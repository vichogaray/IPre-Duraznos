"""
Skeleton Graph Viewer — Conversión de esqueleto a grafo de ramas
=================================================================
INPUT : Una imagen de esqueleto binario
OUTPUT: Interfaz interactiva donde cada rama es un arco del grafo

Interacción:
  · Escribe números en "Merge ramas"    →  ej: "3,4,5"  y presiona Enter o [Merge]
  · Escribe números en "Eliminar ramas" →  ej: "2,6"    y presiona Enter o [Eliminar]
  · [Guardar imagen]                    →  exporta el canvas coloreado como PNG
  · Slider "Rama mín (px)"             →  rehace el grafo filtrando ramas cortas
  · Slider "Grosor línea"              →  cambia el grosor visual
  · Checkboxes                         →  activa/desactiva junctions, endpoints, labels
"""

# =====================================================================
#  CONFIGURACIÓN — MODIFICA AQUÍ
# =====================================================================
SKELETON_IMAGE   = r"E:\Esqueletos sin grafo"
OUTPUT_FOLDER    = r"E:\grafos"

# --- Modo batch (procesa toda una carpeta sin GUI) ---
BATCH_MODE       = False
SKELETON_FOLDER  = r"E:\Esqueletos sin grafo"
MIN_BRANCH_PX    = 5       # Largo mínimo de rama en píxeles
LINE_THICKNESS   = 2       # Grosor visual de las ramas
SHOW_JUNCTIONS   = True    # Mostrar nodos de bifurcación (blanco)
SHOW_ENDPOINTS   = True    # Mostrar puntas de rama (amarillo-naranja)
SHOW_LABELS      = True    # Mostrar número de rama
DARK_BACKGROUND  = True    # Fondo oscuro
CLICK_TOLERANCE  = 8       # Radio de búsqueda al hacer clic (px)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =====================================================================
#  NO MODIFIQUES DEBAJO
# =====================================================================
import sys
import re
import json
import colorsys
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from matplotlib.patches import Patch
from collections import deque
from skimage.morphology import skeletonize
from pathlib import Path


# ---------------------------------------------------------------------------
# 8-connectivity helpers
# ---------------------------------------------------------------------------
NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def get_neighbors(skel, r, c):
    rows, cols = skel.shape
    return [(r+dr, c+dc) for dr, dc in NEIGHBORS_8
            if 0 <= r+dr < rows and 0 <= c+dc < cols and skel[r+dr, c+dc]]


def crossing_number(skel, r, c):
    rows, cols = skel.shape
    order = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    vals = [1 if (0 <= r+dr < rows and 0 <= c+dc < cols and skel[r+dr,c+dc]) else 0
            for dr, dc in order]
    return sum(abs(vals[(i+1) % 8] - vals[i]) for i in range(8)) // 2


# ---------------------------------------------------------------------------
# Junction / endpoint classification
# ---------------------------------------------------------------------------
def classify_skeleton_pixels(skel):
    H, W = skel.shape
    pad  = np.pad(skel > 0, 1, mode='constant').astype(np.int8)

    # --- crossing number (vectorizado) ---
    # Orden circular: (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)
    CN_ORDER = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    planes_cn = [pad[1+dr:1+dr+H, 1+dc:1+dc+W] for dr, dc in CN_ORDER]
    cn_map = np.zeros((H, W), dtype=np.int32)
    for i in range(8):
        cn_map += np.abs(planes_cn[(i+1) % 8].astype(np.int32)
                         - planes_cn[i].astype(np.int32))
    cn_map //= 2

    # --- conteo de vecinos 8-conectados (vectorizado) ---
    NB_ORDER = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    nb_map = sum(pad[1+dr:1+dr+H, 1+dc:1+dc+W].astype(np.int32)
                 for dr, dc in NB_ORDER)

    skel_mask = skel > 0
    ep_mask   = skel_mask & (cn_map == 1) & (nb_map == 1)
    junc_mask = skel_mask & ((cn_map >= 3) | (nb_map >= 3))

    ys_ep,   xs_ep   = np.where(ep_mask)
    ys_junc, xs_junc = np.where(junc_mask)
    endpoints     = set(zip(ys_ep.tolist(),   xs_ep.tolist()))
    raw_junctions = set(zip(ys_junc.tolist(), xs_junc.tolist()))

    junction_map     = {}
    junction_members = {}
    junction_reps    = set()
    visited          = set()

    for start in raw_junctions:
        if start in visited:
            continue
        cluster = []
        queue = deque([start])
        visited.add(start)
        while queue:
            pt = queue.popleft()
            cluster.append(pt)
            for dr, dc in NEIGHBORS_8:
                nb = (pt[0]+dr, pt[1]+dc)
                if nb in raw_junctions and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        mean_r = int(np.mean([p[0] for p in cluster]))
        mean_c = int(np.mean([p[1] for p in cluster]))
        rep = min(cluster, key=lambda p: (p[0]-mean_r)**2 + (p[1]-mean_c)**2)
        junction_reps.add(rep)
        ms = set(cluster)
        junction_members[rep] = ms
        for pt in cluster:
            junction_map[pt] = rep

    node_type = {pt: 'endpoint' for pt in endpoints}
    node_type.update({pt: 'junction' for pt in junction_reps})
    return endpoints, junction_reps, node_type, junction_map, junction_members


# ---------------------------------------------------------------------------
# Branch tracing
# ---------------------------------------------------------------------------
def trace_branches(skel, endpoints, junction_reps, junction_map, junction_members):
    all_junc_px   = set(junction_map.keys())
    branches      = []
    branch_nodes  = []
    globally_visited = set()

    def get_cluster(node):
        return junction_members.get(node, {node})

    def resolve_node(pixel):
        if pixel in endpoints:       return pixel
        if pixel in junction_map:    return junction_map[pixel]
        return None

    all_nodes = endpoints | junction_reps
    node_pixels = endpoints | all_junc_px

    for start_node in all_nodes:
        start_cluster = get_cluster(start_node)

        exit_pixels = []
        for px in start_cluster:
            for nb in get_neighbors(skel, px[0], px[1]):
                if nb not in start_cluster:
                    exit_pixels.append((nb, px))

        seen_exits   = set()
        unique_exits = []
        for ep, fp in exit_pixels:
            if ep not in seen_exits:
                seen_exits.add(ep)
                unique_exits.append((ep, fp))

        for first_step, from_px in unique_exits:
            if first_step in globally_visited:
                continue

            path     = [first_step]
            path_set = {first_step}
            current  = first_step
            prev     = from_px
            found_end = None

            while True:
                node_here = resolve_node(current)
                if node_here is not None and node_here != start_node:
                    found_end = node_here
                    break

                if current in start_cluster:
                    nbs = [n for n in get_neighbors(skel, current[0], current[1])
                           if n != prev and n not in start_cluster and n not in path_set]
                    if not nbs:
                        break
                    prev, current = current, nbs[0]
                    path.append(current)
                    path_set.add(current)
                    continue

                nbs = [n for n in get_neighbors(skel, current[0], current[1])
                       if n != prev and n not in path_set]
                if not nbs:
                    found_end = current
                    break

                if len(nbs) > 1:
                    dr0 = current[0] - prev[0]
                    dc0 = current[1] - prev[1]
                    nbs.sort(key=lambda n: abs((n[0]-current[0])-dr0) +
                                           abs((n[1]-current[1])-dc0))

                prev, current = current, nbs[0]
                path.append(current)
                path_set.add(current)
                if len(path) > skel.shape[0] * skel.shape[1]:
                    break

            if found_end is not None and path:
                end_node = resolve_node(found_end) or found_end
                branches.append(path)
                branch_nodes.append((start_node, end_node))
                for px in path:
                    if px not in node_pixels:
                        globally_visited.add(px)

    return branches, branch_nodes


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_branches(branches, branch_nodes):
    unique_branches, unique_nodes, seen_sets = [], [], []
    for branch, nodes in zip(branches, branch_nodes):
        pset   = frozenset(branch)
        is_dup = False
        for j, seen in enumerate(seen_sets):
            if len(pset & seen) / max(len(pset), len(seen), 1) > 0.6:
                is_dup = True
                if len(branch) > len(unique_branches[j]):
                    unique_branches[j] = branch
                    unique_nodes[j]    = nodes
                    seen_sets[j]       = pset
                break
        if not is_dup:
            unique_branches.append(branch)
            unique_nodes.append(nodes)
            seen_sets.append(pset)
    return unique_branches, unique_nodes


# ---------------------------------------------------------------------------
# Trunk / lateral branch identification (graph diameter via 2-pass BFS)
# ---------------------------------------------------------------------------
def build_branch_graph(bnodes_dict, stats):
    """Adjacency list: node -> [(neighbor, branch_id, pixel_length), ...]"""
    graph = {}
    for lbl, (n0, n1) in bnodes_dict.items():
        length = stats[lbl]['pixels']
        for a, b in [(n0, n1), (n1, n0)]:
            if a not in graph:
                graph[a] = []
            graph[a].append((b, lbl, length))
    return graph


def find_trunk_branches(bnodes_dict, stats, endpoints):
    """
    Identifica las ramas del tronco principal usando el diámetro del grafo.

    Algoritmo (2-pass BFS ponderado por píxeles):
      1. BFS desde cualquier endpoint → encuentra el endpoint más lejano (far1).
      2. BFS desde far1              → encuentra el endpoint más lejano (far2).
      3. El camino far1→far2 son las ramas del tronco.

    Retorna:
      trunk_ids : set de branch IDs que forman el tronco
      trunk_path: lista ordenada de branch IDs del tronco (far1→far2)
      far1, far2: los dos nodos extremo del tronco
    """
    if not bnodes_dict or not endpoints:
        return set(), [], None, None

    graph = build_branch_graph(bnodes_dict, stats)

    def weighted_bfs(start):
        dist    = {start: 0}
        prev    = {start: (None, None)}   # node → (prev_node, branch_id)
        visited = {start}
        queue   = deque([start])
        farthest, max_d = start, 0
        while queue:
            node = queue.popleft()
            for nb, bid, length in graph.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    nd = dist[node] + length
                    dist[nb] = nd
                    prev[nb] = (node, bid)
                    queue.append(nb)
                    if nd > max_d:
                        max_d, farthest = nd, nb
        return farthest, prev, dist

    # Solo endpoints como candidatos a extremos de tronco
    ep_list = list(endpoints)
    start   = ep_list[0]

    far1, _, _     = weighted_bfs(start)
    far2, prev, _  = weighted_bfs(far1)

    # Reconstruir camino far2 → far1
    trunk_path = []
    node = far2
    while True:
        p_node, bid = prev.get(node, (None, None))
        if p_node is None:
            break
        if bid is not None:
            trunk_path.append(bid)
        node = p_node

    trunk_ids = set(trunk_path)
    return trunk_ids, list(reversed(trunk_path)), far1, far2

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

def generate_colors(n):
    if n == 0:
        return []
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95) for i in range(n)]

# ---------------------------------------------------------------------------
# Build graph from skeleton image
# ---------------------------------------------------------------------------
def build_graph(skel_img, min_branch_px):
    _, skel = cv2.threshold(skel_img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(skel) > 0.5:
        skel = 1 - skel
    skel      = skel.astype(np.uint8)
    skel_thin = skeletonize(skel > 0).astype(np.uint8)

    print(f"[INFO] Clasificando píxeles del esqueleto...")
    endpoints, junction_reps, node_type, junction_map, junction_members = \
        classify_skeleton_pixels(skel_thin)
    print(f"[INFO] Endpoints: {len(endpoints)}  Junctions: {len(junction_reps)}")

    print(f"[INFO] Trazando ramas...")
    branches_list, bnodes_list = trace_branches(
        skel_thin, endpoints, junction_reps, junction_map, junction_members)
    branches_list, bnodes_list = deduplicate_branches(branches_list, bnodes_list)
    print(f"[INFO] Ramas tras dedup: {len(branches_list)}")

    filtered = [(b, n) for b, n in zip(branches_list, bnodes_list) if len(b) >= min_branch_px]
    if filtered:
        branches_list, bnodes_list = zip(*filtered)
        branches_list = list(branches_list)
        bnodes_list   = list(bnodes_list)
    else:
        branches_list, bnodes_list = [], []

    # Recompute endpoints/junctions from surviving branches only.
    # This eliminates ghost endpoints from stub branches that were filtered out.
    node_count = {}
    for n0, n1 in bnodes_list:
        node_count[n0] = node_count.get(n0, 0) + 1
        node_count[n1] = node_count.get(n1, 0) + 1
    endpoints     = {n for n, cnt in node_count.items() if cnt == 1}
    junction_reps = {n for n, cnt in node_count.items() if cnt >= 2}

    branches_dict = {i+1: b for i, b in enumerate(branches_list)}
    bnodes_dict   = {i+1: n for i, n in enumerate(bnodes_list)}

    stats = {}
    for lbl in branches_dict:
        pixels   = branches_dict[lbl]
        n0, n1   = bnodes_dict[lbl]
        t0 = node_type.get(n0, 'cont')[:3]
        t1 = node_type.get(n1, 'cont')[:3]
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        stats[lbl] = {
            'pixels':   len(pixels),
            'start':    n0,
            'end':      n1,
            'type':     f"{t0}→{t1}",
            'centroid': (int(np.mean(cs)), int(np.mean(rs))),
        }

    print(f"[INFO] Identificando tronco...")
    trunk_ids, trunk_path, _, _ = find_trunk_branches(
        bnodes_dict, stats, endpoints)
    print(f"[INFO] Tronco: {len(trunk_ids)} ramas  |  Listo, abriendo ventana...")

    return (branches_dict, bnodes_dict, node_type,
            endpoints, junction_reps, stats, skel_thin,
            trunk_ids, trunk_path)


# ---------------------------------------------------------------------------
# Pixel → branch lookup dict
# ---------------------------------------------------------------------------
def build_pixel_lookup(branches_dict):
    lookup = {}

    for lbl, pixels in branches_dict.items():
        for px in pixels:
            lookup[px] = lbl
    return lookup


# ---------------------------------------------------------------------------
# Render colored canvas
# ---------------------------------------------------------------------------
TRUNK_COLOR = (1.0, 0.55, 0.0)   # naranja brillante para el tronco


def render_canvas(img_shape, branches_dict, color_map, thickness, dark,
                  highlight=None, highlight_color=(1.0, 0.9, 0.0),
                  trunk_ids=None):
    """
    highlight  : conjunto opcional de branch IDs a resaltar (preview al escribir)
    trunk_ids  : conjunto de branch IDs que forman el tronco principal
    """
    bg_val = 30 if dark else 240
    canvas = np.full((*img_shape, 3), bg_val, dtype=np.uint8)
    if highlight is None:
        highlight = set()
    if trunk_ids is None:
        trunk_ids = set()

    h, w = img_shape
    kernel = None
    if thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (thickness, thickness))
        pad = thickness

    for lbl, pixels in branches_dict.items():
        if lbl in highlight:
            color_f = highlight_color
        elif lbl in trunk_ids:
            color_f = TRUNK_COLOR
        else:
            color_f = color_map[lbl]
        color_u8 = (int(color_f[0] * 255),
                     int(color_f[1] * 255),
                     int(color_f[2] * 255))

        px = np.array(pixels, dtype=np.intp)
        rs = px[:, 0]
        cs = px[:, 1]

        if kernel is not None:
            # Dilatar solo en la bounding-box de la rama (no imagen completa)
            r_min = max(0, int(rs.min()) - pad)
            r_max = min(h, int(rs.max()) + pad + 1)
            c_min = max(0, int(cs.min()) - pad)
            c_max = min(w, int(cs.max()) + pad + 1)
            roi = np.zeros((r_max - r_min, c_max - c_min), dtype=np.uint8)
            roi[rs - r_min, cs - c_min] = 1
            roi = cv2.dilate(roi, kernel)
            canvas[r_min:r_max, c_min:c_max][roi > 0] = color_u8
        else:
            canvas[np.clip(rs, 0, h - 1), np.clip(cs, 0, w - 1)] = color_u8

    return canvas


# ---------------------------------------------------------------------------
# Export graph to JSON
# ---------------------------------------------------------------------------
def export_graph_json(branches_dict, bnodes_dict, node_type, endpoints,
                      junction_reps, stats, trunk_ids, image_name, out_json):
    """
    Exporta el grafo como JSON con nodos, aristas y pixeles por rama.
    Formato pensado para el metodo Graph Laplacian semi-supervisado.
    """
    all_nodes = endpoints | junction_reps
    nodes = []
    for (r, c) in all_nodes:
        ntype = node_type.get((r, c), 'junction')
        nodes.append({
            "id":   f"{r}_{c}",
            "x":    c,
            "y":    r,
            "type": ntype,
        })

    edges    = []
    branches = []
    for bid in sorted(branches_dict.keys()):
        n0, n1   = bnodes_dict[bid]
        s        = stats[bid]
        is_trunk = bid in trunk_ids
        edges.append({
            "branch_id": bid,
            "from":      f"{n0[0]}_{n0[1]}",
            "to":        f"{n1[0]}_{n1[1]}",
            "length_px": s['pixels'],
            "is_trunk":  is_trunk,
        })
        branches.append({
            "id":         bid,
            "pixels":     [[r, c] for r, c in branches_dict[bid]],
            "centroid_x": s['centroid'][0],
            "centroid_y": s['centroid'][1],
            "is_trunk":   is_trunk,
        })

    data = {
        "image":    image_name,
        "nodes":    nodes,
        "edges":    edges,
        "branches": branches,
    }
    with open(str(out_json), 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print(f"[OK] JSON guardado: {out_json}")


# ===========================================================================
#  VIEWER INTERACTIVO
# ===========================================================================
def run_viewer(image_path_or_folder, output_folder):
    start_path = Path(image_path_or_folder)
    out_path   = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Construir lista de imágenes ordenada por nombre natural ---
    if start_path.is_dir():
        folder = start_path
        files  = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=_natural_key
        )
        if not files:
            print(f"[ERROR] No se encontraron imágenes en: {folder}")
            sys.exit(1)
        current_idx = 0
    else:
        folder = start_path.parent
        files  = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=_natural_key
        )
        current_idx = next((i for i, f in enumerate(files) if f == start_path), 0)

    nav = {'idx': current_idx}

    def load_and_build(min_px=MIN_BRANCH_PX):
        image_path = files[nav['idx']]
        im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"[ERROR] No se pudo cargar: {image_path}")
            sys.exit(1)
        print(f"\n[INFO] [{nav['idx']+1}/{len(files)}] {image_path.name}  {im.shape[1]}x{im.shape[0]} px")
        bd, bn, nt, ep, jr, sv, _, ti, tp = build_graph(im, min_px)
        nc = generate_colors(len(bd))
        cm = {lbl: nc[i % len(nc)] for i, lbl in enumerate(sorted(bd))}
        pl = build_pixel_lookup(bd)
        return im, image_path.name, bd, bn, nt, ep, jr, sv, ti, tp, cm, pl

    (img, name,
     branches_dict, bnodes_dict, node_type,
     endpoints, junction_reps, stats,
     trunk_ids, trunk_path,
     color_map, px_lookup) = load_and_build()

    n_branches = len(branches_dict)
    print(f"[INFO] Ramas detectadas: {n_branches}")
    print(f"[INFO] Junctions: {len(junction_reps)}  |  Endpoints: {len(endpoints)}")

    # Mutable state
    st = {
        'img':            img,
        'name':           name,
        'min_branch':     MIN_BRANCH_PX,
        'thickness':      LINE_THICKNESS,
        'show_junctions': SHOW_JUNCTIONS,
        'show_endpoints': SHOW_ENDPOINTS,
        'show_labels':    SHOW_LABELS,
        'show_trunk':     True,
        'branches':       branches_dict,
        'bnodes':         bnodes_dict,
        'node_type':      node_type,
        'endpoints':      endpoints,
        'junctions':      junction_reps,
        'stats':          stats,
        'color_map':      color_map,
        'px_lookup':      px_lookup,
        'trunk_ids':      trunk_ids,
        'trunk_path':     trunk_path,
    }

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(21, 11))
    fig.patch.set_facecolor('#1a1a1a')

    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        top=0.92, bottom=0.22,
        left=0.01, right=0.99,
        wspace=0.03,
        width_ratios=[5, 5, 3]
    )

    ax_orig     = fig.add_subplot(gs[0, 0])
    ax_branches = fig.add_subplot(gs[0, 1])
    ax_info     = fig.add_subplot(gs[0, 2])

    for ax in [ax_orig, ax_branches, ax_info]:
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')

    ax_orig.set_title('Esqueleto Original',
                      color='white', fontsize=10, pad=4)
    ax_branches.set_title('Grafo de Ramas  (clic para seleccionar)',
                          color='#88ccff', fontsize=10, pad=4)
    ax_info.set_title('Información del Grafo',
                      color='white', fontsize=10, pad=4)

    suptitle = fig.suptitle(
        f"{name}   —   {n_branches} ramas  |  {len(junction_reps)} junctions  |  {len(endpoints)} endpoints",
        color='white', fontsize=11, fontweight='bold')

    # --- imshow artists ---
    ax_orig.imshow(img, cmap='gray', vmin=0, vmax=255)
    canvas0     = render_canvas(img.shape, st['branches'], st['color_map'],
                                st['thickness'], DARK_BACKGROUND)
    im_branches = ax_branches.imshow(canvas0)

    # --- Scatter artists (junctions / endpoints) ---
    junc_scatter = ax_branches.scatter([], [], s=70, c='white',
                                        edgecolors='black', linewidths=0.8,
                                        zorder=6, label='Junction')
    endp_scatter = ax_branches.scatter([], [], s=50, c='#FFC800',
                                        edgecolors='black', linewidths=0.8,
                                        zorder=6, label='Endpoint')

    # --- Branch number labels ---
    label_artists = []

    # --- Info text ---
    info_text = ax_info.text(
        0.04, 0.97, '',
        transform=ax_info.transAxes,
        color='white', fontsize=7.5, va='top', ha='left',
        fontfamily='monospace'
    )

    # -----------------------------------------------------------------------
    # Update helpers
    # -----------------------------------------------------------------------
    def _update_scatter():
        if st['show_junctions'] and st['junctions']:
            jcs = list(st['junctions'])
            junc_scatter.set_offsets([(c, r) for r, c in jcs])
        else:
            junc_scatter.set_offsets(np.empty((0, 2)))

        if st['show_endpoints'] and st['endpoints']:
            eps = list(st['endpoints'])
            endp_scatter.set_offsets([(c, r) for r, c in eps])
        else:
            endp_scatter.set_offsets(np.empty((0, 2)))

    def _update_labels():
        for t in label_artists:
            t.remove()
        label_artists.clear()
        if not st['show_labels']:
            return
        for lbl, s in st['stats'].items():
            cx, cy = s['centroid']
            t = ax_branches.text(
                cx, cy, str(lbl),
                fontsize=7, fontweight='bold',
                color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='black',
                          alpha=0.85, edgecolor='none'),
                zorder=10
            )
            label_artists.append(t)

    def _build_info():
        lines = []
        n     = len(st['branches'])

        trunk_px = sum(st['stats'][i]['pixels']
                       for i in st['trunk_ids'] if i in st['stats'])
        lines.append(f"Ramas totales : {n}")
        lines.append(f"  Tronco      : {len(st['trunk_ids'])} ramas  ({trunk_px} px)")
        lines.append(f"  Laterales   : {n - len(st['trunk_ids'])} ramas")
        lines.append(f"Junctions     : {len(st['junctions'])}")
        lines.append(f"Endpoints     : {len(st['endpoints'])}")
        lines.append("─" * 28)
        lines.append(f"Tronco (orden): {st['trunk_path']}"[:50])

        lines.append("Todas las ramas (↓ por largo):")
        sorted_st = sorted(st['stats'].items(), key=lambda x: -x[1]['pixels'])
        for lbl, s in sorted_st[:35]:
            lines.append(f"  R{lbl:3d}: {s['pixels']:5d} px  {s['type']}")
        if n > 35:
            lines.append(f"  ... +{n-35} ramas más")

        return "\n".join(lines)

    def _set_suptitle():
        n = len(st['branches'])
        suptitle.set_text(
            f"{st['name']}   —   "
            f"{n} ramas  |  {len(st['junctions'])} junctions  |  {len(st['endpoints'])} endpoints")

    def full_redraw(highlight=None, highlight_color=(1.0, 0.9, 0.0)):
        trunk = st['trunk_ids'] if st['show_trunk'] else set()
        canvas = render_canvas(st['img'].shape, st['branches'], st['color_map'],
                               st['thickness'], DARK_BACKGROUND,
                               highlight=highlight, highlight_color=highlight_color,
                               trunk_ids=trunk)
        im_branches.set_data(canvas)
        _update_scatter()
        _update_labels()
        info_text.set_text(_build_info())
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Initial render
    _update_scatter()
    _update_labels()
    info_text.set_text(_build_info())

    # -----------------------------------------------------------------------
    # Sliders
    # -----------------------------------------------------------------------
    sl_color  = '#3a3a3a'
    sl_active = '#5a9fd4'

    ax_sl_branch = fig.add_axes([0.06, 0.13, 0.28, 0.025], facecolor=sl_color)
    ax_sl_thick  = fig.add_axes([0.40, 0.13, 0.22, 0.025], facecolor=sl_color)

    s_branch = Slider(ax_sl_branch, 'Rama mín (px)', 1, 300,
                      valinit=MIN_BRANCH_PX, valstep=1, color=sl_active)
    s_thick  = Slider(ax_sl_thick, 'Grosor línea',   1, 8,
                      valinit=LINE_THICKNESS, valstep=1, color=sl_active)

    for s in [s_branch, s_thick]:
        s.label.set_color('white')
        s.valtext.set_color('white')

    def on_branch_changed(val):
        new_min = int(s_branch.val)
        if new_min == st['min_branch']:
            return
        st['min_branch'] = new_min

        (bd, bn, nt, ep, jr, sv, _, ti, tp) = build_graph(st['img'], new_min)
        new_colors = generate_colors(len(bd))

        st['branches']    = bd
        st['bnodes']      = bn
        st['node_type']   = nt
        st['endpoints']   = ep
        st['junctions']   = jr
        st['stats']       = sv
        st['color_map']   = {lbl: new_colors[i % len(new_colors)]
                              for i, lbl in enumerate(sorted(bd))}
        st['px_lookup']   = build_pixel_lookup(bd)
        st['trunk_ids']   = ti
        st['trunk_path']  = tp

        _set_suptitle()
        full_redraw()

    def on_thick_changed(val):
        st['thickness'] = int(s_thick.val)
        full_redraw()

    s_branch.on_changed(on_branch_changed)
    s_thick.on_changed(on_thick_changed)

    # -----------------------------------------------------------------------
    # Helper: parse IDs from text  ("3,4,5"  o  "3 4 5"  o  "3, 4, 5")
    # -----------------------------------------------------------------------
    import re as _re
    def _parse_ids(text):
        tokens = _re.split(r'[,\s]+', text.strip())
        ids = []
        for t in tokens:
            try:
                ids.append(int(t))
            except ValueError:
                pass
        return ids

    # -----------------------------------------------------------------------
    # TextBoxes + Buttons
    # -----------------------------------------------------------------------
    def _make_btn(left, bottom, width, label, bg, hover):
        ax_b = fig.add_axes([left, bottom, width, 0.04], facecolor=bg)
        b = Button(ax_b, label, color=bg, hovercolor=hover)
        b.label.set_color('white')
        b.label.set_fontsize(8.5)
        return b

    # Fila 1: Merge
    ax_tb_merge = fig.add_axes([0.19, 0.10, 0.22, 0.04], facecolor='#2a2a2a')
    tb_merge    = TextBox(ax_tb_merge, 'Merge ramas (ej: 1,3,5): ',
                          initial='', color='#2a2a2a', hovercolor='#3a3a3a')
    tb_merge.label.set_color('white')
    tb_merge.text_disp.set_color('#88ffcc')

    btn_merge = _make_btn(0.43, 0.10, 0.07, 'Merge', '#1a4a7a', '#2a6aaa')

    # Fila 2: Eliminar
    ax_tb_delete = fig.add_axes([0.19, 0.04, 0.22, 0.04], facecolor='#2a2a2a')
    tb_delete    = TextBox(ax_tb_delete, 'Eliminar ramas (ej: 2,4): ',
                           initial='', color='#2a2a2a', hovercolor='#3a3a3a')
    tb_delete.label.set_color('white')
    tb_delete.text_disp.set_color('#ff9999')

    btn_delete = _make_btn(0.43, 0.04, 0.07, 'Eliminar', '#6a1a1a', '#9a2a2a')

    # Guardar (centrado entre las dos filas)
    btn_save = _make_btn(0.52, 0.07, 0.09, 'Guardar imagen', '#1a6a1a', '#2a9a2a')

    # CheckButtons
    ax_chk = fig.add_axes([0.55, 0.03, 0.22, 0.14], facecolor='#1a1a1a')
    chk = CheckButtons(
        ax_chk,
        ['Junctions', 'Endpoints', 'Labels', 'Tronco'],
        [st['show_junctions'], st['show_endpoints'], st['show_labels'], st['show_trunk']]
    )
    for lbl_txt in chk.labels:
        lbl_txt.set_color('white')
        lbl_txt.set_fontsize(8)

    def on_check(label):
        if label == 'Junctions':
            st['show_junctions'] = not st['show_junctions']
        elif label == 'Endpoints':
            st['show_endpoints'] = not st['show_endpoints']
        elif label == 'Labels':
            st['show_labels'] = not st['show_labels']
        elif label == 'Tronco':
            st['show_trunk'] = not st['show_trunk']
        full_redraw()

    chk.on_clicked(on_check)

    # --- Merge ---
    def on_merge(_):
        ids   = _parse_ids(tb_merge.text)
        valid = [i for i in ids if i in st['branches']]
        if len(valid) < 2:
            print(f"[WARN] Merge: se necesitan ≥2 ramas válidas. Recibidas: {ids}, válidas: {valid}")
            return
        target = valid[0]
        for src in valid[1:]:
            st['branches'][target] = st['branches'][target] + st['branches'][src]
            st['branches'].pop(src, None)
            st['bnodes'].pop(src, None)
            st['stats'].pop(src, None)
            print(f"[MERGE] Rama {src} → Rama {target}")

        pixels = st['branches'][target]
        rs = [p[0] for p in pixels]
        cs = [p[1] for p in pixels]
        old_stat = st['stats'].get(target, {})
        st['stats'][target] = {
            'pixels':   len(pixels),
            'start':    old_stat.get('start'),
            'end':      old_stat.get('end'),
            'type':     old_stat.get('type', '?'),
            'centroid': (int(np.mean(cs)), int(np.mean(rs))),
        }

        st['px_lookup'] = build_pixel_lookup(st['branches'])
        tb_merge.set_val('')

        _set_suptitle()
        full_redraw()

    # --- Delete ---
    def on_delete(_):
        ids   = _parse_ids(tb_delete.text)
        valid = [i for i in ids if i in st['branches']]
        if not valid:
            print(f"[WARN] Eliminar: ninguna rama válida. Recibidas: {ids}")
            return
        for lbl in valid:
            st['branches'].pop(lbl, None)
            st['bnodes'].pop(lbl, None)
            st['stats'].pop(lbl, None)
            print(f"[DELETE] Rama {lbl} eliminada")

        st['px_lookup'] = build_pixel_lookup(st['branches'])
        tb_delete.set_val('')

        _set_suptitle()
        full_redraw()

    # --- Preview al escribir en los textboxes ---
    def on_merge_text(text):
        ids  = _parse_ids(text)
        prev = {i for i in ids if i in st['branches']}
        full_redraw(highlight=prev)

    def on_delete_text(text):
        ids  = _parse_ids(text)
        prev = {i for i in ids if i in st['branches']}
        full_redraw(highlight=prev, highlight_color=(1.0, 0.3, 0.3))

    tb_merge.on_text_change(on_merge_text)
    tb_delete.on_text_change(on_delete_text)

    # --- Save image ---
    def on_save(_):
        p      = Path(st['name'])
        out    = out_path / (p.stem + '_graph' + p.suffix)
        canvas = render_canvas(st['img'].shape, st['branches'], st['color_map'],
                               st['thickness'], DARK_BACKGROUND)
        cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[OK] Imagen guardada: {out}")
        out_json = out_path / (p.stem + '_graph.json')
        export_graph_json(
            st['branches'], st['bnodes'], st['node_type'],
            st['endpoints'], st['junctions'], st['stats'],
            st['trunk_ids'], st['name'], out_json
        )

    btn_merge.on_clicked(on_merge)
    btn_delete.on_clicked(on_delete)
    btn_save.on_clicked(on_save)

    tb_merge.on_submit(on_merge)
    tb_delete.on_submit(on_delete)

    # --- Navegación entre imágenes ---
    def _reload_image():
        (new_img, new_name,
         new_bd, new_bn, new_nt, new_ep, new_jr, new_sv,
         new_ti, new_tp, new_cm, new_pl) = load_and_build(st['min_branch'])
        st['img']       = new_img
        st['name']      = new_name
        st['branches']  = new_bd
        st['bnodes']    = new_bn
        st['node_type'] = new_nt
        st['endpoints'] = new_ep
        st['junctions'] = new_jr
        st['stats']     = new_sv
        st['trunk_ids'] = new_ti
        st['trunk_path']= new_tp
        st['color_map'] = new_cm
        st['px_lookup'] = new_pl
        ax_orig.imshow(new_img, cmap='gray', vmin=0, vmax=255)
        _set_suptitle()
        full_redraw()
        nav_label.set_text(f"{nav['idx']+1} / {len(files)}")
        fig.canvas.draw()

    def on_prev(_):
        if nav['idx'] > 0:
            nav['idx'] -= 1
            _reload_image()

    def on_next(_):
        if nav['idx'] < len(files) - 1:
            nav['idx'] += 1
            _reload_image()

    btn_prev = _make_btn(0.63, 0.10, 0.06, '◀ Anterior', '#333355', '#4444aa')
    btn_next = _make_btn(0.70, 0.10, 0.06, 'Siguiente ▶', '#333355', '#4444aa')
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    ax_nav_lbl = fig.add_axes([0.63, 0.05, 0.13, 0.04], facecolor='#1a1a1a')
    ax_nav_lbl.axis('off')
    nav_label = ax_nav_lbl.text(
        0.5, 0.5, f"{nav['idx']+1} / {len(files)}",
        transform=ax_nav_lbl.transAxes,
        color='#aaaaaa', fontsize=9, ha='center', va='center',
        fontfamily='monospace'
    )

    # --- Legend hint ---
    ax_hint = fig.add_axes([0.79, 0.03, 0.20, 0.12], facecolor='#1a1a1a')
    ax_hint.axis('off')
    ax_hint.text(0.05, 0.92,
                 "◉ Blanco   = Junction\n"
                 "◉ Naranja  = Endpoint\n"
                 "━ Naranja  = Tronco\n\n"
                 "Merge: escribe IDs y\n"
                 "  pulsa Enter o [Merge]\n\n"
                 "Eliminar: escribe IDs y\n"
                 "  pulsa Enter o [Eliminar]\n\n"
                 "Preview: amarillo=merge\n"
                 "         rojo=eliminar",
                 transform=ax_hint.transAxes,
                 color='#aaaaaa', fontsize=7.5, va='top', fontfamily='monospace')

    plt.show()


# =====================================================================
def _natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.name)]


if __name__ == "__main__":
    if BATCH_MODE:
        in_dir  = Path(SKELETON_FOLDER)
        out_dir = Path(OUTPUT_FOLDER)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [f for f in in_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=_natural_key
        )
        print(f"[BATCH] {len(files)} imagenes en {in_dir.name}")

        for idx, img_path in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] {img_path.name}")
            im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                print("  [ERROR] No se pudo cargar, saltando.")
                continue
            try:
                bd, bn, nt, ep, jr, sv, _, ti, tp = build_graph(im, MIN_BRANCH_PX)
            except Exception as exc:
                print(f"  [ERROR] {exc}, saltando.")
                continue
            nc = generate_colors(len(bd))
            cm = {lbl: nc[i % len(nc)] for i, lbl in enumerate(sorted(bd))}
            canvas = render_canvas(im.shape, bd, cm, LINE_THICKNESS, DARK_BACKGROUND)
            out = out_dir / (img_path.stem + '_graph' + img_path.suffix)
            cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            out_json = out_dir / (img_path.stem + '_graph.json')
            export_graph_json(bd, bn, nt, ep, jr, sv, ti, img_path.name, out_json)
            print(f"  [OK] -> {out.name}")

        print(f"\n[BATCH] Listo. {len(files)} imagenes procesadas -> {out_dir}")
        sys.exit(0)

    if len(sys.argv) >= 3:
        image_arg  = sys.argv[1]
        output_arg = sys.argv[2]
    elif len(sys.argv) == 2:
        p = Path(sys.argv[1])
        image_arg  = str(p)
        output_arg = str((p if p.is_dir() else p.parent) / "Grafos")
    else:
        # Si SKELETON_FOLDER existe y no está en batch, úsalo como carpeta de entrada
        image_arg  = SKELETON_FOLDER if Path(SKELETON_FOLDER).is_dir() else SKELETON_IMAGE
        output_arg = OUTPUT_FOLDER
    run_viewer(image_arg, output_arg)
