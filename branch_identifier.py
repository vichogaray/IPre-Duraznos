"""
Branch Identifier for Peach Tree Skeletons (v2 - Graph-based)
==============================================================
Recibe una imagen de esqueleto (JPG) y genera una visualización
con cada rama coloreada de forma distinta.

Uso:
    python branch_identifier.py <ruta_imagen_esqueleto> [--output <ruta_salida>]

Algoritmo (basado en grafo):
    1. Binariza y adelgaza el esqueleto a 1px
    2. Detecta nodos: junctions (CN >= 3) y endpoints (CN == 1)
    3. Traza cada rama recorriendo pixel a pixel desde cada nodo
       hasta llegar a otro nodo → cada recorrido = 1 rama
    4. Resultado: grafo donde nodos = junctions/endpoints, aristas = ramas
    5. Asigna colores distintos a cada rama
    
    Garantía topológica:
      - Toda rama conecta junction→endpoint o junction→junction
      - La única excepción: si el esqueleto entero es una línea sin
        bifurcaciones (endpoint→endpoint)
"""

import sys
import argparse
import numpy as np
import cv2
from scipy import ndimage
from collections import deque
import colorsys


# ---------------------------------------------------------------------------
# 1. Vecindad 8-conectada
# ---------------------------------------------------------------------------
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]


def get_neighbors(skel, r, c):
    """Retorna lista de vecinos activos en 8-conectividad."""
    rows, cols = skel.shape
    neighbors = []
    for dr, dc in NEIGHBORS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and skel[nr, nc]:
            neighbors.append((nr, nc))
    return neighbors


def crossing_number(skel, r, c):
    """
    Número de cruce para clasificar un pixel del esqueleto.
    CN == 1 → endpoint
    CN == 2 → punto de continuación (parte de una rama)
    CN >= 3 → junction / punto de bifurcación
    """
    rows, cols = skel.shape
    order = [(-1, 0), (-1, 1), (0, 1), (1, 1),
             (1, 0), (1, -1), (0, -1), (-1, -1)]
    vals = []
    for dr, dc in order:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            vals.append(1 if skel[nr, nc] else 0)
        else:
            vals.append(0)
    cn = 0
    for i in range(8):
        cn += abs(vals[(i + 1) % 8] - vals[i])
    return cn // 2


# ---------------------------------------------------------------------------
# 2. Clasificación de pixeles y clustering de junctions
# ---------------------------------------------------------------------------
def classify_skeleton_pixels(skel):
    """
    Clasifica cada pixel y agrupa junctions adyacentes en clusters.
    
    Retorna:
      endpoints       : set de (r, c)
      junction_reps   : set de (r, c) — un representante por cluster
      node_type       : dict {(r,c): 'endpoint' | 'junction'}
      junction_map    : dict que mapea cada pixel junction → su representante
      junction_members: dict {representante: set de pixeles del cluster}
    """
    endpoints = set()
    raw_junctions = set()

    ys, xs = np.where(skel > 0)
    for r, c in zip(ys, xs):
        cn = crossing_number(skel, r, c)
        if cn == 1:
            endpoints.add((r, c))
        elif cn >= 3:
            raw_junctions.add((r, c))

    # Agrupar junctions adyacentes en clusters
    junction_map = {}       # pixel → representante
    junction_members = {}   # representante → set de pixeles
    junction_reps = set()
    visited = set()

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
                nb = (pt[0] + dr, pt[1] + dc)
                if nb in raw_junctions and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        # Representante = pixel más cercano al centroide
        mean_r = int(np.mean([p[0] for p in cluster]))
        mean_c = int(np.mean([p[1] for p in cluster]))
        rep = min(cluster, key=lambda p: (p[0]-mean_r)**2 + (p[1]-mean_c)**2)
        junction_reps.add(rep)
        member_set = set(cluster)
        junction_members[rep] = member_set
        for pt in cluster:
            junction_map[pt] = rep

    node_type = {}
    for pt in endpoints:
        node_type[pt] = 'endpoint'
    for pt in junction_reps:
        node_type[pt] = 'junction'

    return endpoints, junction_reps, node_type, junction_map, junction_members


# ---------------------------------------------------------------------------
# 3. Trazado de ramas (recorrido de grafo)
# ---------------------------------------------------------------------------
def trace_branches(skel, endpoints, junction_reps, junction_map, junction_members):
    """
    Traza cada rama recorriendo pixel a pixel desde cada nodo.
    
    Una rama = camino entre dos nodos (endpoint o junction).
    
    Garantías:
      - Cada rama va de junction→endpoint, junction→junction, o 
        endpoint→endpoint (solo si no hay junctions)
      - No hay ramas endpoint→endpoint si existen junctions
    """
    all_junction_pixels = set(junction_map.keys())
    all_node_pixels = endpoints | all_junction_pixels

    branches = []       # lista de listas de (r, c)
    branch_nodes = []   # lista de (nodo_inicio, nodo_fin)

    # Pixeles ya asignados a alguna rama (para evitar duplicados)
    globally_visited = set()

    def get_cluster_pixels(node):
        """Retorna todos los pixeles de un nodo (1 para endpoint, N para junction cluster)."""
        if node in junction_members:
            return junction_members[node]
        return {node}

    def resolve_node(pixel):
        """Dado un pixel, retorna el nodo al que pertenece o None."""
        if pixel in endpoints:
            return pixel
        if pixel in junction_map:
            return junction_map[pixel]
        return None

    # Recorrer desde cada nodo
    all_nodes = endpoints | junction_reps

    for start_node in all_nodes:
        start_cluster = get_cluster_pixels(start_node)

        # Encontrar pixeles de salida: vecinos del cluster que NO son del cluster
        exit_pixels = []
        for px in start_cluster:
            for nb in get_neighbors(skel, px[0], px[1]):
                if nb not in start_cluster:
                    exit_pixels.append((nb, px))

        # Remover duplicados de salida
        seen_exits = set()
        unique_exits = []
        for ep, from_px in exit_pixels:
            if ep not in seen_exits:
                seen_exits.add(ep)
                unique_exits.append((ep, from_px))

        for first_step, from_px in unique_exits:
            # Si este pixel ya fue trazado, saltar
            if first_step in globally_visited:
                continue

            # Trazar la rama pixel a pixel
            path = [first_step]
            current = first_step
            prev = from_px
            found_end_node = None

            while True:
                # ¿El pixel actual es un nodo diferente al de inicio?
                node_here = resolve_node(current)
                if node_here is not None and node_here != start_node:
                    found_end_node = node_here
                    break

                # Si es parte del cluster de inicio (raro, pero posible), seguir
                if current in start_cluster:
                    neighbors = get_neighbors(skel, current[0], current[1])
                    next_options = [n for n in neighbors 
                                    if n != prev and n not in start_cluster and n not in set(path)]
                    if not next_options:
                        break
                    prev = current
                    current = next_options[0]
                    path.append(current)
                    continue

                # Buscar siguiente pixel de continuación
                neighbors = get_neighbors(skel, current[0], current[1])
                next_options = [n for n in neighbors 
                                if n != prev and n not in set(path)]

                if not next_options:
                    # Fin del camino sin encontrar nodo
                    # Esto puede pasar en puntas que no se detectaron como endpoint
                    found_end_node = current  # Tratar como endpoint implícito
                    break

                # Si hay múltiples opciones, preferir la que sigue más recto
                if len(next_options) > 1:
                    dr_prev = current[0] - prev[0]
                    dc_prev = current[1] - prev[1]
                    next_options.sort(
                        key=lambda n: abs((n[0]-current[0]) - dr_prev) + 
                                      abs((n[1]-current[1]) - dc_prev))

                prev = current
                current = next_options[0]
                path.append(current)

                # Seguridad anti-loop
                if len(path) > skel.shape[0] * skel.shape[1]:
                    break

            if found_end_node is not None and len(path) >= 1:
                end_node = resolve_node(found_end_node)
                if end_node is None:
                    end_node = found_end_node

                branches.append(path)
                branch_nodes.append((start_node, end_node))

                # Marcar pixeles intermedios como visitados
                for px in path:
                    if px not in all_node_pixels:
                        globally_visited.add(px)

    return branches, branch_nodes


# ---------------------------------------------------------------------------
# 4. Deduplicación de ramas
# ---------------------------------------------------------------------------
def deduplicate_branches(branches, branch_nodes):
    """Elimina ramas duplicadas basándose en solapamiento de pixeles."""
    unique_branches = []
    unique_nodes = []
    seen_sets = []

    for branch, nodes in zip(branches, branch_nodes):
        pset = frozenset(branch)
        is_dup = False
        for j, seen in enumerate(seen_sets):
            overlap = len(pset & seen) / max(len(pset), len(seen), 1)
            if overlap > 0.6:
                is_dup = True
                if len(branch) > len(unique_branches[j]):
                    unique_branches[j] = branch
                    unique_nodes[j] = nodes
                    seen_sets[j] = pset
                break
        if not is_dup:
            unique_branches.append(branch)
            unique_nodes.append(nodes)
            seen_sets.append(pset)

    return unique_branches, unique_nodes


# ---------------------------------------------------------------------------
# 5. Paleta de colores
# ---------------------------------------------------------------------------
def generate_colors(n, saturation=0.85, value=0.95):
    """Genera n colores bien distribuidos en HSV."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------
def identify_branches(skeleton_path, output_path=None,
                      min_branch_size=5,
                      line_thickness=2, show_junctions=True,
                      show_endpoints=True, dark_background=True):
    """
    Pipeline completo.
    """
    # --- Cargar ---
    img = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {skeleton_path}")
    print(f"[INFO] Imagen: {img.shape[1]}x{img.shape[0]} px")

    # --- Binarizar ---
    _, skel = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(skel) > 0.5:
        skel = 1 - skel
    skel = skel.astype(np.uint8)
    print(f"[INFO] Pixeles esqueleto: {np.sum(skel > 0)}")

    if np.sum(skel > 0) == 0:
        raise ValueError("No se detectó esqueleto en la imagen.")

    # --- Adelgazar ---
    from skimage.morphology import skeletonize
    skel_thin = skeletonize(skel > 0).astype(np.uint8)
    print(f"[INFO] Pixeles tras adelgazamiento: {np.sum(skel_thin > 0)}")

    # --- Clasificar ---
    endpoints, junction_reps, node_type, junction_map, junction_members = \
        classify_skeleton_pixels(skel_thin)
    print(f"[INFO] Endpoints: {len(endpoints)}")
    print(f"[INFO] Junctions (clusters): {len(junction_reps)}")

    # --- Trazar ramas ---
    branches, branch_nodes = trace_branches(
        skel_thin, endpoints, junction_reps, junction_map, junction_members
    )
    print(f"[INFO] Ramas trazadas: {len(branches)}")

    # --- Deduplicar ---
    branches, branch_nodes = deduplicate_branches(branches, branch_nodes)
    print(f"[INFO] Ramas únicas: {len(branches)}")

    # --- Filtrar pequeñas ---
    filtered = [(b, n) for b, n in zip(branches, branch_nodes) if len(b) >= min_branch_size]
    if filtered:
        branches, branch_nodes = zip(*filtered)
        branches, branch_nodes = list(branches), list(branch_nodes)
    else:
        branches, branch_nodes = [], []

    num_branches = len(branches)
    print(f"[INFO] Ramas finales (min={min_branch_size}px): {num_branches}")

    # --- Estadísticas ---
    stats = {}
    for i, (branch, nodes) in enumerate(zip(branches, branch_nodes)):
        lbl = i + 1
        rs = [p[0] for p in branch]
        cs = [p[1] for p in branch]
        t_start = node_type.get(nodes[0], 'continuation')
        t_end = node_type.get(nodes[1], 'continuation')
        stats[lbl] = {
            'pixels': len(branch),
            'start': nodes[0],
            'end': nodes[1],
            'type': f"{t_start} -> {t_end}",
            'centroid': (int(np.mean(cs)), int(np.mean(rs)))
        }

    print(f"\n{'='*65}")
    print(f"  RESUMEN: {num_branches} ramas identificadas")
    print(f"{'='*65}")
    for lbl, s in sorted(stats.items(), key=lambda x: -x[1]['pixels']):
        print(f"  Rama {lbl:3d}: {s['pixels']:5d} px  |  {s['type']:28s}  |  centroide: {s['centroid']}")
    print(f"{'='*65}\n")

    # --- Visualización ---
    if dark_background:
        canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
    else:
        canvas = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 240

    colors = generate_colors(num_branches)

    for i, branch in enumerate(branches):
        color = colors[i % len(colors)]
        mask = np.zeros(skel_thin.shape, dtype=np.uint8)
        for (r, c) in branch:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                mask[r, c] = 1
        if line_thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (line_thickness, line_thickness))
            mask = cv2.dilate(mask, kernel, iterations=1)
        canvas[mask > 0] = color

    if show_junctions:
        for (r, c) in junction_reps:
            cv2.circle(canvas, (c, r), max(4, line_thickness + 2),
                       (255, 255, 255), -1)
            cv2.circle(canvas, (c, r), max(4, line_thickness + 2),
                       (0, 0, 0), 1)

    if show_endpoints:
        for (r, c) in endpoints:
            cv2.circle(canvas, (c, r), max(3, line_thickness + 1),
                       (0, 200, 255), -1)
            cv2.circle(canvas, (c, r), max(3, line_thickness + 1),
                       (0, 0, 0), 1)

    # --- Leyenda ---
    legend_w = 300
    legend_h = 40 + num_branches * 28 + 60
    legend_h = min(legend_h, img.shape[0] - 20)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (8 + legend_w, 8 + legend_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    y_off = 30
    cv2.putText(canvas, f"Ramas: {num_branches}", (16, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_off += 28

    max_items = min(num_branches, (legend_h - 80) // 28)
    sorted_stats = sorted(stats.items(), key=lambda x: -x[1]['pixels'])

    for i, (lbl, s) in enumerate(sorted_stats[:max_items]):
        color = colors[(lbl - 1) % len(colors)]
        cv2.rectangle(canvas, (16, y_off - 10), (36, y_off + 4), color, -1)
        t_short = s['type'].replace('endpoint', 'E').replace('junction', 'J').replace('continuation', 'C')
        cv2.putText(canvas, f"R{lbl}: {s['pixels']}px [{t_short}]", (42, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        y_off += 28

    if num_branches > max_items:
        cv2.putText(canvas, f"... +{num_branches - max_items} mas", (16, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        y_off += 24

    y_off += 8
    if show_junctions:
        cv2.circle(canvas, (24, y_off), 5, (255, 255, 255), -1)
        cv2.putText(canvas, "Junction (bifurcacion)", (36, y_off + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y_off += 22
    if show_endpoints:
        cv2.circle(canvas, (24, y_off), 4, (0, 200, 255), -1)
        cv2.putText(canvas, "Endpoint (punta de rama)", (36, y_off + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Guardar ---
    if output_path is None:
        base = skeleton_path.rsplit('.', 1)[0]
        output_path = f"{base}_branches_colored.png"
    cv2.imwrite(output_path, canvas)
    print(f"[OK] Imagen guardada: {output_path}")

    # --- Comparación ---
    side_path = output_path.rsplit('.', 1)[0] + '_comparison.png'
    orig_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h1, w1 = orig_3ch.shape[:2]
    h2, w2 = canvas.shape[:2]
    if h1 != h2:
        orig_3ch = cv2.resize(orig_3ch, (int(w1 * h2 / h1), h2))
    sep = np.ones((canvas.shape[0], 4, 3), dtype=np.uint8) * 128
    comp = np.hstack([orig_3ch, sep, canvas])
    cv2.putText(comp, "Original", (10, comp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(comp, "Ramas Identificadas",
                (orig_3ch.shape[1] + 14, comp.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.imwrite(side_path, comp)
    print(f"[OK] Comparación: {side_path}")

    return canvas, branches, stats


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Identifica y colorea ramas de esqueletos de durazno (v2 - grafo)")
    parser.add_argument("input", help="Ruta a la imagen del esqueleto")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--min-branch", "-mb", type=int, default=5)
    parser.add_argument("--thickness", "-t", type=int, default=2)
    parser.add_argument("--no-junctions", action="store_true")
    parser.add_argument("--no-endpoints", action="store_true")
    parser.add_argument("--light-bg", action="store_true")
    args = parser.parse_args()

    identify_branches(
        skeleton_path=args.input,
        output_path=args.output,
        min_branch_size=args.min_branch,
        line_thickness=args.thickness,
        show_junctions=not args.no_junctions,
        show_endpoints=not args.no_endpoints,
        dark_background=not args.light_bg
    )


if __name__ == "__main__":
    main()