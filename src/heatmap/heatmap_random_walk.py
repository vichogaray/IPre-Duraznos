"""
Heatmap de Densidad Floral — Mapa de calor por subsecciones de rama (Random Walk)
==================================================================================
Divide cada rama del arbol en subsecciones a lo largo de su eje medial
y colorea cada subseccion segun la cantidad de flores asignadas via Random Walk,
usando un espectro de azul (baja densidad) a rojo (alta densidad).

Input:
  - Carpeta de JSONs de flores (JSON_FLORES_DIR)
    (el grafo y la imagen se detectan automaticamente desde el frame)

Output:
  - Una imagen por frame en OUTPUT_DIR con ramas coloreadas por densidad

Uso:
    Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"

GRAPH_JSON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
OUTPUT_DIR      = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidades\heatmap original"

SECTION_LENGTH   = 30    # Pixeles de esqueleto por subseccion (mas bajo = mas detalle)
BRANCH_THICKNESS = 7     # Grosor visual de las ramas en pixeles (1 = sin engrosar)
DARK_BG          = True

# Parametros Random Walk
K_NEIGHBORS = 3      # ramas mas cercanas por flor
K_FLOWER    = 5      # flores vecinas por flor (0 = desactivar)
SIGMA       = 50.0   # decaimiento de peso: w = exp(-dist / SIGMA)

# =====================================================================
#  NO MODIFICAR DEBAJO
# =====================================================================

import os
import re
import glob
import json
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, eye as speye, diags
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
#  1. Cargar datos
# ---------------------------------------------------------------------------

def load_flowers(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [(float(s['points'][0][0]), float(s['points'][0][1]))
            for s in data.get('shapes', [])
            if s.get('label') == 'flower' and s.get('points')]


def auto_detect_graph_json(flower_json_path, graph_json_dir):
    """frame100.json -> imgs_frame100_*_graph.json"""
    match = re.search(r'frame(\d+)', os.path.basename(flower_json_path))
    if not match:
        raise ValueError(f"No se pudo extraer frame de: {flower_json_path}")
    frame = match.group(1)
    candidates = glob.glob(os.path.join(graph_json_dir, f"*frame{frame}_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontro grafo JSON para frame{frame} en {graph_json_dir}")
    return candidates[0]


def auto_detect_graph_image(image_name, img_dir):
    path = os.path.join(img_dir, image_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No se encontro imagen: {path}")


# ---------------------------------------------------------------------------
#  2. Subsecciones de rama via eje medial
# ---------------------------------------------------------------------------

def _nn_chain(points):
    """Ordena puntos en cadena por vecino mas cercano."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n <= 1:
        return pts
    visited = np.zeros(n, dtype=bool)
    # Iniciar desde el punto mas lejano al centroide (probable extremo)
    centroid = pts.mean(axis=0)
    start = np.argmax(np.sum((pts - centroid) ** 2, axis=1))
    order = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = pts[order[-1]]
        dists = np.sum((pts - cur) ** 2, axis=1)
        dists[visited] = np.inf
        nxt = np.argmin(dists)
        order.append(nxt)
        visited[nxt] = True
    return pts[np.array(order)]


def compute_subsections(branch_pixels, img_shape, section_length):
    """
    Divide una rama en subsecciones a lo largo de su eje medial.

    Proceso:
      1. Esqueletiza los pixeles de la rama -> eje central (1 px ancho)
      2. Ordena el esqueleto en cadena (nearest-neighbor)
      3. Divide la cadena en segmentos de section_length pixeles
      4. Asigna cada pixel de la rama al segmento mas cercano via cKDTree
    """
    pts = np.array(branch_pixels, dtype=np.int32)  # [y, x]
    if len(pts) < 3:
        return [pts]

    # Filtrar pixeles dentro de los limites de la imagen
    valid = ((pts[:, 0] >= 0) & (pts[:, 0] < img_shape[0]) &
             (pts[:, 1] >= 0) & (pts[:, 1] < img_shape[1]))
    pts = pts[valid]
    if len(pts) < 3:
        return [pts]

    # Esqueletizar para obtener el eje medial
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask[pts[:, 0], pts[:, 1]] = 1
    skel = skeletonize(mask > 0)
    skel_yx = np.argwhere(skel)

    if len(skel_yx) < 2:
        return [pts]

    n_subs = max(1, len(skel_yx) // section_length)

    # Ordenar esqueleto en cadena
    ordered = _nn_chain(skel_yx)

    # Etiquetar cada pixel del esqueleto con su segmento
    seg_len = len(ordered) // n_subs
    seg_labels = np.zeros(len(ordered), dtype=int)
    for i in range(n_subs):
        s = i * seg_len
        e = (i + 1) * seg_len if i < n_subs - 1 else len(ordered)
        seg_labels[s:e] = i

    # Asignar cada pixel de la rama al segmento mas cercano del esqueleto
    tree = cKDTree(ordered)
    _, indices = tree.query(pts.astype(np.float64))
    px_labels = seg_labels[indices]

    subsections = []
    for s in range(n_subs):
        sub = pts[px_labels == s]
        if len(sub) > 0:
            subsections.append(sub)

    return subsections if subsections else [pts]


# ---------------------------------------------------------------------------
#  3. Asignacion de flores a ramas via Random Walk
# ---------------------------------------------------------------------------

def assign_flowers_random_walk(flowers, graph_data, k_neighbors, k_flower, sigma):
    """
    Asigna flores a ramas usando Random Walk (Grady 2006).
    Retorna array (n_f,) con el indice de rama por flor.
    """
    branches = graph_data['branches']
    edges    = graph_data['edges']
    n_b = len(branches)
    n_f = len(flowers)
    N   = n_b + n_f

    bid_to_idx = {b['id']: i for i, b in enumerate(branches)}
    A = lil_matrix((N, N), dtype=np.float64)

    # Rama-rama
    for edge in edges:
        i = bid_to_idx.get(edge['from'])
        j = bid_to_idx.get(edge['to'])
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Flor-rama
    branch_pixels = [np.array(b['pixels'], dtype=np.float32) for b in branches]
    for fi, (fx, fy) in enumerate(flowers):
        fi_idx = n_b + fi
        dists = []
        for bi, px in enumerate(branch_pixels):
            d = float(np.sqrt(np.min((px[:, 1] - fx)**2 + (px[:, 0] - fy)**2)))
            dists.append((d, bi))
        dists.sort()
        for d, bi in dists[:k_neighbors]:
            w = np.exp(-d / sigma)
            A[fi_idx, bi] = max(A[fi_idx, bi], w)
            A[bi, fi_idx] = max(A[bi, fi_idx], w)

    # Flor-flor
    if k_flower > 0 and n_f > 1:
        flower_xy = np.array(flowers, dtype=np.float64)
        tree = cKDTree(flower_xy)
        k_q = min(k_flower + 1, n_f)
        dists_knn, idxs_knn = tree.query(flower_xy, k=k_q)
        for fi in range(n_f):
            for d, nb_fi in zip(dists_knn[fi][1:], idxs_knn[fi][1:]):
                fi_idx  = n_b + fi
                nfi_idx = n_b + int(nb_fi)
                w = np.exp(-float(d) / sigma)
                A[fi_idx, nfi_idx] = max(A[fi_idx, nfi_idx], w)
                A[nfi_idx, fi_idx] = max(A[nfi_idx, fi_idx], w)

    A = A.tocsr()
    deg = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    deg[deg == 0] = 1.0
    T = diags(1.0 / deg).dot(A)

    Q  = T[n_b:, n_b:]
    R  = T[n_b:, :n_b]
    IQ = speye(n_f, format='csr') - Q.tocsr()
    R_dense = np.array(R.tocsc().todense(), dtype=np.float64)
    F = spsolve(IQ, R_dense)

    if F.ndim == 1:
        F = F[:, np.newaxis]
    F = np.clip(F, 0.0, None)
    rs = F.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    F /= rs

    return np.argmax(F, axis=1).astype(int)


def count_flowers_per_subsection(flowers, assignments, all_subsections):
    """Cuenta flores por subseccion (dentro de la rama asignada)."""
    counts = {}
    for bi in range(len(all_subsections)):
        for si in range(len(all_subsections[bi])):
            counts[(bi, si)] = 0

    for fi, (fx, fy) in enumerate(flowers):
        bi = assignments[fi]
        subs = all_subsections[bi]
        best_si, best_d = 0, np.inf
        for si, sub_px in enumerate(subs):
            arr = np.array(sub_px, dtype=np.float32)
            d = np.min((arr[:, 1] - fx) ** 2 + (arr[:, 0] - fy) ** 2)
            if d < best_d:
                best_d, best_si = d, si
        counts[(bi, best_si)] += 1

    return counts


# ---------------------------------------------------------------------------
#  4. Renderizado del mapa de calor
# ---------------------------------------------------------------------------

def render_heatmap(img_shape, all_subsections, counts, dark_bg, thickness=7):
    """Genera la imagen BGR del mapa de calor con ramas engrosadas."""
    max_d = max(counts.values()) if counts else 1
    if max_d == 0:
        max_d = 1

    cmap = plt.colormaps['jet']
    norm = Normalize(vmin=0, vmax=max_d)

    bg = 30 if dark_bg else 240
    heatmap = np.full((img_shape[0], img_shape[1], 3), bg, dtype=np.uint8)

    kernel = None
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))

    for bi, subs in enumerate(all_subsections):
        for si, sub_px in enumerate(subs):
            density = counts.get((bi, si), 0)
            rgba = cmap(norm(density))
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            arr = np.array(sub_px, dtype=np.int32)

            sub_mask = np.zeros(img_shape[:2], dtype=np.uint8)
            sub_mask[arr[:, 0], arr[:, 1]] = 255
            if kernel is not None:
                sub_mask = cv2.dilate(sub_mask, kernel)
            heatmap[sub_mask > 0] = bgr

    return heatmap, cmap, norm


def save_figure(heatmap_bgr, cmap, norm, save_path, dark_bg):
    """Guarda la imagen con barra de color."""
    txt = 'white' if dark_bg else 'black'
    bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [20, 1]})
    fig.patch.set_facecolor(bg)

    ax1.imshow(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title('Mapa de Calor — Densidad Floral', color=txt, fontsize=13)
    ax1.axis('off')
    ax1.set_facecolor(bg)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax2)
    cb.set_label('Flores por subseccion', color=txt, fontsize=11)
    cb.ax.yaxis.set_tick_params(color=txt)
    cb.outline.set_edgecolor(txt)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=txt)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
#  5. Pipeline
# ---------------------------------------------------------------------------

def run(flower_json_path):
    print(f"  Flores: {os.path.basename(flower_json_path)}")

    # Cargar flores
    flowers = load_flowers(flower_json_path)
    print(f"  Flores cargadas: {len(flowers)}")
    if not flowers:
        print("  [SKIP] Sin flores.")
        return

    # Auto-detectar grafo JSON e imagen
    graph_json_path = auto_detect_graph_json(flower_json_path, GRAPH_JSON_DIR)
    print(f"  Grafo JSON: {os.path.basename(graph_json_path)}")

    with open(graph_json_path, encoding='utf-8') as f:
        graph_data = json.load(f)

    image_name = graph_data['image']
    img_path   = auto_detect_graph_image(image_name, GRAFOS_IMG_DIR)
    img        = cv2.imread(img_path)
    branches   = graph_data['branches']
    print(f"  Ramas: {len(branches)}")

    # Asignar flores a ramas via Random Walk
    assignments = assign_flowers_random_walk(
        flowers, graph_data, K_NEIGHBORS, K_FLOWER, SIGMA)

    # Calcular subsecciones para cada rama
    all_subsections = []
    for b in branches:
        subs = compute_subsections(b['pixels'], img.shape, SECTION_LENGTH)
        all_subsections.append(subs)

    # Contar flores por subseccion
    counts = count_flowers_per_subsection(flowers, assignments, all_subsections)

    # Renderizar mapa de calor
    heatmap, cmap, norm = render_heatmap(img.shape, all_subsections, counts,
                                         DARK_BG, BRANCH_THICKNESS)

    # Guardar
    frame_match = re.search(r'frame(\d+)', os.path.basename(flower_json_path))
    fname = f"frame{frame_match.group(1)}" if frame_match else "heatmap"
    save_path = os.path.join(OUTPUT_DIR, fname + "_heatmap.png")
    save_figure(heatmap, cmap, norm, save_path, DARK_BG)
    print(f"  [{fname}] -> {save_path}")


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flower_jsons = sorted(glob.glob(os.path.join(JSON_FLORES_DIR, "*.json")))
    if not flower_jsons:
        print(f"[ERROR] No se encontraron JSONs en: {JSON_FLORES_DIR}")
        raise SystemExit(1)

    print(f"[INFO] {len(flower_jsons)} JSONs de flores encontrados\n")

    ok, skipped = 0, 0
    for i, fpath in enumerate(flower_jsons, 1):
        print(f"[{i:3d}/{len(flower_jsons)}] {os.path.basename(fpath)}", end=' ... ', flush=True)
        try:
            run(fpath)
            ok += 1
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP ({e})")
            skipped += 1
        except Exception as e:
            print(f"ERROR ({e})")
            skipped += 1

    print(f"\n[DONE] {ok} heatmaps guardados, {skipped} omitidos -> {OUTPUT_DIR}")
