"""
GT Heatmap Compare - Trunk heatmap del GT vs. de un metodo + similitud
=======================================================================
1. Construye el mapa de calor del esqueleto para el GROUND TRUTH
   (output de gt_annotator.py) usando EXACTAMENTE el mismo mecanismo y
   los mismos parametros que Varilla_heatmap.py:
        heat(p) = sum_v num_flowers(v) * exp(-||p - base_xy(v)||^2 / 2sigma^2)
   sobre los pixeles del esqueleto.
2. Construye el mismo heatmap para un METODO (cualquier JSON con
   varillas[].base_xy + num_flowers: p.ej. el output de varilla_density.py).
3. Los compara con distancia de WASSERSTEIN GEODESICA "a lo largo del
   esqueleto": el costo de mover una unidad de calor de un pixel a otro es
   la distancia recorriendo las ramas (geodesica sobre el grafo de pixeles
   del esqueleto, 8-conexo), NO la distancia euclidiana en el aire.
   Se resuelve con Sinkhorn (Wasserstein regularizado) para que escale.

   Salida: la distancia de Wasserstein en cm (cuanto calor hay que mover,
   y cuan lejos, para convertir un perfil en el otro) + una similitud
   normalizada en [0, 1]. Ademas se puede guardar un panel visual
   GT | metodo | diferencia.

USO:
  1. Ajusta GT_JSON y METHOD_JSON abajo (cambia el input aqui).
  2. F5 / python gt_heatmap_compare.py
"""

# =====================================================================
#  CONFIGURACION  (cambia el input aqui)
# =====================================================================

BASE_DIR       = r"C:\Users\vgara\OneDrive\Desktop\IPre"
GRAPH_JSON_DIR = BASE_DIR + r"\grafos json"
GRAFOS_IMG_DIR = BASE_DIR + r"\Grafos"

# Ground truth (output de gt_annotator.py)
GT_JSON     = r"C:\Users\vgara\OneDrive\Desktop\GT\imgs_frame32_00000_graph_GT.json"

# Metodo a contrastar (cualquier JSON con varillas[].base_xy + num_flowers)
METHOD_JSON = BASE_DIR + r"\densidades\densidad floral varilla\json\imgs_frame32_00000_graph_varilla.json"
METHOD_NAME = "varilla_density"

# Salida del panel visual. None -> muestra en pantalla. "auto" -> se genera
# desde el nombre de la imagen (Desktop\GT\<imagen>_compare.png).
OUTPUT_PNG  = "auto"

# CSV acumulativo: cada corrida agrega/actualiza una fila (imagen+metodo).
# None -> no se escribe CSV.
RESULTS_CSV = r"C:\Users\vgara\OneDrive\Desktop\GT\comparaciones.csv"

# === PARAMETROS DE CALOR (deben coincidir con Varilla_heatmap.py) ===
ASSUMED_TREE_HEIGHT_CM = 350.0
PIXELS_PER_CM_OVERRIDE = None
KERNEL_SIGMA_CM        = 8.0

# === WASSERSTEIN GEODESICO ===
# El esqueleto puede tener miles de pixeles -> Dijkstra all-pairs y Sinkhorn
# a esa escala son caros. Submuestreamos el esqueleto a ~N nodos manteniendo
# la geodesica (los nodos se eligen equiespaciados sobre las ramas). El calor
# se agrega al nodo submuestreado mas cercano por indice de esqueleto.
TARGET_NODES     = 600      # nº de nodos tras submuestreo (sube = mas fiel, mas lento)
SINKHORN_REG     = None     # regularizacion entropica; None -> auto (=media dist * 0.05)
SINKHORN_ITERS   = 400
# Conectividad del grafo geodesico: se construye sobre TODOS los pixeles del
# esqueleto (8-conexo -> topologia fiel de las ramas), NO sobre los nodos
# submuestreados. Este gap conecta pixeles vecinos; sqrt(2)+eps basta para
# 8-conexion, se sube un poco para tolerar micro-huecos del esqueleto.
GEO_CONNECT_GAP_PX = 3.0

# === VISUAL ===
HEAT_CMAP = 'jet'
VMAX_FIJO = 60.0
SKEL_THICKNESS_PX = 3

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import json
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# =====================================================================
#  I/O + escala + esqueleto  (identico a Varilla_heatmap.py)
# =====================================================================

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# La calibracion px <-> cm vive en src/common/geometry.py para que exista una
# sola definicion compartida por todos los metodos. Se anade la raiz del
# repositorio al path para poder ejecutar este archivo directamente (F5).
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.geometry import estimate_pixels_per_cm


def build_skeleton_pixels(graph_data):
    pts = [(px[1], px[0]) for b in graph_data['branches'] for px in b['pixels']]
    return np.asarray(pts, dtype=np.float64)


def compute_heat(skel_xy, varillas, sigma_px):
    """Idem Varilla_heatmap.compute_heat: heat por pixel del esqueleto."""
    n = len(skel_xy)
    heat = np.zeros(n, dtype=np.float64)
    if n == 0 or sigma_px <= 0:
        return heat
    inv_2s2 = 1.0 / (2.0 * sigma_px * sigma_px)
    for v in varillas:
        if v.get('base_xy') is None:
            continue
        bx, by = v['base_xy']
        w = float(v.get('num_flowers', 0))
        if w <= 0:
            continue
        dx = skel_xy[:, 0] - bx
        dy = skel_xy[:, 1] - by
        heat += w * np.exp(-(dx * dx + dy * dy) * inv_2s2)
    return heat


# =====================================================================
#  SUBMUESTREO DEL ESQUELETO + GEODESICA
# =====================================================================

def subsample_skeleton(skel_xy, target_nodes):
    """
    Selecciona ~target_nodes pixeles del esqueleto de forma espacialmente
    uniforme (submuestreo por rejilla). Devuelve indices en skel_xy.
    """
    n = len(skel_xy)
    if n <= target_nodes:
        return np.arange(n)
    # rejilla adaptativa: como el esqueleto es ~1D (lineas finas), el nº de
    # celdas OCUPADAS crece ~lineal con 1/cell, no cuadratico. Ajustamos el
    # tamaño de celda por busqueda para acercarnos a target_nodes celdas.
    xmin, ymin = skel_xy.min(0)
    xmax, ymax = skel_xy.max(0)
    span = max(xmax - xmin, ymax - ymin, 1.0)

    def cells_for(cell):
        keys = {}
        inv = 1.0 / cell
        for i, (x, y) in enumerate(skel_xy):
            keys.setdefault((int((x - xmin) * inv), int((y - ymin) * inv)), []).append(i)
        return keys

    lo, hi = span / (target_nodes * 4.0), span / 2.0
    keys = None
    for _ in range(20):
        cell = 0.5 * (lo + hi)
        keys = cells_for(cell)
        k = len(keys)
        if k > target_nodes * 1.1:
            lo = cell          # celdas mas grandes -> menos nodos
        elif k < target_nodes * 0.9:
            hi = cell
        else:
            break
    idx = [group[len(group) // 2] for group in keys.values()]
    return np.asarray(sorted(idx), dtype=np.int64)


def geodesic_distance_matrix(skel_xy, node_idx, connect_gap_px):
    """
    Geodesica FIEL: construye el grafo sobre TODOS los pixeles del esqueleto
    (8-conexo, arista = dist euclidiana entre pixeles vecinos a <= gap), corre
    Dijkstra desde los nodos submuestreados, y devuelve la submatriz de
    distancias geodesicas ENTRE nodos submuestreados (px).

    Asi la distancia respeta la topologia real de las ramas (caminar por el
    esqueleto), sin saltar entre ramas que solo pasan cerca "por el aire".
    """
    n = len(skel_xy)
    tree = cKDTree(skel_xy)
    pairs = tree.query_pairs(r=connect_gap_px)
    rows, cols, data = [], [], []
    for i, j in pairs:
        d = float(np.hypot(*(skel_xy[i] - skel_xy[j])))
        rows += [i, j]; cols += [j, i]; data += [d, d]
    if not data:
        m = len(node_idx)
        return np.full((m, m), np.inf), 0
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Dijkstra solo desde los nodos submuestreados -> filas = node_idx
    D_full = dijkstra(A, directed=False, indices=node_idx)   # (m, n)
    D = D_full[:, node_idx]                                   # (m, m)
    # reparar desconexiones residuales del esqueleto: inf -> 2x diametro finito
    finite = D[np.isfinite(D)]
    if finite.size:
        big = 2.0 * float(finite.max())
        D[~np.isfinite(D)] = big
    return D, len(pairs)


def aggregate_heat_to_nodes(skel_xy, heat, node_idx):
    """
    Reparte el calor de todos los pixeles del esqueleto al nodo submuestreado
    mas cercano. Devuelve vector de calor por nodo (mismo total que heat.sum()).
    """
    nodes_xy = skel_xy[node_idx]
    tree = cKDTree(nodes_xy)
    _, nn = tree.query(skel_xy)
    agg = np.zeros(len(node_idx), dtype=np.float64)
    np.add.at(agg, nn, heat)
    return agg, nodes_xy


# =====================================================================
#  WASSERSTEIN (SINKHORN) SOBRE LA MATRIZ GEODESICA
# =====================================================================

def sinkhorn_wasserstein(a, b, D, reg, iters):
    """
    Wasserstein regularizado (Sinkhorn) entre distribuciones a y b (se
    normalizan a suma 1) con matriz de costos D. Devuelve el costo de
    transporte <P, D> (en las unidades de D, aqui px).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return np.nan, None
    a = a / sa
    b = b / sb
    K = np.exp(-D / reg)
    # estabilidad: evita ceros exactos
    K = np.maximum(K, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    P = (u[:, None] * K) * v[None, :]
    cost = float(np.sum(P * D))
    return cost, P


# =====================================================================
#  PIPELINE DE COMPARACION
# =====================================================================

def build_heat_for(varilla_data, skel_xy, sigma_px):
    return compute_heat(skel_xy, varilla_data.get('varillas', []), sigma_px)


def compare(gt_data, method_data, graph_data):
    if PIXELS_PER_CM_OVERRIDE is not None:
        px_per_cm = float(PIXELS_PER_CM_OVERRIDE)
    else:
        px_per_cm = estimate_pixels_per_cm(graph_data, ASSUMED_TREE_HEIGHT_CM)
    sigma_px = KERNEL_SIGMA_CM * px_per_cm

    skel_xy = build_skeleton_pixels(graph_data)
    heat_gt = build_heat_for(gt_data, skel_xy, sigma_px)
    heat_me = build_heat_for(method_data, skel_xy, sigma_px)

    # submuestreo + geodesica (una sola vez, compartida)
    node_idx = subsample_skeleton(skel_xy, TARGET_NODES)
    D_px, n_edges = geodesic_distance_matrix(skel_xy, node_idx, GEO_CONNECT_GAP_PX)
    a, nodes_xy = aggregate_heat_to_nodes(skel_xy, heat_gt, node_idx)
    b, _        = aggregate_heat_to_nodes(skel_xy, heat_me, node_idx)

    reg = SINKHORN_REG
    if reg is None:
        finite = D_px[np.isfinite(D_px)]
        reg = float(np.median(finite[finite > 0])) * 0.05 if finite.size else 1.0

    # Sinkhorn DEBIASADO (divergencia de Sinkhorn): resta los auto-transportes
    # para eliminar el piso/sesgo que introduce la regularizacion entropica, de
    # modo que distribuciones identicas den W = 0.
    W_ab, P = sinkhorn_wasserstein(a, b, D_px, reg, SINKHORN_ITERS)
    W_aa, _ = sinkhorn_wasserstein(a, a, D_px, reg, SINKHORN_ITERS)
    W_bb, _ = sinkhorn_wasserstein(b, b, D_px, reg, SINKHORN_ITERS)
    W_px = W_ab - 0.5 * W_aa - 0.5 * W_bb
    W_px = max(W_px, 0.0) if np.isfinite(W_px) else np.nan
    W_cm = W_px / px_per_cm if np.isfinite(W_px) else np.nan

    # similitud normalizada: 0 cm -> 1.  Escala por el "diametro" del arbol.
    diam_px = float(D_px[np.isfinite(D_px)].max()) if np.isfinite(D_px).any() else 1.0
    diam_cm = diam_px / px_per_cm
    sim = float(np.clip(1.0 - (W_cm / diam_cm), 0.0, 1.0)) if np.isfinite(W_cm) else np.nan

    # metricas auxiliares (baratas) sobre el heat completo del esqueleto
    hg = heat_gt / (heat_gt.sum() + 1e-12)
    hm = heat_me / (heat_me.sum() + 1e-12)
    cosine = float(np.dot(hg, hm) / (np.linalg.norm(hg) * np.linalg.norm(hm) + 1e-12))
    if hg.std() > 0 and hm.std() > 0:
        pearson = float(np.corrcoef(heat_gt, heat_me)[0, 1])
    else:
        pearson = np.nan

    result = {
        'px_per_cm': px_per_cm, 'sigma_px': sigma_px,
        'n_skel_px': len(skel_xy), 'n_nodes': len(node_idx), 'n_edges': n_edges,
        'reg': reg, 'diam_cm': diam_cm,
        'W_px': W_px, 'W_cm': W_cm, 'similitud': sim,
        'cosine': cosine, 'pearson': pearson,
        'gt_flores': gt_data.get('n_flowers'),
        'method_flores': method_data.get('n_flowers'),
    }
    viz = {'skel_xy': skel_xy, 'heat_gt': heat_gt, 'heat_me': heat_me,
           'px_per_cm': px_per_cm}
    return result, viz


# =====================================================================
#  VISUALIZACION (panel GT | metodo | diferencia)
# =====================================================================

def paint_skeleton(img_rgb, skel_xy, heat, vmax):
    rgb = img_rgb.astype(np.float64) / 255.0
    h_img, w_img = rgb.shape[:2]
    norm = Normalize(vmin=0.0, vmax=vmax)
    colors = plt.get_cmap(HEAT_CMAP)(norm(heat))[:, :3]
    xs = np.round(skel_xy[:, 0]).astype(int)
    ys = np.round(skel_xy[:, 1]).astype(int)
    order = np.argsort(heat)
    xs, ys, colors = xs[order], ys[order], colors[order]
    r = (max(1, SKEL_THICKNESS_PX) - 1) // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            yo, xo = ys + dy, xs + dx
            m = (yo >= 0) & (yo < h_img) & (xo >= 0) & (xo < w_img)
            rgb[yo[m], xo[m]] = colors[m]
    return rgb


def visualize(viz, result, image_name, save_path):
    img_path = os.path.join(GRAFOS_IMG_DIR, image_name)
    bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    skel = viz['skel_xy']; hg = viz['heat_gt']; hm = viz['heat_me']
    vmax = VMAX_FIJO if VMAX_FIJO else max(hg.max(), hm.max(), 1.0)
    diff = np.abs(hg - hm)

    rgb_gt = paint_skeleton(base.copy(), skel, hg, vmax)
    rgb_me = paint_skeleton(base.copy(), skel, hm, vmax)
    rgb_df = paint_skeleton(base.copy(), skel, diff, vmax)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor('#1e1e1e')
    for ax, rgb, ttl in zip(
            axes, [rgb_gt, rgb_me, rgb_df],
            ['Ground Truth',
             'Metodo: {0}'.format(METHOD_NAME),
             '|GT - metodo|']):
        ax.imshow(rgb); ax.axis('off')
        ax.set_title(ttl, color='white', fontsize=12)

    supt = ("Wasserstein geodesico = {W:.1f} cm   |   similitud = {S:.3f}   |   "
            "cos = {c:.3f}   pearson = {p:.3f}\n"
            "GT flores = {gf}   metodo flores = {mf}   |   nodos = {nn}").format(
        W=result['W_cm'], S=result['similitud'], c=result['cosine'],
        p=result['pearson'], gf=result['gt_flores'], mf=result['method_flores'],
        nn=result['n_nodes'])
    fig.suptitle(supt, color='white', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight', facecolor='#1e1e1e')
        print("[OK] Panel guardado -> " + save_path)
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
#  MAIN
# =====================================================================

def append_csv(csv_path, image_name, method_name, gt_json, result):
    """
    Agrega (o ACTUALIZA) una fila en el CSV acumulativo. La clave es
    (image, metodo): si ya existe una fila con ese par, se reemplaza; si no,
    se agrega. Asi reanotar/recomparar no duplica filas.
    """
    import csv
    from datetime import datetime

    fields = ['image', 'metodo', 'wasserstein_cm', 'similitud',
              'cosine', 'pearson', 'gt_flores', 'metodo_flores',
              'diam_cm', 'px_per_cm', 'n_nodos', 'gt_json', 'timestamp']
    row = {
        'image': image_name,
        'metodo': method_name,
        'wasserstein_cm': round(result['W_cm'], 3),
        'similitud': round(result['similitud'], 4),
        'cosine': round(result['cosine'], 4),
        'pearson': (round(result['pearson'], 4)
                    if np.isfinite(result['pearson']) else ''),
        'gt_flores': result['gt_flores'],
        'metodo_flores': result['method_flores'],
        'diam_cm': round(result['diam_cm'], 1),
        'px_per_cm': round(result['px_per_cm'], 3),
        'n_nodos': result['n_nodes'],
        'gt_json': os.path.basename(gt_json),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # leer filas existentes (si el CSV ya existe)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))

    # upsert por (image, metodo)
    replaced = False
    for r in rows:
        if r.get('image') == row['image'] and r.get('metodo') == row['metodo']:
            r.clear()
            r.update({k: str(v) for k, v in row.items()})
            replaced = True
            break
    if not replaced:
        rows.append({k: str(v) for k, v in row.items()})

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    action = "actualizada" if replaced else "agregada"
    print("  [CSV] fila {0}: {1} / {2}  ->  {3}".format(
        action, row['image'], row['metodo'], os.path.basename(csv_path)))


def resolve_graph_json(gt_data):
    gj = gt_data.get('graph_json')
    if gj:
        p = os.path.join(GRAPH_JSON_DIR, gj)
        if os.path.exists(p):
            return p
    # fallback: derivar del nombre de imagen
    base = os.path.splitext(gt_data['image'])[0]
    p = os.path.join(GRAPH_JSON_DIR, base + ".json")
    if os.path.exists(p):
        return p
    raise FileNotFoundError("No se pudo resolver el graph JSON del GT.")


if __name__ == "__main__":
    gt_data = load_json(GT_JSON)
    method_data = load_json(METHOD_JSON)
    graph_path = resolve_graph_json(gt_data)
    graph_data = load_json(graph_path)

    print("[INFO] GT     :", os.path.basename(GT_JSON),
          "  ({0} puntos, {1} flores)".format(
              gt_data.get('n_points', '?'), gt_data.get('n_flowers', '?')))
    print("[INFO] Metodo :", os.path.basename(METHOD_JSON),
          "  ({0} varillas, {1} flores)".format(
              len(method_data.get('varillas', [])), method_data.get('n_flowers', '?')))

    result, viz = compare(gt_data, method_data, graph_data)

    print("\n===== RESULTADO =====")
    print("  Escala           : {0:.2f} px/cm   sigma = {1:.1f} px".format(
        result['px_per_cm'], result['sigma_px']))
    print("  Esqueleto        : {0} px  ->  {1} nodos ({2} aristas)".format(
        result['n_skel_px'], result['n_nodes'], result['n_edges']))
    print("  Diametro arbol   : {0:.1f} cm".format(result['diam_cm']))
    print("  --------------------------------------------------")
    print("  WASSERSTEIN      : {0:.2f} cm   (calor movido x distancia)".format(
        result['W_cm']))
    print("  SIMILITUD [0-1]  : {0:.3f}".format(result['similitud']))
    print("  cosine / pearson : {0:.3f} / {1:.3f}".format(
        result['cosine'], result['pearson']))

    # CSV acumulativo (upsert por imagen+metodo)
    if RESULTS_CSV:
        append_csv(RESULTS_CSV, gt_data['image'], METHOD_NAME, GT_JSON, result)

    # resolver ruta del panel visual
    out_png = OUTPUT_PNG
    if out_png == "auto":
        base = os.path.splitext(gt_data['image'])[0]
        out_png = os.path.join(os.path.dirname(RESULTS_CSV or GT_JSON),
                               base + "_compare.png")
    visualize(viz, result, gt_data['image'], out_png)
