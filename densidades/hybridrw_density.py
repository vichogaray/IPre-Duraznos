"""
Pipeline HybridRW de Densidad Floral
======================================
Euclidiana para flores con rama claramente dominante (ratio d2/d1 > UMBRAL)
Random Walk para flores ambiguas entre dos ramas   (ratio d2/d1 <= UMBRAL)

Referencia Random Walk: Grady 2006 (Random Walks for Image Segmentation)

Uso: F5
"""

# =====================================================================
#  PARAMETROS
# =====================================================================
GRAPH_JSON_DIR   = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIR   = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
JSON_FLORES_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"
OUTPUT_DIR       = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidades\densidad floral hybridrw"

UMBRAL_AMBIGUEDAD = 1.5   # ratio d2/d1: sobre este umbral -> euclidiana directa

K_VECINOS_RW  = 3      # ramas mas cercanas que conectan cada flor ambigua al grafo
K_FLOWER_RW   = 5      # flores ambiguas vecinas entre si (0 = desactivar)
SIGMA_RW      = 50.0   # decaimiento del peso: w = exp(-dist / SIGMA)

FLOWER_SIZE = 4      # tamano del marcador en el plot
DARK_BG     = True

# None = procesar todas las imagenes; o nombre del JSON para procesar una sola
SINGLE_IMAGE = None  # ej: "imgs_frame1_00000_graph.json"


# =====================================================================
#  IMPORTS
# =====================================================================
import os
import re
import glob
import json
import numpy as np
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import lil_matrix, eye as speye, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree as KDTree


# =====================================================================
#  1. CARGA DE DATOS
# =====================================================================
def load_graph_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_flowers(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return [(float(s['points'][0][0]), float(s['points'][0][1]))
            for s in data.get('shapes', [])
            if s.get('label') == 'flower' and s.get('points')]


def auto_detect_flower_json(image_name, json_flores_dir):
    match = re.search(r'frame(\d+)', image_name)
    if not match:
        raise ValueError(f"No se pudo extraer frame de: {image_name}")
    path = os.path.join(json_flores_dir, f"frame{match.group(1)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro: {path}")
    return path


def auto_detect_graph_image(image_name, grafos_img_dir):
    path = os.path.join(grafos_img_dir, image_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No se encontro imagen: {path}")


# =====================================================================
#  2. DISTANCIAS FLOR -> RAMA
#     Cada rama tiene miles de pixeles del esqueleto;
#     la distancia es el minimo a cualquier pixel de la rama.
# =====================================================================
def compute_min_distances(flowers, branches):
    """
    Calcula la distancia minima de cada flor a cada rama.

    Retorna:
        dist_matrix : np.array (n_flores, n_ramas) con distancias en px
        branch_ids  : lista de branch ids, en el mismo orden que las columnas
    """
    branch_ids    = [b['id'] for b in branches]
    branch_pixels = [np.array(b['pixels'], dtype=np.float32) for b in branches]

    n_f = len(flowers)
    n_b = len(branches)
    dist_matrix = np.zeros((n_f, n_b), dtype=np.float32)

    for fi, (fx, fy) in enumerate(flowers):
        for bi, px in enumerate(branch_pixels):
            # pixels format: [y, x]
            d_sq = (px[:, 1] - fx) ** 2 + (px[:, 0] - fy) ** 2
            dist_matrix[fi, bi] = np.sqrt(np.min(d_sq))

    return dist_matrix, branch_ids


# =====================================================================
#  3. CLASIFICACION POR AMBIGUEDAD
# =====================================================================
def classify_ambiguity(dist_matrix, branch_ids, umbral):
    """
    Separa flores en triviales (euclidiana) y ambiguas (Random Walk).

    Retorna:
        triviales : list de indices de flores triviales
        ambiguas  : list de indices de flores ambiguas
        info      : dict {i: {'d1', 'd2', 'ratio', 'rama_ganadora' o 'rama_d1'/'rama_d2'}}
    """
    n_f = dist_matrix.shape[0]
    triviales, ambiguas = [], []
    info = {}

    for i in range(n_f):
        dists = dist_matrix[i]
        order = np.argsort(dists)

        d1   = float(dists[order[0]])
        bid1 = branch_ids[order[0]]

        if len(order) < 2:
            info[i] = {'d1': d1, 'd2': float('inf'),
                       'ratio': float('inf'), 'rama_ganadora': bid1}
            triviales.append(i)
        else:
            d2    = float(dists[order[1]])
            ratio = d2 / (d1 + 1e-6)
            bid2  = branch_ids[order[1]]
            info[i] = {'d1': d1, 'd2': d2, 'ratio': ratio,
                       'rama_d1': bid1, 'rama_d2': bid2}

            if ratio > umbral:
                info[i]['rama_ganadora'] = bid1
                triviales.append(i)
            else:
                ambiguas.append(i)

    return triviales, ambiguas, info


# =====================================================================
#  4. ASIGNACION EUCLIDIANA (casos triviales)
# =====================================================================
def assign_euclidiana(triviales, info):
    """Asigna flores triviales a la rama mas cercana. Retorna {indice: branch_id}"""
    return {i: info[i]['rama_ganadora'] for i in triviales}


# =====================================================================
#  5. ASIGNACION RANDOM WALK (casos ambiguos)
# =====================================================================
def build_rw_graph(ambiguas, flowers, graph_data, k_branches, k_flower, sigma):
    """
    Construye la matriz de adyacencia para el Random Walk semi-supervisado.
    Solo las flores ambiguas se agregan como nodos no-absorbentes.

    Estructura de filas/columnas:
        0 .. n_b-1       : ramas (nodos absorbentes / etiquetados)
        n_b .. N-1       : flores ambiguas (nodos no-absorbentes)

    Tipos de aristas:
        rama-rama  : del grafo JSON, peso = 1.0
        flor-rama  : k_branches mas cercanas por distancia minima a pixeles,
                     peso = exp(-dist / sigma)
        flor-flor  : k_flower flores ambiguas mas cercanas,
                     peso = exp(-dist / sigma)
    """
    branches = graph_data['branches']
    edges    = graph_data['edges']
    n_b      = len(branches)
    n_f      = len(ambiguas)
    N        = n_b + n_f

    bid_to_idx    = {b['id']: i for i, b in enumerate(branches)}
    branch_pixels = [np.array(b['pixels'], dtype=np.float32) for b in branches]

    A = lil_matrix((N, N), dtype=np.float64)

    # --- Conexiones rama-rama desde el grafo JSON ---
    for edge in edges:
        i = bid_to_idx.get(edge['from'])
        j = bid_to_idx.get(edge['to'])
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Conexiones flor-rama: k_branches vecinos mas cercanos ---
    for fi_local, fi_global in enumerate(ambiguas):
        fx, fy     = flowers[fi_global]
        flower_idx = n_b + fi_local

        dists_bi = []
        for bi, px in enumerate(branch_pixels):
            d_sq = (px[:, 1] - fx) ** 2 + (px[:, 0] - fy) ** 2
            dists_bi.append((float(np.sqrt(np.min(d_sq))), bi))
        dists_bi.sort()

        for dist, bi in dists_bi[:k_branches]:
            w = np.exp(-dist / sigma)
            A[flower_idx, bi] = max(A[flower_idx, bi], w)
            A[bi, flower_idx] = max(A[bi, flower_idx], w)

    # --- Conexiones flor-flor: k_flower flores ambiguas mas cercanas ---
    if k_flower > 0 and n_f > 1:
        ambiguas_xy = np.array([flowers[fi] for fi in ambiguas], dtype=np.float64)
        tree    = KDTree(ambiguas_xy)
        k_query = min(k_flower + 1, n_f)
        dists_knn, idxs_knn = tree.query(ambiguas_xy, k=k_query)

        for fi_local in range(n_f):
            for dist, nb_local in zip(dists_knn[fi_local][1:], idxs_knn[fi_local][1:]):
                fi_idx  = n_b + fi_local
                nfi_idx = n_b + int(nb_local)
                w = np.exp(-float(dist) / sigma)
                A[fi_idx, nfi_idx] = max(A[fi_idx, nfi_idx], w)
                A[nfi_idx, fi_idx] = max(A[nfi_idx, fi_idx], w)

    return A.tocsr(), bid_to_idx, n_b, n_f


def run_random_walk(A, n_b, n_f):
    """
    Resuelve el problema de absorcion Random Walk (Grady 2006).

    T = D^{-1} A  (matriz de transicion estocástica por filas)
    Q = T[flores, flores]    (transiciones flor->flor)
    R = T[flores, ramas]     (transiciones flor->rama)

    Sistema: (I - Q) F = R
    F[i, j] = probabilidad de que una caminata desde flor i sea absorbida por rama j

    Retorna array (n_f,) con el indice de rama asignado a cada flor ambigua.
    """
    d = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    d[d == 0] = 1.0

    T = diags(1.0 / d).dot(A)

    Q  = T[n_b:, n_b:]       # (n_f, n_f)
    R  = T[n_b:, :n_b]       # (n_f, n_b)
    IQ = speye(n_f, format='csr') - Q.tocsr()

    print(f"    Random Walk: {n_f} flores ambiguas, resolviendo sistema ({n_f}x{n_f})...")
    R_dense = np.array(R.tocsc().todense(), dtype=np.float64)
    F = spsolve(IQ, R_dense)

    if F.ndim == 1:
        F = F[:, np.newaxis]

    F = np.clip(F, 0.0, None)
    row_sums = F.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    F /= row_sums

    return np.argmax(F, axis=1).astype(int)


def assign_random_walk(ambiguas, flowers, graph_data, k_branches, k_flower, sigma):
    """
    Asigna flores ambiguas via Random Walk. Retorna {indice_flor: branch_id}
    """
    if not ambiguas:
        return {}

    branches   = graph_data['branches']
    A, _, n_b, n_f = build_rw_graph(ambiguas, flowers, graph_data,
                                     k_branches, k_flower, sigma)
    idx_to_bid = {i: b['id'] for i, b in enumerate(branches)}

    flower_indices = run_random_walk(A, n_b, n_f)

    resultado = {}
    for fi_local, fi_global in enumerate(ambiguas):
        bid = idx_to_bid.get(int(flower_indices[fi_local]), -1)
        resultado[fi_global] = bid
    return resultado


# =====================================================================
#  6. VISUALIZACION
# =====================================================================
def _branch_colors(graph_data):
    """Retorna {branch_id: (r, g, b)} normalizado 0-1."""
    return {n['id']: tuple(c / 255.0 for c in n['color_rgb'])
            for n in graph_data['nodes']}


def visualize_result(img, graph_data, flowers, resultado_final, info,
                     umbral, dark_bg=True, flower_size=4, save_path=None):
    """
    Panel izquierdo : flores coloreadas por rama asignada
                      circulo = euclidiana, estrella = Random Walk
    Panel derecho   : conteo de flores por rama
    """
    bc      = _branch_colors(graph_data)
    txt_col = 'white' if dark_bg else 'black'
    fig_bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    flowers_per_branch = defaultdict(int)
    for bid in resultado_final.values():
        flowers_per_branch[bid] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)

    # --- Panel izquierdo: imagen + flores ---
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fi, bid in resultado_final.items():
        fx, fy     = flowers[fi]
        color      = bc.get(bid, (1, 1, 1))
        es_ambigua = 'rama_ganadora' not in info[fi]
        marker     = '*' if es_ambigua else 'o'
        sz         = flower_size + 2 if es_ambigua else flower_size
        ax1.plot(fx, fy, marker, color=color, markersize=sz,
                 markeredgewidth=0.5, markeredgecolor='white',
                 zorder=5, alpha=0.85)

    n_euclid = sum(1 for fi in resultado_final if 'rama_ganadora' in info[fi])
    n_rw     = len(resultado_final) - n_euclid
    pe = mpatches.Patch(color='lightgray', label=f'o = Euclidiana ({n_euclid})')
    pr = mpatches.Patch(color='lightgray', label=f'* = Random Walk ({n_rw})')
    ax1.legend(handles=[pe, pr], loc='upper right', fontsize=8,
               facecolor='#333333' if dark_bg else 'white', labelcolor=txt_col)
    ax1.set_title(
        f'Pipeline HybridRW  (umbral={umbral})  |  '
        f'{n_euclid} euclidiana + {n_rw} Random Walk',
        color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # --- Panel derecho: flores por rama ---
    ax2.set_facecolor('#2a2a2a' if dark_bg else 'white')
    bids   = sorted(bc.keys())
    counts = [flowers_per_branch.get(b, 0) for b in bids]
    bars   = ax2.barh([f"R{b}" for b in bids], counts,
                      color=[bc[b] for b in bids],
                      edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Flores', color=txt_col, fontsize=10)
    ax2.set_title('Flores por Rama', color=txt_col, fontsize=11)
    ax2.tick_params(colors=txt_col)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color(txt_col)
    ax2.spines['left'].set_color(txt_col)
    ax2.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_width() + 0.3,
                 bar.get_y() + bar.get_height() / 2,
                 str(cnt), va='center', color=txt_col, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_ambiguity_map(img, flowers, info, umbral,
                            dark_bg=True, flower_size=4, save_path=None):
    """
    Verde (circulo) = trivial -> euclidiana
    Rojo  (estrella) = ambigua -> Random Walk
    """
    txt_col = 'white' if dark_bg else 'black'
    fig_bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for fi, d in info.items():
        fx, fy     = flowers[fi]
        es_trivial = 'rama_ganadora' in d
        color      = 'limegreen' if es_trivial else 'tomato'
        marker     = 'o' if es_trivial else '*'
        sz         = flower_size if es_trivial else flower_size + 2
        ax.plot(fx, fy, marker, color=color, markersize=sz,
                markeredgewidth=0.4, markeredgecolor='white',
                zorder=5, alpha=0.9)

    n_t = sum(1 for d in info.values() if 'rama_ganadora' in d)
    n_a = len(info) - n_t
    pt  = mpatches.Patch(color='limegreen', label=f'Trivial (ratio > {umbral}): {n_t}  -> euclidiana')
    pa  = mpatches.Patch(color='tomato',    label=f'Ambigua (ratio <= {umbral}): {n_a}  -> Random Walk')
    ax.legend(handles=[pt, pa], loc='upper right', fontsize=9,
              facecolor='#333333' if dark_bg else 'white', labelcolor=txt_col)
    ax.set_title(
        f'Mapa de Ambiguedad  (umbral={umbral})\n'
        f'Verde={n_t} triviales  |  Rojo={n_a} ambiguas',
        color=txt_col, fontsize=11)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_histogram(all_ratios, umbral, save_path=None):
    """Histograma de ratios para calibrar UMBRAL_AMBIGUEDAD."""
    finitos = [r for r in all_ratios if r != float('inf')]
    if not finitos:
        return

    n_amb = sum(1 for r in finitos if r <= umbral)
    pct   = 100 * n_amb / len(finitos)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(finitos, bins=40, edgecolor='black', color='steelblue', alpha=0.8)
    ax.axvline(x=umbral, color='red', linestyle='--', linewidth=1.5,
               label=f'Umbral = {umbral}  ({pct:.1f}% ambiguas)')
    ax.set_xlabel('Ratio d2/d1')
    ax.set_ylabel('Cantidad de flores')
    ax.set_title('Distribucion de ambiguedad (todas las imagenes procesadas)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
#  7. PIPELINE POR IMAGEN
# =====================================================================
def run_one(graph_json_path):
    """
    Ejecuta el pipeline hybridrw para una imagen.
    Retorna lista de ratios para el histograma global.
    """
    graph_data = load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = auto_detect_flower_json(image_name, JSON_FLORES_DIR)
    flowers     = load_flowers(flower_json)
    img_path    = auto_detect_graph_image(image_name, GRAFOS_IMG_DIR)
    img         = cv2.imread(img_path)

    n_ramas  = len(graph_data['branches'])
    n_flores = len(flowers)
    print(f"    Ramas: {n_ramas}  |  Flores: {n_flores}")

    if not flowers:
        print("    [SKIP] Sin flores.")
        return []

    # Distancias flor -> rama (min sobre todos los pixeles de la rama)
    dist_matrix, branch_ids = compute_min_distances(flowers, graph_data['branches'])

    # Clasificar por ambiguedad
    triviales, ambiguas, info = classify_ambiguity(
        dist_matrix, branch_ids, UMBRAL_AMBIGUEDAD)
    n_t = len(triviales)
    n_a = len(ambiguas)
    print(f"    Triviales={n_t} ({100*n_t/n_flores:.0f}%)  Ambiguas={n_a} ({100*n_a/n_flores:.0f}%)")

    # Asignacion
    res_euclid = assign_euclidiana(triviales, info)
    res_rw     = assign_random_walk(ambiguas, flowers, graph_data,
                                    K_VECINOS_RW, K_FLOWER_RW, SIGMA_RW)
    resultado_final = {**res_euclid, **res_rw}

    # Guardar visualizaciones
    fname = os.path.splitext(os.path.basename(graph_json_path))[0]

    path_hibrido    = os.path.join(OUTPUT_DIR, fname + "_hybridrw.png")
    path_ambiguedad = os.path.join(OUTPUT_DIR, fname + "_hybridrw_ambiguedad.png")

    visualize_result(img, graph_data, flowers, resultado_final, info,
                     UMBRAL_AMBIGUEDAD, DARK_BG, FLOWER_SIZE,
                     save_path=path_hibrido)

    visualize_ambiguity_map(img, flowers, info, UMBRAL_AMBIGUEDAD,
                            DARK_BG, FLOWER_SIZE, save_path=path_ambiguedad)

    print(f"    [OK] -> {os.path.basename(path_hibrido)} + {os.path.basename(path_ambiguedad)}")

    return [info[i]['ratio'] for i in range(n_flores)]


# =====================================================================
#  EJECUCION PRINCIPAL
# =====================================================================
if __name__ == '__main__':
    if SINGLE_IMAGE:
        json_files = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
    else:
        json_files = sorted(glob.glob(os.path.join(GRAPH_JSON_DIR, '*.json')))

    if not json_files:
        raise FileNotFoundError(f"No se encontraron JSONs en: {GRAPH_JSON_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Procesando {len(json_files)} imagen(es)...")
    print(f"[INFO] Umbral ambiguedad : {UMBRAL_AMBIGUEDAD}")
    print(f"[INFO] K ramas RW        : {K_VECINOS_RW}")
    print(f"[INFO] K flores RW       : {K_FLOWER_RW}")
    print(f"[INFO] Sigma RW          : {SIGMA_RW}")
    print(f"[INFO] Output            : {OUTPUT_DIR}\n")

    all_ratios = []
    ok = 0
    errors = 0

    for jf in json_files:
        print(f"  {os.path.basename(jf)}")
        try:
            ratios = run_one(jf)
            all_ratios.extend(ratios)
            ok += 1
        except Exception as e:
            print(f"    [ERROR] {e}")
            errors += 1

    print(f"\n[DONE] {ok} OK, {errors} errores.")

    # Histograma global de ratios para calibrar el umbral
    if all_ratios:
        hist_path = os.path.join(OUTPUT_DIR, "histograma_ratios_hybridrw.png")
        visualize_histogram(all_ratios, UMBRAL_AMBIGUEDAD, save_path=hist_path)
        print(f"[INFO] Histograma guardado en: {hist_path}")

        finitos = [r for r in all_ratios if r != float('inf')]
        n_amb   = sum(1 for r in finitos if r <= UMBRAL_AMBIGUEDAD)
        print(f"\n[RESUMEN GLOBAL]")
        print(f"  Total flores     : {len(all_ratios)}")
        print(f"  Euclidiana       : {len(all_ratios) - n_amb}")
        print(f"  Random Walk      : {n_amb}  ({100*n_amb/len(finitos):.1f}%)")
        print(f"\n  Flores ambiguas por umbral:")
        for u in [1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
            na = sum(1 for r in finitos if r <= u)
            print(f"    umbral {u}: {na} ambiguas ({100*na/len(finitos):.1f}%)")
