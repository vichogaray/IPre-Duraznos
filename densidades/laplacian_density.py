"""
Laplacian Density — Asigna flores a ramas via Graph Laplacian semi-supervisado
===============================================================================
Metodo: Orduz 2019 (https://juanitorduz.github.io/semi_supervised_clustering/)

Input:
  - JSON de grafo (grafos json/)  : ramas con centroide, aristas, pixeles
  - JSON de flores (json flores/) : coordenadas (x,y) de cada flor

Output:
  - Imagen con grafo + flores coloreadas por rama asignada
  - Barra de conteo de flores por rama

Uso:
    1. Configura las rutas abajo
    2. Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"
OUTPUT_DIR      = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidad floral laplacian"

K_NEIGHBORS  = 3      # Cada flor se conecta a las K ramas mas cercanas
ALPHA        = 1.0    # Balance: alto = topologia manda, bajo = labels mandan
FLOWER_SIZE  = 3      # Tamano de la X en el plot
DARK_BG      = True

# Para probar con una sola imagen pon el nombre del JSON aqui, o None para procesar toda la carpeta
SINGLE_IMAGE = None

# =====================================================================
#  NO MODIFICAR DEBAJO
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
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian


# ---------------------------------------------------------------------------
#  1. Cargar datos
# ---------------------------------------------------------------------------

def load_graph_json(json_path):
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


def load_flowers(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    return [(float(s['points'][0][0]), float(s['points'][0][1]))
            for s in data.get('shapes', [])
            if s.get('label') == 'flower' and s.get('points')]


def auto_detect_flower_json(graph_image_name, json_flores_dir):
    match = re.search(r'frame(\d+)', graph_image_name)
    if not match:
        raise ValueError(f"No se pudo extraer frame de: {graph_image_name}")
    json_path = os.path.join(json_flores_dir, f"frame{match.group(1)}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontro: {json_path}")
    return json_path


def auto_detect_graph_image(graph_image_name, grafos_img_dir):
    path = os.path.join(grafos_img_dir, graph_image_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No se encontro imagen: {path}")


# ---------------------------------------------------------------------------
#  2. Construir matriz de adyacencia
# ---------------------------------------------------------------------------

def build_adjacency_matrix(graph_data, flowers, k):
    """
    Construye la matriz de adyacencia A para el grafo combinado:
      - filas/columnas 0..n_branches-1 : ramas (nodos etiquetados)
      - filas/columnas n_branches..N-1  : flores (nodos sin etiqueta)

    Conexiones rama-rama: desde el JSON (adyacencia ya construida)
    Conexiones flor-rama: k ramas mas cercanas por distancia minima a pixeles
    """
    branches = graph_data['branches']
    edges    = graph_data['edges']
    n_b      = len(branches)
    n_f      = len(flowers)
    N        = n_b + n_f

    # Mapeo branch_id -> indice de fila/columna
    bid_to_idx = {b['id']: i for i, b in enumerate(branches)}

    A = lil_matrix((N, N), dtype=np.float64)

    # --- Conexiones rama-rama (peso 1) ---
    for edge in edges:
        i = bid_to_idx.get(edge['from'])
        j = bid_to_idx.get(edge['to'])
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Precomputar pixeles de cada rama como array numpy ---
    branch_pixels = []
    for b in branches:
        px = np.array(b['pixels'], dtype=np.float32)  # [[y,x], ...]
        branch_pixels.append(px)

    # --- Conexiones flor-rama: k-NN por distancia minima a pixeles ---
    for fi, (fx, fy) in enumerate(flowers):
        flower_idx = n_b + fi
        dists = []
        for bi, px in enumerate(branch_pixels):
            # Distancia minima del punto flor a cualquier pixel de la rama
            d_sq = (px[:, 1] - fx) ** 2 + (px[:, 0] - fy) ** 2
            dists.append((np.min(d_sq), bi))

        dists.sort(key=lambda x: x[0])
        for d_sq, bi in dists[:k]:
            w = np.exp(-np.sqrt(d_sq) / 50.0)   # peso decae con distancia
            A[flower_idx, bi] = w
            A[bi, flower_idx] = w

    return A.tocsr(), bid_to_idx, n_b, n_f


# ---------------------------------------------------------------------------
#  3. Graph Laplacian semi-supervisado
# ---------------------------------------------------------------------------

def run_laplacian(A, n_b, n_f, alpha):
    """
    Resuelve el problema de clasificacion semi-supervisada via Graph Laplacian.

    Los nodos etiquetados (0..n_b-1) tienen clase = su propio indice.
    Los nodos sin etiqueta (n_b..N-1) son las flores.

    Retorna array (n_f,) con el indice de rama asignado a cada flor.
    """
    N        = n_b + n_f
    n_classes = n_b  # cada rama es su propia clase

    L = sparse_laplacian(A, normed=False)
    L_dense = np.array(L.todense())

    # U_obs: one-hot para los nodos etiquetados (ramas)
    label_index = list(range(n_b))
    u_obs = np.eye(n_classes, dtype=np.float64)  # (n_b, n_classes)

    def loss(u_flat):
        u = u_flat.reshape(N, n_classes)

        # Softmax loss sobre nodos etiquetados
        u_labeled = u[label_index, :]
        soft = np.apply_along_axis(softmax, 1, u_labeled)
        softmax_loss = -(1.0 / len(label_index)) * np.trace(
            u_obs.T @ np.log(soft + 1e-9))

        # Laplacian smoothness loss
        lap_loss = 0.5 * np.trace(u.T @ L_dense @ u)

        return softmax_loss + alpha * lap_loss

    u0 = np.ones((N, n_classes)) / n_classes
    print(f"    Optimizando ({N} nodos, {n_classes} clases)...")
    result = minimize(fun=loss, x0=u0.flatten(), method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-9})

    u_best = result.x.reshape(N, n_classes)
    u_soft = np.apply_along_axis(softmax, 1, u_best)
    predicted = np.argmax(u_soft, axis=1)

    # Las filas n_b..N-1 corresponden a las flores
    flower_branch_indices = predicted[n_b:]
    return flower_branch_indices


# ---------------------------------------------------------------------------
#  4. Visualizacion
# ---------------------------------------------------------------------------

def visualize(img, graph_data, flowers, flower_branch_indices,
              flower_size=3, dark_bg=True, save_path=None):

    branches  = graph_data['branches']
    nodes     = graph_data['nodes']
    n_b       = len(branches)

    txt_col = 'white' if dark_bg else 'black'
    fig_bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    # Color por rama (desde el JSON, campo color_rgb)
    branch_colors = {}
    for node in nodes:
        bid = node['id']
        r, g, b = node['color_rgb']
        branch_colors[bid] = (r / 255, g / 255, b / 255)

    # Mapeo indice -> branch_id
    idx_to_bid = {i: b['id'] for i, b in enumerate(branches)}

    # Conteo flores por rama
    flowers_per_branch = defaultdict(int)
    for bi in flower_branch_indices:
        bid = idx_to_bid.get(int(bi), -1)
        flowers_per_branch[bid] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)

    # === Panel izquierdo: imagen + flores ===
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fi, (fx, fy) in enumerate(flowers):
        bi  = int(flower_branch_indices[fi])
        bid = idx_to_bid.get(bi, -1)
        color = branch_colors.get(bid, (1, 1, 1))
        ax1.plot(fx, fy, 'x', color=color, markersize=flower_size,
                 markeredgewidth=0.8, zorder=5)
    ax1.set_title('Grafo + Flores  (color = rama asignada por Laplacian)',
                  color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # === Panel derecho: flores por rama ===
    ax2.set_facecolor('#2a2a2a' if dark_bg else 'white')
    sorted_bids  = sorted(branch_colors.keys())
    counts       = [flowers_per_branch.get(bid, 0) for bid in sorted_bids]
    colors_bar   = [branch_colors[bid] for bid in sorted_bids]
    bar_labels   = [f"R{bid}" for bid in sorted_bids]

    bars = ax2.barh(bar_labels, counts, color=colors_bar,
                    edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Flores', color=txt_col, fontsize=10)
    ax2.set_title('Flores por Rama', color=txt_col, fontsize=11)
    ax2.tick_params(colors=txt_col)
    ax2.spines['bottom'].set_color(txt_col)
    ax2.spines['left'].set_color(txt_col)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(count), va='center', color=txt_col, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
#  5. Pipeline por imagen
# ---------------------------------------------------------------------------

def run_one(graph_json_path):
    graph_data = load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = auto_detect_flower_json(image_name, JSON_FLORES_DIR)
    flowers     = load_flowers(flower_json)
    img_path    = auto_detect_graph_image(image_name, GRAFOS_IMG_DIR)
    img         = cv2.imread(img_path)

    print(f"    Ramas: {len(graph_data['nodes'])}  |  Flores: {len(flowers)}")

    if not flowers:
        print("    [SKIP] Sin flores.")
        return

    A, bid_to_idx, n_b, n_f = build_adjacency_matrix(
        graph_data, flowers, K_NEIGHBORS)

    flower_branch_indices = run_laplacian(A, n_b, n_f, ALPHA)

    fname     = os.path.splitext(os.path.basename(graph_json_path))[0]
    save_path = os.path.join(OUTPUT_DIR, fname + "_laplacian.png")
    visualize(img, graph_data, flowers, flower_branch_indices,
              FLOWER_SIZE, DARK_BG, save_path=save_path)
    print(f"    [OK] -> {os.path.basename(save_path)}")


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    if SINGLE_IMAGE:
        json_files = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
    else:
        json_files = sorted(glob.glob(os.path.join(GRAPH_JSON_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No se encontraron JSONs en: {GRAPH_JSON_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Procesando {len(json_files)} imagen(es)...\n")

    ok, errors = 0, 0
    for jf in json_files:
        print(f"  {os.path.basename(jf)}")
        try:
            run_one(jf)
            ok += 1
        except Exception as e:
            print(f"    [ERROR] {e}")
            errors += 1

    print(f"\n[DONE] {ok} OK, {errors} errores.")
