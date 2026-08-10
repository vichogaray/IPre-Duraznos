"""
GLEE Density -- Asigna flores a ramas via GLEE (sin dependencias externas)
===========================================================================
Metodo: GLEE (Geometric Laplacian Eigenmap Embedding)
        Torres et al., Journal of Complex Networks 2020
        Implementado directamente con NumPy/SciPy (sin karateclub)

Input:
  - JSON de grafo (grafos json/)  : ramas con centroide, aristas, pixeles
  - JSON de flores (json flores/) : coordenadas (x,y) de cada flor

Output:
  - Imagen con grafo + flores coloreadas por rama asignada
  - Barra de conteo de flores por rama

Sin instalacion adicional: usa solo NumPy, SciPy y NetworkX.

Uso:
    1. Configura las rutas abajo
    2. Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"
OUTPUT_DIR      = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidad floral glee"

K_NEIGHBORS  = 3    # Cada flor se conecta a las K ramas mas cercanas del esqueleto
N_DIMENSIONS = 8    # Dimensiones del embedding GLEE (probar 4, 8, 16)
FLOWER_SIZE  = 3    # Tamano de la X en el plot
DARK_BG      = True

# Para probar con una sola imagen pon el nombre del JSON aqui, o None para toda la carpeta
# Ejemplo: SINGLE_IMAGE = "imgs_frame1_00000_graph.json"
SINGLE_IMAGE = "imgs_frame153_00000_graph.json"

# =====================================================================
#  NO MODIFICAR DEBAJO
# =====================================================================

import os
import re
import glob
import json
import numpy as np
import networkx as nx
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.sparse.csgraph import laplacian as sparse_laplacian


def glee_embedding(G, dimensions):
    """
    Implementacion de GLEE: eigenvectores del Laplaciano normalizado del grafo.
    Los nodos deben ser enteros 0..N-1.
    Retorna embedding de shape (N, dimensions).
    """
    N = G.number_of_nodes()
    dims = min(dimensions, N - 1)

    # Construir matriz de adyacencia esparsa
    A = lil_matrix((N, N), dtype=np.float64)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    A = A.tocsr()

    # Laplaciano normalizado
    L = sparse_laplacian(A, normed=True)

    # Los dims+1 eigenvectores mas pequenos (el primero es trivial, se descarta)
    k = dims + 1
    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM',
                                          maxiter=N * 20, tol=1e-5)
    except ArpackNoConvergence as e:
        print(f"    [WARN] ARPACK no convergio del todo — usando {len(e.eigenvalues)} eigenvectores parciales")
        eigenvalues  = e.eigenvalues
        eigenvectors = e.eigenvectors
        if len(eigenvalues) < 2:
            raise ValueError("ARPACK convergio menos de 2 eigenvectores, no se puede continuar")

    # Ordenar por eigenvalor y descartar el primero (eigenvalor ~0)
    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    actual_dims = min(dims, eigenvectors.shape[1] - 1)
    embedding = eigenvectors[:, 1:actual_dims + 1]  # shape (N, actual_dims)

    return embedding


# ---------------------------------------------------------------------------
#  1. Cargar datos  (mismas funciones que laplacian_density.py)
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
#  2. Construir grafo a nivel de PIXEL y aplicar GLEE
# ---------------------------------------------------------------------------

def asignar_flores_glee(graph_data, flowers):
    """
    Asigna cada flor a una rama usando GLEE a nivel de pixel.

    En vez de usar solo 1 nodo por rama (centroide), usa TODOS los pixeles
    del esqueleto como nodos, conectados por 8-vecindad. Esto da a GLEE
    la resolucion topologica necesaria para separar ramas.

    Retorna:
        flower_branch_indices: array (n_flores,) con indice de rama asignado
        idx_to_bid: mapeo indice de rama -> branch_id original
    """
    branches = graph_data['branches']

    # --- Paso 1: Crear nodos-pixel con su branch_id ---
    pixel_nodes = []          # [(y, x, branch_id), ...]
    pos_to_node = {}          # (y, x) -> node_index
    branch_pixel_indices = defaultdict(list)  # bid -> [node_indices]

    for b in branches:
        bid = b['id']
        for yx in b['pixels']:
            y, x = int(yx[0]), int(yx[1])
            key = (y, x)
            if key in pos_to_node:
                continue  # evitar duplicados
            node_idx = len(pixel_nodes)
            pixel_nodes.append((y, x, bid))
            pos_to_node[key] = node_idx
            branch_pixel_indices[bid].append(node_idx)

    N_skel = len(pixel_nodes)
    print(f"    Pixeles esqueleto: {N_skel}")

    # --- Paso 2: Grafo por 8-vecindad entre pixeles ---
    G = nx.Graph()
    G.add_nodes_from(range(N_skel))

    for idx, (y, x, _bid) in enumerate(pixel_nodes):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = pos_to_node.get((y + dy, x + dx))
                if nb is not None and nb > idx:
                    G.add_edge(idx, nb)

    # --- Paso 3: Conectar ramas adyacentes por su par de pixeles mas cercano ---
    # (la 8-vecindad puede dejar gaps entre ramas si PROXIMITY_GAP > 1)
    branch_coords = {}
    for bid, indices in branch_pixel_indices.items():
        coords = np.array([(pixel_nodes[i][1], pixel_nodes[i][0])
                           for i in indices], dtype=np.float32)  # (x, y)
        branch_coords[bid] = (indices, coords)

    for edge in graph_data['edges']:
        bid_a, bid_b = edge['from'], edge['to']
        if bid_a not in branch_coords or bid_b not in branch_coords:
            continue
        idx_a, coords_a = branch_coords[bid_a]
        idx_b, coords_b = branch_coords[bid_b]
        dists = cdist(coords_a, coords_b, 'sqeuclidean')
        mi, mj = np.unravel_index(np.argmin(dists), dists.shape)
        G.add_edge(idx_a[mi], idx_b[mj])

    print(f"    Aristas esqueleto: {G.number_of_edges()}")

    # --- Paso 4: Agregar flores conectadas a K pixeles mas cercanos ---
    skel_xy = np.array([(pixel_nodes[i][1], pixel_nodes[i][0])
                        for i in range(N_skel)], dtype=np.float32)  # (x, y)

    for fi, (fx, fy) in enumerate(flowers):
        flower_node = N_skel + fi
        G.add_node(flower_node)
        d_sq = (skel_xy[:, 0] - fx) ** 2 + (skel_xy[:, 1] - fy) ** 2
        nearest = np.argsort(d_sq)[:K_NEIGHBORS]
        for n in nearest:
            G.add_edge(flower_node, int(n))

    # --- Paso 5: GLEE embedding ---
    n_total = G.number_of_nodes()
    dims = min(N_DIMENSIONS, n_total - 2)
    if dims < 1:
        raise ValueError(f"Grafo demasiado pequeno para GLEE (nodos: {n_total})")

    print(f"    GLEE: {n_total} nodos, {dims} dimensiones...")
    embedding = glee_embedding(G, dims)

    # --- Paso 6: Asignar cada flor al pixel-esqueleto mas cercano en embedding ---
    emb_skel   = embedding[:N_skel]
    emb_flores = embedding[N_skel:]

    distancias = cdist(emb_flores, emb_skel, metric='euclidean')
    nearest_skel = np.argmin(distancias, axis=1)

    # Cada flor hereda el branch_id del pixel esqueleto mas cercano en embedding
    bid_set = sorted(set(b['id'] for b in branches))
    idx_to_bid = {i: bid for i, bid in enumerate(bid_set)}
    bid_to_idx = {bid: i for i, bid in enumerate(bid_set)}

    flower_branch_indices = np.array([
        bid_to_idx[pixel_nodes[nearest_skel[fi]][2]]
        for fi in range(len(flowers))
    ])

    return flower_branch_indices, idx_to_bid


# ---------------------------------------------------------------------------
#  4. Visualizacion  (misma estructura que laplacian_density.py)
# ---------------------------------------------------------------------------

def visualize(img, graph_data, flowers, flower_branch_indices, idx_to_bid,
              flower_size=3, dark_bg=True, save_path=None):

    nodes = graph_data['nodes']

    txt_col = 'white' if dark_bg else 'black'
    fig_bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    # Color por branch_id (desde JSON)
    branch_colors = {}
    for node in nodes:
        bid = node['id']
        r, g, b = node['color_rgb']
        branch_colors[bid] = (r / 255, g / 255, b / 255)

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
    ax1.set_title('Grafo + Flores  (color = rama asignada por GLEE)',
                  color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # === Panel derecho: flores por rama ===
    ax2.set_facecolor('#2a2a2a' if dark_bg else 'white')
    sorted_bids = sorted(branch_colors.keys())
    counts      = [flowers_per_branch.get(bid, 0) for bid in sorted_bids]
    colors_bar  = [branch_colors[bid] for bid in sorted_bids]
    bar_labels  = [f"R{bid}" for bid in sorted_bids]

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

    flower_branch_indices, idx_to_bid = asignar_flores_glee(graph_data, flowers)

    fname     = os.path.splitext(os.path.basename(graph_json_path))[0]
    save_path = os.path.join(OUTPUT_DIR, fname + "_glee.png")
    visualize(img, graph_data, flowers, flower_branch_indices, idx_to_bid,
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
