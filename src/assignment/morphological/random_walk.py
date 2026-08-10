"""
Random Walk Density -- Assigns flowers to branches via semi-supervised Random Walk
====================================================================================
Method: Grady 2006 (Random Walks for Image Segmentation)

Math:
  Absorbent nodes : branches (labeled, indices 0..n_b-1)
  Non-absorbent   : flowers  (unlabeled, indices n_b..N-1)

  Weighted adjacency A  (symmetric)
  Transition matrix  T = D^{-1} A   (row-stochastic)

  Partition T into blocks:
    Q = T[flowers, flowers]   -- flower-to-flower transitions
    R = T[flowers, branches]  -- flower-to-branch transitions

  Solve:  (I - Q) F = R
  --> F[i, j] = probability that a random walk starting at flower i
                gets absorbed by branch j

  Assignment: flower_i --> argmax_j F[i, :]

Edge types:
  branch-branch  : from graph JSON edges, weight = 1.0
  flower-branch  : K_NEIGHBORS nearest branches by min-pixel distance,
                   weight = exp(-dist / SIGMA)
  flower-flower  : K_FLOWER nearest flowers by Euclidean distance,
                   weight = exp(-dist / SIGMA)
                   (set K_FLOWER = 0 to disable; without these Q=0
                    and the algorithm reduces to weighted k-NN)

Input:
  - JSON de grafo (grafos json/) : branches, edges, pixels
  - JSON de flores (json flores/): (x, y) por flor

Output:
  - Imagen con grafo + flores coloreadas por rama asignada (plt.show())
  - Barra de conteo por rama

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
OUTPUT_DIR      = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidad floral random walk"

K_NEIGHBORS = 3      # Nearest branches per flower (flower->branch edges)
K_FLOWER    = 5      # Nearest flowers per flower  (flower->flower edges)
                     # Set to 0 to disable flower-flower edges
SIGMA       = 50.0   # Distance decay: w = exp(-dist / SIGMA)
FLOWER_SIZE = 3      # Marker size in the plot
DARK_BG     = True

# Process a single image by name, or None to process the entire folder
SINGLE_IMAGE = None

# =====================================================================
#  DO NOT MODIFY BELOW
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
from scipy.sparse import lil_matrix, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree as KDTree


# ---------------------------------------------------------------------------
#  1. Load data
# ---------------------------------------------------------------------------

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
    m = re.search(r'frame(\d+)', image_name)
    if not m:
        raise ValueError("Cannot extract frame number from: " + image_name)
    path = os.path.join(json_flores_dir, "frame" + m.group(1) + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError("Not found: " + path)
    return path


def auto_detect_graph_image(image_name, img_dir):
    path = os.path.join(img_dir, image_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError("Image not found: " + path)


# ---------------------------------------------------------------------------
#  2. Build graph
# ---------------------------------------------------------------------------

def build_graph(graph_data, flowers, k_neighbors, k_flower, sigma):
    """
    Builds the combined weighted adjacency matrix A of shape (N, N) where:
      rows/cols 0 .. n_b-1  : branch nodes (absorbent, labeled)
      rows/cols n_b .. N-1  : flower nodes (non-absorbent, unlabeled)

    Returns:
      A          : scipy CSR sparse matrix, symmetric
      bid_to_idx : dict branch_id -> row index
      n_b        : number of branches
      n_f        : number of flowers
    """
    branches = graph_data['branches']
    edges    = graph_data['edges']
    n_b = len(branches)
    n_f = len(flowers)
    N   = n_b + n_f

    bid_to_idx = {b['id']: i for i, b in enumerate(branches)}
    A = lil_matrix((N, N), dtype=np.float64)

    # --- Branch-branch edges (weight 1.0) ---
    for edge in edges:
        i = bid_to_idx.get(edge['from'])
        j = bid_to_idx.get(edge['to'])
        if i is not None and j is not None:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # --- Flower-branch edges (k nearest by min-pixel distance) ---
    branch_pixels = [np.array(b['pixels'], dtype=np.float32) for b in branches]

    for fi, (fx, fy) in enumerate(flowers):
        flower_idx = n_b + fi
        dists = []
        for bi, px in enumerate(branch_pixels):
            # px[:, 0] = y, px[:, 1] = x
            d_sq = (px[:, 1] - fx) ** 2 + (px[:, 0] - fy) ** 2
            dists.append((float(np.sqrt(np.min(d_sq))), bi))
        dists.sort()
        for dist, bi in dists[:k_neighbors]:
            w = np.exp(-dist / sigma)
            A[flower_idx, bi] = max(A[flower_idx, bi], w)
            A[bi, flower_idx] = max(A[bi, flower_idx], w)

    # --- Flower-flower edges (k nearest by Euclidean distance) ---
    if k_flower > 0 and n_f > 1:
        flower_xy = np.array(flowers, dtype=np.float64)  # (n_f, 2)  x, y
        # KDTree query: each flower finds k+1 neighbors (first is itself)
        tree = KDTree(flower_xy)
        k_query = min(k_flower + 1, n_f)
        dists_knn, idxs_knn = tree.query(flower_xy, k=k_query)
        for fi in range(n_f):
            for dist, nb_fi in zip(dists_knn[fi][1:], idxs_knn[fi][1:]):
                fi_idx  = n_b + fi
                nfi_idx = n_b + int(nb_fi)
                w = np.exp(-float(dist) / sigma)
                A[fi_idx, nfi_idx] = max(A[fi_idx, nfi_idx], w)
                A[nfi_idx, fi_idx] = max(A[nfi_idx, fi_idx], w)

    return A.tocsr(), bid_to_idx, n_b, n_f


# ---------------------------------------------------------------------------
#  3. Random Walk solver
# ---------------------------------------------------------------------------

def run_random_walk(A, n_b, n_f):
    """
    Solves the Random Walk absorption problem.

    Transition matrix T = D^{-1} A  (row-stochastic)
    Partition into blocks (ordering: branches first, then flowers):
      T_bb  n_b x n_b
      T_fb  n_f x n_b  = R   (flower -> branch transitions)
      T_ff  n_f x n_f  = Q   (flower -> flower transitions)

    System: (I - Q) F = R
    F[i, j] = absorption probability of flower i by branch j

    Returns array of shape (n_f,) with branch index per flower.
    """
    # Degree vector (sum of each row)
    d = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    d[d == 0] = 1.0  # avoid divide-by-zero for isolated nodes

    # T = D^{-1} A  (sparse: scale rows)
    from scipy.sparse import diags
    D_inv = diags(1.0 / d)
    T = D_inv.dot(A)

    # Extract sub-matrices
    # branches: rows/cols 0..n_b-1   flowers: rows/cols n_b..N-1
    Q = T[n_b:, n_b:]   # (n_f, n_f)
    R = T[n_b:, :n_b]   # (n_f, n_b)

    # Solve (I - Q) F = R  column by column
    IQ = speye(n_f, format='csr') - Q.tocsr()
    R_csc = R.tocsc()

    print("    Solving ({0}x{0}) linear system, {1} branches...".format(
        n_f, n_b))

    # spsolve handles multiple RHS when given a dense matrix;
    # convert R to dense since n_b is small (3-6)
    R_dense = np.array(R_csc.todense(), dtype=np.float64)
    F = spsolve(IQ, R_dense)   # (n_f, n_b)

    if F.ndim == 1:
        F = F[:, np.newaxis]

    # Clip numerical noise
    F = np.clip(F, 0.0, None)

    # Row-normalize (probabilities should sum to 1 per flower)
    row_sums = F.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    F /= row_sums

    return np.argmax(F, axis=1).astype(int)


# ---------------------------------------------------------------------------
#  4. Visualization
# ---------------------------------------------------------------------------

def visualize(img, graph_data, flowers, flower_branch_indices,
              flower_size=3, dark_bg=True, save_path=None):

    branches = graph_data['branches']
    nodes    = graph_data['nodes']

    txt_col = 'white' if dark_bg else 'black'
    fig_bg  = '#1e1e1e' if dark_bg else '#f0f0f0'

    # Branch colors from node color_rgb
    branch_colors = {}
    for node in nodes:
        r, g, b = node['color_rgb']
        branch_colors[node['id']] = (r / 255.0, g / 255.0, b / 255.0)

    idx_to_bid = {i: b['id'] for i, b in enumerate(branches)}

    flowers_per_branch = defaultdict(int)
    for bi in flower_branch_indices:
        flowers_per_branch[idx_to_bid.get(int(bi), -1)] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                    gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)

    # Left panel: image + flowers
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fi, (fx, fy) in enumerate(flowers):
        bi  = int(flower_branch_indices[fi])
        bid = idx_to_bid.get(bi, -1)
        color = branch_colors.get(bid, (1, 1, 1))
        ax1.plot(fx, fy, 'x', color=color, markersize=flower_size,
                 markeredgewidth=0.8, zorder=5)
    ax1.set_title('Graph + Flowers  (color = assigned branch, Random Walk)',
                  color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # Right panel: flower count per branch
    ax2.set_facecolor('#2a2a2a' if dark_bg else 'white')
    sorted_bids = sorted(branch_colors.keys())
    counts      = [flowers_per_branch.get(bid, 0) for bid in sorted_bids]
    colors_bar  = [branch_colors[bid] for bid in sorted_bids]
    bar_labels  = ["R" + str(bid) for bid in sorted_bids]

    bars = ax2.barh(bar_labels, counts, color=colors_bar,
                    edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Flowers', color=txt_col, fontsize=10)
    ax2.set_title('Flowers per Branch', color=txt_col, fontsize=11)
    ax2.tick_params(colors=txt_col)
    ax2.spines['bottom'].set_color(txt_col)
    ax2.spines['left'].set_color(txt_col)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.3,
                 bar.get_y() + bar.get_height() / 2,
                 str(count), va='center', color=txt_col, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
#  5. Pipeline per image
# ---------------------------------------------------------------------------

def run_one(graph_json_path):
    graph_data = load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = auto_detect_flower_json(image_name, JSON_FLORES_DIR)
    flowers     = load_flowers(flower_json)
    img_path    = auto_detect_graph_image(image_name, GRAFOS_IMG_DIR)
    img         = cv2.imread(img_path)

    n_b = len(graph_data['branches'])
    print("    Branches: {0}  |  Flowers: {1}  |  K_branch: {2}  |  K_flower: {3}".format(
        n_b, len(flowers), K_NEIGHBORS, K_FLOWER))

    if not flowers:
        print("    [SKIP] No flowers.")
        return

    A, bid_to_idx, n_b, n_f = build_graph(
        graph_data, flowers, K_NEIGHBORS, K_FLOWER, SIGMA)

    flower_branch_indices = run_random_walk(A, n_b, n_f)

    fname     = os.path.splitext(os.path.basename(graph_json_path))[0]
    save_path = os.path.join(OUTPUT_DIR, fname + "_rw.png")
    visualize(img, graph_data, flowers, flower_branch_indices,
              FLOWER_SIZE, DARK_BG, save_path=save_path)
    print("    [OK] -> " + os.path.basename(save_path))


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    if SINGLE_IMAGE:
        json_files = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
    else:
        json_files = sorted(glob.glob(os.path.join(GRAPH_JSON_DIR, "*.json")))

    if not json_files:
        raise FileNotFoundError("No JSON files found in: " + GRAPH_JSON_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[INFO] Processing {0} image(s)...\n".format(len(json_files)))

    ok, errors = 0, 0
    for jf in json_files:
        print("  " + os.path.basename(jf))
        try:
            run_one(jf)
            ok += 1
        except Exception as e:
            print("    [ERROR] " + str(e))
            errors += 1

    print("\n[DONE] {0} OK, {1} errors.".format(ok, errors))
