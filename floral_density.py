"""
Mapa de Densidad Floral — Asigna flores a la rama mas cercana
================================================================
Input:
  - Imagen de grafo coloreado (cada color = una rama)
  - JSON de flores (formato LabelMe, cada punto = una flor)

Output:
  - Visualizacion del grafo con flores superpuestas
  - Conteo de flores por rama y por nivel jerarquico

Uso:
    1. Configura GRAPH_PATH abajo
    2. El JSON se detecta automaticamente desde JSON_DIR
    3. Dale Run (F5)descarg
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_PATH = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"  # carpeta o archivo individual
JSON_DIR   = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"

# None = auto-detectar JSON desde el nombre del grafo. O pon ruta manual:
JSON_PATH  = None

# Parametros de deteccion de ramas (mismos que branch_identifier2)
COLOR_TOLERANCE   = 25
PROXIMITY_GAP     = 7
MIN_BRANCH_PIXELS = 15
TRUNK_SHORT_RATIO = 0.30

# Visualizacion
FLOWER_SIZE       = 3     # Tamano de la X de cada flor
SHOW_LABELS       = True  # Mostrar etiquetas de rama
DARK_BACKGROUND   = True

# =====================================================================
#  NO NECESITAS MODIFICAR NADA DEBAJO DE ESTA LINEA
# =====================================================================

import json
import re
import os
import glob
import numpy as np
import cv2
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # backend sin ventana para guardar archivos
import matplotlib.pyplot as plt

# Importar funciones de branch_identifier2 (mismo directorio)
from branch_identifier2 import (
    detect_background,
    extract_branches_by_color,
    build_adjacency,
    identify_trunk,
    classify_hierarchy,
    _hier_name,
    HIERARCHY,
)


# ---------------------------------------------------------------------------
#  1. Cargar flores desde JSON (formato LabelMe)
# ---------------------------------------------------------------------------
def load_flowers(json_path):
    """
    Lee un JSON de LabelMe y extrae las coordenadas de cada flor.
    Retorna lista de (x, y) en coordenadas de pixel.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    flowers = []
    for shape in data.get('shapes', []):
        if shape.get('label') == 'flower' and shape.get('points'):
            x, y = shape['points'][0]
            flowers.append((float(x), float(y)))

    return flowers


# ---------------------------------------------------------------------------
#  2. Auto-detectar JSON desde nombre del grafo
# ---------------------------------------------------------------------------
def auto_detect_json(graph_path, json_dir):
    """
    imgs_frame1_00000_graph.png  ->  frame1.json
    imgs_frame33_00000_graph.png ->  frame33.json
    """
    basename = os.path.basename(graph_path)
    match = re.search(r'frame(\d+)', basename)
    if match:
        frame_num = match.group(1)
        json_name = f"frame{frame_num}.json"
        json_path = os.path.join(json_dir, json_name)
        if os.path.exists(json_path):
            return json_path
        raise FileNotFoundError(
            f"No se encontro {json_name} en {json_dir}")
    raise ValueError(
        f"No se pudo extraer numero de frame de: {basename}")


# ---------------------------------------------------------------------------
#  3. Asignar cada flor a su rama mas cercana
# ---------------------------------------------------------------------------
def assign_flowers_to_branches(flowers, branches):
    """
    Para cada flor, encuentra la rama cuyo pixel mas cercano esta
    a menor distancia.

    Retorna:
      assignments: lista de branch_id para cada flor
      distances:   lista de distancias (px) a la rama asignada
    """
    if not flowers or not branches:
        return [], []

    # Concatenar todos los pixeles de todas las ramas con sus IDs
    all_yx = []
    all_labels = []
    for bid, branch in branches.items():
        pixels = list(branch['pixels'])
        all_yx.extend(pixels)
        all_labels.extend([bid] * len(pixels))

    all_yx = np.array(all_yx, dtype=np.float32)       # (N, 2) en (y, x)
    all_labels = np.array(all_labels, dtype=np.int32)

    assignments = []
    distances = []

    for fx, fy in flowers:
        # Distancia euclidiana al cuadrado (sin sqrt para comparar)
        dists_sq = (all_yx[:, 1] - fx) ** 2 + (all_yx[:, 0] - fy) ** 2
        idx = np.argmin(dists_sq)
        assignments.append(int(all_labels[idx]))
        distances.append(float(np.sqrt(dists_sq[idx])))

    return assignments, distances


# ---------------------------------------------------------------------------
#  4. Visualizacion
# ---------------------------------------------------------------------------
def visualize(img, branches, levels, trunk_ids, flowers,
              assignments, distances, show_labels=True, dark_bg=True,
              flower_size=3, save_path=None):

    txt_col = 'white' if dark_bg else 'black'
    fig_bg = '#1e1e1e' if dark_bg else '#f0f0f0'

    # --- Colores por rama ---
    branch_colors_rgb = {}
    for bid, branch in branches.items():
        b, g, r = branch['color_bgr']
        branch_colors_rgb[bid] = (r / 255, g / 255, b / 255)

    # --- Conteo de flores por rama ---
    flowers_per_branch = defaultdict(list)
    for i, bid in enumerate(assignments):
        flowers_per_branch[bid].append(i)

    # --- Figura: grafo+flores a la izquierda, barras a la derecha ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)

    # === Panel izquierdo: grafo + flores ===
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i, (fx, fy) in enumerate(flowers):
        bid = assignments[i]
        color = branch_colors_rgb.get(bid, (1, 1, 1))
        ax1.plot(fx, fy, 'x', color=color, markersize=flower_size,
                 markeredgewidth=0.8, zorder=5)
    ax1.set_title('Grafo + Flores (color = rama asignada)', color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # === Panel derecho: barras flores por rama ===
    ax2.set_facecolor('#2a2a2a' if dark_bg else 'white')
    sorted_bids = sorted(branches.keys())
    counts = [len(flowers_per_branch.get(bid, [])) for bid in sorted_bids]
    colors_bar = [branch_colors_rgb.get(bid, (0.5, 0.5, 0.5)) for bid in sorted_bids]
    bar_labels = [f"R{bid}" for bid in sorted_bids]

    bars = ax2.barh(bar_labels, counts, color=colors_bar, edgecolor='white', linewidth=0.5)
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
#  5. Pipeline
# ---------------------------------------------------------------------------
def run(graph_path, json_dir, json_path=None,
        color_tolerance=25, proximity_gap=7,
        min_branch_pixels=15, trunk_short_ratio=0.30,
        flower_size=3, show_labels=True, dark_background=True, save_path=None):

    # --- Cargar imagen ---
    img = cv2.imread(graph_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {graph_path}")
    print(f"[INFO] Imagen: {img.shape[1]}x{img.shape[0]} px")

    # --- Auto-detectar JSON ---
    if json_path is None:
        json_path = auto_detect_json(graph_path, json_dir)
    print(f"[INFO] JSON flores: {json_path}")

    # --- Cargar flores ---
    flowers = load_flowers(json_path)
    print(f"[INFO] Flores cargadas: {len(flowers)}")

    # --- Extraer ramas por color (usando imagen sin texto blanco) ---
    bg_color = detect_background(img)

    # Eliminar pixeles blancos/casi blancos (texto de etiquetas) antes de detectar ramas
    img_clean = img.copy()
    white_mask = (img_clean[:, :, 0] > 200) & (img_clean[:, :, 1] > 200) & (img_clean[:, :, 2] > 200)
    img_clean[white_mask] = bg_color

    branches = extract_branches_by_color(
        img_clean, bg_color, color_tolerance, min_branch_pixels)
    print(f"[INFO] Ramas detectadas: {len(branches)}")

    # --- Adyacencia y jerarquia ---
    adjacency = build_adjacency(branches, img.shape, proximity_gap)
    trunk_ids = identify_trunk(branches, adjacency, trunk_short_ratio)
    levels = classify_hierarchy(branches, adjacency, trunk_ids)
    print(f"[INFO] Tronco: Ramas {sorted(trunk_ids)}")

    # --- Asignar flores a ramas ---
    assignments, distances = assign_flowers_to_branches(flowers, branches)

    # --- Resumen ---
    flowers_per_branch = defaultdict(int)
    for bid in assignments:
        flowers_per_branch[bid] += 1

    flowers_per_level = defaultdict(int)
    for bid, count in flowers_per_branch.items():
        lvl = levels.get(bid, -1)
        flowers_per_level[lvl] += count

    print(f"\n{'=' * 70}")
    print(f"  MAPA DE DENSIDAD FLORAL")
    print(f"{'=' * 70}")
    print(f"  {'Rama':>6}  {'Px':>6}  {'Nivel':>6}  {'Tipo':<20}  {'Flores':>6}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}  {'-'*6}")
    for bid in sorted(branches.keys()):
        b = branches[bid]
        lvl = levels[bid]
        name = _hier_name(lvl)
        fc = flowers_per_branch.get(bid, 0)
        trunk_mark = " *" if bid in trunk_ids else ""
        print(f"  R{bid:>4}  {b['size']:>6}  {lvl:>6}  {name:<20}  {fc:>6}{trunk_mark}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}  {'-'*6}")
    print(f"  {'TOTAL':>6}  {'':>6}  {'':>6}  {'':>20}  {len(flowers):>6}")
    print(f"{'=' * 70}")

    print(f"\n  Flores por nivel:")
    for lvl in sorted(flowers_per_level.keys()):
        name = _hier_name(lvl)
        print(f"    {name}: {flowers_per_level[lvl]}")

    avg_dist = np.mean(distances) if distances else 0
    max_dist = max(distances) if distances else 0
    print(f"\n  Distancia flor-rama: promedio={avg_dist:.1f}px, max={max_dist:.1f}px")
    print(f"{'=' * 70}\n")

    # --- Visualizar ---
    visualize(img, branches, levels, trunk_ids, flowers,
              assignments, distances, show_labels, dark_background,
              flower_size, save_path=save_path)

    return branches, levels, flowers, assignments


# =====================================================================
#  EJECUCION
# =====================================================================
if __name__ == "__main__":
    if os.path.isdir(GRAPH_PATH):
        input_files = sorted(glob.glob(os.path.join(GRAPH_PATH, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No se encontraron .png en: {GRAPH_PATH}")
        output_dir = os.path.join(os.path.dirname(GRAPH_PATH), "densidad floral")
    else:
        input_files = [GRAPH_PATH]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(GRAPH_PATH)), "densidad floral")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Procesando {len(input_files)} imagen(es)...")
    print(f"[INFO] Guardando en: {output_dir}\n")

    for img_path in input_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, fname + "_densidad.png")
        print(f"\n{'=' * 70}")
        print(f"  Procesando: {os.path.basename(img_path)}")
        print(f"{'=' * 70}")
        try:
            run(
                graph_path        = img_path,
                json_dir          = JSON_DIR,
                json_path         = JSON_PATH,
                color_tolerance   = COLOR_TOLERANCE,
                proximity_gap     = PROXIMITY_GAP,
                min_branch_pixels = MIN_BRANCH_PIXELS,
                trunk_short_ratio = TRUNK_SHORT_RATIO,
                flower_size       = FLOWER_SIZE,
                show_labels       = SHOW_LABELS,
                dark_background   = DARK_BACKGROUND,
                save_path         = save_path,
            )
            print(f"  [OK] Guardado: {save_path}")
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e}")

    print(f"\n[DONE] Listo. {len(input_files)} imagen(es) procesada(s).")
