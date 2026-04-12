"""
Branch Hierarchy Classifier — Colored Skeleton Graphs (v3)
===========================================================
Clasifica ramas de un esqueleto coloreado en jerarquía:
  Tronco -> Rama Principal -> Rama Secundaria -> Rama Terciaria -> ...

Lógica:
  1. Agrupa píxeles por color -> cada color = una rama
     (mismo color = misma rama, aunque estén separados por unos píxeles)
  2. Detecta adyacencia espacial entre ramas de distinto color
  3. Identifica el tronco (rama más baja de la imagen)
  4. BFS desde el tronco para asignar niveles jerárquicos

Uso:
    1. Pon tu ruta en SKELETON_PATH
    2. Ajusta COLOR_TOLERANCE y PROXIMITY_GAP si es necesario
    3. Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACIÓN — MODIFICA AQUÍ
# =====================================================================

SKELETON_PATH = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"  # carpeta o archivo individual

COLOR_TOLERANCE   = 25    # Tolerancia RGB para agrupar colores similares
PROXIMITY_GAP     = 7     # Distancia max (px) para considerar ramas "conectadas"
MIN_BRANCH_PIXELS = 15    # Minimo de pixeles para que cuente como rama
TRUNK_SHORT_RATIO = 0.30  # Umbral: rama es "corta" si size < mediana * este valor
LINE_THICKNESS    = 2     # Grosor visual en la imagen de salida
SHOW_LABELS       = True  # Mostrar etiquetas sobre la imagen
DARK_BACKGROUND   = True  # Fondo oscuro en la visualizacion



def merge(*branch_ids):
    """Junta varias ramas en una sola."""
    return ('merge', list(branch_ids))

MANUAL_EDITS = []


import numpy as np
import cv2
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')  # backend sin ventana para guardar archivos
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob


# ---------------------------------------------------------------------------
#  Categorías jerárquicas
# ---------------------------------------------------------------------------
HIERARCHY = [
    # (nombre,           hex,       BGR para OpenCV)
    ("Tronco",           "#E02020", (32,  32,  224)),
    ("Rama Principal",   "#20C820", (0,   200, 32)),
    ("Rama Secundaria",  "#2896FF", (255, 150, 40)),
    ("Rama Terciaria",   "#FF8C00", (0,   140, 255)),
    ("Rama Cuaternaria", "#D020D0", (208, 32,  208)),
    ("Rama Quinaria",    "#00BFBF", (191, 191, 0)),
]


def _hier_name(level):
    if level < 0:
        return "Desconectada"
    if level < len(HIERARCHY):
        return HIERARCHY[level][0]
    return f"Rama Nivel {level}"


def _hier_hex(level):
    if level < 0:
        return "#555555"
    return HIERARCHY[min(level, len(HIERARCHY) - 1)][1]


def _hier_bgr(level):
    if level < 0:
        return (80, 80, 80)
    return HIERARCHY[min(level, len(HIERARCHY) - 1)][2]


# ---------------------------------------------------------------------------
#  1. Detectar fondo
# ---------------------------------------------------------------------------
def detect_background(img):
    """Detecta el color de fondo usando muestras de las esquinas."""
    h, w = img.shape[:2]
    m = min(10, h // 10, w // 10)
    samples = []
    for y in range(m):
        for x in range(m):
            samples.extend([img[y, x], img[y, w-1-x],
                            img[h-1-y, x], img[h-1-y, w-1-x]])
    return np.median(samples, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
#  2. Extraer ramas por color
# ---------------------------------------------------------------------------
def extract_branches_by_color(img, bg_color, color_tol, min_pixels):
    """
    Agrupa todos los píxeles de primer plano por color.
    Mismo color (dentro de tolerancia) = misma rama, sin importar separación espacial.
    """
    # Máscara de primer plano: píxeles que difieren del fondo
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    fg_mask = np.max(diff, axis=2) > color_tol

    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        raise ValueError("No se detectaron ramas en la imagen.")

    colors = img[ys, xs]  # (N, 3) BGR

    # Cuantizar colores para agrupar similares
    q = max(color_tol // 2, 6)
    quantized = ((colors.astype(int) + q // 2) // q) * q
    quantized = np.clip(quantized, 0, 255)

    # Agrupar por color cuantizado
    groups = defaultdict(list)
    for i in range(len(ys)):
        key = (int(quantized[i, 0]), int(quantized[i, 1]), int(quantized[i, 2]))
        groups[key].append((int(ys[i]), int(xs[i])))

    # Construir diccionario de ramas, filtrando grupos pequeños
    branches = {}
    bid = 0
    for color_key, pixel_list in groups.items():
        if len(pixel_list) < min_pixels:
            continue
        ys_b = [p[0] for p in pixel_list]
        xs_b = [p[1] for p in pixel_list]
        # Color representativo: promedio real (no cuantizado) de los píxeles
        indices = [i for i in range(len(ys))
                   if (int(quantized[i, 0]), int(quantized[i, 1]),
                       int(quantized[i, 2])) == color_key]
        avg_color = np.mean(colors[indices], axis=0).astype(np.uint8)

        branches[bid] = {
            'pixels': set(pixel_list),
            'color_bgr': tuple(int(c) for c in avg_color),
            'color_hex': '#{:02x}{:02x}{:02x}'.format(
                int(avg_color[2]), int(avg_color[1]), int(avg_color[0])),
            'avg_y': float(np.mean(ys_b)),
            'avg_x': float(np.mean(xs_b)),
            'max_y': max(ys_b),
            'min_y': min(ys_b),
            'max_x': max(xs_b),
            'min_x': min(xs_b),
            'size': len(pixel_list),
        }
        bid += 1

    if not branches:
        raise ValueError("No se encontraron ramas con suficientes píxeles.")

    return branches


# ---------------------------------------------------------------------------
#  3. Adyacencia espacial entre ramas
# ---------------------------------------------------------------------------
def build_adjacency(branches, img_shape, max_gap):
    """
    Dos ramas son adyacentes si algún píxel de una está a ≤ max_gap
    píxeles de algún píxel de la otra.
    """
    h, w = img_shape[:2]

    # Imagen de etiquetas
    label_img = np.full((h, w), -1, dtype=np.int32)
    for bid, branch in branches.items():
        for (y, x) in branch['pixels']:
            label_img[y, x] = bid

    adjacency = defaultdict(set)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max_gap + 1, 2 * max_gap + 1))

    for bid in branches:
        mask = (label_img == bid).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel)

        neighbors_in_region = label_img[dilated > 0]
        unique_nb = set(int(v) for v in neighbors_in_region
                        if v != -1 and v != bid)
        for nb in unique_nb:
            adjacency[bid].add(nb)
            adjacency[nb].add(bid)

    return adjacency


# ---------------------------------------------------------------------------
#  4. Identificar tronco
# ---------------------------------------------------------------------------
def identify_trunk(branches, adjacency, trunk_short_ratio=0.30):
    """
    El tronco puede estar compuesto por VARIAS ramas.
    1. Semilla: la rama mas baja de la imagen (mayor avg_y)
    2. Expansion: si una rama vecina es CORTA comparada con la mediana
       del arbol, tambien es parte del tronco
    3. Parar cuando los vecinos son largos (esos son ramas principales)

    Retorna un set de branch IDs que forman el tronco.
    """
    # Semilla: rama mas baja
    seed = max(branches.keys(),
               key=lambda bid: (branches[bid]['avg_y'],
                                branches[bid]['max_y']))

    # Referencia: mediana de tamanio de todas las ramas
    sizes = sorted(b['size'] for b in branches.values())
    median_size = sizes[len(sizes) // 2]
    short_threshold = median_size * trunk_short_ratio

    # BFS: expandir tronco incorporando vecinos cortos
    trunk_set = {seed}
    queue = deque([seed])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in trunk_set:
                if branches[neighbor]['size'] <= short_threshold:
                    trunk_set.add(neighbor)
                    queue.append(neighbor)

    return trunk_set


# ---------------------------------------------------------------------------
#  5. Clasificacion jerarquica (BFS desde tronco)
# ---------------------------------------------------------------------------
def classify_hierarchy(branches, adjacency, trunk_ids):
    """
    Nivel 0 = Tronco (conjunto de ramas)
    Nivel 1 = Ramas principales (conectadas al tronco)
    Nivel 2 = Ramas secundarias (conectadas a las principales)
    ...
    """
    levels = {tid: 0 for tid in trunk_ids}
    queue = deque(list(trunk_ids))

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in levels:
                levels[neighbor] = levels[current] + 1
                queue.append(neighbor)

    # Ramas desconectadas
    for bid in branches:
        if bid not in levels:
            levels[bid] = -1

    return levels


# ---------------------------------------------------------------------------
#  6. Ediciones manuales
# ---------------------------------------------------------------------------
def apply_manual_edits(branches, edits):
    """Aplica merge de ramas."""
    if not edits:
        return branches

    for edit in edits:
        op, ids = edit
        if op == 'merge':
            valid = [i for i in ids if i in branches]
            if len(valid) < 2:
                invalid = [i for i in ids if i not in branches]
                if invalid:
                    print(f"  [WARN] merge: rama(s) {invalid} no existen. "
                          f"Disponibles: {sorted(branches.keys())}")
                continue
            target = min(valid)
            for src in valid:
                if src != target:
                    branches[target]['pixels'] |= branches[src]['pixels']
                    del branches[src]
            # Recalcular estadísticas
            ys = [p[0] for p in branches[target]['pixels']]
            xs = [p[1] for p in branches[target]['pixels']]
            branches[target].update({
                'size': len(branches[target]['pixels']),
                'avg_y': float(np.mean(ys)), 'avg_x': float(np.mean(xs)),
                'max_y': max(ys), 'min_y': min(ys),
                'max_x': max(xs), 'min_x': min(xs),
            })
            print(f"  [EDIT] Merge {valid} -> Rama {target}")

    return branches


# ---------------------------------------------------------------------------
#  7. Visualización
# ---------------------------------------------------------------------------
def visualize(img, branches, levels, trunk_ids,
              show_labels=True, dark_bg=True, line_thickness=2, save_path=None):
    h, w = img.shape[:2]

    txt_col = 'white' if dark_bg else 'black'
    fig_bg = '#1e1e1e' if dark_bg else '#f0f0f0'
    bg_val = 30 if dark_bg else 240

    # --- Imagen clasificada (coloreada por nivel jerárquico) ---
    classified = np.full((h, w, 3), bg_val, dtype=np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (line_thickness * 2 + 1, line_thickness * 2 + 1))

    # Dibujar de niveles altos a bajos (tronco encima de todo)
    max_level = max((lv for lv in levels.values() if lv >= 0), default=0)
    for lvl in range(max_level, -1, -1):
        color = _hier_bgr(lvl)
        mask = np.zeros((h, w), dtype=np.uint8)
        for bid, branch in branches.items():
            if levels.get(bid) == lvl:
                for (y, x) in branch['pixels']:
                    if 0 <= y < h and 0 <= x < w:
                        mask[y, x] = 1
        if line_thickness > 1:
            mask = cv2.dilate(mask, kernel)
        classified[mask > 0] = color

    # Ramas desconectadas
    mask_disc = np.zeros((h, w), dtype=np.uint8)
    for bid, branch in branches.items():
        if levels.get(bid, -1) < 0:
            for (y, x) in branch['pixels']:
                if 0 <= y < h and 0 <= x < w:
                    mask_disc[y, x] = 1
    if line_thickness > 1:
        mask_disc = cv2.dilate(mask_disc, kernel)
    classified[mask_disc > 0] = _hier_bgr(-1)

    # --- Matplotlib ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor(fig_bg)
    fig.suptitle('Clasificación Jerárquica de Ramas', fontsize=15,
                 fontweight='bold', color=txt_col)

    # Izquierda: imagen original
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Esqueleto Original (por color)', color=txt_col, fontsize=12)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # Etiquetas con ID de rama sobre imagen original
    if show_labels:
        for bid, branch in branches.items():
            cx, cy = int(branch['avg_x']), int(branch['avg_y'])
            ax1.text(cx, cy, str(bid), fontsize=8, fontweight='bold',
                     color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                               alpha=0.75, edgecolor='none'), zorder=10)

    # Derecha: clasificación jerárquica
    ax2 = axes[1]
    ax2.imshow(cv2.cvtColor(classified, cv2.COLOR_BGR2RGB))
    ax2.set_title('Clasificación Jerárquica', color=txt_col, fontsize=12)
    ax2.axis('off')
    ax2.set_facecolor(fig_bg)

    # Etiquetas con nombre de jerarquía
    if show_labels:
        for bid, branch in branches.items():
            lvl = levels.get(bid, -1)
            name = _hier_name(lvl)
            cx, cy = int(branch['avg_x']), int(branch['avg_y'])
            marker = " *" if bid in trunk_ids else ""
            label = f"R{bid}: {name}{marker}"
            ax2.text(cx, cy, label, fontsize=7, fontweight='bold',
                     color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                               alpha=0.85, edgecolor='none'), zorder=10)

    # Leyenda de jerarquía
    seen_levels = sorted(set(lv for lv in levels.values()))
    legend_elements = []
    for lvl in seen_levels:
        name = _hier_name(lvl)
        color_hex = _hier_hex(lvl)
        count = sum(1 for lv in levels.values() if lv == lvl)
        legend_elements.append(
            Patch(facecolor=color_hex,
                  label=f"{name} ({count} rama{'s' if count != 1 else ''})"))

    legend_bg = '#2a2a2a' if dark_bg else 'white'
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9,
               framealpha=0.9, facecolor=legend_bg,
               edgecolor='gray', labelcolor=txt_col)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
#  8. Pipeline principal
# ---------------------------------------------------------------------------
def run(skeleton_path, color_tolerance=25, proximity_gap=7,
        min_branch_pixels=15, trunk_short_ratio=0.30, manual_edits=None,
        show_labels=True, dark_background=True, line_thickness=2, save_path=None):

    # Cargar imagen
    img = cv2.imread(skeleton_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {skeleton_path}")
    print(f"[INFO] Imagen: {img.shape[1]}x{img.shape[0]} px")

    # Detectar fondo
    bg_color = detect_background(img)
    print(f"[INFO] Fondo detectado (BGR): {tuple(bg_color)}")

    # Extraer ramas por color
    branches = extract_branches_by_color(
        img, bg_color, color_tolerance, min_branch_pixels)
    print(f"[INFO] Ramas detectadas: {len(branches)}")
    for bid in sorted(branches):
        b = branches[bid]
        print(f"  Rama {bid}: {b['size']:5d} px  |  color {b['color_hex']}  |"
              f"  avg_y={b['avg_y']:.0f}")

    # Ediciones manuales
    if manual_edits:
        print(f"\n[EDIT] Aplicando {len(manual_edits)} edición(es)...")
        branches = apply_manual_edits(branches, manual_edits)
        print(f"[EDIT] Ramas después de ediciones: {len(branches)}")

    # Adyacencia
    adjacency = build_adjacency(branches, img.shape, proximity_gap)
    print(f"\n[INFO] Adyacencias:")
    for bid in sorted(adjacency):
        print(f"  Rama {bid} <-> Ramas {sorted(adjacency[bid])}")

    # Tronco (puede ser un conjunto de ramas)
    trunk_ids = identify_trunk(branches, adjacency, trunk_short_ratio)
    print(f"\n[INFO] Tronco identificado: Ramas {sorted(trunk_ids)}")

    # Clasificacion jerarquica
    levels = classify_hierarchy(branches, adjacency, trunk_ids)

    # Resumen
    print(f"\n{'=' * 70}")
    print(f"  CLASIFICACION JERARQUICA")
    print(f"{'=' * 70}")
    for bid in sorted(branches.keys()):
        b = branches[bid]
        lvl = levels[bid]
        name = _hier_name(lvl)
        marker = "  << TRONCO" if bid in trunk_ids else ""
        print(f"  Rama {bid:2d}:  {b['size']:5d} px  |  color {b['color_hex']}"
              f"  |  Nivel {lvl:2d}  ->  {name}{marker}")
    print(f"{'=' * 70}")
    max_lvl = max((lv for lv in levels.values() if lv >= 0), default=0)
    print(f"  Profundidad maxima: {max_lvl}")
    disc = sum(1 for lv in levels.values() if lv < 0)
    if disc:
        print(f"  Ramas desconectadas: {disc}")
    print(f"{'=' * 70}\n")

    # Visualizar
    visualize(img, branches, levels, trunk_ids,
              show_labels, dark_background, line_thickness, save_path=save_path)

    return branches, levels


# =====================================================================
#  EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    # Determinar lista de imágenes a procesar
    if os.path.isdir(SKELETON_PATH):
        input_files = sorted(glob.glob(os.path.join(SKELETON_PATH, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No se encontraron .png en: {SKELETON_PATH}")
        output_dir = os.path.join(os.path.dirname(SKELETON_PATH), "ramas identificadas")
    else:
        input_files = [SKELETON_PATH]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(SKELETON_PATH)), "ramas identificadas")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Procesando {len(input_files)} imagen(es)...")
    print(f"[INFO] Guardando resultados en: {output_dir}\n")

    for img_path in input_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, fname + "_clasificado.png")
        print(f"\n{'=' * 70}")
        print(f"  Procesando: {os.path.basename(img_path)}")
        print(f"{'=' * 70}")
        try:
            run(
                skeleton_path     = img_path,
                color_tolerance   = COLOR_TOLERANCE,
                proximity_gap     = PROXIMITY_GAP,
                min_branch_pixels = MIN_BRANCH_PIXELS,
                trunk_short_ratio = TRUNK_SHORT_RATIO,
                manual_edits      = MANUAL_EDITS,
                show_labels       = SHOW_LABELS,
                dark_background   = DARK_BACKGROUND,
                line_thickness    = LINE_THICKNESS,
                save_path         = save_path,
            )
            print(f"  [OK] Guardado: {save_path}")
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e}")

    print(f"\n[DONE] Listo. {len(input_files)} imagen(es) procesada(s).")
