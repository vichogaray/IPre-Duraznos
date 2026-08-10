"""
Trunk Heatmap - Mapa de calor del esqueleto por carga floral
============================================================
Lee el JSON de varillas (output de varilla_density.py) y dibuja TODO el
esqueleto coloreado por densidad floral acumulada (no por rama).

En cada pixel del esqueleto, el calor es la suma sobre TODAS las varillas
de:
    num_flowers * exp(-dist^2 / (2 * sigma^2))
donde dist es la distancia euclidiana entre el pixel del esqueleto y el
base_xy de la varilla. sigma controla cuanto se "esparce" la carga floral
de cada varilla a lo largo del esqueleto (en cm, convertido a px por
imagen via la calibracion de altura del arbol).

Esto captura ambos factores que pidio el usuario:
  - cuantas varillas hay         -> mas varillas cerca aportan mas calor
  - cual es su carga floral      -> peso = num_flowers de cada varilla

Input:
  - JSON de grafo (grafos json/)
  - JSON de varilla (densidad floral varilla/json/, output de varilla_density.py)

Output:
  - PNG con el esqueleto coloreado por mapa de calor + colorbar

Uso: ejecutar varilla_density.py PRIMERO, despues este script.
"""

# Rutas del proyecto centralizadas en src/common/paths.py. Se anade la raiz
# del repositorio al path para poder ejecutar este archivo directamente (F5).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from src.common.paths import DATA_DIR, GRAPHS_IMG_DIR, GRAPH_JSON_DIR


# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR    = GRAPH_JSON_DIR
VARILLA_JSON_DIR  = _os.path.join(DATA_DIR, "densidades", "densidad floral varilla", "json")
GRAFOS_IMG_DIR    = GRAPHS_IMG_DIR
OUTPUT_DIR        = _os.path.join(DATA_DIR, "densidades", "mapa de calor tronco")

SINGLE_IMAGE = None   # None = procesa todas las imagenes en lote
SHOW_PLOT    = False

# === ESCALA (igual que varilla_density.py) ===
ASSUMED_TREE_HEIGHT_CM = 350.0
PIXELS_PER_CM_OVERRIDE = None

# === KERNEL DE CALOR ===
# Cuanto se esparce la carga floral de cada varilla a lo largo del
# esqueleto (radio caracteristico del gaussiano, en cm).
# Subir -> heatmap mas suavizado (regiones grandes tibias).
# Bajar -> heatmap mas localizado (puntos calientes pequenos).
KERNEL_SIGMA_CM = 8.0

# === VISUALIZACION ===
DARK_BG          = True
HEAT_CMAP        = 'jet'       # azul -> cian -> verde -> amarillo -> rojo
# Escala de color ABSOLUTA compartida con Varilla_heatmap_2.py: fija el tope
# del colormap para que una misma carga floral acumulada reciba SIEMPRE el
# mismo color en v1 y en v2 (sin auto-escalar al maximo de cada imagen).
# None -> auto-escala al heat.max() de la imagen (comportamiento antiguo).
VMAX_FIJO        = 60.0
SKEL_THICKNESS_PX = 3          # grosor del esqueleto pintado (1 = 1px, 3 = 3x3, ...)
FLOWER_SIZE      = 3
SHOW_FLOWERS     = True        # mostrar flores como puntos blancos
SHOW_BASE_DOTS   = False       # marcar base_xy de cada varilla con un anillo

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import glob
import json
import numpy as np
import cv2

import matplotlib
if not SHOW_PLOT:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


# =====================================================================
#  I/O
# =====================================================================

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def auto_detect_varilla_json(graph_json_basename, varilla_json_dir):
    """varilla_density.py guarda como <graph_basename>_varilla.json"""
    base = os.path.splitext(graph_json_basename)[0]
    p = os.path.join(varilla_json_dir, base + "_varilla.json")
    if not os.path.exists(p):
        raise FileNotFoundError("No se encontro varilla JSON: " + p +
                                "\nCorre varilla_density.py primero.")
    return p


def auto_detect_image(graph_image_name, grafos_img_dir):
    p = os.path.join(grafos_img_dir, graph_image_name)
    if not os.path.exists(p):
        raise FileNotFoundError("No se encontro imagen: " + p)
    return p


# =====================================================================
#  ESCALA + ESQUELETO
# =====================================================================

# La calibracion px <-> cm vive en src/common/geometry.py para que exista una
# sola definicion compartida por todos los metodos. Se anade la raiz del
# repositorio al path para poder ejecutar este archivo directamente (F5).
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.geometry import estimate_pixels_per_cm


def build_skeleton_pixels(graph_data):
    """Devuelve array (n, 2) con (x, y) de TODOS los pixeles del esqueleto."""
    pts = []
    for b in graph_data['branches']:
        for px in b['pixels']:
            y, x = px[0], px[1]
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


# =====================================================================
#  CALCULO DEL MAPA DE CALOR
# =====================================================================

def compute_heat(skel_xy, varillas, sigma_px):
    """
    Para cada pixel del esqueleto, acumula:
        sum_v  num_flowers(v) * exp(-||p - base_xy(v)||^2 / (2 sigma^2))

    Usa solo varillas cuya base_xy esta definida.
    """
    n = len(skel_xy)
    heat = np.zeros(n, dtype=np.float64)
    if n == 0 or sigma_px <= 0:
        return heat
    inv_2s2 = 1.0 / (2.0 * sigma_px * sigma_px)
    for v in varillas:
        if v.get('base_xy') is None:
            continue
        bx, by = v['base_xy']
        weight = float(v.get('num_flowers', 0))
        if weight <= 0:
            continue
        dx = skel_xy[:, 0] - bx
        dy = skel_xy[:, 1] - by
        d2 = dx * dx + dy * dy
        heat += weight * np.exp(-d2 * inv_2s2)
    return heat


# =====================================================================
#  VISUALIZACION
# =====================================================================

def visualize_heatmap(img, skel_xy, heat, varilla_data, sigma_cm, save_path):
    txt_col = 'white' if DARK_BG else 'black'
    fig_bg  = '#1e1e1e' if DARK_BG else '#f0f0f0'

    # Tamano de figura segun aspect ratio de la imagen + espacio colorbar
    h_img, w_img = img.shape[:2]
    base_h = 9.0
    fig_w = base_h * (w_img / float(h_img)) + 1.3
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, base_h),
                           constrained_layout=True)
    fig.patch.set_facecolor(fig_bg)

    if VMAX_FIJO is not None:
        heat_max = float(VMAX_FIJO)
    else:
        heat_max = float(heat.max()) if len(heat) and heat.max() > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=heat_max)

    # Fondo en escala de grises -> RGB float [0, 1]. Sobre esta misma
    # matriz REEMPLAZAMOS los pixeles del esqueleto con su color de
    # calor (no es scatter; es pintura directa de pixeles).
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0

    # Color de calor por cada pixel del esqueleto via colormap
    cmap_obj = plt.get_cmap(HEAT_CMAP)
    heat_colors = cmap_obj(norm(heat))[:, :3]  # (n, 3) RGB

    # Indices enteros y mascara dentro del frame
    xs_i = np.round(skel_xy[:, 0]).astype(int)
    ys_i = np.round(skel_xy[:, 1]).astype(int)

    # Pintar primero los frios para que los calientes queden encima en
    # zonas donde dos pixeles del esqueleto caigan en la misma celda
    order = np.argsort(heat)
    xs_i = xs_i[order]
    ys_i = ys_i[order]
    heat_colors = heat_colors[order]

    # Grosor: dilatacion en una vecindad cuadrada de radio (t-1)//2
    t = max(1, int(SKEL_THICKNESS_PX))
    radius = (t - 1) // 2
    for dy_off in range(-radius, radius + 1):
        for dx_off in range(-radius, radius + 1):
            yo = ys_i + dy_off
            xo = xs_i + dx_off
            m = (yo >= 0) & (yo < h_img) & (xo >= 0) & (xo < w_img)
            rgb[yo[m], xo[m]] = heat_colors[m]

    ax.imshow(rgb)

    # ScalarMappable solo para la colorbar (no se dibuja)
    sm = ScalarMappable(cmap=HEAT_CMAP, norm=norm)
    sm.set_array([])

    # Flores opcionales (puntos pequenos blancos)
    if SHOW_FLOWERS:
        for f in varilla_data.get('flowers', []):
            ax.plot(f['x'], f['y'], 'o', color='white',
                    markersize=FLOWER_SIZE, markeredgewidth=0.3,
                    markeredgecolor='black', zorder=5)

    # Anillos en cada base_xy
    if SHOW_BASE_DOTS:
        for v in varilla_data.get('varillas', []):
            if v.get('base_xy') is None:
                continue
            ax.add_patch(Circle(v['base_xy'], 4.0, fill=False,
                                edgecolor='cyan', linewidth=1.0, zorder=6))

    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Carga floral acumulada (flores ponderadas)',
                   color=txt_col, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=txt_col)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=txt_col)
    cbar.outline.set_edgecolor(txt_col)

    title = ("Mapa de calor del esqueleto  |  Varillas: {nv}  |  "
             "Flores: {nf}  |  sigma = {s} cm").format(
                 nv=varilla_data.get('n_varillas', 0),
                 nf=varilla_data.get('n_flowers', 0),
                 s=sigma_cm)
    ax.set_title(title, color=txt_col, fontsize=11)
    ax.axis('off')
    ax.set_facecolor(fig_bg)

    if SHOW_PLOT:
        plt.show()
        plt.close(fig)
    else:
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig_bg)
        plt.close(fig)


# =====================================================================
#  RUN POR IMAGEN
# =====================================================================

def run_one(graph_json_path):
    """Lee el JSON de varilla desde disco y genera el heatmap (uso individual)."""
    varilla_json = auto_detect_varilla_json(
        os.path.basename(graph_json_path), VARILLA_JSON_DIR)
    varilla_data = load_json(varilla_json)
    return run_one_data(graph_json_path, varilla_data)


def run_one_data(graph_json_path, varilla_data, save_path=None):
    """
    Genera el heatmap recibiendo varilla_data EN MEMORIA (no lee de disco).
    Es la entrada que usa el pipeline automático / supremo.
    Si save_path es None, usa el comportamiento por defecto (SHOW_PLOT / OUTPUT_DIR).
    """
    graph_data = load_json(graph_json_path)
    image_name = graph_data['image']

    img_path = auto_detect_image(image_name, GRAFOS_IMG_DIR)
    img = cv2.imread(img_path)

    if PIXELS_PER_CM_OVERRIDE is not None:
        px_per_cm = float(PIXELS_PER_CM_OVERRIDE)
        scale_src = "override"
    else:
        px_per_cm = estimate_pixels_per_cm(graph_data, ASSUMED_TREE_HEIGHT_CM)
        scale_src = "auto"
    sigma_px = KERNEL_SIGMA_CM * px_per_cm
    print("    Escala: {0:.2f} px/cm [{1}]  |  sigma: {2:.1f} px "
          "({3} cm)".format(px_per_cm, scale_src, sigma_px, KERNEL_SIGMA_CM))

    skel_xy = build_skeleton_pixels(graph_data)
    heat = compute_heat(skel_xy, varilla_data['varillas'], sigma_px)
    print("    Pix esqueleto: {0}  |  Varillas: {1}  |  "
          "Heat max: {2:.2f}  |  Heat sum: {3:.2f}".format(
              len(skel_xy), len(varilla_data['varillas']),
              float(heat.max()) if len(heat) else 0.0,
              float(heat.sum()) if len(heat) else 0.0))

    # save_path explícito (supremo) tiene prioridad; si no, comportamiento default
    if save_path is None and not SHOW_PLOT:
        fname = os.path.splitext(os.path.basename(graph_json_path))[0]
        save_path = os.path.join(OUTPUT_DIR, fname + "_heatmap.png")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    visualize_heatmap(img, skel_xy, heat, varilla_data,
                      KERNEL_SIGMA_CM, save_path)
    if save_path:
        print("    [OK] -> {0}".format(os.path.basename(save_path)))
    else:
        print("    [OK] (mostrado en pantalla)")
    return save_path


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    if SINGLE_IMAGE:
        json_files = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
    else:
        # Iterar sobre la carpeta de varilla JSONs: solo procesa las imagenes
        # que ya tienen su varilla calculada (output de varilla_density.py).
        # Para cada *_varilla.json se deriva su graph JSON correspondiente
        # quitando el sufijo "_varilla".
        varilla_files = sorted(glob.glob(
            os.path.join(VARILLA_JSON_DIR, "*_varilla.json")))
        json_files = []
        for vf in varilla_files:
            base = os.path.basename(vf)[:-len("_varilla.json")]
            graph_jf = os.path.join(GRAPH_JSON_DIR, base + ".json")
            if os.path.exists(graph_jf):
                json_files.append(graph_jf)
            else:
                print("[WARN] No se encontro graph JSON para: " + base)
    if not json_files:
        raise FileNotFoundError(
            "No se encontraron varilla JSONs en: " + VARILLA_JSON_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[INFO] Procesando {0} imagen(es)...\n".format(len(json_files)))

    ok, errors = 0, 0
    for jf in json_files:
        print("  " + os.path.basename(jf))
        try:
            run_one(jf)
            ok += 1
        except Exception as e:
            print("    [ERROR] {0}".format(e))
            errors += 1
    print("\n[DONE] {0} OK, {1} errores.".format(ok, errors))
