"""
Varilla Heatmap 2 - Mapa de calor del esqueleto por carga floral (para v2)
==========================================================================
Analogo a heatmap_shoots.py pero para candidate_shoots.py (varillas por
disparo desde el tronco). Dibuja TODO el esqueleto coloreado por densidad
floral acumulada (no por rama).

DIFERENCIA CLAVE con el heatmap de v1:
  - v1 guarda un JSON con base_xy por varilla; este script NO lee JSON: importa
    varilla_density_2 y CALCULA las varillas en memoria (un solo paso).
  - En v2 cada varilla nace en el tronco en el punto 'origin' (donde el rayo
    sale del esqueleto). Ese 'origin' es el analogo exacto del 'base_xy' de v1,
    asi que el calor se deposita ahi.

En cada pixel del esqueleto, el calor es la suma sobre TODAS las varillas de:
    num_flowers * exp(-dist^2 / (2 * sigma^2))
donde dist es la distancia euclidiana entre el pixel del esqueleto y el
'origin' de la varilla. sigma controla cuanto se "esparce" la carga floral
de cada varilla a lo largo del esqueleto (en cm, convertido a px por imagen).

Input:
  - JSON de grafo (grafos json/)
  - JSON de flores (json flores/)  [via varilla_density_2]

Output:
  - PNG con el esqueleto coloreado por mapa de calor + colorbar

Uso: F5. Toma los mismos parametros de clustering/fusion/rayos que
candidate_shoots.py (se importan de ahi).
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR    = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIR    = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
OUTPUT_DIR        = r"C:\Users\vgara\OneDrive\Desktop\IPre\densidades\mapa de calor tronco v2"

# None -> batch sobre todos los graph JSON; o un nombre concreto para una imagen.
SINGLE_IMAGE = "imgs_frame1_00000_graph.json"
SHOW_PLOT    = True

# === ESCALA ===
# Se reutiliza la escala de varilla_density_2 (incluye PIXELS_PER_CM_OVERRIDE).

# === KERNEL DE CALOR ===
# Cuanto se esparce la carga floral de cada varilla a lo largo del esqueleto
# (radio caracteristico del gaussiano, en cm).
# Subir  -> heatmap mas suavizado (regiones grandes tibias).
# Bajar  -> heatmap mas localizado (puntos calientes pequenos).
KERNEL_SIGMA_CM = 8.0

# === VISUALIZACION ===
DARK_BG          = True
HEAT_CMAP        = 'jet'       # azul -> cian -> verde -> amarillo -> rojo
# Escala de color ABSOLUTA compartida con Varilla_heatmap.py: fija el tope del
# colormap para que una misma carga floral acumulada reciba SIEMPRE el mismo
# color en v1 y en v2 (sin auto-escalar al maximo de cada imagen).
# Debe ser IDENTICO al VMAX_FIJO de Varilla_heatmap.py.
# None -> auto-escala al heat.max() de la imagen (comportamiento antiguo).
VMAX_FIJO        = 60.0
SKEL_THICKNESS_PX = 3          # grosor del esqueleto pintado (1 = 1px, 3 = 3x3, ...)
FLOWER_SIZE      = 3
SHOW_FLOWERS     = True        # mostrar flores como puntos blancos
SHOW_BASE_DOTS   = False       # marcar el origin de cada varilla con un anillo

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import sys
import glob
import numpy as np
import cv2

import matplotlib
if not SHOW_PLOT:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

# Importar el pipeline de v2 (renombrado a candidate_shoots y movido a
# src/assignment/shoot_reconstruction/). Se anade la raiz del repositorio al
# path para poder ejecutar este archivo directamente (F5).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src.assignment.shoot_reconstruction.candidate_shoots as v2


# =====================================================================
#  ESQUELETO
# =====================================================================

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
        sum_v  num_flowers(v) * exp(-||p - origin(v)||^2 / (2 sigma^2))

    Usa el 'origin' de cada varilla (punto donde nace en el tronco), que es el
    analogo del 'base_xy' de v1.
    """
    n = len(skel_xy)
    heat = np.zeros(n, dtype=np.float64)
    if n == 0 or sigma_px <= 0:
        return heat
    inv_2s2 = 1.0 / (2.0 * sigma_px * sigma_px)
    for v in varillas:
        origin = v.get('origin')
        if origin is None:
            continue
        bx, by = origin
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

def visualize_heatmap(img, skel_xy, heat, varillas, flowers, n_flowers,
                      sigma_cm, save_path):
    txt_col = 'white' if DARK_BG else 'black'
    fig_bg  = '#1e1e1e' if DARK_BG else '#f0f0f0'

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

    # Fondo en escala de grises -> RGB float [0, 1]. Sobre esta misma matriz
    # REEMPLAZAMOS los pixeles del esqueleto con su color de calor.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0

    cmap_obj = plt.get_cmap(HEAT_CMAP)
    heat_colors = cmap_obj(norm(heat))[:, :3]  # (n, 3) RGB

    xs_i = np.round(skel_xy[:, 0]).astype(int)
    ys_i = np.round(skel_xy[:, 1]).astype(int)

    # Pintar primero los frios para que los calientes queden encima.
    order = np.argsort(heat)
    xs_i = xs_i[order]
    ys_i = ys_i[order]
    heat_colors = heat_colors[order]

    # Grosor: dilatacion en vecindad cuadrada de radio (t-1)//2
    t = max(1, int(SKEL_THICKNESS_PX))
    radius = (t - 1) // 2
    for dy_off in range(-radius, radius + 1):
        for dx_off in range(-radius, radius + 1):
            yo = ys_i + dy_off
            xo = xs_i + dx_off
            m = (yo >= 0) & (yo < h_img) & (xo >= 0) & (xo < w_img)
            rgb[yo[m], xo[m]] = heat_colors[m]

    ax.imshow(rgb)

    sm = ScalarMappable(cmap=HEAT_CMAP, norm=norm)
    sm.set_array([])

    if SHOW_FLOWERS:
        for (fx, fy) in flowers:
            ax.plot(fx, fy, 'o', color='white',
                    markersize=FLOWER_SIZE, markeredgewidth=0.3,
                    markeredgecolor='black', zorder=5)

    if SHOW_BASE_DOTS:
        for v in varillas:
            origin = v.get('origin')
            if origin is None:
                continue
            ax.add_patch(Circle(origin, 4.0, fill=False,
                                edgecolor='cyan', linewidth=1.0, zorder=6))

    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Carga floral acumulada (flores ponderadas)',
                   color=txt_col, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=txt_col)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=txt_col)
    cbar.outline.set_edgecolor(txt_col)

    title = ("Mapa de calor del esqueleto (v2)  |  Varillas: {nv}  |  "
             "Flores: {nf}  |  sigma = {s} cm").format(
                 nv=len(varillas), nf=n_flowers, s=sigma_cm)
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
    graph_data = v2.load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = v2.auto_detect_flower_json(image_name, v2.JSON_FLORES_DIR)
    flowers     = v2.load_flowers(flower_json)
    img_path    = v2.auto_detect_graph_image(image_name, v2.GRAFOS_IMG_DIRS)
    img         = cv2.imread(img_path)

    if not flowers:
        print("    [SKIP] Sin flores.")
        return

    # Escala del HEATMAP: identica a la de Varilla_heatmap.py (v1), que usa la
    # estimacion auto por altura del arbol. Ignoramos PIXELS_PER_CM_OVERRIDE
    # aqui a proposito, para que sigma_px coincida con v1 y la DISPERSION del
    # calor sea la misma. (El pipeline de varillas de v2 sigue usando su propia
    # escala internamente; esto solo afecta el esparcido gaussiano del heatmap.)
    px_per_cm = v2.estimate_pixels_per_cm(
        graph_data, v2.ASSUMED_TREE_HEIGHT_CM)
    scale_src = "auto"
    sigma_px = KERNEL_SIGMA_CM * px_per_cm
    print("    Escala: {0:.2f} px/cm [{1}]  |  sigma: {2:.1f} px "
          "({3} cm)".format(px_per_cm, scale_src, sigma_px, KERNEL_SIGMA_CM))

    # Calcular varillas en memoria con el pipeline completo de v2.
    hp = v2.build_hyperparams(px_per_cm)
    varillas, _ = v2.assign_varillas_v2(graph_data, flowers, hp)

    skel_xy = build_skeleton_pixels(graph_data)
    heat = compute_heat(skel_xy, varillas, sigma_px)
    print("    Pix esqueleto: {0}  |  Varillas: {1}  |  "
          "Heat max: {2:.2f}  |  Heat sum: {3:.2f}".format(
              len(skel_xy), len(varillas),
              float(heat.max()) if len(heat) else 0.0,
              float(heat.sum()) if len(heat) else 0.0))

    if SHOW_PLOT:
        save_path = None
    else:
        fname = os.path.splitext(os.path.basename(graph_json_path))[0]
        save_path = os.path.join(OUTPUT_DIR, fname + "_heatmap_v2.png")
    visualize_heatmap(img, skel_xy, heat, varillas, flowers, len(flowers),
                      KERNEL_SIGMA_CM, save_path)
    if save_path:
        print("    [OK] -> {0}".format(os.path.basename(save_path)))
    else:
        print("    [OK] (mostrado en pantalla)")


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    if SINGLE_IMAGE:
        json_files = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
    else:
        json_files = sorted(glob.glob(os.path.join(GRAPH_JSON_DIR, "*.json")))
    if not json_files:
        raise FileNotFoundError(
            "No se encontraron graph JSONs en: " + GRAPH_JSON_DIR)

    if not SHOW_PLOT:
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
