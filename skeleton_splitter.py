"""
Skeleton Splitter — Corte interactivo con polilínea
====================================================
Divide una imagen de esqueleto en dos partes trazando una línea
de corte de múltiples puntos (polilínea libre).

Flujo:
  1. Pon la ruta de la imagen en SKELETON_PATH
  2. Haz clic en la imagen izquierda para agregar puntos al corte
     (mínimo 2 puntos; puedes agregar todos los que necesites)
  3. Revisa el preview: azul = parte A, naranja = parte B
  4. [Deshacer último] → quita el último punto
  5. [Limpiar corte]   → borra todos los puntos
  6. [Guardar A y B]   → guarda imagen_A.png e imagen_B.png
                         y elimina el original
"""

# =====================================================================
#  CONFIGURACIÓN — MODIFICA AQUÍ
# =====================================================================
SKELETON_PATH = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados\imgs_frame435_00000.png"
OUTPUT_FOLDER = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados"

# =====================================================================
#  NO MODIFIQUES DEBAJO
# =====================================================================
import sys
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from pathlib import Path

if len(sys.argv) >= 2:
    SKELETON_PATH = sys.argv[1]
if len(sys.argv) >= 3:
    OUTPUT_FOLDER = sys.argv[2]

skeleton_path = Path(SKELETON_PATH)
output_path   = Path(OUTPUT_FOLDER)

if not skeleton_path.exists():
    print(f"[ERROR] Imagen no encontrada: {SKELETON_PATH}")
    sys.exit(1)

output_path.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"[ERROR] No se pudo leer la imagen: {SKELETON_PATH}")
    sys.exit(1)

print(f"[INFO] Imagen cargada: {skeleton_path.name}  ({img.shape[1]}×{img.shape[0]} px)")

state = {'cut_pts': []}   # lista de puntos (x, y) de la polilínea


# ---------------------------------------------------------------------------
# Cálculo de máscaras con polilínea
# Para cada píxel: busca el segmento más cercano de la polilínea
# y calcula el producto cruzado → determina de qué lado está.
# ---------------------------------------------------------------------------
def compute_masks_polyline(img_shape, pts):
    h, w = img_shape[:2]
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)

    min_dist_sq = np.full((h, w), np.inf, dtype=np.float32)
    side        = np.zeros((h, w), dtype=np.float32)

    for i in range(len(pts) - 1):
        x1, y1 = float(pts[i][0]),   float(pts[i][1])
        x2, y2 = float(pts[i+1][0]), float(pts[i+1][1])
        dx, dy  = x2 - x1, y2 - y1
        seg_len_sq = dx*dx + dy*dy
        if seg_len_sq < 1e-6:
            continue

        # Proyección de cada píxel sobre el segmento (t clamped [0,1])
        t  = ((cols - x1) * dx + (rows - y1) * dy) / seg_len_sq
        t  = np.clip(t, 0.0, 1.0)
        px = x1 + t * dx
        py = y1 + t * dy

        dist_sq = (cols - px)**2 + (rows - py)**2
        cross   = dx * (rows - y1) - dy * (cols - x1)   # > 0 = lado A

        closer      = dist_sq < min_dist_sq
        min_dist_sq = np.where(closer, dist_sq, min_dist_sq)
        side        = np.where(closer, cross,    side)

    return side >= 0, side < 0


# ---------------------------------------------------------------------------
# Preview en color
# ---------------------------------------------------------------------------
def render_preview(pts):
    canvas = np.full((*img.shape[:2], 3), 0.08, dtype=np.float32)
    skel   = img > 0
    if len(pts) < 2:
        canvas[skel] = [0.9, 0.9, 0.9]
        return canvas
    mask_A, mask_B = compute_masks_polyline(img.shape, pts)
    canvas[skel & mask_A] = [0.27, 0.53, 1.0]   # azul
    canvas[skel & mask_B] = [1.0,  0.53, 0.20]  # naranja
    return canvas


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#1a1a1a')

gs = gridspec.GridSpec(1, 2, figure=fig,
                       top=0.91, bottom=0.15,
                       left=0.02, right=0.98, wspace=0.04)

ax_orig    = fig.add_subplot(gs[0, 0])
ax_preview = fig.add_subplot(gs[0, 1])

for ax in [ax_orig, ax_preview]:
    ax.set_facecolor('#1e1e1e')
    ax.axis('off')

ax_orig.set_title('Original  (haz clic para agregar puntos al corte)',
                  color='#88ccff', fontsize=9, pad=4)
ax_preview.set_title('Preview  (Azul = parte A  |  Naranja = parte B)',
                     color='#ffaa44', fontsize=9, pad=4)

fig.suptitle(skeleton_path.name, color='white', fontsize=11, fontweight='bold')

im_orig    = ax_orig.imshow(img, cmap='gray', vmin=0, vmax=255)
im_preview = ax_preview.imshow(render_preview([]), vmin=0, vmax=1)

hint_text = ax_preview.text(
    0.5, -0.02, 'Haz clic en la imagen izquierda para trazar el corte',
    transform=ax_preview.transAxes,
    color='#aaaaaa', fontsize=8, ha='center', va='top'
)

cut_line = None   # objeto Line2D de la polilínea
cut_dots = []     # marcadores de los puntos


def _refresh():
    global cut_line, cut_dots

    pts = state['cut_pts']

    # Borrar polilínea y puntos anteriores
    if cut_line is not None:
        cut_line.remove()
        cut_line = None
    for d in cut_dots:
        d.remove()
    cut_dots.clear()

    # Dibujar polilínea
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ln, = ax_orig.plot(xs, ys, '-', color='#ffff00',
                           linewidth=1.5, alpha=0.85, zorder=9)
        cut_line = ln

    # Dibujar puntos individuales
    for p in pts:
        d, = ax_orig.plot(p[0], p[1], 'o', color='#ffff00',
                          markersize=6, zorder=10)
        cut_dots.append(d)

    # Actualizar preview
    im_preview.set_data(render_preview(pts))

    # Hint
    n = len(pts)
    if n == 0:
        hint_text.set_text('Haz clic en la imagen izquierda para trazar el corte')
    elif n == 1:
        hint_text.set_text('Agrega más puntos para definir la dirección del corte')
    else:
        hint_text.set_text(f'{n} puntos definidos — agrega más o guarda cuando estés conforme')

    fig.canvas.draw()
    fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Clic para agregar puntos
# ---------------------------------------------------------------------------
def on_click(event):
    if event.inaxes != ax_orig or event.button != 1:
        return
    if event.xdata is None or event.ydata is None:
        return
    x = int(round(event.xdata))
    y = int(round(event.ydata))
    state['cut_pts'].append((x, y))
    _refresh()


fig.canvas.mpl_connect('button_press_event', on_click)


# ---------------------------------------------------------------------------
# Botones
# ---------------------------------------------------------------------------
def _make_btn(left, width, label, bg, hover):
    ax_b = fig.add_axes([left, 0.04, width, 0.07], facecolor=bg)
    b = Button(ax_b, label, color=bg, hovercolor=hover)
    b.label.set_color('white')
    b.label.set_fontsize(8.5)
    return b


btn_undo  = _make_btn(0.10, 0.14, 'Deshacer último', '#4a3a1a', '#7a5a2a')
btn_clear = _make_btn(0.26, 0.12, 'Limpiar corte',   '#5a3a1a', '#8a5a2a')
btn_save  = _make_btn(0.55, 0.14, 'Guardar A y B',   '#1a6a1a', '#2a9a2a')


def on_undo(_):
    if state['cut_pts']:
        state['cut_pts'].pop()
    _refresh()


def on_clear(_):
    state['cut_pts'] = []
    _refresh()


def on_save(_):
    pts = state['cut_pts']
    if len(pts) < 2:
        print("[WARN] Define primero al menos 2 puntos en el corte")
        return

    mask_A, mask_B = compute_masks_polyline(img.shape, pts)

    img_A = img.copy(); img_A[~mask_A] = 0
    img_B = img.copy(); img_B[~mask_B] = 0

    stem  = skeleton_path.stem
    suf   = skeleton_path.suffix
    out_A = output_path / (stem + '_A' + suf)
    out_B = output_path / (stem + '_B' + suf)

    cv2.imwrite(str(out_A), img_A)
    cv2.imwrite(str(out_B), img_B)
    print(f"[OK] Guardado: {out_A.name}  ({int(np.sum(img_A > 0))} px activos)")
    print(f"[OK] Guardado: {out_B.name}  ({int(np.sum(img_B > 0))} px activos)")

    os.remove(str(skeleton_path))
    print(f"[OK] Original eliminado: {skeleton_path.name}")

    # Registrar en JSON de Mascaras filtradas
    log_path = Path(r"C:\Users\vgara\OneDrive\Desktop\IPre\Mascaras filtradas") / "imagenes_divididas.json"
    registro = {}
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            registro = json.load(f)
    registro[skeleton_path.name] = {
        'parte_A': out_A.name,
        'parte_B': out_B.name,
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print(f"[OK] Registrado en: {log_path.name}")

    plt.close(fig)


btn_undo.on_clicked(on_undo)
btn_clear.on_clicked(on_clear)
btn_save.on_clicked(on_save)

# ---------------------------------------------------------------------------
# Leyenda
# ---------------------------------------------------------------------------
ax_leg = fig.add_axes([0.03, 0.02, 0.05, 0.10], facecolor='#1a1a1a')
ax_leg.axis('off')
ax_leg.text(0.05, 0.95,
            "Clic → agrega punto\n"
            "[Deshacer] → quita último\n"
            "[Limpiar]  → reinicia",
            transform=ax_leg.transAxes,
            color='#888888', fontsize=7.5, va='top', fontfamily='monospace')

plt.show()
