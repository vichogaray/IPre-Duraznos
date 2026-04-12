"""
Param Tuner — Ajuste de parámetros sobre máscaras filtradas
============================================================
- Input : carpeta de máscaras filtradas (Mascaras filtradas).
- Output: carpeta de esqueletos filtrados (Esqueletos filtrados).
- Navega con ◀ / ▶ entre imágenes.
- Ajusta los sliders hasta que el esqueleto se vea bien.
- Presiona [Guardar + Procesar] para guardar el esqueleto
  y guardar los parámetros en 'parametros.json'.
- Al navegar a una imagen ya guardada, los sliders se cargan automáticamente
  con sus parámetros guardados.
- El indicador en el título muestra: [✓ guardada] o [pendiente].
"""

FOLDER_PATH   = r"C:\Users\vgara\OneDrive\Desktop\IPre\Mascaras filtradas"
OUTPUT_FOLDER = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados"

# Archivo JSON donde se guardan los parámetros
PARAMS_FILE = r"C:\Users\vgara\OneDrive\Desktop\IPre\Mascaras filtradas\parametros.json"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =====================================================================

import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from skimage.morphology import skeletonize
from pathlib import Path

# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _count_neighbors(skel, r, c):
    rows, cols = skel.shape
    return sum(
        1 for dr, dc in NEIGHBORS_8
        if 0 <= r+dr < rows and 0 <= c+dc < cols and skel[r+dr, c+dc]
    )

def prune_skeleton(skel_bin, min_px):
    skel = (skel_bin > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        rows, cols = skel.shape
        endpoints = [
            (r, c)
            for r in range(rows) for c in range(cols)
            if skel[r, c] and _count_neighbors(skel, r, c) == 1
        ]
        for (r0, c0) in endpoints:
            if not skel[r0, c0]:
                continue
            path = [(r0, c0)]
            prev, cur = None, (r0, c0)
            while True:
                nbs = [
                    (r, c) for dr, dc in NEIGHBORS_8
                    for r, c in [(cur[0]+dr, cur[1]+dc)]
                    if 0 <= r < rows and 0 <= c < cols
                    and skel[r, c] and (r, c) != prev
                ]
                if not nbs:
                    break
                if _count_neighbors(skel, cur[0], cur[1]) > 2:
                    break
                if len(path) >= min_px:
                    break
                prev, cur = cur, nbs[0]
                path.append(cur)
            end_nbs = _count_neighbors(skel, cur[0], cur[1])
            if len(path) < min_px and end_nbs != 1:
                for (r, c) in path:
                    skel[r, c] = 0
                changed = True
    return skel


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process(img_gray, close_k, open_k, thick_dist, dilate_k, min_branch):
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    close_k = max(1, int(close_k)) | 1
    open_k  = max(1, int(open_k))  | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    smoothed = cv2.morphologyEx(binary,   cv2.MORPH_CLOSE, k_close)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,  k_open)

    dist = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)
    thin_mask     = ((smoothed > 0) & (dist <= thick_dist)).astype(np.uint8) * 255
    thick_centers = (dist > thick_dist).astype(np.uint8) * 255

    dilate_k = max(1, int(dilate_k)) | 1
    k_dil    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    thick_region = cv2.dilate(thick_centers, k_dil)
    valid_mask   = cv2.bitwise_or(thin_mask, thick_region)
    valid_mask   = cv2.bitwise_and(valid_mask, smoothed)

    skel = skeletonize(valid_mask > 0).astype(np.uint8)
    skel = prune_skeleton(skel, max(1, int(min_branch)))

    return (skel * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Persistencia de parámetros (JSON)
# ---------------------------------------------------------------------------
params_path = Path(PARAMS_FILE)

def load_params_db():
    if params_path.exists():
        with open(params_path, 'r') as f:
            return json.load(f)
    return {}

def save_params_db(db):
    with open(params_path, 'w') as f:
        json.dump(db, f, indent=2)

params_db = load_params_db()
print(f"[INFO] Parámetros guardados: {len(params_db)} imágenes en {params_path.name}")


# ---------------------------------------------------------------------------
# Cargar lista de imágenes
# ---------------------------------------------------------------------------
if len(sys.argv) >= 2:
    FOLDER_PATH = sys.argv[1]
if len(sys.argv) >= 3:
    OUTPUT_FOLDER = sys.argv[2]

folder      = Path(FOLDER_PATH)
output_path = Path(OUTPUT_FOLDER)

if not folder.exists():
    print(f"[ERROR] No se encontró la carpeta: {FOLDER_PATH}")
    sys.exit(1)

output_path.mkdir(parents=True, exist_ok=True)

image_files = sorted([
    f for f in folder.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
])

if not image_files:
    print(f"[ERROR] No hay imágenes en: {FOLDER_PATH}")
    sys.exit(1)

print(f"[INFO] {len(image_files)} imágenes en {folder.name}")

state = {'idx': 0}

def load_image(idx):
    p = image_files[idx]
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"No se pudo leer: {p}")
    return img, p.name


# ---------------------------------------------------------------------------
# Valores por defecto
# ---------------------------------------------------------------------------
DEFAULTS = dict(close_k=1, open_k=1, thick_dist=1.0, dilate_k=1, min_branch=5)

def params_for(name):
    """Devuelve parámetros guardados para 'name', o los defaults."""
    if name in params_db:
        return params_db[name]
    return dict(DEFAULTS)

img_gray, img_name = load_image(0)
p0   = params_for(img_name)
skel0 = process(img_gray, **p0)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#1e1e1e')

gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    top=0.93, bottom=0.28,
    hspace=0.08, wspace=0.04
)

ax_orig = fig.add_subplot(gs[0, 0])
ax_skel = fig.add_subplot(gs[0, 1])

for ax in [ax_orig, ax_skel]:
    ax.axis('off')
    ax.set_facecolor('#1e1e1e')

title_kw = dict(color='white', fontsize=9, pad=3)
ax_orig.set_title('Máscara original', **title_kw)
ax_skel.set_title('Esqueleto (resultado)', **title_kw)

suptitle = fig.suptitle('', color='white', fontsize=11, fontweight='bold')

def set_title():
    idx  = state['idx']
    name = image_files[idx].name
    tag  = '  [✓ guardada]' if name in params_db else '  [pendiente]'
    suptitle.set_text(f"[{idx+1}/{len(image_files)}]  {name}{tag}")
    suptitle.set_color('#88ff88' if name in params_db else 'white')

set_title()

im_orig = ax_orig.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
im_skel = ax_skel.imshow(skel0,   cmap='gray', vmin=0, vmax=255)

px_text = ax_skel.text(
    0.01, 0.02, f"Píxeles esqueleto: {np.sum(skel0 > 0)}",
    transform=ax_skel.transAxes, color='yellow', fontsize=9,
    verticalalignment='bottom'
)

saved_text = ax_orig.text(
    0.01, 0.02, '',
    transform=ax_orig.transAxes, color='#88ff88', fontsize=9,
    verticalalignment='bottom'
)

# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------
slider_color  = '#3a3a3a'
slider_active = '#5a9fd4'

def make_slider(left, bottom, label, valmin, valmax, valinit, valstep):
    ax_s = fig.add_axes([left, bottom, 0.20, 0.025], facecolor=slider_color)
    s = Slider(ax_s, label, valmin, valmax, valinit=valinit,
               valstep=valstep, color=slider_active)
    s.label.set_color('white')
    s.valtext.set_color('white')
    ax_s.tick_params(colors='white')
    return s

# Fila superior: Cierre, Grosor, Pruning
s_close  = make_slider(0.10, 0.17, 'Cierre',   1,   25,   p0['close_k'],    2)
s_thick  = make_slider(0.43, 0.17, 'Grosor',   1.0, 20.0, p0['thick_dist'], 0.5)
s_prune  = make_slider(0.76, 0.17, 'Pruning',  5,   200,  p0['min_branch'], 5)
# Fila inferior: Apertura, Dilat.
s_open   = make_slider(0.26, 0.11, 'Apertura', 1,   15,   p0['open_k'],     1)
s_dilate = make_slider(0.60, 0.11, 'Dilat.',   1,   31,   p0['dilate_k'],   2)


# ---------------------------------------------------------------------------
# Botones
# ---------------------------------------------------------------------------
ax_prev = fig.add_axes([0.04, 0.03, 0.10, 0.05], facecolor='#3a3a3a')
ax_next = fig.add_axes([0.16, 0.03, 0.10, 0.05], facecolor='#3a3a3a')
ax_save = fig.add_axes([0.40, 0.03, 0.22, 0.05], facecolor='#2a7a2a')

btn_prev = Button(ax_prev, '◀  Anterior',    color='#3a3a3a', hovercolor='#555555')
btn_next = Button(ax_next, 'Siguiente  ▶',   color='#3a3a3a', hovercolor='#555555')
btn_save = Button(ax_save, 'Guardar + Procesar', color='#2a7a2a', hovercolor='#3a9a3a')

for b in [btn_prev, btn_next, btn_save]:
    b.label.set_color('white')
    b.label.set_fontsize(9)


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
def current_params():
    return dict(
        close_k    = int(s_close.val),
        open_k     = int(s_open.val),
        thick_dist = float(s_thick.val),
        dilate_k   = int(s_dilate.val),
        min_branch = int(s_prune.val),
    )

def set_sliders(p):
    """Actualiza los sliders sin disparar callbacks."""
    s_close.eventson = False
    s_open.eventson  = False
    s_thick.eventson = False
    s_dilate.eventson = False
    s_prune.eventson  = False

    s_close.set_val(p['close_k'])
    s_open.set_val(p['open_k'])
    s_thick.set_val(p['thick_dist'])
    s_dilate.set_val(p['dilate_k'])
    s_prune.set_val(p['min_branch'])

    s_close.eventson  = True
    s_open.eventson   = True
    s_thick.eventson  = True
    s_dilate.eventson = True
    s_prune.eventson  = True

def redraw(new_img=None):
    global img_gray
    if new_img is not None:
        img_gray = new_img

    p  = current_params()
    sk = process(img_gray, **p)

    im_orig.set_data(img_gray)
    im_skel.set_data(sk)

    h, w = img_gray.shape
    for ax in [ax_orig, ax_skel]:
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

    px_text.set_text(f"Píxeles esqueleto: {np.sum(sk > 0)}")
    set_title()
    fig.canvas.draw()
    fig.canvas.flush_events()

def update(_=None):
    redraw()

def navigate(delta):
    idx = (state['idx'] + delta) % len(image_files)
    state['idx'] = idx
    new_img, name = load_image(idx)
    # Cargar parámetros guardados (o defaults) para esta imagen
    set_sliders(params_for(name))
    saved_text.set_text('')
    redraw(new_img)

def on_prev(_): navigate(-1)
def on_next(_): navigate(+1)

def on_save(_):
    name = image_files[state['idx']].name
    p    = current_params()

    # Guardar parámetros en JSON
    params_db[name] = p
    save_params_db(params_db)

    # Procesar y guardar esqueleto
    sk = process(img_gray, **p)
    out_file = output_path / name
    cv2.imwrite(str(out_file), sk)

    saved_text.set_text(f'Sobreescrito → {out_file.name}')
    set_title()
    print(f"[OK] {name}  →  sobreescrito  ({np.sum(sk>0)} px)")
    fig.canvas.draw()
    fig.canvas.flush_events()


s_close.on_changed(update)
s_open.on_changed(update)
s_thick.on_changed(update)
s_dilate.on_changed(update)
s_prune.on_changed(update)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_save.on_clicked(on_save)

plt.show()
