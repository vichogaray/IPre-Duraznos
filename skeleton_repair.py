"""
Skeleton Repair — Reparación interactiva de gaps de 1 píxel
============================================================
Lee un esqueleto, detecta endpoints donde falta exactamente 1 píxel
para reconectarse con otra parte del esqueleto, y permite aceptar o
rechazar cada reparación individualmente.

Panel izquierdo : vista global del esqueleto
  · Rojo    = endpoint actual bajo revisión
  · Amarillo = otros endpoints reparables pendientes
  · Verde    = reparaciones ya aceptadas

Panel derecho : zoom alrededor del endpoint
  · Gris   = esqueleto existente
  · Rojo   = endpoint (punto de partida del gap)
  · Verde  = píxel propuesto a agregar
  · Azul   = píxel destino (donde se reconecta)

Botones:
  [◀ Anterior]  [Aceptar]  [Rechazar]  [Siguiente ▶]  [Guardar esqueleto]

OUTPUT: mismo nombre + '_reparado.png' en la misma carpeta
"""

# =====================================================================
#  CONFIGURACIÓN
# =====================================================================
BATCH_MODE    = True   # True = procesa toda la lista sin interfaz gráfica
INPUT_LIST    = r"C:\Users\vgara\OneDrive\Desktop\IPre\esqueletos_sin_grafo.txt"
SKELETON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados"
OUTPUT_DIR    = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos sin grafo"

# Solo para modo interactivo (BATCH_MODE = False):
SKELETON_PATH = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados\imgs_frame1_00000.png"
ZOOM_RADIUS   = 30    # píxeles alrededor del endpoint en el panel zoom
LOCAL_STEPS   = 8     # pasos BFS para definir la rama local del endpoint

# =====================================================================
#  NO MODIFIQUES DEBAJO
# =====================================================================
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from pathlib import Path

if not BATCH_MODE:
    if len(sys.argv) >= 2:
        SKELETON_PATH = sys.argv[1]

    skeleton_path = Path(SKELETON_PATH)
    if not skeleton_path.exists():
        print(f"[ERROR] Imagen no encontrada: {SKELETON_PATH}")
        sys.exit(1)

    raw = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"[ERROR] No se pudo leer: {SKELETON_PATH}")
        sys.exit(1)

    _, skel_orig = cv2.threshold(raw, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(skel_orig) > 0.5:
        skel_orig = 1 - skel_orig
    skel_orig = skel_orig.astype(np.uint8)

    print(f"[INFO] Imagen: {skeleton_path.name}  ({raw.shape[1]}x{raw.shape[0]} px)")

# -----------------------------------------------------------------------
# Helpers de conectividad
# -----------------------------------------------------------------------
NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def get_neighbors_8(skel, r, c):
    h, w = skel.shape
    return [(r+dr, c+dc) for dr, dc in NEIGHBORS_8
            if 0 <= r+dr < h and 0 <= c+dc < w and skel[r+dr, c+dc]]


def find_endpoints(skel):
    eps = []
    ys, xs = np.where(skel > 0)
    for r, c in zip(ys, xs):
        if len(get_neighbors_8(skel, r, c)) == 1:
            eps.append((r, c))
    return eps


def local_branch(skel, ep, steps):
    """BFS desde ep, devuelve todos los píxeles a ≤ steps pasos."""
    visited = {ep}
    frontier = [ep]
    for _ in range(steps):
        nf = []
        for p in frontier:
            for nb in get_neighbors_8(skel, p[0], p[1]):
                if nb not in visited:
                    visited.add(nb)
                    nf.append(nb)
        frontier = nf
    return visited


# -----------------------------------------------------------------------
# Algoritmo de detección de gaps de 1 píxel
# -----------------------------------------------------------------------
def find_repairs(skel, endpoints, local_steps=LOCAL_STEPS):
    """
    Para cada endpoint E:
      1. Calcula la dirección de la rama (desde su único vecino N → E)
      2. Busca píxeles vacíos C adyacentes a E que tengan vecino esqueleto S
         donde S no pertenezca a la rama local de E
      3. Elige el candidato más alineado con la dirección de la rama

    Devuelve: {endpoint: (pixel_a_agregar, pixel_destino)}
    """
    h, w = skel.shape
    repairs = {}

    for ep in endpoints:
        er, ec = ep
        nbs = get_neighbors_8(skel, er, ec)
        if len(nbs) != 1:
            continue  # solo procesar endpoints puros

        nr, nc = nbs[0]
        # Dirección extrapolada: desde vecino hacia endpoint (y más allá)
        dr_dir = er - nr
        dc_dir = ec - nc

        loc = local_branch(skel, ep, local_steps)

        # Recolectar todos los candidatos de reparación
        candidates = []
        for dr1, dc1 in NEIGHBORS_8:
            cr, cc = er + dr1, ec + dc1
            if not (0 <= cr < h and 0 <= cc < w):
                continue
            if skel[cr, cc]:
                continue  # C debe estar vacío

            for dr2, dc2 in NEIGHBORS_8:
                sr, sc = cr + dr2, cc + dc2
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                if skel[sr, sc] and (sr, sc) not in loc:
                    # Alineación: producto punto con dirección de rama
                    alignment = dr1 * dr_dir + dc1 * dc_dir
                    candidates.append((alignment, (cr, cc), (sr, sc)))
                    break  # un destino por píxel candidato es suficiente

        if candidates:
            # Elegir el más alineado con la dirección actual de la rama
            candidates.sort(reverse=True, key=lambda x: x[0])
            _, bridge_px, dest_px = candidates[0]
            repairs[ep] = (bridge_px, dest_px)

    return repairs


# -----------------------------------------------------------------------
# MODO BATCH — acepta todas las reparaciones automáticamente
# -----------------------------------------------------------------------
if BATCH_MODE:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(INPUT_LIST, encoding='utf-8') as _f:
        _files = [l.strip() for l in _f if l.strip()]

    print(f"[BATCH] {len(_files)} imagenes a procesar")
    print(f"[BATCH] Destino: {out_dir}\n")

    _total = 0
    for _i, _fname in enumerate(_files, 1):
        _img_path = Path(SKELETON_DIR) / _fname
        if not _img_path.exists():
            print(f"  [{_i:3d}/{len(_files)}] FALTA   : {_fname}")
            continue

        _raw = cv2.imread(str(_img_path), cv2.IMREAD_GRAYSCALE)
        if _raw is None:
            print(f"  [{_i:3d}/{len(_files)}] ERROR   : {_fname}")
            continue

        _, _skel = cv2.threshold(_raw, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(_skel) > 0.5:
            _skel = 1 - _skel
        _skel = _skel.astype(np.uint8)

        _eps     = find_endpoints(_skel)
        _repairs = find_repairs(_skel, _eps, LOCAL_STEPS)

        for (_bridge, _) in _repairs.values():
            _skel[_bridge[0], _bridge[1]] = 1

        cv2.imwrite(str(out_dir / _fname), (_skel * 255).astype(np.uint8))
        _total += len(_repairs)
        print(f"  [{_i:3d}/{len(_files)}] {_fname}  ->  {len(_repairs)} reparaciones")

    print(f"\n[BATCH] Listo. {_total} pixeles reparados en {len(_files)} imagenes.")
    sys.exit(0)


# -----------------------------------------------------------------------
# Detección inicial
# -----------------------------------------------------------------------
endpoints_all = find_endpoints(skel_orig)
repairs_all   = find_repairs(skel_orig, endpoints_all)
repair_eps    = sorted(repairs_all.keys())  # lista ordenada de endpoints reparables

print(f"[INFO] Endpoints totales    : {len(endpoints_all)}")
print(f"[INFO] Endpoints reparables : {len(repair_eps)}")

if not repair_eps:
    print("[INFO] No se encontraron gaps de 1 píxel. El esqueleto parece completo.")
    sys.exit(0)

# Estado mutable
state = {
    'idx':      0,               # índice actual en repair_eps
    'accepted': set(),           # índices aceptados
    'rejected': set(),           # índices rechazados
    'skel':     skel_orig.copy(),
}


# -----------------------------------------------------------------------
# Render helpers
# -----------------------------------------------------------------------
def render_full(skel):
    h, w = skel.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    canvas[skel > 0] = [0.55, 0.55, 0.55]

    idx     = state['idx']
    current = repair_eps[idx] if idx < len(repair_eps) else None

    # Reparaciones aceptadas → verde
    for i in state['accepted']:
        ep = repair_eps[i]
        (br, bc), _ = repairs_all[ep]
        canvas[br, bc] = [0.0, 0.9, 0.3]

    # Endpoints pendientes → amarillo
    for i, ep in enumerate(repair_eps):
        if i not in state['accepted'] and i not in state['rejected'] and i != idx:
            canvas[ep[0], ep[1]] = [1.0, 0.85, 0.0]

    # Endpoint actual → rojo
    if current:
        canvas[current[0], current[1]] = [1.0, 0.18, 0.18]

    return canvas


def render_zoom(ep):
    skel  = state['skel']
    h, w  = skel.shape
    er, ec = ep
    r0 = max(0, er - ZOOM_RADIUS); r1 = min(h, er + ZOOM_RADIUS + 1)
    c0 = max(0, ec - ZOOM_RADIUS); c1 = min(w, ec + ZOOM_RADIUS + 1)

    crop   = skel[r0:r1, c0:c1]
    canvas = np.zeros((r1-r0, c1-c0, 3), dtype=np.float32)
    canvas[crop > 0] = [0.65, 0.65, 0.65]

    # Endpoint: rojo
    canvas[er-r0, ec-c0] = [1.0, 0.18, 0.18]

    (br, bc), (dr, dc) = repairs_all[ep]

    # Píxel propuesto: verde
    if r0 <= br < r1 and c0 <= bc < c1:
        canvas[br-r0, bc-c0] = [0.0, 1.0, 0.35]

    # Píxel destino: azul
    if r0 <= dr < r1 and c0 <= dc < c1:
        canvas[dr-r0, dc-c0] = [0.2, 0.55, 1.0]

    return canvas


# -----------------------------------------------------------------------
# Figura
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(17, 9))
fig.patch.set_facecolor('#1a1a1a')

gs = gridspec.GridSpec(1, 2, figure=fig,
                       top=0.90, bottom=0.15,
                       left=0.02, right=0.98, wspace=0.04)

ax_full = fig.add_subplot(gs[0, 0])
ax_zoom = fig.add_subplot(gs[0, 1])

for ax in [ax_full, ax_zoom]:
    ax.set_facecolor('#1e1e1e')
    ax.axis('off')

ax_full.set_title('Vista global  (rojo=actual · amarillo=pendiente · verde=aceptada)',
                  color='white', fontsize=8.5, pad=4)
ax_zoom.set_title('Zoom  (rojo=endpoint · verde=píxel propuesto · azul=destino)',
                  color='#88ccff', fontsize=8.5, pad=4)

n_total   = len(repair_eps)
suptitle  = fig.suptitle('', color='white', fontsize=11, fontweight='bold')
hint_text = ax_zoom.text(0.5, -0.03, '',
                         transform=ax_zoom.transAxes,
                         color='#aaaaaa', fontsize=8, ha='center', va='top')

im_full = ax_full.imshow(render_full(state['skel']), vmin=0, vmax=1,
                          interpolation='nearest')
im_zoom = ax_zoom.imshow(render_zoom(repair_eps[0]), vmin=0, vmax=1,
                          interpolation='nearest')


def _refresh():
    idx = state['idx']
    ep  = repair_eps[idx]
    (br, bc), (dr, dc) = repairs_all[ep]

    suptitle.set_text(
        f"{skeleton_path.name}   —   Endpoint {idx+1}/{n_total}  "
        f"·  aceptadas: {len(state['accepted'])}  "
        f"·  rechazadas: {len(state['rejected'])}"
    )

    status = ('✓ aceptada' if idx in state['accepted'] else
              '✗ rechazada' if idx in state['rejected'] else
              'pendiente')
    hint_text.set_text(
        f"Endpoint ({ep[1]},{ep[0]})  →  agregar ({bc},{br})  →  conecta a ({dc},{dr})   [{status}]"
    )

    im_full.set_data(render_full(state['skel']))
    im_zoom.set_data(render_zoom(ep))

    # Centrar la vista global en el endpoint actual
    h, w = state['skel'].shape
    margin = 120
    ax_full.set_xlim(max(0, ep[1]-margin), min(w, ep[1]+margin))
    ax_full.set_ylim(min(h, ep[0]+margin), max(0, ep[0]-margin))

    fig.canvas.draw()
    fig.canvas.flush_events()


_refresh()


# -----------------------------------------------------------------------
# Botones
# -----------------------------------------------------------------------
def _btn(left, label, bg, hover):
    ax_b = fig.add_axes([left, 0.04, 0.11, 0.07], facecolor=bg)
    b = Button(ax_b, label, color=bg, hovercolor=hover)
    b.label.set_color('white')
    b.label.set_fontsize(9)
    return b


btn_prev   = _btn(0.08, '◀ Anterior',       '#3a3a3a', '#5a5a5a')
btn_accept = _btn(0.21, 'Aceptar',           '#1a5a1a', '#2a8a2a')
btn_reject = _btn(0.34, 'Rechazar',          '#5a1a1a', '#8a2a2a')
btn_next   = _btn(0.47, 'Siguiente ▶',       '#3a3a3a', '#5a5a5a')
btn_save   = _btn(0.70, 'Guardar esqueleto', '#1a4a7a', '#2a6aaa')


def on_prev(_):
    state['idx'] = (state['idx'] - 1) % n_total
    _refresh()


def on_next(_):
    state['idx'] = (state['idx'] + 1) % n_total
    _refresh()


def on_accept(_):
    idx = state['idx']
    state['accepted'].add(idx)
    state['rejected'].discard(idx)
    # Aplicar la reparación al esqueleto activo
    ep = repair_eps[idx]
    (br, bc), _ = repairs_all[ep]
    state['skel'][br, bc] = 1
    print(f"[OK] Reparación aceptada: endpoint ({ep[1]},{ep[0]}) → píxel ({bc},{br}) agregado")
    on_next(None)


def on_reject(_):
    idx = state['idx']
    state['rejected'].add(idx)
    state['accepted'].discard(idx)
    # Deshacer reparación si fue aceptada antes
    ep = repair_eps[idx]
    (br, bc), _ = repairs_all[ep]
    state['skel'][br, bc] = 0
    print(f"[--] Reparación rechazada: endpoint ({ep[1]},{ep[0]})")
    on_next(None)


def on_save(_):
    out = skeleton_path.parent / (skeleton_path.stem + '_reparado' + skeleton_path.suffix)
    result = (state['skel'] * 255).astype(np.uint8)
    cv2.imwrite(str(out), result)
    n_acc = len(state['accepted'])
    print(f"[OK] Guardado: {out.name}  ({n_acc} píxeles reparados)")


btn_prev.on_clicked(on_prev)
btn_accept.on_clicked(on_accept)
btn_reject.on_clicked(on_reject)
btn_next.on_clicked(on_next)
btn_save.on_clicked(on_save)

# -----------------------------------------------------------------------
# Leyenda
# -----------------------------------------------------------------------
ax_leg = fig.add_axes([0.84, 0.02, 0.15, 0.10], facecolor='#1a1a1a')
ax_leg.axis('off')
ax_leg.text(0.05, 0.95,
            "◉ Rojo    = endpoint actual\n"
            "◉ Verde   = píxel a agregar\n"
            "◉ Azul    = destino conexión\n"
            "◉ Amarillo= otros pendientes\n"
            "◉ Verde   = ya aceptadas",
            transform=ax_leg.transAxes,
            color='#888888', fontsize=7.5, va='top', fontfamily='monospace')

plt.show()
