"""
Skeletonize Batch — Esqueletización masiva de máscaras binarias
===============================================================
Pipeline por imagen:
  1. Binarización (Otsu)
  2. Suavizado morfológico: cierre → apertura
  3. Filtro por grosor via transformada de distancia
  4. Dilatar centros gruesos para reconstruir la región válida
  5. Esqueletización (skimage)
  6. Pruning: eliminar ramas menores a min_branch píxeles

Prioridad de parámetros por imagen (de mayor a menor):
  1. MANUAL    — si la imagen está en parametros.json (calibrada con el tuner),
                 se usan EXACTAMENTE esos valores. Es ground truth: no se toca.
  2. APRENDIDO — si existe defaults_aprendidos.json, se usan los valores de
                 consenso (moda) aprendidos de tus calibraciones manuales.
                 Genera estos defaults con: python aprender_parametros.py
  3. FIJO      — los valores definidos abajo en el bloque CONFIG (fallback
                 final si no hay defaults aprendidos).

El antiguo modo ADAPTIVE (estimar grosor/dilatación con Otsu sobre la
transformada de distancia) quedó deshabilitado por defecto: se comprobó que
los defaults aprendidos de tu ground truth dan mejores resultados que el
estimado automático. Se puede reactivar poniendo ADAPTIVE = True.
"""

# Rutas del proyecto centralizadas en src/common/paths.py. Se anade la raiz
# del repositorio al path para poder ejecutar este archivo directamente (F5).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
from src.common.paths import MASKS_DIR, SKELETONS_DIR


# =====================================================================
#  CONFIGURACIÓN — MODIFICA AQUÍ
# =====================================================================

INPUT_FOLDER  = MASKS_DIR
OUTPUT_FOLDER = SKELETONS_DIR

# --- Imagen única / ajuste manual --------------------------------------
# SINGLE_IMAGE = None  -> procesa TODA la carpeta en batch (automático).
# SINGLE_IMAGE = "imgs_frameXX_00000.png" -> procesa SOLO esa máscara.
#
# MANUAL_TUNE solo aplica cuando SINGLE_IMAGE está definido:
#   False -> esqueletiza automático con los parámetros aprendidos para esa foto.
#   True  -> abre la interfaz de tuneo del suavizado (precargada con los
#            parámetros predichos). Al pulsar "Guardar + Esqueletizar", el
#            proceso continúa: esqueletiza con esos parámetros y guarda el PNG.
SINGLE_IMAGE = None
MANUAL_TUNE  = False

# --- Modo automático ---------------------------------------------------
# Por defecto se usan los DEFAULTS APRENDIDOS del ground truth manual
# (defaults_aprendidos.json). Pon ADAPTIVE = True para volver al antiguo
# estimador de grosor por Otsu sobre la transformada de distancia.
ADAPTIVE = False

# El calibrador aprendido (predice open_k/min_branch desde la imagen y usa
# consenso para el resto) está INCRUSTADO más abajo (CONSENSO + LINMODEL).
# No depende de ningún archivo externo.

# Parámetros fijos (fallback solo si se desactiva el calibrador incrustado)
CLOSE_KSIZE        = 7      # Kernel del cierre morfológico
OPEN_KSIZE         = 4      # Kernel de la apertura morfológica
THICKNESS_MAX_DIST = 5.5    # Umbral grosor
THICK_DILATE_KSIZE = 11     # Dilatación centros gruesos
MIN_BRANCH_PX      = 70     # Largo mínimo de rama en pruning

# --- Reparación automática de gaps -------------------------------------
# Tras esqueletizar cada imagen, reconecta endpoints separados por
# exactamente 1 píxel (gaps de 1px). Se aplica antes de guardar.
REPAIR_GAPS         = True   # True = reparar gaps de 1px automáticamente
REPAIR_LOCAL_STEPS  = 8      # Pasos BFS para definir la rama local de un endpoint

# Extensiones de imagen a procesar
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =====================================================================
#  NO NECESITAS MODIFICAR NADA DEBAJO DE ESTA LÍNEA
# =====================================================================

import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Estimación adaptativa de parámetros de grosor
# ---------------------------------------------------------------------------
def estimate_thickness_params(dist: np.ndarray):
    """
    Estima THICKNESS_MAX_DIST y THICK_DILATE_KSIZE automáticamente
    a partir de la transformada de distancia de la imagen suavizada.

    Estrategia:
      - Extrae los valores de distancia de los píxeles activos.
      - Aplica umbral de Otsu sobre el histograma de distancias para
        separar la zona "fina" (ramas) de la zona "gruesa" (nodos/tronco).
      - DILATE_KSIZE = 2 × umbral, redondeado al entero impar más cercano.

    Si la máscara es muy delgada (sin zona gruesa), devuelve los valores
    globales de fallback.
    """
    vals = dist[dist > 0].astype(np.float32)
    if vals.size == 0:
        return THICKNESS_MAX_DIST, THICK_DILATE_KSIZE

    max_val = float(vals.max())
    if max_val < 2.0:
        # Máscara casi esquelética — no hay zona gruesa diferenciable
        return max_val * 0.6, 3

    # Normalizar a 0–255 para poder usar cv2.threshold con Otsu
    vals_norm = (vals / max_val * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(vals_norm, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convertir umbral Otsu de vuelta a unidades de píxeles
    threshold = float(otsu_val) / 255.0 * max_val

    # Acotar: mínimo 2px, máximo el 80% del valor máximo
    threshold = float(np.clip(threshold, 2.0, max_val * 0.8))

    # Tamaño de dilatación impar ≥ 3
    dilate_k = int(round(threshold * 2))
    dilate_k = max(3, dilate_k | 1)   # forzar impar

    return threshold, dilate_k


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

def prune_skeleton(skel_bin: np.ndarray, min_px: int) -> np.ndarray:
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
# Reparación de gaps de 1 píxel (integrado de skeleton_repair.py)
# ---------------------------------------------------------------------------
def _get_neighbors_8(skel, r, c):
    h, w = skel.shape
    return [(r+dr, c+dc) for dr, dc in NEIGHBORS_8
            if 0 <= r+dr < h and 0 <= c+dc < w and skel[r+dr, c+dc]]


def _find_endpoints(skel):
    eps = []
    ys, xs = np.where(skel > 0)
    for r, c in zip(ys, xs):
        if len(_get_neighbors_8(skel, r, c)) == 1:
            eps.append((r, c))
    return eps


def _local_branch(skel, ep, steps):
    """Conjunto de píxeles alcanzables desde un endpoint en 'steps' pasos BFS."""
    visited = {ep}
    frontier = [ep]
    for _ in range(steps):
        nf = []
        for p in frontier:
            for nb in _get_neighbors_8(skel, p[0], p[1]):
                if nb not in visited:
                    visited.add(nb)
                    nf.append(nb)
        frontier = nf
    return visited


def _find_repairs(skel, endpoints, local_steps):
    """
    Para cada endpoint, busca un píxel puente vacío que, al activarlo,
    lo reconecte con otra rama (gap de exactamente 1px). Elige el puente
    más alineado con la dirección local de la rama.
    """
    h, w = skel.shape
    repairs = {}
    for ep in endpoints:
        er, ec = ep
        nbs = _get_neighbors_8(skel, er, ec)
        if len(nbs) != 1:
            continue
        nr, nc = nbs[0]
        dr_dir = er - nr
        dc_dir = ec - nc
        loc = _local_branch(skel, ep, local_steps)
        candidates = []
        for dr1, dc1 in NEIGHBORS_8:
            cr, cc = er + dr1, ec + dc1
            if not (0 <= cr < h and 0 <= cc < w):
                continue
            if skel[cr, cc]:
                continue
            for dr2, dc2 in NEIGHBORS_8:
                sr, sc = cr + dr2, cc + dc2
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                if skel[sr, sc] and (sr, sc) not in loc:
                    alignment = dr1 * dr_dir + dc1 * dc_dir
                    candidates.append((alignment, (cr, cc), (sr, sc)))
                    break
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            _, bridge_px, dest_px = candidates[0]
            repairs[ep] = (bridge_px, dest_px)
    return repairs


def repair_skeleton(skel, local_steps=REPAIR_LOCAL_STEPS):
    """
    Reconecta gaps de 1px. Recibe skel binario (0/1), devuelve
    (skel_reparado, n_reparaciones).
    """
    eps     = _find_endpoints(skel)
    repairs = _find_repairs(skel, eps, local_steps)
    for (bridge, _) in repairs.values():
        skel[bridge[0], bridge[1]] = 1
    return skel, len(repairs)


# ---------------------------------------------------------------------------
# Pipeline con parámetros explícitos (usado para imágenes con params manuales)
# ---------------------------------------------------------------------------
def _run_pipeline(img_gray, close_k, open_k, thick_dist, dilate_k, min_branch):
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    ck = max(1, int(close_k)) | 1
    ok = max(1, int(open_k))  | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    smoothed = cv2.morphologyEx(binary,   cv2.MORPH_CLOSE, k_close)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,  k_open)

    dist = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)
    thin_mask     = ((smoothed > 0) & (dist <= thick_dist)).astype(np.uint8) * 255
    thick_centers = (dist > thick_dist).astype(np.uint8) * 255

    dk = max(3, int(dilate_k) | 1)
    k_dil        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
    thick_region = cv2.dilate(thick_centers, k_dil)
    valid_mask   = cv2.bitwise_or(thin_mask, thick_region)
    valid_mask   = cv2.bitwise_and(valid_mask, smoothed)

    skel = skeletonize(valid_mask > 0).astype(np.uint8)
    skel = prune_skeleton(skel, max(1, int(min_branch)))
    return (skel * 255).astype(np.uint8)


# ===========================================================================
#  CALIBRADOR APRENDIDO (incrustado — NO depende de archivos externos)
# ===========================================================================
# El modelo está aprendido de las 289 máscaras calibradas a mano. En vez de
# guardar un modelo en disco, sus coeficientes viven aquí como código:
#
#   - close_k / thick_dist / dilate_k -> CONSENSO (la moda del ground truth).
#       Se comprobó que NO son predecibles desde la imagen.
#   - open_k / min_branch -> REGRESIÓN LINEAL sobre 16 features de la máscara.
#       open_k mejora ~13% sobre usar el valor típico; min_branch algo menos.
#       (Se destiló de un RandomForest a una lineal sin perder casi nada, para
#        que el aprendizaje quepa como fórmula y no requiera scikit-learn.)
#
# Para REAPRENDER con nuevas calibraciones: vuelve a entrenar una regresión
# lineal sobre las features de FEATURE_NAMES y reemplaza CONSENSO / LINMODEL.
# ---------------------------------------------------------------------------

# Consenso (moda de las 289 calibraciones manuales)
CONSENSO = {'close_k': 7, 'thick_dist': 5.5, 'dilate_k': 11}

# Orden FIJO de las features (no reordenar sin reentrenar los coeficientes)
FEATURE_NAMES = [
    'area_frac', 'dist_mean', 'dist_std', 'dist_p90', 'dist_max',
    'frac_gruesa', 'frac_muy_fina', 'n_components', 'frac_en_mayor',
    'n_cc_pequenos', 'n_cc_medianos', 'perim_area_ratio',
    'bbox_h', 'bbox_w', 'bbox_fill', 'n_huecos',
]

# Coeficientes de la regresión lineal entrenada con las 289 (intercept + 16 coef)
LINMODEL = {
    'open_k': {
        'intercept': -47.410295,
        'coef': [-63.38993348, 1.84279037, -1.44001408, 0.05660939, 0.05621225,
                 19.12823518, 222.1255056, 0.20235371, 6.74392716, -0.22085721,
                 -0.7351967, -134.32654084, 0.00689731, 0.01042349, 54.09694702,
                 -0.01142687],
    },
    'min_branch': {
        'intercept': 37.232889,
        'coef': [-84.0930219, -1.27909852, 4.73778706, 0.37602549, -0.42387433,
                 -4.31641049, -13.52995101, -0.54059273, 6.31112496, 0.85115127,
                 1.17218938, -114.70112638, 0.0244663, 0.00024658, 5.54948928,
                 0.82505752],
    },
}


def extract_features(img_gray: np.ndarray) -> dict:
    """Mide las 16 características de la máscara (mismo cálculo con que se entrenó)."""
    _, b = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(b) > 127:
        b = cv2.bitwise_not(b)
    fg = b > 0
    area = int(fg.sum())
    f = {k: 0.0 for k in FEATURE_NAMES}
    if area == 0:
        return f

    f['area_frac'] = area / b.size
    dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    dv = dist[dist > 0]
    if dv.size:
        f['dist_mean']     = float(dv.mean())
        f['dist_std']      = float(dv.std())
        f['dist_p90']      = float(np.percentile(dv, 90))
        f['dist_max']      = float(dv.max())
        f['frac_gruesa']   = float((dv > 5.5).mean())
        f['frac_muy_fina'] = float((dv < 2).mean())

    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(b)
    f['n_components'] = n_cc - 1
    if n_cc > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        f['frac_en_mayor'] = float(areas.max() / area)
        f['n_cc_pequenos'] = int((areas < 50).sum())
        f['n_cc_medianos'] = int(((areas >= 50) & (areas < 500)).sum())

    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    f['perim_area_ratio'] = perim / area if area else 0.0

    ys, xs = np.where(fg)
    if ys.size:
        bh = int(ys.max() - ys.min()); bw = int(xs.max() - xs.min())
        f['bbox_h'] = bh
        f['bbox_w'] = bw
        f['bbox_fill'] = float(area / max(1, bh * bw))

    n_inv, _, si, _ = cv2.connectedComponentsWithStats(255 - b)
    f['n_huecos'] = int((si[1:, cv2.CC_STAT_AREA] < area).sum()) - 1 if n_inv > 1 else 0
    return f


def predict_params(img_gray: np.ndarray) -> dict:
    """
    Predice los 5 parámetros para una máscara con el calibrador incrustado:
      - open_k / min_branch  -> regresión lineal (coeficientes en LINMODEL)
      - close_k / thick_dist / dilate_k -> consenso (CONSENSO)
    """
    f = extract_features(img_gray)
    x = [f[k] for k in FEATURE_NAMES]
    params = dict(CONSENSO)
    for k, m in LINMODEL.items():
        params[k] = m['intercept'] + sum(c * xi for c, xi in zip(m['coef'], x))
    return _clean_params(params)


def _clean_params(p: dict) -> dict:
    """Redondea a enteros (salvo thick_dist) y aplica los defaults faltantes."""
    return dict(
        close_k    = int(round(p.get('close_k',    CLOSE_KSIZE))),
        open_k     = int(round(p.get('open_k',     OPEN_KSIZE))),
        thick_dist = round(float(p.get('thick_dist', THICKNESS_MAX_DIST)), 1),
        dilate_k   = int(round(p.get('dilate_k',   THICK_DILATE_KSIZE))),
        min_branch = int(round(p.get('min_branch', MIN_BRANCH_PX))),
    )


# ---------------------------------------------------------------------------
# Interfaz de tuneo manual del suavizado (integrada)
# ---------------------------------------------------------------------------
def tune_gui(img_gray: np.ndarray, init_params: dict, title: str = "") -> Optional[dict]:
    """
    Abre la interfaz para ajustar a mano el suavizado de UNA máscara.
    Muestra máscara | esqueleto en vivo según los sliders, precargados con
    init_params (los parámetros predichos por el modelo).

    Devuelve los parámetros confirmados al pulsar "Guardar + Esqueletizar",
    o None si se cierra la ventana sin guardar (se cancela el procesamiento).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    p0 = _clean_params(init_params)
    result = {'params': None}

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1e1e1e')
    ax_orig = fig.add_axes([0.05, 0.30, 0.42, 0.62]); ax_orig.axis('off')
    ax_skel = fig.add_axes([0.53, 0.30, 0.42, 0.62]); ax_skel.axis('off')
    ax_orig.set_title('Máscara', color='white', fontsize=10)
    ax_skel.set_title('Esqueleto (en vivo)', color='white', fontsize=10)
    sup = fig.suptitle(f'AJUSTE MANUAL — {title}', color='#5a9fd4', fontsize=12, fontweight='bold')

    im_orig = ax_orig.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    skel0 = _run_pipeline(img_gray, p0['close_k'], p0['open_k'],
                          p0['thick_dist'], p0['dilate_k'], p0['min_branch'])
    im_skel = ax_skel.imshow(skel0, cmap='gray', vmin=0, vmax=255)
    px_text = ax_skel.text(0.01, 0.02, '', transform=ax_skel.transAxes,
                           color='yellow', fontsize=9, va='bottom')

    def mk(left, bottom, label, lo, hi, init, step):
        ax = fig.add_axes([left, bottom, 0.20, 0.025], facecolor='#3a3a3a')
        s = Slider(ax, label, lo, hi, valinit=init, valstep=step, color='#5a9fd4')
        s.label.set_color('white'); s.valtext.set_color('white')
        return s

    s_close  = mk(0.10, 0.20, 'Cierre',   1,   25,   p0['close_k'],    2)
    s_thick  = mk(0.43, 0.20, 'Grosor',   1.0, 20.0, p0['thick_dist'], 0.5)
    s_prune  = mk(0.73, 0.20, 'Pruning',  5,   200,  p0['min_branch'], 5)
    s_open   = mk(0.26, 0.14, 'Apertura', 1,   15,   p0['open_k'],     1)
    s_dilate = mk(0.57, 0.14, 'Dilat.',   1,   31,   p0['dilate_k'],   2)

    def cur():
        return dict(close_k=int(s_close.val), open_k=int(s_open.val),
                    thick_dist=float(s_thick.val), dilate_k=int(s_dilate.val),
                    min_branch=int(s_prune.val))

    def redraw(_=None):
        p = cur()
        sk = _run_pipeline(img_gray, p['close_k'], p['open_k'],
                           p['thick_dist'], p['dilate_k'], p['min_branch'])
        im_skel.set_data(sk)
        px_text.set_text(f"Píxeles: {int((sk>0).sum())}   params: {p}")
        fig.canvas.draw_idle()

    for s in (s_close, s_thick, s_prune, s_open, s_dilate):
        s.on_changed(redraw)
    redraw()

    ax_btn = fig.add_axes([0.40, 0.04, 0.22, 0.06], facecolor='#2a7a2a')
    btn = Button(ax_btn, 'Guardar + Esqueletizar', color='#2a7a2a', hovercolor='#3a9a3a')
    btn.label.set_color('white')

    def on_save(_):
        result['params'] = cur()
        plt.close(fig)

    btn.on_clicked(on_save)
    plt.show()
    return result['params']


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def process_mask(img_gray: np.ndarray, adaptive: bool = True,
                 defaults: Optional[dict] = None):
    """
    Esqueletiza una máscara en modo automático.

    Prioridad de los parámetros close_k / open_k / thick_dist / dilate_k /
    min_branch:
      - Si 'defaults' viene dado (predicho por el calibrador aprendido para
        esta imagen), se usan esos valores.
      - Si no, y adaptive=True, se estiman grosor/dilatación con Otsu y el
        resto sale de los valores fijos del CONFIG.
      - Si no, se usan los valores fijos del CONFIG.

    Retorna (skel_uint8, params_dict) con los parámetros realmente usados.
    """
    d = defaults or {}
    close_k    = int(d.get('close_k',    CLOSE_KSIZE))
    open_k     = int(d.get('open_k',     OPEN_KSIZE))
    thick_dist = float(d.get('thick_dist', THICKNESS_MAX_DIST))
    dilate_k   = int(d.get('dilate_k',   THICK_DILATE_KSIZE))
    min_branch = int(d.get('min_branch', MIN_BRANCH_PX))

    # 1. Binarizar
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 2. Suavizado morfológico
    ck = max(1, close_k) | 1
    ok = max(1, open_k)  | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    smoothed = cv2.morphologyEx(binary,   cv2.MORPH_CLOSE, k_close)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,  k_open)

    # 3. Transformada de distancia
    dist = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)

    # 4. Grosor/dilatación: solo se estiman con Otsu si NO hay defaults y adaptive=True
    if defaults is None and adaptive:
        thick_dist, dilate_k = estimate_thickness_params(dist)

    dilate_k = max(3, int(dilate_k) | 1)

    # 5. Filtro por grosor
    thin_mask     = ((smoothed > 0) & (dist <= thick_dist)).astype(np.uint8) * 255
    thick_centers = (dist > thick_dist).astype(np.uint8) * 255

    # 6. Dilatar centros gruesos y reconstruir región válida
    k_dil        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    thick_region = cv2.dilate(thick_centers, k_dil)
    valid_mask   = cv2.bitwise_or(thin_mask, thick_region)
    valid_mask   = cv2.bitwise_and(valid_mask, smoothed)

    # 7. Esqueletizar
    skel = skeletonize(valid_mask > 0).astype(np.uint8)

    # 8. Pruning
    skel = prune_skeleton(skel, max(1, min_branch))

    params = {
        'close_k':    ck,
        'open_k':     ok,
        'thick_dist': round(float(thick_dist), 2),
        'dilate_k':   dilate_k,
        'min_branch': min_branch,
        'dist_max':   round(float(dist.max()), 2),
    }
    return (skel * 255).astype(np.uint8), params


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def load_params_db(input_folder: str) -> dict:
    """Carga parametros.json si existe junto a las máscaras."""
    p = Path(input_folder) / "parametros.json"
    if p.exists():
        with open(p, 'r') as f:
            db = json.load(f)
        print(f"[INFO] parametros.json encontrado: {len(db)} imágenes con parámetros manuales")
        return db
    print("[INFO] parametros.json no encontrado — se usará el modo automático para todas")
    return {}


def batch_skeletonize(input_folder: str, output_folder: str) -> None:
    input_path  = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"[ERROR] La carpeta de entrada no existe: {input_folder}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"[WARN] No se encontraron imágenes en: {input_folder}")
        return

    params_db = load_params_db(input_folder)

    if ADAPTIVE:
        mode_str = "ADAPTATIVO (Otsu sobre transformada de distancia)"
    else:
        mode_str = "CALIBRADOR APRENDIDO incrustado (regresión + consenso)"
    print(f"[INFO] Modo automático (imágenes sin calibración manual): {mode_str}")
    print(f"[INFO] Imágenes encontradas: {len(image_files)}")
    print(f"[INFO] Guardando en: {output_folder}\n")

    ok = errors = total_repairs = 0

    for i, img_file in enumerate(image_files, 1):
        name = img_file.name
        print(f"[{i:4d}/{len(image_files)}] {name}", end=" ... ")

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("ERROR (no se pudo leer)")
            errors += 1
            continue

        try:
            if name in params_db:
                # Parámetros guardados manualmente desde el tuner
                p = params_db[name]
                skel = _run_pipeline(img, p['close_k'], p['open_k'],
                                     p['thick_dist'], p['dilate_k'], p['min_branch'])
                source = "manual"
            else:
                # Calibración automática: predice los parámetros de ESTA imagen
                pred = None if ADAPTIVE else predict_params(img)
                skel, params = process_mask(img, adaptive=ADAPTIVE,
                                            defaults=pred)
                source = "adaptativo" if ADAPTIVE else "calibrado"

            # Reparación automática de gaps de 1px (antes de guardar)
            n_repairs = 0
            if REPAIR_GAPS:
                skel_bin = (skel > 0).astype(np.uint8)
                skel_bin, n_repairs = repair_skeleton(skel_bin, REPAIR_LOCAL_STEPS)
                skel = (skel_bin * 255).astype(np.uint8)
        except Exception as e:
            print(f"ERROR ({e})")
            errors += 1
            continue

        out_file = output_path / name
        if cv2.imwrite(str(out_file), skel):
            px = np.sum(skel > 0)
            repair_str = f" | {n_repairs} gaps" if REPAIR_GAPS else ""
            print(f"OK  ({px} px | {source}{repair_str})")
            ok += 1
            total_repairs += n_repairs
        else:
            print("ERROR (no se pudo guardar)")
            errors += 1

    print(f"\n{'='*55}")
    print(f"  Procesadas : {ok + errors}")
    print(f"  Exitosas   : {ok}")
    print(f"  Con error  : {errors}")
    if REPAIR_GAPS:
        print(f"  Gaps reparados : {total_repairs}")
    print(f"{'='*55}")
    print(f"  Esqueletos guardados en: {output_folder}")


# ---------------------------------------------------------------------------
# Imagen única (automático o con ajuste manual)
# ---------------------------------------------------------------------------
def process_single(input_folder: str, output_folder: str,
                   name: str, manual_tune: bool = False) -> None:
    """
    Pipeline para UNA máscara:
      1. Carga la máscara binaria.
      2. Predice los parámetros de suavizado aprendidos para esa foto.
      3. Si manual_tune=True, abre la interfaz para ajustarlos; al guardar,
         continúa con los parámetros confirmados. Si la cierras sin guardar,
         se cancela.
      4. Esqueletiza, repara gaps y guarda el PNG.
    """
    input_path  = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    img_file = input_path / name
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] No se pudo leer: {img_file}")
        return

    # 2. Parámetros: ground truth manual > calibrador aprendido > fijos
    params_db = load_params_db(input_folder)
    if name in params_db:
        params = _clean_params(params_db[name])
        print(f"[INFO] {name}: usando parámetros MANUALES de parametros.json")
    else:
        params = predict_params(img)
        print(f"[INFO] {name}: parámetros APRENDIDOS -> {params}")

    # 3. Ajuste manual opcional
    if manual_tune:
        print("[INFO] Abriendo interfaz de ajuste manual del suavizado...")
        tuned = tune_gui(img, params, title=name)
        if tuned is None:
            print("[CANCELADO] Cerraste sin guardar — no se procesó la imagen.")
            return
        params = _clean_params(tuned)
        print(f"[INFO] Parámetros confirmados a mano -> {params}")

    # 4. Esqueletizar con los parámetros elegidos
    skel = _run_pipeline(img, params['close_k'], params['open_k'],
                         params['thick_dist'], params['dilate_k'], params['min_branch'])

    n_repairs = 0
    if REPAIR_GAPS:
        skel_bin = (skel > 0).astype(np.uint8)
        skel_bin, n_repairs = repair_skeleton(skel_bin, REPAIR_LOCAL_STEPS)
        skel = (skel_bin * 255).astype(np.uint8)

    out_file = output_path / name
    if cv2.imwrite(str(out_file), skel):
        repair_str = f" | {n_repairs} gaps" if REPAIR_GAPS else ""
        print(f"[OK] {name}: {int((skel>0).sum())} px{repair_str}  ->  {out_file}")
    else:
        print(f"[ERROR] No se pudo guardar: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        INPUT_FOLDER  = sys.argv[1]
        OUTPUT_FOLDER = sys.argv[2]

    if SINGLE_IMAGE:
        process_single(INPUT_FOLDER, OUTPUT_FOLDER, SINGLE_IMAGE, manual_tune=MANUAL_TUNE)
    else:
        batch_skeletonize(INPUT_FOLDER, OUTPUT_FOLDER)
