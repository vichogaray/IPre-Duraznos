"""
Skeletonize Batch — Esqueletización masiva de máscaras binarias
===============================================================
Pipeline por imagen:
  1. Binarización (Otsu)
  2. Suavizado morfológico: cierre → apertura
  3. Filtro por grosor via transformada de distancia
  4. Dilatar centros gruesos para reconstruir la región válida
  5. Esqueletización (skimage)
  6. Pruning: eliminar ramas menores a MIN_BRANCH_PX píxeles

Modos:
  ADAPTIVE = True  →  THICKNESS_MAX_DIST y THICK_DILATE_KSIZE se estiman
                       automáticamente para cada imagen usando Otsu sobre
                       la distribución de la transformada de distancia.
  ADAPTIVE = False →  Se usan los valores fijos definidos abajo.
"""

# =====================================================================
#  CONFIGURACIÓN — MODIFICA AQUÍ
# =====================================================================

INPUT_FOLDER  = r"C:\Users\vgara\OneDrive\Desktop\IPre\Mascaras filtradas"
OUTPUT_FOLDER = r"C:\Users\vgara\OneDrive\Desktop\IPre\Esqueletos filtrados"

# --- Modo adaptativo ---------------------------------------------------
ADAPTIVE = True   # True = parámetros automáticos por imagen

# Parámetros fijos (se usan siempre, o como fallback si ADAPTIVE falla)
CLOSE_KSIZE        = 7      # Kernel del cierre morfológico
OPEN_KSIZE         = 4      # Kernel de la apertura morfológica
THICKNESS_MAX_DIST = 5.5    # Umbral grosor (solo si ADAPTIVE = False)
THICK_DILATE_KSIZE = 11     # Dilatación centros gruesos (solo si ADAPTIVE = False)
MIN_BRANCH_PX      = 70     # Largo mínimo de rama en pruning (siempre fijo)

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


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def process_mask(img_gray: np.ndarray, adaptive: bool = True):
    """
    Retorna (skel_uint8, params_dict) donde params_dict contiene los
    parámetros usados (útil para logging).
    """
    # 1. Binarizar
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # 2. Suavizado morfológico
    close_k = max(1, CLOSE_KSIZE) | 1
    open_k  = max(1, OPEN_KSIZE)  | 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    smoothed = cv2.morphologyEx(binary,   cv2.MORPH_CLOSE, k_close)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN,  k_open)

    # 3. Transformada de distancia
    dist = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)

    # 4. Parámetros de grosor: adaptativos o fijos
    if adaptive:
        thick_dist, dilate_k = estimate_thickness_params(dist)
    else:
        thick_dist, dilate_k = THICKNESS_MAX_DIST, THICK_DILATE_KSIZE

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
    skel = prune_skeleton(skel, MIN_BRANCH_PX)

    params = {
        'thick_dist': round(float(thick_dist), 2),
        'dilate_k':   dilate_k,
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
    print("[INFO] parametros.json no encontrado — se usará modo adaptativo para todas")
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

    mode_str = "ADAPTATIVO" if ADAPTIVE else "FIJO"
    print(f"[INFO] Modo fallback: {mode_str}")
    print(f"[INFO] Imágenes encontradas: {len(image_files)}")
    print(f"[INFO] Guardando en: {output_folder}\n")

    ok = errors = 0

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
                skel, params = process_mask(img, adaptive=ADAPTIVE)
                source = "adaptativo" if ADAPTIVE else "fijo"
        except Exception as e:
            print(f"ERROR ({e})")
            errors += 1
            continue

        out_file = output_path / name
        if cv2.imwrite(str(out_file), skel):
            px = np.sum(skel > 0)
            print(f"OK  ({px} px | {source})")
            ok += 1
        else:
            print("ERROR (no se pudo guardar)")
            errors += 1

    print(f"\n{'='*55}")
    print(f"  Procesadas : {ok + errors}")
    print(f"  Exitosas   : {ok}")
    print(f"  Con error  : {errors}")
    print(f"{'='*55}")
    print(f"  Esqueletos guardados en: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        INPUT_FOLDER  = sys.argv[1]
        OUTPUT_FOLDER = sys.argv[2]

    batch_skeletonize(INPUT_FOLDER, OUTPUT_FOLDER)
