"""
Build Graph JSON — Construye JSON de grafo desde imagenes PNG de Grafos
=======================================================================
Lee cada PNG de la carpeta Grafos, extrae la estructura de ramas usando
branch_identifier2, y guarda un JSON con nodos, aristas y pixeles por rama.

Estos JSONs son el input para el metodo Graph Laplacian en floral_density.

Uso:
    Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAFOS_DIR        = r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"
OUTPUT_DIR        = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"

COLOR_TOLERANCE   = 25
PROXIMITY_GAP     = 7
MIN_BRANCH_PIXELS = 15
TRUNK_SHORT_RATIO = 0.30

# =====================================================================
#  NO MODIFICAR DEBAJO
# =====================================================================

import os
import glob
import json
import numpy as np
import cv2

from branch_identifier2 import (
    detect_background,
    extract_branches_by_color,
    build_adjacency,
    identify_trunk,
    classify_hierarchy,
    _hier_name,
)


def build_json_from_png(img_path):
    """
    Dado un PNG de Grafos, retorna el dict JSON con la estructura del grafo.

    Estructura del JSON:
      nodes  — una entrada por rama, con id, centroide, jerarquia, color
      edges  — pares de ramas adyacentes (aristas del grafo de ramas)
      branches — pixeles completos de cada rama (para conectar flores por k-NN)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {img_path}")

    # Eliminar texto blanco antes de detectar ramas
    bg_color = detect_background(img)
    img_clean = img.copy()
    white_mask = (
        (img_clean[:, :, 0] > 200) &
        (img_clean[:, :, 1] > 200) &
        (img_clean[:, :, 2] > 200)
    )
    img_clean[white_mask] = bg_color

    branches  = extract_branches_by_color(img_clean, bg_color,
                                          COLOR_TOLERANCE, MIN_BRANCH_PIXELS)
    adjacency = build_adjacency(branches, img.shape, PROXIMITY_GAP)
    trunk_ids = identify_trunk(branches, adjacency, TRUNK_SHORT_RATIO)
    levels    = classify_hierarchy(branches, adjacency, trunk_ids)

    # --- Nodos: una entrada por rama ---
    nodes = []
    for bid, branch in branches.items():
        b, g, r = branch['color_bgr']
        nodes.append({
            "id":         bid,
            "centroid_x": round(branch['avg_x'], 1),
            "centroid_y": round(branch['avg_y'], 1),
            "level":      levels.get(bid, -1),
            "level_name": _hier_name(levels.get(bid, -1)),
            "is_trunk":   bid in trunk_ids,
            "size_px":    branch['size'],
            "color_rgb":  [r, g, b],
        })

    # --- Aristas: pares de ramas adyacentes ---
    edges = []
    seen  = set()
    for bid, neighbors in adjacency.items():
        for nb in neighbors:
            key = (min(bid, nb), max(bid, nb))
            if key not in seen:
                seen.add(key)
                edges.append({"from": bid, "to": nb})

    # --- Ramas: pixeles completos (para k-NN con flores) ---
    branches_out = []
    for bid, branch in branches.items():
        branches_out.append({
            "id":     bid,
            "pixels": [[int(y), int(x)] for y, x in branch['pixels']],
        })

    return {
        "image":    os.path.basename(img_path),
        "nodes":    nodes,
        "edges":    edges,
        "branches": branches_out,
    }


# =====================================================================
#  EJECUCION
# =====================================================================
if __name__ == "__main__":
    input_files = sorted(glob.glob(os.path.join(GRAFOS_DIR, "*.png")))
    if not input_files:
        raise FileNotFoundError(f"No se encontraron PNG en: {GRAFOS_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Procesando {len(input_files)} imagen(es)...\n")

    ok, errors = 0, 0
    for img_path in input_files:
        fname    = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, fname + ".json")
        try:
            data = build_json_from_png(img_path)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            n_nodes  = len(data['nodes'])
            n_edges  = len(data['edges'])
            print(f"  [OK] {os.path.basename(img_path)} -> {n_nodes} ramas, {n_edges} aristas -> {os.path.basename(out_path)}")
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e}")
            errors += 1

    print(f"\n[DONE] {ok} OK, {errors} errores.")
