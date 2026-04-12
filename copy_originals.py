"""
Copia las imagenes originales que corresponden a cada grafo.
================================================================
Para cada grafo en GRAPHS_DIR, extrae el numero de frame,
busca la imagen original en IMGS_DIR y la copia a OUTPUT_DIR.

Uso:
    1. Configura los tres paths abajo
    2. Dale Run (F5)
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPHS_DIR = r"C:\Users\vgara\OneDrive\Desktop\7mo\bruno 1-184"
IMGS_DIR   = r"C:\Users\vgara\OneDrive\Desktop\IPre\IMGS"
OUTPUT_DIR = r"C:\Users\vgara\OneDrive\Desktop\7mo\originales 1-184"

# =====================================================================
#  NO NECESITAS MODIFICAR NADA DEBAJO DE ESTA LINEA
# =====================================================================

import os
import re
import shutil

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graph_files = [f for f in os.listdir(GRAPHS_DIR)
                   if re.search(r'frame\d+', f)]

    if not graph_files:
        print("[ERROR] No se encontraron grafos en:", GRAPHS_DIR)
        return

    print(f"[INFO] Grafos encontrados: {len(graph_files)}")
    print(f"[INFO] Destino: {OUTPUT_DIR}\n")

    copied = 0
    not_found = []

    for gfile in sorted(graph_files):
        match = re.search(r'frame(\d+)', gfile)
        if not match:
            continue
        frame_num = match.group(1)

        # Buscar imagen original con cualquier extension
        candidates = [f for f in os.listdir(IMGS_DIR)
                      if re.fullmatch(rf'imgs_frame{frame_num}\.[a-zA-Z]+', f)]

        if not candidates:
            not_found.append(gfile)
            print(f"  [FALTA]  frame{frame_num}  <- no se encontro en {IMGS_DIR}")
            continue

        src = os.path.join(IMGS_DIR, candidates[0])
        dst = os.path.join(OUTPUT_DIR, candidates[0])
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [OK]  {candidates[0]}")

    print(f"\n{'=' * 50}")
    print(f"  Copiadas: {copied}")
    if not_found:
        print(f"  No encontradas: {len(not_found)}")
        for f in not_found:
            print(f"    - {f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
