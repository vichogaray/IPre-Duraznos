"""
GT Annotator - Ground Truth de carga floral por punto del esqueleto
====================================================================
Herramienta manual para construir el "ground truth" contra el que se
contrastan los metodos de densidad floral.

QUE HACES:
  - Se muestra la imagen del esqueleto (grafo coloreado por rama).
  - Haces CLIC IZQUIERDO sobre una rama -> el punto se "pega" (snap) al
    pixel del esqueleto mas cercano y se te pide cuantas flores hay en
    ese punto.
  - CLIC DERECHO cerca de un punto existente -> lo borra.
  - Tecla 's' -> guarda el JSON.  Tecla 'u' -> deshace el ultimo punto.
  - El punto queda asociado automaticamente a la RAMA a la que pertenece
    ese pixel del esqueleto (rama_id del grafo).

SALIDA (formato que caracteriza lo que hiciste):
  Un JSON con la MISMA estructura que consume Varilla_heatmap.compute_heat,
  de modo que el heatmap del GT se construye con el MISMO codigo y los
  MISMOS parametros que el de cualquier metodo:
     {
       "image": "...png",
       "graph_json": "...graph.json",
       "n_points": K,
       "n_flowers": <suma de flores>,
       "varillas": [ {"varilla_id": i, "rama_id": r,
                      "num_flowers": n, "base_xy": [x, y]}, ... ],
       "points":   [ {"x":.., "y":.., "num_flowers":.., "rama_id":..}, ... ]
     }
  ("varillas" se incluye para que compute_heat lo lea tal cual: cada
   punto GT = una "varilla" con su base_xy y su num_flowers.)

USO:
  1. Ajusta INPUT_GRAPH_JSON abajo (cambia de imagen aqui).
  2. F5 / python gt_annotator.py
  3. Anota, pulsa 's' para guardar.
"""

# =====================================================================
#  CONFIGURACION  (cambia el input aqui)
# =====================================================================

BASE_DIR        = r"C:\Users\vgara\OneDrive\Desktop\IPre"
GRAPH_JSON_DIR  = BASE_DIR + r"\grafos json"
GRAFOS_IMG_DIR  = BASE_DIR + r"\Grafos"
OUTPUT_DIR      = r"C:\Users\vgara\OneDrive\Desktop\GT"   # donde se guarda el GT

# Imagen a anotar: nombre del JSON de grafo dentro de GRAPH_JSON_DIR
INPUT_GRAPH_JSON = "imgs_frame172_00000_graph.json"

# Radio maximo (px) para que un clic se considere "sobre el esqueleto".
# Si clicas mas lejos que esto de cualquier pixel de rama, se ignora.
SNAP_MAX_DIST_PX = 40.0

# Valor de flores por defecto que se propone en el cuadro de texto
DEFAULT_FLOWERS  = 1

# Apariencia
POINT_COLOR   = 'yellow'
POINT_SIZE    = 90
LABEL_COLOR   = 'white'
DARK_BG       = True

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.spatial import cKDTree


# =====================================================================
#  I/O
# =====================================================================

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build_skeleton(graph_data):
    """Devuelve (skel_xy Nx2 en (x,y), rama_id_por_pixel N)."""
    pts, rid = [], []
    for b in graph_data['branches']:
        bid = b['id']
        for px in b['pixels']:
            y, x = px[0], px[1]
            pts.append((x, y))
            rid.append(bid)
    return np.asarray(pts, dtype=np.float64), np.asarray(rid, dtype=np.int64)


# =====================================================================
#  ANOTADOR
# =====================================================================

class Annotator:
    def __init__(self, graph_json_path):
        self.graph_json_path = graph_json_path
        self.graph = load_json(graph_json_path)
        self.image_name = self.graph['image']

        img_path = os.path.join(GRAFOS_IMG_DIR, self.image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError("No se encontro imagen: " + img_path)
        bgr = cv2.imread(img_path)
        self.img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.skel_xy, self.skel_rid = build_skeleton(self.graph)
        if len(self.skel_xy) == 0:
            raise ValueError("El grafo no tiene pixeles de esqueleto.")
        self.tree = cKDTree(self.skel_xy)

        # puntos anotados: lista de dicts {x, y, num_flowers, rama_id}
        self.points = []

        self._pending_xy = None     # clic esperando confirmar nº flores
        self._build_fig()

    # -------------------------------------------------- figura / widgets
    def _build_fig(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 11))
        if DARK_BG:
            self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.imshow(self.img_rgb)
        self.ax.set_title(self._title(), color='white' if DARK_BG else 'black',
                          fontsize=10)
        self.ax.axis('off')

        # cuadro de texto para nº de flores
        axbox = self.fig.add_axes([0.15, 0.02, 0.20, 0.05])
        self.textbox = TextBox(axbox, 'Flores: ', initial=str(DEFAULT_FLOWERS))
        self.textbox.on_submit(self._on_submit_flowers)

        axsave = self.fig.add_axes([0.40, 0.02, 0.15, 0.05])
        self.btn_save = Button(axsave, 'Guardar (s)')
        self.btn_save.on_clicked(lambda e: self.save())

        axundo = self.fig.add_axes([0.58, 0.02, 0.15, 0.05])
        self.btn_undo = Button(axundo, 'Deshacer (u)')
        self.btn_undo.on_clicked(lambda e: self.undo())

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._redraw()

    def _title(self):
        nf = sum(p['num_flowers'] for p in self.points)
        return ("{img}\nPuntos: {k}   Flores: {nf}   "
                "[Izq=agregar  Der=borrar  s=guardar  u=deshacer]").format(
            img=self.image_name, k=len(self.points), nf=nf)

    # -------------------------------------------------- eventos
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if event.button == 1:          # izquierdo -> agregar
            d, idx = self.tree.query([x, y])
            if d > SNAP_MAX_DIST_PX:
                self._flash("Clic lejos del esqueleto (>{:.0f}px), ignorado."
                            .format(SNAP_MAX_DIST_PX))
                return
            sx, sy = self.skel_xy[idx]
            rid = int(self.skel_rid[idx])
            # usa el valor actual del cuadro de texto como nº flores
            try:
                n = int(float(self.textbox.text))
            except ValueError:
                n = DEFAULT_FLOWERS
            if n <= 0:
                self._flash("nº flores debe ser > 0.")
                return
            self.points.append({'x': float(sx), 'y': float(sy),
                                 'num_flowers': int(n), 'rama_id': rid})
            self._redraw()

        elif event.button == 3:        # derecho -> borrar el mas cercano
            if not self.points:
                return
            px = np.array([[p['x'], p['y']] for p in self.points])
            dd = np.hypot(px[:, 0] - x, px[:, 1] - y)
            j = int(np.argmin(dd))
            if dd[j] < 25:
                self.points.pop(j)
                self._redraw()

    def _on_submit_flowers(self, text):
        # solo valida; el valor se usa en el proximo clic izquierdo
        try:
            int(float(text))
        except ValueError:
            self.textbox.set_val(str(DEFAULT_FLOWERS))

    def _on_key(self, event):
        if event.key == 's':
            self.save()
        elif event.key == 'u':
            self.undo()

    # -------------------------------------------------- acciones
    def undo(self):
        if self.points:
            self.points.pop()
            self._redraw()

    def _flash(self, msg):
        print("[i] " + msg)

    def _redraw(self):
        # limpia scatter/labels previos redibujando desde cero
        for art in list(self.ax.collections):
            art.remove()
        for txt in list(self.ax.texts):
            txt.remove()
        if self.points:
            px = np.array([[p['x'], p['y']] for p in self.points])
            self.ax.scatter(px[:, 0], px[:, 1], s=POINT_SIZE,
                            facecolors='none', edgecolors=POINT_COLOR,
                            linewidths=1.8, zorder=5)
            for p in self.points:
                self.ax.annotate(str(p['num_flowers']),
                                 (p['x'], p['y']),
                                 color=LABEL_COLOR, fontsize=8,
                                 ha='center', va='center', zorder=6)
        self.ax.set_title(self._title(),
                          color='white' if DARK_BG else 'black', fontsize=10)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------- guardar
    def save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.graph_json_path))[0]
        out_path = os.path.join(OUTPUT_DIR, base + "_GT.json")

        varillas = []
        for i, p in enumerate(self.points):
            varillas.append({
                'varilla_id': i,
                'rama_id': p['rama_id'],
                'num_flowers': p['num_flowers'],
                'base_xy': [p['x'], p['y']],
            })
        data = {
            'image': self.image_name,
            'graph_json': os.path.basename(self.graph_json_path),
            'source': 'ground_truth_manual',
            'n_points': len(self.points),
            'n_varillas': len(self.points),
            'n_flowers': int(sum(p['num_flowers'] for p in self.points)),
            'varillas': varillas,
            'points': self.points,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("[OK] Guardado GT: {0}  ({1} puntos, {2} flores)".format(
            out_path, len(self.points), data['n_flowers']))
        # feedback visual en el titulo
        self.ax.set_title("GUARDADO -> " + os.path.basename(out_path),
                          color='lime', fontsize=10)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    graph_path = os.path.join(GRAPH_JSON_DIR, INPUT_GRAPH_JSON)
    if not os.path.exists(graph_path):
        raise FileNotFoundError("No existe el grafo JSON: " + graph_path)
    ann = Annotator(graph_path)
    print("[INFO] Anotando:", ann.image_name)
    print("       Clic IZQ = agregar punto (usa el valor de 'Flores:')")
    print("       Clic DER = borrar punto cercano")
    print("       's' = guardar   'u' = deshacer")
    ann.run()
