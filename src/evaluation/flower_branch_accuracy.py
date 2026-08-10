"""
GT Flor->Rama Eval - Similitud de metodos contra el ground truth
=================================================================
Evalua, para cada imagen con GT anotado (gt_annotator.py), la similitud de
cada metodo contra el GT. DOS tipos de metrica segun el metodo:

  A) ACCURACY FLOR->RAMA  (todos los metodos)
     Para cada flor real (json flores/), se compara la rama que le asigno el
     metodo contra la rama VERDADERA del GT. La rama verdadera de una flor se
     DERIVA del GT: es la rama del punto-GT mas cercano que anotaste.
     accuracy = % de flores con rama_metodo == rama_GT.
     (Solo importa la rama, NO el punto especifico dentro de ella.)

  B) WASSERSTEIN GEODESICO  (solo varilla_density 1 y 2)
     Similitud de la FORMA del perfil de carga floral a lo largo del
     esqueleto (reusa heatmap_wasserstein). Para metodos que producen
     varillas con base_xy + num_flowers.

Metodos evaluados:
  - varilla_density (v1) : accuracy flor->rama  +  Wasserstein
  - varilla_density_2 (v2): accuracy flor->rama  +  Wasserstein
  - glee, laplacian, random_walk : accuracy flor->rama
    (usan branch_id del JSON de grafo -> IDs alineados con el GT)

NOTA: euclidian / hybrid detectan ramas por COLOR del PNG -> sus branch_id NO
estan alineados con el grafo/GT, por eso quedan fuera de esta evaluacion.

Salida: tabla resumen en consola + CSV acumulativo (flor_rama_resumen.csv).

USO: F5 / python flower_branch_accuracy.py
"""

import os
import sys
import json
import glob
import importlib.util
import numpy as np
from scipy.spatial import cKDTree

# =====================================================================
#  CONFIG
# =====================================================================

# Raiz del repositorio, deducida de la ubicacion de este archivo
# (src/evaluation/ -> dos niveles arriba). Evita depender del directorio
# desde el que se ejecute.
REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DESKTOP    = r"C:\Users\vgara\OneDrive\Desktop"
GT_DIR     = DESKTOP + r"\GT"
BASE_DIR   = REPO_ROOT
FLORES_DIR = os.path.join(BASE_DIR, "json flores")
GRAPH_DIR  = os.path.join(BASE_DIR, "grafos json")
V1_JSON_DIR= os.path.join(BASE_DIR, "densidades", "densidad floral varilla", "json")

# Modulos de metodo, tras la reestructuracion del repositorio.
# Antes se cargaban por nombre desde codigos/densidades/; ahora viven en
# src/assignment/ y src/evaluation/ con nombres en ingles.
SRC_DIR    = os.path.join(BASE_DIR, "src")
MORPH_DIR  = os.path.join(SRC_DIR, "assignment", "morphological")
SHOOT_DIR  = os.path.join(SRC_DIR, "assignment", "shoot_reconstruction")

OUT_CSV    = GT_DIR + r"\flor_rama_resumen.csv"

# =====================================================================
#  Carga dinamica de los modulos de metodo (sin correr su plotting)
# =====================================================================

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

# matplotlib en modo Agg para que ningun import abra ventanas
import matplotlib
matplotlib.use('Agg')

MOD = {}
MOD['glee'] = _load_module('m_glee', os.path.join(MORPH_DIR, 'glee.py'))
MOD['lap']  = _load_module('m_lap',  os.path.join(MORPH_DIR, 'graph_laplacian.py'))
MOD['rw']   = _load_module('m_rw',   os.path.join(MORPH_DIR, 'random_walk.py'))
MOD['v2']   = _load_module('m_v2',   os.path.join(SHOOT_DIR, 'candidate_shoots.py'))

# heatmap_wasserstein (antes gt_heatmap_compare, en el escritorio) aporta el
# Wasserstein geodesico para v1/v2. Vive en esta misma carpeta.
HC = _load_module('m_hc', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'heatmap_wasserstein.py'))


# =====================================================================
#  I/O comun
# =====================================================================

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_flowers_xy(image_name):
    """Lista [(x,y)] de flores en el MISMO orden que usan los metodos."""
    return MOD['glee'].load_flowers(
        MOD['glee'].auto_detect_flower_json(image_name, FLORES_DIR))


# =====================================================================
#  RAMA VERDADERA POR FLOR (derivada del GT)
# =====================================================================

def gt_branch_per_flower(gt_data, flowers_xy):
    """
    Rama verdadera de cada flor = rama_id del punto-GT mas cercano.
    Devuelve array (n_flowers,) de rama_id.
    """
    pts = gt_data['points']
    P = np.array([[p['x'], p['y']] for p in pts], dtype=np.float64)
    R = np.array([p['rama_id'] for p in pts], dtype=np.int64)
    tree = cKDTree(P)
    F = np.array(flowers_xy, dtype=np.float64)
    _, nn = tree.query(F)
    return R[nn]


# =====================================================================
#  ASIGNACION FLOR->RAMA POR METODO  ->  array (n_flowers,) de rama_id
# =====================================================================

def assign_v1(graph_data, flowers_xy, image_name):
    """v1 ya tiene su JSON en disco: leemos flowers[].rama_id."""
    base = os.path.splitext(os.path.basename(graph_data['_graph_json_name']))[0]
    p = os.path.join(V1_JSON_DIR, base + "_varilla.json")
    d = load_json(p)
    # flowers en el JSON de v1 estan en el mismo orden que load_flowers
    return np.array([f.get('rama_id', -1) for f in d['flowers']], dtype=np.int64)


def assign_v2(graph_data, flowers_xy, image_name):
    m = MOD['v2']
    px_per_cm = m.estimate_pixels_per_cm(graph_data, m.ASSUMED_TREE_HEIGHT_CM)
    hp = m.build_hyperparams(px_per_cm)
    _, flower_assignments = m.assign_varillas_v2(graph_data, flowers_xy, hp)
    return np.array([fa['rama_id'] for fa in flower_assignments], dtype=np.int64)


def assign_glee(graph_data, flowers_xy, image_name):
    m = MOD['glee']
    fbi, idx_to_bid = m.asignar_flores_glee(graph_data, flowers_xy)
    return np.array([idx_to_bid.get(int(bi), -1) for bi in fbi], dtype=np.int64)


def assign_lap(graph_data, flowers_xy, image_name):
    m = MOD['lap']
    A, bid_to_idx, n_b, n_f = m.build_adjacency_matrix(
        graph_data, flowers_xy, m.K_NEIGHBORS)
    fbi = m.run_laplacian(A, n_b, n_f, m.ALPHA)
    branches = graph_data['branches']
    idx_to_bid = {i: b['id'] for i, b in enumerate(branches)}
    return np.array([idx_to_bid.get(int(bi), -1) for bi in fbi], dtype=np.int64)


def assign_rw(graph_data, flowers_xy, image_name):
    m = MOD['rw']
    A, bid_to_idx, n_b, n_f = m.build_graph(
        graph_data, flowers_xy, m.K_NEIGHBORS, m.K_FLOWER, m.SIGMA)
    fbi = m.run_random_walk(A, n_b, n_f)
    branches = graph_data['branches']
    idx_to_bid = {i: b['id'] for i, b in enumerate(branches)}
    return np.array([idx_to_bid.get(int(bi), -1) for bi in fbi], dtype=np.int64)


METHODS = [
    ('varilla_density',   assign_v1,   True),   # (nombre, fn, tiene_wasserstein)
    ('varilla_density_2', assign_v2,   True),
    ('glee',              assign_glee, False),
    ('laplacian',         assign_lap,  False),
    ('random_walk',       assign_rw,   False),
]


# =====================================================================
#  WASSERSTEIN para v1 / v2  (reusa heatmap_wasserstein)
# =====================================================================

def build_v2_varilla_data(graph_data, flowers_xy, image_name):
    """Genera el dict tipo-metodo (varillas[].base_xy + num_flowers) para v2."""
    m = MOD['v2']
    px_per_cm = m.estimate_pixels_per_cm(graph_data, m.ASSUMED_TREE_HEIGHT_CM)
    hp = m.build_hyperparams(px_per_cm)
    varillas, _ = m.assign_varillas_v2(graph_data, flowers_xy, hp)
    out = []
    for v in varillas:
        # base_xy = punto de contacto con la rama (origin/impact) si existe,
        # si no el basal de la varilla
        if v.get('found') and v.get('impact') is not None:
            bxy = [float(v['impact'][0]), float(v['impact'][1])]
        else:
            bxy = [float(v['p_basal'][0]), float(v['p_basal'][1])]
        out.append({'varilla_id': v['varilla_id'], 'rama_id': v['rama_id'],
                    'num_flowers': v['num_flowers'], 'base_xy': bxy})
    return {'image': image_name, 'n_flowers': int(sum(v['num_flowers'] for v in varillas)),
            'varillas': out}


def wasserstein_for(method_name, graph_data, gt_data, flowers_xy, image_name):
    if method_name == 'varilla_density':
        base = os.path.splitext(os.path.basename(graph_data['_graph_json_name']))[0]
        md = load_json(os.path.join(V1_JSON_DIR, base + "_varilla.json"))
    elif method_name == 'varilla_density_2':
        md = build_v2_varilla_data(graph_data, flowers_xy, image_name)
    else:
        return None
    res, _ = HC.compare(gt_data, md, graph_data)
    return res


# =====================================================================
#  MAIN
# =====================================================================

def eval_one(gt_path):
    gt_data = load_json(gt_path)
    image_name = gt_data['image']
    graph_json_name = gt_data.get('graph_json') or (
        os.path.splitext(image_name)[0] + ".json")
    graph_data = load_json(os.path.join(GRAPH_DIR, graph_json_name))
    graph_data['_graph_json_name'] = graph_json_name

    flowers_xy = load_flowers_xy(image_name)
    gt_rama = gt_branch_per_flower(gt_data, flowers_xy)
    n_f = len(flowers_xy)

    rows = []
    for name, fn, has_w in METHODS:
        try:
            pred = fn(graph_data, flowers_xy, image_name)
            if len(pred) != n_f:
                # desajuste de conteo -> recortar al minimo comun
                k = min(len(pred), n_f)
                acc = float(np.mean(pred[:k] == gt_rama[:k]))
                note = "n_pred={0}!=n_gt={1}".format(len(pred), n_f)
            else:
                acc = float(np.mean(pred == gt_rama))
                note = ""
        except Exception as e:
            acc, note = np.nan, "ERR:{0}".format(type(e).__name__)
            print("    [ERR] {0}: {1}".format(name, e))

        W_cm, sim = np.nan, np.nan
        if has_w and np.isfinite(acc):
            try:
                res = wasserstein_for(name, graph_data, gt_data, flowers_xy, image_name)
                if res:
                    W_cm, sim = res['W_cm'], res['similitud']
            except Exception as e:
                note = (note + " " if note else "") + "W_ERR:{0}".format(type(e).__name__)

        rows.append({'image': image_name, 'metodo': name,
                     'acc_flor_rama': acc, 'wasserstein_cm': W_cm,
                     'sim_forma': sim, 'n_flores': n_f, 'nota': note})
    return rows


def write_csv(all_rows):
    import csv
    fields = ['image', 'metodo', 'acc_flor_rama', 'wasserstein_cm',
              'sim_forma', 'n_flores', 'nota']
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                'image': r['image'], 'metodo': r['metodo'],
                'acc_flor_rama': ('' if not np.isfinite(r['acc_flor_rama'])
                                  else round(r['acc_flor_rama'], 4)),
                'wasserstein_cm': ('' if not np.isfinite(r['wasserstein_cm'])
                                   else round(r['wasserstein_cm'], 2)),
                'sim_forma': ('' if not np.isfinite(r['sim_forma'])
                              else round(r['sim_forma'], 4)),
                'n_flores': r['n_flores'], 'nota': r['nota'],
            })
    print("\n[CSV] -> " + OUT_CSV)


if __name__ == "__main__":
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*_GT.json")))
    # excluir GT de prueba (frame100 era sintetico)
    gt_files = [g for g in gt_files if 'frame100_' not in os.path.basename(g)]
    if not gt_files:
        raise FileNotFoundError("No hay *_GT.json en " + GT_DIR)

    print("[INFO] {0} imagenes con GT: {1}\n".format(
        len(gt_files), [os.path.basename(g) for g in gt_files]))

    all_rows = []
    for gt in gt_files:
        print("=== " + os.path.basename(gt) + " ===")
        rows = eval_one(gt)
        all_rows.extend(rows)
        for r in rows:
            acc = '  n/a' if not np.isfinite(r['acc_flor_rama']) else "{0:5.1%}".format(r['acc_flor_rama'])
            w   = '' if not np.isfinite(r['wasserstein_cm']) else "W={0:.1f}cm sim={1:.3f}".format(r['wasserstein_cm'], r['sim_forma'])
            print("    {0:20s} acc_flor_rama={1}  {2}  {3}".format(r['metodo'], acc, w, r['nota']))

    write_csv(all_rows)

    # ---- tabla resumen (promedios por metodo) ----
    print("\n" + "=" * 64)
    print("  RESUMEN (promedio sobre {0} imagenes)".format(len(gt_files)))
    print("=" * 64)
    print("  {0:20s} {1:>14s} {2:>14s}".format("metodo", "acc_flor_rama", "sim_forma(W)"))
    print("  " + "-" * 60)
    for name, _, _ in METHODS:
        accs = [r['acc_flor_rama'] for r in all_rows if r['metodo'] == name and np.isfinite(r['acc_flor_rama'])]
        sims = [r['sim_forma'] for r in all_rows if r['metodo'] == name and np.isfinite(r['sim_forma'])]
        acc_m = np.mean(accs) if accs else np.nan
        sim_m = np.mean(sims) if sims else np.nan
        acc_s = '   n/a' if not np.isfinite(acc_m) else "{0:6.1%}".format(acc_m)
        sim_s = '     -' if not np.isfinite(sim_m) else "{0:6.3f}".format(sim_m)
        print("  {0:20s} {1:>14s} {2:>14s}".format(name, acc_s, sim_s))
    print("=" * 64)
