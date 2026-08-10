"""
Varilla L-System - Asignacion flor -> varilla (L-System) -> rama del tronco
===========================================================================
OBJETIVO: asignar CADA flor a una rama del esqueleto. La flor no toca el tronco;
entre medio hay una VARILLA invisible (la conexion flor<->tronco) que hay que
predecir. El L-System es el motor generativo que construye esa varilla.

    FLOR  ->  [VARILLA generada con L-System]  ->  RAMA del tronco

Pipeline:
  1. Clusterizar flores en varillas candidatas (DBSCAN anisotropico, >=3 flores)
  2. Para cada cluster, AJUSTAR la varilla L-System a las flores observadas
     (modelo forward generar_varilla + refinamiento least_squares) -> da P0
  3. Extrapolar el gap basal: P0 es el punto de insercion en el tronco
  4. Asignar P0 a la RAMA mas cercana del scaffold -> rama madre del cluster
  5. Reasignar cada flor suelta (ruido) a la varilla valida mas cercana
     -> COBERTURA 100%: toda flor queda asignada a una rama

Output: un solo plot de la imagen 100, flores coloreadas por la RAMA a la que
quedaron asignadas. NO guarda archivos.

Uso: F5. Edita los parametros nombrados abajo para tunear.
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIRS = [r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"]
JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"

# Imagen unica de salida.
#   - Si es un nombre de archivo -> procesa SOLO esa imagen.
#   - Si es None                 -> procesa TODOS los *_graph.json de
#                                   GRAPH_JSON_DIR (carpeta entera).
SINGLE_IMAGE = None

# Al procesar la carpeta entera (SINGLE_IMAGE = None):
#   - SHOW_PLOTS = True  -> abre un plot por imagen (uno tras otro; cierra cada
#                           ventana para pasar a la siguiente).
#   - SHOW_PLOTS = False -> NO abre ventanas, solo imprime el resumen por imagen
#                           (recomendado para lotes grandes).
SHOW_PLOTS = False

# Guardar cada plot como PNG en disco (en vez de / ademas de mostrarlo):
#   - SAVE_DIR = None        -> no guarda nada (solo muestra segun SHOW_PLOTS).
#   - SAVE_DIR = r"...ruta"  -> guarda un PNG por imagen en esa carpeta
#                               (la crea si no existe). Lo tipico para un lote:
#                               SAVE_DIR = "...", SHOW_PLOTS = False.
SAVE_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\salidas_lsystem"
SAVE_DPI = 150

# === ESCALA (calibracion px <-> cm POR IMAGEN) ===
# Auto: altura del esqueleto (max y - min y) / ASSUMED_TREE_HEIGHT_CM.
# Override calibrado en frame100: la distancia mediana flor-vecina es ~8 px y
# biologicamente es el internodo (~2.5 cm) -> 8/2.5 ~ 3.2 px/cm. La auto da 1.34
# px/cm, incoherente con la separacion visible de las flores. (None = auto.)
ASSUMED_TREE_HEIGHT_CM = 350.0
PIXELS_PER_CM_OVERRIDE = 3.2

# === PRIORS BOTANICOS (fijados por la botanica) ===
DELTA_CM      = 2.5     # longitud de internodo (espaciado entre flores)
GAP_BASAL_CM  = 9.0     # tramo basal sin flores (se extrapola hacia el tronco)
LONG_MIN_CM   = 30.0    # longitud min de varilla (validacion / score)
LONG_MAX_CM   = 80.0    # longitud max de varilla (validacion / score)
FLORES_MIN    = 20      # flores min por varilla (validacion / score)
FLORES_MAX    = 40      # flores max por varilla (validacion / score)

# === CLUSTERING (DBSCAN anisotropico) ===
# X' = X * ANISOTROPY_LAMBDA, Y' = Y. Penaliza separacion horizontal porque las
# varillas son aprox verticales -> favorece clusters colineales verticales.
EPS_CM             = 7.5   # radio DBSCAN en cm (~3 internodos)
DBSCAN_MIN_SAMPLES = 3     # min flores para formar una varilla FIABLE
ANISOTROPY_LAMBDA  = 4.0   # escala en X

# === AJUSTE L-SYSTEM (refinamiento del eje generado) ===
RANSAC_RESIDUAL_CM = 1.0   # tolerancia de inlier RANSAC (semilla del ajuste)
RANSAC_MAX_TRIALS  = 200
REFINE_LSYSTEM     = True  # True: refina (p0,theta0,kappa) con least_squares
KAPPA_INIT         = 0.0   # curvatura inicial por internodo (0 = recta)

# === ASIGNACION P0 -> RAMA ===
UMBRAL_D2D1 = 1.5    # d2/d1 <= umbral -> insercion ambigua (marca visual)

# === VALIDACION (score; NO descarta flores) ===
TOL_ESPACIADO = 0.4  # error relativo tolerado en el espaciado medio

# === VISUALIZACION ===
DARK_BG       = True
FLOWER_SIZE   = 7
LINE_WIDTH    = 1.4
P0_SIZE       = 70
SHOW_SCAFFOLD = True  # dibuja los pixeles del esqueleto por debajo

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import re
import json
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from skimage.measure import LineModelND, ransac


# =====================================================================
#  1. CARGA DE DATOS
# =====================================================================

def load_graph_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_flowers(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for s in data.get('shapes', []):
        if s.get('label') == 'flower' and s.get('points'):
            x, y = s['points'][0]
            out.append((float(x), float(y)))
    return out


def auto_detect_flower_json(graph_image_name, json_flores_dir):
    m = re.search(r'frame(\d+)', graph_image_name)
    if not m:
        raise ValueError("No se pudo extraer frame de: " + graph_image_name)
    p = os.path.join(json_flores_dir, "frame" + m.group(1) + ".json")
    if not os.path.exists(p):
        raise FileNotFoundError("No se encontro: " + p)
    return p


def auto_detect_graph_image(graph_image_name, grafos_img_dirs):
    if isinstance(grafos_img_dirs, str):
        grafos_img_dirs = [grafos_img_dirs]
    tried = []
    for d in grafos_img_dirs:
        p = os.path.join(d, graph_image_name)
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontro imagen:\n  " + "\n  ".join(tried))


# La calibracion px <-> cm vive en src/common/geometry.py para que exista una
# sola definicion compartida por todos los metodos. Se anade la raiz del
# repositorio al path para poder ejecutar este archivo directamente (F5).
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.common.geometry import estimate_pixels_per_cm


def build_scaffold(graph_data):
    """
    Junta TODOS los pixeles de TODAS las ramas como nube (x, y) + el branch_id
    de cada pixel. Pixeles en JSON estan en formato [y, x] -> los convertimos a
    (x, y) para coherencia con las coords de flores.

    Devuelve (scaffold_xy (M,2), pix_branch_id (M,), tree cKDTree).
    """
    pts = []
    bid = []
    for b in graph_data['branches']:
        for px in b['pixels']:
            y, x = px[0], px[1]
            pts.append((x, y))
            bid.append(b['id'])
    scaffold_xy = np.array(pts, dtype=np.float64)
    pix_branch_id = np.array(bid, dtype=np.int32)
    tree = cKDTree(scaffold_xy)
    return scaffold_xy, pix_branch_id, tree


# =====================================================================
#  2. MODELO GENERATIVO L-SYSTEM (forward) - eje unico
# =====================================================================

def generar_varilla(p0, theta0, n_internodos, kappa, delta_px, n_base):
    """
    Modelo forward de una varilla. Avanza delta_px por internodo, gira kappa
    radianes (curvatura), deposita flor lateral tras el gap basal (n_base).

    Devuelve (flores_pred (Nf,2), nodos_eje (n_internodos+1,2)).
    """
    x, y = float(p0[0]), float(p0[1])
    theta = float(theta0)
    nodos = [np.array([x, y])]
    flores = []
    for n in range(n_internodos):
        x += delta_px * np.cos(theta)
        y += delta_px * np.sin(theta)
        nodos.append(np.array([x, y]))
        if n >= n_base:
            flores.append(np.array([x, y]))
        theta += kappa
    flores_pred = np.array(flores) if flores else np.empty((0, 2))
    return flores_pred, np.array(nodos)


# =====================================================================
#  3. PROBLEMA INVERSO - construir la varilla a partir de las flores
# =====================================================================

def clusterizar_flores(flores_xy, eps_px, min_muestras, lambda_x):
    """
    DBSCAN anisotropico: X' = X * lambda_x, Y' = Y. Favorece colinealidad
    vertical. Retorna labels (-1 = ruido).
    """
    if len(flores_xy) == 0:
        return np.array([], dtype=np.int32)
    F = np.array(flores_xy, dtype=np.float64).copy()
    F[:, 0] *= lambda_x
    return DBSCAN(eps=eps_px, min_samples=min_muestras).fit_predict(F)


def ajustar_eje_ransac(flores_xy, residual_threshold, max_trials):
    """Eje semilla por RANSAC. Devuelve (origin, direction unit, inliers)."""
    model, inliers = ransac(
        flores_xy, LineModelND,
        min_samples=2, residual_threshold=residual_threshold,
        max_trials=max_trials,
    )
    origin = np.asarray(model.params[0], dtype=float)
    direction = np.asarray(model.params[1], dtype=float)
    direction = direction / np.linalg.norm(direction)
    return origin, direction, inliers


def proyectar_y_ordenar(flores_xy, origin, direction):
    """Proyecta flores sobre el eje y las ordena. Devuelve (orden, t_ord)."""
    t = (flores_xy - origin) @ direction
    orden = np.argsort(t)
    return orden, t[orden]


def orientar_direccion(flores_ord, direction, scaffold_xy):
    """
    Orienta 'direction' para que apunte DESDE el scaffold HACIA el apice; el
    extremo basal es el mas cercano al scaffold. Devuelve (base_xy, direction).
    """
    extremo_a = flores_ord[0]
    extremo_b = flores_ord[-1]
    da = np.min(np.linalg.norm(scaffold_xy - extremo_a, axis=1))
    db = np.min(np.linalg.norm(scaffold_xy - extremo_b, axis=1))
    if da <= db:
        base = extremo_a
        if (extremo_b - extremo_a) @ direction < 0:
            direction = -direction
    else:
        base = extremo_b
        if (extremo_a - extremo_b) @ direction < 0:
            direction = -direction
    return base, direction


def refinar_lsystem(flores_obs, p0_init, theta0_init, n_internodos,
                    kappa_init, delta_px, n_base):
    """
    Ajuste fino del L-System: mueve (p0, theta0, kappa) para que las flores
    GENERADAS por el modelo forward calcen con las observadas (least_squares).
    El L-System es aqui el motor: la varilla final es la que el modelo produce.

    Devuelve (p0 (2,), theta0, kappa, cost).
    """
    flores_obs = np.asarray(flores_obs, dtype=float)

    def residuos(params):
        p0x, p0y, theta0, kappa = params
        pred, _ = generar_varilla((p0x, p0y), theta0, n_internodos,
                                  kappa, delta_px, n_base)
        m = min(len(pred), len(flores_obs))
        if m == 0:
            return np.full(flores_obs.size, 1e3)
        r = (pred[:m] - flores_obs[:m]).ravel()
        # rellena si faltan flores predichas (penaliza longitudes incompatibles)
        if r.size < flores_obs.size:
            r = np.concatenate([r, np.full(flores_obs.size - r.size, 1e2)])
        return r

    x0 = [float(p0_init[0]), float(p0_init[1]),
          float(theta0_init), float(kappa_init)]
    try:
        sol = least_squares(residuos, x0, method='lm', max_nfev=200)
        p0 = np.array([sol.x[0], sol.x[1]], dtype=float)
        return p0, float(sol.x[2]), float(sol.x[3]), float(sol.cost)
    except Exception:
        return np.asarray(p0_init, float), float(theta0_init), \
               float(kappa_init), float('inf')


def asignar_p0_a_rama(p0, scaffold_xy, pix_branch_id, tree, umbral_d2d1):
    """
    Asigna P0 a la RAMA del pixel de scaffold mas cercano.
    Devuelve (branch_id, p_scaffold (2,), d1, d2, ambiguo).
    """
    dists, idxs = tree.query(p0, k=2)
    i1 = int(idxs[0])
    d1 = float(dists[0])
    d2 = float(dists[1])
    branch_id = int(pix_branch_id[i1])
    ambiguo = (d2 / max(d1, 1e-9)) <= umbral_d2d1
    return branch_id, scaffold_xy[i1], d1, d2, ambiguo


def validar_varilla(t_ordenado, n_flores, px_por_cm):
    """Score [0,1] de la varilla contra los priors botanicos (NO descarta)."""
    if len(t_ordenado) < 2:
        return dict(ok=False, score=0.0)
    espaciados_cm = np.diff(t_ordenado) / px_por_cm
    espaciado_med_cm = float(np.median(espaciados_cm))
    longitud_fertil_cm = float((t_ordenado[-1] - t_ordenado[0]) / px_por_cm)
    longitud_total_cm = longitud_fertil_cm + GAP_BASAL_CM
    ok_long = LONG_MIN_CM <= longitud_total_cm <= LONG_MAX_CM
    ok_n = FLORES_MIN <= n_flores <= FLORES_MAX
    err_esp = abs(espaciado_med_cm - DELTA_CM) / DELTA_CM
    ok_esp = err_esp <= TOL_ESPACIADO
    score = (1.0 - min(err_esp, 1.0)) * (0.5 * ok_long + 0.5 * ok_n)
    return dict(ok=bool(ok_long and ok_n and ok_esp), score=float(score),
                longitud_total_cm=longitud_total_cm,
                espaciado_med_cm=espaciado_med_cm)


# =====================================================================
#  4. RECONSTRUCCION COMPLETA (una imagen) - COBERTURA 100%
# =====================================================================

def reconstruir(graph_data, flores_xy, px_por_cm):
    """
    Devuelve:
      varillas        : lista de dicts (eje, p0, branch_id, flores, ...)
      flower_branch   : (N,) branch_id asignado a CADA flor (cobertura 100%)
      flower_varilla  : (N,) indice de varilla de cada flor (-1 si reasignada)
      scaffold_xy, pix_branch_id
    """
    delta_px = DELTA_CM * px_por_cm
    n_base = int(round(GAP_BASAL_CM / DELTA_CM))
    eps_px = EPS_CM * px_por_cm
    res_thr = RANSAC_RESIDUAL_CM * px_por_cm

    scaffold_xy, pix_branch_id, tree = build_scaffold(graph_data)
    flores_xy = np.array(flores_xy, dtype=np.float64)
    N = len(flores_xy)

    etiquetas = clusterizar_flores(flores_xy, eps_px, DBSCAN_MIN_SAMPLES,
                                   ANISOTROPY_LAMBDA)

    flower_branch = np.full(N, -1, dtype=np.int32)
    flower_varilla = np.full(N, -1, dtype=np.int32)

    varillas = []
    for c in sorted(set(etiquetas.tolist()) - {-1}):
        idx_cluster = np.where(etiquetas == c)[0]
        cluster = flores_xy[idx_cluster]
        if len(cluster) < 2:
            continue

        # --- eje semilla (RANSAC) ---
        try:
            origin, direction, _ = ajustar_eje_ransac(
                cluster, res_thr, RANSAC_MAX_TRIALS)
        except Exception:
            continue
        if direction is None or not np.all(np.isfinite(direction)):
            continue

        orden, _ = proyectar_y_ordenar(cluster, origin, direction)
        flores_ord = cluster[orden]
        idx_ord = idx_cluster[orden]
        base, direction = orientar_direccion(flores_ord, direction, scaffold_xy)

        # P0 semilla: extrapola el gap basal hacia atras desde la flor basal
        theta0 = float(np.arctan2(direction[1], direction[0]))
        p0_seed = flores_ord[0] - direction * (n_base * delta_px)

        # --- L-SYSTEM como motor: refina (p0, theta0, kappa) ---
        n_internodos = n_base + len(flores_ord)
        if REFINE_LSYSTEM:
            p0, theta0, kappa, _ = refinar_lsystem(
                flores_ord, p0_seed, theta0, n_internodos,
                KAPPA_INIT, delta_px, n_base)
            direction = np.array([np.cos(theta0), np.sin(theta0)])
        else:
            p0, kappa = p0_seed, KAPPA_INIT

        # eje generado por el L-System (para dibujar la varilla real)
        _, nodos_eje = generar_varilla(p0, theta0, n_internodos, kappa,
                                       delta_px, n_base)

        # --- asignar P0 a una RAMA ---
        branch_id, p_scaf, d1, d2, ambiguo = asignar_p0_a_rama(
            p0, scaffold_xy, pix_branch_id, tree, UMBRAL_D2D1)

        # t ordenado para validacion
        _, t_ord = proyectar_y_ordenar(flores_ord, p0, direction)
        t_ord = np.sort(t_ord)
        val = validar_varilla(t_ord, len(cluster), px_por_cm)

        vidx = len(varillas)
        flower_branch[idx_ord] = branch_id
        flower_varilla[idx_ord] = vidx

        varillas.append(dict(
            label=int(c), p0=p0, theta0=theta0, kappa=kappa,
            base=base, direction=direction, nodos_eje=nodos_eje,
            flores=flores_ord, flores_idx=idx_ord,
            branch_id=branch_id, p_scaffold=p_scaf,
            ambiguo=ambiguo, validacion=val,
        ))

    # --- 5. REASIGNAR flores sueltas (ruido) a la varilla mas cercana ---
    # garantiza cobertura 100%: ninguna flor queda sin rama
    sueltas = np.where(flower_branch < 0)[0]
    n_reasignadas = 0
    if len(sueltas) and varillas:
        cent = np.array([v['flores'].mean(axis=0) for v in varillas])
        vtree = cKDTree(cent)
        for fi in sueltas:
            _, vj = vtree.query(flores_xy[fi], k=1)
            flower_branch[fi] = varillas[int(vj)]['branch_id']
            flower_varilla[fi] = int(vj)
            n_reasignadas += 1

    # ultimo recurso: si aun queda alguna sin varilla, asigna a la rama mas
    # cercana del scaffold directamente (cobertura garantizada)
    restantes = np.where(flower_branch < 0)[0]
    for fi in restantes:
        _, ii = tree.query(flores_xy[fi], k=1)
        flower_branch[fi] = int(pix_branch_id[int(ii)])

    return (varillas, flower_branch, flower_varilla,
            scaffold_xy, pix_branch_id, n_reasignadas)


# =====================================================================
#  5. VISUALIZACION (un solo plot) - flores coloreadas por RAMA
# =====================================================================

def visualizar(img, scaffold_xy, pix_branch_id, varillas,
               flores_xy, flower_branch, title, save_path=None, show=True):
    if DARK_BG:
        plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 11))

    if img is not None:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha=0.45)
    else:
        ax.invert_yaxis()  # sin imagen: +Y hacia abajo (coords de imagen)

    # color por branch_id
    branch_ids = sorted(set(pix_branch_id.tolist()))
    cmap = plt.get_cmap('tab10')
    col_of = {bid: cmap(k % 10) for k, bid in enumerate(branch_ids)}

    # scaffold por debajo
    if SHOW_SCAFFOLD:
        for bid in branch_ids:
            m = pix_branch_id == bid
            ax.scatter(scaffold_xy[m, 0], scaffold_xy[m, 1], s=2,
                       color=col_of[bid], alpha=0.30, edgecolors='none')

    # flores coloreadas por la RAMA a la que quedaron asignadas (OUTPUT)
    fb = np.asarray(flower_branch)
    for bid in branch_ids:
        m = fb == bid
        if m.any():
            ax.scatter(flores_xy[m, 0], flores_xy[m, 1], s=FLOWER_SIZE,
                       color=col_of[bid], edgecolors='none')

    # varillas: eje generado por el L-System + gap basal + P0 -> rama
    for v in varillas:
        bid = v['branch_id']
        col = col_of.get(bid, (1, 1, 1, 1))
        ne = v['nodos_eje']
        ax.plot(ne[:, 0], ne[:, 1], '-', color=col, lw=LINE_WIDTH, alpha=0.9)
        # gap basal (P0 -> primera flor): tramo invisible inferido
        ax.plot([v['p0'][0], v['base'][0]], [v['p0'][1], v['base'][1]],
                '--', color=col, lw=LINE_WIDTH, alpha=0.7)
        # link P0 -> punto del scaffold (insercion en la rama)
        ps = v['p_scaffold']
        ax.plot([v['p0'][0], ps[0]], [v['p0'][1], ps[1]],
                ':', color='white', lw=0.6, alpha=0.5)
        # P0
        marker = 'X' if not v['ambiguo'] else 'P'
        ax.scatter([v['p0'][0]], [v['p0'][1]], marker=marker, s=P0_SIZE,
                   color=col, edgecolors='white', linewidths=0.7, zorder=5)

    n_asig = int((fb >= 0).sum())
    ax.set_title(
        "{0}\n{1} flores asignadas a rama (de {2})  |  {3} varillas  |  "
        "color = rama madre  |  X=P0 inequivoco  P=ambiguo".format(
            title, n_asig, len(flores_xy), len(varillas)), fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print("    [GUARDADO] " + save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
#  6. RUN (una imagen)
# =====================================================================

def run_one(graph_json_path):
    graph_data = load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = auto_detect_flower_json(image_name, JSON_FLORES_DIR)
    flowers = load_flowers(flower_json)
    try:
        img_path = auto_detect_graph_image(image_name, GRAFOS_IMG_DIRS)
        img = cv2.imread(img_path)
    except FileNotFoundError:
        img = None

    print("    Ramas: {0}  |  Flores: {1}".format(
        len(graph_data['branches']), len(flowers)))
    if not flowers:
        print("    [SKIP] Sin flores.")
        return

    if PIXELS_PER_CM_OVERRIDE is not None:
        px_per_cm = float(PIXELS_PER_CM_OVERRIDE)
        scale_src = "override"
    else:
        px_per_cm = estimate_pixels_per_cm(graph_data, ASSUMED_TREE_HEIGHT_CM)
        scale_src = "auto (altura esqueleto / {0:.0f} cm)".format(
            ASSUMED_TREE_HEIGHT_CM)
    print("    Escala: {0:.2f} px/cm  [{1}]".format(px_per_cm, scale_src))
    print("    N_BASE (internodos de gap): {0}".format(
        int(round(GAP_BASAL_CM / DELTA_CM))))

    (varillas, flower_branch, flower_varilla,
     scaffold_xy, pix_branch_id, n_reasig) = reconstruir(
        graph_data, flowers, px_per_cm)

    flores_xy = np.array(flowers, dtype=np.float64)
    n_asig = int((flower_branch >= 0).sum())
    n_ok = sum(1 for v in varillas if v['validacion'].get('ok', False))
    print("    Varillas: {0}  ({1} validas por priors)".format(
        len(varillas), n_ok))
    print("    Flores asignadas a rama: {0}/{1}  ({2} sueltas reasignadas)".format(
        n_asig, len(flores_xy), n_reasig))
    # reparto por rama
    import collections
    rep = collections.Counter(flower_branch.tolist())
    print("    Reparto por rama: " + ", ".join(
        "rama{0}={1}".format(b, rep[b]) for b in sorted(rep)))

    base = os.path.splitext(os.path.basename(graph_json_path))[0]
    save_path = None
    if SAVE_DIR is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, base + ".png")

    # genera la figura si hay que mostrarla o guardarla
    if SHOW_PLOTS or save_path is not None:
        visualizar(img, scaffold_xy, pix_branch_id, varillas,
                   flores_xy, flower_branch, base,
                   save_path=save_path, show=SHOW_PLOTS)


def listar_graph_jsons(graph_json_dir):
    """Devuelve los *_graph.json de la carpeta, en orden por numero de frame."""
    nombres = [f for f in os.listdir(graph_json_dir)
               if f.endswith("_graph.json")]

    def clave(nombre):
        m = re.search(r'frame(\d+)', nombre)
        return (int(m.group(1)) if m else 1 << 30, nombre)

    return sorted(nombres, key=clave)


if __name__ == "__main__":
    if SINGLE_IMAGE is not None:
        # --- una sola imagen ---
        rutas = [os.path.join(GRAPH_JSON_DIR, SINGLE_IMAGE)]
        if not os.path.exists(rutas[0]):
            raise FileNotFoundError(
                "No se encontro el JSON de grafo: " + rutas[0])
    else:
        # --- carpeta entera ---
        nombres = listar_graph_jsons(GRAPH_JSON_DIR)
        if not nombres:
            raise FileNotFoundError(
                "No se encontraron *_graph.json en: " + GRAPH_JSON_DIR)
        rutas = [os.path.join(GRAPH_JSON_DIR, n) for n in nombres]

    print("[INFO] Asignando flores -> varilla (L-System) -> rama")
    print("[INFO] {0} imagen(es) a procesar.\n".format(len(rutas)))

    fallidas = []
    for i, path in enumerate(rutas, 1):
        print("[{0}/{1}] {2}".format(i, len(rutas), os.path.basename(path)))
        try:
            run_one(path)
        except Exception as e:
            print("    [ERROR] {0}".format(e))
            fallidas.append((os.path.basename(path), str(e)))
        print("")

    if fallidas:
        print("[AVISO] {0} imagen(es) fallaron:".format(len(fallidas)))
        for nombre, err in fallidas:
            print("    - {0}: {1}".format(nombre, err))

    print("[DONE]")
