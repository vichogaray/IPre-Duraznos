"""
Varilla Density - Asignacion flor -> varilla (ramo mixto) -> rama del esqueleto
==============================================================================
Idea: las flores no nacen del esqueleto visible sino de varillas (ramos mixtos)
de 30-80 cm que crecen del esqueleto y NO son visibles. Este algoritmo:

  1. Agrupa flores en clusters lineales (DBSCAN anisotropico)
  2. Ajusta una direccion principal por cluster (PCA + trim de outliers)
  3. Extiende la varilla mas alla del flor mas bajo (gap basal sin flores)
  4. Proyecta el extremo basal al esqueleto -> rama madre

Input:
  - JSON de grafo (grafos json/) con ramas etiquetadas
  - JSON de flores (json flores/) con coords (x, y)

Output:
  - PNG visualizacion: flores coloreadas por varilla + linea de cada varilla
  - JSON asignacion: flor -> varilla -> rama

Uso:
    1. Ajusta la CONFIGURACION abajo (sobre todo ASSUMED_TREE_HEIGHT_CM)
    2. Pon IMAGE_NAME con el JSON de grafo a procesar
    3. F5 -> abre el plot de matplotlib
"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIRS = [r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"]
JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"

# Imagen a procesar: nombre del JSON de grafo dentro de GRAPH_JSON_DIR
IMAGE_NAME = "imgs_frame101_00000_graph.json"

# === ESCALA (calibracion px <-> cm POR IMAGEN) ===
# Auto-calibracion: medimos la altura del esqueleto (max y - min y de TODOS
# los pixeles de ramas) y la dividimos por ASSUMED_TREE_HEIGHT_CM. Un duraznero
# adulto mide entre 3 y 4 m -> 350 cm es el valor central razonable.
# PIXELS_PER_CM_OVERRIDE permite forzar un valor fijo (None = auto).
ASSUMED_TREE_HEIGHT_CM = 350.0
PIXELS_PER_CM_OVERRIDE = None

# === GEOMETRIA DE VARILLA ===
VARILLA_LENGTH_MIN_CM   = 8.0    # bajo el min descarta el cluster
VARILLA_LENGTH_MAX_CM   = 80.0   # arriba del max marca como sospechoso
INTERNODE_CM            = 3.5    # espaciamiento tipico entre flores
BASAL_GAP_CM            = 7.0    # zona basal sin flores
MAX_BASAL_EXTENSION_CM  = 18.0   # cuanto extender mas alla del ultimo flor

# === CLUSTERING ===
DBSCAN_EPS_CM        = 4.0       # ~1.5 internodos
DBSCAN_MIN_SAMPLES   = 1    # 1 -> cero ruido: TODA flor cae en alguna varilla
ANISOTROPY_LAMBDA    = 4.0       # cuanto penalizar separacion en X vs Y
                                 # (varillas son aprox verticales -> X scale > 1)

# === FORMA DE LA VARILLA (parabola muy poco achatada) ===
USE_QUADRATIC_THRESHOLD = 10      # con >= N flores, ajustar parabola
MAX_CURVATURE_COEF      = 0.0012 # 1/px, tope de curvatura (mantener "poco achatada")

# === FILTROS (solo informativos: no descartan flores) ===
FLOWERS_PER_VARILLA_MIN = 3
FLOWERS_PER_VARILLA_MAX = 60

# === FUSION DE VARILLAS PARALELAS Y CERCANAS ===
# Tras el clustering inicial, se fusionan varillas que cumplen LOS TRES:
#   - angulo entre ejes principales <= MERGE_ANGLE_DEG
#   - distancia perpendicular entre centroides <= MERGE_LATERAL_CM
#   - distancia minima entre extremos/centroides <= MERGE_MAX_DIST_CM
# La fusion es transitiva (union-find): si A~B y B~C, los tres se vuelven uno.
ENABLE_MERGE        = True
MERGE_ANGLE_DEG     = 20.0
MERGE_LATERAL_CM    = 36.0
MERGE_MAX_DIST_CM   = 50.0

# === PROYECCION AL ESQUELETO ===
SKELETON_PROXIMITY_TOL_PX = 6.0  # distancia maxima para considerar "encontrado"

# === VISUALIZACION ===
DARK_BG       = True
FLOWER_SIZE   = 4
LINE_WIDTH    = 1.4
BASE_RADIUS   = 3
VARILLA_ALPHA = 0.6              # transparencia de las curvas de varilla

# =====================================================================
#  IMPORTS
# =====================================================================

import os
import re
import json
import numpy as np
import cv2
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


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
    """Busca el PNG del grafo en una lista de carpetas (en orden)."""
    if isinstance(grafos_img_dirs, str):
        grafos_img_dirs = [grafos_img_dirs]
    tried = []
    for d in grafos_img_dirs:
        p = os.path.join(d, graph_image_name)
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontro imagen en ninguna carpeta:\n  "
                            + "\n  ".join(tried))


# =====================================================================
#  2. ESQUELETO -> KD-TREE PARA BUSQUEDA POR PIXEL
# =====================================================================

def estimate_pixels_per_cm(graph_data, assumed_tree_height_cm):
    """
    Calibra px <-> cm midiendo la altura vertical del esqueleto.

    Asume que la extension vertical (max y - min y) de TODOS los pixeles de
    las ramas representa la altura total del arbol y que esa altura es
    ~assumed_tree_height_cm (3-4 m para duraznero adulto). Esto da una escala
    POR IMAGEN, robusta a zoom/distancia de camara.
    """
    ys = []
    for b in graph_data['branches']:
        for px in b['pixels']:
            ys.append(px[0])  # pixels en formato [y, x]
    if len(ys) < 2:
        return 1.0
    height_px = float(max(ys) - min(ys))
    if height_px <= 1.0 or assumed_tree_height_cm <= 0:
        return 1.0
    return height_px / float(assumed_tree_height_cm)


def build_skeleton_kdtree(graph_data):
    """
    Construye un cKDTree con TODOS los pixeles de TODAS las ramas, mas
    un array paralelo que dice a que branch_id pertenece cada pixel.

    Pixeles en JSON estan en formato [y, x] (fila, columna). Convertimos
    a (x, y) para coherencia con coords de flores.
    """
    pts_xy = []
    pix_to_bid = []
    for b in graph_data['branches']:
        for px in b['pixels']:
            y, x = px[0], px[1]
            pts_xy.append((x, y))
            pix_to_bid.append(b['id'])
    pts_xy = np.array(pts_xy, dtype=np.float32)
    pix_to_bid = np.array(pix_to_bid, dtype=np.int32)
    tree = cKDTree(pts_xy)
    return tree, pix_to_bid, pts_xy


# =====================================================================
#  3. CLUSTERING ANISOTROPICO
# =====================================================================

def _dbscan_aniso(F, eps_px, min_samples, lambda_x, lambda_y):
    """DBSCAN sobre coords escaladas X'=X*lambda_x, Y'=Y*lambda_y."""
    F_scaled = F.copy()
    F_scaled[:, 0] *= lambda_x
    F_scaled[:, 1] *= lambda_y
    db = DBSCAN(eps=eps_px, min_samples=min_samples, metric='euclidean')
    return db.fit_predict(F_scaled)


def cluster_flowers(flowers_xy, eps_px, min_samples, lambda_x):
    """
    DBSCAN anisotropico DEPENDIENTE DE LA ALTURA.

    Las varillas de la mitad inferior del arbol no crecen tan verticales como
    las de arriba. Por eso partimos las flores por la mitad de la altura y
    aplicamos una anisotropia distinta a cada mitad:

      - Mitad SUPERIOR (y < y_medio): X' = X * lambda_x, Y' = Y
        -> penaliza separacion horizontal -> favorece clusters VERTICALES.
      - Mitad INFERIOR (y >= y_medio): X' = X, Y' = Y * lambda_x
        -> penaliza separacion vertical -> favorece clusters HORIZONTALES,
           en la MISMA magnitud que la mitad de arriba favorece la vertical.

    y_medio = (y_min + y_max) / 2 sobre TODAS las flores (linea vertical entre
    el pixel de flor mas bajo y el mas alto). Cada mitad se clusteriza con un
    DBSCAN independiente y las etiquetas se reunen con un offset para que no
    colisionen.

    Retorna: array (n_flowers,) con label de cluster (-1 = ruido), en el mismo
    orden que flowers_xy.
    """
    if len(flowers_xy) == 0:
        return np.array([], dtype=np.int32)
    F = np.array(flowers_xy, dtype=np.float64)

    y = F[:, 1]
    y_mid = 0.5 * (float(y.min()) + float(y.max()))
    upper_mask = y < y_mid          # mitad superior (favorece vertical)
    lower_mask = ~upper_mask        # mitad inferior (favorece horizontal)

    labels = np.full(len(F), -1, dtype=np.int32)
    next_label = 0

    for mask, (lx, ly) in ((upper_mask, (lambda_x, 1.0)),
                           (lower_mask, (1.0, lambda_x))):
        if not np.any(mask):
            continue
        sub_labels = _dbscan_aniso(F[mask], eps_px, min_samples, lx, ly)
        # reindexar: ruido (-1) se conserva; clusters validos reciben offset
        out = np.full(sub_labels.shape, -1, dtype=np.int32)
        valid = sub_labels >= 0
        if np.any(valid):
            out[valid] = sub_labels[valid] + next_label
            next_label += int(sub_labels[valid].max()) + 1
        labels[mask] = out

    return labels


# =====================================================================
#  4. AJUSTE DE VARILLA (PCA + parabola opcional, sin trim)
# =====================================================================

def fit_varilla(cluster_xy, use_quadratic_threshold, max_curvature):
    """
    Ajusta una varilla al cluster por regresion lineal de minimos cuadrados
    (OLS), pasando por el centroide:
      - n=1: singleton, eje vertical asumido
      - Si Var(y) >= Var(x): cluster mas vertical -> regreso x = a + b*y
        (slope = Sxy / Syy). Direccion u = (b, 1) normalizada.
      - Si Var(x) > Var(y): cluster mas horizontal -> regreso y = a + b*x
        (slope = Sxy / Sxx). Direccion u = (1, b) normalizada.
      - n>=use_quadratic_threshold: anade curvatura r = c*t^2 (parabola)
        con c acotado a +/- max_curvature.

    Elegir el eje de regresion segun la varianza dominante evita la
    inestabilidad de OLS cuando el cluster es casi paralelo al eje
    independiente (slope tiende a infinito).

    Retorna centroid (2,), u (2,), v (2,), proj_t (n,), proj_r (n,), c.
    Todas las flores se conservan (sin trim).
    """
    P = np.array(cluster_xy, dtype=np.float64)
    n = len(P)
    centroid = P.mean(axis=0)

    if n == 1:
        u = np.array([0.0, -1.0])
        v = np.array([1.0, 0.0])
        proj_t = np.array([0.0])
        proj_r = np.array([0.0])
        return centroid, u, v, proj_t, proj_r, 0.0

    Pc = P - centroid
    Sxx = float(np.sum(Pc[:, 0] ** 2))
    Syy = float(np.sum(Pc[:, 1] ** 2))
    Sxy = float(np.sum(Pc[:, 0] * Pc[:, 1]))

    if Syy >= Sxx:
        # Cluster mas vertical: regreso x sobre y
        if Syy > 1e-12:
            slope = Sxy / Syy
            u = np.array([slope, 1.0])
        else:
            u = np.array([0.0, 1.0])
    else:
        # Cluster mas horizontal: regreso y sobre x
        if Sxx > 1e-12:
            slope = Sxy / Sxx
            u = np.array([1.0, slope])
        else:
            u = np.array([1.0, 0.0])

    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]])
    proj_t = Pc @ u
    proj_r = Pc @ v

    c = 0.0
    if n >= use_quadratic_threshold:
        t2 = proj_t ** 2
        denom = float(np.sum(t2 * t2))
        if denom > 1e-9:
            c = float(np.sum(t2 * proj_r) / denom)
            c = float(np.clip(c, -max_curvature, max_curvature))

    return centroid, u, v, proj_t, proj_r, c


def curve_point(centroid, u, v, c, t):
    """Punto sobre la varilla parametrizado por t."""
    return centroid + t * u + (c * t * t) * v


def curve_tangent(u, v, c, t):
    """Tangente unitaria a la curva en t (dp/dt normalizada)."""
    tg = u + (2.0 * c * t) * v
    n = np.linalg.norm(tg) + 1e-9
    return tg / n


def varilla_extent(proj_t):
    """Rango [t_min, t_max] del cluster sobre el eje principal."""
    if len(proj_t) == 0:
        return 0.0, 0.0
    return float(proj_t.min()), float(proj_t.max())


def endpoints_of_varilla(centroid, u, v, c, t_min, t_max):
    p_a = curve_point(centroid, u, v, c, t_min)
    p_b = curve_point(centroid, u, v, c, t_max)
    return p_a, p_b


def pick_bottom_endpoint(centroid, u, v, c, t_min, t_max):
    """
    Devuelve (p_bottom, p_top, dir_basal). El extremo "basal" es el de mayor Y
    (gravedad apunta a +y en imagen). dir_basal es la tangente a la curva en
    el extremo basal, orientada hacia afuera (hacia donde se proyectara la base).
    """
    p_a = curve_point(centroid, u, v, c, t_min)
    p_b = curve_point(centroid, u, v, c, t_max)

    if p_a[1] >= p_b[1]:
        p_bottom, p_top = p_a, p_b
        t_bottom = t_min
    else:
        p_bottom, p_top = p_b, p_a
        t_bottom = t_max

    if abs(t_max - t_min) < 1e-6:
        # Singleton o cluster degenerado: proyectar hacia +y (gravedad).
        dir_basal = np.array([0.0, 1.0])
        return p_bottom, p_top, dir_basal

    tg = curve_tangent(u, v, c, t_bottom)
    diff = p_bottom - p_top
    if float(tg @ diff) < 0.0:
        tg = -tg
    return p_bottom, p_top, tg


# =====================================================================
#  5. PROYECCION DE LA BASE AL ESQUELETO
# =====================================================================

def project_base_to_skeleton(p_bottom, dir_basal, tree, pix_to_bid,
                             basal_gap_px, max_ext_px, tol_px, step_px=1.0):
    """
    Desde p_bottom extiende a lo largo de dir_basal hasta encontrar el esqueleto.
    Empieza despues de basal_gap_px (zona basal sin flores) y avanza hasta
    max_ext_px en pasos de step_px.

    Retorna dict: {found, base_xy, branch_id, distance_to_skeleton, n_steps}
    """
    found_branch = -1
    found_xy = None
    found_dist = None
    n_steps = 0
    d = float(basal_gap_px)
    d_max = float(basal_gap_px + max_ext_px)
    while d <= d_max:
        candidate = p_bottom + d * dir_basal
        dist, idx = tree.query(candidate, k=1)
        n_steps += 1
        if dist <= tol_px:
            found_xy = tree.data[idx]
            found_branch = int(pix_to_bid[idx])
            found_dist = float(dist)
            break
        d += step_px

    if found_xy is None:
        # fallback: tomar el pixel mas cercano dentro del rango total
        best_dist = np.inf
        best_idx = -1
        d = 0.0
        while d <= d_max:
            candidate = p_bottom + d * dir_basal
            dist, idx = tree.query(candidate, k=1)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
            d += step_px
        if best_idx >= 0:
            return {
                'found': False,
                'base_xy': tuple(tree.data[best_idx].tolist()),
                'branch_id': int(pix_to_bid[best_idx]),
                'distance_to_skeleton': float(best_dist),
                'n_steps': n_steps,
            }
        return {'found': False, 'base_xy': None, 'branch_id': -1,
                'distance_to_skeleton': None, 'n_steps': n_steps}

    return {
        'found': True,
        'base_xy': (float(found_xy[0]), float(found_xy[1])),
        'branch_id': found_branch,
        'distance_to_skeleton': found_dist,
        'n_steps': n_steps,
    }


# =====================================================================
#  6. FUSION DE VARILLAS PARALELAS Y CERCANAS
# =====================================================================

def should_merge(va, vb, hp):
    """
    True si va y vb son lo bastante paralelas y cercanas para fusionarse.
    """
    ua = np.array(va['u'])
    ub = np.array(vb['u'])
    # Angulo (independiente de signo: u y -u son la misma direccion)
    cos_ang = abs(float(np.dot(ua, ub)))
    cos_ang = max(-1.0, min(1.0, cos_ang))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))
    if angle_deg > hp['merge_angle_deg']:
        return False

    # Distancia perpendicular de cb al eje de a (lateral offset)
    ca = np.array(va['centroid'])
    cb = np.array(vb['centroid'])
    v_a = np.array(va['v'])
    lateral = abs(float(np.dot(cb - ca, v_a)))
    if lateral > hp['merge_lateral_px']:
        return False

    # Distancia minima entre extremos/centroides (cercania espacial)
    pts_a = [np.array(va['p_top']), np.array(va['p_bottom']), ca]
    pts_b = [np.array(vb['p_top']), np.array(vb['p_bottom']), cb]
    min_dist = min(float(np.linalg.norm(pa - pb))
                   for pa in pts_a for pb in pts_b)
    if min_dist > hp['merge_max_dist_px']:
        return False

    return True


def _build_varilla_dict(vid, combined_idx, flowers_arr, tree, pix_to_bid, hp,
                       merged_from):
    """Re-ajusta una varilla a partir de un set de indices de flores."""
    cluster_xy = flowers_arr[combined_idx]
    centroid, u, v_perp, proj_t, proj_r, c_curv = fit_varilla(
        cluster_xy, hp['use_quad_threshold'], hp['max_curvature'])
    t_min, t_max = varilla_extent(proj_t)
    length_px = t_max - t_min
    length_cm = length_px / hp['px_per_cm']
    n_in = len(combined_idx)

    p_bottom, p_top, dir_basal = pick_bottom_endpoint(
        centroid, u, v_perp, c_curv, t_min, t_max)
    base_info = project_base_to_skeleton(
        p_bottom, dir_basal, tree, pix_to_bid,
        hp['basal_gap_px'], hp['max_ext_px'], hp['tol_px'])
    rama_id = base_info['branch_id']
    if rama_id < 0:
        _, idx_nn = tree.query(centroid, k=1)
        rama_id = int(pix_to_bid[idx_nn])
        if base_info['base_xy'] is None:
            base_info['base_xy'] = (float(tree.data[idx_nn][0]),
                                    float(tree.data[idx_nn][1]))
            base_info['distance_to_skeleton'] = float(
                np.linalg.norm(tree.data[idx_nn] - centroid))

    if n_in > 1:
        rms = float(np.sqrt(np.mean(proj_r ** 2)))
    else:
        rms = 0.0
    rms_term = float(np.exp(-rms / max(hp['eps_px'], 1.0)))
    if base_info['found']:
        sk_term = float(np.exp(-base_info['distance_to_skeleton'] /
                               hp['tol_px']))
    else:
        sk_term = 0.2
    too_short = length_cm < hp['len_min_cm']
    too_long  = length_cm > hp['len_max_cm']
    bad_count = (n_in < hp['flowers_min']) or (n_in > hp['flowers_max'])
    length_term = 1.0
    if too_short:
        length_term = max(0.2, length_cm / max(hp['len_min_cm'], 1.0))
    elif too_long:
        length_term = max(0.2, hp['len_max_cm'] / max(length_cm, 1.0))
    bio_term = 0.5 if bad_count else 1.0
    confidence = (0.40 * rms_term + 0.40 * sk_term +
                  0.20 * length_term) * bio_term
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        'varilla_id': vid,
        'rama_id': int(rama_id),
        'num_flowers': n_in,
        'length_px': float(length_px),
        'length_cm': float(length_cm),
        'centroid': (float(centroid[0]), float(centroid[1])),
        'u': (float(u[0]), float(u[1])),
        'v': (float(v_perp[0]), float(v_perp[1])),
        'c': float(c_curv),
        't_min': t_min, 't_max': t_max,
        'p_top': (float(p_top[0]), float(p_top[1])),
        'p_bottom': (float(p_bottom[0]), float(p_bottom[1])),
        'base_xy': base_info['base_xy'],
        'base_found': bool(base_info['found']),
        'distance_to_skeleton_px': base_info['distance_to_skeleton'],
        'rms_residual_px': rms,
        'confidence': confidence,
        'flag_too_short': bool(too_short),
        'flag_too_long': bool(too_long),
        'flag_bad_count': bool(bad_count),
        'merged_from': int(merged_from),
        'flower_indices': [int(i) for i in combined_idx],
    }


def merge_close_varillas(varillas, flower_assignments, flowers_arr,
                         tree, pix_to_bid, hp):
    """
    Fusion codiciosa que preserva la propiedad de CLIQUE: un grupo solo
    crece si TODOS los pares entre sus miembros cumplen should_merge. Asi
    se evita que varillas A y C terminen en el mismo grupo si entre ellas
    no se cumple el criterio (aunque ambas lo cumplan con un B intermedio).

    Estrategia:
      1. Calcular matriz de compatibilidad pairwise.
      2. Ordenar pares compatibles por cercania entre centroides (mas cerca
         primero -> prioriza fusiones evidentes).
      3. Para cada par (i, j), intentar fusionar sus grupos actuales: solo
         se fusiona si todos los pares (a, b) con a en G_i y b en G_j son
         compatibles. Caso contrario, se descarta.
    """
    n = len(varillas)
    if n <= 1:
        return varillas, flower_assignments

    # 1. Matriz de compatibilidad
    compatible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(varillas[i], varillas[j], hp):
                compatible[i, j] = True
                compatible[j, i] = True

    # 2. Pares candidatos ordenados por distancia entre centroides
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if compatible[i, j]:
                ca = np.array(varillas[i]['centroid'])
                cb = np.array(varillas[j]['centroid'])
                d = float(np.linalg.norm(cb - ca))
                pairs.append((d, i, j))
    pairs.sort(key=lambda p: p[0])

    # 3. Fusion codiciosa preservando clique
    group_of = list(range(n))                    # group id de cada varilla
    members_of = [[i] for i in range(n)]         # miembros por group id
    for _, i, j in pairs:
        gi = group_of[i]
        gj = group_of[j]
        if gi == gj:
            continue
        clique_ok = True
        for a in members_of[gi]:
            for b in members_of[gj]:
                if not compatible[a, b]:
                    clique_ok = False
                    break
            if not clique_ok:
                break
        if not clique_ok:
            continue
        # Fusionar gj dentro de gi
        for member in members_of[gj]:
            group_of[member] = gi
        members_of[gi].extend(members_of[gj])
        members_of[gj] = []

    final_groups = [g for g in members_of if g]

    if all(len(g) == 1 for g in final_groups):
        return varillas, flower_assignments

    new_varillas = []
    new_assignments = [dict(a) for a in flower_assignments]
    for new_vid, members in enumerate(final_groups):
        if len(members) == 1:
            v = dict(varillas[members[0]])
            v['varilla_id'] = new_vid
            v['merged_from'] = 1
            for fi in v['flower_indices']:
                new_assignments[fi] = {
                    'varilla_id': new_vid,
                    'rama_id': v['rama_id'],
                    'confidence': v['confidence'],
                }
            new_varillas.append(v)
        else:
            combined_idx = []
            for m in members:
                combined_idx.extend(varillas[m]['flower_indices'])
            combined_idx = sorted(set(combined_idx))
            new_v = _build_varilla_dict(new_vid, combined_idx, flowers_arr,
                                        tree, pix_to_bid, hp,
                                        merged_from=len(members))
            for fi in combined_idx:
                new_assignments[fi] = {
                    'varilla_id': new_vid,
                    'rama_id': new_v['rama_id'],
                    'confidence': new_v['confidence'],
                }
            new_varillas.append(new_v)

    return new_varillas, new_assignments


# =====================================================================
#  7. PIPELINE PRINCIPAL
# =====================================================================

def assign_varillas(graph_data, flowers, hp):
    """
    hp: dict con todos los hiperparametros (en pixeles ya convertidos).

    Retorna:
        varillas: lista de dicts (uno por varilla)
        flower_assignments: lista (one per flower) de dicts
                            {varilla_id, rama_id, confidence}
    """
    n_f = len(flowers)
    flower_assignments = [{'varilla_id': -1, 'rama_id': -1, 'confidence': 0.0}
                          for _ in range(n_f)]

    if n_f == 0:
        return [], flower_assignments

    # 1. Clustering (min_samples=1 garantiza que NO hay ruido)
    labels = cluster_flowers(flowers, hp['eps_px'],
                             hp['min_samples'], hp['lambda_x'])

    # 2. Esqueleto
    tree, pix_to_bid, _ = build_skeleton_kdtree(graph_data)

    # 3. Por cluster, ajustar varilla y proyectar base al esqueleto
    varillas = []
    next_vid = 0
    flowers_arr = np.array(flowers, dtype=np.float64)

    unique_labels = sorted(set(labels.tolist()))
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        n_in = len(idx)

        cluster_xy = flowers_arr[idx]
        centroid, u, v, proj_t, proj_r, c_curv = fit_varilla(
            cluster_xy, hp['use_quad_threshold'], hp['max_curvature'])

        t_min, t_max = varilla_extent(proj_t)
        length_px = t_max - t_min
        length_cm = length_px / hp['px_per_cm']

        # Flags informativos (no descartan flores)
        too_short = length_cm < hp['len_min_cm']
        too_long  = length_cm > hp['len_max_cm']
        bad_count = (n_in < hp['flowers_min']) or (n_in > hp['flowers_max'])

        p_bottom, p_top, dir_basal = pick_bottom_endpoint(
            centroid, u, v, c_curv, t_min, t_max)

        base_info = project_base_to_skeleton(
            p_bottom, dir_basal, tree, pix_to_bid,
            hp['basal_gap_px'], hp['max_ext_px'], hp['tol_px'])

        rama_id = base_info['branch_id']

        # Fallback duro: si la proyeccion no encontro rama, asignar la rama
        # mas cercana al centroide del cluster -> garantiza rama_id valido.
        if rama_id < 0:
            _, idx_nn = tree.query(centroid, k=1)
            rama_id = int(pix_to_bid[idx_nn])
            if base_info['base_xy'] is None:
                base_info['base_xy'] = (float(tree.data[idx_nn][0]),
                                        float(tree.data[idx_nn][1]))
                base_info['distance_to_skeleton'] = float(
                    np.linalg.norm(tree.data[idx_nn] - centroid))

        # Confianza heuristica
        if n_in > 1:
            rms = float(np.sqrt(np.mean(proj_r ** 2)))
        else:
            rms = 0.0
        rms_term = float(np.exp(-rms / max(hp['eps_px'], 1.0)))
        if base_info['found']:
            sk_dist = base_info['distance_to_skeleton']
            sk_term = float(np.exp(-sk_dist / hp['tol_px']))
        else:
            sk_term = 0.2
        length_term = 1.0
        if too_short:
            length_term = max(0.2, length_cm / max(hp['len_min_cm'], 1.0))
        elif too_long:
            length_term = max(0.2, hp['len_max_cm'] / max(length_cm, 1.0))
        bio_term = 0.5 if bad_count else 1.0
        confidence = (0.40 * rms_term + 0.40 * sk_term +
                      0.20 * length_term) * bio_term
        confidence = float(np.clip(confidence, 0.0, 1.0))

        vid = next_vid
        next_vid += 1
        for fi in idx:
            flower_assignments[fi] = {
                'varilla_id': vid,
                'rama_id': int(rama_id),
                'confidence': confidence,
            }

        varillas.append({
            'varilla_id': vid,
            'rama_id': int(rama_id),
            'num_flowers': n_in,
            'length_px': float(length_px),
            'length_cm': float(length_cm),
            'centroid': (float(centroid[0]), float(centroid[1])),
            'u': (float(u[0]), float(u[1])),
            'v': (float(v[0]), float(v[1])),
            'c': float(c_curv),
            't_min': t_min, 't_max': t_max,
            'p_top': (float(p_top[0]), float(p_top[1])),
            'p_bottom': (float(p_bottom[0]), float(p_bottom[1])),
            'base_xy': base_info['base_xy'],
            'base_found': bool(base_info['found']),
            'distance_to_skeleton_px': base_info['distance_to_skeleton'],
            'rms_residual_px': rms,
            'confidence': confidence,
            'flag_too_short': bool(too_short),
            'flag_too_long': bool(too_long),
            'flag_bad_count': bool(bad_count),
            'merged_from': 1,
            'flower_indices': [int(i) for i in idx.tolist()],
        })

    # 4. Fusion de varillas paralelas y cercanas
    if hp['enable_merge'] and len(varillas) > 1:
        n_before = len(varillas)
        varillas, flower_assignments = merge_close_varillas(
            varillas, flower_assignments, flowers_arr, tree, pix_to_bid, hp)
        n_after = len(varillas)
        if n_after < n_before:
            print("    Fusion: {0} -> {1} varillas".format(n_before, n_after))

    return varillas, flower_assignments


# =====================================================================
#  7. VISUALIZACION
# =====================================================================

def sample_varilla_curve(v, n_samples=30):
    """Genera arrays 1D xs, ys a lo largo de la curva de la varilla."""
    cx, cy = v['centroid']
    t_min = v['t_min']
    t_max = v['t_max']
    if abs(t_max - t_min) < 1e-6:
        return np.array([cx]), np.array([cy])
    ux, uy = v['u']
    vx, vy = v['v']
    cc = v['c']
    ts = np.linspace(t_min, t_max, n_samples)
    xs = cx + ts * ux + (cc * ts * ts) * vx
    ys = cy + ts * uy + (cc * ts * ts) * vy
    return xs, ys


def visualize(img, graph_data, flowers, varillas, flower_assignments,
              save_path=None):
    n_v = len(varillas)

    txt_col = 'white' if DARK_BG else 'black'
    fig_bg  = '#1e1e1e' if DARK_BG else '#f0f0f0'

    # Color por rama del esqueleto (heredado del JSON)
    skel_colors = {}
    for node in graph_data['nodes']:
        r, g, b = node['color_rgb']
        skel_colors[node['id']] = (r / 255.0, g / 255.0, b / 255.0)

    # Conteo de FLORES por rama madre
    flowers_per_rama = defaultdict(int)
    for a in flower_assignments:
        flowers_per_rama[a['rama_id']] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Tallo recto base_xy -> centroide del cluster.
    for v in varillas:
        rama_id = v['rama_id']
        col = skel_colors.get(rama_id, (1.0, 1.0, 1.0))
        if v['base_xy'] is not None:
            bx, by = v['base_xy']
            cx, cy = v['centroid']
            ax1.plot([bx, cx], [by, cy], '-', color=col,
                     linewidth=LINE_WIDTH, alpha=VARILLA_ALPHA, zorder=4)
            circ = Circle(v['base_xy'], BASE_RADIUS, fill=False,
                          edgecolor=col, linewidth=1.4, zorder=6)
            ax1.add_patch(circ)

    # Flores coloreadas por rama
    for fi, (fx, fy) in enumerate(flowers):
        rama_id = flower_assignments[fi]['rama_id']
        col = skel_colors.get(rama_id, (1.0, 1.0, 1.0))
        ax1.plot(fx, fy, 'o', color=col, markersize=FLOWER_SIZE,
                 markeredgewidth=0.4, markeredgecolor='black', zorder=5)

    title = ("Flores asignadas via varillas a rama madre  |  "
             "Varillas: {n_v}  |  Flores: {tot}").format(
                 n_v=n_v, tot=len(flowers))
    ax1.set_title(title, color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # Panel derecho: FLORES por rama
    ax2.set_facecolor('#2a2a2a' if DARK_BG else 'white')
    sorted_bids = sorted(skel_colors.keys())
    counts = [flowers_per_rama.get(bid, 0) for bid in sorted_bids]
    colors_bar = [skel_colors[bid] for bid in sorted_bids]
    labels_bar = ["R{0}".format(bid) for bid in sorted_bids]
    bars = ax2.barh(labels_bar, counts, color=colors_bar,
                    edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Flores', color=txt_col, fontsize=10)
    ax2.set_title('Flores por rama', color=txt_col, fontsize=11)
    ax2.tick_params(colors=txt_col)
    ax2.spines['bottom'].set_color(txt_col)
    ax2.spines['left'].set_color(txt_col)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(count), va='center', color=txt_col, fontsize=9)

    plt.tight_layout()
    if save_path:
        import os as _os
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig_bg, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
#  8. RUN POR IMAGEN
# =====================================================================

def build_hyperparams(px_per_cm):
    px_per_cm = float(px_per_cm)
    return {
        'px_per_cm':          px_per_cm,
        'eps_px':             DBSCAN_EPS_CM * px_per_cm,
        'min_samples':        DBSCAN_MIN_SAMPLES,
        'lambda_x':           ANISOTROPY_LAMBDA,
        'use_quad_threshold': USE_QUADRATIC_THRESHOLD,
        'max_curvature':      MAX_CURVATURE_COEF,
        'len_min_cm':         VARILLA_LENGTH_MIN_CM,
        'len_max_cm':         VARILLA_LENGTH_MAX_CM,
        'flowers_min':        FLOWERS_PER_VARILLA_MIN,
        'flowers_max':        FLOWERS_PER_VARILLA_MAX,
        'basal_gap_px':       BASAL_GAP_CM * px_per_cm,
        'max_ext_px':         MAX_BASAL_EXTENSION_CM * px_per_cm,
        'tol_px':             SKELETON_PROXIMITY_TOL_PX,
        'enable_merge':       ENABLE_MERGE,
        'merge_angle_deg':    MERGE_ANGLE_DEG,
        'merge_lateral_px':   MERGE_LATERAL_CM * px_per_cm,
        'merge_max_dist_px':  MERGE_MAX_DIST_CM * px_per_cm,
    }


def compute_varilla_data(graph_json_path):
    """
    Corre TODO el pipeline de varillas sobre un grafo y DEVUELVE el resultado
    como un dict (mismo formato que el JSON de varilla), sin mostrar plot ni
    escribir en disco. Es la función que usa el pipeline automático / supremo.

    Devuelve (varilla_data, contexto) donde contexto = (img, graph_data,
    flowers, varillas, flower_assignments) por si se quiere visualizar después.
    Si no hay flores, devuelve (None, None).
    """
    graph_data = load_graph_json(graph_json_path)
    image_name = graph_data['image']

    flower_json = auto_detect_flower_json(image_name, JSON_FLORES_DIR)
    flowers     = load_flowers(flower_json)
    img_path    = auto_detect_graph_image(image_name, GRAFOS_IMG_DIRS)
    img         = cv2.imread(img_path)

    print("    Ramas: {0}  |  Flores: {1}".format(
        len(graph_data['nodes']), len(flowers)))

    if not flowers:
        print("    [SKIP] Sin flores.")
        return None, None

    if PIXELS_PER_CM_OVERRIDE is not None:
        px_per_cm = float(PIXELS_PER_CM_OVERRIDE)
        scale_src = "override"
    else:
        px_per_cm = estimate_pixels_per_cm(graph_data, ASSUMED_TREE_HEIGHT_CM)
        scale_src = "auto (altura esqueleto / {0:.0f} cm)".format(
            ASSUMED_TREE_HEIGHT_CM)
    print("    Escala: {0:.2f} px/cm  [{1}]".format(px_per_cm, scale_src))

    hp = build_hyperparams(px_per_cm)
    varillas, flower_assignments = assign_varillas(graph_data, flowers, hp)
    n_quad = sum(1 for v in varillas if abs(v.get('c', 0.0)) > 1e-7)
    print("    Varillas: {0}  ({1} con curvatura)".format(
        len(varillas), n_quad))

    # Serializar al formato del JSON de varilla
    flowers_out = []
    for fi, (fx, fy) in enumerate(flowers):
        a = flower_assignments[fi]
        flowers_out.append({
            'flower_id': fi,
            'x': float(fx), 'y': float(fy),
            'varilla_id': a['varilla_id'],
            'rama_id': a['rama_id'],
            'confidence': float(a['confidence']),
        })
    n_orphan = sum(1 for a in flower_assignments if a['varilla_id'] < 0)
    varilla_data = {
        'image': image_name,
        'n_flowers': len(flowers),
        'n_varillas': len(varillas),
        'n_orphan_flowers': n_orphan,
        'varillas': varillas,
        'flowers': flowers_out,
    }
    contexto = (img, graph_data, flowers, varillas, flower_assignments)
    return varilla_data, contexto


def run_one(graph_json_path, show_plot=True):
    """Computa las varillas y, si show_plot, muestra el plot (uso manual)."""
    varilla_data, contexto = compute_varilla_data(graph_json_path)
    if varilla_data is None:
        return None
    if show_plot:
        img, graph_data, flowers, varillas, flower_assignments = contexto
        visualize(img, graph_data, flowers, varillas, flower_assignments)
        print("    [OK] plot mostrado")
    return varilla_data


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    json_path = os.path.join(GRAPH_JSON_DIR, IMAGE_NAME)
    if not os.path.exists(json_path):
        raise FileNotFoundError("No se encontro el JSON: " + json_path)

    print("[INFO] Procesando " + os.path.basename(json_path) + "\n")
    run_one(json_path)
