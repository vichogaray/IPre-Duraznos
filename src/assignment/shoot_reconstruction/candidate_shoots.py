"""


Mecanismos (modulares, intercambiables):
  A. cluster_flowers        -> DBSCAN anisotropico (elipse achatada, favor vertical)
  B. fit_cluster_segment    -> OLS por cluster, ACOTADO a un segmento [ext1, ext2]
  C. shoot_rays             -> desde cada punto del tronco dispara rayos
                              (angulo vs. normal local del tronco, 0-90 desde 45)
  D. connect_rays_to_clusters -> impacto rayo-segmento (extremo basal preferido);
                              al tocar deja de disparar en ese punto; si varios
                              rayos llegan al extremo de un cluster gana el MAS
                              COLINEAL con el eje del cluster; fallback a la rama
                              mas cercana (nunca se pierden flores).
  E. smooth_varilla         -> spline/Bezier que une rayo + segmento en una curva.
  F. discretizacion         -> parametros consumidos por C y D (stride, paso
                              angular, largo max del rayo, muestreo de la curva).

Input : UNA imagen (IMAGE_NAME, JSON de grafo).
Output: un plot de matplotlib. NO guarda ninguna imagen ni JSON.

"""

# =====================================================================
#  CONFIGURACION
# =====================================================================

GRAPH_JSON_DIR  = r"C:\Users\vgara\OneDrive\Desktop\IPre\grafos json"
GRAFOS_IMG_DIRS = [r"C:\Users\vgara\OneDrive\Desktop\IPre\Grafos"]
JSON_FLORES_DIR = r"C:\Users\vgara\OneDrive\Desktop\IPre\json flores"

# Imagen a procesar: nombre del JSON de grafo dentro de GRAPH_JSON_DIR
IMAGE_NAME = "imgs_frame1_00000_graph.json"

# === ESCALA (calibracion px <-> cm POR IMAGEN) ===
ASSUMED_TREE_HEIGHT_CM = 350.0
# Escala forzada: la calibracion por altura del esqueleto subestima (~2x) porque
# el esqueleto visible NO abarca un arbol de 350cm. 4.3 px/cm se deduce del
# espaciamiento mediano entre flores (~12.8 px) tomado como 1 internodo (~3 cm).
PIXELS_PER_CM_OVERRIDE = 4.3

# === A: CLUSTERING DIRECCIONAL (region growing) ===
# Un cluster es valido si las flores forman una LINEA en CUALQUIER orientacion
# (vertical, diagonal o chueca), no solo vertical. Se penaliza solo lo que se
# desvia del eje LOCAL del grupo, que se actualiza a medida que el cluster crece.
#   - GROW_RADIUS_CM   : que tan lejos buscar la siguiente flor candidata.
#   - GROW_ANGLE_DEG   : desvio angular maximo respecto al eje actual del grupo.
#   - SEED_RADIUS_CM   : distancia maxima del par semilla inicial.
GROW_RADIUS_CM = 7.0    # ~2 internodos (re-tuneado con escala correcta 4.3)
GROW_ANGLE_DEG = 18.0   # tolerancia de "alineado" (menor = clusters mas rectos)
SEED_RADIUS_CM = 6.0    # par semilla: dos flores mas cercanas que esto

# (legacy DBSCAN, ya no se usa por defecto; se conserva por si se quiere volver)
DBSCAN_EPS_CM      = 7.0
DBSCAN_MIN_SAMPLES = 1
ANISOTROPY_LAMBDA  = 1.5

# === C: DISPARO DE RAYOS DESDE EL TRONCO ===
# El angulo se mide respecto a la NORMAL LOCAL del tronco (varilla sale hacia
# afuera). Se barre 0-90 grados empezando en 45 (mas frecuente) y abriendo a
# ambos lados de la normal. Las varillas crecen HACIA ARRIBA (nunca hacia +y).
SHOOT_ANGLE_CENTER_DEG = 45.0   # angulo de arranque (mas frecuente)
SHOOT_ANGLE_MIN_DEG    = 0.0    # limite inferior del barrido
SHOOT_ANGLE_MAX_DEG    = 90.0   # limite superior del barrido

# === D: CONEXION / COMPETENCIA ===
# Penalizacion de colinealidad: gana el rayo cuya direccion sea MAS ALINEADA con
# el eje del segmento del cluster (penaliza llegar perpendicular).
IMPACT_TOL_PX          = 4.0    # tolerancia para considerar "toca el segmento"
BASAL_PREFERENCE_PX    = 30.0   # ventana cerca del extremo basal del cluster
                                # donde el impacto es preferido (suma bonus)

# === FUSION DE VARILLAS PARALELAS Y CERCANAS (portado de v1) ===
# Tras el clustering (A) y el ajuste de segmento (B), se fusionan clusters cuyos
# segmentos cumplen LOS TRES:
#   - angulo entre ejes principales <= MERGE_ANGLE_DEG
#   - distancia perpendicular entre centroides <= MERGE_LATERAL_CM
#   - distancia minima entre extremos/centroides <= MERGE_MAX_DIST_CM
# La fusion preserva la propiedad de CLIQUE (si A~B y B~C, solo se unen los tres
# si tambien A~C). Se re-ajusta el segmento del cluster fusionado antes de C/D.
ENABLE_MERGE        = True
MERGE_ANGLE_DEG     = 20.0
MERGE_LATERAL_CM    = 36.0
MERGE_MAX_DIST_CM   = 50.0

# === E: SUAVIZADO ===
SMOOTH_TENSION   = 0.25    # 0 = recta (origin->apical), 1 = codo duro en el impacto
CURVE_N_SAMPLES  = 30      # puntos de la curva suavizada (dibujo)

# === F: DISCRETIZACION (valores iniciales, sujetos a cambio) ===
TRUNK_STRIDE_PX    = 5.0    # submuestreo de puntos de disparo en el esqueleto
ANGLE_STEP_DEG     = 5.0    # paso del barrido angular (~19 rayos por punto)
MAX_RAY_LEN_CM     = 80.0   # largo maximo que recorre un rayo buscando impacto
NORMAL_NEIGHBOR_PX = 6.0    # radio para estimar la tangente/normal local del tronco

# === VISUALIZACION ===
DARK_BG       = True
FLOWER_SIZE   = 4
LINE_WIDTH    = 1.4
BASE_RADIUS   = 3
VARILLA_ALPHA = 0.7

# === DEBUG: correr solo hasta un mecanismo ===
# 'A'  -> corre solo el clustering y colorea flores por TAMANO de cluster
#         (1 flor = azul, mas flores = mas rojo).
# None -> pipeline completo (A..E).
STOP_AFTER = None

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

from scipy.spatial import cKDTree


# =====================================================================
#  CARGA DE DATOS (reutilizado de v1)
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
    raise FileNotFoundError("No se encontro imagen en ninguna carpeta:\n  "
                            + "\n  ".join(tried))


# =====================================================================
#  ESQUELETO -> ESCALA + KD-TREE (reutilizado de v1)
# =====================================================================

# La calibracion px <-> cm vive en src/common/geometry.py para que exista una
# sola definicion compartida por todos los metodos. Se anade la raiz del
# repositorio al path para poder ejecutar este archivo directamente (F5).
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.common.geometry import estimate_pixels_per_cm


def build_skeleton_arrays(graph_data):
    """
    Devuelve:
      pts_xy     (M,2)  todos los pixeles de ramas en (x, y)
      pix_to_bid (M,)   branch_id de cada pixel
      tree              cKDTree sobre pts_xy
    (Pixeles en JSON estan en [y, x]; convertimos a (x, y).)
    """
    pts_xy = []
    pix_to_bid = []
    for b in graph_data['branches']:
        for px in b['pixels']:
            y, x = px[0], px[1]
            pts_xy.append((x, y))
            pix_to_bid.append(b['id'])
    pts_xy = np.array(pts_xy, dtype=np.float64)
    pix_to_bid = np.array(pix_to_bid, dtype=np.int32)
    tree = cKDTree(pts_xy)
    return pts_xy, pix_to_bid, tree


# =====================================================================
#  MECANISMO A: CLUSTERING DIRECCIONAL (region growing)
# =====================================================================

def cluster_flowers_directional(flowers_xy, grow_radius_px, grow_angle_deg,
                                seed_radius_px):
    """
    Agrupa flores en lineas de CUALQUIER orientacion (vertical, diagonal, chueca)
    por crecimiento de regiones direccional:

      1. Semilla: el par de flores no asignadas mas cercano (< seed_radius_px).
         Define el eje inicial del grupo.
      2. Crecer: repetidamente se busca, entre las flores no asignadas dentro de
         grow_radius_px de CUALQUIER miembro, aquella cuyo vector hacia el miembro
         mas cercano este ALINEADO con el eje del grupo (desvio <= grow_angle_deg).
         Se anade la mejor candidata y se RE-ESTIMA el eje (PCA del grupo).
      3. Cuando no hay candidata valida, el cluster se cierra. Las flores que
         nunca formaron par/semilla quedan como singletons (su propio cluster).

    La anisotropia es LOCAL y orientada al grupo: no se favorece el eje vertical,
    se favorece la COLINEALIDAD con el eje propio del cluster en formacion.

    Retorna labels (n_flowers,) >= 0 (sin ruido: toda flor cae en un cluster).
    """
    F = np.array(flowers_xy, dtype=np.float64)
    n = len(F)
    if n == 0:
        return np.array([], dtype=np.int32)

    tree = cKDTree(F)
    labels = np.full(n, -1, dtype=np.int32)
    cos_tol = np.cos(np.radians(grow_angle_deg))

    def group_axis(member_idx):
        """Eje principal (PCA) del grupo; para n=2 es la recta entre ambos."""
        pts = F[member_idx]
        c = pts.mean(axis=0)
        Pc = pts - c
        cov = Pc.T @ Pc
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return axis / (np.linalg.norm(axis) + 1e-12)

    # Orden de semillas: por distancia al vecino mas cercano (pares mas apretados
    # primero -> las varillas evidentes se forman antes).
    nn_dist, _ = tree.query(F, k=2)
    seed_order = np.argsort(nn_dist[:, 1])

    next_label = 0
    for s in seed_order:
        if labels[s] != -1:
            continue
        # buscar el vecino libre mas cercano para formar la semilla
        idxs = tree.query_ball_point(F[s], seed_radius_px)
        cand = [(np.linalg.norm(F[i] - F[s]), i) for i in idxs
                if i != s and labels[i] == -1]
        if not cand:
            # sin par: singleton
            labels[s] = next_label
            next_label += 1
            continue
        cand.sort()
        partner = cand[0][1]
        members = [s, partner]
        labels[s] = next_label
        labels[partner] = next_label
        axis = group_axis(members)

        # crecer
        while True:
            best = None  # (alineacion, distancia, idx)
            for m in members:
                for i in tree.query_ball_point(F[m], grow_radius_px):
                    if labels[i] != -1:
                        continue
                    d = F[i] - F[m]
                    dist = np.linalg.norm(d)
                    if dist < 1e-9:
                        continue
                    align = abs(float(np.dot(d / dist, axis)))  # 1=alineado
                    if align < cos_tol:
                        continue
                    # preferir mas alineado y, a igualdad, mas cercano
                    key = (align, -dist)
                    if best is None or key > best[0]:
                        best = (key, i)
            if best is None:
                break
            new_i = best[1]
            labels[new_i] = next_label
            members.append(new_i)
            axis = group_axis(members)

        next_label += 1

    return labels


def cluster_flowers(flowers_xy, hp):
    """Wrapper del Mecanismo A (firma estable para el pipeline)."""
    return cluster_flowers_directional(
        flowers_xy, hp['grow_radius_px'], hp['grow_angle_deg'],
        hp['seed_radius_px'])


# =====================================================================
#  MECANISMO B: REGRESION LINEAL -> SEGMENTO ACOTADO A LOS EXTREMOS
# =====================================================================

def fit_cluster_segment(cluster_xy):
    """
    Ajusta una recta OLS al cluster (eligiendo eje de regresion segun varianza
    dominante para evitar slope->inf) y la ACOTA al segmento entre los dos
    extremos del cluster proyectados sobre esa recta.

    Retorna dict:
      centroid (2,), dir (2,) unitaria,
      p_basal (2,)  extremo de mayor Y (lado tronco / gravedad),
      p_apical(2,)  extremo de menor Y,
      t_min, t_max  (proyecciones escalares sobre dir),
      rms           residual perpendicular (px)
    """
    P = np.array(cluster_xy, dtype=np.float64)
    n = len(P)
    centroid = P.mean(axis=0)

    if n == 1:
        d = np.array([0.0, -1.0])  # eje vertical hacia arriba por defecto
        return {
            'centroid': centroid, 'dir': d,
            'p_basal': centroid.copy(), 'p_apical': centroid.copy(),
            't_min': 0.0, 't_max': 0.0, 'rms': 0.0,
        }

    Pc = P - centroid
    Sxx = float(np.sum(Pc[:, 0] ** 2))
    Syy = float(np.sum(Pc[:, 1] ** 2))
    Sxy = float(np.sum(Pc[:, 0] * Pc[:, 1]))

    if Syy >= Sxx:  # cluster mas vertical: regreso x sobre y
        slope = Sxy / Syy if Syy > 1e-12 else 0.0
        d = np.array([slope, 1.0])
    else:           # cluster mas horizontal: regreso y sobre x
        slope = Sxy / Sxx if Sxx > 1e-12 else 0.0
        d = np.array([1.0, slope])
    d = d / (np.linalg.norm(d) + 1e-12)
    perp = np.array([-d[1], d[0]])

    proj_t = Pc @ d
    proj_r = Pc @ perp
    t_min, t_max = float(proj_t.min()), float(proj_t.max())
    rms = float(np.sqrt(np.mean(proj_r ** 2)))

    p_end_a = centroid + t_min * d
    p_end_b = centroid + t_max * d
    # Extremo basal = mayor Y (gravedad apunta a +y en la imagen)
    if p_end_a[1] >= p_end_b[1]:
        p_basal, p_apical = p_end_a, p_end_b
    else:
        p_basal, p_apical = p_end_b, p_end_a

    return {
        'centroid': centroid, 'dir': d,
        'p_basal': p_basal, 'p_apical': p_apical,
        't_min': t_min, 't_max': t_max, 'rms': rms,
    }


# =====================================================================
#  FUSION DE CLUSTERS PARALELOS Y CERCANOS (portado de v1)
# =====================================================================

def should_merge(sa, sb, hp):
    """
    True si los segmentos sa y sb (de fit_cluster_segment) son lo bastante
    paralelos y cercanos para fusionarse. Mismos criterios y parametros que v1,
    adaptados a la nomenclatura de segmentos de v2 (dir, p_basal, p_apical).
    """
    ua = np.asarray(sa['dir'], dtype=np.float64)
    ub = np.asarray(sb['dir'], dtype=np.float64)
    # Angulo (independiente de signo: u y -u son la misma direccion)
    cos_ang = abs(float(np.dot(ua, ub)))
    cos_ang = max(-1.0, min(1.0, cos_ang))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))
    if angle_deg > hp['merge_angle_deg']:
        return False

    # Distancia perpendicular del centroide de b al eje de a (lateral offset)
    ca = np.asarray(sa['centroid'], dtype=np.float64)
    cb = np.asarray(sb['centroid'], dtype=np.float64)
    v_a = np.array([-ua[1], ua[0]])           # perpendicular al eje de a
    lateral = abs(float(np.dot(cb - ca, v_a)))
    if lateral > hp['merge_lateral_px']:
        return False

    # Distancia minima entre extremos/centroides (cercania espacial)
    pts_a = [np.asarray(sa['p_apical']), np.asarray(sa['p_basal']), ca]
    pts_b = [np.asarray(sb['p_apical']), np.asarray(sb['p_basal']), cb]
    min_dist = min(float(np.linalg.norm(pa - pb))
                   for pa in pts_a for pb in pts_b)
    if min_dist > hp['merge_max_dist_px']:
        return False

    return True


def merge_close_clusters(cluster_idx, segments, flowers_arr, hp):
    """
    Fusion codiciosa que preserva la propiedad de CLIQUE: un grupo solo crece si
    TODOS los pares entre sus miembros cumplen should_merge (si A~B y B~C, los
    tres solo se unen si tambien A~C). Identica logica a v1.

    Recibe los clusters como lista de arrays de indices de flores y sus segmentos.
    Retorna (new_cluster_idx, new_segments) con los clusters ya fusionados y sus
    segmentos re-ajustados.
    """
    n = len(segments)
    if n <= 1:
        return cluster_idx, segments

    # 1. Matriz de compatibilidad
    compatible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(segments[i], segments[j], hp):
                compatible[i, j] = True
                compatible[j, i] = True

    # 2. Pares candidatos ordenados por distancia entre centroides
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if compatible[i, j]:
                ca = np.asarray(segments[i]['centroid'])
                cb = np.asarray(segments[j]['centroid'])
                d = float(np.linalg.norm(cb - ca))
                pairs.append((d, i, j))
    pairs.sort(key=lambda p: p[0])

    # 3. Fusion codiciosa preservando clique
    group_of = list(range(n))
    members_of = [[i] for i in range(n)]
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
        for member in members_of[gj]:
            group_of[member] = gi
        members_of[gi].extend(members_of[gj])
        members_of[gj] = []

    final_groups = [g for g in members_of if g]
    if all(len(g) == 1 for g in final_groups):
        return cluster_idx, segments

    new_cluster_idx = []
    new_segments = []
    for members in final_groups:
        combined = np.concatenate([cluster_idx[m] for m in members])
        combined = np.unique(combined)
        new_cluster_idx.append(combined)
        new_segments.append(fit_cluster_segment(flowers_arr[combined]))
    return new_cluster_idx, new_segments


# =====================================================================
#  MECANISMO C (soporte): NORMAL LOCAL DEL TRONCO
# =====================================================================

def estimate_local_normal(point_xy, tree, pts_xy, neighbor_px):
    """
    Estima la tangente local del esqueleto en point_xy via PCA de los vecinos
    dentro de neighbor_px, y devuelve la NORMAL unitaria (perpendicular a la
    tangente) orientada hacia ARRIBA (componente y negativa: las varillas
    crecen contra la gravedad).
    """
    idx = tree.query_ball_point(point_xy, neighbor_px)
    if len(idx) < 2:
        normal = np.array([0.0, -1.0])  # default: hacia arriba
        return normal
    nbrs = pts_xy[idx]
    nbrs_c = nbrs - nbrs.mean(axis=0)
    cov = nbrs_c.T @ nbrs_c
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, np.argmax(eigvals)]  # mayor varianza = direccion rama
    normal = np.array([-tangent[1], tangent[0]])
    if normal[1] > 0:        # forzar componente y negativa (hacia arriba)
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-12)


def _rotate(vec, deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


# =====================================================================
#  MECANISMO C: DISPARO DE RAYOS DESDE EL TRONCO
# =====================================================================

def shoot_directions(normal, center_deg, min_deg, max_deg, step_deg):
    """
    Genera direcciones de rayo rotando la normal local. Barre desde center_deg
    abriendo simetricamente a ambos lados, recortado a [min_deg, max_deg] medido
    como desviacion |angulo| respecto a la normal. Solo se conservan rayos que
    apuntan hacia arriba (componente y <= 0).

    Retorna lista de (dir_unit, dev_deg) ordenada por |dev| creciente (empieza
    en el angulo mas frecuente y se abre).
    """
    # Barrido simetrico alrededor de center_deg, recortado a [min_deg, max_deg].
    half = max(max_deg - center_deg, center_deg - min_deg)
    offsets = [0.0]
    o = step_deg
    while o <= half + 1e-9:
        offsets.append(o)
        offsets.append(-o)
        o += step_deg

    dirs = []
    for off in offsets:
        ang_from_normal = center_deg + off
        if ang_from_normal < min_deg - 1e-9 or ang_from_normal > max_deg + 1e-9:
            continue
        # Rotar la normal +ang y -ang (dos lados del tronco)
        for sgn in (+1.0, -1.0):
            d = _rotate(normal, sgn * ang_from_normal)
            if d[1] > 1e-6:        # descarta rayos que apuntan hacia abajo
                continue
            dirs.append((d / (np.linalg.norm(d) + 1e-12),
                         abs(ang_from_normal)))
    # ordenar por cercania al angulo central (45 -> mas frecuente primero)
    dirs.sort(key=lambda t: abs(t[1] - center_deg))
    return dirs


def shoot_rays(pts_xy, pix_to_bid, tree, hp):
    """
    Submuestrea puntos del esqueleto (stride) y, en cada uno, genera los rayos
    candidatos (origen, dir, branch_id). El barrido angular se evalua despues en
    connect_rays_to_clusters (que decide cuando "dejar de disparar").

    Retorna lista de dicts: {origin (2,), dirs [(dir, dev_deg)], branch_id}.
    """
    stride = hp['trunk_stride_px']
    # submuestreo simple: recorrer pts y quedarse con puntos separados >= stride
    chosen_idx = []
    chosen_pts = []
    for i in range(len(pts_xy)):
        p = pts_xy[i]
        ok = True
        # comparar contra los ya elegidos cercanos via tree de elegidos seria
        # mas rapido; para una sola imagen, este filtro por grilla basta:
        key = (int(p[0] // stride), int(p[1] // stride))
        chosen_pts.append(key)
    # quedarnos con un representante por celda de grilla (stride x stride)
    seen = {}
    rays = []
    for i in range(len(pts_xy)):
        p = pts_xy[i]
        key = (int(p[0] // stride), int(p[1] // stride))
        if key in seen:
            continue
        seen[key] = True
        normal = estimate_local_normal(p, tree, pts_xy, hp['normal_neighbor_px'])
        dirs = shoot_directions(normal,
                                hp['angle_center_deg'],
                                hp['angle_min_deg'],
                                hp['angle_max_deg'],
                                hp['angle_step_deg'])
        rays.append({
            'origin': p.copy(),
            'dirs': dirs,
            'branch_id': int(pix_to_bid[i]),
        })
    return rays


# =====================================================================
#  MECANISMO D: CONEXION RAYO -> SEGMENTO + COMPETENCIA + FALLBACK
# =====================================================================

def ray_segment_intersection(o, d, a, b, max_len):
    """
    Interseccion analitica del rayo o + t*d (t>=0, t<=max_len) con el segmento
    [a, b]. Retorna (t_ray, point, s) o None.  s in [0,1] = posicion sobre [a,b].
    """
    ab = b - a
    denom = d[0] * (-ab[1]) - d[1] * (-ab[0])
    if abs(denom) < 1e-12:
        return None  # paralelos
    diff = a - o
    t = (diff[0] * (-ab[1]) - diff[1] * (-ab[0])) / denom
    s = (d[0] * diff[1] - d[1] * diff[0]) / denom
    if t < 0 or t > max_len:
        return None
    if s < 0 or s > 1:
        return None
    return t, o + t * d, s


def connect_rays_to_clusters(rays, segments, hp):
    """
    Para cada cluster (segmento B) busca el mejor rayo ganador.

    Reglas:
      - Un rayo "toca" si intersecta el segmento (tol implicita por la geometria;
        IMPACT_TOL_PX se usa como holgura extendiendo el segmento).
      - Impacto cerca del extremo BASAL del cluster se prefiere (bonus).
      - Al lograr impacto, ese punto del tronco "deja de disparar" (probamos sus
        direcciones en orden y nos quedamos con la primera que toca).
      - Competencia: gana el rayo MAS COLINEAL con el eje del segmento (penaliza
        llegar perpendicular). score = colinealidad + bonus_basal.
      - Fallback: si ningun rayo toca un cluster, se conecta a la rama mas cercana
        a su extremo basal (nunca se pierden flores).

    Retorna lista de varillas ganadoras (una por cluster):
      {cluster_id, branch_id, origin, impact, s_on_segment, score,
       colinearity, basal_bonus, found}
    """
    max_len = hp['max_ray_len_px']
    tol = hp['impact_tol_px']

    winners = []
    for cid, seg in enumerate(segments):
        a = seg['p_basal']
        b = seg['p_apical']
        seg_dir = b - a
        seg_len = np.linalg.norm(seg_dir) + 1e-12
        seg_u = seg_dir / seg_len
        # extender el segmento por tol en ambos extremos (holgura)
        a_ext = a - seg_u * tol
        b_ext = b + seg_u * tol

        best = None
        for ray in rays:
            o = ray['origin']
            for d, dev in ray['dirs']:
                hit = ray_segment_intersection(o, d, a_ext, b_ext, max_len)
                if hit is None:
                    continue
                t_ray, point, s = hit
                # colinealidad: |d . seg_u| -> 1 alineado, 0 perpendicular
                colin = abs(float(np.dot(d, seg_u)))
                # bonus basal: 1 en el extremo basal, decae con la distancia
                dist_to_basal = np.linalg.norm(point - a)
                basal_bonus = float(np.exp(-dist_to_basal /
                                           max(hp['basal_pref_px'], 1.0)))
                score = colin + 0.5 * basal_bonus
                cand = {
                    'cluster_id': cid,
                    'branch_id': ray['branch_id'],
                    'origin': o.copy(),
                    'impact': point,
                    's_on_segment': s,
                    'score': score,
                    'colinearity': colin,
                    'basal_bonus': basal_bonus,
                    'found': True,
                }
                if best is None or cand['score'] > best['score']:
                    best = cand
                # "deja de disparar": tomamos la primera direccion que toca
                # (las dirs ya van ordenadas por cercania a 45). Pero seguimos
                # evaluando otros puntos del tronco para la competencia global.
                break

        if best is None:
            # Fallback: rama mas cercana al extremo basal del cluster
            best = {
                'cluster_id': cid,
                'branch_id': None,   # se resuelve fuera (necesita tree)
                'origin': None,
                'impact': a.copy(),
                's_on_segment': 0.0,
                'score': 0.0,
                'colinearity': 0.0,
                'basal_bonus': 0.0,
                'found': False,
            }
        winners.append(best)
    return winners


def resolve_fallbacks(winners, segments, pix_to_bid, tree, pts_xy):
    """Para varillas sin impacto, asigna la rama mas cercana al extremo basal."""
    for w in winners:
        if w['found']:
            continue
        seg = segments[w['cluster_id']]
        _, idx = tree.query(seg['p_basal'], k=1)
        w['branch_id'] = int(pix_to_bid[idx])
        w['origin'] = pts_xy[idx].copy()
        w['impact'] = seg['p_basal'].copy()
    return winners


# =====================================================================
#  MECANISMO E: SUAVIZADO (curva que une rayo + segmento)
# =====================================================================

def smooth_varilla(winner, segment, tension, n_samples):
    """
    Construye una curva suave que va del origen en el tronco, redondea el codo en
    el punto de impacto, y sigue el eje del cluster hasta el extremo apical.

    Usa una Bezier cuadratica por tramos con el punto de impacto como control:
      P0 = origin (tronco)
      P1 = impact (control = codo a suavizar)
      P2 = p_apical (punta de la varilla)
    La tension mezcla el control entre la cuerda P0-P2 (recta) y el codo P1:
      tension = 0 -> recta origin->apical ; tension = 1 -> codo duro en impact.

    Retorna (xs, ys).
    """
    p0 = np.array(winner['origin'], dtype=np.float64)
    p1 = np.array(winner['impact'], dtype=np.float64)
    p2 = np.array(segment['p_apical'], dtype=np.float64)

    # tension: interpolar el control entre la cuerda recta y el codo real.
    #   tension = 0 -> ctrl = punto medio de la cuerda -> Bezier = RECTA origin->apical
    #   tension = 1 -> ctrl = impact -> codo duro (maxima curvatura)
    chord_mid = 0.5 * (p0 + p2)
    ctrl = (1.0 - tension) * chord_mid + tension * p1

    ts = np.linspace(0.0, 1.0, n_samples)
    one = (1.0 - ts)
    xs = one * one * p0[0] + 2 * one * ts * ctrl[0] + ts * ts * p2[0]
    ys = one * one * p0[1] + 2 * one * ts * ctrl[1] + ts * ts * p2[1]
    return xs, ys


# =====================================================================
#  ASIGNACION FLOR -> VARILLA -> RAMA (orquestacion A->B->C->D->E)
# =====================================================================

def assign_varillas_v2(graph_data, flowers, hp):
    """
    Conecta los mecanismos. Retorna:
      varillas: lista de dicts (una por cluster)  con curva suavizada y rama madre
      flower_assignments: lista (una por flor) {varilla_id, rama_id}
    """
    n_f = len(flowers)
    flower_assignments = [{'varilla_id': -1, 'rama_id': -1} for _ in range(n_f)]
    if n_f == 0:
        return [], flower_assignments

    flowers_arr = np.array(flowers, dtype=np.float64)
    pts_xy, pix_to_bid, tree = build_skeleton_arrays(graph_data)

    # A. Clustering
    labels = cluster_flowers(flowers, hp)
    unique_labels = sorted(set(labels.tolist()))
    cluster_idx = [np.where(labels == lab)[0] for lab in unique_labels]

    # B. Segmento por cluster
    segments = [fit_cluster_segment(flowers_arr[idx]) for idx in cluster_idx]

    # B.5 Fusion de clusters paralelos y cercanos (portado de v1)
    if hp['enable_merge'] and len(segments) > 1:
        n_before = len(segments)
        cluster_idx, segments = merge_close_clusters(
            cluster_idx, segments, flowers_arr, hp)
        if len(segments) < n_before:
            print("    Fusion: {0} -> {1} clusters".format(
                n_before, len(segments)))

    # C. Disparo de rayos desde el tronco
    rays = shoot_rays(pts_xy, pix_to_bid, tree, hp)

    # D. Conexion + competencia + fallback
    winners = connect_rays_to_clusters(rays, segments, hp)
    winners = resolve_fallbacks(winners, segments, pix_to_bid, tree, pts_xy)

    # E. Suavizado + ensamblar varillas y asignar flores
    varillas = []
    for cid, (seg, win) in enumerate(zip(segments, winners)):
        xs, ys = smooth_varilla(win, seg, hp['smooth_tension'],
                                hp['curve_n_samples'])
        rama_id = int(win['branch_id'])
        for fi in cluster_idx[cid]:
            flower_assignments[fi] = {'varilla_id': cid, 'rama_id': rama_id}
        varillas.append({
            'varilla_id': cid,
            'rama_id': rama_id,
            'num_flowers': int(len(cluster_idx[cid])),
            'centroid': (float(seg['centroid'][0]), float(seg['centroid'][1])),
            'p_basal': (float(seg['p_basal'][0]), float(seg['p_basal'][1])),
            'p_apical': (float(seg['p_apical'][0]), float(seg['p_apical'][1])),
            'origin': (float(win['origin'][0]), float(win['origin'][1])),
            'impact': (float(win['impact'][0]), float(win['impact'][1])),
            'found': bool(win['found']),
            'colinearity': float(win['colinearity']),
            'score': float(win['score']),
            'curve_xs': xs, 'curve_ys': ys,
            'flower_indices': [int(i) for i in cluster_idx[cid].tolist()],
        })

    n_found = sum(1 for v in varillas if v['found'])
    print("    Clusters: {0}  |  con impacto: {1}  |  fallback: {2}".format(
        len(varillas), n_found, len(varillas) - n_found))
    return varillas, flower_assignments


# =====================================================================
#  VISUALIZACION
# =====================================================================

def visualize_clusters(img, graph_data, flowers, labels):
    """
    Debug del Mecanismo A: colorea cada flor segun el TAMANO de su cluster.
    1 flor = azul, mas flores = mas rojo (colormap 'jet' sobre size_de_cluster).
    """
    txt_col = 'white' if DARK_BG else 'black'
    fig_bg  = '#1e1e1e' if DARK_BG else '#f0f0f0'

    labels = np.asarray(labels)
    # tamano del cluster al que pertenece cada flor
    from collections import Counter
    counts = Counter(labels.tolist())
    sizes = np.array([counts[l] for l in labels], dtype=float)

    n_clusters = len(counts)
    max_size = int(sizes.max()) if len(sizes) else 1

    # normalizacion para el colormap: 1 -> 0.0 (azul), max -> 1.0 (rojo)
    if max_size > 1:
        norm = (sizes - 1.0) / float(max_size - 1.0)
    else:
        norm = np.zeros_like(sizes)
    cmap = plt.get_cmap('jet')

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(fig_bg)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    F = np.array(flowers, dtype=np.float64)
    sc = ax.scatter(F[:, 0], F[:, 1], c=norm, cmap=cmap, vmin=0.0, vmax=1.0,
                    s=FLOWER_SIZE * 6, edgecolors='black', linewidths=0.3,
                    zorder=5)

    # Contorno por cluster: convex hull (>=3 flores), linea (2) o circulo (1).
    # Encierra exactamente las flores del cluster y ninguna ajena.
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon
    rng = np.random.default_rng(0)
    for lab in counts:
        if lab < 0:
            continue
        pts = F[labels == lab]
        hull_col = rng.uniform(0.35, 1.0, size=3)  # color distinto por cluster
        if len(pts) >= 3:
            try:
                h = ConvexHull(pts)
                poly = pts[h.vertices]
                ax.add_patch(MplPolygon(poly, closed=True, fill=False,
                                        edgecolor=hull_col, linewidth=1.0,
                                        alpha=0.9, zorder=4))
            except Exception:
                ax.plot(pts[:, 0], pts[:, 1], '-', color=hull_col,
                        linewidth=1.0, alpha=0.9, zorder=4)
        elif len(pts) == 2:
            ax.plot(pts[:, 0], pts[:, 1], '-', color=hull_col,
                    linewidth=1.0, alpha=0.9, zorder=4)
        else:
            ax.add_patch(Circle(pts[0], BASE_RADIUS * 1.5, fill=False,
                                edgecolor=hull_col, linewidth=1.0,
                                alpha=0.9, zorder=4))

    # barra de color con etiquetas de tamano real (1..max_size)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('Flores por cluster', color=txt_col)
    if max_size > 1:
        ticks = np.linspace(0.0, 1.0, min(max_size, 8))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(round(1 + t * (max_size - 1))))
                             for t in ticks])
    cbar.ax.yaxis.set_tick_params(color=txt_col)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=txt_col)

    singles = int(np.sum(sizes == 1))
    title = ("Mecanismo A (clustering)  |  Flores: {nf}  |  Clusters: {nc}  "
             "|  Singletons: {ns}  |  Cluster max: {mx}").format(
                 nf=len(flowers), nc=n_clusters, ns=singles, mx=max_size)
    ax.set_title(title, color=txt_col, fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize(img, graph_data, flowers, varillas, flower_assignments):
    n_v = len(varillas)
    txt_col = 'white' if DARK_BG else 'black'
    fig_bg  = '#1e1e1e' if DARK_BG else '#f0f0f0'

    skel_colors = {}
    for node in graph_data['nodes']:
        r, g, b = node['color_rgb']
        skel_colors[node['id']] = (r / 255.0, g / 255.0, b / 255.0)

    flowers_per_rama = defaultdict(int)
    for a in flower_assignments:
        flowers_per_rama[a['rama_id']] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9),
                                   gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(fig_bg)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Curva suavizada de cada varilla (color de la rama madre)
    for v in varillas:
        col = skel_colors.get(v['rama_id'], (1.0, 1.0, 1.0))
        ax1.plot(v['curve_xs'], v['curve_ys'], '-', color=col,
                 linewidth=LINE_WIDTH, alpha=VARILLA_ALPHA, zorder=4)
        # circulo en la base (origen en el tronco)
        circ = Circle(v['origin'], BASE_RADIUS, fill=False,
                      edgecolor=col, linewidth=1.4, zorder=6)
        ax1.add_patch(circ)
        # marca el punto de impacto si hubo
        if v['found']:
            ax1.plot(v['impact'][0], v['impact'][1], 'x', color=col,
                     markersize=5, zorder=6)

    # Flores coloreadas por rama madre
    for fi, (fx, fy) in enumerate(flowers):
        col = skel_colors.get(flower_assignments[fi]['rama_id'], (1.0, 1.0, 1.0))
        ax1.plot(fx, fy, 'o', color=col, markersize=FLOWER_SIZE,
                 markeredgewidth=0.4, markeredgecolor='black', zorder=5)

    title = ("Varillas por disparo desde tronco  |  "
             "Varillas: {n_v}  |  Flores: {tot}").format(
                 n_v=n_v, tot=len(flowers))
    ax1.set_title(title, color=txt_col, fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor(fig_bg)

    # Panel derecho: flores por rama
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
    plt.show()


# =====================================================================
#  RUN POR IMAGEN
# =====================================================================

def build_hyperparams(px_per_cm):
    px_per_cm = float(px_per_cm)
    return {
        'px_per_cm':          px_per_cm,
        # A
        'grow_radius_px':     GROW_RADIUS_CM * px_per_cm,
        'grow_angle_deg':     GROW_ANGLE_DEG,
        'seed_radius_px':     SEED_RADIUS_CM * px_per_cm,
        # C
        'angle_center_deg':   SHOOT_ANGLE_CENTER_DEG,
        'angle_min_deg':      SHOOT_ANGLE_MIN_DEG,
        'angle_max_deg':      SHOOT_ANGLE_MAX_DEG,
        # D
        'impact_tol_px':      IMPACT_TOL_PX,
        'basal_pref_px':      BASAL_PREFERENCE_PX,
        # Fusion (portado de v1)
        'enable_merge':       ENABLE_MERGE,
        'merge_angle_deg':    MERGE_ANGLE_DEG,
        'merge_lateral_px':   MERGE_LATERAL_CM * px_per_cm,
        'merge_max_dist_px':  MERGE_MAX_DIST_CM * px_per_cm,
        # E
        'smooth_tension':     SMOOTH_TENSION,
        'curve_n_samples':    CURVE_N_SAMPLES,
        # F
        'trunk_stride_px':    TRUNK_STRIDE_PX,
        'angle_step_deg':     ANGLE_STEP_DEG,
        'max_ray_len_px':     MAX_RAY_LEN_CM * px_per_cm,
        'normal_neighbor_px': NORMAL_NEIGHBOR_PX,
    }


def run_one(graph_json_path):
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
        return

    if PIXELS_PER_CM_OVERRIDE is not None:
        px_per_cm = float(PIXELS_PER_CM_OVERRIDE)
        scale_src = "override"
    else:
        px_per_cm = estimate_pixels_per_cm(graph_data, ASSUMED_TREE_HEIGHT_CM)
        scale_src = "auto (altura esqueleto / {0:.0f} cm)".format(
            ASSUMED_TREE_HEIGHT_CM)
    print("    Escala: {0:.2f} px/cm  [{1}]".format(px_per_cm, scale_src))

    hp = build_hyperparams(px_per_cm)

    # --- DEBUG: correr solo el Mecanismo A ---
    if STOP_AFTER == 'A':
        labels = cluster_flowers(flowers, hp)
        from collections import Counter
        counts = Counter(labels.tolist())
        sizes = list(counts.values())
        print("    [A] Clusters: {0}  |  Singletons: {1}  |  "
              "Cluster max: {2}  |  Tam. medio: {3:.2f}".format(
                  len(counts), sum(1 for s in sizes if s == 1),
                  max(sizes), sum(sizes) / len(sizes)))
        visualize_clusters(img, graph_data, flowers, labels)
        print("    [OK] plot de clustering mostrado")
        return

    varillas, flower_assignments = assign_varillas_v2(graph_data, flowers, hp)

    visualize(img, graph_data, flowers, varillas, flower_assignments)
    print("    [OK] plot mostrado")


# =====================================================================
#  EJECUCION
# =====================================================================

if __name__ == "__main__":
    json_path = os.path.join(GRAPH_JSON_DIR, IMAGE_NAME)
    if not os.path.exists(json_path):
        raise FileNotFoundError("No se encontro el JSON: " + json_path)

    print("[INFO] Procesando " + os.path.basename(json_path) + "\n")
    run_one(json_path)
