# CONTEXTO DEL PROYECTO — IPre Duraznos

> Pega este archivo al inicio de una conversación nueva con Claude para que retome el proyecto sin perder tiempo. Ruta: `C:\Users\vgara\OneDrive\Desktop\IPre\CONTEXTO_PROYECTO.md`

---

## 1. Qué es el proyecto

Pipeline de visión por computador para analizar **árboles de durazno (Prunus persica)** a partir de imágenes. Objetivo final: cuantificar la **densidad floral** y entender la estructura de ramas del árbol.

El proyecto va desde máscaras binarias → esqueletización → grafo de ramas clasificado → asignación de flores a ramas → mapas de densidad/calor.

---

## 2. Concepto biológico clave (IMPORTANTE)

Las flores del duraznero **no nacen de las ramas grandes visibles** (tronco, ramas primarias, secundarias). Nacen de **varillas** (ramos mixtos):

- Ramos delgados de **30–60 cm** de largo.
- Crecen aproximadamente **verticales** (contra la gravedad).
- **NO son visibles** en la imagen / esqueleto (muy delgados, ocultos entre las flores).
- Una varilla tiene un tramo basal **sin flores** (~7 cm) antes de la primera flor.
- Las flores en una varilla están separadas por **internodos** de ~2.5 cm.

El reto central: inferir dónde están las varillas invisibles a partir de cómo se disponen las flores, y conectar cada varilla a su rama madre del esqueleto.

---

## 3. Estructura de carpetas

```
IPre/
├── codigos/                      ← TODOS los scripts (repo git)
│   ├── densidades/               ← scripts de densidad floral
│   │   ├── varilla_density.py    ← [PRINCIPAL] flor→varilla→rama
│   │   ├── trunk_heatmap.py      ← mapa de calor del esqueleto
│   │   └── (otros métodos de densidad: laplacian, glee, random walk...)
│   ├── branch_identifier2.py     ← clasifica ramas (tronco/primaria/secundaria)
│   ├── build_graph_json.py       ← genera los JSON de grafo
│   ├── skeleton_graph_viewer.py  ← esqueleto → grafo coloreado
│   └── README.md                 ← documentación del pipeline completo
├── grafos json/                  ← 293 JSON de grafo (input)
├── json flores/                  ← JSON de flores formato LabelMe (input)
├── Grafos/                       ← PNG del esqueleto coloreado por rama
├── densidad floral varilla/json/ ← output JSON de varilla_density.py (292 archivos)
├── mapa de calor tronco/         ← output PNG de trunk_heatmap.py
└── (Esqueletos, MASKS, etc.)
```

---

## 4. Los dos scripts en foco activo

### `codigos/densidades/varilla_density.py` — Asignación flor → varilla → rama

Pipeline en 6 pasos:

1. **Calibración de escala**: mide la altura del esqueleto en px, asume árbol de 350 cm → px/cm por imagen.
2. **Clustering de flores**: DBSCAN con métrica **anisotrópica** (X multiplicada por `ANISOTROPY_LAMBDA=4`, penaliza separación horizontal). Agrupa flores en hileras verticales = varillas. `min_samples=1` → toda flor cae en algún cluster.
3. **Ajuste de varilla**: regresión OLS por cluster (elige regresar X/Y según varianza dominante). Con ≥5 flores añade curvatura cuadrática suave `r = c·t²`.
4. **Proyección al esqueleto**: toma el extremo basal (mayor Y = gravedad), camina por la tangente saltando el `BASAL_GAP_CM=7`, busca el esqueleto con un cKDTree. El píxel encontrado = `base_xy`, su rama = rama madre.
5. **Fusión de varillas**: une varillas casi paralelas y cercanas (preservando propiedad de clique — todos los pares deben ser compatibles).
6. **Asignación + confidence**: cada flor recibe varilla, rama madre y un score de confianza heurístico (`0.40·rms + 0.40·dist_esqueleto + 0.20·largo`, por `bio_term`).

Es **determinístico** (sin azar) pero **heurístico** (combina factores con umbrales blandos). Los límites de largo (`VARILLA_LENGTH_MIN/MAX_CM`) son solo flags informativos, no descartan clusters.

**Output**: PNG (flores coloreadas por rama madre + línea por varilla) y JSON en `densidad floral varilla/json/`.

**Visualización actual**: por cada varilla dibuja un tallo recto `base_xy → centroide del cluster` + círculo en la base. (Se quitó la curva de regresión sobre las flores por pedido del usuario.)

### `codigos/densidades/trunk_heatmap.py` — Mapa de calor del esqueleto

Lee el JSON output de `varilla_density.py` + el JSON de grafo. Para cada píxel del esqueleto calcula calor acumulado:

```
heat(p) = Σ_varillas  num_flowers · exp(−‖p − base_xy‖² / (2σ²))
```

con `KERNEL_SIGMA_CM=8`. Pinta el esqueleto con colormap `'jet'` (azul→rojo) **reemplazando píxeles** sobre un fondo en escala de grises (NO scatter, NO colores per-rama). Grosor `SKEL_THICKNESS_PX=3`.

`SHOW_PLOT=True` → muestra ventana; `False` → guarda PNG en `mapa de calor tronco/`.

**Dependencia**: requiere correr `varilla_density.py` primero.

---

## 5. Formatos de datos

**JSON de grafo** (`grafos json/imgs_frameXX_00000_graph.json`):
```
{ "image": "...png", "nodes": [...], "edges": [...], "branches": [...] }
```
- `nodes[i]`: `id, centroid_x/y, level, level_name, is_trunk, color_rgb`
- `branches[i]`: `id, pixels` — pixels en formato **[y, x]** (fila, columna)

**JSON de flores** (`json flores/frameXX.json`, formato LabelMe):
- `shapes[i]`: `label="flower", points=[[x, y]]` — coords en formato **(x, y)**

**JSON output de varilla** (`densidad floral varilla/json/..._varilla.json`):
- `varillas[i]`: `varilla_id, rama_id, num_flowers, centroid, base_xy, u, v, c, t_min/max, confidence, ...`
- `flowers[i]`: `flower_id, x, y, varilla_id, rama_id, confidence`

⚠️ **Convención de coordenadas**: branches usan `[y, x]`, flores usan `[x, y]`. No confundir.

---

## 6. Convenciones

- Nombres: `imgs_frameXX_00000.png` ↔ `frameXX.json` (se emparejan por número de frame).
- En coords de imagen, **+Y apunta hacia abajo** (gravedad).
- Cada script tiene un bloque `CONFIG` arriba y una variable `SINGLE_IMAGE` (None = batch).
- Entorno: Windows, PowerShell. Python con numpy, opencv-python, scikit-image, scikit-learn, scipy, matplotlib.

---

## 7. Estado actual / pendientes conocidos

- `varilla_density.py` y `trunk_heatmap.py` funcionando; ambos corridos en batch.
- Discrepancia sin resolver: el config dice `VARILLA_LENGTH_MIN/MAX_CM = 8/80` pero la info biológica real es **30–60 cm**. El "largo" que mide el código es solo el tramo florido (no incluye la brecha basal). Si se quiere alinear: ajustar valores, o pasar de flag blando a filtro duro, o cortar clusters largos.
- Imagen de prueba habitual: `imgs_frame100_00000_graph.json`.

---

## 8. Cómo trabajar en este proyecto

- Antes de editar, leer el bloque CONFIG del script relevante.
- Cambios de visualización: el usuario itera mucho, hacer cambios mínimos y precisos.
- Para correr: F5 sobre el script con `SINGLE_IMAGE` seteado a una imagen para tunear, o `None` para batch.
