# Auditoria del repositorio `IPre-Duraznos`

**Paper asociado:** *RGB image processing of trees for flower load across tree and branch scales by structure-aware geometric assignment*
**Destino:** Computers and Electronics in Agriculture / Biosystems Engineering (Q1)
**Fecha:** 2026-08-10
**Remote:** `https://github.com/vichogaray/IPre-Duraznos.git`
**Alcance:** auditoria de solo lectura. No se modifico ningun archivo.

---

## 0. Resumen ejecutivo

El repositorio actual es la carpeta `IPre/codigos/` (no `IPre/`, que no es un repo git). Contiene **18 scripts Python** (~370 KB de codigo) que cubren de forma bastante completa las etapas 1-6 del pipeline del paper. La calidad algoritmica es alta; el problema es exclusivamente de **empaquetado cientifico**.

Cinco hallazgos criticos, en orden de impacto sobre la reproducibilidad exigida por un Q1:

| # | Hallazgo | Severidad |
|---|---|---|
| 1 | **Toda la capa de evaluacion/validacion esta fuera del repo** (3 scripts, ~44 KB, en el Desktop, sin versionar). Un paper Q1 no es publicable sin el codigo que produce sus metricas. | Bloqueante |
| 2 | **No existe el mapa de densidad floral en flores/cm** (etapa 7). Solo hay heatmaps cualitativos sin unidades fisicas. Es la contribucion final del paper. | Bloqueante |
| 3 | **18/18 scripts tienen rutas absolutas** `C:\Users\vgara\OneDrive\Desktop\IPre\...` hardcodeadas. Nadie fuera de tu maquina puede ejecutar nada. | Bloqueante |
| 4 | Faltan `requirements.txt`, `LICENSE`, `CITATION.cff`, enlace a Zenodo, `data/sample/`. | Alto |
| 5 | Duplicacion masiva: `load_flowers` x10, `load_graph_json` x8, `estimate_pixels_per_cm` x4, etc. Sin modulo comun. | Alto |

Ademas: 3 scripts sin trackear con trabajo real (`varilla_density_2.py`, `Varilla_heatmap_2.py`, `script_supremo.py`), `__pycache__/` presente en disco, y ~30 `print()` con acentos que rompen bajo consola cp1252.

---

## 1. Inventario de archivos

### 1.1 Arbol actual del repositorio

```
IPre/codigos/                       <- RAIZ DEL REPO GIT
├── .gitattributes                      309 B    [tracked]
├── .gitignore                          134 B    [tracked]
├── README.md                          5.2 KB    [tracked]
├── Esqueletizacion.py                29.7 KB    [tracked, MODIFICADO sin commit]
├── Grafo.py                          38.4 KB    [tracked]
├── Identificador_de_ramas.py         18.9 KB    [tracked, MODIFICADO sin commit]
├── Param_tuner_esqueletizacion.py    12.4 KB    [tracked]
├── build_graph_json.py                4.8 KB    [tracked]
├── script_supremo.py                 13.3 KB    [SIN TRACKEAR]
├── __pycache__/                       ~123 KB   [en disco; ignorado por .gitignore]
└── densidades/
    ├── euclidian_density.py          12.8 KB    [tracked]
    ├── laplacian_density.py          11.9 KB    [tracked]
    ├── random_walk_density.py        13.4 KB    [tracked]
    ├── glee_density.py               13.9 KB    [tracked]
    ├── hybridlaplac_density.py       19.8 KB    [tracked]
    ├── hybridrw_density.py           21.0 KB    [tracked]
    ├── varilla_density.py            37.8 KB    [tracked, MODIFICADO sin commit]
    ├── varilla_density_2.py          42.6 KB    [SIN TRACKEAR]
    ├── varilla_lsystem.py            24.8 KB    [tracked, MODIFICADO sin commit]
    ├── RW_heatmap.py                 14.4 KB    [tracked]
    ├── Varilla_heatmap.py            13.7 KB    [tracked, MODIFICADO sin commit]
    ├── Varilla_heatmap_2.py          11.7 KB    [SIN TRACKEAR]
    └── __pycache__/                   ~127 KB   [en disco; ignorado]
```

**Total codigo fuente: 18 archivos `.py`, ~371 KB.**

### 1.2 Codigo relevante FUERA del repositorio

En `C:\Users\vgara\OneDrive\Desktop\` (sueltos, sin control de versiones):

| Archivo | Tamano | Que hace |
|---|---|---|
| `gt_annotator.py` | 11.0 KB | Herramienta GUI de anotacion de ground truth: clic sobre el esqueleto -> snap al pixel mas cercano -> se ingresa el numero de flores en ese punto. Asocia cada punto a su `rama_id`. Genera el GT contra el que se validan todos los metodos. |
| `gt_heatmap_compare.py` | 19.9 KB | Compara el heatmap del GT contra el de un metodo mediante **distancia de Wasserstein geodesica** sobre el esqueleto (Sinkhorn). Devuelve distancia en cm + similitud normalizada [0,1]. |
| `gt_flor_rama_eval.py` | 12.8 KB | Evaluacion multi-metodo: accuracy flor->rama contra el GT para todos los metodos + Wasserstein para varilla v1/v2. Exporta `flor_rama_resumen.csv`. |

Estos 3 archivos (~44 KB) son **la seccion de resultados del paper**. Que no esten versionados es el hallazgo mas grave de la auditoria.

### 1.3 Descripcion de cada script del repo

#### Etapa 1-2: Mascaras y esqueletizacion

| Script | Descripcion (del docstring/codigo) |
|---|---|
| `Esqueletizacion.py` | Esqueletizacion masiva de mascaras binarias. Pipeline: Otsu -> suavizado morfologico (cierre->apertura) -> filtro por grosor via transformada de distancia -> dilatacion de centros -> `skimage.skeletonize` -> pruning de ramas cortas -> reparacion de gaps de 1 px. Jerarquia de parametros de 3 niveles: MANUAL (`parametros.json`, ground truth) > APRENDIDO (`defaults_aprendidos.json`, moda de calibraciones) > FIJO (bloque CONFIG). El modo ADAPTIVE quedo deshabilitado. |
| `Param_tuner_esqueletizacion.py` | GUI matplotlib con sliders para calibrar a mano los parametros de esqueletizacion imagen por imagen. Persiste en `parametros.json`, que alimenta el nivel MANUAL de `Esqueletizacion.py`. Es la herramienta que genero el ground truth de parametros. |

#### Etapa 3-4: Grafo y jerarquia de ramas

| Script | Descripcion |
|---|---|
| `Grafo.py` | Convierte un esqueleto binario en grafo de ramas coloreado. Traza ramas por topologia (numero de cruce -> endpoints/junctions/slabs). GUI interactiva: merge de ramas, eliminacion, sliders de poda (`MIN_BRANCH_PX`) y grosor, export PNG. Soporta `BATCH_MODE`. El archivo mas grande del repo (38 KB), mayoritariamente GUI. |
| `Identificador_de_ramas.py` | Libreria (no script) de clasificacion jerarquica. Agrupa pixeles por color -> detecta adyacencia espacial -> identifica el tronco (rama mas baja) -> BFS desde el tronco asignando niveles Tronco -> Principal -> Secundaria -> ... Importada por `build_graph_json.py`. |
| `build_graph_json.py` | Lee cada PNG de `Grafos/`, extrae la estructura con `Identificador_de_ramas`, y escribe un JSON por imagen con nodos, aristas, jerarquia y pixeles por rama. **Es el hub del pipeline**: su output alimenta los 10 metodos de densidad. |

#### Etapa 5: Asignacion flor -> rama

| Script | Familia | Descripcion |
|---|---|---|
| `euclidian_density.py` | Morfologico | Baseline: cada flor a la rama con distancia euclidiana minima sobre los pixeles de la rama. **Detecta ramas por color del PNG**, no por el JSON de grafo -> sus `branch_id` NO estan alineados con el GT. |
| `laplacian_density.py` | Morfologico | Clasificacion semi-supervisada con Graph Laplacian (Orduz 2019): perdida softmax + termino de suavidad Laplaciana. `K_NEIGHBORS=3`, `ALPHA=1.0`. |
| `random_walk_density.py` | Morfologico | Random Walk absorbente (Grady 2006). Ramas = nodos absorbentes, flores = no absorbentes. Resuelve `(I-Q)F = R`; asignacion por `argmax`. Aristas flor-rama y flor-flor con peso `exp(-d/sigma)`. |
| `glee_density.py` | Morfologico | GLEE / Geometric Laplacian Eigenmap Embedding (Torres et al. 2020), implementado con NumPy/SciPy puro (sin `karateclub`). |
| `hybridlaplac_density.py` | Morfologico | Hibrido: euclidiana cuando la rama es dominante (`d2/d1 > 1.5`), Laplaciano solo para flores ambiguas. Mismo problema de `branch_id` por color. |
| `hybridrw_density.py` | Morfologico | Idem anterior pero con Random Walk para las ambiguas. |
| `varilla_density.py` | Reconstruccion de brotes | **Metodo principal v1.** Flor -> varilla -> rama. DBSCAN anisotropico (`ANISOTROPY_LAMBDA=4`, favorece hileras verticales) -> ajuste OLS por cluster + curvatura cuadratica opcional -> extension del gap basal (~7 cm) -> proyeccion al esqueleto por cKDTree -> fusion de varillas paralelas -> score de confianza. Calibracion de escala px/cm por altura asumida del arbol. |
| `varilla_density_2.py` | Reconstruccion de brotes | **v2, sin trackear.** Rediseno modular por mecanismos A-F: (A) clustering DBSCAN anisotropico, (B) ajuste OLS acotado a segmento, (C) disparo de rayos desde el tronco con paso angular, (D) conexion rayo-cluster con desempate por colinealidad y fallback a rama mas cercana (nunca pierde flores), (E) suavizado spline/Bezier, (F) discretizacion. Invierte la logica de v1: en vez de proyectar la flor hacia el tronco, dispara desde el tronco hacia las flores. Solo plotea, no guarda. |
| `varilla_lsystem.py` | Reconstruccion de brotes | Varillas generadas con un **L-System** ajustado a las flores observadas via `least_squares`. Extrapola el gap basal para obtener P0 (punto de insercion) y lo asigna a la rama del scaffold mas cercana. Reasigna flores sueltas -> cobertura 100%. Solo plotea. |

#### Etapa 6: Visualizacion / heatmaps

| Script | Descripcion |
|---|---|
| `Varilla_heatmap.py` | Mapa de calor del esqueleto: `heat(p) = sum_v num_flowers(v) * exp(-||p - base_xy(v)||^2 / 2*sigma^2)`, `KERNEL_SIGMA_CM=8`. Colorea el esqueleto con colormap `jet`. Lee el JSON de `varilla_density.py`. |
| `Varilla_heatmap_2.py` | Idem para v2. **Sin trackear.** No lee JSON: importa `varilla_density_2` y calcula las varillas en memoria; el calor se deposita en el `origin` del rayo (analogo de `base_xy`). |
| `RW_heatmap.py` | Heatmap por subsecciones de rama (`SECTION_LENGTH=30 px`) coloreadas por conteo de flores asignadas via Random Walk. Espectro azul->rojo. |

#### Orquestacion

| Script | Descripcion |
|---|---|
| `script_supremo.py` | **Sin trackear.** Orquestador end-to-end sobre UNA mascara: mascara -> esqueleto -> grafo -> grafo-json -> varilla density -> heatmap, con panel final de todas las etapas. Automatiza las dos etapas antes manuales con parametros aprendidos del GT (`GRAFO_MIN_BRANCH_PX=180`, error ~1.7 ramas), con escapes `TUNE_SKELETON` / `TUNE_GRAFO` que abren las GUI existentes. **Es el mejor candidato a "demo de reproducibilidad" del paper.** |

---

## 2. Cobertura del pipeline

| # | Etapa del paper | Estado | Implementado en |
|---|---|---|---|
| 1 | Suavizado morfologico de mascaras | **Completo** | `Esqueletizacion.py` (cierre->apertura + filtro de grosor) |
| 2 | Esqueletizacion (Zhang-Suen, scikit-image) | **Completo** | `Esqueletizacion.py`, `Param_tuner_esqueletizacion.py` |
| 3 | Grafo jerarquico (endpoint/junction/slab) | **Completo con reserva** | `Grafo.py`, `build_graph_json.py` |
| 4 | Identificacion de ramas por orden de ramificacion | **Completo** | `Identificador_de_ramas.py` (BFS desde tronco) |
| 5a | Asignacion morfologica: Euclidiana | **Completo** | `euclidian_density.py` |
| 5b | Asignacion morfologica: Graph Laplacian | **Completo** | `laplacian_density.py` |
| 5c | Asignacion morfologica: Random Walk | **Completo** | `random_walk_density.py` |
| 5d | Asignacion morfologica: GLEE | **Completo** | `glee_density.py` |
| 5e | (extra) Hibridos euclidiano+Laplaciano / +RW | **Completo** | `hybridlaplac_density.py`, `hybridrw_density.py` |
| 6a | Reconstruccion de brotes: L-Systems | **Completo** | `varilla_lsystem.py` |
| 6b | Reconstruccion: brotes candidatos | **Completo** | `varilla_density_2.py` (rayos desde el tronco) |
| 6c | Reconstruccion: proyeccion de clusters (**mejor metodo**) | **Completo** | `varilla_density.py` |
| 7 | **Mapa de densidad floral (flores/cm)** | **AUSENTE** | -- |
| 8 | **Evaluacion contra ground truth** | **Fuera del repo** | `gt_*.py` en el Desktop |

### 2.1 Brechas

**Brecha critica 1 -- la etapa 7 no existe.**
El paper promete densidad floral en **flores·cm⁻¹**. Lo que hay son tres heatmaps *cualitativos*: `Varilla_heatmap*.py` acumula `num_flowers * gaussiana` en unidades arbitrarias, y `RW_heatmap.py` colorea por conteo bruto por subseccion. `estimate_pixels_per_cm()` ya existe en 4 archivos, asi que la conversion es factible, pero **nadie divide por la longitud de rama en cm**. Falta un script que produzca, por rama y por arbol, una densidad lineal con unidades fisicas -- que es literalmente el titulo del paper ("flower load across tree and branch scales").

**Brecha critica 2 -- la evaluacion no esta versionada.**
Los revisores Q1 piden el codigo que genera las tablas. `gt_annotator.py` + `gt_heatmap_compare.py` + `gt_flor_rama_eval.py` deben entrar al repo. Nota tecnica del propio `gt_flor_rama_eval.py`: euclidian e hibridos quedan **excluidos de la evaluacion** porque detectan ramas por color del PNG y sus `branch_id` no se alinean con el grafo/GT. Eso es una inconsistencia real del pipeline que el paper tendra que justificar o corregir (migrar esos 3 metodos a leer el JSON de grafo).

**Brecha menor 3 -- terminologia grafo.**
El paper habla de NetworkX con nodos endpoint/junction/slab. `Grafo.py` calcula esa topologia (numero de cruce) pero el grafo se materializa como **PNG coloreado + JSON propio**, y NetworkX solo aparece en los metodos de densidad. Hay un desajuste entre la descripcion del paper y la implementacion: o se documenta como esta, o se refactoriza `build_graph_json.py` para emitir un `nx.Graph` explicito.

**Brecha menor 4 -- calibracion de escala.**
`estimate_pixels_per_cm` asume una altura fija de arbol (350 cm) para toda imagen. Para un paper Q1 esto es un supuesto fuerte que necesita, como minimo, un analisis de sensibilidad; idealmente una referencia metrica en campo.

---

## 3. Estado del README

**Actual:** 5.2 KB, en espanol. Contiene diagrama ASCII del pipeline, tablas de descripcion por script (bien escritas), lista de dependencias con comando `pip install`, y 3 referencias bibliograficas.

Como README de trabajo interno es **bueno**. Como README de repo Q1 le falta la mayor parte.

| Elemento estandar Q1 | Estado | Comentario |
|---|---|---|
| Titulo del paper | Falta | Dice "IPre Duraznos", no el titulo del articulo |
| Badges (DOI Zenodo, licencia, Python) | Falta | |
| Abstract / resumen del metodo | Parcial | Hay parrafo sobre varillas; falta el resumen cientifico |
| Cita (BibTeX) | Falta | Critico. Tambien falta `CITATION.cff` |
| Autores y afiliacion | Falta | |
| Instalacion reproducible | Debil | Solo `pip install <lista>`; sin versiones fijadas ni `requirements.txt` |
| Version de Python | Falta | No dice 3.9 en ninguna parte |
| Enlace al dataset (Zenodo) | Falta | Critico |
| Guia de reproduccion paso a paso | Falta | No hay "corre esto y obtienes la Figura 3" |
| Estructura de carpetas | Falta | |
| Formatos de datos I/O | Falta | Vive en `CONTEXTO_PROYECTO.md`, fuera del repo |
| Tabla de resultados/metricas | Falta | |
| Licencia | Falta | Critico: sin licencia, legalmente nadie puede reusarlo |
| Contacto | Falta | |
| **Precision** | **Con errores** | Documenta `copy_masks_sin_grafo.py`, **que no existe**. Omite `varilla_density_2.py`, `Varilla_heatmap_2.py`, `script_supremo.py`, `varilla_lsystem.py` y los 3 `gt_*.py` |
| Idioma | A decidir | Esta en espanol; revista Q1 internacional espera ingles |

Lista `gensim` como dependencia, pero **ningun archivo lo importa** (`glee_density.py` dice explicitamente que evita `karateclub`/`gensim`). Dependencia fantasma.

---

## 4. Problemas detectados

### 4.1 Bloqueantes para reproducibilidad

**P1 -- Rutas absolutas hardcodeadas en los 18 scripts.**
Cada script tiene entre 1 y 4 rutas `C:\Users\vgara\OneDrive\Desktop\IPre\...`. Peor: apuntan a carpetas *fuera del repo* (`Mascaras/`, `Grafos/`, `grafos json/`, `json flores/`). Consecuencia: recien clonado, **ningun script corre**. Recuento por archivo:

```
4 rutas: varilla_lsystem, random_walk_density, laplacian_density,
         hybridrw_density, hybridlaplac_density, glee_density,
         Varilla_heatmap, RW_heatmap
3 rutas: varilla_density, varilla_density_2, Varilla_heatmap_2,
         Param_tuner_esqueletizacion, Grafo
2 rutas: euclidian_density, build_graph_json, Esqueletizacion
1 ruta : script_supremo, Identificador_de_ramas
```

**P2 -- Sin `requirements.txt` ni versiones fijadas.** Un revisor con scikit-learn nuevo puede obtener otro comportamiento de DBSCAN. Dependencias reales detectadas por imports: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `scipy`, `matplotlib`, `networkx`. (`gensim` listado pero no usado.)

**P3 -- Sin LICENSE.** Sin licencia explicita, el codigo es "todos los derechos reservados" por defecto. Varias revistas rechazan el material suplementario en esa condicion.

**P4 -- Sin datos de ejemplo.** No hay `data/sample/`. Aunque el dataset de 292 arboles vaya a Zenodo, el repo necesita 2-3 casos para un smoke test.

**P5 -- Configuracion por edicion de codigo.** El patron es "edita el bloque CONFIG y presiona F5". No hay CLI (`argparse`) en ningun script. Un revisor no puede ejecutar un experimento sin editar fuentes.

### 4.2 Duplicacion de codigo

Funciones definidas de forma independiente en multiples archivos:

| Funcion | Copias | Que implica |
|---|---|---|
| `run_one` | 10 | Cada metodo reimplementa su bucle por imagen |
| `load_flowers` | 10 | 10 parsers del mismo JSON LabelMe |
| `auto_detect_graph_image` | 9 | |
| `load_graph_json` | 8 | 8 lectores del formato de grafo |
| `auto_detect_flower_json` | 8 | |
| `visualize` | 7 | |
| `estimate_pixels_per_cm` | 4 | **La calibracion px/cm esta duplicada 4 veces: riesgo de divergencia silenciosa entre metodos** |
| `cluster_flowers`, `build_hyperparams`, `compute_heat`, `prune_skeleton`, `_count_neighbors`, ... | 2 c/u | |

Riesgo cientifico concreto: si dos metodos calibran px/cm distinto, la comparacion del paper deja de ser valida. Un modulo `common/io.py` + `common/geometry.py` eliminaria ~40% del codigo.

### 4.3 Higiene del repositorio

| Problema | Detalle |
|---|---|
| **3 archivos sin trackear con trabajo real** | `varilla_density_2.py` (42.6 KB, mecanismos A-F), `Varilla_heatmap_2.py` (11.7 KB), `script_supremo.py` (13.3 KB). Un `git clone` pierde el metodo v2 completo y el orquestador. |
| **5 archivos modificados sin commit** | `Esqueletizacion.py`, `Identificador_de_ramas.py`, `varilla_density.py`, `varilla_lsystem.py`, `Varilla_heatmap.py` |
| **`__pycache__/` en disco** | ~250 KB en 2 carpetas. Correctamente ignorados por `.gitignore` y no trackeados; solo limpieza local. |
| **`.gitignore` incompleto** | No ignora `*.png`, `*.json` de output, `results/`, `data/` (salvo sample), `parametros.json`, `defaults_aprendidos.json`, `*.csv`. Riesgo real de commitear por accidente cientos de MB de outputs. |
| **BOM UTF-8 en `varilla_density_2.py`** | Empieza con `EF BB BF`. En Python 3.9 no rompe, pero ensucia diffs y puede romper herramientas. |
| **Encoding mixto** | 14 archivos UTF-8, 4 puro ASCII. Sin declaracion `# -*- coding -*-` en ninguno. |
| **~30 `print()` con caracteres no-ASCII** | Viola tu restriccion cp1252. Ejemplos: `Esqueletizacion.py:570` (`imágenes`), `:601` (`automático`), `Grafo.py:348` (`píxeles`), `glee_density.py:87` (guion largo `—`). Bajo `chcp 1252` en Windows esto lanza `UnicodeEncodeError` y aborta el batch. |
| **Sin tests** | Ni un `tests/`. Un smoke test sobre `data/sample/` bastaria. |
| **Sin CI** | Sin `.github/workflows/`. |
| **Sin `CITATION.cff`** | GitHub muestra "Cite this repository" automaticamente si existe. |
| **Nomenclatura inconsistente** | Mezcla de espanol/ingles y de convenciones: `Esqueletizacion.py` y `Grafo.py` en CapitalizadoEspanol, `build_graph_json.py` en snake_case ingles, `Varilla_heatmap.py` en hibrido. |
| **Sufijos `_2` como versionado** | `varilla_density_2.py`, `Varilla_heatmap_2.py` versionan por nombre de archivo en vez de por git. |
| **Codigo obsoleto embebido** | Modo `ADAPTIVE` de `Esqueletizacion.py` deshabilitado pero presente. |
| **Typo en docstring** | `euclidian_density.py:12`: `"Dale Run (F5)descarg"`. Ademas `euclidian` es un error ortografico de `euclidean`. |
| **Documentacion fuera del repo** | `CONTEXTO_PROYECTO.md` (el mejor documento del proyecto) vive en `IPre/`, sin versionar. |

---

## 5. Propuesta de reestructuracion

### 5.1 Decision previa: donde vive la raiz del repo

**Recomendacion: mover el repo git de `IPre/codigos/` a `IPre/`.**

Justificacion: hoy el repo solo contiene codigo, pero `data/sample/`, `docs/` y `results/` son hermanos de `codigos/`, no hijos. Para meterlos habria que moverlos dentro de `codigos/`, lo que hace que la carpeta deje de significar "codigos". Promover la raiz a `IPre/` permite la estructura estandar sin contorsiones, con un `.gitignore` que excluya los datasets pesados (`Mascaras/`, `IMGS/`, `Grafos/`, `grafos json/`, `json flores/`, `Esqueletizacion/`).

*Alternativa conservadora:* dejar la raiz en `codigos/` y renombrar la carpeta a `IPre-Duraznos/`. Menos disruptivo, pero obliga a duplicar las imagenes de muestra dentro del repo.

### 5.2 Estructura propuesta

```
IPre-Duraznos/                        <- raiz del repo git
│
├── README.md                         reescrito, en ingles, estandar Q1
├── LICENSE                           MIT (codigo) -- ver 5.6
├── CITATION.cff                      cita del paper
├── requirements.txt                  con versiones fijadas
├── .gitignore                        endurecido
├── .gitattributes                    (sin cambios)
├── environment.yml                   opcional, conda py3.9
│
├── src/
│   ├── common/                       <- NUEVO: elimina la duplicacion
│   │   ├── __init__.py
│   │   ├── io.py                     load_flowers, load_graph_json,
│   │   │                             auto_detect_* (hoy x8-x10)
│   │   ├── geometry.py               estimate_pixels_per_cm (hoy x4),
│   │   │                             prune_skeleton, _count_neighbors
│   │   ├── paths.py                  rutas relativas / config.yaml.
│   │   │                             MATA las 18 rutas absolutas
│   │   └── viz.py                    paletas y helpers de plot
│   │
│   ├── preprocessing/
│   │   ├── skeletonize.py            <- Esqueletizacion.py
│   │   └── tune_skeleton_params.py   <- Param_tuner_esqueletizacion.py
│   │
│   ├── graph/
│   │   ├── build_skeleton_graph.py   <- Grafo.py
│   │   ├── branch_hierarchy.py       <- Identificador_de_ramas.py
│   │   └── export_graph_json.py      <- build_graph_json.py
│   │
│   ├── assignment/
│   │   ├── morphological/
│   │   │   ├── euclidean.py          <- euclidian_density.py (typo corregido)
│   │   │   ├── graph_laplacian.py    <- laplacian_density.py
│   │   │   ├── random_walk.py        <- random_walk_density.py
│   │   │   ├── glee.py               <- glee_density.py
│   │   │   ├── hybrid_laplacian.py   <- hybridlaplac_density.py
│   │   │   └── hybrid_random_walk.py <- hybridrw_density.py
│   │   └── shoot_reconstruction/
│   │       ├── cluster_projection.py <- varilla_density.py   [METODO PRINCIPAL]
│   │       ├── candidate_shoots.py   <- varilla_density_2.py
│   │       └── lsystem.py            <- varilla_lsystem.py
│   │
│   ├── density/
│   │   ├── flower_load_map.py        <- NUEVO: flores/cm por rama y arbol
│   │   ├── heatmap_shoots.py         <- Varilla_heatmap.py (+ _2 fusionado)
│   │   └── heatmap_random_walk.py    <- RW_heatmap.py
│   │
│   ├── evaluation/                   <- NUEVO: entra el codigo del Desktop
│   │   ├── gt_annotator.py           <- Desktop/gt_annotator.py
│   │   ├── heatmap_wasserstein.py    <- Desktop/gt_heatmap_compare.py
│   │   └── flower_branch_accuracy.py <- Desktop/gt_flor_rama_eval.py
│   │
│   └── run_pipeline.py               <- script_supremo.py, con argparse
│
├── notebooks/
│   ├── 01_pipeline_demo.ipynb        mascara -> mapa de carga en 1 imagen
│   ├── 02_method_comparison.ipynb    reproduce la tabla comparativa
│   └── 03_paper_figures.ipynb        genera las figuras del paper
│
├── data/
│   └── sample/                       2-3 arboles completos (< 10 MB)
│       ├── masks/
│       ├── skeletons/
│       ├── graphs/
│       ├── graph_json/
│       ├── flowers_json/
│       └── ground_truth/
│
├── results/                          .gitignored salvo .gitkeep + metricas
│   ├── figures/
│   ├── metrics/                      flor_rama_resumen.csv (SI versionado)
│   └── .gitkeep
│
├── docs/
│   ├── PIPELINE.md                   <- CONTEXTO_PROYECTO.md, en ingles
│   ├── DATA_FORMATS.md               convenciones [y,x] vs (x,y)
│   ├── BIOLOGY.md                    justificacion del modelo de varilla
│   ├── PARAMETERS.md                 tabla de todos los parametros
│   └── REPRODUCING_PAPER.md          figura por figura, tabla por tabla
│
└── configs/
    ├── default.yaml                  todos los CONFIG en un solo lugar
    └── paper_experiments.yaml        parametros exactos del paper
```

### 5.3 Tabla de movimientos

| Actual | Destino | Justificacion |
|---|---|---|
| `Esqueletizacion.py` | `src/preprocessing/skeletonize.py` | Ingles + etapa explicita |
| `Param_tuner_esqueletizacion.py` | `src/preprocessing/tune_skeleton_params.py` | Junto a lo que calibra |
| `Grafo.py` | `src/graph/build_skeleton_graph.py` | "Grafo" es ambiguo |
| `Identificador_de_ramas.py` | `src/graph/branch_hierarchy.py` | Es libreria, no script |
| `build_graph_json.py` | `src/graph/export_graph_json.py` | Coherencia del modulo |
| `densidades/euclidian_density.py` | `src/assignment/morphological/euclidean.py` | Corrige typo; agrupa por familia del paper |
| `densidades/laplacian_density.py` | `src/assignment/morphological/graph_laplacian.py` | Nombre del paper |
| `densidades/random_walk_density.py` | `src/assignment/morphological/random_walk.py` | |
| `densidades/glee_density.py` | `src/assignment/morphological/glee.py` | |
| `densidades/hybridlaplac_density.py` | `src/assignment/morphological/hybrid_laplacian.py` | Nombre legible |
| `densidades/hybridrw_density.py` | `src/assignment/morphological/hybrid_random_walk.py` | |
| `densidades/varilla_density.py` | `src/assignment/shoot_reconstruction/cluster_projection.py` | Es "proyeccion de clusters" del paper |
| `densidades/varilla_density_2.py` | `src/assignment/shoot_reconstruction/candidate_shoots.py` | Es "brotes candidatos". **Trackear ya** |
| `densidades/varilla_lsystem.py` | `src/assignment/shoot_reconstruction/lsystem.py` | |
| `densidades/Varilla_heatmap.py` + `_2.py` | `src/density/heatmap_shoots.py` | **Fusionar**: mismo algoritmo, distinta fuente. Un flag `--method` basta |
| `densidades/RW_heatmap.py` | `src/density/heatmap_random_walk.py` | |
| `script_supremo.py` | `src/run_pipeline.py` | Entry point publico. **Trackear ya** |
| `Desktop/gt_annotator.py` | `src/evaluation/gt_annotator.py` | **Entrar al repo: es la validacion del paper** |
| `Desktop/gt_heatmap_compare.py` | `src/evaluation/heatmap_wasserstein.py` | Idem |
| `Desktop/gt_flor_rama_eval.py` | `src/evaluation/flower_branch_accuracy.py` | Idem; produce la tabla de resultados |
| `IPre/CONTEXTO_PROYECTO.md` | `docs/PIPELINE.md` | Mejor doc del proyecto, hoy sin versionar |
| `README.md` | reescrito | Ver seccion 3 |

### 5.4 Archivos a crear

| Archivo | Por que |
|---|---|
| `src/common/{io,geometry,paths,viz}.py` | Elimina la duplicacion (P5) y unifica la calibracion px/cm |
| `src/density/flower_load_map.py` | **La etapa 7 del paper no existe.** Debe emitir flores/cm por rama y por arbol |
| `requirements.txt` | Reproducibilidad |
| `LICENSE` | Requisito legal |
| `CITATION.cff` | Estandar Q1 |
| `configs/default.yaml` | Saca los CONFIG del codigo |
| `tests/test_smoke.py` | Verifica que el pipeline corre sobre `data/sample/` |
| `docs/*.md` | Documentacion del metodo |

### 5.5 Que eliminar

| Elemento | Accion | Justificacion |
|---|---|---|
| `__pycache__/` (ambos) | Borrar del disco | Ya ignorados; ~250 KB de ruido |
| Modo `ADAPTIVE` en `Esqueletizacion.py` | Borrar (queda en git) | Deshabilitado y superado por los defaults aprendidos |
| Referencia a `copy_masks_sin_grafo.py` en README | Borrar | El archivo no existe |
| `gensim` de las dependencias | Borrar | No lo importa nadie |
| BOM de `varilla_density_2.py` | Borrar | Higiene |
| Acentos en ~30 `print()` | Reemplazar por ASCII | Tu restriccion cp1252 |

**Nada de codigo se elimina.** Los hibridos, aunque queden fuera de la evaluacion por el desalineamiento de `branch_id`, tienen valor como baselines: conviene corregirlos (que lean el JSON de grafo) en vez de descartarlos.

### 5.6 Notas sobre licencia y dataset

- **Codigo:** MIT o BSD-3-Clause. MIT es lo mas comun en CEA/Biosystems Engineering.
- **Dataset (Zenodo):** CC-BY-4.0. Licencia separada de la del codigo.
- **Zenodo:** activa la integracion GitHub-Zenodo *antes* de taggear `v1.0.0`, para que el DOI se acuñe automaticamente al publicar el release. El README debe llevar el badge del DOI y `CITATION.cff` debe referenciarlo.
- **`data/sample/`:** con 2-3 arboles bastan ~5-10 MB. Incluir la cadena completa (mascara -> esqueleto -> grafo -> graph JSON -> flores JSON -> GT) para que el smoke test cubra el pipeline entero.

---

## 6. Orden de trabajo sugerido

**Fase 0 -- salvar el trabajo en riesgo (hacer ya, antes de reestructurar)**
1. Commitear los 5 archivos modificados.
2. Trackear `varilla_density_2.py`, `Varilla_heatmap_2.py`, `script_supremo.py`.
3. Copiar los 3 `gt_*.py` del Desktop al repo y commitear.
4. Taggear `v0.1-pre-restructure` como punto de retorno.

**Fase 1 -- desbloquear reproducibilidad**
5. `src/common/paths.py` + `configs/default.yaml`; eliminar las rutas absolutas.
6. `requirements.txt` con versiones fijadas (congelar del entorno py3.9 actual).
7. `data/sample/` con 2-3 arboles.
8. Sanear los ~30 `print()` no-ASCII.

**Fase 2 -- reestructurar**
9. Mover archivos con `git mv` (preserva historial).
10. Extraer `src/common/{io,geometry}.py` y desduplicar.

**Fase 3 -- cerrar el paper**
11. Implementar `src/density/flower_load_map.py` (flores/cm). **La brecha cientifica.**
12. Corregir el `branch_id` de euclidian/hibridos para que entren en la evaluacion.
13. Reescribir README en ingles + `LICENSE` + `CITATION.cff`.
14. `notebooks/` que reproduzcan figuras y tablas.
15. Release + DOI Zenodo.

---

## 7. Preguntas abiertas

1. **Idioma del repo:** ¿ingles completo (recomendado para Q1) o codigo en ingles + docs bilingues?
2. **Raiz del repo:** ¿promover a `IPre/` (recomendado) o mantener en `codigos/`?
3. **Metodo principal:** el paper dice que "proyeccion de clusters" es el mejor, es decir `varilla_density.py` (v1). Pero v2 (`varilla_density_2.py`) es mas nuevo y grande. ¿Confirmas que v1 es el metodo del paper y v2 es una variante comparada?
4. **Altura del arbol:** ¿los 350 cm asumidos son constantes para el dataset, o hay medicion por arbol? Afecta toda conversion a flores/cm.
5. **Hibridos:** ¿entran al paper como baselines (hay que arreglarles el `branch_id`) o quedan fuera?
6. **Renombrar "varilla":** ¿traducir a `shoot`/`fruiting_shoot` en el codigo, o conservar el termino en espanol por continuidad?
