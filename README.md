# IPre Duraznos — Carga floral en duraznos por asignacion geometrica estructural

Pipeline de vision por computador para estimar la **carga floral** de arboles de
durazno (*Prunus persica*) a partir de imagenes RGB individuales, a escala de
arbol y de rama.

Codigo asociado al articulo *RGB image processing of trees for flower load
across tree and branch scales by structure-aware geometric assignment*.

> **Estado:** repositorio en preparacion para publicacion. Ver
> [`docs/AUDITORIA_REPO.md`](docs/AUDITORIA_REPO.md) para el plan de trabajo
> pendiente.

---

## Idea central: las varillas

Las flores del duraznero **no nacen de las ramas visibles** (tronco, primarias,
secundarias) sino de **varillas** (ramos mixtos de 30–60 cm) que crecen del
esqueleto y **no son visibles** en la imagen: son demasiado delgadas y quedan
ocultas entre las flores.

El problema central del metodo es inferir donde estan esas varillas invisibles a
partir de como se disponen las flores, y conectar cada varilla a su rama madre.

---

## Pipeline

```
Mascaras binarias
      |
      v
[1] Esqueletizacion              src/preprocessing/
      |                          skeletonize.py
      |                          Otsu, suavizado morfologico, Zhang-Suen,
      v                          pruning, reparacion de gaps
[2] Grafo jerarquico de ramas    src/preprocessing/
      |                          build_skeleton_graph.py, branch_hierarchy.py,
      |                          export_graph_json.py
      v                          topologia -> nodos, aristas, orden de
      |                          ramificacion via BFS desde el tronco
      v
[3] Asignacion flor -> rama      src/assignment/
      |                          metodos morfologicos y de reconstruccion
      v                          de brotes
[4] Mapa de carga floral         src/heatmap/
      |
      v
[5] Evaluacion contra GT         src/evaluation/
```

---

## Estructura del repositorio

```
.
├── src/
│   ├── preprocessing/           Esqueletizacion y grafo de ramas
│   │   ├── skeletonize.py           Pipeline masivo de esqueletizacion
│   │   ├── tune_skeleton_params.py  GUI de calibracion de parametros
│   │   ├── build_skeleton_graph.py  Esqueleto -> grafo coloreado (GUI + batch)
│   │   ├── branch_hierarchy.py      Clasificacion por orden de ramificacion
│   │   └── export_graph_json.py     Exporta el grafo a JSON (hub del pipeline)
│   │
│   ├── assignment/              Asignacion flor -> rama
│   │   ├── morphological/           Familia morfologica
│   │   │   ├── euclidean.py             Distancia euclidiana (baseline)
│   │   │   ├── graph_laplacian.py       Graph Laplacian semi-supervisado
│   │   │   ├── random_walk.py           Random Walk absorbente
│   │   │   ├── glee.py                  GLEE (Geometric Laplacian Eigenmap)
│   │   │   ├── hybrid_laplacian.py      Euclidiana + Laplaciano (ambiguas)
│   │   │   └── hybrid_random_walk.py    Euclidiana + Random Walk (ambiguas)
│   │   │
│   │   └── shoot_reconstruction/    Familia de reconstruccion de brotes
│   │       ├── cluster_projection.py    Proyeccion de clusters [PRINCIPAL]
│   │       ├── candidate_shoots.py      Brotes candidatos (rayos desde tronco)
│   │       └── lsystem.py               Varillas generadas con L-System
│   │
│   ├── heatmap/                 Mapas de carga floral
│   │   ├── heatmap_shoots.py        Heatmap del esqueleto (metodo principal)
│   │   ├── heatmap_shoots_v2.py     Heatmap para brotes candidatos
│   │   └── heatmap_random_walk.py   Heatmap por subsecciones de rama
│   │
│   └── evaluation/              Validacion contra ground truth
│       ├── gt_annotator.py              GUI de anotacion del GT
│       ├── heatmap_wasserstein.py       Wasserstein geodesico GT vs. metodo
│       └── flower_branch_accuracy.py    Accuracy flor->rama multi-metodo
│
├── docs/                        Documentacion del metodo
│   ├── PIPELINE.md                  Descripcion detallada y formatos de datos
│   └── AUDITORIA_REPO.md            Auditoria y plan de trabajo
└── requirements.txt
```

**El dataset completo (292 arboles) no esta en este repositorio**: se deposita
en Zenodo. Las carpetas de datos locales (`IMGS/`, `Mascaras/`, `Grafos/`,
`grafos json/`, `json flores/`, `densidades/`) estan excluidas via `.gitignore`.

---

## Instalacion

Requiere **Python 3.9** (probado en 3.9.11, Windows 11).

```bash
git clone https://github.com/vichogaray/IPre-Duraznos.git
cd IPre-Duraznos
pip install -r requirements.txt
```

---

## Uso

Cada script tiene un bloque `CONFIGURACION` al inicio con las rutas y
parametros. Se ejecutan directamente (F5 en el editor, o `python <script>`).

```bash
# 1. Esqueletizar las mascaras binarias
python src/preprocessing/skeletonize.py

# 2. Construir el grafo de ramas y exportarlo a JSON
python src/preprocessing/build_skeleton_graph.py
python src/preprocessing/export_graph_json.py

# 3. Asignar flores a ramas (metodo principal)
python src/assignment/shoot_reconstruction/cluster_projection.py

# 4. Generar el mapa de carga floral
python src/heatmap/heatmap_shoots.py

# 5. Evaluar contra el ground truth
python src/evaluation/flower_branch_accuracy.py
```

> **Nota:** las rutas de entrada/salida estan actualmente hardcodeadas en cada
> script y apuntan a carpetas locales de la maquina de desarrollo, por lo que
> los scripts no se ejecutan tal cual tras clonar el repositorio.
> Centralizarlas es la principal tarea pendiente (ver la auditoria).

---

## Formatos de datos

| Formato | Convencion de coordenadas |
|---|---|
| JSON de grafo — `branches[i].pixels` | `[y, x]` (fila, columna) |
| JSON de flores (LabelMe) — `shapes[i].points` | `[x, y]` |

En coordenadas de imagen, **+Y apunta hacia abajo** (direccion de la gravedad).
Detalle completo en [`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## Resultados preliminares

Evaluacion de `src/evaluation/flower_branch_accuracy.py` sobre las **4 imagenes
con ground truth anotado a mano** (frames 32, 46, 64, 172):

| Metodo | Accuracy flor->rama | Similitud de forma (Wasserstein) |
|---|---|---|
| random_walk | 78.1 % | - |
| glee | 77.7 % | - |
| laplacian | 71.3 % | - |
| varilla_density (proyeccion de clusters) | 64.4 % | 0.898 |
| varilla_density_2 (brotes candidatos) | 53.0 % | 0.874 |

Dos metricas complementarias: **accuracy** mide cuantas flores reciben la rama
correcta; **similitud de forma** compara donde se concentra la carga floral a lo
largo del esqueleto, y solo se calcula para los metodos que reconstruyen
varillas.

> **La accuracy NO permite comparar familias de metodos entre si.** El GT si
> anota la rama madre de cada flor, siguiendo la varilla a ojo, pero lo hace de
> forma **agregada**: cada clic marca una base de varilla sobre el esqueleto y
> registra cuantas flores cuelgan de ella (p. ej. frame46: 110 flores
> distribuidas en 18 puntos). Queda anotado *cuantas* flores nacen de cada
> rama, no *cuales*.
>
> Para calcular la accuracy hay que desagregar eso, y `gt_branch_per_flower`
> lo hace emparejando cada flor con el punto-GT **mas cercano en linea recta**.
> Ahi entra el sesgo: los puntos estan en el esqueleto y las flores al extremo
> de varillas de 30-60 cm, asi que una flor rara vez esta cerca de su propia
> base. El emparejamiento por proximidad reproduce el criterio de los metodos
> morfologicos, que quedan favorecidos, y penaliza a los de varilla cuando
> asignan la flor a la rama de la que realmente nace. La similitud de forma no
> depende de esa desagregacion y ordena los metodos al reves.
>
> **Preliminar.** 4 arboles anotados sobre 292 es una base pequena para
> sostener conclusiones: ampliar el ground truth es trabajo pendiente. Los
> metodos `euclidean` y los hibridos quedan fuera porque detectan ramas por
> color del PNG y sus `branch_id` no se alinean con el grafo ni con el GT.

---

## Pendiente

- [ ] Mapa de densidad en flores·cm⁻¹ (`src/heatmap/flower_load_map.py`)
- [ ] Centralizar rutas y parametros en `configs/`
- [x] ~~Unificar `estimate_pixels_per_cm`~~ -> `src/common/geometry.py`. Estaba
      reescrita en 5 scripts; ahora hay una sola definicion. Verificado sobre 40
      imagenes: los 200 valores son identicos a los de antes del cambio
- [ ] Unificar tambien `load_flowers` (x10) y `load_graph_json` (x8). Menos
      urgente: son lectores de JSON y una divergencia haria fallar el script de
      inmediato, no falsear resultados en silencio
- [ ] `data/sample/` con 2–3 arboles de ejemplo
- [ ] `LICENSE` y `CITATION.cff`
- [ ] Enlace al dataset en Zenodo (DOI)
- [x] ~~Arreglar `src/heatmap/heatmap_shoots.py:38` (`SINGLE_IMAGE` sin valor)~~

---

## Referencias

- Orduz, J. (2019). *Semi-supervised clustering with graph Laplacian*.
  [juanitorduz.github.io](https://juanitorduz.github.io/semi_supervised_clustering/)
- Grady, L. (2006). Random walks for image segmentation. *IEEE TPAMI*, 28(11), 1768–1783.
- Torres, L., Chan, K. S., & Eliassi-Rad, T. (2020). GLEE: Geometric Laplacian
  Eigenmap Embedding. *Journal of Complex Networks*, 8(2).
- Zhang, T. Y., & Suen, C. Y. (1984). A fast parallel algorithm for thinning
  digital patterns. *Communications of the ACM*, 27(3), 236–239.
