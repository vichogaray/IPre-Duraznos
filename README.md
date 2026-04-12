# IPre Duraznos — Análisis de Ramas y Densidad Floral

Pipeline completo para el análisis estructural de árboles de durazno a partir de imágenes: desde la esqueletización de máscaras binarias hasta la asignación de flores a ramas mediante métodos de distancia euclidiana y Graph Laplacian semi-supervisado.

---

## Pipeline general

```
Máscaras binarias
      │
      ▼
[1] Esqueletización          skeletonize_batch.py  /  param_tuner.py
      │
      ▼
[2] Reparación y corte       skeleton_repair.py  /  skeleton_splitter.py
      │
      ▼
[3] Grafo de ramas coloreado skeleton_graph_viewer.py  →  Grafos/
      │
      ▼
[4] Clasificación jerárquica branch_identifier2.py
      │
      ├──────────────────────────────────────────┐
      ▼                                          ▼
[5a] Densidad Euclidiana               [5b] Densidad Laplacian
     floral_density.py                      build_graph_json.py
     (distancia mínima pixel)               laplacian_density.py
                                            (semi-supervisado)
```

---

## Descripción de cada script

### Esqueletización

| Script | Descripción |
|---|---|
| `skeletonize_batch.py` | Esqueletización masiva con binarización Otsu, suavizado morfológico, filtro por grosor vía transformada de distancia y pruning de ramas cortas. Soporta modo adaptativo (parámetros automáticos por imagen). |
| `skeletonize_simple.py` | Versión mínima para probar una imagen individual. |
| `param_tuner.py` | Interfaz gráfica con sliders para ajustar parámetros de esqueletización imagen por imagen. Guarda configuraciones en `parametros.json`. |

### Procesamiento del esqueleto

| Script | Descripción |
|---|---|
| `skeleton_repair.py` | Detecta endpoints donde falta exactamente 1 píxel para reconectar el esqueleto. Modo batch o interactivo (con botones Aceptar/Rechazar por gap). |
| `skeleton_splitter.py` | Herramienta interactiva para dividir un esqueleto en dos partes trazando una polilínea de corte libre. |
| `skeleton_graph_viewer.py` | Convierte un esqueleto binario en un grafo de ramas coloreado. Permite hacer merge de ramas, eliminar ramas y exportar el PNG final. Soporta modo batch. |

### Identificación y clasificación de ramas

| Script | Descripción |
|---|---|
| `branch_identifier.py` | Primera versión. Detecta ramas trazando pixel a pixel desde junctions y endpoints mediante número de cruce (CN). |
| `branch_identifier2.py` | Versión actual. Agrupa píxeles por color RGB, detecta adyacencia espacial entre ramas, identifica el tronco por posición vertical y clasifica toda la estructura en jerarquía (Tronco → Rama Principal → Secundaria → ...) mediante BFS. Usado como librería por otros scripts. |

### Densidad floral

| Script | Descripción |
|---|---|
| `build_graph_json.py` | Genera un JSON por imagen con la estructura completa del grafo (nodos, aristas, píxeles por rama y centroides). Input requerido para `laplacian_density.py`. |
| `floral_density.py` | Asigna cada flor a la rama más cercana por distancia euclidiana mínima sobre todos los píxeles de la rama. Genera visualización con conteo por rama y nivel jerárquico. |
| `laplacian_density.py` | Asigna flores a ramas modelando el grafo como problema de clasificación semi-supervisada (Orduz 2019). Las ramas son nodos etiquetados, las flores nodos sin etiqueta; se minimiza una función de pérdida softmax + suavidad Laplaciana via L-BFGS-B. |

### Utilidades

| Script | Descripción |
|---|---|
| `esqueletizacion.py` | Script exploratorio con implementación manual del adelgazamiento Zhang-Suen y comparación con `skimage.skeletonize`. |
| `copy_originals.py` | Copia imágenes originales que corresponden a un conjunto de grafos, usando el número de frame como clave. |

---

## Estructura de carpetas esperada

```
IPre/
├── codigos/             ← scripts (este repositorio)
├── MASKS/               ← máscaras binarias originales
├── Mascaras filtradas/  ← máscaras preprocesadas
├── Esqueletos filtrados/← esqueletos generados por skeletonize_batch
├── Esqueletos sin grafo/← esqueletos reparados
├── Grafos/              ← imágenes de grafo coloreado (PNG)
├── grafos json/         ← JSONs de grafo (generados por build_graph_json)
├── json flores/         ← JSONs de flores en formato LabelMe
├── densidad floral/     ← salida de floral_density.py
└── densidad floral laplacian/ ← salida de laplacian_density.py
```

Los JSONs de flores siguen el formato [LabelMe](https://github.com/labelmeai/labelme), con `label: "flower"` y un punto por flor.

---

## Dependencias

```
numpy
opencv-python
scikit-image
scikit-learn
scipy
matplotlib
```

Instalar con:

```bash
pip install numpy opencv-python scikit-image scikit-learn scipy matplotlib
```

---

## Convención de nombres

Los archivos de imagen siguen el patrón `imgs_frameXX_00000.png`. Los scripts usan el número de frame para emparejar automáticamente la imagen del grafo con su JSON de flores correspondiente (`frameXX.json`).

---

## Referencias

- Orduz, J. (2019). *Semi-supervised clustering with graph Laplacian*. [juanitorduz.github.io](https://juanitorduz.github.io/semi_supervised_clustering/)
- Zhang, T. Y., & Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3), 236–239.
