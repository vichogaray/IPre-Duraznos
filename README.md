# IPre Duraznos — Análisis de Ramas y Densidad Floral

Pipeline de visión por computador para el análisis estructural de árboles de
durazno (*Prunus persica*) a partir de imágenes: desde la esqueletización de
máscaras binarias hasta la estimación de densidad floral.

El enfoque principal es el método de **varillas**: las flores del duraznero no
nacen de las ramas grandes visibles (tronco, primarias, secundarias) sino de
*varillas* (ramos mixtos de 30–60 cm) que crecen del esqueleto y no son
visibles en la imagen. El pipeline infiere esas varillas a partir de la
disposición de las flores y las conecta a su rama madre.

---

## Pipeline general

```
Máscaras binarias
      │
      ▼
[1] Esqueletización            Esqueletizacion.py  /  Param_tuner_esqueletizacion.py
      │                        (incluye reparación automática de gaps de 1px)
      ▼
[2] Grafo de ramas coloreado   Grafo.py
      │
      ▼
[3] Estructura de ramas        build_graph_json.py  (usa Identificador_de_ramas.py)
      │                        -> JSON con nodos, aristas, jerarquía y píxeles
      ▼
[4] Densidad floral            varilla_density.py  (+ otros métodos)
      │
      ▼
[5] Visualización              Varilla_heatmap.py  /  RW_heatmap.py
```

---

## Descripción de los scripts

### Esqueletización

| Script | Descripción |
|---|---|
| `Esqueletizacion.py` | Esqueletización masiva de máscaras binarias: binarización Otsu, suavizado morfológico, filtro por grosor vía transformada de distancia, pruning de ramas cortas y reparación automática de gaps de 1 píxel. Soporta modo adaptativo (parámetros automáticos por imagen) y parámetros manuales desde `parametros.json`. |
| `Param_tuner_esqueletizacion.py` | Interfaz gráfica con sliders para ajustar los parámetros de esqueletización imagen por imagen. Guarda las configuraciones en `parametros.json`, que `Esqueletizacion.py` lee para usar parámetros manuales. |

### Grafo de ramas

| Script | Descripción |
|---|---|
| `Grafo.py` | Convierte un esqueleto binario en un grafo de ramas coloreado. Traza las ramas por topología (número de cruce) y les asigna colores. Interfaz interactiva para hacer merge de ramas, eliminar ramas y exportar el PNG. Soporta modo batch. |
| `Identificador_de_ramas.py` | Librería de identificación y clasificación de ramas. Agrupa píxeles por color, detecta adyacencia espacial entre ramas, identifica el tronco y clasifica la estructura en jerarquía (Tronco → Rama Principal → Secundaria → ...) mediante BFS. Usada por otros scripts. |
| `build_graph_json.py` | Lee cada PNG de grafo coloreado, extrae la estructura con `Identificador_de_ramas`, y guarda un JSON por imagen con nodos, aristas, jerarquía y píxeles por rama. Estos JSON son el input de todos los métodos de densidad. |

### Densidad floral

| Script | Descripción |
|---|---|
| `varilla_density.py` | **Método principal.** Asigna flor → varilla → rama madre. Agrupa flores en clusters lineales (DBSCAN anisotrópico), ajusta una varilla por cluster (regresión OLS + curvatura opcional), extiende la base hasta el esqueleto y la proyecta a su rama madre. |
| `euclidian_density.py` | Asigna cada flor a la rama más cercana por distancia euclidiana mínima sobre los píxeles de la rama. |
| `laplacian_density.py` | Asigna flores a ramas como un problema de clasificación semi-supervisada con Graph Laplacian (Orduz 2019), minimizando pérdida softmax + suavidad Laplaciana. |
| `glee_density.py` | Asignación de flores a ramas mediante embeddings de grafo (GLEE). |
| `random_walk_density.py` | Asignación de flores a ramas mediante Random Walk sobre el grafo (Grady 2006). |
| `hybridlaplac_density.py` | Método híbrido basado en Graph Laplacian, con análisis de ambigüedad. |
| `hybridrw_density.py` | Método híbrido basado en Random Walk, con análisis de ambigüedad. |

### Visualización

| Script | Descripción |
|---|---|
| `Varilla_heatmap.py` | Mapa de calor del esqueleto basado en el modelo de varillas. Para cada píxel del esqueleto acumula la carga floral de las varillas cercanas (kernel gaussiano) y lo colorea con un espectro de calor. Usa el output de `varilla_density.py`. |
| `RW_heatmap.py` | Mapa de calor de densidad floral por subsecciones de rama, asignando flores con Random Walk. |


## Dependencias

```
numpy
opencv-python
scikit-image
scikit-learn
scipy
matplotlib
gensim
networkx
```

Instalar con:

```bash
pip install numpy opencv-python scikit-image scikit-learn scipy matplotlib gensim networkx
```

---

## Referencias

- Orduz, J. (2019). *Semi-supervised clustering with graph Laplacian*. [juanitorduz.github.io](https://juanitorduz.github.io/semi_supervised_clustering/)
- Grady, L. (2006). Random walks for image segmentation. *IEEE TPAMI*, 28(11), 1768–1783.
- Zhang, T. Y., & Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3), 236–239.
