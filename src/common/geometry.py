"""
Geometry - Utilidades geometricas compartidas por todo el pipeline
==================================================================
Este modulo existe para que la calibracion px <-> cm este definida en UN
solo lugar. Antes estaba reescrita en cinco scripts distintos; si una de
esas copias se editaba y las demas no, los metodos pasaban a medir con
escalas diferentes y dejaban de ser comparables entre si, sin que nada
fallara de forma visible.
"""

__all__ = ["estimate_pixels_per_cm"]


def estimate_pixels_per_cm(graph_data, assumed_tree_height_cm):
    """
    Calibra px <-> cm midiendo la altura vertical del esqueleto.

    Asume que la extension vertical (max y - min y) de TODOS los pixeles de
    las ramas representa la altura total del arbol, y que esa altura es
    ~assumed_tree_height_cm (3-4 m para un duraznero adulto). Esto da una
    escala POR IMAGEN, robusta a zoom y distancia de camara.

    Devuelve 1.0 (escala neutra) si el grafo tiene menos de dos pixeles, si
    el arbol resulta degenerado en altura, o si la altura asumida no es
    positiva.

    Parametros
    ----------
    graph_data : dict
        JSON de grafo ya cargado. Se leen los pixeles de graph_data['branches'],
        en formato [y, x].
    assumed_tree_height_cm : float
        Altura real asumida del arbol, en centimetros.

    Retorna
    -------
    float
        Pixeles por centimetro.
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
