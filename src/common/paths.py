"""
Paths - Rutas del proyecto en un solo lugar
============================================
Antes cada script llevaba sus rutas absolutas escritas dentro
(C:\\Users\\vgara\\OneDrive\\Desktop\\IPre\\...), asi que el repositorio no
corria en ninguna maquina que no fuera la de desarrollo. Este modulo las
centraliza y las deduce de la ubicacion del propio archivo.

Como cambiar donde estan los datos, por orden de prioridad:

  1. Variable de entorno IPRE_DATA_DIR
         set IPRE_DATA_DIR=D:\\datos\\duraznos          (Windows, cmd)
         $env:IPRE_DATA_DIR = "D:\\datos\\duraznos"     (PowerShell)

  2. Archivo configs/paths.json en la raiz del repositorio:
         {"data_dir": "D:/datos/duraznos", "gt_dir": "D:/datos/GT"}

  3. Por defecto: la raiz del repositorio (donde estan las carpetas de datos
     hoy), y GT junto a ella.

Uso desde un script:

    from src.common.paths import GRAPH_JSON_DIR, FLOWERS_JSON_DIR
"""

import os
import json

__all__ = [
    "REPO_ROOT", "DATA_DIR", "GT_DIR",
    "MASKS_DIR", "SKELETONS_DIR", "GRAPHS_IMG_DIR", "GRAPH_JSON_DIR",
    "FLOWERS_JSON_DIR", "OUTPUT_DIR", "SAMPLE_DIR",
    "output_subdir", "describe",
]


# ---------------------------------------------------------------------------
#  Raiz del repositorio: dos niveles arriba de src/common/
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CONFIG_FILE = os.path.join(REPO_ROOT, "configs", "paths.json")


def _load_overrides():
    """Lee configs/paths.json si existe. Nunca falla: si el archivo esta
    corrupto se ignora, porque unas rutas por defecto son preferibles a que
    todos los scripts dejen de arrancar."""
    if not os.path.exists(_CONFIG_FILE):
        return {}
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_OVER = _load_overrides()


def _resolve(env_var, config_key, default):
    """Prioridad: variable de entorno > configs/paths.json > valor por defecto."""
    val = os.environ.get(env_var) or _OVER.get(config_key) or default
    return os.path.abspath(os.path.expanduser(str(val)))


# ---------------------------------------------------------------------------
#  Directorios raiz
# ---------------------------------------------------------------------------
# Donde viven las carpetas de datos del proyecto. Por defecto la raiz del
# repositorio, que es donde estan hoy (y estan excluidas por .gitignore).
DATA_DIR = _resolve("IPRE_DATA_DIR", "data_dir", REPO_ROOT)

# Anotaciones de ground truth (output de gt_annotator.py).
GT_DIR = _resolve("IPRE_GT_DIR", "gt_dir",
                  os.path.join(os.path.dirname(REPO_ROOT), "GT"))

# Resultados generados (figuras, CSV de metricas).
OUTPUT_DIR = _resolve("IPRE_OUTPUT_DIR", "output_dir",
                      os.path.join(DATA_DIR, "resultados"))


# ---------------------------------------------------------------------------
#  Etapas del pipeline
# ---------------------------------------------------------------------------
MASKS_DIR        = os.path.join(DATA_DIR, "Mascaras")
SKELETONS_DIR    = os.path.join(DATA_DIR, "Esqueletizacion")
GRAPHS_IMG_DIR   = os.path.join(DATA_DIR, "Grafos")
GRAPH_JSON_DIR   = os.path.join(DATA_DIR, "grafos json")
FLOWERS_JSON_DIR = os.path.join(DATA_DIR, "json flores")

# Imagenes de ejemplo versionadas en el repositorio (2-3 arboles).
SAMPLE_DIR = os.path.join(REPO_ROOT, "data", "sample")


def output_subdir(name, create=True):
    """Ruta de una subcarpeta de resultados, creandola si hace falta.

        OUT = output_subdir("densidad floral glee")
    """
    p = os.path.join(OUTPUT_DIR, name)
    if create:
        try:
            os.makedirs(p, exist_ok=True)
        except OSError:
            pass
    return p


def describe():
    """Imprime las rutas activas y si existen. Util para diagnosticar por que
    un script no encuentra los datos."""
    rows = [
        ("REPO_ROOT",        REPO_ROOT),
        ("DATA_DIR",         DATA_DIR),
        ("GT_DIR",           GT_DIR),
        ("OUTPUT_DIR",       OUTPUT_DIR),
        ("MASKS_DIR",        MASKS_DIR),
        ("SKELETONS_DIR",    SKELETONS_DIR),
        ("GRAPHS_IMG_DIR",   GRAPHS_IMG_DIR),
        ("GRAPH_JSON_DIR",   GRAPH_JSON_DIR),
        ("FLOWERS_JSON_DIR", FLOWERS_JSON_DIR),
    ]
    print("Rutas activas del proyecto")
    print("  fuente: IPRE_DATA_DIR > configs/paths.json > raiz del repositorio")
    print("")
    for name, path in rows:
        mark = "ok " if os.path.isdir(path) else "NO "
        print("  [%s] %-17s %s" % (mark, name, path))


if __name__ == "__main__":
    describe()
