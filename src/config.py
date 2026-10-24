"""Configuración global del proyecto: rutas, semilla y constantes del problema.

Todas las decisiones numéricas del diseño (umbral de magnitud, tamaño de celda, corte de prueba,
gap, etc.) viven aquí para que el Jupyter Book, los scripts y el dashboard usen exactamente los mismos
valores.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Semilla global (se propaga a numpy, random, scikit-learn, XGBoost, Optuna y DEAP)
# ---------------------------------------------------------------------------
SEMILLA = 42

# ---------------------------------------------------------------------------
# Definición del problema
# ---------------------------------------------------------------------------
M_MIN = 2.5                 # magnitud mínima de los sismos que se cuentan
CELDA_GRADOS = 0.5          # tamaño de la celda espacial (grados)
LAT_MIN, LAT_MAX = -5.0, 14.0
LON_MIN, LON_MAX = -82.0, -66.0
INICIO = "2004-01-01"       # primer día del catálogo usado
CORTE_PRUEBA = "2021-01-01" # primer mes objetivo del conjunto de prueba
CORTES_ROBUSTEZ = ("2018-01-01", "2021-01-01", "2023-01-01")
GAP_MESES = 1               # meses objetivo entre el final del entrenamiento y la prueba
MIN_EVENTOS_CELDA = 20      # sismos mínimos en entrenamiento para que una celda sea "activa"
HISTORIA_MESES = 12         # meses de historia necesarios para construir las variables
N_PLIEGUES = 5              # pliegues de la validación cruzada temporal (ventana creciente)

# Rutas
RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data"
CRUDOS = DATOS / "crudos"
CATALOGO = DATOS / "sismos_sgc_2004_2026.csv"
TABLAS = RAIZ / "tables"
FIGURAS = RAIZ / "figures"
RESULTADOS = RAIZ / "resultados"

FUENTE_URL = "http://www.sismologia.sgc.gov.co/"
FUENTE_CATALOGO_NUEVO = "https://www.sgc.gov.co/sismos"
LICENCIA = ("Catálogo de consulta pública de la Red Sismológica Nacional de Colombia (SGC). No se encontró una "
            "licencia específica (p. ej. Creative Commons) para el catálogo; los términos generales de uso de "
            "información del SGC exigen citar la fuente y prohíben comercializar o redistribuir los datos sin "
            "autorización escrita. Uso aquí: académico, sin fines comerciales y con atribución.")
LICENCIA_URL = "https://www2.sgc.gov.co/sgc/mapas/Documents/CondGenTermLicUso.pdf"


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fija la semilla en ``random``, ``numpy`` y la variable ``PYTHONHASHSEED``.

    Parameters
    ----------
    semilla : int
        Valor de la semilla global. scikit-learn, XGBoost, Optuna y DEAP la reciben explícitamente
        como ``random_state``/``seed`` en cada llamada (ver ``src.modelos``).
    """
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)


for _d in (TABLAS, FIGURAS, RESULTADOS):
    _d.mkdir(parents=True, exist_ok=True)
