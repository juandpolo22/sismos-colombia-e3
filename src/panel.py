"""Panel celda × mes con todas las columnas que usan los benchmarks y el modelo original.

Une las 18 variables base, las 13 de excitación, la historia de 12 meses (``h0`` … ``h11``), el mes objetivo y los
identificadores. Con ``incluir_siguiente=True`` agrega las filas del mes siguiente al último del catálogo, cuyo
objetivo aún no se conoce (``y`` = NaN): son las que el dashboard pronostica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import datos
from .original_model import EXCITACION, HISTORIA


def construir(corte: str = cfg.CORTE_PRUEBA, incluir_siguiente: bool = False) -> pd.DataFrame:
    """Panel en formato largo.

    Parameters
    ----------
    corte : str
        Fecha que define las celdas activas (≥ 20 sismos antes del corte). Por defecto 2021-01-01, las mismas
        163 celdas del Entregable 2.
    incluir_siguiente : bool
        Agrega las filas cuyo mes objetivo es el siguiente al último mes del catálogo (sin objetivo).

    Returns
    -------
    DataFrame con las columnas de :data:`src.original_model.COLUMNAS` más ``y``, ``celda``, ``fecha_obj``,
    ``persist`` y ``estacional``.
    """
    ev, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    n_tot = n_m + 1 if incluir_siguiente else n_m          # un mes vacío al final crea las filas a pronosticar
    X, info = datos.construir_panel(ev, n_tot, corte)
    XE = datos.variables_excitacion(ev, info)
    C, cc, tt = info["C"], info["cc"], info["tt"]
    P = pd.concat([X.reset_index(drop=True), XE[EXCITACION].reset_index(drop=True)], axis=1)
    for j in range(12):
        P[HISTORIA[j]] = C[cc, tt - j]
    P["mes"] = info["tm"]
    P["y"] = info["y"].astype(float)
    if incluir_siguiente:
        P.loc[P["mes"] >= n_m, "y"] = np.nan                # mes aún no observado
    P["celda"] = cc
    P["fecha_obj"] = datos.fecha_de_mes(info["tm"])
    P["persist"], P["estacional"] = info["persist"], info["estacional"]
    return P
