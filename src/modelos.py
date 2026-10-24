"""Preprocesamiento (Pipeline) y modelos base.

El preprocesamiento sigue los hallazgos del EDA:

* **Conteos** (``n_*``, ``n35_*``, ``vec_*``, ``n_mismo_mes_anio_prev``): muy asimétricos y con atípicos
  reales (cap. 4 y 6) → ``log1p`` y estandarización (sin recortar).
* **Magnitudes de ventana**: faltan cuando no hubo sismos (faltante estructural, cap. 2) → imputación
  (la estrategia se elige por validación cruzada temporal en el cap. 10) y estandarización.
* **Profundidad media histórica**: 1,4 % faltante (celdas aún sin sismos) → mediana de entrenamiento.
* **Calendario y ubicación**: ya numéricas (componentes cíclicos; centro de celda) → estandarización.
* **Objetivo**: ``log1p`` dentro de ``TransformedTargetRegressor`` (cap. 3); las predicciones vuelven a
  la escala de conteos con ``expm1`` y se recortan en 0.

Todo se ajusta solo con los datos de entrenamiento del pliegue correspondiente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVR

from . import config as cfg

CONTEOS = ["n_1m", "n_3m", "n_6m", "n_12m", "n35_3m", "n35_12m", "vec_1m", "vec_3m", "n_mismo_mes_anio_prev"]
MAGNITUDES = ["mag_media_3m", "mag_media_12m", "mag_max_3m", "mag_max_12m"]
OTRAS = ["prof_media_hist"]
CICLICAS = ["sen_mes", "cos_mes"]
UBICACION = ["lat", "lon"]


def _expm1_recortado(x):
    return np.expm1(np.clip(x, -20, 20))


def preprocesador(columnas, imputacion_magnitud: str = "constante") -> ColumnTransformer:
    """ColumnTransformer del proyecto.

    Parameters
    ----------
    columnas : list
        Columnas disponibles (permite usar subconjuntos de variables).
    imputacion_magnitud : {"constante", "mediana"}
        ``"constante"`` imputa la magnitud mínima del catálogo (2,5: «no hubo sismos por encima del
        umbral»); ``"mediana"`` imputa la mediana de entrenamiento.
    """
    columnas = list(columnas)
    sel = lambda grupo: [c for c in grupo if c in columnas]  # noqa: E731
    imp_mag = (SimpleImputer(strategy="constant", fill_value=cfg.M_MIN) if imputacion_magnitud == "constante"
               else SimpleImputer(strategy="median"))
    partes = []
    if sel(CONTEOS):
        partes.append(("conteos", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                            ("escala", StandardScaler())]), sel(CONTEOS)))
    if sel(MAGNITUDES):
        partes.append(("magnitudes", Pipeline([("imputa", imp_mag), ("escala", StandardScaler())]), sel(MAGNITUDES)))
    if sel(OTRAS):
        partes.append(("profundidad", Pipeline([("imputa", SimpleImputer(strategy="median")), ("escala", StandardScaler())]), sel(OTRAS)))
    resto = sel(CICLICAS) + sel(UBICACION) + [c for c in columnas if c not in CONTEOS + MAGNITUDES + OTRAS + CICLICAS + UBICACION]
    if resto:
        partes.append(("resto", Pipeline([("imputa", SimpleImputer(strategy="median")), ("escala", StandardScaler())]), resto))
    return ColumnTransformer(partes, remainder="drop", verbose_feature_names_out=False)


def con_objetivo_log(estimador) -> TransformedTargetRegressor:
    """Entrena sobre log(1 + y) y devuelve predicciones en conteos (expm1, recortadas en 0 al evaluar)."""
    return TransformedTargetRegressor(regressor=estimador, func=np.log1p, inverse_func=_expm1_recortado, check_inverse=False)


def svr_lineal(columnas, C: float = 0.1, epsilon: float = 0.0, imputacion_magnitud: str = "constante",
               semilla: int = cfg.SEMILLA) -> TransformedTargetRegressor:
    """Modelo base del enunciado (ruta C/D): SVR lineal dentro del Pipeline completo."""
    pipe = Pipeline([("prep", preprocesador(columnas, imputacion_magnitud)),
                     ("svr", LinearSVR(C=C, epsilon=epsilon, loss="epsilon_insensitive", dual="auto",
                                       max_iter=20000, random_state=semilla))])
    return con_objetivo_log(pipe)


# ---------------------------------------------------------------------------
# Líneas base triviales (no aprenden más que una constante o una tabla)
# ---------------------------------------------------------------------------
class MediaPorCelda(BaseEstimator, RegressorMixin):
    """Línea base «media por zona»: media del objetivo de cada celda en el entrenamiento.

    Necesita la columna de identificación de celda ``celda`` (se pasa aparte en ``fit``/``predict``).
    Para celdas no vistas usa la media global.
    """

    def fit(self, celdas, y):
        """Guarda la media histórica del objetivo en cada celda y la media global.

        Returns
        -------
        self
        """
        s = pd.Series(np.asarray(y, float)).groupby(np.asarray(celdas)).mean()
        self.tabla_ = s.to_dict()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, celdas):
        """Media histórica de cada celda; la media global para celdas no vistas."""
        return np.array([self.tabla_.get(c, self.global_) for c in np.asarray(celdas)])
