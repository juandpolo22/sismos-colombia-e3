"""Explicaciones SHAP de los modelos benchmark (escala log1p, la misma en que se entrenan).

* Modelos de árboles (XGBoost, Random Forest, árbol de decisión): TreeSHAP exacto sobre el modelo interno del
  pipeline, con las variables ya preprocesadas (cada variable preprocesada corresponde a una sola variable original).
* Resto (KNN, SVR, Ridge, Lasso): SHAP por permutaciones sobre el pipeline completo con un fondo de referencia.

El modelo original A+B se explica con :meth:`src.original_model.ModeloOriginalAB.explicar` (aporte de cada miembro).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

ARBOLES = ("XGBRegressor", "RandomForestRegressor", "DecisionTreeRegressor")


def _interno(mod):
    pipe = mod.regressor_
    return pipe.named_steps["prep"], pipe.named_steps["modelo"]


def shap_filas(mod, X: pd.DataFrame, fondo: pd.DataFrame, max_evals: int = 400, n_fondo: int = 100) -> np.ndarray:
    """Valores SHAP (n × p, en el orden de las columnas de X) del pronóstico log1p."""
    prep, est = _interno(mod)
    if type(est).__name__ in ARBOLES:
        Xt = prep.transform(X)
        nombres = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        S = np.asarray(shap.TreeExplainer(est).shap_values(Xt))
        return pd.DataFrame(S, columns=nombres)[list(X.columns)].to_numpy()
    f = lambda Z: np.log1p(np.clip(mod.predict(pd.DataFrame(Z, columns=X.columns)), 0, None))  # noqa: E731
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(fondo.to_numpy(float), max_samples=n_fondo), seed=0)
    return expl(X.to_numpy(float), max_evals=max_evals, silent=True).values


def valor_base(mod, fondo: pd.DataFrame) -> float:
    """Pronóstico log1p medio sobre el fondo (punto de partida de la explicación)."""
    return float(np.mean(np.log1p(np.clip(mod.predict(fondo), 0, None))))
