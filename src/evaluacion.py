"""Métricas, intervalos de confianza y pruebas de comparación de pronósticos.

Unidad de remuestreo: el **mes objetivo**. Las celdas de un mismo mes comparten el mismo estado sísmico
(y los mismos sismos de vecindad), así que no son independientes; remuestrear filas daría intervalos
demasiado estrechos. Por eso el bootstrap remuestrea meses completos y las pruebas de pronóstico usan la
serie de pérdidas mensuales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from . import config as cfg


# ---------------------------------------------------------------------------
# Métricas de regresión
# ---------------------------------------------------------------------------
def rmsle(y, p) -> float:
    """Raíz del error cuadrático medio en escala log1p (métrica principal)."""
    p = np.clip(np.asarray(p, float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def metricas_regresion(y, p) -> dict:
    """RMSLE, RMSE, MAE, MAPE (solo sobre y > 0), sMAPE, R² y R² en escala log.

    El MAPE se indefine cuando y = 0 (77 % de las filas): se reporta restringido a y > 0 y se
    acompaña del sMAPE, que sí está definido con ceros (vale 0 cuando y = p = 0).
    """
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, None)
    pos = y > 0
    den = np.abs(y) + np.abs(p)
    smape = np.where(den > 0, 2 * np.abs(p - y) / np.where(den > 0, den, 1), 0.0)
    return {"RMSLE": rmsle(y, p), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": float(np.mean(np.abs(p - y))),
            "MAPE_y>0_%": float(100 * np.mean(np.abs(p[pos] - y[pos]) / y[pos])) if pos.any() else np.nan,
            "sMAPE_%": float(100 * np.mean(smape)), "R2": float(r2_score(y, p)),
            "R2_log": float(r2_score(np.log1p(y), np.log1p(p)))}


def perdida_mensual(y, p, tm) -> tuple:
    """Error cuadrático en escala log1p promediado por mes objetivo. Devuelve (meses, pérdidas)."""
    meses = np.unique(tm)
    pos = np.searchsorted(meses, tm)
    e = (np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2
    return meses, np.bincount(pos, e) / np.bincount(pos)


# ---------------------------------------------------------------------------
# Bootstrap por bloques de meses
# ---------------------------------------------------------------------------
def _agregados(y, p, tm):
    meses = np.unique(tm)
    pos = np.searchsorted(meses, tm)
    p = np.clip(p, 0, None)
    M = len(meses)
    return dict(n=np.bincount(pos, minlength=M).astype(float),
                sl=np.bincount(pos, (np.log1p(p) - np.log1p(y)) ** 2, M),
                s2=np.bincount(pos, (p - y) ** 2, M), s1=np.bincount(pos, np.abs(p - y), M),
                sy=np.bincount(pos, y, M), syy=np.bincount(pos, y ** 2, M))


def _met_desde_agregados(a, idx):
    n = a["n"][idx].sum(axis=-1)
    sse = a["s2"][idx].sum(axis=-1)
    sy, syy = a["sy"][idx].sum(axis=-1), a["syy"][idx].sum(axis=-1)
    sst = syy - sy ** 2 / n
    return {"RMSLE": np.sqrt(a["sl"][idx].sum(axis=-1) / n), "RMSE": np.sqrt(sse / n),
            "MAE": a["s1"][idx].sum(axis=-1) / n, "R2": 1 - sse / sst}


def bootstrap_meses(y, p, tm, B: int = 2000, alpha: float = 0.05, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Intervalos de confianza percentil y BCa remuestreando meses completos (bootstrap de bloques).

    Returns
    -------
    DataFrame con métrica, estimación puntual, IC percentil y IC BCa.
    """
    a = _agregados(np.asarray(y, float), np.asarray(p, float), tm)
    M = len(a["n"])
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, M, (B, M))
    boot = _met_desde_agregados(a, idx)
    todo = _met_desde_agregados(a, np.arange(M)[None, :])
    jack_idx = np.array([np.delete(np.arange(M), j) for j in range(M)])
    jack = _met_desde_agregados(a, jack_idx)
    filas = []
    for k in boot:
        th, bt, jk = float(todo[k][0]), boot[k], jack[k]
        z0 = stats.norm.ppf(np.clip((bt < th).mean(), 1e-6, 1 - 1e-6))
        d = jk.mean() - jk
        acc = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5 + 1e-300))
        zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
        q = stats.norm.cdf(z0 + (z0 + zs) / (1 - acc * (z0 + zs)))
        filas.append({"métrica": k, "valor": th, "IC_perc_inf": np.percentile(bt, 100 * alpha / 2),
                      "IC_perc_sup": np.percentile(bt, 100 * (1 - alpha / 2)),
                      "IC_BCa_inf": np.quantile(bt, q[0]), "IC_BCa_sup": np.quantile(bt, q[1])})
    return pd.DataFrame(filas)


def bootstrap_diferencia(y, p_a, p_b, tm, B: int = 2000, semilla: int = cfg.SEMILLA) -> dict:
    """Diferencia de RMSLE (A - B; negativo = A mejor) con IC por bootstrap de meses y p-valor bilateral."""
    a, b = _agregados(y, p_a, tm), _agregados(y, p_b, tm)
    M = len(a["n"])
    idx = np.random.default_rng(semilla).integers(0, M, (B, M))
    d = _met_desde_agregados(a, idx)["RMSLE"] - _met_desde_agregados(b, idx)["RMSLE"]
    d0 = float(_met_desde_agregados(a, np.arange(M)[None])["RMSLE"][0] - _met_desde_agregados(b, np.arange(M)[None])["RMSLE"][0])
    return {"dif": d0, "IC_inf": float(np.percentile(d, 2.5)), "IC_sup": float(np.percentile(d, 97.5)),
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


# ---------------------------------------------------------------------------
# Diebold-Mariano con corrección de Harvey-Leybourne-Newbold
# ---------------------------------------------------------------------------
def varianza_newey_west(d: np.ndarray, lag: int | None = None) -> float:
    """Varianza de largo plazo de una serie (núcleo de Bartlett)."""
    T = len(d)
    lag = int(np.floor(T ** (1 / 3))) if lag is None else lag
    dc = d - d.mean()
    v = dc @ dc / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    return max(v, 1e-18)


def diebold_mariano(perdida_a, perdida_b, h: int = 1) -> dict:
    """Prueba DM-HLN de igual capacidad predictiva.

    ``d_t = L_a(t) - L_b(t)``; H0: E[d] = 0. Positivo = B es mejor que A.
    """
    d = np.asarray(perdida_a, float) - np.asarray(perdida_b, float)
    T = len(d)
    dm = d.mean() / np.sqrt(varianza_newey_west(d) / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return {"dif_media": float(d.mean()), "DM_HLN": float(hln), "p": float(p), "T": T}


# ---------------------------------------------------------------------------
# Diagnóstico de residuos
# ---------------------------------------------------------------------------
def pruebas_residuos(residuos, ajustados) -> dict:
    """Normalidad (Jarque-Bera), heterocedasticidad (White y Breusch-Pagan) e independencia (BDS).

    Las pruebas de White y Breusch-Pagan usan como regresores los valores ajustados (y su cuadrado en
    White), la versión «especial» habitual cuando el modelo no es una regresión OLS.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.tsa.stattools import bds

    r, f = np.asarray(residuos, float), np.asarray(ajustados, float)
    exog = sm.add_constant(f)
    jb = stats.jarque_bera(r)
    w = het_white(r, exog)
    bp = het_breuschpagan(r, exog)
    rng = np.random.default_rng(cfg.SEMILLA)
    sub = r if len(r) <= 3000 else r[np.sort(rng.choice(len(r), 3000, replace=False))]
    b = bds(sub, max_dim=2)
    return {"JB_estad": float(jb.statistic), "JB_p": float(jb.pvalue), "asimetria": float(stats.skew(r)),
            "curtosis_exceso": float(stats.kurtosis(r)), "White_LM": float(w[0]), "White_p": float(w[1]),
            "BP_LM": float(bp[0]), "BP_p": float(bp[1]), "BDS_estad": float(np.atleast_1d(b[0])[0]),
            "BDS_p": float(np.atleast_1d(b[1])[0])}


def ljung_box(serie, lags=(1, 3, 6, 12)) -> pd.DataFrame:
    """Prueba de Ljung-Box de autocorrelación serial para varios rezagos."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return acorr_ljungbox(np.asarray(serie, float), lags=list(lags), return_df=True).reset_index().rename(columns={"index": "rezago"})
