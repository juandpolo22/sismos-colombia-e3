"""Carga y caché de los datos del dashboard. Todo se lee una sola vez por proceso (``lru_cache``).

Los archivos los genera ``scripts/preparar_datos_app.py`` en ``data/app/``; el servidor no entrena ni recalcula
modelos.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as cfg
from .datos import VARIABLES_BASE

APP = cfg.DATOS / "app"

DESCRIPCION = {
    "n_1m": "Sismos M ≥ 2,5 en la celda en el último mes", "n_3m": "Sismos en la celda en los últimos 3 meses",
    "n_6m": "Sismos en la celda en los últimos 6 meses", "n_12m": "Sismos en la celda en los últimos 12 meses",
    "n35_3m": "Sismos M ≥ 3,5 en los últimos 3 meses", "n35_12m": "Sismos M ≥ 3,5 en los últimos 12 meses",
    "mag_media_3m": "Magnitud media en 3 meses (faltante si no hubo sismos)", "mag_media_12m": "Magnitud media en 12 meses",
    "mag_max_3m": "Magnitud máxima en 3 meses", "mag_max_12m": "Magnitud máxima en 12 meses",
    "prof_media_hist": "Profundidad media histórica de la celda (km)", "vec_1m": "Sismos en las 8 celdas vecinas el último mes",
    "vec_3m": "Sismos en las 8 celdas vecinas en 3 meses", "n_mismo_mes_anio_prev": "Sismos de la celda el mismo mes del año anterior",
    "sen_mes": "Seno del mes objetivo (estacionalidad)", "cos_mes": "Coseno del mes objetivo", "lat": "Latitud del centro de la celda",
    "lon": "Longitud del centro de la celda",
}
UNIDADES = {**{k: "sismos" for k in ("n_1m", "n_3m", "n_6m", "n_12m", "n35_3m", "n35_12m", "vec_1m", "vec_3m",
                                      "n_mismo_mes_anio_prev")},
            **{k: "magnitud" for k in ("mag_media_3m", "mag_media_12m", "mag_max_3m", "mag_max_12m")},
            "prof_media_hist": "km", "sen_mes": "—", "cos_mes": "—", "lat": "grados", "lon": "grados"}


@lru_cache(maxsize=1)
def catalogo() -> pd.DataFrame:
    """Catálogo limpio del SGC (una fila por sismo)."""
    c = pd.read_csv(cfg.CATALOGO, parse_dates=["time"])
    c["mes"] = c.time.dt.to_period("M").dt.to_timestamp()
    ultimo = c.time.max()
    if (ultimo + pd.Timedelta(days=1)).month == ultimo.month:          # el último mes aún no termina: se descarta
        c = c[c.mes < c.mes.max()]
    return c.reset_index(drop=True)


@lru_cache(maxsize=1)
def panel() -> pd.DataFrame:
    """Panel celda × mes (incluye las filas del mes a pronosticar, con ``y`` vacío)."""
    P = pd.read_pickle(APP / "panel.pkl.gz")
    P["depto"] = P.celda.map(depto_celda())
    return P


@lru_cache(maxsize=1)
def depto_celda() -> dict:
    """Departamento más frecuente de los sismos de cada celda (para filtrar y rotular)."""
    c = catalogo()
    from .datos import asignar_celda
    celda = asignar_celda(c.latitude, c.longitude, cfg.CELDA_GRADOS)
    return pd.Series(c.depto.fillna("SIN DATO").values, index=celda).groupby(level=0).agg(lambda s: s.value_counts().index[0]).to_dict()


@lru_cache(maxsize=1)
def oos() -> dict:
    """Predicciones fuera de muestra del origen móvil 2021–2026 (y, mes, celda y una columna por modelo)."""
    z = np.load(APP / "oos.npz", allow_pickle=False)
    nombres = [str(n) for n in z["nombres"]]
    return {"y": z["y"], "tm": z["tm"], "celda": z["celda"], **{n: z[f"m{i}"] for i, n in enumerate(nombres)}, "_nombres": nombres}


@lru_cache(maxsize=1)
def lideres() -> pd.DataFrame:
    return pd.read_csv(APP / "lideres.csv")


@lru_cache(maxsize=1)
def pronostico() -> pd.DataFrame:
    return pd.read_csv(APP / "pronostico.csv", parse_dates=["fecha_obj"])


@lru_cache(maxsize=1)
def intervalos() -> dict:
    return json.load(open(APP / "intervalos.json", encoding="utf-8"))


@lru_cache(maxsize=1)
def shap_global() -> dict:
    return json.load(open(APP / "shap_global.json", encoding="utf-8"))


@lru_cache(maxsize=1)
def original() -> dict:
    return json.load(open(APP / "original.json", encoding="utf-8"))


@lru_cache(maxsize=1)
def fondo_shap() -> pd.DataFrame:
    return pd.read_pickle(APP / "fondo_shap.pkl.gz")


@lru_cache(maxsize=None)
def tabla(nombre: str) -> pd.DataFrame:
    return pd.read_csv(APP / nombre)


def modelos() -> list:
    """Etiquetas de los modelos del dashboard (A+B primero)."""
    return lideres().etiqueta.tolist()[-1:] + lideres().etiqueta.tolist()[:-1]


# Límites físicos de cada variable (fuera de ellos la entrada es inválida). El rango observado en el entrenamiento se
# usa aparte: un valor válido pero fuera de lo observado se acepta con una advertencia de extrapolación.
LIMITES = {**{c: (0.0, 20000.0) for c in ("n_1m", "n_3m", "n_6m", "n_12m", "n35_3m", "n35_12m", "vec_1m", "vec_3m", "n_mismo_mes_anio_prev")},
           **{c: (2.5, 9.5) for c in ("mag_media_3m", "mag_media_12m", "mag_max_3m", "mag_max_12m")},
           "prof_media_hist": (0.0, 700.0), "sen_mes": (-1.0, 1.0), "cos_mes": (-1.0, 1.0)}


def esquema_formulario() -> list:
    """Variables editables del formulario: tipo, límites válidos y rango observado en el entrenamiento."""
    P = panel()
    P = P[P.y.notna()]
    out = []
    for c in VARIABLES_BASE:
        v = P[c]
        lo, hi = LIMITES.get(c, (float(np.nanmin(v)), float(np.nanmax(v))))
        out.append({"columna": c, "descripcion": DESCRIPCION[c], "unidad": UNIDADES[c], "min": float(np.nanmin(v)),
                    "max": float(np.nanmax(v)), "min_valido": lo, "max_valido": hi,
                    "entero": bool(np.all(v.dropna() == np.round(v.dropna()))), "admite_faltante": bool(v.isna().any())})
    return out


def intervalo(etiqueta: str, p) -> tuple:
    """Intervalo de predicción al 80 % (cuantiles de residuos fuera de muestra del tramo del pronóstico)."""
    t = intervalos()[etiqueta]
    lp = np.log1p(np.atleast_1d(np.asarray(p, float)))
    b = np.array(t["bordes"])
    j = np.clip(np.searchsorted(b, lp, side="right") - 1, 0, len(b) - 2)
    return np.clip(np.expm1(lp + np.array(t["q10"])[j]), 0, None), np.expm1(lp + np.array(t["q90"])[j])
