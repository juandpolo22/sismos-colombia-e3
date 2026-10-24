"""Herramientas del análisis exploratorio: formato, estadística descriptiva, asociación, faltantes,
deriva y estadística espacial y espacio-temporal.

Todas las funciones son puras (no leen archivos) y están documentadas con el supuesto que usan.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as cfg

RADIO_TIERRA_KM = 6371.0088


# ---------------------------------------------------------------------------
# Formato y figuras
# ---------------------------------------------------------------------------
def configurar_graficos() -> None:
    """Estilo común de las figuras de matplotlib/seaborn."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.max_open_warning": 0})


def guardar_figura(fig, nombre: str) -> Path:
    """Guarda la figura en ``figures/`` y devuelve la ruta."""
    ruta = cfg.FIGURAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def guardar_tabla(df: pd.DataFrame, nombre: str, index: bool = False) -> Path:
    """Guarda una tabla en ``tables/`` (CSV UTF-8) y devuelve la ruta."""
    ruta = cfg.TABLAS / nombre
    df.to_csv(ruta, index=index)
    return ruta


def es(x, dec: int = 1, pct: bool = False) -> str:
    """Formato numérico en español (1.234,5; con ``pct=True`` multiplica por 100 y añade ' %')."""
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "—"
    v = 100 * x if pct else x
    txt = f"{v:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return txt + (" %" if pct else "")


def p_fmt(p: float) -> str:
    """P-valor legible (``< 0,001`` cuando corresponde)."""
    if p is None or np.isnan(p):
        return "—"
    return "< 0,001" if p < 0.001 else es(p, 3)


# ---------------------------------------------------------------------------
# Estadística descriptiva y asociación
# ---------------------------------------------------------------------------
def resumen_numerico(X: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos por variable numérica (incluye % de atípicos por la regla IQR de Tukey)."""
    q = X.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    q1, q3 = q[0.25], q[0.75]
    iqr = q3 - q1
    fuera = X.lt(q1 - 1.5 * iqr, axis=1) | X.gt(q3 + 1.5 * iqr, axis=1)
    return pd.DataFrame({
        "n": X.count(), "faltantes": X.isna().sum(), "media": X.mean(), "mediana": q[0.5], "desv_est": X.std(),
        "mín": X.min(), "p01": q[0.01], "p25": q1, "p75": q3, "p99": q[0.99], "máx": X.max(),
        "asimetría": X.skew(), "curtosis": X.kurt(), "ceros_%": 100 * X.eq(0).mean(),
        "únicos": X.nunique(), "atípicos_IQR_%": 100 * fuera.sum() / X.notna().sum().replace(0, np.nan),
    })


def cramers_v(x: pd.Series, y: pd.Series) -> tuple:
    """V de Cramér con corrección de sesgo (Bergsma, 2013), χ² y p-valor."""
    tabla = pd.crosstab(x.astype("object").fillna("Faltante"), y.astype("object").fillna("Faltante"))
    if min(tabla.shape) < 2:
        return np.nan, np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(tabla, correction=False)
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-12))), float(chi2), float(p)


def holm(pvalores) -> np.ndarray:
    """P-valores ajustados por Holm-Bonferroni (control del error de familia)."""
    p = np.asarray(pvalores, float)
    orden = np.argsort(p)
    aj, acum, m = np.empty(len(p)), 0.0, len(p)
    for r, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - r) * p[i]))
        aj[i] = acum
    return aj


def fdr_bh(pvalores) -> np.ndarray:
    """P-valores ajustados por Benjamini-Hochberg (tasa de falsos descubrimientos)."""
    p = np.asarray(pvalores, float)
    m = len(p)
    orden = np.argsort(p)
    esc = p[orden] * m / np.arange(1, m + 1)
    mono = np.minimum.accumulate(esc[::-1])[::-1]
    out = np.empty(m)
    out[orden] = np.clip(mono, 0, 1)
    return out


def vif(X: np.ndarray) -> np.ndarray:
    """Factor de inflación de la varianza de cada columna: 1 / (1 - R²_j)."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        y, otros = X[:, j], np.delete(X, j, axis=1)
        b, *_ = np.linalg.lstsq(otros, y, rcond=None)
        r = y - otros @ b
        r2 = 1 - (r @ r) / (y @ y) if (y @ y) > 0 else 1.0
        out[j] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return out


def epsilon_cuadrado(h: float, n: int) -> float:
    """Tamaño de efecto de Kruskal-Wallis: ε² = H / ((n² - 1) / (n + 1))."""
    return float(h / ((n ** 2 - 1) / (n + 1)))


def biserial_rangos(u: float, n1: int, n2: int) -> float:
    """Correlación biserial de rangos para Mann-Whitney: r = 1 - 2U / (n1 n2)."""
    return float(1 - 2 * u / (n1 * n2))


def delta_cliff(a, b, max_n: int = 4000, semilla: int = cfg.SEMILLA) -> float:
    """Delta de Cliff P(a > b) - P(a < b) (submuestra si los grupos son grandes)."""
    rng = np.random.default_rng(semilla)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    s = np.sign(a[:, None] - b[None, :])
    return float(s.mean())


def psi(esperado, actual, bins: int = 10) -> float:
    """Population Stability Index (deriva): < 0,1 estable; 0,1–0,25 moderada; > 0,25 fuerte."""
    eps = 1e-6
    esperado, actual = np.asarray(esperado, float), np.asarray(actual, float)
    esperado, actual = esperado[~np.isnan(esperado)], actual[~np.isnan(actual)]
    bordes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    if len(bordes) < 3:
        vals = np.union1d(np.unique(esperado), np.unique(actual))
        e = np.array([(esperado == v).mean() for v in vals]) + eps
        a = np.array([(actual == v).mean() for v in vals]) + eps
    else:
        bordes[0], bordes[-1] = -np.inf, np.inf
        e = np.histogram(esperado, bordes)[0] / len(esperado) + eps
        a = np.histogram(actual, bordes)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def prueba_little_mcar(df: pd.DataFrame) -> dict:
    """Prueba MCAR de Little (1988) para variables numéricas.

    H0: los datos faltan completamente al azar. Usa las medias y la covarianza estimadas con los casos
    disponibles por pares (aproximación habitual cuando no se usa EM). Devuelve χ², g.l. y p-valor.
    """
    X = df.astype(float)
    cols = X.columns
    mu = X.mean().to_numpy()
    S = X.cov().to_numpy()
    patrones = X.isna().to_numpy()
    claves = pd.Series([tuple(r) for r in patrones])
    d2, gl = 0.0, 0
    for patron, idx in claves.groupby(claves).groups.items():
        obs = ~np.array(patron)
        if obs.sum() == 0:
            continue
        sub = X.iloc[list(idx), obs].to_numpy()
        nj = len(sub)
        dif = sub.mean(0) - mu[obs]
        Sj = S[np.ix_(obs, obs)]
        d2 += nj * dif @ np.linalg.pinv(Sj) @ dif
        gl += int(obs.sum())
    gl -= len(cols)
    p = float(stats.chi2.sf(d2, gl)) if gl > 0 else np.nan
    return {"chi2": float(d2), "gl": gl, "p": p, "patrones": int(claves.nunique())}


# ---------------------------------------------------------------------------
# Estadística espacial
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia de gran círculo en km (no se usa distancia euclidiana sobre grados)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pesos_knn(lat, lon, k: int = 8) -> np.ndarray:
    """Matriz de pesos de k vecinos más cercanos (haversine), estandarizada por filas."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = len(lat)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    k = min(k, n - 1)
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), np.argsort(D, axis=1)[:, :k].ravel()] = 1.0
    return W / W.sum(1, keepdims=True)


def pesos_banda(lat, lon, d_km: float) -> np.ndarray:
    """Pesos binarios por banda de distancia (``d <= d_km``), estandarizados por filas (filas vacías = 0)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    W = ((D <= d_km) & (D > 0)).astype(float)
    s = W.sum(1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def moran_global(valores, W, permutaciones: int = 999, semilla: int = cfg.SEMILLA) -> dict:
    """I de Moran global con prueba de permutación (H0: aleatoriedad espacial)."""
    z = np.asarray(valores, float) - np.mean(valores)
    n, s0 = len(z), W.sum()
    den = float(z @ z)
    obs = (n / s0) * float(z @ W @ z) / den
    rng = np.random.default_rng(semilla)
    sims = np.array([(n / s0) * float((zp := rng.permutation(z)) @ W @ zp) / den for _ in range(permutaciones)])
    p = (1 + np.sum(np.abs(sims - sims.mean()) >= abs(obs - sims.mean()))) / (permutaciones + 1)
    return {"I": float(obs), "E[I]": -1 / (n - 1), "z": float((obs - sims.mean()) / sims.std()), "p": float(p)}


def moran_local(valores, W, permutaciones: int = 999, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """LISA (I de Moran local) con permutación condicional y cuadrante HH/LL/HL/LH."""
    x = np.asarray(valores, float)
    z = (x - x.mean()) / x.std()
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(semilla)
    p = np.empty(len(z))
    for i in range(len(z)):
        otros = np.delete(z, i)
        vec = W[i] > 0
        k, w = int(vec.sum()), W[i, vec]
        if k == 0:
            p[i] = 1.0
            continue
        sims = np.array([z[i] * (w @ rng.choice(otros, k, replace=False)) for _ in range(permutaciones)])
        p[i] = (1 + np.sum(np.abs(sims) >= abs(Ii[i]))) / (permutaciones + 1)
    cuad = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0)], ["HH", "LL", "HL"], "LH")
    return pd.DataFrame({"z": z, "rezago": lag, "I_local": Ii, "p": p, "cuadrante": cuad})


def getis_ord_gi(valores, W) -> np.ndarray:
    """Estadístico Gi* de Getis-Ord (z) con la propia unidad incluida en el vecindario."""
    x = np.asarray(valores, float)
    n = len(x)
    Wb = (W > 0).astype(float)
    np.fill_diagonal(Wb, 1.0)
    xb, s = x.mean(), np.sqrt((x ** 2).mean() - x.mean() ** 2)
    ws, w2 = Wb.sum(1), (Wb ** 2).sum(1)
    return (Wb @ x - xb * ws) / (s * np.sqrt((n * w2 - ws ** 2) / (n - 1)))


def clark_evans(lat, lon, area_km2: float) -> dict:
    """Índice del vecino más cercano (Clark-Evans): R < 1 agrupado; R ≈ 1 aleatorio; R > 1 disperso."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    obs = float(D.min(1).mean())
    dens = len(lat) / area_km2
    esp = 0.5 / np.sqrt(dens)
    se = 0.26136 / np.sqrt(len(lat) * dens)
    return {"R": obs / esp, "dist_media_obs_km": obs, "dist_esperada_km": esp, "z": (obs - esp) / se,
            "p": float(2 * stats.norm.sf(abs((obs - esp) / se)))}


def ripley_l(lat, lon, radios_km, area_km2: float) -> pd.DataFrame:
    """Función K de Ripley (sin corrección de borde) y L(r) - r (> 0 indica agrupamiento a escala r)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    n = len(lat)
    K = np.array([area_km2 * (D < r).sum() / (n * (n - 1)) for r in radios_km])
    return pd.DataFrame({"r_km": radios_km, "K": K, "L_menos_r": np.sqrt(K / np.pi) - np.asarray(radios_km)})


def prueba_knox(lat, lon, t_dias, d_km: float, t_umbral: float, permutaciones: int = 499,
                semilla: int = cfg.SEMILLA) -> dict:
    """Prueba de Knox de interacción espacio-tiempo.

    Cuenta los pares de sismos cercanos a la vez en espacio (<= ``d_km``) y tiempo (<= ``t_umbral``
    días) y lo compara con su distribución al permutar los tiempos (H0: sin interacción).
    """
    lat, lon, t = map(lambda a: np.asarray(a, float), (lat, lon, t_dias))
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) <= d_km
    iu = np.triu_indices(len(t), 1)
    cerca_s = D[iu]
    obs = int((cerca_s & (np.abs(t[:, None] - t[None, :])[iu] <= t_umbral)).sum())
    rng = np.random.default_rng(semilla)
    sims = np.empty(permutaciones)
    for b in range(permutaciones):
        tp = rng.permutation(t)
        sims[b] = (cerca_s & (np.abs(tp[:, None] - tp[None, :])[iu] <= t_umbral)).sum()
    return {"pares_observados": obs, "pares_esperados": float(sims.mean()), "razon": obs / max(sims.mean(), 1e-9),
            "p": float((1 + (sims >= obs).sum()) / (permutaciones + 1))}
