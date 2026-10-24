"""Comparación estadística de modelos (Entregable 2, sección 6).

Clasificación: Friedman (+ Iman-Davenport) → Nemenyi y diagrama de Diferencia Crítica → DeLong con Holm →
delta de Cliff.
Regresión: Model Confidence Set (Hansen, Lunde y Nason, 2011) → Giacomini-White (2006) / Clark-West (2007)
→ Diebold-Mariano HLN (en :mod:`src.evaluacion`) con bootstrap estacionario (Politis y Romano, 1994) →
d de Cohen; además SPA de Hansen (2005) para el control de *data snooping*.
Calibración: ECE y curva de confiabilidad.

Todas las pruebas sobre series de pérdidas usan el **mes** como unidad (ver :mod:`src.evaluacion`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .evaluacion import varianza_newey_west


# ---------------------------------------------------------------------------
# Friedman, Nemenyi y diagrama de Diferencia Crítica
# ---------------------------------------------------------------------------
def friedman(M: np.ndarray, menor_mejor: bool = True) -> dict:
    """Prueba de Friedman sobre una matriz bloques × modelos (cada fila un bloque: pliegue o mes).

    Devuelve el estadístico chi², su p-valor, la corrección F de Iman-Davenport y los rangos medios
    (rango 1 = mejor).
    """
    M = np.asarray(M, float)
    N, k = M.shape
    R = stats.rankdata(M if menor_mejor else -M, axis=1)
    rm = R.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(rm ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi2, k - 1)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) - chi2 > 0 else np.inf
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    return {"chi2": float(chi2), "gl": k - 1, "p": float(p), "F_ID": float(ff), "p_ID": float(p_ff), "N": N, "k": k, "rangos": rm}


def diferencia_critica(k: int, N: int, alpha: float = 0.05) -> float:
    """CD de Nemenyi: q_α(k, ∞)/√2 · √(k(k+1)/(6N))."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * N)))


def nemenyi_pares(rangos, N: int) -> pd.DataFrame:
    """p-valores de Nemenyi para todos los pares (aproximación por rango estudentizado)."""
    k = len(rangos)
    se = np.sqrt(k * (k + 1) / (6 * N))
    P = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(rangos[i] - rangos[j]) / se * np.sqrt(2)
            P[i, j] = P[j, i] = stats.studentized_range.sf(q, k, np.inf)
    return pd.DataFrame(P)


def grafico_cd(nombres, rangos, cd, ax, titulo=""):
    """Diagrama de Diferencia Crítica (Demšar, 2006): eje de rangos, modelos y barras de grupos no distintos."""
    orden = np.argsort(rangos)
    r = np.asarray(rangos)[orden]
    n = [nombres[i] for i in orden]
    k = len(r)
    lo, hi = np.floor(r.min()), np.ceil(r.max())
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-(k // 2 + 2) - 0.5, 1.6)
    ax.axhline(0, color="k", lw=1)
    for t in np.arange(lo, hi + 1):
        ax.plot([t, t], [0, 0.12], color="k", lw=1)
        ax.text(t, 0.25, f"{t:.0f}", ha="center", fontsize=8)
    largo = min(cd, hi - lo)
    ax.plot([lo, lo + largo], [1.1, 1.1], color="crimson", lw=2, ls="-" if cd <= hi - lo else "--")
    ax.text(lo + largo / 2, 1.3, f"CD = {cd:.2f}" + ("" if cd <= hi - lo else " (más larga que el eje)"), ha="center", color="crimson", fontsize=8)
    mitad = (k + 1) // 2
    for i, (ri, ni) in enumerate(zip(r, n)):
        izq = i < mitad
        yy = -(i + 1) * 0.55 if izq else -(k - i) * 0.55
        xx = lo - 0.25 if izq else hi + 0.25
        ax.plot([ri, ri, xx], [0, yy, yy], color="0.35", lw=0.8)
        ax.text(xx, yy, f"{ni} ({ri:.2f})", ha="right" if izq else "left", va="center", fontsize=7)
    # grupos maximales sin diferencia significativa
    grupos, nivel = [], 0
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for a, b in grupos:
        nivel += 1
        ax.plot([r[a] - 0.03, r[b] + 0.03], [-0.15 * nivel, -0.15 * nivel], color="k", lw=3)
    ax.axis("off")
    ax.set_title(titulo, fontsize=10)
    return grupos


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _componentes_delong(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    todo = np.concatenate([pos, neg])
    r = stats.rankdata(todo)
    rp, rn = stats.rankdata(pos), stats.rankdata(neg)
    v10 = (r[:m] - rp) / n            # componentes estructurales de Sen
    v01 = 1 - (r[m:] - rn) / m
    auc = v10.mean()
    return auc, v10, v01


def delong(y, s1, s2) -> dict:
    """Prueba de DeLong (1988) para dos AUC correlacionadas sobre las mismas observaciones."""
    y = np.asarray(y).astype(int)
    a1, v10a, v01a = _componentes_delong(y, np.asarray(s1, float))
    a2, v10b, v01b = _componentes_delong(y, np.asarray(s2, float))
    m, n = len(v10a), len(v01a)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(max(var, 1e-18))
    return {"AUC_1": float(a1), "AUC_2": float(a2), "dif": float(a1 - a2), "z": float(z), "p": float(2 * stats.norm.sf(abs(z))),
            "IC_inf": float(a1 - a2 - 1.96 * np.sqrt(var)), "IC_sup": float(a1 - a2 + 1.96 * np.sqrt(var))}


def bootstrap_auc_meses(y, s1, s2, tm, B: int = 1000, semilla: int = cfg.SEMILLA) -> dict:
    """IC 95 % de la diferencia de AUC remuestreando meses completos."""
    from sklearn.metrics import roc_auc_score
    meses = np.unique(tm)
    grupos = [np.where(tm == m)[0] for m in meses]
    rng = np.random.default_rng(semilla)
    d = []
    for _ in range(B):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(meses), len(meses))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    return {"IC_boot_inf": float(np.percentile(d, 2.5)), "IC_boot_sup": float(np.percentile(d, 97.5))}


# ---------------------------------------------------------------------------
# Bootstrap estacionario
# ---------------------------------------------------------------------------
def indices_estacionarios(T: int, B: int, largo_medio: float, semilla: int = cfg.SEMILLA) -> np.ndarray:
    """Índices del bootstrap estacionario de Politis y Romano (bloques de largo geométrico)."""
    rng = np.random.default_rng(semilla)
    p = 1.0 / largo_medio
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, B)
    nuevo = rng.random((B, T)) < p
    aleatorio = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(nuevo[:, t], aleatorio[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def largo_bloque(T: int) -> float:
    """Regla práctica T^(1/3) para el largo medio del bloque."""
    return max(2.0, T ** (1 / 3))


def bootstrap_estacionario_media(d, B: int = 2000, semilla: int = cfg.SEMILLA) -> dict:
    """IC 95 % y p-valor bilateral para la media de una serie de diferencias de pérdida."""
    d = np.asarray(d, float)
    idx = indices_estacionarios(len(d), B, largo_bloque(len(d)), semilla)
    bm = d[idx].mean(1)
    centr = bm - d.mean()
    return {"media": float(d.mean()), "IC_inf": float(np.percentile(bm, 2.5)), "IC_sup": float(np.percentile(bm, 97.5)),
            "p_boot": float(np.mean(np.abs(centr) >= abs(d.mean())))}


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------
def mcs(L: np.ndarray, nombres, alpha: float = 0.10, B: int = 2000, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """Model Confidence Set con estadístico T_max y bootstrap estacionario (Hansen, Lunde y Nason, 2011).

    ``L``: matriz T × m de pérdidas (filas = meses). Elimina iterativamente el modelo con mayor pérdida
    relativa estandarizada mientras la hipótesis de igual capacidad predictiva se rechace a nivel α.
    Devuelve el p-valor MCS de cada modelo (un modelo está en el conjunto si p_MCS ≥ α).
    """
    L = np.asarray(L, float)
    T, m = L.shape
    idx = indices_estacionarios(T, B, largo_bloque(T), semilla)
    Lb = L[idx]                                   # B × T × m
    medias_b = Lb.mean(1)                         # B × m
    vivos = list(range(m))
    p_mcs = np.zeros(m)
    p_acum = 0.0
    orden_elim = []
    while len(vivos) > 1:
        Lv = L[:, vivos]
        dbar_i = Lv.mean(0) - Lv.mean()           # pérdida relativa de i frente al promedio del conjunto
        mb = medias_b[:, vivos]
        dstar = (mb - mb.mean(1, keepdims=True)) - dbar_i
        var = (dstar ** 2).mean(0)
        t_i = dbar_i / np.sqrt(var)
        tmax = t_i.max()
        tmax_b = (dstar / np.sqrt(var)).max(1)
        p = float(np.mean(tmax_b >= tmax))
        p_acum = max(p_acum, p)
        peor = vivos[int(np.argmax(t_i))]
        p_mcs[peor] = p_acum
        orden_elim.append(peor)
        vivos.remove(peor)
    p_mcs[vivos[0]] = 1.0
    df = pd.DataFrame({"modelo": list(nombres), "perdida_media": L.mean(0), "p_MCS": p_mcs})
    df["rango_perdida"] = df.perdida_media.rank().astype(int)
    df["en_MCS"] = df.p_MCS >= alpha
    return df.sort_values("perdida_media").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPA de Hansen
# ---------------------------------------------------------------------------
def spa(L_bench, L_modelos, B: int = 2000, semilla: int = cfg.SEMILLA) -> dict:
    """Test de Habilidad Predictiva Superior (Hansen, 2005).

    H0: ningún modelo de ``L_modelos`` (T × k) supera al benchmark ``L_bench`` (T). Devuelve los p-valores
    consistente, inferior y superior (este último equivale al Reality Check de White).
    """
    d = np.asarray(L_bench, float)[:, None] - np.asarray(L_modelos, float)   # > 0: el modelo es mejor
    T, k = d.shape
    dbar = d.mean(0)
    w = np.sqrt(np.array([varianza_newey_west(d[:, j]) for j in range(k)]))
    est = max(0.0, np.max(np.sqrt(T) * dbar / w))
    idx = indices_estacionarios(T, B, largo_bloque(T), semilla)
    db = d[idx].mean(1)                            # B × k
    umbral = -np.sqrt(w ** 2 / T * 2 * np.log(np.log(T)))
    # recentrado g(d̄): inferior max(d̄,0); consistente d̄·1{d̄ ≥ −A_T}; superior d̄ (Reality Check)
    g = {"inferior": np.maximum(dbar, 0), "consistente": np.where(dbar >= umbral, dbar, 0.0), "superior": dbar}
    out = {"estadistico": float(est)}
    for nombre, gi in g.items():
        z = np.sqrt(T) * (db - gi) / w
        tb = np.maximum(z.max(1), 0)
        out[f"p_{nombre}"] = float(np.mean(tb >= est))
    return out


# ---------------------------------------------------------------------------
# Giacomini-White y Clark-West
# ---------------------------------------------------------------------------
def giacomini_white(La, Lb, rezagos: int = 1) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006).

    Instrumentos h_{t-1} = (1, d_{t-1}, ..., d_{t-rezagos}); estadístico de Wald T·Z̄'Ω⁻¹Z̄ ~ χ²(q).
    Con solo la constante se reduce a una prueba incondicional tipo DM. Positivo en ``dif_media`` = B mejor.
    """
    d = np.asarray(La, float) - np.asarray(Lb, float)
    T = len(d)
    H = np.column_stack([np.ones(T - rezagos)] + [d[rezagos - j - 1: T - j - 1] for j in range(rezagos)])
    Z = H * d[rezagos:, None]
    n = len(Z)
    Zb = Z.mean(0)
    Zc = Z - Zb
    lag = int(np.floor(n ** (1 / 3)))
    Om = Zc.T @ Zc / n
    for k in range(1, lag + 1):
        G = Zc[k:].T @ Zc[:-k] / n
        Om += (1 - k / (lag + 1)) * (G + G.T)
    W = float(n * Zb @ np.linalg.pinv(Om) @ Zb)
    q = Z.shape[1]
    return {"dif_media": float(d.mean()), "GW": W, "gl": q, "p": float(stats.chi2.sf(W, q))}


def clark_west(y, p_restringido, p_amplio, tm) -> dict:
    """Prueba de Clark y West (2007) para modelos anidados, en escala log1p y agregada por mes.

    f_t = e_r² − (e_a² − (ŷ_r − ŷ_a)²); H0: igual MSPE poblacional; H1 (unilateral): el modelo amplio es mejor.
    """
    ly = np.log1p(np.asarray(y, float))
    lr, la = np.log1p(np.clip(p_restringido, 0, None)), np.log1p(np.clip(p_amplio, 0, None))
    f = (lr - ly) ** 2 - ((la - ly) ** 2 - (lr - la) ** 2)
    meses = np.unique(tm)
    pos = np.searchsorted(meses, tm)
    fm = np.bincount(pos, f) / np.bincount(pos)
    T = len(fm)
    t = fm.mean() / np.sqrt(varianza_newey_west(fm) / T)
    return {"CW": float(t), "p_unilateral": float(stats.norm.sf(t)), "T": T}


# ---------------------------------------------------------------------------
# Tamaños de efecto
# ---------------------------------------------------------------------------
def cohen_d_pareado(La, Lb) -> float:
    """d de Cohen para diferencias de pérdida pareadas (media / desviación de d_t). Positivo = B mejor."""
    d = np.asarray(La, float) - np.asarray(Lb, float)
    return float(d.mean() / (d.std(ddof=1) + 1e-18))


def magnitud_cliff(delta: float) -> str:
    """Etiqueta de magnitud del δ de Cliff (umbrales de Romano et al., 2006: 0,147; 0,33; 0,474)."""
    a = abs(delta)
    return "despreciable" if a < 0.147 else "pequeño" if a < 0.33 else "mediano" if a < 0.474 else "grande"


def magnitud_cohen(d: float) -> str:
    """Etiqueta de magnitud de la d de Cohen (umbrales de Cohen, 1988: 0,2; 0,5; 0,8)."""
    a = abs(d)
    return "despreciable" if a < 0.2 else "pequeño" if a < 0.5 else "mediano" if a < 0.8 else "grande"


# ---------------------------------------------------------------------------
# Calibración
# ---------------------------------------------------------------------------
def ece(y, p, bins: int = 10) -> float:
    """Error de Calibración Esperado con bins de igual ancho."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            tot += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(tot)


def curva_confiabilidad(y, p, bins: int = 10) -> pd.DataFrame:
    """Datos del diagrama de confiabilidad de un clasificador.

    Parameters
    ----------
    y : array-like
        Clase observada (0/1).
    p : array-like
        Probabilidad pronosticada de la clase 1.
    bins : int
        Número de intervalos iguales en [0, 1].

    Returns
    -------
    pd.DataFrame
        Por intervalo no vacío: probabilidad media, frecuencia observada y número de casos.
    """
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    b = np.minimum((p * bins).astype(int), bins - 1)
    filas = [{"bin": k, "p_media": p[b == k].mean(), "frecuencia": y[b == k].mean(), "n": int((b == k).sum())}
             for k in range(bins) if (b == k).any()]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Bootstrap BCa genérico por meses
# ---------------------------------------------------------------------------
def bca_meses(estadistico, tm, B: int = 1000, alpha: float = 0.05, semilla: int = cfg.SEMILLA) -> dict:
    """IC BCa de ``estadistico(indices_filas)`` remuestreando meses completos (bootstrap de bloques).

    La aceleración se estima con jackknife dejando fuera un mes cada vez.
    """
    meses = np.unique(tm)
    grupos = [np.where(tm == m)[0] for m in meses]
    M = len(meses)
    theta = estadistico(np.concatenate(grupos))
    rng = np.random.default_rng(semilla)
    bt = np.array([estadistico(np.concatenate([grupos[i] for i in rng.integers(0, M, M)])) for _ in range(B)])
    jk = np.array([estadistico(np.concatenate([g for k, g in enumerate(grupos) if k != j])) for j in range(M)])
    z0 = stats.norm.ppf(np.clip((bt < theta).mean(), 1e-6, 1 - 1e-6))
    d = jk.mean() - jk
    a = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5 + 1e-300))
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    q = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    return {"valor": float(theta), "IC_perc_inf": float(np.percentile(bt, 100 * alpha / 2)), "IC_perc_sup": float(np.percentile(bt, 100 * (1 - alpha / 2))),
            "IC_BCa_inf": float(np.quantile(bt, q[0])), "IC_BCa_sup": float(np.quantile(bt, q[1])), "sesgo_z0": float(z0), "aceleracion": float(a)}


# ---------------------------------------------------------------------------
# Independencia de residuos (pruebas complementarias)
# ---------------------------------------------------------------------------
def escanciano_lobato(e, d: int = 12, q: float = 2.4) -> dict:
    """Prueba portmanteau automática de Escanciano y Lobato (2009, J. Econometrics).

    Autocorrelaciones robustas a heterocedasticidad condicional (ρ̃_j = γ_j / τ_j) y número de rezagos elegido por
    un criterio tipo AIC/BIC; bajo H0 (sin autocorrelación) el estadístico se distribuye χ²(1).
    """
    e = np.asarray(e, float)
    n = len(e)
    ec = e - e.mean()
    g0 = ec @ ec / n
    rho, Qs = [], []
    for j in range(1, d + 1):
        gj = (ec[j:] @ ec[:-j]) / n
        tj = np.sum((ec[j:] ** 2) * (ec[:-j] ** 2)) / n
        rho.append(gj ** 2 / tj if tj > 0 else 0.0)
        Qs.append(n * np.sum(rho))
    rho = np.array(rho)
    umbral = np.sqrt(q * np.log(n))
    max_r = np.max(np.sqrt(n * rho))
    pen = [(p * np.log(n)) if max_r <= umbral else 2 * p for p in range(1, d + 1)]
    L = np.array(Qs) - np.array(pen)
    p_sel = int(np.argmax(L)) + 1
    est = float(Qs[p_sel - 1])
    return {"estadistico": est, "rezagos": p_sel, "p": float(stats.chi2.sf(est, 1))}


def breusch_godfrey(e, rezagos: int = 12) -> dict:
    """Prueba LM de Breusch-Godfrey de autocorrelación de orden ``rezagos`` sobre una serie de residuos."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    e = np.asarray(e, float)
    r = sm.OLS(e, np.ones(len(e))).fit()
    lm, p_lm, f, p_f = acorr_breusch_godfrey(r, nlags=rezagos)
    return {"LM": float(lm), "p": float(p_lm), "F": float(f), "p_F": float(p_f)}


def bds_serie(e, dims=(2, 3)) -> dict:
    """BDS (Brock, Dechert, Scheinkman y LeBaron, 1996) para dependencia no lineal; devuelve la dimensión con menor p."""
    from statsmodels.tsa.stattools import bds
    e = np.asarray(e, float)
    st, p = bds(e, max_dim=max(dims))
    st, p = np.atleast_1d(st), np.atleast_1d(p)
    return {"estadisticos": [float(v) for v in st], "p_por_dimension": [float(v) for v in p], "p_min": float(np.min(p))}


def moran_por_mes(res, tm, lat, lon, k: int = 8, permutaciones: int = 499, semilla: int = cfg.SEMILLA) -> pd.DataFrame:
    """I de Moran de los residuos entre celdas, mes a mes (pesos de k vecinos más cercanos, p por permutación)."""
    from .eda import moran_global, pesos_knn
    filas = []
    for m in np.unique(tm):
        s = tm == m
        W = pesos_knn(lat[s], lon[s], k=k)
        r = moran_global(res[s], W, permutaciones=permutaciones, semilla=semilla)
        filas.append({"mes": m, **r})
    return pd.DataFrame(filas)
