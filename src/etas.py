"""Modelo ETAS espacio-temporal (implementación propia con numpy/scipy) como miembro del ensamble.

Se estima por máxima verosimilitud sobre los **eventos individuales** del catálogo (no sobre el panel) y se
usa para pronosticar el número de sismos M >= m0 de cada fila (celda ``c``, mes ``t``) del panel en el mes
``t + 1`` con la información hasta el cierre del mes ``t``.

Modelo (Ogata, 1998; Zhuang, Ogata y Vere-Jones, 2002)
------------------------------------------------------
Intensidad condicional (tiempo en días, distancias en km)::

    λ(t, x | H_t) = μ_c(x) / A_c(x) + Σ_{i: t_i < t} κ_i · h(t - t_i) · f(x - x_i | m_i)

* Fondo: tasa constante ``μ_c`` (sismos/día) en cada celda ``c`` de la rejilla, repartida uniformemente en su
  área ``A_c`` (km²).
* Productividad: ``κ_i = K · exp(α (m_i - m0)) · exp(δ · 1[prof_i > prof_profunda])`` (los sismos de
  profundidad intermedia, p. ej. el nido de Bucaramanga, tienen su propio factor de productividad).
* Omori-Utsu (forma de Ogata) truncado en ``t_max``: ``h(s) = (s + c)^{-p}``, ``0 < s <= t_max``. Gracias al
  truncamiento la integral es finita para cualquier ``p > 0`` (también ``p <= 1``).
* Núcleo espacial de ley de potencia normalizado en el plano y truncado en ``r_max``::

      f(r | m) = (q-1) / (π σ_m²) · (1 + r² / σ_m²)^{-q},   σ_m² = D · exp(γ (m - m0)) · exp(ε · 1[profundo])

Log-verosimilitud (eventos en [0, T))::

    log L = Σ_j log λ(t_j, x_j) - Σ_c μ_c T - Σ_i κ_i · H(min(T - t_i, t_max)) · S(r_max | m_i)

con ``H(s) = ∫_0^s (u + c)^{-p} du = c^{1-p} (exp((1-p) log(1 + s/c)) - 1) / (1-p)`` (``= log(1 + s/c)`` si p = 1) y ``S(R | m) = 1 - (1 + R²/σ_m²)^{1-q}`` (se ignora la pérdida de masa por
el borde de la región, aproximación habitual). Gradiente analítico respecto a los 9 parámetros de disparo
(en escala no restringida) y a ``log μ_c``; optimización con L-BFGS-B.

Pronóstico de la fila (c, t), ventana W = [inicio del mes t+1, inicio del mes t+2)::

    Λ1_c = μ_c |W| + Σ_{i: t_i < inicio(W)} κ_i [H(fin(W) - t_i) - H(inicio(W) - t_i)] · ∫_{celda c} f(x - x_i | m_i) dx
    Λ_c  = Λ1_c / (1 - n̄ · H̄_W)        (cascada dentro de la ventana, aproximación de campo medio)

donde ``n̄`` es el promedio de κ·S en los eventos del ajuste y ``H̄_W = (1/|W|) ∫_0^|W| H(s) ds`` la descendencia
media (por unidad de κ·S) que un evento de la ventana genera dentro de ella. La integral espacial sobre la celda
se calcula con una cuadratura determinista de cuantiles radiales × ángulos del propio núcleo.

Pseudocódigo
------------
::

    fit(X, y):
        T  <- inicio del mes (max(t de X) + 2)                     # fin del último mes objetivo observado
        E  <- eventos con tiempo < T y m >= m0
        P  <- pares (i -> j) con 0 < t_j - t_i <= t_max y r_ij <= r_max      # barrido temporal por bloques
        θ0 <- (K, α, c, p, D, γ, q, δ, ε) iniciales;  μ_c <- 0,5 · N_c / T
        θ, μ <- L-BFGS-B( -log L(θ, μ) / N, gradiente analítico )
        si ventana_fondo: μ_c <- Σ_{j∈c, t_j ≥ T-ventana} φ_j / ventana,  φ_j = (μ_c/A_c) / λ_j   # desagrupamiento
        n̄ <- media_i κ_i S_i ;  factor_cascada <- 1 / (1 - n̄ H̄_W)

    predict(X):
        S  <- matriz dispersa evento × celda con ∫_celda f(x - x_i | m_i) dx   # cuadratura, una vez por evento
        para cada mes t de X:  w_i(t) <- κ_i [H(fin - t_i) - H(inicio - t_i)]  (eventos con t_i < inicio)
        Λ  <- factor_cascada · (μ_c |W_t| + Sᵀ w(t))[c]

Complejidad
-----------
``N`` eventos, ``P`` pares dentro de (t_max, r_max), ``C`` celdas con eventos, ``Q = n_radial · n_angular``.

* Construcción de pares: O(Σ_j #{i : t_j - t_max < t_i < t_j}) operaciones (barrido por tiempo), memoria O(P).
* Cada evaluación de log L y gradiente: O(P + N + C); L-BFGS con ``I`` iteraciones: O(I (P + N + C)).
* Fondo por ventana: O(N). Integrales espaciales: O(N Q). Pronóstico de ``M`` meses: O(N · t_max/30 + nnz(S) · M).

En el catálogo del SGC (≈ 37 000 eventos hasta 2020, t_max = 365 d, r_max = 100 km) ``P`` ≈ 1,7·10⁷: cada
evaluación tarda ≈ 1 s y el ajuste unos minutos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from . import config as cfg

KM_LAT = 110.574                    # km por grado de latitud
KM_LON = 111.320                    # km por grado de longitud en el ecuador (se multiplica por cos(lat))
NOMBRES_THETA = ("log_K", "alpha", "log_c", "p", "log_D", "gamma", "log_q_menos_1", "delta_prof", "eps_prof")
# Límites de los parámetros no restringidos (evitan regiones degeneradas durante L-BFGS-B).
LIMITES_THETA = ((-16.0, 6.0), (0.0, 3.5), (np.log(1e-5), np.log(30.0)), (0.2, 3.0),
                 (np.log(0.05), np.log(1e4)), (0.0, 3.5), (np.log(0.05), np.log(4.0)), (-8.0, 4.0), (-6.0, 6.0))


# ---------------------------------------------------------------------------
# Núcleos (funciones puras, también usadas por el simulador y las pruebas)
# ---------------------------------------------------------------------------
def _phi(x):
    """φ(x) = (e^x - 1)/x y φ'(x), estables cerca de x = 0."""
    x = np.asarray(x, float)
    peq = np.abs(x) < 1e-5
    xs = np.where(peq, 1.0, x)
    f = np.where(peq, 1 + x / 2 + x * x / 6, np.expm1(xs) / xs)
    d = np.where(peq, 0.5 + x / 3 + x * x / 8, (xs * np.exp(xs) - np.expm1(xs)) / (xs * xs))
    return f, d


def omori_acumulada(s, c, p, t_max, derivadas=False):
    """H(s) = ∫_0^min(s, t_max) (u + c)^{-p} du; 0 si s <= 0. Con ``derivadas`` también ∂H/∂log c y ∂H/∂p.

    H = c^{1-p} Δ φ((1-p) Δ) con Δ = log(1 + s/c) y φ(x) = (e^x - 1)/x (válida y estable para todo p > 0).
    """
    s = np.clip(np.asarray(s, float), 0.0, t_max)
    lc = np.log(c)
    dl = np.log1p(s / c)
    e = 1.0 - p
    f, fd = _phi(e * dl)
    a = np.exp(e * lc)
    H = a * dl * f
    if not derivadas:
        return H
    dH_lc = c * (s + c) ** (-p) - c ** (1 - p)            # c ∂H/∂c
    dH_p = -(lc * H + a * dl * dl * fd)                    # ∂H/∂p = -∂H/∂e
    return H, dH_lc, dH_p


def omori_inversa(v, c, p):
    """s tal que H(s) = v (sin truncamiento)."""
    e = 1.0 - p
    if abs(e) < 1e-10:
        return c * np.expm1(v)
    return c * ((1 + e * v * c ** (-e)) ** (1 / e) - 1)


def masa_espacial(r, sigma2, q):
    """S(r) = ∫_{|x|<=r} f(x) dx del núcleo de ley de potencia con escala ``sigma2`` (km²)."""
    return 1.0 - (1.0 + np.asarray(r, float) ** 2 / sigma2) ** (1.0 - q)


def theta_a_dict(theta) -> dict:
    """Parámetros no restringidos -> parámetros físicos (K, α, c, p, D, γ, q, δ, ε)."""
    th = np.asarray(theta, float)
    return {"K": float(np.exp(th[0])), "alpha": float(th[1]), "c": float(np.exp(th[2])), "p": float(th[3]),
            "D": float(np.exp(th[4])), "gamma": float(th[5]), "q": float(1 + np.exp(th[6])), "delta_prof": float(th[7]),
            "eps_prof": float(th[8])}


def dict_a_theta(par: dict) -> np.ndarray:
    """Convierte los parámetros ETAS de diccionario al vector θ sin restricciones (log de los positivos)."""
    return np.array([np.log(par["K"]), par["alpha"], np.log(par["c"]), par["p"], np.log(par["D"]),
                     par["gamma"], np.log(par["q"] - 1), par.get("delta_prof", 0.0), par.get("eps_prof", 0.0)], float)


def _km(lat1, lon1, lat2, lon2):
    """Componentes (dx, dy) en km de una proyección equirectangular local (adecuada para r <= unos cientos de km)."""
    dy = (lat2 - lat1) * KM_LAT
    dx = (lon2 - lon1) * KM_LON * np.cos(np.radians(0.5 * (lat1 + lat2)))
    return dx, dy


def construir_pares(t, lat, lon, t_max, r_max, bloque=2000):
    """Pares (padre i, hijo j) con 0 < t_j - t_i <= t_max y distancia <= r_max (``t`` ordenado).

    Barrido temporal por bloques de hijos: para cada hijo los candidatos son un rango contiguo de índices.
    Devuelve (i, j, Δt, r²) con i, j en int32.
    """
    n = len(t)
    lo = np.searchsorted(t, t - t_max, side="left")
    hi = np.searchsorted(t, t, side="left")            # excluye simultáneos (Δt = 0)
    salida = []
    for j0 in range(0, n, bloque):
        j = np.arange(j0, min(n, j0 + bloque))
        k = hi[j] - lo[j]
        if k.sum() == 0:
            continue
        jj = np.repeat(j, k)
        ii = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + np.repeat(lo[j], k)
        dx, dy = _km(lat[ii], lon[ii], lat[jj], lon[jj])
        r2 = dx * dx + dy * dy
        ok = r2 <= r_max ** 2
        salida.append((ii[ok].astype(np.int32), jj[ok].astype(np.int32), (t[jj] - t[ii])[ok], r2[ok]))
    if not salida:
        z = np.zeros(0)
        return z.astype(np.int32), z.astype(np.int32), z, z
    return tuple(np.concatenate(a) for a in zip(*salida))


# ---------------------------------------------------------------------------
# Log-verosimilitud y gradiente analítico
# ---------------------------------------------------------------------------
def log_verosimilitud(theta, log_mu, datos, con_gradiente=True):
    """log L(θ, μ) y su gradiente. ``datos``: dict con los arreglos del ajuste (ver ``ETAS._preparar``).

    ``log_mu`` contiene log μ_c (sismos/día) de las celdas con eventos ``datos["celdas_u"]``.
    """
    lK, al, lc, p, lD, ga, ch, de, ep = theta
    c, q = np.exp(lc), 1 + np.exp(ch)
    dm, prof, T, m0 = datos["dm"], datos["prof"], datos["T"], datos["m0"]
    pi_, pj, dt, r2 = datos["pi"], datos["pj"], datos["dt"], datos["r2"]
    t_max, r_max = datos["t_max"], datos["r_max"]
    mu = np.exp(log_mu)

    # por evento padre
    log_kap = lK + al * dm + de * prof
    log_s2 = lD + ga * dm + ep * prof
    # pares
    lt = np.log(dt + c)
    u = r2 * np.exp(-log_s2[pi_])
    lu = np.log1p(u)
    log_g = log_kap[pi_] - p * lt + np.log(q - 1) - np.log(np.pi) - log_s2[pi_] - q * lu
    g = np.exp(log_g)
    fondo = mu[datos["idx_mu"]] / datos["area"]
    lam = fondo + np.bincount(pj, g, minlength=len(dm))
    ll = np.log(lam).sum()

    # compensador
    H, dH_lc, dH_p = omori_acumulada(T - datos["t"], c, p, t_max, derivadas=True)
    v = r_max ** 2 * np.exp(-log_s2)
    b = (1 + v) ** (1 - q)
    S = 1 - b
    kap = np.exp(log_kap)
    comp_i = kap * H * S
    ll -= mu.sum() * T + comp_i.sum()
    if not con_gradiente:
        return ll, lam

    G = g / lam[pj]
    d_s = -1 + q * u / (1 + u)                  # ∂ log f / ∂ log σ²
    gr = np.empty(9)
    # término Σ log λ
    gr[0] = G.sum()
    gr[1] = (G * dm[pi_]).sum()
    gr[2] = -(G * (p * c / (dt + c))).sum()
    gr[3] = -(G * lt).sum()
    Gs = G * d_s
    gr[4] = Gs.sum()
    gr[5] = (Gs * dm[pi_]).sum()
    gr[6] = (G * (1 - (q - 1) * lu)).sum()
    gr[7] = (G * prof[pi_]).sum()
    gr[8] = (Gs * prof[pi_]).sum()
    # compensador
    dS_s = -b * (q - 1) * v / (1 + v)
    dS_ch = b * (q - 1) * np.log1p(v)
    gr[0] -= comp_i.sum()
    gr[1] -= (comp_i * dm).sum()
    gr[2] -= (kap * S * dH_lc).sum()
    gr[3] -= (kap * S * dH_p).sum()
    kHs = kap * H * dS_s
    gr[4] -= kHs.sum()
    gr[5] -= (kHs * dm).sum()
    gr[6] -= (kap * H * dS_ch).sum()
    gr[7] -= (comp_i * prof).sum()
    gr[8] -= (kHs * prof).sum()
    g_mu = np.bincount(datos["idx_mu"], fondo / lam, minlength=len(mu)) - mu * T
    return ll, gr, g_mu


# ---------------------------------------------------------------------------
# Estimador compatible con scikit-learn
# ---------------------------------------------------------------------------
class ETAS(RegressorMixin, BaseEstimator):
    """ETAS espacio-temporal estimado por máxima verosimilitud sobre los eventos del catálogo.

    Parameters
    ----------
    eventos : DataFrame
        Catálogo con columnas ``time`` (fecha-hora), ``latitude``, ``longitude``, ``depth`` y ``mag``. Es la fuente
        de información del modelo: ``fit`` solo usa los eventos anteriores al fin del último mes objetivo de ``X``
        y ``predict`` solo los anteriores al inicio del mes pronosticado de cada fila.
    m0 : float
        Magnitud mínima (los conteos son de M >= m0).
    t_max, r_max : float
        Truncamiento del núcleo de disparo en días y km.
    prof_profunda : float
        Profundidad (km) a partir de la cual el evento usa los factores ``δ`` (productividad) y ``ε`` (escala espacial).
    ventana_fondo : float or None
        Si no es None, tras el ajuste la tasa de fondo de cada celda se reestima con el desagrupamiento estocástico
        de los últimos ``ventana_fondo`` días (adapta el fondo a cambios lentos de la red o de la actividad).
    cascada : bool
        Si es True corrige el pronóstico por los sismos disparados dentro del propio mes (campo medio).
    n_radial, n_angular : int
        Puntos de la cuadratura determinista de la integral del núcleo espacial sobre cada celda.
    max_iter, tol : int, float
        Control de L-BFGS-B.
    inicio, celda, lat_min, lat_max, lon_min, lon_max :
        Origen de los índices de mes y rejilla (iguales a los del panel: ``src.datos``).

    Notes
    -----
    ``X`` es una matriz (n, 2) con el índice de celda de la rejilla y el índice de mes ``t`` (enero de 2004 = 0) de
    cada fila del panel; el objetivo es el conteo del mes ``t + 1``. ``y`` no se usa en ``fit`` (la verosimilitud
    es la de los eventos) y se acepta por compatibilidad. ``predict`` devuelve el número esperado de sismos;
    ``predecir_log1p`` devuelve E[log(1 + N)] con N ~ Poisson(Λ) (pronóstico puntual para RMSLE).
    """

    def __init__(self, eventos=None, m0=cfg.M_MIN, t_max=365.0, r_max=100.0, prof_profunda=70.0, ventana_fondo=None,
                 cascada=True, n_radial=40, n_angular=36, max_iter=500, tol=1e-9, inicio=cfg.INICIO,
                 celda=cfg.CELDA_GRADOS, lat_min=cfg.LAT_MIN, lat_max=cfg.LAT_MAX, lon_min=cfg.LON_MIN, lon_max=cfg.LON_MAX):
        self.eventos = eventos
        self.m0 = m0
        self.t_max = t_max
        self.r_max = r_max
        self.prof_profunda = prof_profunda
        self.ventana_fondo = ventana_fondo
        self.cascada = cascada
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.max_iter = max_iter
        self.tol = tol
        self.inicio = inicio
        self.celda = celda
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    # ------------------------------------------------------------------ utilidades de rejilla y tiempo
    def _rejilla(self):
        n_lat = int(np.ceil((self.lat_max - self.lat_min) / self.celda - 1e-9))
        n_lon = int(np.ceil((self.lon_max - self.lon_min) / self.celda - 1e-9))
        return n_lat, n_lon

    def _celda_de(self, lat, lon):
        """Índice de celda (fila-mayor); -1 fuera de la rejilla."""
        n_lat, n_lon = self._rejilla()
        i = np.floor((np.asarray(lat) - self.lat_min) / self.celda).astype(int)
        j = np.floor((np.asarray(lon) - self.lon_min) / self.celda).astype(int)
        fuera = (i < 0) | (i >= n_lat) | (j < 0) | (j >= n_lon)
        return np.where(fuera, -1, i * n_lon + j)

    def _celda_panel(self, lat, lon):
        """Igual que ``src.datos.asignar_celda`` (los eventos del borde se asignan a la celda más cercana)."""
        n_lat, n_lon = self._rejilla()
        i = np.clip(((np.asarray(lat) - self.lat_min) / self.celda).astype(int), 0, n_lat - 1)
        j = np.clip(((np.asarray(lon) - self.lon_min) / self.celda).astype(int), 0, n_lon - 1)
        return i * n_lon + j

    def _areas(self):
        n_lat, n_lon = self._rejilla()
        lat_c = self.lat_min + (np.arange(n_lat) + 0.5) * self.celda
        a = (self.celda * KM_LAT) * (self.celda * KM_LON * np.cos(np.radians(lat_c)))
        return np.repeat(a, n_lon)

    def _dia_mes(self, m):
        """Día (desde ``inicio``) en que empieza el mes de índice ``m``."""
        t0 = pd.Timestamp(self.inicio)
        m = np.asarray(m, int) + (t0.month - 1)
        f = pd.to_datetime(pd.DataFrame({"year": t0.year + m // 12, "month": m % 12 + 1, "day": 1}))
        return ((f - t0).dt.total_seconds() / 86400.0).to_numpy()

    # ------------------------------------------------------------------ validación
    def _validar_hiper(self):
        for nombre in ("t_max", "r_max", "celda"):
            v = getattr(self, nombre)
            if not np.isscalar(v) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{nombre} debe ser un número positivo; se recibió {v!r}")
        if self.ventana_fondo is not None and (not np.isscalar(self.ventana_fondo) or not self.ventana_fondo > 0):
            raise ValueError(f"ventana_fondo debe ser None o positivo; se recibió {self.ventana_fondo!r}")
        for nombre in ("n_radial", "n_angular", "max_iter"):
            v = getattr(self, nombre)
            if int(v) != v or v < 1:
                raise ValueError(f"{nombre} debe ser un entero >= 1; se recibió {v!r}")
        if not np.isfinite(self.m0):
            raise ValueError("m0 debe ser finito")
        if self.lat_max <= self.lat_min or self.lon_max <= self.lon_min:
            raise ValueError("la región (lat_min, lat_max, lon_min, lon_max) está vacía")

    def _catalogo(self):
        ev = self.eventos
        req = ("time", "latitude", "longitude", "depth", "mag")
        if not isinstance(ev, pd.DataFrame):
            raise ValueError("eventos debe ser un DataFrame con columnas time, latitude, longitude, depth, mag")
        falta = [c for c in req if c not in ev.columns]
        if falta:
            raise ValueError(f"a eventos le faltan las columnas {falta}")
        ev = ev.loc[ev["mag"] >= self.m0 - 1e-9, list(req)].dropna()
        if len(ev) == 0:
            raise ValueError(f"eventos no tiene sismos con M >= {self.m0}")
        t = ((pd.to_datetime(ev["time"]) - pd.Timestamp(self.inicio)).dt.total_seconds() / 86400.0).to_numpy()
        o = np.argsort(t, kind="stable")
        cat = {"t": t[o], "lat": ev["latitude"].to_numpy(float)[o], "lon": ev["longitude"].to_numpy(float)[o],
               "m": ev["mag"].to_numpy(float)[o], "prof": (ev["depth"].to_numpy(float)[o] > self.prof_profunda).astype(float)}
        ok = cat["t"] >= 0
        cat = {k: v[ok] for k, v in cat.items()}
        cat["celda"] = self._celda_panel(cat["lat"], cat["lon"])
        return cat

    def _validar_X(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X debe tener forma (n, 2) [celda, mes]; se recibió {X.shape}")
        if len(X) == 0:
            raise ValueError("X está vacío")
        if not np.all(np.isfinite(X.astype(float))) or np.any(X.astype(float) != np.round(X.astype(float))):
            raise ValueError("X debe contener índices enteros finitos")
        X = X.astype(int)
        n_lat, n_lon = self._rejilla()
        if X[:, 0].min() < 0 or X[:, 0].max() >= n_lat * n_lon:
            raise ValueError("índice de celda fuera de la rejilla")
        if X[:, 1].min() < 0:
            raise ValueError("índice de mes negativo")
        return X

    # ------------------------------------------------------------------ ajuste
    def _preparar(self, cat, T):
        sel = cat["t"] < T
        d = {k: v[sel] for k, v in cat.items()}
        if len(d["t"]) < 10:
            raise ValueError("se necesitan al menos 10 eventos antes del fin del periodo de ajuste")
        pi_, pj, dt, r2 = construir_pares(d["t"], d["lat"], d["lon"], self.t_max, self.r_max)
        celdas_u, idx_mu = np.unique(d["celda"], return_inverse=True)
        d.update(pi=pi_, pj=pj, dt=dt, r2=r2, T=T, m0=self.m0, t_max=float(self.t_max), r_max=float(self.r_max),
                 dm=d["m"] - self.m0, celdas_u=celdas_u, idx_mu=idx_mu, area=self._areas()[d["celda"]])
        return d

    def fit(self, X, y=None):
        """Estima (K, α, c, p, D, γ, q, δ, ε) y μ_c por máxima verosimilitud con los eventos hasta el fin del
        último mes objetivo de ``X`` (filas (celda, t) -> mes t + 1)."""
        self._validar_hiper()
        X = self._validar_X(X)
        if y is not None and len(np.asarray(y)) != len(X):
            raise ValueError("X e y tienen longitudes distintas")
        cat = self._catalogo()
        T = float(self._dia_mes([X[:, 1].max() + 2])[0])
        d = self._preparar(cat, T)
        N, nc = len(d["t"]), len(d["celdas_u"])
        th0 = dict_a_theta({"K": 0.01, "alpha": 1.0, "c": 0.01, "p": 1.1, "D": 25.0, "gamma": 0.5, "q": 1.5})
        conteo = np.bincount(d["idx_mu"], minlength=nc)
        x0 = np.concatenate([th0, np.log(0.5 * conteo / T)])
        lim = list(LIMITES_THETA) + [(-30.0, 10.0)] * nc

        def objetivo(x):
            ll, g, gmu = log_verosimilitud(x[:9], x[9:], d)
            return -ll / N, -np.concatenate([g, gmu]) / N

        res = minimize(objetivo, x0, jac=True, method="L-BFGS-B", bounds=lim,
                       options=dict(maxiter=int(self.max_iter), ftol=self.tol, gtol=1e-7, maxcor=30))
        self.theta_ = res.x[:9].copy()
        self.parametros_ = theta_a_dict(self.theta_)
        mu = np.zeros(self._rejilla()[0] * self._rejilla()[1])
        mu[d["celdas_u"]] = np.exp(res.x[9:])
        self.mu_mle_ = mu.copy()
        self.log_verosimilitud_ = float(-res.fun * N)
        self.n_iter_ = int(res.nit)
        self.convergio_ = bool(res.success)
        self.n_eventos_ = N
        self.n_pares_ = len(d["pi"])
        self.T_ajuste_ = T
        lam = log_verosimilitud(self.theta_, res.x[9:], d, con_gradiente=False)[1]
        self.prob_fondo_ = (np.exp(res.x[9:])[d["idx_mu"]] / d["area"]) / lam   # desagrupamiento estocástico
        self._datos_fondo = (d["t"], d["celda"])
        self.mu_ = self._fondo(self.ventana_fondo)
        self.kappa_masa_ = float(np.mean(self._kappa(d["m"], d["prof"]) * self._masa(d["m"], d["prof"])))
        P = self.parametros_
        self.razon_ramificacion_ = self.kappa_masa_ * float(omori_acumulada(self.t_max, P["c"], P["p"], self.t_max))
        self.factor_cascada_ = self._factor_cascada() if self.cascada else 1.0
        return self

    def reestimar_fondo(self, ventana):
        """Cambia ``ventana_fondo`` de un modelo ya ajustado sin repetir la máxima verosimilitud."""
        check_is_fitted(self, "theta_")
        self.ventana_fondo = ventana
        self._validar_hiper()
        self.mu_ = self._fondo(ventana)
        return self

    def _fondo(self, ventana):
        """Tasa de fondo por celda (sismos/día): MLE o desagrupamiento de los últimos ``ventana`` días."""
        if ventana is None:
            return self.mu_mle_.copy()
        t, celda = self._datos_fondo
        sel = t >= self.T_ajuste_ - ventana
        dur = min(float(ventana), self.T_ajuste_)
        mu = np.bincount(celda[sel], self.prob_fondo_[sel], minlength=len(self.mu_mle_)) / dur
        piso = 0.5 / self.T_ajuste_                    # celdas sin eventos en la ventana: medio evento en todo el periodo
        return np.where(self.mu_mle_ > 0, np.maximum(mu, piso * (self.mu_mle_ > 0)), 0.0)

    def _kappa(self, m, prof):
        P = self.parametros_
        return P["K"] * np.exp(P["alpha"] * (m - self.m0) + P["delta_prof"] * prof)

    def _sigma2(self, m, prof):
        P = self.parametros_
        return P["D"] * np.exp(P["gamma"] * (m - self.m0) + P["eps_prof"] * prof)

    def _masa(self, m, prof):
        return masa_espacial(self.r_max, self._sigma2(m, prof), self.parametros_["q"])

    def _factor_cascada(self, L=30.4375, n=2000):
        P = self.parametros_
        s = (np.arange(n) + 0.5) * L / n
        Hm = omori_acumulada(s, P["c"], P["p"], self.t_max).mean()     # (1/L) ∫_0^L H(s) ds
        nw = min(self.kappa_masa_ * Hm, 0.9)
        self.n_ventana_ = float(nw)
        return 1.0 / (1.0 - nw)

    # ------------------------------------------------------------------ pronóstico
    def _fracciones_celda(self, lat, lon, m, prof):
        """Matriz dispersa (eventos × celdas) con ∫_celda f(x - x_i | m_i) dx por cuadratura determinista."""
        q = self.parametros_["q"]
        s2 = self._sigma2(m, prof)
        FR = masa_espacial(self.r_max, s2, q)
        nr, na = int(self.n_radial), int(self.n_angular)
        u = (np.arange(nr) + 0.5) / nr
        ang = 2 * np.pi * (np.arange(na)[None, :] + 0.5 * (np.arange(nr)[:, None] % 2)) / na   # anillos alternados
        n_cel = int(np.prod(self._rejilla()))
        filas, cols, vals = [], [], []
        for i0 in range(0, len(m), 2000):
            sl = slice(i0, i0 + 2000)
            r = np.sqrt(s2[sl, None] * ((1 - u[None, :] * FR[sl, None]) ** (1 / (1 - q)) - 1))      # (b, nr)
            dy = (r[:, :, None] * np.sin(ang)[None]) / KM_LAT
            la = lat[sl, None, None] + dy
            dx = (r[:, :, None] * np.cos(ang)[None]) / (KM_LON * np.cos(np.radians(lat[sl, None, None] + 0.5 * dy)))
            lo = lon[sl, None, None] + dx
            ce = self._celda_de(la, lo).reshape(la.shape[0], -1)
            ev = np.repeat(np.arange(i0, i0 + la.shape[0]), nr * na)
            ce = ce.ravel()
            ok = ce >= 0
            clave = ev[ok].astype(np.int64) * n_cel + ce[ok]
            cu, cnt = np.unique(clave, return_counts=True)
            filas.append(cu // n_cel)
            cols.append(cu % n_cel)
            vals.append(cnt / (nr * na) * FR[cu // n_cel])
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(len(m), n_cel))

    def tasa_mensual(self, meses):
        """Número esperado de sismos por celda (todas las celdas de la rejilla) en el mes ``t + 1`` para cada
        ``t`` de ``meses`` con la información hasta el cierre del mes ``t``. Devuelve matriz (celdas × meses)."""
        check_is_fitted(self, "theta_")
        meses = np.asarray(meses, int)
        cat = self._catalogo()
        ini = self._dia_mes(meses + 1)
        fin = self._dia_mes(meses + 2)
        usa = cat["t"] < ini.max()
        t, m, prof = cat["t"][usa], cat["m"][usa], cat["prof"][usa]
        S = self._fracciones_celda(cat["lat"][usa], cat["lon"][usa], m, prof)
        P = self.parametros_
        kap = self._kappa(m, prof)
        # matriz dispersa eventos × meses con κ_i [H(fin - t_i) - H(ini - t_i)]
        fil, col, val = [], [], []
        for k, (a, b) in enumerate(zip(ini, fin)):
            i = np.where((t < a) & (t >= a - self.t_max))[0]
            w = kap[i] * (omori_acumulada(b - t[i], P["c"], P["p"], self.t_max) - omori_acumulada(a - t[i], P["c"], P["p"], self.t_max))
            fil.append(i); col.append(np.full(len(i), k)); val.append(w)
        W = sparse.csr_matrix((np.concatenate(val), (np.concatenate(fil), np.concatenate(col))), shape=(len(t), len(meses)))
        disparo = np.asarray((S.T @ W).todense())
        fondo = self.mu_[:, None] * (fin - ini)[None, :]
        return self.factor_cascada_ * (fondo + disparo)

    def predict(self, X):
        """Número esperado de sismos M >= m0 de la celda en el mes t + 1 (filas [celda, t])."""
        check_is_fitted(self, "theta_")
        X = self._validar_X(X)
        mu_, inv = np.unique(X[:, 1], return_inverse=True)
        L = self.tasa_mensual(mu_)
        return L[X[:, 0], inv]

    def predecir_log1p(self, X, k_max=None):
        """E[log(1 + N)] con N ~ Poisson(Λ) y Λ = ``predict(X)``."""
        return esperado_log1p_poisson(self.predict(X), k_max)


def esperado_log1p_poisson(lam, k_max=None):
    """E[log(1 + N)], N ~ Poisson(λ), sumando la función de probabilidad hasta k_max."""
    lam = np.maximum(np.asarray(lam, float), 1e-12)
    if k_max is None:
        k_max = int(np.ceil(lam.max() + 12 * np.sqrt(lam.max()) + 30))
    k = np.arange(k_max + 1)
    out = np.empty(len(lam))
    for i in range(0, len(lam), 2000):
        l = lam[i:i + 2000, None]
        lp = k * np.log(l) - l - gammaln(k + 1)
        out[i:i + 2000] = (np.exp(lp) * np.log1p(k)).sum(1)
    return out


# ---------------------------------------------------------------------------
# Simulador (datos sintéticos con parámetros conocidos)
# ---------------------------------------------------------------------------
def simular_etas(par: dict, mu_densidad: float, T: float, region: tuple, m0=2.5, b=1.0, m_max=7.0, t_max=365.0,
                 r_max=100.0, inicio=cfg.INICIO, semilla=0, profundidad=10.0) -> pd.DataFrame:
    """Catálogo ETAS sintético por generaciones (fondo homogéneo ``mu_densidad`` sismos/día/km² en ``region``).

    ``region`` = (lat_min, lat_max, lon_min, lon_max). Magnitudes Gutenberg-Richter truncadas en ``m_max``.
    Los hijos que caen fuera de la región se descartan (como en un catálogo real con frontera).
    """
    rng = np.random.default_rng(semilla)
    la0, la1, lo0, lo1 = region
    lat_c = 0.5 * (la0 + la1)
    area = (la1 - la0) * KM_LAT * (lo1 - lo0) * KM_LON * np.cos(np.radians(lat_c))
    beta = b * np.log(10)

    def mags(n):
        u = rng.random(n)
        return m0 - np.log(1 - u * (1 - np.exp(-beta * (m_max - m0)))) / beta

    n0 = rng.poisson(mu_densidad * area * T)
    t = rng.random(n0) * T
    lat = la0 + rng.random(n0) * (la1 - la0)
    lon = lo0 + rng.random(n0) * (lo1 - lo0)
    m = mags(n0)
    todos = [(t, lat, lon, m)]
    gen = (t, lat, lon, m)
    c, p, D, q = par["c"], par["p"], par["D"], par["q"]
    HT = float(omori_acumulada(t_max, c, p, t_max))
    while len(gen[0]):
        tp, lap, lop, mp = gen
        s2 = D * np.exp(par["gamma"] * (mp - m0))
        nk = rng.poisson(par["K"] * np.exp(par["alpha"] * (mp - m0)) * HT * masa_espacial(r_max, s2, q))
        if nk.sum() == 0:
            break
        pa = np.repeat(np.arange(len(tp)), nk)
        u = rng.random(len(pa)) * HT
        dt = omori_inversa(u, c, p)
        FR = masa_espacial(r_max, s2[pa], q)
        r = np.sqrt(s2[pa] * ((1 - rng.random(len(pa)) * FR) ** (1 / (1 - q)) - 1))
        th = rng.random(len(pa)) * 2 * np.pi
        la = lap[pa] + r * np.sin(th) / KM_LAT
        lo = lop[pa] + r * np.cos(th) / (KM_LON * np.cos(np.radians(la)))
        tt = tp[pa] + dt
        ok = (tt < T) & (la >= la0) & (la < la1) & (lo >= lo0) & (lo < lo1)
        gen = (tt[ok], la[ok], lo[ok], mags(ok.sum()))
        todos.append(gen)
    t, lat, lon, m = (np.concatenate(a) for a in zip(*todos))
    o = np.argsort(t)
    return pd.DataFrame({"time": pd.Timestamp(inicio) + pd.to_timedelta(t[o], unit="D"), "latitude": lat[o],
                         "longitude": lon[o], "depth": profundidad, "mag": np.round(m[o], 6)})
