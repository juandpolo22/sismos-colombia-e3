"""Modelo propuesto A+B (implementación propia; ver capítulo del modelo propuesto).

Piezas, todas escritas en este proyecto (las librerías solo aportan álgebra y L-BFGS; los árboles débiles de B
son el CART propio de src/arbol_cart.py):

* :class:`ZINB` — regresión binomial negativa inflada en ceros con excitación tipo ETAS; verosimilitud y
  gradiente analíticos; pronóstico puntual óptimo para RMSLE: E[log(1+y)].
* :class:`BoostingZINB` — **modelo B**: boosting de la verosimilitud ZINB (log-media y logit de ceros
  aditivos en árboles débiles con paso de Newton; dispersión por máxima verosimilitud; parada temprana
  cronológica). Familia: boosting de gradiente (Friedman, 2001); GAMLSS por boosting (Mayr et al., 2012).
* :class:`CombinadorSoftmax` — **modelo A**: combinación de pronósticos con pesos softmax que dependen de
  meta-características de cada celda-mes (idea de FFORMA, Montero-Manso et al., 2020), entrenada con
  gradiente analítico y L-BFGS-B minimizando el error cuadrático en escala log1p.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma, expit, gammaln, softmax

from . import config as cfg
from .arbol_cart import ArbolCART

SEMILLA = cfg.SEMILLA


class ZINB:
    """Regresión binomial negativa inflada en ceros (ZINB) ajustada por máxima verosimilitud.

    La media del conteo es ``exp(Xm β)``, la probabilidad de cero estructural ``σ(W γ)`` y la dispersión ``α = exp(λ)``.
    Se ajusta con L-BFGS-B y gradiente analítico, con penalización L2 ``l2``.

    Parameters
    ----------
    l2 : float
        Peso de la penalización L2 sobre los coeficientes.
    """
    def __init__(self, l2=1e-3):
        self.l2 = l2

    def _nll(self, th, Xm, W, y):
        pm, pw = Xm.shape[1], W.shape[1]
        beta, gam, la = th[:pm], th[pm:pm + pw], th[-1]
        la = np.clip(la, -6, 4)
        al = np.exp(la)
        r = 1.0 / al
        eta = np.clip(Xm @ beta, -12, 9)
        mu = np.exp(eta)
        pi = expit(np.clip(W @ gam, -30, 30))
        lr = np.log(r / (r + mu))
        p0 = np.exp(r * lr)
        y0 = y == 0
        lognb = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * lr + y * (eta - np.log(r + mu))
        q = pi + (1 - pi) * p0
        ll = np.where(y0, np.log(np.maximum(q, 1e-300)), np.log(np.maximum(1 - pi, 1e-300)) + lognb)
        # gradientes por fila
        d_eta = np.where(y0, (1 - pi) * (-p0 * mu / (1 + al * mu)) / np.maximum(q, 1e-300), (y - mu) / (1 + al * mu))
        d_xi = np.where(y0, pi * (1 - pi) * (1 - p0) / np.maximum(q, 1e-300), -pi)
        d_r_nb = digamma(y + r) - digamma(r) + lr + (mu - y) / (r + mu)
        d_r_p0 = p0 * (lr + mu / (r + mu))
        d_la = np.where(y0, (1 - pi) * d_r_p0 * (-r) / np.maximum(q, 1e-300), d_r_nb * (-r))
        pen = self.l2 * (np.sum(beta[1:] ** 2) + np.sum(gam[1:] ** 2))
        f = -ll.sum() + pen * len(y) / 1000.0
        g = np.concatenate([-(Xm.T @ d_eta), -(W.T @ d_xi), [-d_la.sum()]])
        g[1:pm] += 2 * self.l2 * beta[1:] * len(y) / 1000.0
        g[pm + 1:pm + pw] += 2 * self.l2 * gam[1:] * len(y) / 1000.0
        return f, g

    def fit(self, Xm, W, y):
        """Ajusta los parámetros por máxima verosimilitud penalizada.

        Parameters
        ----------
        Xm : np.ndarray de forma (n, pm)
            Diseño de la media (con intercepto).
        W : np.ndarray de forma (n, pw)
            Diseño de la inflación de ceros (con intercepto).
        y : np.ndarray de forma (n,)
            Conteos observados.

        Returns
        -------
        self
        """
        pm, pw = Xm.shape[1], W.shape[1]
        th0 = np.zeros(pm + pw + 1)
        th0[0] = np.log(max(y.mean(), 0.05))
        th0[pm] = -0.5
        th0[-1] = 0.0
        r = minimize(self._nll, th0, args=(Xm, W, y), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.th, self.pm, self.pw, self.res = r.x, pm, pw, r
        return self

    def params(self, Xm, W):
        """Parámetros ajustados para nuevas filas.

        Returns
        -------
        tuple
            ``(mu, pi, alpha)``: media de la parte binomial negativa, probabilidad de cero estructural y dispersión.
        """
        beta, gam, la = self.th[:self.pm], self.th[self.pm:self.pm + self.pw], np.clip(self.th[-1], -6, 4)
        return np.exp(np.clip(Xm @ beta, -12, 9)), expit(np.clip(W @ gam, -30, 30)), np.exp(la)

    def predecir(self, Xm, W, K=900, bloque=1500):
        """Pronósticos para nuevas filas.

        Returns
        -------
        tuple
            ``(E[log1p Y], E[Y])``. El primero minimiza el RMSLE y es el que se combina; la suma sobre k se trunca en ``K``.
        """
        mu, pi, al = self.params(Xm, W)
        r = 1.0 / al
        ks = np.arange(1, K + 1)
        lk = np.log1p(ks)
        e_log = np.zeros(len(mu))
        for i in range(0, len(mu), bloque):
            m = mu[i:i + bloque, None]
            lp = (gammaln(ks + r) - gammaln(r) - gammaln(ks + 1) + r * np.log(r / (r + m)) + ks * (np.log(m) - np.log(r + m)))
            e_log[i:i + bloque] = (1 - pi[i:i + bloque]) * (np.exp(lp) * lk).sum(1)
        return e_log, (1 - pi) * mu


def disenar(X, XE, idx_cols):
    """Matrices de diseño (mu y pi) con estandarización de train."""
    c = lambda n: np.log1p(X[n].values.astype(float))  # noqa: E731
    Z = np.column_stack([c("n_1m"), c("n_3m"), c("n_12m"), c("n35_12m"), c("vec_1m"),
                         XE["exc_a0.5_s30_r0.4"].values, XE["exc_a0.9_s30_r0.8"].values,
                         XE["exc_a0.5_s15_r0.4"].values, XE["exc_a0.9_s60_r0.8"].values,
                         X["sen_mes"].values, X["cos_mes"].values])
    Wz = np.column_stack([c("n_12m"), c("n_3m"), XE["exc_a0.5_s30_r0.4"].values])
    mu_, sd_ = Z[idx_cols].mean(0), Z[idx_cols].std(0) + 1e-9
    mw, sw = Wz[idx_cols].mean(0), Wz[idx_cols].std(0) + 1e-9
    Z, Wz = (Z - mu_) / sd_, (Wz - mw) / sw
    one = np.ones((len(Z), 1))
    return np.hstack([one, Z]), np.hstack([one, Wz])


class CombinadorSoftmax:
    """Combinador contextual: pesos softmax que dependen del estado de cada celda.

    El pronóstico es ``Σ_k w_k(z) b_k`` con ``w(z) = softmax(z T)``; se minimiza el error cuadrático en escala log1p más
    una penalización L2 sobre ``T`` (sin la fila del intercepto), con gradiente analítico y L-BFGS-B.

    Parameters
    ----------
    l2 : float
        Peso de la penalización L2 (λ).
    """
    def __init__(self, l2=1e-2):
        self.l2 = l2

    def _f(self, th, Z, B, y):
        d, K = Z.shape[1], B.shape[1]
        T = th.reshape(d, K)
        W = softmax(Z @ T, axis=1)
        pred = (W * B).sum(1)
        r = pred - y
        n = len(y)
        pen = self.l2 * np.sum(T[1:] ** 2)
        f = np.mean(r ** 2) + pen
        dlog = (2 * r / n)[:, None] * W * (B - pred[:, None])
        G = Z.T @ dlog
        G[1:] += 2 * self.l2 * T[1:]
        return f, G.ravel()

    def fit(self, Z, B, y):
        """Ajusta la matriz de pesos ``T``.

        Parameters
        ----------
        Z : np.ndarray de forma (n, d)
            Variables de contexto (con intercepto).
        B : np.ndarray de forma (n, K)
            Pronósticos log1p de los K miembros, fuera de pliegue.
        y : np.ndarray de forma (n,)
            Objetivo en escala log1p.

        Returns
        -------
        self
        """
        th0 = np.zeros(Z.shape[1] * B.shape[1])
        r = minimize(self._f, th0, args=(Z, B, y), jac=True, method="L-BFGS-B", options=dict(maxiter=400))
        self.T = r.x.reshape(Z.shape[1], B.shape[1])
        self.res = r
        return self

    def pesos(self, Z):
        """Pesos de cada miembro por fila, de forma (n, K); cada fila suma 1."""
        return softmax(Z @ self.T, axis=1)

    def predict(self, Z, B):
        """Pronóstico combinado en escala log1p: suma de los pronósticos de los miembros ponderados por fila."""
        return (self.pesos(Z) * B).sum(1)


# ----------------------------------------------------------------------------
# Modelo B: boosting ZINB propio
# ----------------------------------------------------------------------------
def _comp(eta, xi, la, y):
    al = np.exp(la)
    r = 1.0 / al
    mu = np.exp(np.clip(eta, -12, 9))
    pi = expit(np.clip(xi, -30, 30))
    lr = np.log(r / (r + mu))
    p0 = np.exp(r * lr)
    return al, r, mu, pi, lr, p0


def nll_zinb(eta, xi, la, y):
    """Log-verosimilitud negativa de la ZINB en los predictores aditivos.

    Parameters
    ----------
    eta : np.ndarray
        Log de la media de la binomial negativa.
    xi : np.ndarray
        Logit de la probabilidad de cero estructural.
    la : float
        Log de la dispersión.
    y : np.ndarray
        Conteos observados.

    Returns
    -------
    float
        Suma de la log-verosimilitud negativa (lo que minimiza el boosting).
    """
    al, r, mu, pi, lr, p0 = _comp(eta, xi, la, y)
    lognb = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * lr + y * (np.log(mu) - np.log(r + mu))
    q = pi + (1 - pi) * p0
    ll = np.where(y == 0, np.log(np.maximum(q, 1e-300)), np.log(np.maximum(1 - pi, 1e-300)) + lognb)
    return -ll.sum()


def grads_zinb(eta, xi, la, y):
    """derivadas de la log-verosimilitud respecto a eta y xi, y curvaturas de Fisher aproximadas"""
    al, r, mu, pi, lr, p0 = _comp(eta, xi, la, y)
    y0 = y == 0
    q = np.maximum(pi + (1 - pi) * p0, 1e-300)
    g_eta = np.where(y0, (1 - pi) * (-p0 * mu / (1 + al * mu)) / q, (y - mu) / (1 + al * mu))
    g_xi = np.where(y0, pi * (1 - pi) * (1 - p0) / q, -pi)
    h_eta = np.maximum((1 - pi) * mu / (1 + al * mu), 1e-3)
    h_xi = np.maximum(pi * (1 - pi), 1e-3)
    return g_eta, g_xi, h_eta, h_xi


def esperado_log1p(mu, pi, al, K=900, bloque=1500):
    """Valor esperado de log(1 + Y) bajo la ZINB, que es el pronóstico que minimiza el RMSLE.

    Parameters
    ----------
    mu, pi, al : np.ndarray o float
        Media, probabilidad de cero estructural y dispersión.
    K : int
        Truncamiento de la suma sobre k = 1..K (la masa por encima es despreciable para las medias del panel).
    bloque : int
        Filas procesadas a la vez, para acotar la memoria.

    Returns
    -------
    np.ndarray
        ``E[log1p Y]`` por fila.
    """
    r = 1.0 / al
    ks = np.arange(1, K + 1)
    lk = np.log1p(ks)
    out = np.zeros(len(mu))
    for i in range(0, len(mu), bloque):
        m = mu[i:i + bloque, None]
        lp = gammaln(ks + r) - gammaln(r) - gammaln(ks + 1) + r * np.log(r / (r + m)) + ks * (np.log(m) - np.log(r + m))
        out[i:i + bloque] = (1 - pi[i:i + bloque]) * (np.exp(lp) * lk).sum(1)
    return out


class BoostingZINB:
    """Boosting de la verosimilitud ZINB (miembro B del modelo propuesto).

    En cada iteración ajusta dos árboles CART propios por pasos de Newton (al cociente −g/h con pesos h): uno para el
    log de la media y otro para el logit de la inflación de ceros. La dispersión se estima por máxima verosimilitud (búsqueda acotada) y el número
    de iteraciones se fija con parada temprana cronológica.

    Parameters
    ----------
    n_max : int
        Iteraciones máximas.
    lr : float
        Tasa de aprendizaje.
    depth : int
        Profundidad máxima de cada árbol.
    min_leaf : int
        Mínimo de filas por hoja.
    seed : int
        Semilla (por defecto la semilla global).
    """
    def __init__(self, n_max=300, lr=0.08, depth=3, min_leaf=50, seed=SEMILLA):
        self.n_max, self.lr, self.depth, self.min_leaf, self.seed = n_max, lr, depth, min_leaf, seed

    def _run(self, X, y, n_iter, Xv=None, yv=None):
        n = len(y)
        eta = np.full(n, np.log(max(y.mean(), 0.05)))
        xi = np.full(n, -0.8)
        la = 0.0
        self.eta0, self.xi0 = eta[0], xi[0]
        self.te, self.tx = [], []
        if Xv is not None:
            ev, xv = np.full(len(yv), self.eta0), np.full(len(yv), self.xi0)
        hist = []
        orden = ArbolCART.ordenar(X)                      # el orden de las filas por variable se calcula una vez
        for m in range(n_iter):
            g_e, g_x, h_e, h_x = grads_zinb(eta, xi, la, y)
            t1 = ArbolCART(max_depth=self.depth, min_samples_leaf=self.min_leaf, random_state=self.seed + m)  # CART propio
            t1.fit(X, np.clip(g_e / h_e, -3, 3), sample_weight=h_e, orden=orden)
            eta = eta + self.lr * t1.predict(X)
            g_e, g_x, h_e, h_x = grads_zinb(eta, xi, la, y)
            t2 = ArbolCART(max_depth=self.depth, min_samples_leaf=self.min_leaf, random_state=self.seed + 7919 + m)
            t2.fit(X, np.clip(g_x / h_x, -3, 3), sample_weight=h_x, orden=orden)
            xi = xi + self.lr * t2.predict(X)
            self.te.append(t1)
            self.tx.append(t2)
            if (m + 1) % 20 == 0:
                la = minimize_scalar(lambda v: nll_zinb(eta, xi, v, y), bounds=(-4, 3), method="bounded").x
            self.la = la
            if Xv is not None:
                ev = ev + self.lr * t1.predict(Xv)
                xv = xv + self.lr * t2.predict(Xv)
                hist.append(nll_zinb(ev, xv, la, yv))
        return hist

    def fit(self, X, y, tm_rows):
        """elige el número de iteraciones con el último 20 % de meses; reajusta con todo"""
        meses = np.unique(tm_rows)
        corte = meses[int(len(meses) * 0.8)]
        a, b = tm_rows < corte, tm_rows >= corte
        hist = self._run(X[a], y[a], self.n_max, X[b], y[b])
        self.n_best = int(np.argmin(hist)) + 1
        self.curva_validacion_ = np.asarray(hist)      # NLL de validación por iteración (curva de convergencia)
        self._run(X, y, self.n_best)
        return self

    def params(self, X):
        """Parámetros ZINB para nuevas filas, sumando la contribución de todos los árboles.

        Returns
        -------
        tuple
            ``(mu, pi, alpha)``.
        """
        eta = np.full(len(X), self.eta0)
        xi = np.full(len(X), self.xi0)
        for t1, t2 in zip(self.te, self.tx):
            eta = eta + self.lr * t1.predict(X)
            xi = xi + self.lr * t2.predict(X)
        return np.exp(np.clip(eta, -12, 9)), expit(np.clip(xi, -30, 30)), np.exp(self.la)

    def predecir(self, X):
        """Pronósticos para nuevas filas.

        Returns
        -------
        tuple
            ``(E[log1p Y], E[Y])``.
        """
        mu, pi, al = self.params(X)
        return esperado_log1p(mu, pi, al), (1 - pi) * mu




# ----------------------------------------------------------------------------
# Predicciones fuera de pliegue cronológicas (solo pasado)
# ----------------------------------------------------------------------------
def pliegues_cronologicos(tm_rows, k=5, gap=1):
    """Divide los meses en ``k`` bloques contiguos y, para cada bloque f ≥ 1, entrena con los meses anteriores
    al inicio del bloque menos ``gap`` y predice el bloque. El primer bloque no tiene predicción (no hay pasado).

    Returns
    -------
    lista de (índices de ajuste, índices a predecir) y máscara booleana de filas con predicción.
    """
    meses = np.unique(tm_rows)
    bloque = np.minimum((np.searchsorted(meses, tm_rows) * k) // len(meses), k - 1)
    pares = []
    for f in range(1, k):
        inicio = tm_rows[bloque == f].min()
        pares.append((np.where(tm_rows < inicio - gap)[0], np.where(bloque == f)[0]))
    return pares, bloque >= 1

