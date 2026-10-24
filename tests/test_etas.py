"""Pruebas del miembro ETAS (src/etas.py). Ejecutar con:  python -m pytest tests/test_etas.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.etas import (ETAS, NOMBRES_THETA, dict_a_theta, esperado_log1p_poisson, log_verosimilitud, masa_espacial,
                      omori_acumulada, omori_inversa, simular_etas)

REGION = (2.0, 6.0, -76.0, -72.0)          # 8 × 8 celdas de 0,5°
VERDAD = {"K": 0.011, "alpha": 1.5, "c": 0.02, "p": 1.2, "D": 20.0, "gamma": 0.6, "q": 1.6}
MU = 5e-6                                  # sismos/día/km² de fondo
T_DIAS = 3652.0
REJ = dict(lat_min=REGION[0], lat_max=REGION[1], lon_min=REGION[2], lon_max=REGION[3])


def _X(meses, n_cel=64):
    cc, tt = np.meshgrid(np.arange(n_cel), np.asarray(meses), indexing="ij")
    return np.c_[cc.ravel(), tt.ravel()]


@pytest.fixture(scope="module")
def catalogo():
    return simular_etas(VERDAD, MU, T_DIAS, REGION, semilla=7)


@pytest.fixture(scope="module")
def ajustado(catalogo):
    return ETAS(catalogo, **REJ).fit(_X([118]))          # último mes objetivo 119 = fin de los 10 años


# ---------------------------------------------------------------- núcleos
@pytest.mark.parametrize("p", [0.6, 1.0, 1.0 + 1e-7, 1.3])
def test_omori_acumulada_coincide_con_integral_numerica(p):
    c, s = 0.05, 40.0
    num = quad(lambda u: (u + c) ** (-p), 0, s, limit=200)[0]
    assert omori_acumulada(s, c, p, 365.0) == pytest.approx(num, rel=1e-7)
    assert omori_acumulada(500.0, c, p, 365.0) == pytest.approx(omori_acumulada(365.0, c, p, 365.0))   # truncamiento
    assert omori_acumulada(-1.0, c, p, 365.0) == 0.0
    assert omori_inversa(omori_acumulada(s, c, p, 1e9), c, p) == pytest.approx(s, rel=1e-6)


def test_derivadas_omori():
    c, p, s, h = 0.03, 1.1, 20.0, 1e-6
    H, dlc, dp = omori_acumulada(s, c, p, 365.0, derivadas=True)
    assert dlc == pytest.approx((omori_acumulada(s, c * np.exp(h), p, 365.0) - omori_acumulada(s, c * np.exp(-h), p, 365.0)) / (2 * h), rel=1e-5)
    assert dp == pytest.approx((omori_acumulada(s, c, p + h, 365.0) - omori_acumulada(s, c, p - h, 365.0)) / (2 * h), rel=1e-5)


def test_masa_espacial_normalizada():
    s2, q = 30.0, 1.7
    num = quad(lambda r: 2 * np.pi * r * (q - 1) / (np.pi * s2) * (1 + r * r / s2) ** (-q), 0, 50.0)[0]
    assert masa_espacial(50.0, s2, q) == pytest.approx(num, rel=1e-8)


def test_esperado_log1p_poisson():
    lam = np.array([0.01, 0.7, 3.0, 40.0])
    k = np.arange(400)
    exacto = [(poisson.pmf(k, l) * np.log1p(k)).sum() for l in lam]
    assert np.allclose(esperado_log1p_poisson(lam), exacto, rtol=1e-10)


# ---------------------------------------------------------------- verosimilitud
def test_gradiente_analitico_igual_a_diferencias_finitas(catalogo):
    m = ETAS(catalogo, **REJ)
    cat = m._catalogo()
    cat["prof"] = (np.arange(len(cat["t"])) % 3 == 0).astype(float)      # activa los factores de profundidad
    d = m._preparar(cat, T_DIAS)
    th = dict_a_theta({**VERDAD, "delta_prof": 0.3, "eps_prof": -0.4}) + 0.05
    lm = np.log(0.5 * np.bincount(d["idx_mu"]) / T_DIAS)
    _, g, gmu = log_verosimilitud(th, lm, d)
    h = 1e-6
    for k in range(9):
        e = np.zeros(9)
        e[k] = h
        fd = (log_verosimilitud(th + e, lm, d, False)[0] - log_verosimilitud(th - e, lm, d, False)[0]) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-4, abs=1e-3), NOMBRES_THETA[k]
    e = np.zeros(len(lm))
    e[5] = h
    fd = (log_verosimilitud(th, lm + e, d, False)[0] - log_verosimilitud(th, lm - e, d, False)[0]) / (2 * h)
    assert gmu[5] == pytest.approx(fd, rel=1e-4, abs=1e-3)


# ---------------------------------------------------------------- datos sintéticos con parámetros conocidos
def test_recupera_parametros_conocidos(ajustado):
    P = ajustado.parametros_
    assert ajustado.convergio_
    assert P["alpha"] == pytest.approx(VERDAD["alpha"], abs=0.25)
    assert P["p"] == pytest.approx(VERDAD["p"], abs=0.1)
    assert np.log(P["c"]) == pytest.approx(np.log(VERDAD["c"]), abs=0.7)
    assert P["gamma"] == pytest.approx(VERDAD["gamma"], abs=0.3)
    # tasa de fondo total (sismos/día) y razón de ramificación
    area = (4 * 110.574) * (4 * 111.320 * np.cos(np.radians(4.0)))
    assert ajustado.mu_.sum() == pytest.approx(MU * area, rel=0.1)
    m_ver = ETAS(None, **REJ)
    m_ver.parametros_ = {**VERDAD, "delta_prof": 0.0, "eps_prof": 0.0}
    cat = ajustado._catalogo()
    n_ver = np.mean(m_ver._kappa(cat["m"], 0 * cat["m"]) * m_ver._masa(cat["m"], 0 * cat["m"])) * omori_acumulada(365, VERDAD["c"], VERDAD["p"], 365)
    assert ajustado.razon_ramificacion_ == pytest.approx(n_ver, abs=0.07)


def test_pronostico_calibrado_en_agregado(catalogo, ajustado):
    meses = np.arange(12, 119)
    L = ajustado.tasa_mensual(meses)                       # celdas × meses
    m = ETAS(catalogo, **REJ)
    cat = m._catalogo()
    ini, fin = m._dia_mes(meses + 1), m._dia_mes(meses + 2)
    obs = np.array([((cat["t"] >= a) & (cat["t"] < b)).sum() for a, b in zip(ini, fin)])
    assert L.sum() == pytest.approx(obs.sum(), rel=0.08)


def test_integral_espacial_por_celda_coincide_con_monte_carlo(ajustado):
    """La cuadratura determinista de ∫_celda f(x - x_i | m_i) dx coincide con Monte Carlo (10⁶ puntos)."""
    lat, lon, mag = np.array([4.1]), np.array([-74.05]), np.array([4.5])
    S = ajustado._fracciones_celda(lat, lon, mag, np.zeros(1)).toarray()[0]
    P = ajustado.parametros_
    s2, q = ajustado._sigma2(mag, np.zeros(1))[0], P["q"]
    rng = np.random.default_rng(0)
    FR = masa_espacial(ajustado.r_max, s2, q)
    r = np.sqrt(s2 * ((1 - rng.random(10 ** 6) * FR) ** (1 / (1 - q)) - 1))
    th = rng.random(10 ** 6) * 2 * np.pi
    la = lat[0] + r * np.sin(th) / 110.574
    lo = lon[0] + r * np.cos(th) / (111.320 * np.cos(np.radians(0.5 * (la + lat[0]))))
    ce = ajustado._celda_de(la, lo)
    mc = np.bincount(ce[ce >= 0], minlength=len(S)) / 10 ** 6 * FR
    assert S.sum() == pytest.approx(mc.sum(), abs=0.01)
    assert np.abs(S - mc).max() < 0.02


def test_predict_por_filas_coincide_con_tasa_mensual(ajustado):
    X = _X([30, 31, 77])
    p = ajustado.predict(X)
    L = ajustado.tasa_mensual(np.array([30, 31, 77]))
    assert p.shape == (len(X),) and np.all(p > 0)
    assert np.allclose(p, L[X[:, 0], np.searchsorted([30, 31, 77], X[:, 1])])
    assert np.allclose(ajustado.predecir_log1p(X), esperado_log1p_poisson(p))


# ---------------------------------------------------------------- causalidad (sin fuga)
def test_sin_fuga_en_pronostico_ni_en_ajuste(catalogo, ajustado):
    """Cambiar los sismos desde el inicio del mes pronosticado no cambia el pronóstico; cambiar los posteriores
    al fin del periodo de ajuste no cambia los parámetros."""
    t_corte = pd.Timestamp("2004-01-01") + pd.DateOffset(months=61)      # inicio del mes 61
    alterado = catalogo.copy()
    fut = alterado.time >= t_corte
    alterado.loc[fut, "mag"] = alterado.loc[fut, "mag"] + 1.0
    alterado = pd.concat([alterado, alterado[fut]])                          # y duplica el futuro
    X = _X([60])
    m1 = ETAS(catalogo, **REJ).fit(_X([40]))
    m2 = ETAS(alterado, **REJ).fit(_X([40]))
    assert np.allclose(m1.theta_, m2.theta_)
    assert np.allclose(m1.predict(X), m2.predict(X))
    assert not np.allclose(m1.predict(_X([70])), m2.predict(_X([70])))


# ---------------------------------------------------------------- reproducibilidad y API de scikit-learn
def test_reproducible(catalogo):
    X = _X([60])
    a = ETAS(catalogo, **REJ).fit(X)
    b = ETAS(catalogo, **REJ).fit(X)
    assert np.array_equal(a.theta_, b.theta_)
    assert np.array_equal(a.predict(X), b.predict(X))


def test_clone_y_parametros(ajustado):
    c = clone(ajustado)
    assert c.get_params(deep=False).keys() == ajustado.get_params(deep=False).keys()
    assert c.t_max == ajustado.t_max
    pd.testing.assert_frame_equal(c.eventos, ajustado.eventos)
    assert not hasattr(c, "theta_")
    with pytest.raises(NotFittedError):
        c.predict(_X([5]))
    c.set_params(t_max=100.0, ventana_fondo=730.0)
    assert c.get_params()["t_max"] == 100.0
    assert ajustado.t_max == 365.0


def test_reestimar_fondo(ajustado):
    import copy
    m = copy.deepcopy(ajustado)
    m.reestimar_fondo(3 * 365.25)
    assert m.ventana_fondo == pytest.approx(3 * 365.25)
    assert m.mu_.sum() == pytest.approx(ajustado.mu_.sum(), rel=0.25)
    assert not np.allclose(m.mu_, ajustado.mu_)


def test_score_r2(ajustado, catalogo):
    m = ETAS(catalogo, **REJ)
    cat = m._catalogo()
    X = _X([80, 81])
    ini, fin = m._dia_mes(X[:, 1] + 1), m._dia_mes(X[:, 1] + 2)
    y = np.array([((cat["celda"] == c) & (cat["t"] >= a) & (cat["t"] < b)).sum() for c, a, b in zip(X[:, 0], ini, fin)])
    assert np.isfinite(ajustado.score(X, y))


@pytest.mark.parametrize("kwargs, X, msg", [
    (dict(eventos=None), _X([10]), "DataFrame"),
    (dict(eventos=pd.DataFrame({"time": [], "mag": []})), _X([10]), "faltan"),
    (dict(t_max=-1.0), _X([10]), "t_max"),
    (dict(r_max=0.0), _X([10]), "r_max"),
    (dict(ventana_fondo=-5.0), _X([10]), "ventana_fondo"),
    (dict(n_radial=0), _X([10]), "n_radial"),
    (dict(m0=9.5), _X([10]), "M >="),
    (dict(), np.zeros((3, 3)), "forma"),
    (dict(), np.array([[0, 10.5]]), "enteros"),
    (dict(), np.array([[64, 10]]), "rejilla"),
    (dict(), np.array([[0, -1]]), "negativo"),
    (dict(), np.zeros((0, 2)), "vacío"),
])
def test_entradas_invalidas(catalogo, kwargs, X, msg):
    kw = {"eventos": catalogo, **REJ, **kwargs}
    with pytest.raises(ValueError, match=msg):
        ETAS(**kw).fit(X)


def test_y_de_longitud_distinta(catalogo):
    with pytest.raises(ValueError, match="longitudes"):
        ETAS(catalogo, **REJ).fit(_X([10]), np.zeros(3))


def test_predict_con_X_invalido(ajustado):
    with pytest.raises(ValueError):
        ajustado.predict(np.array([[0, 1, 2]]))
