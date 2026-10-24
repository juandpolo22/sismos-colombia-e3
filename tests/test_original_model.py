"""Pruebas del modelo original A+B (src/original_model.py), exigidas por la guía del Entregable 3 (sección 5.4):
(i) aprende en datos sintéticos con solución conocida, (ii) reproducibilidad con la misma semilla,
(iii) manejo de entradas inválidas y (iv) compatibilidad con sklearn.base.clone."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.original_model import COLUMNAS, ModeloOriginalAB

# Configuración liviana para que las pruebas tarden segundos: un solo miembro benchmark (Ridge) y B corto.
RAPIDO = dict(miembros=[("Ridge", "Grid", {"alpha": 1.0})], iter_max_b=30, hoja_b=20)


def panel_sintetico(n_celdas=30, n_meses=80, semilla=0):
    """Panel con la estructura real; la media verdadera depende de n_1m: E[y] = 0,2 + 0,6·n_1m."""
    rng = np.random.default_rng(semilla)
    filas = []
    for c in range(n_celdas):
        nivel = rng.gamma(1.0, 1.0)
        serie = rng.poisson(nivel, n_meses + 13)
        for t in range(12, n_meses + 12):
            h = serie[t - np.arange(12)]
            filas.append(dict(celda=c, mes=t + 1, h=h, y=rng.poisson(0.2 + 0.6 * h[0])))
    D = pd.DataFrame({k: 0.0 for k in COLUMNAS}, index=range(len(filas)))
    H = np.array([f["h"] for f in filas], float)
    for j in range(12):
        D[f"h{j}"] = H[:, j]
    D["n_1m"], D["n_3m"], D["n_6m"], D["n_12m"] = H[:, 0], H[:, :3].sum(1), H[:, :6].sum(1), H.sum(1)
    D["n_mismo_mes_anio_prev"] = H[:, 11]
    D["mag_media_3m"] = D["mag_media_12m"] = D["mag_max_3m"] = D["mag_max_12m"] = 3.0
    D["prof_media_hist"] = 30.0
    D["mes"] = [f["mes"] for f in filas]
    D["sen_mes"], D["cos_mes"] = np.sin(2 * np.pi * D["mes"] / 12), np.cos(2 * np.pi * D["mes"] / 12)
    D["lat"] = [f["celda"] % 6 for f in filas]
    D["lon"] = [f["celda"] // 6 for f in filas]
    for c in [c for c in COLUMNAS if c.startswith("exc_") or c == "media_expansiva"]:
        D[c] = np.log1p(D["n_3m"]) * 0.5
    return D, np.array([f["y"] for f in filas], float)


@pytest.fixture(scope="module")
def datos():
    X, y = panel_sintetico()
    corte = X["mes"].quantile(0.8)
    return X[X.mes <= corte], y[X.mes <= corte], X[X.mes > corte], y[X.mes > corte]


@pytest.fixture(scope="module")
def ajustado(datos):
    Xtr, ytr, _, _ = datos
    return ModeloOriginalAB(**RAPIDO).fit(Xtr, ytr)


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def test_aprende_la_senal_conocida(datos, ajustado):
    """Supera claramente a pronosticar la media y se acerca al error de la media verdadera."""
    _, ytr, Xte, yte = datos
    p = ajustado.predict(Xte)
    verdad = 0.2 + 0.6 * Xte["h0"].to_numpy()
    assert rmsle(yte, p) < 0.9 * rmsle(yte, np.full(len(yte), ytr.mean()))
    assert rmsle(yte, p) < rmsle(yte, verdad) + 0.05
    assert (p >= 0).all()


def test_pesos_y_explicacion_consistentes(datos, ajustado):
    _, _, Xte, _ = datos
    W = ajustado.pesos(Xte)
    assert np.allclose(W.sum(1), 1) and (W >= 0).all()
    E = ajustado.explicar(Xte)
    assert np.allclose(np.clip(E.sum(1).to_numpy(), 0, None), ajustado.predict_log(Xte))


def test_reproducible_con_la_misma_semilla(datos, ajustado):
    Xtr, ytr, Xte, _ = datos
    otro = ModeloOriginalAB(**RAPIDO).fit(Xtr, ytr)
    assert np.array_equal(otro.predict(Xte), ajustado.predict(Xte))


def test_entradas_invalidas(datos):
    Xtr, ytr, _, _ = datos
    with pytest.raises(TypeError):
        ModeloOriginalAB(**RAPIDO).fit(Xtr.to_numpy(), ytr)
    with pytest.raises(ValueError):
        ModeloOriginalAB(**RAPIDO).fit(Xtr.drop(columns=["n_1m"]), ytr)
    with pytest.raises(ValueError):
        ModeloOriginalAB(**RAPIDO).fit(Xtr, -ytr - 1)
    with pytest.raises(ValueError):
        ModeloOriginalAB(**RAPIDO).fit(Xtr, ytr[:-1])
    with pytest.raises(Exception):
        ModeloOriginalAB(**RAPIDO).predict(Xtr)          # sin ajustar


def test_compatible_con_clone(datos, ajustado):
    Xtr, ytr, Xte, _ = datos
    c = clone(ajustado)
    assert c.get_params() == ajustado.get_params()
    assert not hasattr(c, "combinador_")
    assert np.array_equal(c.fit(Xtr, ytr).predict(Xte), ajustado.predict(Xte))


def test_ablacion_cambia_los_miembros(datos):
    Xtr, ytr, _, _ = datos
    sin_b = ModeloOriginalAB(**RAPIDO, usar_b=False).fit(Xtr, ytr)
    assert "B: boosting ZINB (propio)" not in sin_b.nombres_
