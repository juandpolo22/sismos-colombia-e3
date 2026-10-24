"""Pruebas del árbol CART propio (src/arbol_cart.py), aprendiz débil del modelo B."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.arbol_cart import ArbolCART


def _datos(n=3000, p=6, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, p))
    X[:, 2] = rng.integers(0, 4, n)                      # variable discreta con muchos empates de valor
    y = np.sin(X[:, 0]) + 0.5 * X[:, 2] + rng.normal(0, 0.1, n)
    w = rng.uniform(0.1, 2.0, n)
    return X, y, w


def test_aprende_un_escalon_conocido():
    """Con y = 1{x0 > 0}, un árbol de profundidad 1 debe cortar en x0 ≈ 0 y predecir 0 y 1."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(2000, 3))
    y = (X[:, 0] > 0).astype(float)
    t = ArbolCART(max_depth=1, min_samples_leaf=5).fit(X, y)
    assert t.variable_[0] == 0 and abs(t.umbral_[0]) < 0.01
    assert np.allclose(np.sort(np.unique(t.predict(X))), [0.0, 1.0])


@pytest.mark.parametrize("prof,hoja", [(1, 1), (3, 50), (5, 10)])
def test_coincide_con_cart_de_referencia(prof, hoja):
    """Mismo criterio que CART de scikit-learn: mismas predicciones (hasta el redondeo)."""
    X, y, w = _datos()
    ref = DecisionTreeRegressor(max_depth=prof, min_samples_leaf=hoja, random_state=7).fit(X, y, sample_weight=w)
    propio = ArbolCART(prof, hoja, random_state=7).fit(X, y, w)
    assert np.abs(ref.predict(X) - propio.predict(X)).max() < 1e-9


def test_reproducible_con_la_misma_semilla():
    X, y, w = _datos()
    a = ArbolCART(4, 20, random_state=3).fit(X, y, w).predict(X)
    b = ArbolCART(4, 20, random_state=3).fit(X, y, w).predict(X)
    assert np.array_equal(a, b)


def test_respeta_min_samples_leaf_y_profundidad():
    X, y, w = _datos()
    t = ArbolCART(3, 100).fit(X, y, w)
    hojas, cuentas = np.unique(t.apply(X), return_counts=True)
    assert cuentas.min() >= 100 and len(hojas) <= 2 ** 3


def test_entradas_invalidas():
    X, y, w = _datos(200)
    with pytest.raises(ValueError):
        ArbolCART(3, 0)
    with pytest.raises(ValueError):
        ArbolCART().fit(X, y[:-1])
    Xn = X.copy(); Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        ArbolCART().fit(Xn, y)
    with pytest.raises(ValueError):
        ArbolCART().fit(X, y, -w)
    with pytest.raises(RuntimeError):
        ArbolCART().predict(X)
