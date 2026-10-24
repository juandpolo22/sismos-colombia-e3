"""Árbol de regresión CART ponderado, programado desde cero con numpy (aprendiz débil del modelo B).

Sustituye a ``sklearn.tree.DecisionTreeRegressor`` dentro de :class:`src.propuesto.BoostingZINB`, de modo que todo
el núcleo del algoritmo (construcción del árbol, regla de actualización y pérdida) es implementación propia.
Reproduce el criterio CART de Breiman et al. (1984) con búsqueda exacta de cortes.

Algoritmo (crecimiento en profundidad, búsqueda exacta)
------------------------------------------------------
Entradas: X (n × p), objetivo r (n), pesos w > 0 (n), profundidad máxima D, mínimo de filas por hoja m.

1. Ordenar una sola vez las filas por cada variable (n log n por variable).
2. Para un nodo con filas S, |S| ≥ 2m y profundidad < D, para cada variable j recorrer S en orden de x_j y,
   con sumas acumuladas W_L = Σ w_i y Y_L = Σ w_i r_i, evaluar cada corte entre dos valores distintos
   consecutivos que deje al menos m filas a cada lado:

       ganancia(j, k) = Y_L² / W_L + Y_R² / W_R            (ecuación 1)

   Maximizar (1) equivale a minimizar el error cuadrático ponderado de los dos hijos, porque
   Σ_S w (r − c_S)² = Σ_S w r² − Y_S² / W_S y el primer término no depende del corte.
3. Elegir el corte (j*, k*) de máxima ganancia; umbral = punto medio entre los dos valores. Si el nodo no admite
   cortes válidos o su impureza es ~0, es hoja. Empates: las variables se recorren en un orden aleatorio con semilla
   (Fisher-Yates con un generador xorshift) y, entre ganancias iguales salvo redondeo (tolerancia relativa 1e-10),
   gana la primera visitada; dentro de una variable, el primer corte. Es la convención de CART en scikit-learn.
4. Valor de cada hoja: c = Y_S / W_S (media ponderada; minimiza el error cuadrático ponderado).

Criterio de parada: profundidad D, menos de 2m filas, nodo puro o sin cortes válidos. Termina siempre: cada corte
deja al menos m ≥ 1 filas a cada lado, así que el número de nodos es finito (≤ 2^(D+1) − 1).

Complejidad: entrenamiento O(p · n log n) para el orden inicial + O(D · p · n) para las sumas acumuladas por nivel;
inferencia O(D) por fila.

Como en scikit-learn, X se representa en float32 para elegir y aplicar los umbrales (así ambos coinciden).
"""
from __future__ import annotations

import numpy as np

_UMBRAL_CONSTANTE = 1e-7          # dos valores se consideran distintos si difieren en más de esto
_EPS = np.finfo(np.float64).eps
_TOL_EMPATE = 1e-10           # tolerancia relativa para considerar dos ganancias iguales


class ArbolCART:
    """Árbol de regresión CART con pesos por fila.

    Parameters
    ----------
    max_depth : int
        Profundidad máxima D (la raíz tiene profundidad 0).
    min_samples_leaf : int
        Mínimo de filas m en cada hoja.

    Attributes
    ----------
    variable_, umbral_, izq_, der_, valor_ : np.ndarray
        Representación plana del árbol: para el nodo k, variable de corte (−1 si es hoja), umbral, hijos y valor.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 1, random_state: int = 0):
        if max_depth < 0 or min_samples_leaf < 1:
            raise ValueError("max_depth debe ser ≥ 0 y min_samples_leaf ≥ 1")
        self.max_depth, self.min_samples_leaf, self.random_state = int(max_depth), int(min_samples_leaf), random_state

    # ------------------------------------------------- generador para desempates
    @staticmethod
    def _xorshift(estado: int) -> tuple[int, int]:
        """Generador xorshift de 32 bits (Marsaglia, 2003). Devuelve (nuevo estado, entero en [0, 2^31))."""
        if estado == 0:
            estado = 1
        estado ^= (estado << 13) & 0xFFFFFFFF
        estado ^= estado >> 17
        estado ^= (estado << 5) & 0xFFFFFFFF
        return estado, estado % 0x80000000

    # ------------------------------------------------------------------ ajuste
    @staticmethod
    def ordenar(X: np.ndarray) -> np.ndarray:
        """Orden de las filas por cada variable (p × n). Se puede calcular una vez y reutilizar entre árboles."""
        Xf = np.asarray(X, dtype=np.float32)
        return np.argsort(Xf, axis=0, kind="stable").T.copy()

    def fit(self, X, y, sample_weight=None, orden: np.ndarray | None = None):
        """Ajusta el árbol.

        Parameters
        ----------
        X : array (n, p);  y : array (n,);  sample_weight : array (n,) de pesos > 0 (por defecto 1).
        orden : salida de :meth:`ordenar` para este X (opcional, acelera el boosting).
        """
        Xf = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if Xf.ndim != 2 or len(y) != len(Xf):
            raise ValueError("X debe ser 2D y tener tantas filas como y")
        if not np.isfinite(Xf).all() or not np.isfinite(y).all():
            raise ValueError("X e y no pueden tener NaN ni infinitos")
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("los pesos deben ser positivos")
        XT = Xf.T.astype(np.float64)                          # p × n, valores float32 representados en float64
        O = self.ordenar(Xf) if orden is None else orden
        self._var, self._umb, self._izq, self._der, self._val = [], [], [], [], []
        # Desempates: las variables se recorren en orden aleatorio (Fisher-Yates con semilla) y, ante ganancias
        # iguales, gana la primera visitada. Es la misma convención de CART en scikit-learn, con la que se verificó
        # que el árbol coincide exactamente.
        self._rng = int(np.random.RandomState(self.random_state).randint(0, 0x7FFFFFFF))
        self._feat = np.arange(XT.shape[0])
        self._const = np.zeros(XT.shape[0], int)
        self._crecer(XT, y, w, O, np.ones(len(y), bool), 0, 0)
        self.variable_ = np.array(self._var, int)
        self.umbral_ = np.array(self._umb, float)
        self.izq_ = np.array(self._izq, int)
        self.der_ = np.array(self._der, int)
        self.valor_ = np.array(self._val, float)
        del self._var, self._umb, self._izq, self._der, self._val, self._rng, self._feat, self._const
        return self

    def _nuevo(self, valor):
        self._var.append(-1); self._umb.append(np.nan); self._izq.append(-1); self._der.append(-1); self._val.append(valor)
        return len(self._var) - 1

    def _crecer(self, XT, y, w, O, mask, prof, n_const):
        Wn = w[mask].sum()
        Yn = (w[mask] * y[mask]).sum()
        nodo = self._nuevo(Yn / Wn)
        n = int(mask.sum())
        m = self.min_samples_leaf
        impureza = (w[mask] * y[mask] ** 2).sum() / Wn - (Yn / Wn) ** 2
        if prof >= self.max_depth or n < 2 * m or n < 2 or impureza <= _EPS:
            return nodo
        p = XT.shape[0]
        S = O[mask[O]].reshape(p, n)                          # filas del nodo ordenadas por cada variable
        xs = np.take_along_axis(XT, S, axis=1)
        ws, ys = w[S], y[S]
        WL = np.cumsum(ws, axis=1)[:, :-1]
        YL = np.cumsum(ws * ys, axis=1)[:, :-1]
        WR, YR = Wn - WL, Yn - YL
        pos = np.arange(1, n)                                 # filas a la izquierda si se corta tras la posición i
        valido = (xs[:, 1:] > xs[:, :-1] + _UMBRAL_CONSTANTE) & (pos >= m) & (n - pos >= m)
        with np.errstate(divide="ignore", invalid="ignore"):
            gan = np.where(valido, YL ** 2 / WL + YR ** 2 / WR, -np.inf)   # ecuación (1)
        # Ganancias que difieren en menos de una tolerancia relativa (ruido de redondeo) se tratan como empates:
        # así la elección no depende del orden en que se suman los términos.
        g_max = gan.max(axis=1)
        k_mejor = np.argmax(gan >= (g_max - _TOL_EMPATE * np.abs(g_max))[:, None], axis=1)   # primer casi-máximo
        g_mejor = g_max
        constante = xs[:, -1] <= xs[:, 0] + _UMBRAL_CONSTANTE
        # recorrido de variables en orden aleatorio (Fisher-Yates), descartando las constantes en el nodo
        F, C = self._feat, self._const
        f_i, n_known, n_total, n_found, n_drawn, visit = p, n_const, n_const, 0, 0, 0
        j, mejor = -1, -np.inf
        while f_i > n_total and (visit < p or visit <= n_found + n_drawn):
            visit += 1
            self._rng, r = self._xorshift(self._rng)
            f_j = n_drawn + r % (f_i - n_found - n_drawn)
            if f_j < n_known:
                F[n_drawn], F[f_j] = F[f_j], F[n_drawn]
                n_drawn += 1
                continue
            f_j += n_found
            v = F[f_j]
            if constante[v]:
                F[f_j], F[n_total] = F[n_total], F[f_j]
                n_found += 1; n_total += 1
                continue
            f_i -= 1
            F[f_i], F[f_j] = F[f_j], F[f_i]
            if g_mejor[v] > (mejor + _TOL_EMPATE * abs(mejor) if np.isfinite(mejor) else mejor):
                j, mejor = v, g_mejor[v]
        F[:n_known] = C[:n_known]
        C[n_known:n_known + n_found] = F[n_known:n_known + n_found]
        if j < 0 or not np.isfinite(mejor):
            return nodo
        k = k_mejor[j]
        umbral = xs[j, k] / 2.0 + xs[j, k + 1] / 2.0
        if umbral == np.inf or umbral == xs[j, k + 1]:
            umbral = xs[j, k]
        izq = mask & (XT[j] <= umbral)
        der = mask & ~izq
        self._var[nodo], self._umb[nodo] = int(j), float(umbral)
        self._izq[nodo] = self._crecer(XT, y, w, O, izq, prof + 1, n_total)
        self._der[nodo] = self._crecer(XT, y, w, O, der, prof + 1, n_total)
        return nodo

    # ------------------------------------------------------------- inferencia
    def apply(self, X) -> np.ndarray:
        """Índice de la hoja de cada fila."""
        if not hasattr(self, "valor_"):
            raise RuntimeError("el árbol no está ajustado: llame a fit primero")
        Xf = np.asarray(X, dtype=np.float32).astype(np.float64)
        nodo = np.zeros(len(Xf), int)
        for _ in range(self.max_depth + 1):
            v = self.variable_[nodo]
            interno = v >= 0
            if not interno.any():
                break
            idx = np.where(interno)[0]
            va = Xf[idx, v[idx]] <= self.umbral_[nodo[idx]]
            nodo[idx] = np.where(va, self.izq_[nodo[idx]], self.der_[nodo[idx]])
        return nodo

    def predict(self, X) -> np.ndarray:
        """Valor de la hoja de cada fila."""
        hojas = self.apply(X)
        return self.valor_[hojas]
