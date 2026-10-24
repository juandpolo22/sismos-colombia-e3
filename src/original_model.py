"""Modelo original A+B como estimador compatible con scikit-learn.

``ModeloOriginalAB`` empaqueta, en una sola clase con ``fit``/``predict``, el procedimiento completo que se evaluó en
el origen móvil 2021–2026 (``scripts/origen_movil_final.py``):

1. **Miembros.** Modelos benchmark optimizados (hiperparámetros fijados con datos anteriores a 2016), el núcleo
   ZINB propio, el modelo B (boosting de la verosimilitud ZINB sobre árboles CART propios), persistencia e ingenuo
   estacional. Todos producen pronósticos en escala log1p.
2. **Predicciones fuera de pliegue cronológicas.** Los meses de entrenamiento se dividen en 5 bloques contiguos;
   cada bloque f ≥ 1 se predice con modelos entrenados solo con meses anteriores (gap de 1 mes).
3. **Combinador A.** Pesos softmax w(z) = softmax(zᵀΘ) que dependen de meta-características z de cada celda-mes;
   Θ minimiza el error cuadrático en log1p con penalización L2 (λ elegido con el 20 % final de los meses).
4. **Calibración final.** Recta log1p(y) = c0 + c1·ŷ estimada con el 20 % final de los meses (Mincer-Zarnowitz);
   pronóstico = expm1(max(0, c0 + c1·ŷ)).

Todo se estima solo con las filas que recibe ``fit``. ``predict`` no reentrena nada.

Columnas de entrada (``X`` es un DataFrame)
------------------------------------------
* las 18 variables de :data:`src.datos.VARIABLES_BASE`;
* las 13 variables de excitación de :func:`src.datos.variables_excitacion` (``exc_*`` y ``media_expansiva``);
* ``h0`` … ``h11``: sismos de la celda en los meses t, t−1, …, t−11 (historia para el contexto de A);
* ``mes``: índice del mes objetivo (solo se usa para ordenar en el tiempo; no es predictor).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from . import benchmark as bm
from . import config as cfg
from .datos import VARIABLES_BASE
from .propuesto import BoostingZINB, CombinadorSoftmax, ZINB, pliegues_cronologicos

EXCITACION = [f"exc_a{a}_s{s}_r{r}" for a in (0.5, 0.9) for s in (15, 30, 60) for r in (0.4, 0.8)] + ["media_expansiva"]
HISTORIA = [f"h{j}" for j in range(12)]
COLUMNAS = VARIABLES_BASE + EXCITACION + HISTORIA + ["mes"]
L2_REJILLA = (1e-3, 1e-2, 1e-1, 1.0)

PSEUDOCODIGO = """Algoritmo A+B v2  (entrada: panel de entrenamiento {(x_i, z_i, y_i)} ordenado por mes; salida: ŷ para filas nuevas)
 1. Para cada miembro j (10 benchmarks optimizados, núcleo ZINB, B, persistencia, estacional):
      dividir los meses en 5 bloques contiguos; para f = 1..4 entrenar j con los meses < inicio del bloque f − 1
      y predecir el bloque f  →  f_j(x_i) en escala log1p (predicciones fuera de pliegue cronológicas)
 2. Modelo B (boosting ZINB):  η ← log ȳ, ξ ← −0,8, α ← 1
      repetir m = 1..M:  g, h ← gradiente y hessiano de la NLL ZINB respecto de η
                         ajustar árbol CART propio a −g/h con pesos h;  η ← η + ν·árbol(x)
                         ídem para ξ (logit de cero estructural);  cada 20 iteraciones re-estimar α
      M = argmin de la NLL en el 20 % final de los meses (parada temprana cronológica)
      salida: E[log(1+Y)] bajo ZINB(μ = e^η, π = σ(ξ), α)
 3. Combinador A:  w(z) = softmax(zᵀΘ);  Θ = argmin Σ (Σ_j w_j(z_i) f_j(x_i) − log(1+y_i))² + λ‖Θ‖²   (L-BFGS)
      λ ∈ {0,001; 0,01; 0,1; 1} elegido con el 20 % final de los meses
 4. Calibración:  (c0, c1) = MCO de log(1+y) sobre ŷ_A en el 20 % final  (Mincer-Zarnowitz)
 5. Predicción:  ŷ = expm1( max(0, c0 + c1 · Σ_j w_j(z) f_j(x)) )
 Parada: M iteraciones de B (parada temprana), L-BFGS hasta tolerancia; cada árbol se detiene por profundidad 3,
 mínimo 50 filas por hoja o ausencia de cortes válidos."""


def miembros_por_defecto(dir_hp: Path | None = None) -> list:
    """Miembros benchmark de A+B: las 4 versiones de XGBoost y de Random Forest y la mejor de KNN y de Ridge.

    Los hiperparámetros son los fijados con datos anteriores a 2016 (``resultados/origen_movil/hp_bench``).
    Devuelve una lista de tuplas ``(modelo, optimizador, params)``.
    """
    dir_hp = Path(dir_hp) if dir_hp else cfg.RESULTADOS / "origen_movil" / "hp_bench"
    leer = lambda m, o: json.load(open(dir_hp / f"{bm_slug(m, o)}.json"))  # noqa: E731
    mejor = lambda m: min(bm.OPTIMIZADORES, key=lambda o: leer(m, o)["RMSLE_CV"])  # noqa: E731
    pares = [(m, o) for m in ("XGBoost", "Random Forest") for o in bm.OPTIMIZADORES] + [("KNN", mejor("KNN")), ("Ridge", mejor("Ridge"))]
    return [(m, o, leer(m, o)["params"]) for m, o in pares]


def bm_slug(m, o):
    return f"{m}__{o}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


class ModeloOriginalAB(RegressorMixin, BaseEstimator):
    """Modelo original A+B: combinador contextual (A) de miembros que incluye un boosting ZINB propio (B).

    Parameters
    ----------
    miembros : list of (modelo, optimizador, params) or None
        Modelos benchmark que entran como miembros. ``None`` usa :func:`miembros_por_defecto`.
    lr_b, profundidad_b, hoja_b, iter_max_b : float, int, int, int
        Tasa de aprendizaje, profundidad, mínimo de filas por hoja y máximo de iteraciones del modelo B.
    n_bloques : int
        Bloques cronológicos para las predicciones fuera de pliegue.
    calibrar : bool
        Si se aplica la calibración final de Mincer-Zarnowitz.
    usar_b, usar_zinb : bool
        Permiten la ablación (quitar B o el núcleo ZINB).
    n_jobs : int
        Núcleos para los miembros benchmark.
    random_state : int
        Semilla global.

    Attributes
    ----------
    l2_ : float                 penalización elegida para el combinador
    c0_, c1_ : float            coeficientes de la calibración
    nombres_ : list of str      nombre de cada miembro (columnas de la matriz de pronósticos)
    """

    def __init__(self, miembros=None, lr_b=0.08, profundidad_b=3, hoja_b=50, iter_max_b=300, n_bloques=5,
                 calibrar=True, usar_b=True, usar_zinb=True, n_jobs=1, random_state=cfg.SEMILLA):
        self.miembros = miembros
        self.lr_b = lr_b
        self.profundidad_b = profundidad_b
        self.hoja_b = hoja_b
        self.iter_max_b = iter_max_b
        self.n_bloques = n_bloques
        self.calibrar = calibrar
        self.usar_b = usar_b
        self.usar_zinb = usar_zinb
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _validar(X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame con las columnas de src.original_model.COLUMNAS")
        faltan = [c for c in COLUMNAS if c not in X.columns]
        if faltan:
            raise ValueError(f"faltan columnas en X: {faltan[:5]}{' …' if len(faltan) > 5 else ''}")
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim != 1 or len(y) != len(X):
                raise ValueError("y debe ser un vector con una fila por fila de X")
            if not np.isfinite(y).all() or (y < 0).any():
                raise ValueError("y debe contener conteos finitos y no negativos")
        return y

    def _matriz_b(self, X):
        """Matriz del modelo B: 18 variables base + 13 de excitación, faltantes como −1."""
        return np.nan_to_num(X[VARIABLES_BASE + EXCITACION].to_numpy(float), nan=-1.0)

    def _diseno_zinb(self, X):
        """Matrices de diseño del núcleo ZINB (media y ceros) estandarizadas con el entrenamiento."""
        c = lambda n: np.log1p(X[n].to_numpy(float))  # noqa: E731
        Zm = np.column_stack([c("n_1m"), c("n_3m"), c("n_12m"), c("n35_12m"), c("vec_1m"),
                              X["exc_a0.5_s30_r0.4"].to_numpy(), X["exc_a0.9_s30_r0.8"].to_numpy(),
                              X["exc_a0.5_s15_r0.4"].to_numpy(), X["exc_a0.9_s60_r0.8"].to_numpy(),
                              X["sen_mes"].to_numpy(), X["cos_mes"].to_numpy()])
        Wz = np.column_stack([c("n_12m"), c("n_3m"), X["exc_a0.5_s30_r0.4"].to_numpy()])
        if not hasattr(self, "_est_zinb"):
            self._est_zinb = (Zm.mean(0), Zm.std(0) + 1e-9, Wz.mean(0), Wz.std(0) + 1e-9)
        mu, sd, mw, sw = self._est_zinb
        uno = np.ones((len(X), 1))
        return np.hstack([uno, (Zm - mu) / sd]), np.hstack([uno, (Wz - mw) / sw])

    def _contexto(self, X):
        """Meta-características z de A (estandarizadas con el entrenamiento) con la constante."""
        hist = X[HISTORIA].to_numpy(float)
        lg = lambda n: np.log1p(X[n].to_numpy(float))  # noqa: E731
        ctx = self.pca_.transform(np.log1p(hist))
        Zr = np.column_stack([lg("n_1m"), lg("n_3m"), lg("n_12m"), lg("n35_12m"), lg("vec_1m"),
                              X["exc_a0.5_s30_r0.4"].to_numpy(), X["exc_a0.9_s60_r0.8"].to_numpy(), ctx, (hist == 0).mean(1),
                              np.nan_to_num(X["mag_max_12m"].to_numpy(float), nan=cfg.M_MIN), X["sen_mes"].to_numpy(),
                              X["cos_mes"].to_numpy()])
        if not hasattr(self, "_est_z"):
            self._est_z = (Zr.mean(0), Zr.std(0) + 1e-9)
        return np.hstack([np.ones((len(Zr), 1)), (Zr - self._est_z[0]) / self._est_z[1]])

    def _base(self, m, params):
        return bm.construir("reg", m, params, VARIABLES_BASE, semilla=self.random_state, n_jobs=self.n_jobs)

    # ------------------------------------------------------------------ ajuste
    def fit(self, X, y):
        """Ajusta miembros, B, el combinador A y la calibración con las filas de ``X``.

        Parameters
        ----------
        X : DataFrame con :data:`COLUMNAS`;  y : conteos del mes objetivo (≥ 0).
        """
        y = self._validar(X, y)
        X = X.reset_index(drop=True)
        cfg.fijar_semillas(self.random_state)
        for a in ("_est_zinb", "_est_z"):
            if hasattr(self, a):
                delattr(self, a)
        tm = X["mes"].to_numpy()
        ylog = np.log1p(y)
        PL, MASK = pliegues_cronologicos(tm, k=self.n_bloques, gap=cfg.GAP_MESES)
        FILAS = np.where(MASK)[0]
        if len(FILAS) < 50:
            raise ValueError("muy pocos meses para las predicciones fuera de pliegue cronológicas")
        miembros = self.miembros if self.miembros is not None else miembros_por_defecto()
        Xb = X[VARIABLES_BASE]
        cols, self.miembros_, self.nombres_ = [], [], []
        for m, o, par in miembros:
            oof = np.full(len(X), np.nan)
            for a, b in PL:
                oof[b] = np.log1p(np.clip(self._base(m, par).fit(Xb.iloc[a], y[a]).predict(Xb.iloc[b]), 0, None))
            cols.append(oof)
            self.miembros_.append(self._base(m, par).fit(Xb, y))
            self.nombres_.append(f"{m} ({o})")
        if self.usar_zinb:
            Xm, W = self._diseno_zinb(X)
            oof = np.full(len(X), np.nan)
            for a, b in PL:
                oof[b], _ = ZINB().fit(Xm[a], W[a], y[a]).predecir(Xm[b], W[b])
            cols.append(oof)
            self.zinb_ = ZINB().fit(Xm, W, y)
            self.nombres_.append("Núcleo ZINB (propio)")
        if self.usar_b:
            XB = self._matriz_b(X)
            oof = np.full(len(X), np.nan)
            par_b = dict(n_max=self.iter_max_b, lr=self.lr_b, depth=self.profundidad_b, min_leaf=self.hoja_b, seed=self.random_state)
            for a, b in PL:
                oof[b], _ = BoostingZINB(**par_b).fit(XB[a], y[a], tm[a]).predecir(XB[b])
            cols.append(oof)
            self.b_ = BoostingZINB(**par_b).fit(XB, y, tm)
            self.nombres_.append("B: boosting ZINB (propio)")
        cols += [np.log1p(X["h0"].to_numpy(float)), np.log1p(X["h11"].to_numpy(float))]
        self.nombres_ += ["Persistencia", "Ingenuo estacional"]
        B = np.column_stack(cols)
        self.pca_ = PCA(n_components=4, random_state=self.random_state).fit(np.log1p(X[HISTORIA].to_numpy(float)))
        Z = self._contexto(X)
        # λ del combinador con el 20 % final de los meses con predicción fuera de pliegue
        meses_m = np.unique(tm[FILAS])
        corte = meses_m[int(len(meses_m) * 0.8)]
        a, b = FILAS[tm[FILAS] < corte], FILAS[tm[FILAS] >= corte]
        err = {l2: np.mean((CombinadorSoftmax(l2).fit(Z[a], B[a], ylog[a]).predict(Z[b], B[b]) - ylog[b]) ** 2) for l2 in L2_REJILLA}
        self.l2_ = min(err, key=err.get)
        self.combinador_ = CombinadorSoftmax(self.l2_).fit(Z[FILAS], B[FILAS], ylog[FILAS])
        if self.calibrar:
            pv = CombinadorSoftmax(self.l2_).fit(Z[a], B[a], ylog[a]).predict(Z[b], B[b])
            self.c0_, self.c1_ = np.linalg.lstsq(np.column_stack([np.ones(len(b)), pv]), ylog[b], rcond=None)[0]
        else:
            self.c0_, self.c1_ = 0.0, 1.0
        self.n_features_in_ = len(COLUMNAS)
        return self

    # ------------------------------------------------------------------ inferencia
    def pronosticos_miembros(self, X) -> np.ndarray:
        """Matriz n × K de pronósticos de cada miembro en escala log1p."""
        check_is_fitted(self, "combinador_")
        self._validar(X)
        X = X.reset_index(drop=True)
        cols = [np.log1p(np.clip(mod.predict(X[VARIABLES_BASE]), 0, None)) for mod in self.miembros_]
        if self.usar_zinb:
            Xm, W = self._diseno_zinb(X)
            cols.append(self.zinb_.predecir(Xm, W)[0])
        if self.usar_b:
            cols.append(self.b_.predecir(self._matriz_b(X))[0])
        cols += [np.log1p(X["h0"].to_numpy(float)), np.log1p(X["h11"].to_numpy(float))]
        return np.column_stack(cols)

    def pesos(self, X) -> np.ndarray:
        """Pesos w_j(z) del combinador para cada fila (suman 1)."""
        check_is_fitted(self, "combinador_")
        return self.combinador_.pesos(self._contexto(X.reset_index(drop=True)))

    def predict_log(self, X) -> np.ndarray:
        """Pronóstico calibrado en escala log1p."""
        B = self.pronosticos_miembros(X)
        plog = self.combinador_.predict(self._contexto(X.reset_index(drop=True)), B)
        return np.clip(self.c0_ + self.c1_ * plog, 0, None)

    def predict(self, X) -> np.ndarray:
        """Número esperado de sismos M ≥ 2,5 en el mes objetivo (≥ 0)."""
        return np.expm1(self.predict_log(X))

    def explicar(self, X) -> pd.DataFrame:
        """Aporte de cada miembro al pronóstico log1p de cada fila: c1 · w_j(z) · f_j(x) (más c0 en «calibración»).

        La suma de las columnas reproduce ``predict_log`` (salvo el recorte en 0).
        """
        B = self.pronosticos_miembros(X)
        W = self.pesos(X)
        A = pd.DataFrame(self.c1_ * W * B, columns=self.nombres_)
        A["calibración (c0)"] = self.c0_
        return A
