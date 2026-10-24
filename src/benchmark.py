"""Diseño experimental del Entregable 2: modelos benchmark × balanceo × optimizadores con validación
cruzada anidada temporal.

Estructura
----------
* :data:`ESPACIOS` — espacio de búsqueda de cada modelo (tipo, rango y escala de cada hiperparámetro).
* :data:`REJILLAS` — rejilla explícita de Grid Search (como mucho ``presupuesto`` puntos).
* :func:`construir` — Pipeline completo (preprocesamiento + balanceo + modelo) para unos hiperparámetros.
* :class:`Optimizador` — Grid, Random, Bayesiano (Optuna/TPE) y Genético (DEAP) con el **mismo
  presupuesto** de evaluaciones; cada evaluación es una validación cruzada temporal interna en la que el
  Pipeline se ajusta solo con la parte de ajuste del pliegue interno (sin fuga).
* :func:`correr_combinacion` — validación anidada (bucle externo temporal) + reoptimización con todo el
  entrenamiento + predicción de la prueba; registra cada evaluación, tiempos y errores.

Todas las funciones reciben la semilla explícitamente.
"""
from __future__ import annotations

import json
import math
import time
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from . import config as cfg
from . import modelos as mdl

warnings.filterwarnings("ignore")

MODELOS_REG = ["KNN", "Ridge", "Lasso", "Árbol de decisión", "Random Forest", "XGBoost", "SVR"]
MODELOS_CLF = ["KNN", "Naive Bayes", "Regresión logística (L1/L2)", "Árbol de decisión", "Random Forest", "XGBoost", "SVM"]
BALANCEOS = ["sin balanceo", "SMOTE", "ADASYN", "class_weight"]
OPTIMIZADORES = ["Grid", "Random", "Bayesiano (TPE)", "Genético (DEAP)"]
N_MAX_KERNEL = 4000          # submuestra para SVR/SVC con núcleo RBF (coste O(n²)–O(n³); ver cap. de optimización computacional)


# ---------------------------------------------------------------------------
# Envolturas propias
# ---------------------------------------------------------------------------
class Submuestra(BaseEstimator):
    """Ajusta el estimador interno con una submuestra aleatoria de ``n_max`` filas (para métodos de núcleo).

    En clasificación la submuestra es estratificada por clase.
    """

    def __init__(self, estimador=None, n_max: int = N_MAX_KERNEL, semilla: int = cfg.SEMILLA):
        self.estimador, self.n_max, self.semilla = estimador, n_max, semilla

    def fit(self, X, y, **kw):
        """Ajusta el estimador interno sobre una submuestra de a lo sumo ``n_max`` filas.

        En clasificación la submuestra es estratificada (conserva la proporción de cada clase).

        Parameters
        ----------
        X : array-like de forma (n, p)
            Variables predictoras ya preprocesadas.
        y : array-like de forma (n,)
            Objetivo.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        rng = np.random.default_rng(self.semilla)
        if len(y) > self.n_max:
            if getattr(self.estimador, "_estimator_type", "") == "classifier":
                idx = np.concatenate([rng.choice(np.where(y == c)[0], max(1, int(round(self.n_max * np.mean(y == c)))), replace=False)
                                      for c in np.unique(y)])
            else:
                idx = rng.choice(len(y), self.n_max, replace=False)
            X, y = X[idx], y[idx]
        self.estimador_ = clone(self.estimador).fit(X, y)
        if hasattr(self.estimador_, "classes_"):
            self.classes_ = self.estimador_.classes_
        return self

    def predict(self, X):
        """Predicción del estimador ajustado sobre la submuestra."""
        return self.estimador_.predict(X)

    def decision_function(self, X):
        """Función de decisión del estimador interno (SVM sin probabilidades)."""
        return self.estimador_.decision_function(X)

    def predict_proba(self, X):
        """Probabilidades del estimador interno, si las ofrece."""
        return self.estimador_.predict_proba(X)


class SubmuestraReg(Submuestra, RegressorMixin):
    """Envoltura de :class:`Submuestra` para regresores (SVR con núcleo RBF)."""
    _estimator_type = "regressor"


class SubmuestraClf(Submuestra, ClassifierMixin):
    """Envoltura de :class:`Submuestra` para clasificadores (SVM con núcleo RBF)."""
    _estimator_type = "classifier"


class CorreccionPrior(BaseEstimator, ClassifierMixin):
    """Equivalente de ``class_weight='balanced'`` para clasificadores sin ese parámetro (KNN).

    Divide la probabilidad estimada de cada clase por su prevalencia en entrenamiento y renormaliza:
    equivale a decidir con costos inversamente proporcionales a la frecuencia de la clase (regla de
    Bayes con pérdida balanceada).
    """

    def __init__(self, estimador=None):
        self.estimador = estimador

    def fit(self, X, y):
        """Ajusta el clasificador y guarda la frecuencia de cada clase en entrenamiento.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.estimador_ = clone(self.estimador).fit(X, y)
        self.classes_ = self.estimador_.classes_
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict_proba(self, X):
        """Probabilidades divididas por el prior de entrenamiento y renormalizadas (prior uniforme)."""
        p = self.estimador_.predict_proba(X) / self.prior_
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        """Clase con mayor probabilidad corregida."""
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class XGBBalanceado(XGBClassifier):
    """XGBoost con ``scale_pos_weight = negativos / positivos`` calculado en cada ajuste (class_weight)."""

    def fit(self, X, y, **kw):
        """Ajusta XGBoost con ``scale_pos_weight = n_negativos / n_positivos`` (equivalente a class_weight).

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kw)


# ---------------------------------------------------------------------------
# Espacios de búsqueda (justificados en el capítulo 13)
# ---------------------------------------------------------------------------
# (nombre, tipo, mínimo, máximo, escala_log) o (nombre, "cat", opciones)
ESPACIOS = {
    ("reg", "KNN"): [("n_neighbors", "int", 5, 200, True), ("weights", "cat", ["uniform", "distance"])],
    ("reg", "Ridge"): [("alpha", "float", 1e-3, 1e4, True)],
    ("reg", "Lasso"): [("alpha", "float", 1e-5, 1e-1, True)],
    ("reg", "Árbol de decisión"): [("max_depth", "int", 2, 20, False), ("min_samples_leaf", "int", 5, 500, True)],
    ("reg", "Random Forest"): [("n_estimators", "int", 50, 250, False), ("max_depth", "int", 4, 20, False),
                               ("min_samples_leaf", "int", 1, 50, True), ("max_features", "float", 0.2, 1.0, False)],
    ("reg", "XGBoost"): [("n_estimators", "int", 100, 600, False), ("max_depth", "int", 2, 8, False),
                         ("learning_rate", "float", 0.01, 0.3, True), ("subsample", "float", 0.5, 1.0, False),
                         ("colsample_bytree", "float", 0.5, 1.0, False), ("min_child_weight", "float", 1, 50, True)],
    ("reg", "SVR"): [("C", "float", 0.01, 30, True), ("epsilon", "float", 0.01, 0.5, True), ("gamma", "float", 1e-3, 1, True)],
    # Miembros adicionales del modelo propuesto (no forman parte de los 7 modelos benchmark del enunciado)
    ("reg", "LightGBM"): [("n_estimators", "int", 100, 600, False), ("num_leaves", "int", 8, 128, True),
                          ("learning_rate", "float", 0.01, 0.3, True), ("subsample", "float", 0.5, 1.0, False),
                          ("colsample_bytree", "float", 0.5, 1.0, False), ("min_child_samples", "int", 5, 200, True)],
    ("reg", "CatBoost"): [("iterations", "int", 200, 800, False), ("depth", "int", 3, 8, False),
                          ("learning_rate", "float", 0.01, 0.3, True), ("l2_leaf_reg", "float", 1, 30, True)],
    ("clf", "KNN"): [("n_neighbors", "int", 5, 200, True), ("weights", "cat", ["uniform", "distance"])],
    ("clf", "Naive Bayes"): [("var_smoothing", "float", 1e-12, 1e-1, True)],
    ("clf", "Regresión logística (L1/L2)"): [("C", "float", 1e-3, 1e2, True), ("penalty", "cat", ["l1", "l2"])],
    ("clf", "Árbol de decisión"): [("max_depth", "int", 2, 20, False), ("min_samples_leaf", "int", 5, 500, True)],
    ("clf", "Random Forest"): [("n_estimators", "int", 50, 250, False), ("max_depth", "int", 4, 20, False),
                               ("min_samples_leaf", "int", 1, 50, True), ("max_features", "float", 0.2, 1.0, False)],
    ("clf", "XGBoost"): [("n_estimators", "int", 100, 600, False), ("max_depth", "int", 2, 8, False),
                         ("learning_rate", "float", 0.01, 0.3, True), ("subsample", "float", 0.5, 1.0, False),
                         ("colsample_bytree", "float", 0.5, 1.0, False), ("min_child_weight", "float", 1, 50, True)],
    ("clf", "SVM"): [("C", "float", 0.01, 30, True), ("gamma", "float", 1e-3, 1, True)],
}


def _niveles(p, k):
    if p[1] == "cat":
        return list(p[2])[:k]
    lo, hi, log = p[2], p[3], p[4]
    v = np.geomspace(lo, hi, k) if log else np.linspace(lo, hi, k)
    return [int(round(x)) for x in v] if p[1] == "int" else [float(x) for x in v]


def rejilla(tarea: str, modelo: str, presupuesto: int) -> list:
    """Rejilla de Grid Search con como mucho ``presupuesto`` puntos.

    Reparte los niveles entre hiperparámetros (más niveles a los primeros, que son los más influyentes).
    Cuando hay más hiperparámetros de los que caben, los últimos quedan fijos en el centro de su rango:
    es la limitación clásica de Grid Search en alta dimensión que el capítulo 13 discute.
    """
    esp = ESPACIOS[(tarea, modelo)]
    d = len(esp)
    k = [1] * d
    i = 0
    while True:
        prueba = k.copy()
        prueba[i % d] += 1
        if p_cat := (esp[i % d][1] == "cat" and prueba[i % d] > len(esp[i % d][2])):
            i += 1
            if all((e[1] == "cat" and kk >= len(e[2])) for e, kk in zip(esp, k)):
                break
            continue
        if math.prod(prueba) > presupuesto:
            break
        k = prueba
        i += 1
    listas = []
    for p, kk in zip(esp, k):
        if kk == 1:
            listas.append([_centro(p)])
        else:
            listas.append(_niveles(p, kk))
    puntos = [dict(zip([p[0] for p in esp], c)) for c in _producto(listas)]
    return puntos[:presupuesto]


def _centro(p):
    if p[1] == "cat":
        return p[2][0]
    v = math.sqrt(p[2] * p[3]) if p[4] else (p[2] + p[3]) / 2
    return int(round(v)) if p[1] == "int" else float(v)


def _producto(listas):
    out = [[]]
    for l in listas:
        out = [o + [x] for o in out for x in l]
    return out


def desde_unitario(tarea, modelo, u) -> dict:
    """Convierte un vector en [0, 1]^d (genes del AG o muestra aleatoria) en hiperparámetros."""
    params = {}
    for p, ui in zip(ESPACIOS[(tarea, modelo)], u):
        ui = float(np.clip(ui, 0, 1 - 1e-12))
        if p[1] == "cat":
            params[p[0]] = p[2][int(ui * len(p[2]))]
        else:
            lo, hi = p[2], p[3]
            v = math.exp(math.log(lo) + ui * (math.log(hi) - math.log(lo))) if p[4] else lo + ui * (hi - lo)
            params[p[0]] = int(round(v)) if p[1] == "int" else float(v)
    return params


# ---------------------------------------------------------------------------
# Construcción del Pipeline completo
# ---------------------------------------------------------------------------
def construir(tarea: str, modelo: str, params: dict, columnas, balanceo: str = "sin balanceo",
              semilla: int = cfg.SEMILLA, n_jobs: int = 1):
    """Pipeline completo: preprocesamiento (cap. 10) [+ balanceo] + modelo con los hiperparámetros dados."""
    prep = mdl.preprocesador(columnas, "constante")
    p = dict(params)
    if tarea == "reg":
        est = {
            "KNN": lambda: KNeighborsRegressor(n_jobs=n_jobs, **p),
            "Ridge": lambda: Ridge(random_state=semilla, **p),
            "Lasso": lambda: Lasso(max_iter=5000, random_state=semilla, **p),
            "Árbol de decisión": lambda: DecisionTreeRegressor(random_state=semilla, **p),
            "Random Forest": lambda: RandomForestRegressor(random_state=semilla, n_jobs=n_jobs, max_samples=0.5, **p),
            "XGBoost": lambda: XGBRegressor(tree_method="hist", random_state=semilla, n_jobs=n_jobs, verbosity=0, **p),
            "SVR": lambda: SubmuestraReg(SVR(kernel="rbf", **p), semilla=semilla),
            "LightGBM": lambda: __import__("lightgbm").LGBMRegressor(subsample_freq=1, random_state=semilla, n_jobs=n_jobs, verbose=-1, **p),
            "CatBoost": lambda: __import__("catboost").CatBoostRegressor(random_seed=semilla, thread_count=max(n_jobs, 1) if n_jobs > 0 else -1, verbose=0, **p),
        }[modelo]()
        return mdl.con_objetivo_log(Pipeline([("prep", prep), ("modelo", est)]))
    cw = "balanced" if balanceo == "class_weight" else None
    est = {
        "KNN": lambda: KNeighborsClassifier(n_jobs=n_jobs, **p),
        "Naive Bayes": lambda: GaussianNB(priors=[0.5, 0.5] if cw else None, **p),
        "Regresión logística (L1/L2)": lambda: LogisticRegression(solver="saga", max_iter=3000, class_weight=cw, random_state=semilla, **p),
        "Árbol de decisión": lambda: DecisionTreeClassifier(class_weight=cw, random_state=semilla, **p),
        "Random Forest": lambda: RandomForestClassifier(class_weight=cw, random_state=semilla, n_jobs=n_jobs, max_samples=0.5, **p),
        "XGBoost": lambda: (XGBBalanceado if cw else XGBClassifier)(tree_method="hist", random_state=semilla, n_jobs=n_jobs,
                                                                    verbosity=0, eval_metric="logloss", **p),
        "SVM": lambda: SubmuestraClf(SVC(kernel="rbf", class_weight=cw, random_state=semilla, **p), semilla=semilla),
    }[modelo]()
    if modelo == "KNN" and cw:
        est = CorreccionPrior(est)
    pasos = [("prep", prep)]
    if balanceo == "SMOTE":
        from imblearn.over_sampling import SMOTE
        pasos.append(("balanceo", SMOTE(random_state=semilla)))
    elif balanceo == "ADASYN":
        from imblearn.over_sampling import ADASYN
        pasos.append(("balanceo", ADASYN(random_state=semilla)))
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline(pasos + [("modelo", est)])


def puntaje_clf(modelo, X) -> np.ndarray:
    """Puntaje continuo para AUC: probabilidad si existe; si no, función de decisión."""
    if hasattr(modelo, "predict_proba"):
        try:
            return modelo.predict_proba(X)[:, 1]
        except Exception:  # noqa: BLE001  (SVC sin probability=True)
            pass
    return modelo.decision_function(X)


# ---------------------------------------------------------------------------
# Optimizadores con el mismo presupuesto
# ---------------------------------------------------------------------------
@dataclass
class Registro:
    """Historial de evaluaciones de una búsqueda (para curvas anytime y presupuesto)."""
    params: list = field(default_factory=list)
    valores: list = field(default_factory=list)
    tiempos: list = field(default_factory=list)
    diversidad: list = field(default_factory=list)


class Optimizador:
    """Búsqueda de hiperparámetros con presupuesto fijo de evaluaciones.

    Parameters
    ----------
    metodo : str
        Uno de :data:`OPTIMIZADORES`.
    objetivo : callable
        ``objetivo(params) -> float`` (se **maximiza**; en regresión se pasa −RMSLE).
    tarea, modelo : str
        Para el espacio de búsqueda.
    presupuesto : int
        Número máximo de evaluaciones del objetivo.
    """

    def __init__(self, metodo, objetivo, tarea, modelo, presupuesto=15, semilla=cfg.SEMILLA):
        self.metodo, self.f, self.tarea, self.modelo = metodo, objetivo, tarea, modelo
        self.B, self.semilla = presupuesto, semilla
        self.reg = Registro()
        self._cache = {}

    def _evaluar(self, params):
        clave = json.dumps(params, sort_keys=True, default=str)
        if clave in self._cache:                       # el AG puede repetir individuos (élite): no gasta presupuesto
            return self._cache[clave]
        t0 = time.perf_counter()
        v = float(self.f(params))
        self.reg.params.append(params); self.reg.valores.append(v); self.reg.tiempos.append(time.perf_counter() - t0)
        self._cache[clave] = v
        return v

    def correr(self) -> tuple:
        """Ejecuta el método de optimización elegido con el presupuesto fijado.

        Returns
        -------
        tuple
            ``(mejores_params, mejor_valor, registro)``: el valor es la métrica interna a maximizar (−RMSLE en regresión,
            AUC en clasificación) y el registro guarda todas las evaluaciones con sus tiempos (curva *anytime*).
        """
        getattr(self, {"Grid": "_grid", "Random": "_random", "Bayesiano (TPE)": "_tpe", "Genético (DEAP)": "_ga"}[self.metodo])()
        i = int(np.argmax(self.reg.valores))
        return self.reg.params[i], self.reg.valores[i], self.reg

    def _grid(self):
        for p in rejilla(self.tarea, self.modelo, self.B):
            self._evaluar(p)

    def _random(self):
        rng = np.random.default_rng(self.semilla)
        d = len(ESPACIOS[(self.tarea, self.modelo)])
        intentos = 0
        while len(self.reg.valores) < self.B and intentos < 10 * self.B:
            self._evaluar(desde_unitario(self.tarea, self.modelo, rng.random(d)))
            intentos += 1

    def _tpe(self):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        esp = ESPACIOS[(self.tarea, self.modelo)]

        def obj(trial):
            p = {}
            for q in esp:
                if q[1] == "cat":
                    p[q[0]] = trial.suggest_categorical(q[0], q[2])
                elif q[1] == "int":
                    p[q[0]] = trial.suggest_int(q[0], q[2], q[3], log=q[4])
                else:
                    p[q[0]] = trial.suggest_float(q[0], q[2], q[3], log=q[4])
            return self._evaluar(p)

        sampler = optuna.samplers.TPESampler(seed=self.semilla, n_startup_trials=min(5, self.B // 4 or 1), multivariate=True)
        estudio = optuna.create_study(direction="maximize", sampler=sampler)
        while len(self.reg.valores) < self.B and len(estudio.trials) < 3 * self.B:
            estudio.optimize(obj, n_trials=1)

    def _ga(self):
        from deap import base, creator, tools
        d = len(ESPACIOS[(self.tarea, self.modelo)])
        if not hasattr(creator, "AptitudMax"):
            creator.create("AptitudMax", base.Fitness, weights=(1.0,))
            creator.create("Individuo", list, fitness=creator.AptitudMax)
        rng = np.random.default_rng(self.semilla)
        tb = base.Toolbox()
        tb.register("evaluar", lambda ind: (self._evaluar(desde_unitario(self.tarea, self.modelo, ind)),))
        tb.register("cruce", tools.cxBlend, alpha=0.5)
        tb.register("mutar", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.5)
        tb.register("seleccionar", tools.selTournament, tournsize=3)
        tam = max(4, self.B // 4)                      # p. ej. presupuesto 20 -> población 5, 3 generaciones
        pob = [creator.Individuo(rng.random(d).tolist()) for _ in range(tam)]
        import random as _r
        _r.seed(self.semilla)

        def diversidad(p):
            A = np.array(p)
            return float(np.mean([np.linalg.norm(a - b) for i, a in enumerate(A) for b in A[i + 1:]])) if len(A) > 1 else 0.0

        for ind in pob:
            ind.fitness.values = tb.evaluar(ind)
        self.reg.diversidad.append(diversidad(pob))
        while len(self.reg.valores) < self.B:
            elite = tools.selBest(pob, 1)[0]
            hijos = [creator.Individuo(list(x)) for x in tb.seleccionar(pob, len(pob) - 1)]
            for a, b in zip(hijos[::2], hijos[1::2]):
                if _r.random() < 0.7:
                    tb.cruce(a, b)
            for h in hijos:
                if _r.random() < 0.3:
                    tb.mutar(h)
                h[:] = list(np.clip(h, 0, 1))
            antes = len(self.reg.valores)
            for h in hijos:
                if len(self.reg.valores) >= self.B:
                    h.fitness.values = (-np.inf,)
                    continue
                h.fitness.values = tb.evaluar(h)
            pob = [creator.Individuo(list(elite))] + hijos   # elitismo: el mejor pasa intacto
            pob[0].fitness.values = elite.fitness.values
            self.reg.diversidad.append(diversidad(pob))
            if len(self.reg.valores) == antes:             # población convergida a individuos ya evaluados
                for h in hijos:
                    h[:] = list(rng.random(d))             # reinyección aleatoria para no gastar ciclos sin evaluar
                    if len(self.reg.valores) < self.B:
                        h.fitness.values = tb.evaluar(h)


# ---------------------------------------------------------------------------
# Métricas por tarea
# ---------------------------------------------------------------------------
def metricas_clf(y, s, umbral: float = 0.5, es_prob: bool = True) -> dict:
    """Métricas de clasificación. Con puntajes que no son probabilidad (SVM) el umbral es 0."""
    pred = (s >= (umbral if es_prob else 0.0)).astype(int)
    out = {"ROC_AUC": roc_auc_score(y, s), "AP": average_precision_score(y, s), "Accuracy": accuracy_score(y, pred),
           "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred), "F1": f1_score(y, pred)}
    out["Brier"] = brier_score_loss(y, np.clip(s, 0, 1)) if es_prob else np.nan
    return out


def metricas_reg(y, p) -> dict:
    """Métricas de regresión de conteos usadas por todos los modelos.

    Parameters
    ----------
    y : array-like
        Conteos observados (≥ 0).
    p : array-like
        Pronósticos; los negativos se recortan a 0 antes de calcular.

    Returns
    -------
    dict
        ``{"RMSLE", "RMSE", "MAE"}``. El RMSLE es la métrica principal del proyecto.
    """
    p = np.clip(p, 0, None)
    return {"RMSLE": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))), "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
            "MAE": float(mean_absolute_error(y, p))}


# ---------------------------------------------------------------------------
# Validación anidada de una combinación
# ---------------------------------------------------------------------------
def pliegues_externos(tm_tr, n=3, meses_bloque=12, gap=cfg.GAP_MESES):
    """Bucle externo: ``n`` bloques consecutivos de ``meses_bloque`` meses al final del entrenamiento."""
    meses = np.unique(tm_tr)
    out = []
    for k in range(n, 0, -1):
        bloque = meses[len(meses) - k * meses_bloque: len(meses) - (k - 1) * meses_bloque]
        out.append((np.where(tm_tr < bloque[0] - gap)[0], np.where(np.isin(tm_tr, bloque))[0]))
    return out


def pliegues_internos(tm_sub, n=3, gap=cfg.GAP_MESES):
    """Pliegues temporales del bucle interno de la validación anidada.

    Parameters
    ----------
    tm_sub : array-like
        Índice de mes de cada fila del conjunto de ajuste.
    n : int
        Número de pliegues.
    gap : int
        Meses de separación entre entrenamiento y validación (evita fuga por rezagos).

    Returns
    -------
    list de tuple
        Pares ``(idx_entrenamiento, idx_validacion)`` en orden cronológico.

    Notes
    -----
    Supone que las filas de entrenamiento son siempre anteriores a las de validación.
    """
    from .datos import pliegues_temporales
    return pliegues_temporales(tm_sub, n_pliegues=n, gap=gap, min_frac=0.5)


def correr_combinacion(tarea, modelo, balanceo, optimizador, X, y, tm, tr, te, presupuesto=15,
                       n_ext=3, n_int=3, semilla=cfg.SEMILLA) -> dict:
    """Validación anidada completa de una combinación (modelo, balanceo, optimizador).

    1. Bucle externo temporal (``n_ext`` bloques de 12 meses al final del entrenamiento, con gap).
    2. En cada pliegue externo: el optimizador busca hiperparámetros con validación temporal interna
       (``n_int`` pliegues) sobre la parte de entrenamiento externa; se reajusta con ella y se evalúa en el
       bloque externo (esta es la métrica reportada).
    3. Final: el optimizador vuelve a buscar con todo el entrenamiento; el modelo final predice la prueba.

    Returns
    -------
    dict con métricas externas (media y desviación), hiperparámetros finales, historiales, tiempos y
    predicciones (fuera de pliegue externo y de prueba).
    """
    Xv = X
    cols = list(X.columns)
    y_obj = y if tarea == "reg" else (y >= 1).astype(int)

    def objetivo_para(idx_ajuste):
        """Construye la función objetivo del bucle interno para un conjunto de ajuste.

        Parameters
        ----------
        idx_ajuste : np.ndarray
            Índices de las filas de entrenamiento del pliegue externo.

        Returns
        -------
        callable
            ``f(params) -> float``: media de −RMSLE (regresión) o AUC (clasificación) sobre los pliegues internos; mayor es
            mejor.
        """
        internos = pliegues_internos(tm[idx_ajuste], n_int)

        def f(params):
            vals = []
            for a, v in internos:
                ia, iv = idx_ajuste[a], idx_ajuste[v]
                m = construir(tarea, modelo, params, cols, balanceo, semilla).fit(Xv.iloc[ia], y_obj[ia])
                if tarea == "reg":
                    vals.append(-metricas_reg(y[iv], m.predict(Xv.iloc[iv]))["RMSLE"])
                else:
                    vals.append(roc_auc_score(y_obj[iv], puntaje_clf(m, Xv.iloc[iv])))
            return float(np.mean(vals))
        return f

    t_total = time.perf_counter()
    externos = pliegues_externos(tm[tr], n_ext)
    filas_ext, hist, oof_idx, oof_pred = [], [], [], []
    for k, (a, v) in enumerate(externos):
        ia, iv = tr[a], tr[v]
        opt = Optimizador(optimizador, objetivo_para(ia), tarea, modelo, presupuesto, semilla + k)
        params, mejor_int, reg = opt.correr()
        t0 = time.perf_counter()
        m = construir(tarea, modelo, params, cols, balanceo, semilla).fit(Xv.iloc[ia], y_obj[ia])
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        if tarea == "reg":
            p = m.predict(Xv.iloc[iv]); met = metricas_reg(y[iv], p)
        else:
            p = puntaje_clf(m, Xv.iloc[iv]); met = metricas_clf(y_obj[iv], p, es_prob=hasattr(m, "predict_proba") and modelo != "SVM")
        t_pred = time.perf_counter() - t0
        filas_ext.append({"pliegue": k, "mejor_interno": mejor_int, "t_ajuste_s": t_fit, "t_inferencia_s": t_pred, **met})
        hist.append({"pliegue": k, "params": reg.params, "valores": reg.valores, "tiempos": reg.tiempos, "diversidad": reg.diversidad})
        oof_idx.append(iv); oof_pred.append(p)
    opt = Optimizador(optimizador, objetivo_para(tr), tarea, modelo, presupuesto, semilla + 99)
    params_fin, mejor_fin, reg_fin = opt.correr()
    m = construir(tarea, modelo, params_fin, cols, balanceo, semilla).fit(Xv.iloc[tr], y_obj[tr])
    p_te = m.predict(Xv.iloc[te]) if tarea == "reg" else puntaje_clf(m, Xv.iloc[te])
    ext = pd.DataFrame(filas_ext)
    principal = "RMSLE" if tarea == "reg" else "ROC_AUC"
    return {"tarea": tarea, "modelo": modelo, "balanceo": balanceo if tarea == "clf" else "—", "optimizador": optimizador,
            "presupuesto": presupuesto, "semilla": semilla, "estado": "completado", "error": "",
            "metrica_principal": principal, "externo_media": float(ext[principal].mean()), "externo_desv": float(ext[principal].std(ddof=1)),
            "externo": ext.to_dict("list"), "params_finales": params_fin, "mejor_interno_final": mejor_fin,
            "evaluaciones": int(sum(len(h["valores"]) for h in hist) + len(reg_fin.valores)),
            "t_total_s": time.perf_counter() - t_total, "historial": hist,
            "historial_final": {"params": reg_fin.params, "valores": reg_fin.valores, "tiempos": reg_fin.tiempos, "diversidad": reg_fin.diversidad},
            "oof_idx": np.concatenate(oof_idx), "oof_pred": np.concatenate(oof_pred), "pred_prueba": np.asarray(p_te, float)}


def correr_seguro(*args, **kw) -> dict:
    """Envuelve :func:`correr_combinacion` y registra el fallo en lugar de detener el experimento."""
    try:
        return correr_combinacion(*args, **kw)
    except Exception as e:  # noqa: BLE001
        tarea, modelo, balanceo, optimizador = args[:4]
        return {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": optimizador, "estado": "error",
                "error": f"{type(e).__name__}: {e}", "traza": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Reducción justificada del diseño (capítulo 13)
# ---------------------------------------------------------------------------
EXCLUIDAS_A_PRIORI = {
    ("KNN", "class_weight"): "equivalente a «sin balanceo» en ROC-AUC: la corrección por prior es una transformación "
                             "monótona de la probabilidad (mismo orden de las predicciones)",
    ("Naive Bayes", "class_weight"): "equivalente a «sin balanceo» en ROC-AUC: priors = (0,5; 0,5) solo reescala la razón "
                                     "de verosimilitudes por una constante (transformación monótona)",
}


def tamizaje_par(modelo, balanceo, X, y, tm, tr, presupuesto=5, n_int=2, semilla=cfg.SEMILLA) -> dict:
    """Fase 1 (baja fidelidad) para un par (modelo, balanceo) de clasificación.

    Solo el **último** pliegue externo (12 meses), Random Search con ``presupuesto`` evaluaciones y ``n_int``
    pliegues internos. Cuesta ≈ ``presupuesto * n_int + 1`` ajustes (11 con los valores por defecto) frente a
    ≈ 184 de la validación anidada completa.
    """
    t0 = time.perf_counter()
    cols = list(X.columns)
    yb = (y >= 1).astype(int)
    a, v = pliegues_externos(tm[tr], 3)[-1]
    ia, iv = tr[a], tr[v]
    internos = pliegues_internos(tm[ia], n_int)

    def f(params):
        vals = []
        for aa, vv in internos:
            m = construir("clf", modelo, params, cols, balanceo, semilla).fit(X.iloc[ia[aa]], yb[ia[aa]])
            vals.append(roc_auc_score(yb[ia[vv]], puntaje_clf(m, X.iloc[ia[vv]])))
        return float(np.mean(vals))

    params, interno, reg = Optimizador("Random", f, "clf", modelo, presupuesto, semilla).correr()
    t_fit = time.perf_counter()
    m = construir("clf", modelo, params, cols, balanceo, semilla).fit(X.iloc[ia], yb[ia])
    t_fit = time.perf_counter() - t_fit
    s = puntaje_clf(m, X.iloc[iv])
    met = metricas_clf(yb[iv], s, es_prob=modelo != "SVM")
    return {"modelo": modelo, "balanceo": balanceo, "AUC_tamizaje": met["ROC_AUC"], "AP_tamizaje": met["AP"],
            "Recall_tamizaje": met["Recall"], "F1_tamizaje": met["F1"], "AUC_interno": interno, "params": json.dumps(params, default=str),
            "evaluaciones": len(reg.valores), "t_total_s": time.perf_counter() - t0, "t_ajuste_final_s": t_fit}


def seleccionar_fase2(tamizaje: pd.DataFrame) -> pd.DataFrame:
    """Regla preregistrada para pasar a la fase 2 (validación anidada completa con los 4 optimizadores).

    Por cada modelo pasan:
    1. **«sin balanceo»** siempre (línea base de la comparación de balanceo exigida por la guía);
    2. la técnica de balanceo con **mayor AUC de tamizaje** entre las no excluidas a priori;
    3. cualquier otra técnica cuyo AUC quede a menos de 0,005 de esa mejor (empate práctico), para no
       descartar por ruido de un solo pliegue.
    """
    filas = []
    for modelo, g in tamizaje.groupby("modelo"):
        alt = g[g.balanceo != "sin balanceo"]
        mejor = alt.AUC_tamizaje.max() if len(alt) else -np.inf
        for _, r in g.iterrows():
            if r.balanceo == "sin balanceo":
                pasa, motivo = True, "línea base de balanceo (siempre pasa)"
            elif r.AUC_tamizaje >= mejor - 0.005:
                pasa, motivo = True, "mejor técnica de balanceo del modelo (o a < 0,005 de la mejor)"
            else:
                pasa, motivo = False, f"dominada: AUC {mejor - r.AUC_tamizaje:.3f} por debajo de la mejor técnica del modelo"
            filas.append({**r.to_dict(), "pasa_fase2": pasa, "motivo": motivo})
    return pd.DataFrame(filas)
