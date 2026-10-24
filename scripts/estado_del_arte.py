"""Comparación con métodos recientes del estado del arte, además de los 7 benchmarks vistos en clase.

* LightGBM (Ke et al., 2017) y CatBoost (Prokhorenkova et al., 2018), con el objetivo en log1p como los benchmarks;
* boosting de gradiente por histogramas con pérdida de Poisson (scikit-learn), que modela el conteo directamente.

Mismo protocolo y mismo presupuesto que los benchmarks: hiperparámetros con optimización bayesiana (TPE) de 15
evaluaciones y validación temporal interna de 3 pliegues, una sola vez, con datos anteriores a 2016; después, origen
móvil 2021–2026 reentrenando cada año. Se compara con A+B con Diebold-Mariano (HLN) y corrección de Holm.

Uso:  python scripts/estado_del_arte.py [--trabajos 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src import benchmark as bm
from src import config as cfg
from src import datos, eda
from src import evaluacion as ev
from src import modelos as mdl

OUT = cfg.RESULTADOS / "e3" / "estado_del_arte"
OUT.mkdir(parents=True, exist_ok=True)
OM = cfg.RESULTADOS / "origen_movil"
ANIOS = list(range(2021, 2027))
MODELOS = ["LightGBM", "CatBoost", "HGB Poisson"]
TPE = "Bayesiano (TPE)"
ESPACIO_HGB = [("learning_rate", "float", 0.01, 0.3, True), ("max_leaf_nodes", "int", 8, 128, True),
                                       ("min_samples_leaf", "int", 10, 300, True), ("l2_regularization", "float", 1e-4, 10.0, True),
                                       ("max_iter", "int", 100, 600, False)]


def _registrar_espacio():
    """Espacio de búsqueda del boosting de Poisson (se registra en cada proceso de joblib)."""
    bm.ESPACIOS[("reg", "HGB Poisson")] = ESPACIO_HGB


def construir(m, p, columnas):
    if m == "HGB Poisson":       # pérdida de Poisson sobre el conteo (sin transformar el objetivo)
        return Pipeline([("prep", mdl.preprocesador(columnas, "constante")),
                         ("modelo", HistGradientBoostingRegressor(loss="poisson", random_state=cfg.SEMILLA, **p))])
    return bm.construir("reg", m, p, columnas, n_jobs=1)


def panel(corte):
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, corte)
    tr, gp, te = datos.particion(info)
    return X, info, tr, te


def afinar(m):
    ruta = OUT / f"hp_{m.replace(' ', '_')}.json"
    if ruta.exists():
        return f"{m}: ya existía"
    t0 = time.time()
    _registrar_espacio()
    cfg.fijar_semillas()
    X, info, tr, te = panel("2016-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    internos = bm.pliegues_internos(tm[tr], 3)

    def f(params):
        r = []
        for a, v in internos:
            mod = construir(m, params, X.columns).fit(X.iloc[tr[a]], y[tr[a]])
            r.append(-ev.rmsle(y[tr[v]], np.clip(mod.predict(X.iloc[tr[v]]), 0, None)))
        return float(np.mean(r))
    params, val, _ = bm.Optimizador(TPE, f, "reg", m, 15, cfg.SEMILLA).correr()
    json.dump({"params": params, "RMSLE_CV": -val}, open(ruta, "w"), indent=2, default=float)
    return f"{m}: RMSLE CV {-val:.4f} ({time.time() - t0:.0f} s)"


def un_origen(anio):
    ruta = OUT / f"pred_{anio}.npz"
    if ruta.exists():
        return f"{anio}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    X, info, tr, te_todo = panel(f"{anio}-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    te = te_todo[tm[te_todo] < datos.indice_mes([f"{anio + 1}-01-01"])[0]]
    ref = np.load(OM / "final" / f"origen_{anio}.npz")
    assert np.array_equal(ref["y"], y[te]), "filas distintas a las de A+B"
    out = {"y": y[te], "tm": tm[te]}
    for m in MODELOS:
        p = json.load(open(OUT / f"hp_{m.replace(' ', '_')}.json"))["params"]
        out[m] = np.clip(construir(m, p, X.columns).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, None)
    np.savez(ruta, **out)
    return f"{anio}: listo ({time.time() - t0:.0f} s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=2)
    a = ap.parse_args()
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(afinar)(m) for m in MODELOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(un_origen)(y_) for y_ in ANIOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    partes = [np.load(OUT / f"pred_{a_}.npz") for a_ in ANIOS]
    y, tm = np.concatenate([p["y"] for p in partes]), np.concatenate([p["tm"] for p in partes])
    ab = np.concatenate([np.load(OM / "final" / f"origen_{a_}.npz")["AmasB"] for a_ in ANIOS])
    _, Lab = ev.perdida_mensual(y, ab, tm)
    filas = []
    for m in MODELOS:
        pr = np.concatenate([p[m] for p in partes])
        _, L = ev.perdida_mensual(y, pr, tm)
        d = ev.diebold_mariano(L, Lab)
        filas.append({"método": m, "RMSLE rival": ev.rmsle(y, pr), "RMSLE A+B": ev.rmsle(y, ab), "dif. pérdida (rival − A+B)": d["dif_media"],
                      "p DM": d["p"], "d de Cohen": float(np.mean(L - Lab) / np.std(L - Lab, ddof=1))})
    T = pd.DataFrame(filas)
    T["p Holm (3)"] = eda.holm(T["p DM"].to_numpy())
    T["veredicto"] = np.where(T["p Holm (3)"] < 0.05, np.where(T["dif. pérdida (rival − A+B)"] > 0, "A+B mejor", "A+B peor"), "sin diferencia significativa")
    T.to_csv(OUT / "comparacion.csv", index=False)
    print(T.round(5).to_string(index=False))
