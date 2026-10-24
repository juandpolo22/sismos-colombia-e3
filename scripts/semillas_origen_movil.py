"""Sensibilidad a la semilla en el origen móvil (cap. 20.2).

Para el campeón de cada modelo aleatorio (y KNN como referencia determinista), con los hiperparámetros fijados
antes de 2016, se reentrena el modelo del primer origen (datos hasta 2020) con 6 semillas (1–5 y 42) y se evalúa
en los 12 meses de 2021 que ese origen pronostica. Guarda resultados/e2/semillas_modelos_om.csv.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import benchmark as bm
from src import config as cfg
from src import datos
from src import evaluacion as ev
from src import resultados_e2 as r2

ruta = cfg.RESULTADOS / "e2" / "semillas_modelos_om.csv"
ev_, X, info, tr, gp, te = datos.panel_completo()
y, tm = info["y"].astype(float), info["tm"]
te = te[tm[te] < datos.indice_mes(["2022-01-01"])[0]]
filas = []
cr, cc = r2.campeones_om("reg"), r2.campeones_om("clf")
trabajos = [("reg", f.modelo, None, f.variante) for _, f in cr[cr.modelo.isin(["XGBoost", "Random Forest", "Árbol de decisión", "KNN"])].iterrows()]
trabajos += [("clf", f.modelo, f.variante, r2.TPE) for _, f in cc[cc.modelo.isin(["XGBoost", "Random Forest", "Árbol de decisión", "SVM"])].iterrows()]
for tarea, mod, bal, opt in trabajos:
    if tarea == "reg":
        params = json.load(open(r2.OM / "hp_bench" / f"{r2._slug(mod, opt)}.json"))["params"]
    else:
        params = json.load(open(r2.OM / "hp_clf" / f"{r2._slug(mod, bal, opt)}.json"))["params"]
    for s in (1, 2, 3, 4, 5, 42):
        t0 = time.perf_counter()
        if tarea == "reg":
            m = bm.construir("reg", mod, params, X.columns, semilla=s, n_jobs=2).fit(X.iloc[tr], y[tr])
            val, met = ev.rmsle(y[te], m.predict(X.iloc[te])), "RMSLE"
        else:
            m = bm.construir("clf", mod, params, X.columns, bal, semilla=s, n_jobs=2).fit(X.iloc[tr], (y[tr] >= 1).astype(int))
            val, met = roc_auc_score((y[te] >= 1).astype(int), bm.puntaje_clf(m, X.iloc[te])), "ROC_AUC"
        filas.append({"tarea": tarea, "modelo": mod, "balanceo": bal, "optimizador": opt, "semilla": s, "metrica": met, "valor": val,
                      "t_s": time.perf_counter() - t0})
        print(f"{tarea} {mod} semilla {s}: {val:.4f}", flush=True)
pd.DataFrame(filas).to_csv(ruta, index=False)
