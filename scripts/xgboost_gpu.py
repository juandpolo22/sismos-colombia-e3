"""Aceleración por GPU de XGBoost (sección 4.3 de la guía del Entregable 2).

Entrena el mismo XGBoost (regresión, objetivo log1p, 18 variables del panel) en CPU y en GPU con
varios tamaños de muestra y repite cada medición 3 veces. Guarda ``resultados/benchmark/xgboost_gpu.csv``
con tiempos, factor de aceleración y RMSLE en la prueba (para verificar que el resultado no cambia).

Uso:  python scripts/xgboost_gpu.py        (requiere GPU NVIDIA con CUDA y xgboost con soporte CUDA)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xgboost as xgb

from src import config as cfg
from src import datos, evaluacion

ev_, X, info, tr, gp, te = datos.panel_completo()
y = info["y"].astype(float)
Xtr, Xte = X.iloc[tr].to_numpy(float), X.iloc[te].to_numpy(float)
ytr = np.log1p(y[tr])
filas = []
for disp in ("cpu", "cuda"):
    for frac in (0.25, 0.5, 1.0):
        n = int(len(tr) * frac)
        for rep in range(3):
            m = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, tree_method="hist", device=disp,
                                 random_state=cfg.SEMILLA, n_jobs=-1 if disp == "cpu" else None)
            try:
                t0 = time.perf_counter(); m.fit(Xtr[:n], ytr[:n]); t_fit = time.perf_counter() - t0
                t0 = time.perf_counter(); p = np.expm1(m.predict(Xte)).clip(0); t_pred = time.perf_counter() - t0
                filas.append({"dispositivo": disp, "filas": n, "repeticion": rep, "t_ajuste_s": t_fit, "t_inferencia_s": t_pred,
                              "RMSLE_prueba": evaluacion.rmsle(y[te], p), "estado": "ok"})
            except Exception as e:  # noqa: BLE001  (sin GPU o sin CUDA)
                filas.append({"dispositivo": disp, "filas": n, "repeticion": rep, "estado": f"error: {e}"})
            print(filas[-1], flush=True)
r = pd.DataFrame(filas)
r.to_csv(cfg.RESULTADOS / "benchmark" / "xgboost_gpu.csv", index=False)
ok = r[r.estado == "ok"]
if ok.dispositivo.nunique() == 2:
    g = ok.groupby(["filas", "dispositivo"]).t_ajuste_s.mean().unstack()
    g["aceleracion_GPU"] = g["cpu"] / g["cuda"]
    print(g)
