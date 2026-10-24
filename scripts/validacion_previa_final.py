"""Validación previa del modelo propuesto final (A+B v2 con calibración) en el 20 % final de los meses de
entrenamiento (< 2021), sin tocar 2021-2026. Requiere las predicciones fuera de pliegue de los miembros que guarda
scripts/miembros_fuera_de_pliegue.py en resultados/propuesto/cronologico/.

Uso:  python scripts/validacion_previa_final.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from src import benchmark as bm, config as cfg, datos, evaluacion as ev, modelos as mdl
from src.propuesto import CombinadorSoftmax, pliegues_cronologicos
from sklearn.pipeline import Pipeline

CACHE = cfg.RESULTADOS / "propuesto" / "cronologico"
cfg.fijar_semillas()
ev_, X, info, tr, gp, te = datos.panel_completo()
y = info["y"].astype(float); tm = info["tm"]; ylog = np.log1p(y)
XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
PL, MASK = pliegues_cronologicos(tm[tr], k=5, gap=cfg.GAP_MESES)
orden = ["XGBoost__Grid", "XGBoost__Random", "XGBoost__Bayesiano_TPE", "XGBoost__Genético_DEAP",
         "Random_Forest__Grid", "Random_Forest__Random", "Random_Forest__Bayesiano_TPE", "Random_Forest__Genético_DEAP",
         "KNN__Random", "Ridge__Genético_DEAP", "ZINB", "BZINB_lr0.0800_d3_m50"]
B = [np.load(CACHE / f"miembro_{n}.npz")["oof"] for n in orden]
B += [np.log1p(info["persist"][tr]), np.log1p(info["estacional"][tr])]
B = np.column_stack(B)
C, cc, tt = info["C"], info["cc"], info["tt"]
hist = np.stack([C[cc, tt - j] for j in range(12)], axis=1)
ctx = PCA(n_components=4, random_state=cfg.SEMILLA).fit(np.log1p(hist[tr])).transform(np.log1p(hist))
lg = lambda n: np.log1p(X[n].to_numpy(float))
Zr = np.column_stack([lg("n_1m"), lg("n_3m"), lg("n_12m"), lg("n35_12m"), lg("vec_1m"), XE["exc_a0.5_s30_r0.4"].to_numpy(),
                      XE["exc_a0.9_s60_r0.8"].to_numpy(), ctx, (hist == 0).mean(1),
                      np.nan_to_num(X["mag_max_12m"].to_numpy(float), nan=cfg.M_MIN), X["sen_mes"].to_numpy(), X["cos_mes"].to_numpy()])
FILAS = np.where(MASK)[0]
meses_m = np.unique(tm[tr][FILAS]); corte_v = meses_m[int(len(meses_m) * 0.8)]
ia, iv = FILAS[tm[tr][FILAS] < corte_v], FILAS[tm[tr][FILAS] >= corte_v]


from src import eda
Zs = (Zr - Zr[tr].mean(0)) / (Zr[tr].std(0) + 1e-9); Z = np.hstack([np.ones((len(Zs), 1)), Zs])[tr]
m_a = tm[tr][ia]; c = np.unique(m_a)[int(len(np.unique(m_a)) * 0.8)]; a, b = ia[m_a < c], ia[m_a >= c]
errs = {l2: np.mean((CombinadorSoftmax(l2).fit(Z[a], B[a], ylog[tr][a]).predict(Z[b], B[b]) - ylog[tr][b]) ** 2) for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
l2 = min(errs, key=errs.get)
pb = CombinadorSoftmax(l2).fit(Z[a], B[a], ylog[tr][a]).predict(Z[b], B[b])
c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(len(b)), pb]), ylog[tr][b], rcond=None)[0]
pv = np.expm1(np.clip(c0 + c1 * CombinadorSoftmax(l2).fit(Z[ia], B[ia], ylog[tr][ia]).predict(Z[iv], B[iv]), 0, None))
yv = y[tr][iv]; tv = tm[tr][iv]
nombres = ["XGBoost (Grid)", "XGBoost (Random)", "XGBoost (Bayesiano (TPE))", "XGBoost (Genético (DEAP))", "Random Forest (Grid)", "Random Forest (Random)",
           "Random Forest (Bayesiano (TPE))", "Random Forest (Genético (DEAP))", "KNN (Random)", "Ridge (Genético (DEAP))", "Núcleo ZINB (propio)",
           "B: boosting ZINB (propio)", "Persistencia", "Ingenuo estacional"]
_, Lab = ev.perdida_mensual(yv, pv, tv); filas = []
for j, n in enumerate(nombres):
    pm = np.expm1(B[iv, j]); _, Lm = ev.perdida_mensual(yv, pm, tv); d = ev.diebold_mariano(Lm, Lab)
    filas.append({"miembro": n, "RMSLE_miembro": ev.rmsle(yv, pm), "dif": d["dif_media"], "p": d["p"]})
V = pd.DataFrame(filas); V["p_holm"] = eda.holm(V.p.values); V.insert(0, "RMSLE_AB", ev.rmsle(yv, pv))
V.to_csv(cfg.RESULTADOS / "propuesto" / "validacion_previa_final.csv", index=False)
print("c0, c1:", round(c0, 4), round(c1, 4)); print(V.round(4).to_string(index=False))
np.savez(cfg.RESULTADOS / "propuesto" / "AB_final_validacion.npz", oof_val=pv, idx_val=tr[iv])
