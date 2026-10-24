"""Miembros del modelo propuesto con datos anteriores a 2021: predicciones fuera de pliegue CRONOLÓGICAS.

Miembros: las 4 versiones optimizadas (Grid, Random, TPE, Genético) de XGBoost y de Random Forest + el mejor KNN y
el mejor Ridge según el bucle externo de la validación anidada (solo entrenamiento) + núcleo ZINB propio +
B (boosting ZINB propio) + persistencia + estacional.

1. Hiperparámetros de cada miembro: los finales de la validación anidada (solo entrenamiento).
2. Predicciones fuera de pliegue cronológicas: 5 bloques contiguos de meses; cada bloque f ≥ 1 se predice con un
   modelo entrenado solo con los meses anteriores (gap de 1 mes). El primer bloque no se usa.
3. Validación previa del combinador sin calibración (80 % inicial / 20 % final de esos meses).

Las predicciones se guardan en resultados/propuesto/cronologico/ y las usa scripts/validacion_previa_final.py.
La evaluación 2021–2026 se hace en el origen móvil (scripts/origen_movil_final.py).

Uso:  python scripts/miembros_fuera_de_pliegue.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import benchmark as bm
from src import config as cfg
from src import datos, eda
from src import evaluacion as ev
from src.propuesto import BoostingZINB, CombinadorSoftmax, ZINB, disenar, pliegues_cronologicos

OUT = cfg.RESULTADOS / "propuesto"
CACHE = OUT / "cronologico"
CACHE.mkdir(parents=True, exist_ok=True)
t00 = time.time()
cfg.fijar_semillas()
ev_, X, info, tr, gp, te = datos.panel_completo()
y = info["y"].astype(float); tm = info["tm"]; ylog = np.log1p(y)
XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
XEv = np.nan_to_num(XE.to_numpy(float), nan=-1.0)
meses_tr = np.unique(tm[tr])
PLIEGUES, MASK = pliegues_cronologicos(tm[tr], k=5, gap=cfg.GAP_MESES)   # solo pasado
print("Filas con predicción fuera de pliegue cronológica:", int(MASK.sum()), "de", len(tr), flush=True)

# 1. miembros optimizados elegidos por el bucle externo (solo entrenamiento)
maestro = pd.read_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv")
reg = maestro[(maestro.tarea == "reg") & (maestro.estado == "completado")]
elegidos = {}
for m in ("XGBoost", "Random Forest"):
    for _, fila in reg[reg.modelo == m].iterrows():
        elegidos[(m, fila.optimizador)] = json.loads(fila.params_finales)
for m in ("KNN", "Ridge"):
    fila = reg[reg.modelo == m].sort_values("externo_media").iloc[0]
    elegidos[(m, fila.optimizador)] = json.loads(fila.params_finales)
pd.DataFrame([{"miembro": k[0], "optimizador": k[1], "params": json.dumps(v)} for k, v in elegidos.items()]).to_csv(OUT / "miembros_v2.csv", index=False)
print("Miembros:", list(elegidos), flush=True)


def oof_y_prueba(fabrica, Xm, nombre):
    ruta = CACHE / f"miembro_{nombre}.npz"
    if ruta.exists():
        z = np.load(ruta); return z["oof"], z["te"]
    oof = np.full(len(tr), np.nan)
    for a, b in PLIEGUES:
        m = fabrica().fit(Xm(tr[a]), y[tr[a]])
        oof[b] = np.log1p(np.clip(m.predict(Xm(tr[b])), 0, None))
    m = fabrica().fit(Xm(tr), y[tr])
    p = np.log1p(np.clip(m.predict(Xm(te)), 0, None))
    np.savez(ruta, oof=oof, te=p)
    print(f"  {nombre} listo [{time.time() - t00:.0f} s]", flush=True)
    return oof, p


B_tr, B_te, nombres = [], [], []
for (m, opt), par in elegidos.items():
    slug = f"{m}__{opt}".replace(" ", "_").replace("(", "").replace(")", "")
    o, p = oof_y_prueba(lambda m=m, par=par: bm.construir("reg", m, par, X.columns, n_jobs=-1), lambda i: X.iloc[i], slug)
    B_tr.append(o); B_te.append(p); nombres.append(f"{m} ({opt})")

# 2. núcleo ZINB (propio) y modelo B (boosting ZINB propio)
ruta = CACHE / "miembro_ZINB.npz"
if ruta.exists():
    z = np.load(ruta); core_oof, core_te = z["oof"], z["te"]
else:
    Xm_, W = disenar(X, XE, tr)
    core_oof = np.full(len(tr), np.nan)
    for a, b in PLIEGUES:
        mf = ZINB().fit(Xm_[tr][a], W[tr][a], y[tr][a])
        core_oof[b], _ = mf.predecir(Xm_[tr][b], W[tr][b])
    core_te, _ = ZINB().fit(Xm_[tr], W[tr], y[tr]).predecir(Xm_[te], W[te])
    np.savez(ruta, oof=core_oof, te=core_te)
B_PAR = {"lr": 0.08, "depth": 3, "min_leaf": 50}
print("Hiperparámetros de B:", B_PAR, flush=True)
ruta = CACHE / f"miembro_BZINB_lr{B_PAR['lr']:.4f}_d{B_PAR['depth']}_m{B_PAR['min_leaf']}.npz"
if ruta.exists():
    z = np.load(ruta); bz_oof, bz_te = z["oof"], z["te"]
else:
    bz_oof = np.full(len(tr), np.nan)
    for f, (a, b) in enumerate(PLIEGUES, start=1):
        mb = BoostingZINB(**B_PAR).fit(XEv[tr][a], y[tr][a], tm[tr][a])
        bz_oof[b], _ = mb.predecir(XEv[tr][b])
        print(f"  B bloque {f}: {mb.n_best} iteraciones [{time.time() - t00:.0f} s]", flush=True)
    bz_te, _ = BoostingZINB(**B_PAR).fit(XEv[tr], y[tr], tm[tr]).predecir(XEv[te])
    np.savez(ruta, oof=bz_oof, te=bz_te)
B_tr += [core_oof, bz_oof, np.log1p(info["persist"][tr]), np.log1p(info["estacional"][tr])]
B_te += [core_te, bz_te, np.log1p(info["persist"][te]), np.log1p(info["estacional"][te])]
nombres += ["Núcleo ZINB (propio)", "B: boosting ZINB (propio)", "Persistencia", "Ingenuo estacional"]
B_tr, B_te = np.column_stack(B_tr), np.column_stack(B_te)

# 3. meta-características (solo pasado; estandarizadas con entrenamiento)
C, cc, tt = info["C"], info["cc"], info["tt"]
hist = np.stack([C[cc, tt - j] for j in range(12)], axis=1)
ctx = PCA(n_components=4, random_state=cfg.SEMILLA).fit(np.log1p(hist[tr])).transform(np.log1p(hist))
lg = lambda n: np.log1p(X[n].to_numpy(float))  # noqa: E731
mmax = np.nan_to_num(X["mag_max_12m"].to_numpy(float), nan=cfg.M_MIN)
Zr = np.column_stack([lg("n_1m"), lg("n_3m"), lg("n_12m"), lg("n35_12m"), lg("vec_1m"), XE["exc_a0.5_s30_r0.4"].to_numpy(),
                      XE["exc_a0.9_s60_r0.8"].to_numpy(), ctx, (hist == 0).mean(1), mmax, X["sen_mes"].to_numpy(), X["cos_mes"].to_numpy()])
Zr = (Zr - Zr[tr].mean(0)) / (Zr[tr].std(0) + 1e-9)
Z = np.hstack([np.ones((len(Zr), 1)), Zr])


def ajustar(idx_a, idx_v_list):
    """Elige l2 con el 20 % final de idx_a y ajusta con todo idx_a."""
    m_a = tm[tr][idx_a]
    corte = np.unique(m_a)[int(len(np.unique(m_a)) * 0.8)]
    a, b = idx_a[m_a < corte], idx_a[m_a >= corte]
    errs = {l2: np.mean((CombinadorSoftmax(l2).fit(Z[tr][a], B_tr[a], ylog[tr][a]).predict(Z[tr][b], B_tr[b]) - ylog[tr][b]) ** 2)
            for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
    l2 = min(errs, key=errs.get)
    return CombinadorSoftmax(l2).fit(Z[tr][idx_a], B_tr[idx_a], ylog[tr][idx_a]), l2


assert np.isfinite(B_tr[MASK]).all() and np.isnan(B_tr[~MASK, 0]).all()
FILAS = np.where(MASK)[0]                       # filas con predicciones cronológicas de todos los miembros

# 4. validación previa (80 % / 20 % de los meses con predicción cronológica)
meses_m = np.unique(tm[tr][FILAS])
corte_v = meses_m[int(len(meses_m) * 0.8)]
ia, iv = FILAS[tm[tr][FILAS] < corte_v], FILAS[tm[tr][FILAS] >= corte_v]
comb_v, l2_v = ajustar(ia, None)
p_v = np.expm1(comb_v.predict(Z[tr][iv], B_tr[iv]))
_, L_ab = ev.perdida_mensual(y[tr][iv], p_v, tm[tr][iv])
filas = []
for j, n in enumerate(nombres):
    _, L_m = ev.perdida_mensual(y[tr][iv], np.expm1(B_tr[iv, j]), tm[tr][iv])
    d = ev.diebold_mariano(L_m, L_ab)
    filas.append({"miembro": n, "RMSLE_miembro": ev.rmsle(y[tr][iv], np.expm1(B_tr[iv, j])), "dif": d["dif_media"], "p": d["p"]})
val = pd.DataFrame(filas)
val["p_holm"] = eda.holm(val.p)
val.insert(0, "RMSLE_AB", ev.rmsle(y[tr][iv], p_v))
val.to_csv(OUT / "validacion_previa_v2.csv", index=False)
print("\nVALIDACIÓN PREVIA (20 % final del entrenamiento):\n", val.round(4).to_string(index=False), flush=True)
print(f"[{(time.time() - t00) / 60:.1f} min]")
