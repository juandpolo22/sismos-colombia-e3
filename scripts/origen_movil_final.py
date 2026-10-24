"""Modelo propuesto final (A+B v2: miembros + B + combinador contextual + calibración final) en el origen móvil
2021-2026, con su ablación (B solo, núcleo ZINB solo, A sin B y A sin B ni núcleo ZINB, todas con la misma calibración).

Calibración final: con el combinador ajustado en el 80 % inicial de los meses con predicción fuera de pliegue se
predice el 20 % final y se ajusta log1p(y) = c0 + c1·ŷ por mínimos cuadrados; el pronóstico es
expm1(max(0, c0 + c1·ŷ)) del combinador ajustado con todos los meses. Solo usa el entrenamiento de cada origen.

En cada origen anual entrena los miembros, B y el combinador con datos anteriores al año evaluado y, con los mismos
miembros y la misma regla de elección de l2, ajusta el combinador sin B y sin B ni núcleo ZINB.
Guarda resultados/origen_movil/final/origen_{año}.npz.

Uso:  python scripts/origen_movil_final.py [--trabajos 2]
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
from sklearn.decomposition import PCA

from src import benchmark as bm
from src import config as cfg
from src import datos
from src import evaluacion as ev
from src.propuesto import BoostingZINB, CombinadorSoftmax, ZINB, disenar, pliegues_cronologicos

OUT = cfg.RESULTADOS / "origen_movil"
HP, PR = OUT / "hp_bench", OUT / "pred_bench"
AB = OUT / "final"
AB.mkdir(parents=True, exist_ok=True)
ANIOS = list(range(2021, 2027))


def slug(m, o):
    return f"{m}__{o}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def mejor(m):
    return min(bm.OPTIMIZADORES, key=lambda o: json.load(open(HP / f"{slug(m, o)}.json"))["RMSLE_CV"])


MIEMBROS = [(m, o) for m in ("XGBoost", "Random Forest") for o in bm.OPTIMIZADORES] + [("KNN", mejor("KNN")), ("Ridge", mejor("Ridge"))]


def un_origen(anio, matrices=False):
    destino = OUT / "matrices" if matrices else AB
    destino.mkdir(exist_ok=True)
    ruta = destino / f"origen_{anio}.npz"
    if ruta.exists():
        return f"{anio}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, f"{anio}-01-01")
    tr, gp, te_todo = datos.particion(info)
    y, tm = info["y"].astype(float), info["tm"]
    te = te_todo[tm[te_todo] < datos.indice_mes([f"{anio + 1}-01-01"])[0]]
    ylog = np.log1p(y)
    XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
    XEv = np.nan_to_num(XE.to_numpy(float), nan=-1.0)
    meses_tr = np.unique(tm[tr])
    PL, MASK = pliegues_cronologicos(tm[tr], k=5, gap=cfg.GAP_MESES)      # OOF solo con pasado
    FILAS = np.where(MASK)[0]
    B_tr, B_te = [], []
    for m, o in MIEMBROS:
        par = json.load(open(HP / f"{slug(m, o)}.json"))["params"]
        fab = lambda m=m, par=par: bm.construir("reg", m, par, X.columns, n_jobs=1)  # noqa: E731
        oof = np.full(len(tr), np.nan)
        for a_, b_ in PL:
            mod = fab().fit(X.iloc[tr[a_]], y[tr[a_]])
            oof[b_] = np.log1p(np.clip(mod.predict(X.iloc[tr[b_]]), 0, None))
        B_tr.append(oof); B_te.append(np.log1p(np.clip(fab().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, None)))
    Xm_, W = disenar(X, XE, tr)
    core_oof = np.full(len(tr), np.nan)
    for a_, b_ in PL:
        core_oof[b_], _ = ZINB().fit(Xm_[tr][a_], W[tr][a_], y[tr][a_]).predecir(Xm_[tr][b_], W[tr][b_])
    core_te, _ = ZINB().fit(Xm_[tr], W[tr], y[tr]).predecir(Xm_[te], W[te])
    bz_oof = np.full(len(tr), np.nan)
    for a_, b_ in PL:
        bz_oof[b_], _ = BoostingZINB().fit(XEv[tr][a_], y[tr][a_], tm[tr][a_]).predecir(XEv[tr][b_])
    bz_te, _ = BoostingZINB().fit(XEv[tr], y[tr], tm[tr]).predecir(XEv[te])
    B_tr = np.column_stack(B_tr + [core_oof, bz_oof, np.log1p(info["persist"][tr]), np.log1p(info["estacional"][tr])])
    B_te = np.column_stack(B_te + [core_te, bz_te, np.log1p(info["persist"][te]), np.log1p(info["estacional"][te])])
    C, cc, tt = info["C"], info["cc"], info["tt"]
    hist = np.stack([C[cc, tt - j] for j in range(12)], axis=1)
    ctx = PCA(n_components=4, random_state=cfg.SEMILLA).fit(np.log1p(hist[tr])).transform(np.log1p(hist))
    lg = lambda n: np.log1p(X[n].to_numpy(float))  # noqa: E731
    Zr = np.column_stack([lg("n_1m"), lg("n_3m"), lg("n_12m"), lg("n35_12m"), lg("vec_1m"), XE["exc_a0.5_s30_r0.4"].to_numpy(),
                          XE["exc_a0.9_s60_r0.8"].to_numpy(), ctx, (hist == 0).mean(1),
                          np.nan_to_num(X["mag_max_12m"].to_numpy(float), nan=cfg.M_MIN), X["sen_mes"].to_numpy(), X["cos_mes"].to_numpy()])
    Zr = (Zr - Zr[tr].mean(0)) / (Zr[tr].std(0) + 1e-9)
    Z = np.hstack([np.ones((len(Zr), 1)), Zr])
    meses_m = np.unique(tm[tr][FILAS])
    corte = meses_m[int(len(meses_m) * 0.8)]
    a, b = FILAS[tm[tr][FILAS] < corte], FILAS[tm[tr][FILAS] >= corte]
    Zt, yt = Z[tr], ylog[tr]
    errs = {l2: np.mean((CombinadorSoftmax(l2).fit(Zt[a], B_tr[a], yt[a]).predict(Zt[b], B_tr[b]) - yt[b]) ** 2)
            for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
    def ajustar_calibrado(cols):
        """Combinador con la regla de l2 + calibración final; devuelve pronóstico, (c0, c1) y pesos medios."""
        e2 = {l2: np.mean((CombinadorSoftmax(l2).fit(Zt[a], B_tr[a][:, cols], yt[a]).predict(Zt[b], B_tr[b][:, cols]) - yt[b]) ** 2)
              for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
        l2b = min(e2, key=e2.get)
        cmb = CombinadorSoftmax(l2b).fit(Zt[FILAS], B_tr[FILAS][:, cols], yt[FILAS])
        plog = cmb.predict(Z[te], B_te[:, cols])
        pv = CombinadorSoftmax(l2b).fit(Zt[a], B_tr[a][:, cols], yt[a]).predict(Zt[b], B_tr[b][:, cols])
        c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(len(b)), pv]), yt[b], rcond=None)[0]
        return np.expm1(np.clip(c0 + c1 * plog, 0, None)), np.expm1(plog).clip(0), (c0, c1), cmb.pesos(Z[te]).mean(0)
    todas = list(range(B_tr.shape[1]))
    p, p_sin, (c0, c1), w = ajustar_calibrado(todas)
    out = {"y": y[te], "tm": tm[te], "AmasB": p, "AmasB_sin_calibrar": p_sin, "c0": c0, "c1": c1, "pesos": w,
           "B_solo": np.expm1(B_te[:, 11]).clip(0), "ZINB_solo": np.expm1(B_te[:, 10]).clip(0)}
    for nombre, quitar in (("A_sin_B", [11]), ("A_sin_B_ni_ZINB", [10, 11])):
        out[nombre] = ajustar_calibrado([j for j in todas if j not in quitar])[0]
    if matrices:      # insumos del combinador para contrastar reglas de combinación estándar (cap. 5 del E3)
        out.update({"B_ajuste": B_tr[FILAS], "B_prueba": B_te, "Z_ajuste": Zt[FILAS], "Z_prueba": Z[te], "ylog_ajuste": yt[FILAS],
                    "tm_ajuste": tm[tr][FILAS], "corte_calibracion": corte})
    np.savez(ruta, **out)
    return f"{anio}: A+B {ev.rmsle(y[te], p):.4f} (c1 = {c1:.3f}), A sin B {ev.rmsle(y[te], out['A_sin_B']):.4f} ({time.time() - t0:.0f} s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=2)
    ap.add_argument("--matrices", action="store_true", help="repite el origen móvil en resultados/origen_movil/matrices/ "
                    "guardando además los insumos del combinador (no toca los resultados oficiales)")
    a = ap.parse_args()

    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(un_origen)(y_, a.matrices) for y_ in ANIOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
