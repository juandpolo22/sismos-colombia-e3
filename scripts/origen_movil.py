"""Evaluación de origen móvil (expanding window) del modelo propuesto A+B frente a XGBoost y Random Forest.

Protocolo fijado antes de ver resultados:
* Orígenes anuales 2016, 2017, ..., 2026: para cada año Y se construye el panel con corte en Y-01
  (celdas activas y normalizaciones solo con datos anteriores), se entrena con todo lo anterior (gap de
  un mes) y se pronostican los meses objetivo del año Y.
* Hiperparámetros de XGBoost, Random Forest y de los miembros de A+B (XGBoost, Random Forest, KNN,
  Ridge): Random Search de 15 evaluaciones con validación temporal interna de 3 pliegues, UNA sola vez,
  con el panel de corte 2016 (solo datos anteriores a 2016). Después quedan fijos.
* A+B se usa tal cual (B con sus hiperparámetros originales).
* Comparación: Diebold-Mariano (HLN) sobre todos los meses fuera de muestra concatenados.

Uso:  python scripts/origen_movil.py [--trabajos 2]
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
from src.propuesto import BoostingZINB, CombinadorSoftmax, ZINB, disenar

OUT = cfg.RESULTADOS / "origen_movil"
OUT.mkdir(parents=True, exist_ok=True)
ANIOS = list(range(2016, 2027))
MIEMBROS = ("XGBoost", "Random Forest", "KNN", "Ridge")


def panel(corte):
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, corte)
    tr, gp, te = datos.particion(info)
    return ev_, X, info, tr, te


def afinar():
    """Random Search (15 evaluaciones) con validación temporal interna, solo datos < 2016."""
    ruta = OUT / "hiperparametros_2016.json"
    if ruta.exists():
        return json.load(open(ruta))
    ev_, X, info, tr, te = panel("2016-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    internos = bm.pliegues_internos(tm[tr], 3)
    out = {}
    for m in MIEMBROS:
        def f(params, m=m):
            r = []
            for a, v in internos:
                mod = bm.construir("reg", m, params, X.columns, n_jobs=-1).fit(X.iloc[tr[a]], y[tr[a]])
                r.append(-ev.rmsle(y[tr[v]], mod.predict(X.iloc[tr[v]])))
            return float(np.mean(r))
        params, val, _ = bm.Optimizador("Random", f, "reg", m, 15, cfg.SEMILLA).correr()
        out[m] = params
        print(f"  {m}: {params} (RMSLE CV {-val:.4f})", flush=True)
    json.dump(out, open(ruta, "w"), indent=2)
    return out


def un_origen(anio, hp):
    ruta = OUT / f"origen_{anio}.npz"
    if ruta.exists():
        return f"{anio}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    ev_, X, info, tr, te_todo = panel(f"{anio}-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    fin = datos.indice_mes([f"{anio + 1}-01-01"])[0]
    te = te_todo[tm[te_todo] < fin]
    if len(te) == 0:
        return f"{anio}: sin meses de prueba"
    ylog = np.log1p(y)
    XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
    XEv = np.nan_to_num(XE.to_numpy(float), nan=-1.0)
    meses_tr = np.unique(tm[tr])
    fold = np.minimum((np.searchsorted(meses_tr, tm[tr]) * 5) // len(meses_tr), 4)
    preds = {}
    B_tr, B_te = [], []
    for m in MIEMBROS:
        fab = lambda m=m: bm.construir("reg", m, hp[m], X.columns, n_jobs=1)  # noqa: E731
        oof = np.zeros(len(tr))
        for f in range(5):
            mod = fab().fit(X.iloc[tr[fold != f]], y[tr[fold != f]])
            oof[fold == f] = np.log1p(np.clip(mod.predict(X.iloc[tr[fold == f]]), 0, None))
        p = np.clip(fab().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, None)
        if m in ("XGBoost", "Random Forest"):
            preds[m] = p                                     # rivales: el mismo modelo optimizado, reentrenado en el origen
        B_tr.append(oof); B_te.append(np.log1p(p))
    Xm_, W = disenar(X, XE, tr)
    core_oof = np.zeros(len(tr))
    for f in range(5):
        core_oof[fold == f], _ = ZINB().fit(Xm_[tr][fold != f], W[tr][fold != f], y[tr][fold != f]).predecir(Xm_[tr][fold == f], W[tr][fold == f])
    core_te, _ = ZINB().fit(Xm_[tr], W[tr], y[tr]).predecir(Xm_[te], W[te])
    bz_oof = np.zeros(len(tr))
    for f in range(5):
        bz_oof[fold == f], _ = BoostingZINB().fit(XEv[tr][fold != f], y[tr][fold != f], tm[tr][fold != f]).predecir(XEv[tr][fold == f])
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
    corte = meses_tr[int(len(meses_tr) * 0.8)]
    a, b = tm[tr] < corte, tm[tr] >= corte
    errs = {l2: np.mean((CombinadorSoftmax(l2).fit(Z[tr][a], B_tr[a], ylog[tr][a]).predict(Z[tr][b], B_tr[b]) - ylog[tr][b]) ** 2)
            for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
    comb = CombinadorSoftmax(min(errs, key=errs.get)).fit(Z[tr], B_tr, ylog[tr])
    preds["A+B"] = np.expm1(comb.predict(Z[te], B_te)).clip(0)
    np.savez(ruta, y=y[te], tm=tm[te], celda=cc[te], **{k.replace(" ", "_").replace("+", "mas"): v for k, v in preds.items()})
    return f"{anio}: {len(te)} filas, RMSLE A+B {ev.rmsle(y[te], preds['A+B']):.4f} | XGB {ev.rmsle(y[te], preds['XGBoost']):.4f} | RF {ev.rmsle(y[te], preds['Random Forest']):.4f} ({time.time() - t0:.0f} s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=2)
    a = ap.parse_args()
    t00 = time.time()
    print("Afinando hiperparámetros con datos < 2016...", flush=True)
    hp = afinar()
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(un_origen)(y_, hp) for y_ in ANIOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    partes = [np.load(OUT / f"origen_{y_}.npz") for y_ in ANIOS if (OUT / f"origen_{y_}.npz").exists()]
    y = np.concatenate([p["y"] for p in partes]); tm = np.concatenate([p["tm"] for p in partes])
    P = {k: np.concatenate([p[k] for p in partes]) for k in ("AmasB", "XGBoost", "Random_Forest")}
    _, L_ab = ev.perdida_mensual(y, P["AmasB"], tm)
    filas = []
    for r in ("XGBoost", "Random_Forest"):
        _, L_r = ev.perdida_mensual(y, P[r], tm)
        d = ev.diebold_mariano(L_r, L_ab)
        filas.append({"rival": r, "RMSLE_rival": ev.rmsle(y, P[r]), "RMSLE_AB": ev.rmsle(y, P["AmasB"]), **d})
    res = pd.DataFrame(filas)
    from src.eda import holm
    res["p_Holm"] = holm(res.p.to_numpy())
    res.to_csv(OUT / "resultado_origen_movil.csv", index=False)
    print("\n", res.round(4).to_string(index=False), f"\nTotal {(time.time() - t00) / 60:.1f} min")
