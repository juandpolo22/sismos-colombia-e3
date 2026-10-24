"""Origen móvil para las 28 combinaciones benchmark de regresión (7 modelos × 4 optimizadores) frente a A+B.

Mismo protocolo que scripts/origen_movil.py:
* Hiperparámetros: cada optimizador (Grid, Random, TPE, Genético) con 15 evaluaciones y validación
  temporal interna de 3 pliegues, UNA sola vez, con el panel de corte 2016 (solo datos < 2016).
* Orígenes anuales 2016..2026: se reentrena cada combinación con todo el pasado y se pronostica el año.
* Las predicciones de A+B se toman de resultados/origen_movil/origen_{año}.npz (no se recalculan).
* Diebold-Mariano (HLN) de cada combinación vs A+B sobre los 129 meses, con Holm sobre las 28.

Reanudable: cada afinamiento y cada (año, combinación) se guarda en su propio archivo.
Uso:  python scripts/origen_movil_benchmark.py [--trabajos 2]
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

from src import benchmark as bm
from src import config as cfg
from src import datos
from src import evaluacion as ev

OUT = cfg.RESULTADOS / "origen_movil"
HP = OUT / "hp_bench"
PR = OUT / "pred_bench"
HP.mkdir(parents=True, exist_ok=True); PR.mkdir(parents=True, exist_ok=True)
ANIOS = list(range(2016, 2027))
COMBOS = [(m, o) for m in bm.MODELOS_REG for o in bm.OPTIMIZADORES]


def slug(m, o):
    return f"{m}__{o}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def panel(corte):
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, corte)
    tr, gp, te = datos.particion(info)
    return X, info, tr, te


def afinar(m, o):
    ruta = HP / f"{slug(m, o)}.json"
    if ruta.exists():
        return f"{m} / {o}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    X, info, tr, te = panel("2016-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    internos = bm.pliegues_internos(tm[tr], 3)

    def f(params):
        r = []
        for a, v in internos:
            mod = bm.construir("reg", m, params, X.columns, n_jobs=1).fit(X.iloc[tr[a]], y[tr[a]])
            r.append(-ev.rmsle(y[tr[v]], mod.predict(X.iloc[tr[v]])))
        return float(np.mean(r))
    params, val, _ = bm.Optimizador(o, f, "reg", m, 15, cfg.SEMILLA).correr()
    json.dump({"params": params, "RMSLE_CV": -val}, open(ruta, "w"), indent=2, default=float)
    return f"{m} / {o}: RMSLE CV {-val:.4f} ({time.time() - t0:.0f} s)"


def un_origen(anio):
    pend = [(m, o) for m, o in COMBOS if not (PR / f"{anio}__{slug(m, o)}.npy").exists()]
    if not pend:
        return f"{anio}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    X, info, tr, te_todo = panel(f"{anio}-01-01")
    y, tm = info["y"].astype(float), info["tm"]
    fin = datos.indice_mes([f"{anio + 1}-01-01"])[0]
    te = te_todo[tm[te_todo] < fin]
    ref = np.load(OUT / f"origen_{anio}.npz")
    assert np.array_equal(ref["tm"], tm[te]) and np.array_equal(ref["y"], y[te]), "filas distintas a las de A+B"
    for m, o in pend:
        p = json.load(open(HP / f"{slug(m, o)}.json"))["params"]
        mod = bm.construir("reg", m, p, X.columns, n_jobs=1).fit(X.iloc[tr], y[tr])
        np.save(PR / f"{anio}__{slug(m, o)}.npy", np.clip(mod.predict(X.iloc[te]), 0, None))
    return f"{anio}: {len(pend)} combinaciones ({time.time() - t0:.0f} s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=2)
    a = ap.parse_args()
    t00 = time.time()
    print("1) Afinando 28 combinaciones con datos < 2016...", flush=True)
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(afinar)(m, o) for m, o in COMBOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    print("2) Orígenes anuales...", flush=True)
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(un_origen)(y_) for y_ in ANIOS):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    partes = [np.load(OUT / f"origen_{y_}.npz") for y_ in ANIOS]
    y = np.concatenate([p["y"] for p in partes]); tm = np.concatenate([p["tm"] for p in partes])
    ab = np.concatenate([p["AmasB"] for p in partes])
    _, L_ab = ev.perdida_mensual(y, ab, tm)
    filas = []
    for m, o in COMBOS:
        pr = np.concatenate([np.load(PR / f"{y_}__{slug(m, o)}.npy") for y_ in ANIOS])
        _, L = ev.perdida_mensual(y, pr, tm)
        d = ev.diebold_mariano(L, L_ab)
        filas.append({"modelo": m, "optimizador": o, "RMSLE_rival": ev.rmsle(y, pr), "RMSLE_AB": ev.rmsle(y, ab), **d})
    res = pd.DataFrame(filas)
    from src.eda import holm
    res["p_Holm"] = holm(res.p.to_numpy())
    res["veredicto"] = np.where(res.p_Holm < 0.05, np.where(res.dif_media > 0, "A+B gana", "A+B pierde"), "empate")
    res.to_csv(OUT / "resultado_origen_movil_28.csv", index=False)
    print("\n", res.round(4).to_string(index=False), "\n", res.veredicto.value_counts().to_string(), f"\nTotal {(time.time() - t00) / 60:.1f} min")
