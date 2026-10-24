"""Origen móvil 2021–2026 para la clasificación: 26 combinaciones (7 modelos × 4 balanceos, sin los 2 pares equivalentes).

Reducción fijada antes de correr, con evidencia anterior a 2021 (validación anidada hasta 2020):
* Se omiten KNN y Naive Bayes con class_weight: equivalentes a «sin balanceo» en ROC-AUC (cap. 13.6 y 14.7).
* Un solo optimizador, Bayesiano (TPE): los cuatro optimizadores dan la misma métrica externa (Friedman,
  cap. 14.2) y TPE es el más eficiente por unidad de presupuesto (cap. 14.4).

Mismo protocolo que el origen móvil de regresión (scripts/origen_movil_benchmark.py):
* Hiperparámetros: TPE con 15 evaluaciones y validación temporal interna de 3 pliegues, UNA sola vez, con el
  panel de corte 2016 (solo datos anteriores a 2016). Métrica: ROC-AUC.
* Orígenes anuales 2021..2026: cada combinación se reentrena con todo el pasado (gap de 1 mes) y se pronostican
  los meses objetivo del año. Las filas coinciden con las de A+B en el origen móvil.
* El balanceo (SMOTE, ADASYN, class_weight) va dentro del Pipeline, así que solo se aplica a los datos de ajuste.

Reanudable: cada afinamiento y cada (año, combinación) se guarda en su propio archivo.
Uso:  python scripts/origen_movil_clasificacion.py --trabajos N     (N = núcleos físicos − 1; con 8 GB de RAM, máx. 4)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src import benchmark as bm
from src import config as cfg
from src import datos

OUT = cfg.RESULTADOS / "origen_movil"
HP = OUT / "hp_clf"
PR = OUT / "pred_clf"
HP.mkdir(parents=True, exist_ok=True); PR.mkdir(parents=True, exist_ok=True)
ANIOS = list(range(2021, 2027))
COMBOS = [(m, b, "Bayesiano (TPE)") for m in bm.MODELOS_CLF for b in bm.BALANCEOS if (m, b) not in bm.EXCLUIDAS_A_PRIORI]


def slug(m, b, o):
    return f"{m}__{b}__{o}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def panel(corte):
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, corte)
    tr, gp, te = datos.particion(info)
    return X, info, tr, te


def afinar(m, b, o):
    ruta = HP / f"{slug(m, b, o)}.json"
    if ruta.exists():
        return f"{m} / {b} / {o}: ya existía"
    t0 = time.time()
    cfg.fijar_semillas()
    X, info, tr, te = panel("2016-01-01")
    yb, tm = (info["y"] >= 1).astype(int), info["tm"]
    internos = bm.pliegues_internos(tm[tr], 3)

    def f(params):
        r = []
        for a, v in internos:
            mod = bm.construir("clf", m, params, X.columns, b, n_jobs=1).fit(X.iloc[tr[a]], yb[tr[a]])
            r.append(roc_auc_score(yb[tr[v]], bm.puntaje_clf(mod, X.iloc[tr[v]])))
        return float(np.mean(r))
    try:
        params, val, _ = bm.Optimizador(o, f, "clf", m, 15, cfg.SEMILLA).correr()
    except Exception as e:  # noqa: BLE001
        json.dump({"error": f"{type(e).__name__}: {e}"}, open(ruta, "w"), indent=2)
        return f"{m} / {b} / {o}: ERROR {type(e).__name__}: {e}"
    json.dump({"params": params, "AUC_CV": val, "t_s": time.time() - t0}, open(ruta, "w"), indent=2, default=float)
    return f"{m} / {b} / {o}: AUC CV {val:.4f} ({time.time() - t0:.0f} s)"


def un_origen(anio, m, b, o):
    ruta = PR / f"{anio}__{slug(m, b, o)}.npy"
    if ruta.exists():
        return None
    hp = json.load(open(HP / f"{slug(m, b, o)}.json"))
    if "params" not in hp:
        return f"{anio} {m} / {b} / {o}: sin hiperparámetros (error en el afinamiento)"
    cfg.fijar_semillas()
    X, info, tr, te_todo = panel(f"{anio}-01-01")
    yb, tm = (info["y"] >= 1).astype(int), info["tm"]
    fin = datos.indice_mes([f"{anio + 1}-01-01"])[0]
    te = te_todo[tm[te_todo] < fin]
    ref = np.load(OUT / f"origen_{anio}.npz")
    assert np.array_equal(ref["tm"], tm[te]) and np.array_equal(ref["y"], info["y"][te]), "filas distintas a las de A+B"
    mod = bm.construir("clf", m, hp["params"], X.columns, b, n_jobs=1).fit(X.iloc[tr], yb[tr])
    np.save(ruta, bm.puntaje_clf(mod, X.iloc[te]))
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabajos", type=int, default=2)
    a = ap.parse_args()
    t00 = time.time()
    print(f"1) Afinando {len(COMBOS)} combinaciones con datos < 2016...", flush=True)
    orden = sorted(COMBOS, key=lambda c: (c[0] not in ("Random Forest", "SVM", "XGBoost"), c))   # las lentas primero
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(afinar)(*c) for c in orden):
        print(time.strftime("%H:%M:%S"), msg, flush=True)
    print("2) Orígenes anuales 2021-2026...", flush=True)
    tareas = [(y_, *c) for y_ in ANIOS for c in orden]
    hechas = 0
    for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator")(delayed(un_origen)(*t) for t in tareas):
        hechas += 1
        if msg:
            print(time.strftime("%H:%M:%S"), msg, flush=True)
        if hechas % 26 == 0:
            print(time.strftime("%H:%M:%S"), f"{hechas}/{len(tareas)} ajustes", flush=True)
    faltan = [t for t in tareas if not (PR / f"{t[0]}__{slug(*t[1:])}.npy").exists()]
    print(f"\nTerminado en {(time.time() - t00) / 60:.1f} min. Ajustes faltantes: {len(faltan)}", flush=True)
