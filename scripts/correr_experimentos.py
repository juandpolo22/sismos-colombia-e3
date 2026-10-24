"""Corre las 140 combinaciones del Entregable 2 (28 de regresión + 112 de clasificación) con validación
anidada temporal y guarda un punto de control por combinación.

Uso (desde la raíz del proyecto):
    python scripts/correr_experimentos.py --tarea reg                  # 28 combinaciones de regresión (completas)
    python scripts/correr_experimentos.py --tarea clf --fase 1         # tamizaje de clasificación (26 pares, barato)
    python scripts/correr_experimentos.py --tarea clf --fase 2         # validación anidada de los pares que pasan
    python scripts/correr_experimentos.py --tarea clf --fase todo      # fase 1 y luego fase 2 (recomendado)
    python scripts/correr_experimentos.py --trabajos 4                 # número de procesos en paralelo
    python scripts/correr_experimentos.py --tabla                      # solo recompila la tabla maestra

La reducción del diseño de clasificación (exclusiones a priori + tamizaje) está justificada en el capítulo 13.

Cada combinación se guarda en ``resultados/benchmark/combos/*.pkl``; si existe, no se repite (punto de
control). La tabla maestra queda en ``resultados/benchmark/experimentos_maestro.csv``.
"""
import argparse
import json
import pickle
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

CARPETA = cfg.RESULTADOS / "benchmark" / "combos"
CARPETA.mkdir(parents=True, exist_ok=True)
# costo relativo estimado en el piloto (para repartir primero las combinaciones largas entre los trabajadores)
COSTO = {"Random Forest": 20, "XGBoost": 2, "SVM": 2, "SVR": 2, "KNN": 1.5}


def nombre_archivo(t, m, b, o):
    limpio = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)  # noqa: E731
    return CARPETA / f"{t}__{limpio(m)}__{limpio(b)}__{limpio(o)}.pkl"


TAMIZAJE = cfg.RESULTADOS / "benchmark" / "tamizaje_clasificacion.csv"


def combinaciones(tarea=None, solo_fase2=False):
    out = []
    if tarea in (None, "reg"):
        out += [("reg", m, "—", o) for m in bm.MODELOS_REG for o in bm.OPTIMIZADORES]
    if tarea in (None, "clf"):
        pares = [(m, b) for m in bm.MODELOS_CLF for b in bm.BALANCEOS]
        if solo_fase2:
            sel = bm.seleccionar_fase2(pd.read_csv(TAMIZAJE))
            ok = set(zip(sel[sel.pasa_fase2].modelo, sel[sel.pasa_fase2].balanceo))
            pares = [p for p in pares if p in ok]
        out += [("clf", m, b, o) for m, b in pares for o in bm.OPTIMIZADORES]
    return out


def fase1(trabajos):
    pares = [(m, b) for m in bm.MODELOS_CLF for b in bm.BALANCEOS if (m, b) not in bm.EXCLUIDAS_A_PRIORI]
    hechos = pd.read_csv(TAMIZAJE) if TAMIZAJE.exists() else pd.DataFrame(columns=["modelo", "balanceo"])
    pend = [p for p in pares if p not in set(zip(hechos.modelo, hechos.balanceo))]
    print(f"Fase 1 (tamizaje): {len(pend)} pares pendientes de {len(pares)}", flush=True)

    def uno(m, b):
        cfg.fijar_semillas()
        ev_, X, info, tr, gp, te = datos.panel_completo()
        return bm.tamizaje_par(m, b, X, info["y"].astype(float), info["tm"], tr)

    filas = list(hechos.to_dict("records"))
    for r in Parallel(n_jobs=trabajos, backend="loky", return_as="generator")(delayed(uno)(m, b) for m, b in pend):
        filas.append(r)
        pd.DataFrame(filas).to_csv(TAMIZAJE, index=False)              # punto de control tras cada par
        print(time.strftime("%H:%M:%S"), r["modelo"], "|", r["balanceo"], "| AUC", round(r["AUC_tamizaje"], 4), f"({r['t_total_s']:.0f} s)", flush=True)


def una(combo, presupuesto):
    ruta = nombre_archivo(*combo)
    if ruta.exists():
        return "ya existía"
    cfg.fijar_semillas()
    ev_, X, info, tr, gp, te = datos.panel_completo()
    r = bm.correr_seguro(*combo, X, info["y"].astype(float), info["tm"], tr, te, presupuesto=presupuesto)
    r["fecha_fin"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ruta, "wb") as f:
        pickle.dump(r, f)
    return f"{combo}: {r['estado']} {r.get('externo_media', '')} ({r.get('t_total_s', 0):.0f} s)"


def tabla_maestra():
    filas = []
    sel = bm.seleccionar_fase2(pd.read_csv(TAMIZAJE)).set_index(["modelo", "balanceo"]) if TAMIZAJE.exists() else None
    for combo in combinaciones():
        ruta = nombre_archivo(*combo)
        base = dict(zip(["tarea", "modelo", "balanceo", "optimizador"], combo))
        par = (combo[1], combo[2])
        if not ruta.exists():
            if combo[0] == "clf" and par in bm.EXCLUIDAS_A_PRIORI:
                filas.append({**base, "estado": "descartada a priori", "error": bm.EXCLUIDAS_A_PRIORI[par]})
            elif combo[0] == "clf" and sel is not None and par in sel.index and not sel.loc[par, "pasa_fase2"]:
                filas.append({**base, "estado": "descartada en tamizaje", "error": sel.loc[par, "motivo"],
                              "AUC_tamizaje": sel.loc[par, "AUC_tamizaje"]})
            else:
                filas.append({**base, "estado": "pendiente"})
            continue
        r = pickle.load(open(ruta, "rb"))
        fila = {**base, "estado": r["estado"], "error": r.get("error", "")}
        if combo[0] == "clf":
            if par in bm.EXCLUIDAS_A_PRIORI:
                fila["decision_previa"] = "descartada a priori (se ejecutó igual)"
            elif sel is not None and par in sel.index and not sel.loc[par, "pasa_fase2"]:
                fila["decision_previa"] = "descartada en tamizaje (se ejecutó igual)"
                fila["AUC_tamizaje"] = sel.loc[par, "AUC_tamizaje"]
            else:
                fila["decision_previa"] = "pasa tamizaje"
                if sel is not None and par in sel.index:
                    fila["AUC_tamizaje"] = sel.loc[par, "AUC_tamizaje"]
        if r["estado"] == "completado":
            ext = pd.DataFrame(r["externo"])
            met = [c for c in ext.columns if c not in ("pliegue", "mejor_interno", "t_ajuste_s", "t_inferencia_s")]
            fila.update({"metrica_principal": r["metrica_principal"], "externo_media": r["externo_media"], "externo_desv": r["externo_desv"],
                         **{f"{c}_media": ext[c].mean() for c in met}, **{f"{c}_desv": ext[c].std(ddof=1) for c in met},
                         "params_finales": json.dumps(r["params_finales"], default=str), "evaluaciones": r["evaluaciones"],
                         "t_total_s": r["t_total_s"], "t_por_evaluacion_s": r["t_total_s"] / max(r["evaluaciones"], 1),
                         "t_ajuste_medio_s": ext.t_ajuste_s.mean(), "t_inferencia_medio_s": ext.t_inferencia_s.mean(),
                         "presupuesto": r["presupuesto"], "semilla": r["semilla"], "fecha_fin": r.get("fecha_fin", "")})
        filas.append(fila)
    t = pd.DataFrame(filas)
    t.to_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarea", choices=["reg", "clf"])
    ap.add_argument("--presupuesto", type=int, default=15)
    ap.add_argument("--trabajos", type=int, default=2)
    ap.add_argument("--tabla", action="store_true")
    ap.add_argument("--fase", choices=["1", "2", "todo"], default="todo", help="solo para --tarea clf")
    ap.add_argument("--todas", action="store_true", help="clasificación: ejecuta también las 28 combinaciones descartadas (140 en total)")
    a = ap.parse_args()
    if a.tarea == "clf" and a.fase in ("1", "todo") and not a.tabla:
        fase1(a.trabajos)
    if a.tarea == "clf" and a.fase == "1":
        a.tabla = True
    if not a.tabla:
        pend = [c for c in combinaciones(a.tarea, solo_fase2=(a.tarea == "clf" and not a.todas)) if not nombre_archivo(*c).exists()]
        pend.sort(key=lambda c: -COSTO.get(c[1], 1) * (2 if c[2] in ("SMOTE", "ADASYN") else 1))
        print(f"{len(pend)} combinaciones pendientes; {a.trabajos} trabajadores (backend loky, 1 hilo por modelo)", flush=True)
        for msg in Parallel(n_jobs=a.trabajos, backend="loky", return_as="generator", verbose=0)(delayed(una)(c, a.presupuesto) for c in pend):
            print(time.strftime("%H:%M:%S"), msg, flush=True)
    t = tabla_maestra()
    print(t.estado.value_counts().to_string())
