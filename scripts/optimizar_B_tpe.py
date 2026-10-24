"""Optimización bayesiana (TPE, Optuna) de los hiperparámetros del modelo B con datos anteriores a 2016.

Mismo protocolo que los benchmarks: panel con corte 2016, validación temporal interna de 3 pliegues
(src.benchmark.pliegues_internos), 15 evaluaciones y semilla global. La configuración usada hasta ahora
(ν = 0,08, profundidad 3, 50 filas por hoja) se evalúa como primer candidato; gana la de menor RMSLE en esta validación.
Ningún mes posterior a 2015 interviene en la elección.

Uso:  python scripts/optimizar_B_tpe.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import optuna
import pandas as pd

from src import benchmark as bm
from src import config as cfg
from src import datos
from src import evaluacion as ev
from src.propuesto import BoostingZINB

OUT = cfg.RESULTADOS / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ACTUAL = {"lr": 0.08, "depth": 3, "min_leaf": 50}


def main():
    cfg.fijar_semillas()
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    X, info = datos.construir_panel(ev_, n_m, "2016-01-01")
    tr, gp, te = datos.particion(info)
    XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
    XEv = np.nan_to_num(XE.to_numpy(float), nan=-1.0)
    y, tm = info["y"].astype(float), info["tm"]
    internos = bm.pliegues_internos(tm[tr], 3)
    t0 = time.time()

    def objetivo(trial):
        p = {"lr": trial.suggest_float("lr", 0.02, 0.3, log=True), "depth": trial.suggest_int("depth", 2, 6),
             "min_leaf": trial.suggest_int("min_leaf", 10, 300, log=True)}
        r = []
        for a, v in internos:
            ia, iv = tr[a], tr[v]
            pred, _ = BoostingZINB(**p).fit(XEv[ia], y[ia], tm[ia]).predecir(XEv[iv])
            r.append(ev.rmsle(y[iv], np.expm1(pred)))
        print(f"  {trial.number:2d} {p} RMSLE {np.mean(r):.5f} [{time.time() - t0:.0f} s]", flush=True)
        return float(np.mean(r))

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.SEMILLA))
    estudio.enqueue_trial(ACTUAL)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio.optimize(objetivo, n_trials=15)
    H = estudio.trials_dataframe()[["number", "value", "params_lr", "params_depth", "params_min_leaf"]]
    H.to_csv(OUT / "optimizacion_B_tpe.csv", index=False)
    mejor = estudio.best_params
    res = {"metodo": "Bayesiano (TPE, Optuna)", "evaluaciones": 15, "datos": "panel con corte 2016 (meses objetivo < 2016)",
           "validacion": "3 pliegues temporales internos", "actual": ACTUAL, "rmsle_actual": float(H.value.iloc[0]),
           "mejor": mejor, "rmsle_mejor": float(estudio.best_value),
           "decision": ("la configuración actual es también la mejor B sola" if mejor == ACTUAL else
                        "la mejor B sola en la validación < 2016 es la optimizada; la decisión final sobre A+B se toma en "
                        "scripts/comparar_B_validacion.py")}
    json.dump(res, open(OUT / "optimizacion_B_tpe.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
