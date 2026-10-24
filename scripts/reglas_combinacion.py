"""Contraste de la regla de combinación propia (A) con reglas estándar sobre los mismos miembros (guía E3, Cuadro 1).

Reglas, todas en escala log1p con las predicciones fuera de pliegue cronológicas del entrenamiento de cada origen:
* mejor miembro individual (menor error fuera de pliegue);
* promedio simple;
* pesos por error global: w_j ∝ 1 / ECM_j fuera de pliegue;
* stacking lineal: mínimos cuadrados no negativos (Breiman, 1996) de log(1 + y) sobre los pronósticos;
cada una sin y con la misma calibración final de A+B (recta estimada con el 20 % final de los meses). Se comparan con
A+B (misma corrida) con Diebold-Mariano HLN sobre la pérdida mensual y corrección de Holm.

Requiere:  python scripts/origen_movil_final.py --matrices
Uso:       python scripts/reglas_combinacion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src import config as cfg, eda, evaluacion as ev

DIR = cfg.RESULTADOS / "origen_movil" / "matrices"
ANIOS = range(2021, 2027)


def reglas(B, l, Bp):
    """Pronósticos log1p en prueba de cada regla ajustada con (B, l)."""
    ecm = np.mean((B - l[:, None]) ** 2, axis=0)
    w_inv = (1 / ecm) / (1 / ecm).sum()
    w_nnls, _ = nnls(np.column_stack([np.ones(len(B)), B]), l)
    return {"mejor miembro individual": Bp[:, np.argmin(ecm)], "promedio simple": Bp.mean(1),
            "pesos por error global (1/ECM)": Bp @ w_inv, "stacking lineal (NNLS)": np.column_stack([np.ones(len(Bp)), Bp]) @ w_nnls}


def main():
    filas, ys, tms, preds = [], [], [], {}
    for a in ANIOS:
        z = np.load(DIR / f"origen_{a}.npz")
        B, l, Bp, tm = z["B_ajuste"], z["ylog_ajuste"], z["B_prueba"], z["tm_ajuste"]
        ini, fin = tm < z["corte_calibracion"], tm >= z["corte_calibracion"]
        crudo = reglas(B, l, Bp)
        parcial = reglas(B[ini], l[ini], B[fin])
        for k, p in crudo.items():
            c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(fin.sum()), parcial[k]]), l[fin], rcond=None)[0]
            preds.setdefault(k, []).append(np.expm1(np.clip(p, 0, None)))
            preds.setdefault(k + " + calibración", []).append(np.expm1(np.clip(c0 + c1 * p, 0, None)))
        preds.setdefault("A+B v2 (regla propia)", []).append(z["AmasB"])
        preds.setdefault("A sin calibración", []).append(z["AmasB_sin_calibrar"])
        ys.append(z["y"]); tms.append(z["tm"])
    y, tm = np.concatenate(ys), np.concatenate(tms)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    _, Lab = ev.perdida_mensual(y, P["A+B v2 (regla propia)"], tm)
    for k, p in P.items():
        _, L = ev.perdida_mensual(y, p, tm)
        d = ev.diebold_mariano(L, Lab) if k != "A+B v2 (regla propia)" else {"p": np.nan, "dif_media": 0.0}
        filas.append({"regla": k, "RMSLE 2021–2026": ev.rmsle(y, p), "dif. pérdida vs A+B": d["dif_media"], "p DM": d["p"]})
    T = pd.DataFrame(filas)
    m = T["p DM"].notna()
    T.loc[m, "p Holm"] = eda.holm(T.loc[m, "p DM"].to_numpy())
    T = T.sort_values("RMSLE 2021–2026")
    (cfg.RESULTADOS / "e3").mkdir(exist_ok=True)
    T.to_csv(cfg.RESULTADOS / "e3" / "reglas_combinacion.csv", index=False)
    print(T.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
