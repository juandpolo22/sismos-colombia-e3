"""Decisión sobre los hiperparámetros de B con la validación previa (< 2021), sin usar 2021–2026.

Regla (fijada antes de ejecutar este script):
    Se arma A+B con la B optimizada por TPE (scripts/optimizar_B_tpe.py) y se evalúa en el 20 % final de los meses de
    entrenamiento anteriores a 2021, con el mismo procedimiento que la validación previa del modelo actual. Se adopta la
    B optimizada solo si A+B con la B nueva tiene menor RMSLE que (i) el A+B actual y (ii) XGBoost · AG, y la diferencia
    es significativa en ambos casos (Diebold-Mariano HLN sobre la pérdida mensual, p < 0,05). Si no, se conserva la B
    actual (ν = 0,08, profundidad 3, 50 filas por hoja).

Uso:  python scripts/comparar_B_validacion.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import config as cfg, datos, evaluacion as ev
from src.propuesto import BoostingZINB, CombinadorSoftmax, pliegues_cronologicos

CACHE = cfg.RESULTADOS / "propuesto" / "cronologico"
OUT = cfg.RESULTADOS / "e3"
MIEMBROS = ["XGBoost__Grid", "XGBoost__Random", "XGBoost__Bayesiano_TPE", "XGBoost__Genético_DEAP",
            "Random_Forest__Grid", "Random_Forest__Random", "Random_Forest__Bayesiano_TPE", "Random_Forest__Genético_DEAP",
            "KNN__Random", "Ridge__Genético_DEAP", "ZINB"]


def main():
    t0 = time.time()
    cfg.fijar_semillas()
    opt = json.load(open(OUT / "optimizacion_B_tpe.json"))
    nuevo, actual = opt["mejor"], opt["actual"]
    ev_, X, info, tr, gp, te = datos.panel_completo()
    y, tm = info["y"].astype(float), info["tm"]
    ylog = np.log1p(y)
    XE = pd.concat([X.reset_index(drop=True), datos.variables_excitacion(ev_, info)], axis=1)
    XEv = np.nan_to_num(XE.to_numpy(float), nan=-1.0)
    PL, MASK = pliegues_cronologicos(tm[tr], k=5, gap=cfg.GAP_MESES)

    def b_oof(par):
        ruta = CACHE / f"miembro_BZINB_lr{par['lr']:.4f}_d{par['depth']}_m{par['min_leaf']}.npz"
        if ruta.exists():
            return np.load(ruta)["oof"]
        oof = np.full(len(tr), np.nan)
        for a, b in PL:
            oof[b], _ = BoostingZINB(**par).fit(XEv[tr][a], y[tr][a], tm[tr][a]).predecir(XEv[tr][b])
        np.savez(ruta, oof=oof)
        print(f"  B {par}: predicciones fuera de pliegue listas [{time.time() - t0:.0f} s]", flush=True)
        return oof

    base = [np.load(CACHE / f"miembro_{n}.npz")["oof"] for n in MIEMBROS]
    persist, estac = np.log1p(info["persist"][tr]), np.log1p(info["estacional"][tr])
    C, cc, tt = info["C"], info["cc"], info["tt"]
    hist = np.stack([C[cc, tt - j] for j in range(12)], axis=1)
    ctx = PCA(n_components=4, random_state=cfg.SEMILLA).fit(np.log1p(hist[tr])).transform(np.log1p(hist))
    lg = lambda n: np.log1p(X[n].to_numpy(float))  # noqa: E731
    Zr = np.column_stack([lg("n_1m"), lg("n_3m"), lg("n_12m"), lg("n35_12m"), lg("vec_1m"), XE["exc_a0.5_s30_r0.4"].to_numpy(),
                          XE["exc_a0.9_s60_r0.8"].to_numpy(), ctx, (hist == 0).mean(1),
                          np.nan_to_num(X["mag_max_12m"].to_numpy(float), nan=cfg.M_MIN), X["sen_mes"].to_numpy(), X["cos_mes"].to_numpy()])
    Zs = (Zr - Zr[tr].mean(0)) / (Zr[tr].std(0) + 1e-9)
    Z = np.hstack([np.ones((len(Zs), 1)), Zs])[tr]
    FILAS = np.where(MASK)[0]
    meses_m = np.unique(tm[tr][FILAS]); corte_v = meses_m[int(len(meses_m) * 0.8)]
    ia, iv = FILAS[tm[tr][FILAS] < corte_v], FILAS[tm[tr][FILAS] >= corte_v]
    m_a = tm[tr][ia]; c = np.unique(m_a)[int(len(np.unique(m_a)) * 0.8)]
    a, b = ia[m_a < c], ia[m_a >= c]
    yt = ylog[tr]

    def ab_validacion(bcol):
        B = np.column_stack(base + [bcol, persist, estac])
        errs = {l2: np.mean((CombinadorSoftmax(l2).fit(Z[a], B[a], yt[a]).predict(Z[b], B[b]) - yt[b]) ** 2) for l2 in (1e-3, 1e-2, 1e-1, 1.0)}
        l2 = min(errs, key=errs.get)
        pb = CombinadorSoftmax(l2).fit(Z[a], B[a], yt[a]).predict(Z[b], B[b])
        c0, c1 = np.linalg.lstsq(np.column_stack([np.ones(len(b)), pb]), yt[b], rcond=None)[0]
        return np.expm1(np.clip(c0 + c1 * CombinadorSoftmax(l2).fit(Z[ia], B[ia], yt[ia]).predict(Z[iv], B[iv]), 0, None))

    yv, tv = y[tr][iv], tm[tr][iv]
    p_act = ab_validacion(b_oof(actual))
    p_new = ab_validacion(b_oof(nuevo))
    p_xgb = np.expm1(np.load(CACHE / "miembro_XGBoost__Genético_DEAP.npz")["oof"][iv])
    _, L_new = ev.perdida_mensual(yv, p_new, tv)
    filas = []
    for nom, p in (("A+B actual (B 0,08 · 3 · 50)", p_act), ("XGBoost · AG", p_xgb)):
        _, L = ev.perdida_mensual(yv, p, tv)
        d = ev.diebold_mariano(L, L_new)
        filas.append({"rival": nom, "RMSLE rival": ev.rmsle(yv, p), "RMSLE A+B con B optimizada": ev.rmsle(yv, p_new),
                      "dif. pérdida (rival − nuevo)": d["dif_media"], "p DM": d["p"],
                      "nuevo mejor y significativo": bool(d["dif_media"] > 0 and d["p"] < 0.05)})
    T = pd.DataFrame(filas)
    adoptar = bool(T["nuevo mejor y significativo"].all())
    T.to_csv(OUT / "decision_B_validacion.csv", index=False)
    json.dump({"B_actual": actual, "B_optimizada": nuevo, "adoptar_B_optimizada": adoptar, "meses_validacion": int(len(np.unique(tv))),
               "regla": "adoptar solo si A+B con B optimizada supera con DM p < 0,05 al A+B actual y a XGBoost · AG en la validación previa (< 2021)"},
              open(OUT / "decision_B_validacion.json", "w"), indent=1, ensure_ascii=False)
    print(T.round(5).to_string(index=False))
    print("DECISIÓN:", "adoptar la B optimizada" if adoptar else "conservar la B actual", f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
