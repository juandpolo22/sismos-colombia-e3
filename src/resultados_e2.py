"""Carga de los resultados guardados del Entregable 2 (combinaciones benchmark y modelo propuesto)."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from . import config as cfg

BENCH = cfg.RESULTADOS / "benchmark"
COMBOS = BENCH / "combos"
PROP = cfg.RESULTADOS / "propuesto"
OM = cfg.RESULTADOS / "origen_movil"
E2 = cfg.RESULTADOS / "e2"
ABREV_OPT = {"Grid": "Grid", "Random": "Random", "Bayesiano (TPE)": "TPE", "Genético (DEAP)": "AG"}
ABREV_BAL = {"sin balanceo": "sin bal.", "SMOTE": "SMOTE", "ADASYN": "ADASYN", "class_weight": "c_weight"}


def maestro() -> pd.DataFrame:
    """Tabla maestra de las 140 combinaciones (estado, métricas del bucle externo, tiempos)."""
    return pd.read_csv(BENCH / "experimentos_maestro.csv")


def etiqueta(r) -> str:
    """Nombre corto de una combinación: modelo · optimizador (y · balanceo en clasificación)."""
    if r["tarea"] == "reg":
        return f"{r['modelo']} · {ABREV_OPT[r['optimizador']]}"
    return f"{r['modelo']} · {ABREV_BAL[r['balanceo']]} · {ABREV_OPT[r['optimizador']]}"


def cargar(tarea: str) -> list:
    """Lista de diccionarios de las combinaciones completadas de una tarea (con historiales y predicciones)."""
    out = []
    for f in sorted(COMBOS.glob(f"{tarea}__*.pkl")):
        r = pickle.load(open(f, "rb"))
        if r.get("estado") == "completado":
            r["etiqueta"] = etiqueta(r)
            out.append(r)
    orden = {m: i for i, m in enumerate(maestro().modelo.unique())}
    return sorted(out, key=lambda r: (orden.get(r["modelo"], 99), r.get("balanceo", ""), list(ABREV_OPT).index(r["optimizador"])))


def predicciones_prueba(combos) -> pd.DataFrame:
    """Tabla de pronósticos en el periodo de prueba, una columna por combinación."""
    return pd.DataFrame({r["etiqueta"]: r["pred_prueba"] for r in combos})


def ab2_prueba() -> dict:
    """Validación previa del modelo propuesto final (20 % final de los meses de entrenamiento, < 2021):
    ``idx_val`` (filas del panel) y ``oof_val`` (pronóstico). Ver scripts/validacion_previa_final.py."""
    z = np.load(PROP / "AB_final_validacion.npz")
    return {k: z[k] for k in z.files}


OM_DESDE = "2021-01-01"   # periodo de evaluación del origen móvil = el mismo de la partición fija (cap. 13.11)


def origen_movil(desde: str = OM_DESDE) -> dict:
    """Predicciones del origen móvil 2021–2026: y, tm, el modelo propuesto final («A+B v2», con calibración;
    scripts/origen_movil_final.py) y las 28 combinaciones benchmark."""
    anios = ANIOS_OM
    p1 = [np.load(OM / f"origen_{a}.npz") for a in anios]
    pf = [np.load(OM / "final" / f"origen_{a}.npz") for a in anios]
    out = {"y": np.concatenate([p["y"] for p in p1]), "tm": np.concatenate([p["tm"] for p in p1]),
           "A+B v2": np.concatenate([p["AmasB"] for p in pf])}
    assert np.array_equal(out["tm"], np.concatenate([p["tm"] for p in pf]))
    from .benchmark import MODELOS_REG, OPTIMIZADORES
    for m in MODELOS_REG:
        for o in OPTIMIZADORES:
            slug = f"{m}__{o}".replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
            out[f"{m} · {ABREV_OPT[o]}"] = np.concatenate([np.load(OM / "pred_bench" / f"{a}__{slug}.npy") for a in anios])
    from .datos import indice_mes
    mask = out["tm"] >= indice_mes([desde])[0]
    return {k: v[mask] for k, v in out.items()}


def variantes_propuestas() -> dict:
    """Modelo propuesto final y las versiones de su ablación en el origen móvil 2021–2026 (para el SPA, cap. 21.7)."""
    f = [np.load(OM / "final" / f"origen_{a}.npz") for a in ANIOS_OM]
    cat = lambda k: np.concatenate([p[k] for p in f])  # noqa: E731
    return {"A+B v2 (final)": cat("AmasB"), "Sin calibración final": cat("AmasB_sin_calibrar"),
            "Sin B": cat("A_sin_B"), "Sin B ni núcleo ZINB": cat("A_sin_B_ni_ZINB")}


def params(r) -> str:
    """Hiperparámetros finales de una combinación como texto JSON, con los reales redondeados a 4 decimales."""
    p = r["params_finales"]
    return json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Origen móvil 2021–2026: evaluación definitiva (cap. 13.11)
# ---------------------------------------------------------------------------
ANIOS_OM = list(range(2021, 2027))
TPE = "Bayesiano (TPE)"


def _slug(*partes) -> str:
    return "__".join(partes).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")


def filas_origen_movil() -> dict:
    """Celda (``cc``), mes de las variables (``tt``) y prevalencia de la clase positiva en el entrenamiento de su
    origen, para cada fila del origen móvil 2021–2026, en el mismo orden que las predicciones. Se reconstruyen con el panel de cada origen y se guardan en caché."""
    ruta = OM / "filas_2021_2026.npz"
    if ruta.exists():
        z = np.load(ruta)
        return {k: z[k] for k in z.files}
    from . import datos
    ev_, n_m = datos.preparar_eventos(datos.cargar_catalogo())
    cc, tt, tm, prev = [], [], [], []
    for a in ANIOS_OM:
        X, info = datos.construir_panel(ev_, n_m, f"{a}-01-01")
        tr, gp, te = datos.particion(info)
        te = te[info["tm"][te] < datos.indice_mes([f"{a + 1}-01-01"])[0]]
        ref = np.load(OM / f"origen_{a}.npz")
        assert np.array_equal(ref["tm"], info["tm"][te])
        cc.append(info["cc"][te]); tt.append(info["tt"][te]); tm.append(info["tm"][te])
        prev.append(np.full(len(te), float((info["y"][tr] >= 1).mean())))
    out = {"cc": np.concatenate(cc), "tt": np.concatenate(tt), "tm": np.concatenate(tm),
           "prevalencia_entrenamiento": np.concatenate(prev), "celda_grados": np.array(info["celda"])}
    np.savez(ruta, **out)
    return out


def campeones_om(tarea: str) -> pd.DataFrame:
    """Campeón de cada familia en el origen móvil, elegido con la validación interna anterior a 2016
    (regresión: menor RMSLE_CV entre los 4 optimizadores; clasificación: mayor AUC_CV entre las técnicas de
    balanceo, todas con TPE)."""
    from .benchmark import EXCLUIDAS_A_PRIORI, MODELOS_CLF, MODELOS_REG, BALANCEOS, OPTIMIZADORES
    filas = []
    if tarea == "reg":
        for m in MODELOS_REG:
            for o in OPTIMIZADORES:
                v = json.load(open(OM / "hp_bench" / f"{_slug(m, o)}.json"))
                filas.append({"modelo": m, "variante": o, "etiqueta": f"{m} · {ABREV_OPT[o]}", "validación < 2016": v["RMSLE_CV"]})
        T = pd.DataFrame(filas)
        idx = T.groupby("modelo")["validación < 2016"].idxmin()
        return T.loc[idx].sort_values("validación < 2016").reset_index(drop=True)
    for m in MODELOS_CLF:
        for b in BALANCEOS:
            if (m, b) in EXCLUIDAS_A_PRIORI:
                continue
            v = json.load(open(OM / "hp_clf" / f"{_slug(m, b, TPE)}.json"))
            filas.append({"modelo": m, "variante": b, "etiqueta": f"{m} · {ABREV_BAL[b]}", "validación < 2016": v["AUC_CV"]})
    T = pd.DataFrame(filas)
    idx = T.groupby("modelo")["validación < 2016"].idxmax()
    return T.loc[idx].sort_values("validación < 2016", ascending=False).reset_index(drop=True)


def origen_movil_clf() -> dict:
    """Clasificación en el origen móvil 2021–2026: ``y`` (≥ 1 sismo), ``tm``, ``params`` y los puntajes de las
    26 combinaciones (7 modelos × 4 balanceos con TPE, sin los 2 pares equivalentes)."""
    from .benchmark import EXCLUIDAS_A_PRIORI, MODELOS_CLF, BALANCEOS
    partes = [np.load(OM / f"origen_{a}.npz") for a in ANIOS_OM]
    out = {"y": (np.concatenate([p["y"] for p in partes]) >= 1).astype(int), "tm": np.concatenate([p["tm"] for p in partes]),
           "combos": {}, "params": {}, "AUC_CV": {}}
    for m in MODELOS_CLF:
        for b in BALANCEOS:
            if (m, b) in EXCLUIDAS_A_PRIORI:
                continue
            s = _slug(m, b, TPE)
            etq = f"{m} · {ABREV_BAL[b]}"
            out["combos"][etq] = np.concatenate([np.load(OM / "pred_clf" / f"{a}__{s}.npy") for a in ANIOS_OM])
            hp = json.load(open(OM / "hp_clf" / f"{s}.json"))
            out["params"][etq], out["AUC_CV"][etq] = hp["params"], hp["AUC_CV"]
    return out


def ablacion_om() -> dict:
    """Ablación del modelo propuesto final en el origen móvil 2021–2026 (scripts/origen_movil_final.py); todas las
    variantes con la misma calibración final."""
    partes = [np.load(OM / "final" / f"origen_{a}.npz") for a in ANIOS_OM]
    claves = ["y", "tm", "AmasB", "B_solo", "ZINB_solo", "A_sin_B", "A_sin_B_ni_ZINB"]
    out = {k: np.concatenate([p[k] for p in partes]) for k in claves}
    out["AmasB_v2"] = out.pop("AmasB")
    out["c"] = pd.DataFrame({"año": list(ANIOS_OM), "c0": [float(p["c0"]) for p in partes], "c1": [float(p["c1"]) for p in partes]})
    return out
