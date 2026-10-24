"""Pruebas de independencia de los residuos de prueba.

Evaluación: origen móvil 2021–2026 (cap. 13.11). Modelos: A+B v2 y el campeón de cada uno de los 7 modelos de
regresión (elegido con la validación interna anterior a 2016).

Residuo de cada fila: r = log1p(ŷ) − log1p(y) (la escala de la métrica RMSLE).

1. Dependencia temporal de la serie mensual de residuos medios (69 meses):
   Escanciano-Lobato (portmanteau automático, robusto a heterocedasticidad), Breusch-Godfrey (12 rezagos) y BDS
   (dependencia no lineal, dimensiones 2 y 3).
2. Dependencia temporal dentro de cada celda: Escanciano-Lobato sobre la serie de residuos de cada una de las
   163 celdas; se reporta la fracción de celdas con p < 0,05 (se esperaría ≈ 5 % bajo independencia).
3. Dependencia espacial: I de Moran de los residuos entre celdas, mes a mes (8 vecinos, 499 permutaciones).
4. Serie de diferencias de pérdida mensual A+B − rival (la que usa Diebold-Mariano): Escanciano-Lobato y
   Breusch-Godfrey. Si hay autocorrelación, justifica el error estándar HAC (Newey-West) de la prueba DM.

Salidas: resultados/e2/independencia_residuos.csv, independencia_celdas.csv, independencia_moran.csv,
independencia_diferencias.csv.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as cfg
from src import datos
from src import estadistica as es
from src import evaluacion as ev

OUT = cfg.RESULTADOS / "e2"
OUT.mkdir(parents=True, exist_ok=True)
COMBOS = cfg.RESULTADOS / "benchmark" / "combos"


def archivo(m, o):
    limpio = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)  # noqa: E731
    return COMBOS / f"reg__{limpio(m)}__{limpio('—')}__{limpio(o)}.pkl"


def main():
    _, _, info, tr, gp, te = datos.panel_completo()
    from src import resultados_e2 as r2
    O = r2.origen_movil(); Fi = r2.filas_origen_movil()
    y, tm, cc = O["y"], O["tm"], Fi["cc"]
    assert np.array_equal(Fi["tm"], tm)
    lat, lon = datos.centro_celda(cc, info["celda"])
    modelos = {"A+B (propuesto)": O["A+B v2"]}
    for _, f in r2.campeones_om("reg").iterrows():
        modelos[f"{f.modelo} ({f.variante})"] = O[f.etiqueta]

    filas, celdas, moran, difs = [], [], [], []
    meses = np.unique(tm)
    _, L_ab = ev.perdida_mensual(y, modelos["A+B (propuesto)"], tm)
    for nombre, p in modelos.items():
        res = np.log1p(np.clip(p, 0, None)) - np.log1p(y)
        serie = np.array([res[tm == m].mean() for m in meses])
        el, bg, bd = es.escanciano_lobato(serie), es.breusch_godfrey(serie, 12), es.bds_serie(serie)
        filas.append({"modelo": nombre, "RMSLE": ev.rmsle(y, p), "EL_estadistico": el["estadistico"], "EL_rezagos": el["rezagos"],
                      "EL_p": el["p"], "BG_LM": bg["LM"], "BG_p": bg["p"], "BDS_p_dim2": bd["p_por_dimension"][0],
                      "BDS_p_dim3": bd["p_por_dimension"][-1]})
        # por celda
        ps = []
        for c in np.unique(cc):
            s = cc == c
            orden = np.argsort(tm[s])
            e = res[s][orden]
            if np.std(e) > 0:
                ps.append(es.escanciano_lobato(e)["p"])
        ps = np.array(ps)
        celdas.append({"modelo": nombre, "celdas": len(ps), "frac_p<0.05": float((ps < 0.05).mean()),
                       "frac_p<0.05_Holm": float(es_holm(ps))})
        # espacial
        mo = es.moran_por_mes(res, tm, lat, lon, k=8, permutaciones=499)
        mo.insert(0, "modelo", nombre)
        moran.append(mo)
        # diferencias de pérdida (solo rivales)
        if nombre != "A+B (propuesto)":
            _, L = ev.perdida_mensual(y, p, tm)
            d = np.asarray(L_ab) - np.asarray(L)
            el, bg = es.escanciano_lobato(d), es.breusch_godfrey(d, 12)
            difs.append({"rival": nombre, "EL_p": el["p"], "EL_rezagos": el["rezagos"], "BG_p": bg["p"]})
        print(nombre, "listo", flush=True)

    pd.DataFrame(filas).to_csv(OUT / "independencia_residuos.csv", index=False)
    pd.DataFrame(celdas).to_csv(OUT / "independencia_celdas.csv", index=False)
    M = pd.concat(moran)
    M.to_csv(OUT / "independencia_moran.csv", index=False)
    pd.DataFrame(difs).to_csv(OUT / "independencia_diferencias.csv", index=False)
    print(pd.DataFrame(filas).round(4).to_string())
    print(pd.DataFrame(celdas).round(3).to_string())
    pcol = "p" if "p" in M.columns else [c for c in M.columns if c.startswith("p")][0]
    icol = "I" if "I" in M.columns else [c for c in M.columns if c not in ("modelo", "mes", pcol)][0]
    print(M.groupby("modelo").agg(I_mediana=(icol, "median"), frac_meses_p05=(pcol, lambda v: (v < 0.05).mean())).round(3).to_string())
    print(pd.DataFrame(difs).round(4).to_string())


def es_holm(p):
    """Fracción de celdas que siguen significativas tras Holm (control de comparaciones múltiples)."""
    p = np.sort(np.asarray(p))
    m = len(p)
    rechaza = 0
    for i, v in enumerate(p):
        if v <= 0.05 / (m - i):
            rechaza += 1
        else:
            break
    return rechaza / m if m else np.nan


if __name__ == "__main__":
    main()
