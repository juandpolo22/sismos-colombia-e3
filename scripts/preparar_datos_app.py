"""Prepara los datos livianos que lee el dashboard (data/app/). Se ejecuta una vez, fuera del servidor.

Requiere los artefactos de scripts/entrenar_artefactos.py para el pronóstico del mes siguiente y los SHAP.

Archivos que genera:
* panel.pkl.gz        panel celda × mes (variables, objetivo, identificadores) para el EDA y el formulario.
* oos.npz             predicciones fuera de muestra del origen móvil 2021–2026 (campeones y A+B) para diagnósticos.
* intervalos.json     cuantiles de los residuos log1p fuera de muestra por tramo de pronóstico (intervalo al 80 %).
* lideres.csv         tabla de líderes (mejor combinación de cada modelo).
* pronostico.csv      pronóstico del mes siguiente por celda y modelo, con intervalo.
* shap_global.json    importancia SHAP global por modelo (muestra del origen móvil).
* fondo_shap.pkl.gz   filas de referencia para las explicaciones locales.
* original.json       curva de convergencia de B, pesos medios de A y calibración.

Uso:  python scripts/preparar_datos_app.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from src import config as cfg
from src import panel, resultados_e2 as r2
from src.datos import VARIABLES_BASE
from src.original_model import COLUMNAS
from src.artefactos import slug as _slug

APP = cfg.DATOS / "app"
APP.mkdir(parents=True, exist_ok=True)
ART = cfg.RAIZ / "artifacts"


def main():
    cfg.fijar_semillas()
    # ---------------------------------------------------------------- panel
    P = panel.construir(incluir_siguiente=True)
    P.to_pickle(APP / "panel.pkl.gz")
    print("panel:", P.shape, "| mes a pronosticar:", P.fecha_obj.max().date(), flush=True)
    # ---------------------------------------------------------------- fuera de muestra 2021–2026
    O = r2.origen_movil()
    camp = r2.campeones_om("reg")
    claves = ["A+B v2"] + camp.etiqueta.tolist()
    cc = np.concatenate([np.load(cfg.RESULTADOS / "origen_movil" / f"origen_{a}.npz")["celda"] for a in r2.ANIOS_OM])
    tm_ = np.concatenate([np.load(cfg.RESULTADOS / "origen_movil" / f"origen_{a}.npz")["tm"] for a in r2.ANIOS_OM])
    cc = cc[tm_ >= O["tm"].min()]
    assert len(cc) == len(O["y"])
    np.savez_compressed(APP / "oos.npz", y=O["y"], tm=O["tm"], celda=cc,
                        **{f"m{i}": O[k] for i, k in enumerate(claves)}, nombres=np.array(claves))
    # intervalos: cuantiles 10 % y 90 % del residuo log1p por decil del pronóstico (conformal por tramos)
    inter = {}
    for k in claves:
        lp = np.log1p(O[k]); r = np.log1p(O["y"]) - lp
        bordes = np.unique(np.quantile(lp, np.linspace(0, 1, 11)))
        tramo = np.clip(np.searchsorted(bordes, lp, side="right") - 1, 0, len(bordes) - 2)
        inter[k] = {"bordes": bordes.tolist(), "q10": [float(np.quantile(r[tramo == j], 0.10)) for j in range(len(bordes) - 1)],
                    "q90": [float(np.quantile(r[tramo == j], 0.90)) for j in range(len(bordes) - 1)],
                    "cobertura_80": float(np.mean((r >= np.array([np.quantile(r[tramo == j], 0.10) for j in tramo])) &
                                                  (r <= np.array([np.quantile(r[tramo == j], 0.90) for j in tramo]))))}
    json.dump(inter, open(APP / "intervalos.json", "w"), indent=1)
    # ---------------------------------------------------------------- tablas estadísticas que muestra el dashboard
    for f in ("e2_ab_vs_campeones.csv", "e2_stats_mcs_origen_movil.csv", "e2_stats_gw_cw.csv", "e2_stats_spa.csv",
              "e2_stats_dm.csv", "e2_stats_familia_holm.csv", "e2_stats_independencia.csv", "e2_stats_diferencias_mensuales.csv",
              "e2_ab_ablacion.csv"):
        shutil.copy(cfg.TABLAS / f, APP / f)
    for f in ("reglas_combinacion.csv",):               # contraste con reglas de combinación estándar (scripts/reglas_combinacion.py)
        if (cfg.RESULTADOS / "e3" / f).exists():
            shutil.copy(cfg.RESULTADOS / "e3" / f, APP / f)
    # ---------------------------------------------------------------- tabla de líderes
    M = pd.read_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv")
    M = M[(M.tarea == "reg") & (M.estado == "completado")]
    filas = []
    for _, c in camp.iterrows():
        meta = json.load(open(ART / f"{_slug(c.etiqueta)}.metadata.json")) if (ART / f"{_slug(c.etiqueta)}.metadata.json").exists() else {}
        mejor_anidada = M[M.modelo == c.modelo].sort_values("externo_media").iloc[0]
        ext = M[(M.modelo == c.modelo) & (M.optimizador == c.variante)].iloc[0]
        filas.append({"modelo": c.modelo, "balanceo": "no aplica (regresión)", "optimizador": c.variante,
                      "RMSLE bucle externo (media ± d.e.)": f"{ext.RMSLE_media:.4f} ± {ext.RMSLE_desv:.4f}",
                      "RMSLE_externo": ext.RMSLE_media, "RMSLE validación < 2016": c["validación < 2016"],
                      "RMSLE origen móvil 2021–2026": float(np.sqrt(np.mean((np.log1p(O[c.etiqueta]) - np.log1p(O['y'])) ** 2))),
                      "mejor optimizador por bucle externo": mejor_anidada.optimizador,
                      "inferencia (ms/fila)": meta.get("tiempo_inferencia_ms_por_fila"), "etiqueta": c.etiqueta})
    meta = json.load(open(ART / "A+B_v2.metadata.json")) if (ART / "A+B_v2.metadata.json").exists() else {}
    filas.append({"modelo": "A+B v2 (propio)", "balanceo": "no aplica (regresión)", "optimizador": "rejilla interna de λ",
                  "RMSLE bucle externo (media ± d.e.)": "— (evaluado en el origen móvil)", "RMSLE_externo": np.nan,
                  "RMSLE validación < 2016": np.nan,
                  "RMSLE origen móvil 2021–2026": float(np.sqrt(np.mean((np.log1p(O['A+B v2']) - np.log1p(O['y'])) ** 2))),
                  "mejor optimizador por bucle externo": "—", "inferencia (ms/fila)": meta.get("tiempo_inferencia_ms_por_fila"),
                  "etiqueta": "A+B v2"})
    pd.DataFrame(filas).to_csv(APP / "lideres.csv", index=False)
    if not (ART / "A+B_v2.joblib").exists():
        print("Sin artefactos: se omiten el pronóstico y los SHAP.")
        return
    # ---------------------------------------------------------------- pronóstico del mes siguiente
    F = P[P.y.isna()].reset_index(drop=True)
    fondo = P[P.y.notna() & (P.mes >= P.mes.max() - 24)].sample(200, random_state=cfg.SEMILLA)
    fondo.to_pickle(APP / "fondo_shap.pkl.gz")
    pron = F[["celda", "lat", "lon", "fecha_obj", "n_1m", "n_12m"]].copy()
    shap_global = {}
    muestra = P[P.y.notna() & (P.mes >= P.mes.max() - 11)].sample(200, random_state=cfg.SEMILLA)
    for k in ["A+B v2"] + camp.etiqueta.tolist():
        mod = joblib.load(ART / f"{_slug(k)}.joblib")
        cols = COLUMNAS if k == "A+B v2" else VARIABLES_BASE
        p = np.clip(mod.predict(F[cols]), 0, None)
        lo, hi = intervalo(inter[k], p)
        pron[f"{k}|pred"], pron[f"{k}|lo"], pron[f"{k}|hi"] = p, lo, hi
        if k == "A+B v2":
            E = mod.explicar(muestra[cols]).abs().mean().sort_values(ascending=False)
            shap_global[k] = {"tipo": "aporte de cada miembro (c1·w·f), escala log1p", "valores": E.round(5).to_dict()}
            json.dump({"curva_validacion_B": mod.b_.curva_validacion_.tolist(), "n_iter_B": int(mod.b_.n_best),
                       "pesos_medios": dict(zip(mod.nombres_, mod.pesos(muestra[cols]).mean(0).round(4).tolist())),
                       "c0": float(mod.c0_), "c1": float(mod.c1_), "l2": float(mod.l2_)},
                      open(APP / "original.json", "w"), ensure_ascii=False, indent=1)
        else:
            from src.explicacion import shap_filas
            S = shap_filas(mod, muestra[VARIABLES_BASE], fondo[VARIABLES_BASE], max_evals=300, n_fondo=50)
            shap_global[k] = {"tipo": "SHAP (escala log1p)", "valores": dict(sorted(
                zip(VARIABLES_BASE, np.abs(S).mean(0).round(5).tolist()), key=lambda t: -t[1]))}
        print("listo", k, flush=True)
    pron.to_csv(APP / "pronostico.csv", index=False)
    json.dump(shap_global, open(APP / "shap_global.json", "w"), ensure_ascii=False, indent=1)


def intervalo(tabla: dict, p: np.ndarray) -> tuple:
    """Intervalo de predicción al 80 % a partir de los cuantiles de residuos log1p del tramo del pronóstico."""
    lp = np.log1p(p)
    b = np.array(tabla["bordes"])
    j = np.clip(np.searchsorted(b, lp, side="right") - 1, 0, len(b) - 2)
    return np.clip(np.expm1(lp + np.array(tabla["q10"])[j]), 0, None), np.expm1(lp + np.array(tabla["q90"])[j])


if __name__ == "__main__":
    main()
