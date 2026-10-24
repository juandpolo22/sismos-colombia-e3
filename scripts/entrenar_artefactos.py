"""Reentrenamiento final y serialización de los modelos del dashboard (guía del Entregable 3, sección 4.2).

Para cada uno de los 7 modelos base se toma su campeón (la combinación con mejor validación interna anterior a 2016,
la misma que se comparó estadísticamente en el Entregable 2) y, además, el modelo original A+B. Cada pipeline completo
(preprocesamiento + modelo) se reentrena con TODOS los meses con objetivo conocido y se guarda con joblib (compress=3)
en ``artifacts/`` junto con su ``metadata.json``. Se verifica la ida y vuelta: el modelo cargado debe producir
exactamente las mismas predicciones que el modelo en memoria sobre un conjunto fijo.

El dashboard solo carga estos archivos: nunca entrena ni optimiza.

Uso:  python scripts/entrenar_artefactos.py [--solo-benchmarks]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src import benchmark as bm
from src import config as cfg
from src import panel, resultados_e2 as r2
from src.datos import VARIABLES_BASE
from src.original_model import COLUMNAS, ModeloOriginalAB
from src.artefactos import slug

ART = cfg.RAIZ / "artifacts"
ART.mkdir(exist_ok=True)


def esquema(P: pd.DataFrame, columnas) -> list:
    """Nombre, tipo y rango válido (mínimo y máximo observados en el entrenamiento) de cada columna de entrada."""
    out = []
    for c in columnas:
        v = P[c]
        entero = bool(np.all(np.isnan(v) | (v == np.round(v))))
        out.append({"columna": c, "tipo": "entero" if entero else "real", "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                    "admite_faltante": bool(v.isna().any())})
    return out


def metricas_externas(m: str, o: str) -> dict:
    """Métricas del bucle externo de la validación anidada (tabla maestra del Entregable 2)."""
    M = pd.read_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv")
    f = M[(M.tarea == "reg") & (M.modelo == m) & (M.optimizador == o)]
    if f.empty:
        return {}
    f = f.iloc[0]
    return {k: float(f[k]) for k in ("RMSLE_media", "RMSLE_desv", "RMSE_media", "RMSE_desv", "MAE_media", "MAE_desv")}


def un_hilo(obj, vistos=None):
    """Pone n_jobs = 1 en todos los estimadores internos: la suma en paralelo de los árboles de Random Forest cambia el
    orden de redondeo entre ejecuciones y rompería la igualdad exacta de la prueba de ida y vuelta."""
    vistos = set() if vistos is None else vistos
    if id(obj) in vistos or obj is None or isinstance(obj, (str, int, float, bool, np.ndarray, pd.DataFrame)):
        return
    vistos.add(id(obj))
    if hasattr(obj, "n_jobs"):
        try:
            obj.n_jobs = 1
        except Exception:
            pass
    hijos = list(vars(obj).values()) if hasattr(obj, "__dict__") else []
    for h in hijos:
        for x in (h if isinstance(h, (list, tuple)) else [h]):
            if hasattr(x, "__dict__") or isinstance(x, (list, tuple)):
                un_hilo(x[1] if isinstance(x, tuple) and len(x) == 2 and hasattr(x[1], "__dict__") else x, vistos)


def guardar(nombre: str, modelo, P_tr, X_fijo, columnas, extra: dict) -> dict:
    """Serializa, mide tamaño y verifica la ida y vuelta."""
    un_hilo(modelo)
    ruta = ART / f"{nombre}.joblib"
    p_mem = np.asarray(modelo.predict(X_fijo))
    joblib.dump(modelo, ruta, compress=3)
    t0 = time.perf_counter()
    cargado = joblib.load(ruta)
    t_carga = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_disco = np.asarray(cargado.predict(X_fijo))
    t_inf = (time.perf_counter() - t0) / len(X_fijo)
    ida_vuelta = bool(np.array_equal(p_mem, p_disco))
    np.save(ART / f"{nombre}.ida_vuelta.npy", p_mem)
    meta = {
        "modelo": nombre, "archivo": ruta.name, "tamano_MB": round(ruta.stat().st_size / 2**20, 2),
        "semilla": cfg.SEMILLA, "python": platform.python_version(),
        "librerias": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                      "xgboost": xgboost.__version__, "joblib": joblib.__version__},
        "entrenamiento": {"filas": int(len(P_tr)), "primer_mes_objetivo": str(P_tr.fecha_obj.min().date()),
                          "ultimo_mes_objetivo": str(P_tr.fecha_obj.max().date()), "celdas": int(P_tr.celda.nunique())},
        "esquema_entrada": esquema(P_tr, columnas),
        "prueba_ida_y_vuelta": {"filas": int(len(X_fijo)), "identicas": ida_vuelta, "archivo": f"{nombre}.ida_vuelta.npy"},
        "tiempo_carga_s": round(t_carga, 3), "tiempo_inferencia_ms_por_fila": round(1000 * t_inf, 4),
        **extra,
    }
    json.dump(meta, open(ART / f"{nombre}.metadata.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"{nombre}: {meta['tamano_MB']} MB, ida y vuelta {'OK' if ida_vuelta else 'FALLA'}, "
          f"{meta['tiempo_inferencia_ms_por_fila']} ms/fila", flush=True)
    if not ida_vuelta:
        raise SystemExit(f"La prueba de ida y vuelta falló para {nombre}")
    return meta


def main(solo_benchmarks: bool):
    cfg.fijar_semillas()
    P = panel.construir(incluir_siguiente=True)
    P_tr = P[P.y.notna()].reset_index(drop=True)
    X_fijo = P_tr[P_tr.mes >= P_tr.mes.max() - 11].reset_index(drop=True)      # últimos 12 meses (conjunto fijo)
    y = P_tr.y.to_numpy()
    om = pd.read_csv(cfg.TABLAS / "e2_ab_vs_campeones.csv").set_index("campeón")
    indice = []
    for _, c in r2.campeones_om("reg").iterrows():
        m, o, etq = c["modelo"], c["variante"], c["etiqueta"]
        params = json.load(open(cfg.RESULTADOS / "origen_movil" / "hp_bench" / f"{r2._slug(m, o)}.json"))["params"]
        t0 = time.perf_counter()
        mod = bm.construir("reg", m, params, VARIABLES_BASE, n_jobs=2).fit(P_tr[VARIABLES_BASE], y)
        t_fit = time.perf_counter() - t0
        meta = guardar(slug(etq), mod, P_tr, X_fijo[VARIABLES_BASE], VARIABLES_BASE, {
            "familia": m, "optimizador": o, "etiqueta": etq, "hiperparametros": params,
            "criterio_seleccion": "menor RMSLE en la validación interna anterior a 2016 (campeón de la familia)",
            "rmsle_validacion_previa_2016": float(c["validación < 2016"]),
            "metricas_bucle_externo_validacion_anidada": metricas_externas(m, o),
            "rmsle_origen_movil_2021_2026": float(om.loc[etq, "RMSLE campeón"]), "tiempo_entrenamiento_s": round(t_fit, 1)})
        indice.append(meta)
    if not solo_benchmarks:
        t0 = time.perf_counter()
        ab = ModeloOriginalAB(n_jobs=2).fit(P_tr[COLUMNAS], y)
        t_fit = time.perf_counter() - t0
        meta = guardar(slug("A+B v2"), ab, P_tr, X_fijo[COLUMNAS], COLUMNAS, {
            "familia": "Modelo original", "optimizador": "λ por rejilla interna; B y miembros con hiperparámetros previos a 2016",
            "etiqueta": "A+B v2 (propio)", "hiperparametros": {k: v for k, v in ab.get_params().items() if k != "miembros"},
            "miembros": ab.nombres_, "l2_elegido": float(ab.l2_), "calibracion": {"c0": float(ab.c0_), "c1": float(ab.c1_)},
            "rmsle_origen_movil_2021_2026": float(om["RMSLE A+B v2"].iloc[0]), "tiempo_entrenamiento_s": round(t_fit, 1)})
        indice.append(meta)
    json.dump([{k: m[k] for k in ("modelo", "archivo", "etiqueta", "tamano_MB")} for m in indice],
              open(ART / "indice.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-benchmarks", action="store_true")
    main(ap.parse_args().solo_benchmarks)
