"""Verifica que la clase ModeloOriginalAB reproduce exactamente el modelo evaluado en el origen móvil.

Ajusta la clase con el entrenamiento del origen 2021 (meses objetivo < dic-2020) y predice 2021; compara con
resultados/origen_movil/final/origen_2021.npz (scripts/origen_movil_final.py).

Uso:  python scripts/verificar_modelo_original.py [año]
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config as cfg, datos, evaluacion as ev, panel
from src.original_model import ModeloOriginalAB

anio = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
t0 = time.time()
P = panel.construir(f"{anio}-01-01")
m_test = int(datos.indice_mes([f"{anio}-01-01"])[0])
tr = P["mes"].to_numpy() < m_test - cfg.GAP_MESES
te = (P["mes"].to_numpy() >= m_test) & (P["mes"].to_numpy() < datos.indice_mes([f"{anio + 1}-01-01"])[0])
mod = ModeloOriginalAB(n_jobs=1).fit(P[tr], P["y"][tr])
p = mod.predict(P[te])
ref = np.load(cfg.RESULTADOS / "origen_movil" / "final" / f"origen_{anio}.npz")
assert np.array_equal(ref["y"], P["y"][te].to_numpy())
print(f"RMSLE clase {ev.rmsle(ref['y'], p):.6f} | evaluado {ev.rmsle(ref['y'], ref['AmasB']):.6f}")
print(f"dif. máx. de predicción {np.abs(p - ref['AmasB']).max():.2e} | c1 {mod.c1_:.4f} vs {float(ref['c1']):.4f} | {time.time() - t0:.0f} s")
