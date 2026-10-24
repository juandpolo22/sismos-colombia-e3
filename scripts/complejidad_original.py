"""Complejidad empírica del modelo original: tiempo de entrenamiento al variar n (filas) y p (variables).

Mide (i) el árbol CART propio, (ii) una iteración del boosting B (dos árboles + gradientes) y (iii) el combinador A
(L-BFGS), con datos reales del panel (meses anteriores a 2021). Repite cada medición y guarda la mediana en
resultados/e3/complejidad.csv. El exponente empírico es la pendiente de log(tiempo) contra log(n) o log(p).

Uso:  python scripts/complejidad_original.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as cfg, panel
from src.arbol_cart import ArbolCART
from src.datos import VARIABLES_BASE
from src.original_model import EXCITACION
from src.propuesto import CombinadorSoftmax, grads_zinb

OUT = cfg.RESULTADOS / "e3"
OUT.mkdir(parents=True, exist_ok=True)
REP = 5


def cronometrar(f):
    t = []
    for _ in range(REP):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    cfg.fijar_semillas()
    P = panel.construir()
    P = P[P.mes < P.mes.quantile(0.7)].reset_index(drop=True)
    X = np.nan_to_num(P[VARIABLES_BASE + EXCITACION].to_numpy(float), nan=-1.0)
    y = P.y.to_numpy()
    rng = np.random.default_rng(cfg.SEMILLA)
    filas = []
    for n in (2000, 4000, 8000, 16000, len(y)):
        i = rng.choice(len(y), n, replace=False)
        Xi, yi = X[i], y[i]
        eta, xi = np.full(n, np.log(max(yi.mean(), .05))), np.full(n, -0.8)
        g, _, h, _ = grads_zinb(eta, xi, 0.0, yi)
        r = np.clip(g / h, -3, 3)
        O = ArbolCART.ordenar(Xi)
        filas.append({"componente": "árbol CART (sin orden previo)", "variable": "n", "valor": n,
                      "segundos": cronometrar(lambda: ArbolCART(3, 50).fit(Xi, r, h))})
        filas.append({"componente": "árbol CART (con orden reutilizado)", "variable": "n", "valor": n,
                      "segundos": cronometrar(lambda: ArbolCART(3, 50).fit(Xi, r, h, orden=O))})
        K = 14
        B = np.log1p(np.abs(yi[:, None] + rng.normal(0, 1, (n, K))))
        Z = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 15))])
        filas.append({"componente": "combinador A (L-BFGS)", "variable": "n", "valor": n,
                      "segundos": cronometrar(lambda: CombinadorSoftmax(0.01).fit(Z, B, np.log1p(yi)))})
        print("n", n, flush=True)
    n = 16000
    i = rng.choice(len(y), n, replace=False)
    for p in (4, 8, 16, 31):
        Xi, yi = X[i][:, :p], y[i]
        eta, xi = np.full(n, np.log(max(yi.mean(), .05))), np.full(n, -0.8)
        g, _, h, _ = grads_zinb(eta, xi, 0.0, yi)
        r = np.clip(g / h, -3, 3)
        O = ArbolCART.ordenar(Xi)
        filas.append({"componente": "árbol CART (con orden reutilizado)", "variable": "p", "valor": p,
                      "segundos": cronometrar(lambda: ArbolCART(3, 50).fit(Xi, r, h, orden=O))})
        print("p", p, flush=True)
    T = pd.DataFrame(filas)
    T.to_csv(OUT / "complejidad.csv", index=False)
    for (c, v), d in T.groupby(["componente", "variable"]):
        b = np.polyfit(np.log(d.valor), np.log(d.segundos), 1)[0]
        print(f"{c} vs {v}: exponente empírico {b:.2f}")


if __name__ == "__main__":
    main()
