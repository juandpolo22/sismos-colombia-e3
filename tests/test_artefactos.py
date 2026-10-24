"""Prueba de ida y vuelta de los artefactos (guía del Entregable 3, sección 4.2): cada modelo serializado en artifacts/
debe reproducir las predicciones guardadas al entrenarlo sobre un conjunto fijo (los últimos 12 meses).

La comparación admite un error relativo de 1e-9: entre máquinas o números de núcleos, el álgebra lineal (BLAS) suma en
otro orden y cambia el último dígito (diferencias del orden de 1e-14); cualquier error real de serialización es mucho
mayor."""
import json

import numpy as np
import pytest

from src import config as cfg, panel
from src.artefactos import ART, modelo
from src.datos import VARIABLES_BASE
from src.original_model import COLUMNAS

INDICE = ART / "indice.json"
pytestmark = pytest.mark.skipif(not INDICE.exists(), reason="faltan los artefactos (python scripts/entrenar_artefactos.py)")


@pytest.fixture(scope="module")
def conjunto_fijo():
    P = panel.construir(incluir_siguiente=True)
    P = P[P.y.notna()].reset_index(drop=True)
    return P[P.mes >= P.mes.max() - 11].reset_index(drop=True)


@pytest.mark.parametrize("entrada", json.load(open(INDICE, encoding="utf-8")) if INDICE.exists() else [])
def test_ida_y_vuelta(entrada, conjunto_fijo):
    meta = json.load(open(ART / f"{entrada['modelo']}.metadata.json", encoding="utf-8"))
    cols = COLUMNAS if entrada["etiqueta"].startswith("A+B") else VARIABLES_BASE
    esperado = np.load(ART / meta["prueba_ida_y_vuelta"]["archivo"])
    obtenido = modelo(entrada["etiqueta"]).predict(conjunto_fijo[cols])
    np.testing.assert_allclose(np.asarray(obtenido), esperado, rtol=1e-9, atol=1e-12)
    assert meta["semilla"] == cfg.SEMILLA
    assert {e["columna"] for e in meta["esquema_entrada"]} == set(cols)
