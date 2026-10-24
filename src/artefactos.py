"""Nombres de archivo de los artefactos y carga perezosa (una sola vez por proceso)."""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache

import joblib

from . import config as cfg

ART = cfg.RAIZ / "artifacts"


def slug(etiqueta: str) -> str:
    """Nombre de archivo de un modelo: «Árbol de decisión · TPE» → «Arbol_de_decision__TPE»; «A+B v2» → «A+B_v2»."""
    if etiqueta.startswith("A+B"):
        return "A+B_v2"
    s = unicodedata.normalize("NFKD", etiqueta).encode("ascii", "ignore").decode()
    return s.replace(" · ", "__").replace(" ", "_")


@lru_cache(maxsize=None)
def modelo(etiqueta: str):
    """Carga el pipeline serializado (solo la primera vez que se pide)."""
    return joblib.load(ART / f"{slug(etiqueta)}.joblib")


@lru_cache(maxsize=None)
def metadata(etiqueta: str) -> dict:
    return json.load(open(ART / f"{slug(etiqueta)}.metadata.json", encoding="utf-8"))
