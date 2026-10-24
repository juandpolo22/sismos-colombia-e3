"""Utilidades para los capítulos del Jupyter Book (mostrar texto con cifras calculadas y figuras Plotly)."""
from __future__ import annotations

import pandas as pd
from IPython.display import HTML, Markdown, display

_estado = {"plotly": False}


def md(texto: str) -> None:
    """Muestra Markdown cuyas cifras se calculan en la propia celda (nada se escribe a mano)."""
    display(Markdown(texto))


def mostrar_plotly(fig) -> None:
    """Muestra una figura Plotly interactiva (la primera de la página carga plotly.js desde la CDN)."""
    import plotly.io as pio
    incluir = "cdn" if not _estado["plotly"] else False
    _estado["plotly"] = True
    display(HTML(pio.to_html(fig, full_html=False, include_plotlyjs=incluir,
                             config={"displaylogo": False, "responsive": True})))


def tabla(df: pd.DataFrame, dec: int = 3):
    """Tabla con formato numérico uniforme para el libro."""
    return df.style.format(precision=dec, thousands=".", decimal=",", na_rep="—")
