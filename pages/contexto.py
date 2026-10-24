"""Sección 1 · Contexto: problema, pregunta de investigación, datos, metodología y limitaciones."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import dash_table, dcc, html

from src import data as D
from src import plots as pl

dash.register_page(__name__, path="/", name="Contexto", order=0)


def _cifra(valor, etiqueta):
    return dbc.Col(html.Div([html.Div(valor, className="cifra"), html.Div(etiqueta, className="cifra-et")], className="tarjeta"),
                   xs=6, md=4, lg=2)


def _paso(titulo, texto, propio=False):
    return dbc.Col(html.Div([html.B(titulo), texto], className="paso propio" if propio else "paso"), xs=12, sm=6, lg=True)


def _lectura_celdas(celdas):
    """Frase de lectura calculada: concentración de la actividad en las celdas más activas."""
    c = celdas.assign(depto=celdas.celda.map(D.depto_celda())).sort_values("y", ascending=False)
    k = max(1, int(round(0.05 * len(c))))
    parte = 100 * c.y.head(k).sum() / c.y.sum()
    top = "; ".join(f"{str(r.depto).title()} ({r.y:.1f} sismos/mes)" for r in c.head(3).itertuples())
    return [html.B("Lectura: "), f"la actividad está muy concentrada: el 5 % de las celdas más activas ({k}) reúne el {parte:.0f} % "
            f"de los sismos. Las tres más activas están en {top}; la mediana de las celdas es {c.y.median():.2f} sismos/mes."]


def layout():
    cat, P = D.catalogo(), D.panel()
    obs = P[P.y.notna()]
    n_meses = cat.mes.nunique()
    anual = cat.groupby(cat.time.dt.year).size()
    anual = anual[anual.index < cat.time.dt.year.max()]
    celdas = obs.groupby(["celda", "lat", "lon"]).y.mean().reset_index()
    dicc = [{"variable": c, "significado": D.DESCRIPCION[c], "unidad": D.UNIDADES[c]} for c in D.DESCRIPCION]
    dicc.insert(0, {"variable": "y (objetivo)", "significado": "Sismos M ≥ 2,5 en la celda durante el mes t+1 (conteo ≥ 0)", "unidad": "sismos"})
    return html.Div([
        html.Div([
            html.H1("¿Cuántos sismos habrá el próximo mes en cada zona de Colombia?"),
            html.P("Pronóstico mensual del número de sismos de magnitud M ≥ 2,5 en cada celda de 0,5° × 0,5° (≈ 55 km) del "
                   "territorio colombiano, con modelos de machine learning y un modelo original (A+B) que combina, según el "
                   "contexto de cada celda, varios pronosticadores, entre ellos un boosting de la distribución binomial negativa "
                   "inflada en ceros programado desde cero."),
        ], className="hero"),
        dbc.Row([
            _cifra(f"{len(cat):,}".replace(",", " "), "sismos M ≥ 2,5 en el catálogo del SGC"),
            _cifra(f"{anual.mean():,.0f}".replace(",", " "), f"sismos por año en promedio ({anual.index.min()}–{anual.index.max()})"),
            _cifra(f"{P.celda.nunique()}", "celdas activas (≥ 20 sismos antes de 2021)"),
            _cifra(f"{len(obs):,}".replace(",", " "), "filas celda × mes con objetivo conocido"),
            _cifra(f"{100 * (obs.y == 0).mean():.1f} %", "celdas-mes sin ningún sismo"),
            _cifra(f"{n_meses}", "meses de catálogo (2004-01 a 2026-09)"),
        ], className="g-2"),
        html.P(["Fuente de las cifras: cálculo propio sobre el catálogo sísmico del ",
                html.A("Servicio Geológico Colombiano", href="https://www.sgc.gov.co/sismos", target="_blank"),
                " (Red Sismológica Nacional de Colombia), consultado en octubre de 2026."], className="sub"),
        dbc.Row([
            dbc.Col(html.Div([
                html.H2("Planteamiento y relevancia"),
                html.P("Colombia está en el borde noroccidental de Suramérica, donde interactúan las placas de Nazca, Caribe y "
                       "Suramérica; la sismicidad es frecuente y está concentrada en zonas como el nido de Bucaramanga. "
                       "Saber dónde se espera más actividad el mes siguiente ayuda a priorizar el monitoreo, la revisión de "
                       "estaciones y la comunicación del riesgo."),
                html.P("El modelo no predice un sismo puntual (fecha, lugar exacto o magnitud), que hoy no es posible: estima la "
                       "tasa de actividad esperada de cada celda en el mes siguiente."),
                html.H3("Pregunta de investigación"),
                html.P("¿Puede un modelo que combina pronosticadores según el contexto de cada celda, con un componente propio de "
                       "conteo inflado en ceros, pronosticar el número mensual de sismos M ≥ 2,5 por celda con un error menor, de "
                       "forma estadísticamente significativa, que los mejores modelos de machine learning de cada familia?"),
                html.H3("Objetivos"),
                html.Ul([
                    html.Li("Construir un panel celda × mes sin fuga de información a partir del catálogo del SGC."),
                    html.Li("Optimizar y comparar 7 familias de modelos con 4 métodos de búsqueda de hiperparámetros."),
                    html.Li("Diseñar, implementar y validar estadísticamente un modelo original (A+B)."),
                    html.Li("Poner los mejores modelos a disposición en este tablero, con pronóstico y explicación."),
                ]),
            ], className="tarjeta"), lg=6),
            dbc.Col(html.Div([
                html.H2("Actividad media por celda"),
                html.P("Sismos M ≥ 2,5 por mes en cada celda activa (promedio de todos los meses observados).", className="sub"),
                dcc.Graph(figure=pl.mapa_celdas(celdas.lon, celdas.lat, celdas.y, "", "sismos/mes",
                                                texto=[f"({a:.2f}, {o:.2f}) · {v:.2f} sismos/mes" for a, o, v in
                                                       zip(celdas.lat, celdas.lon, celdas.y)]).update_layout(height=430),
                          config={"displayModeBar": False}),
                html.Div(_lectura_celdas(celdas), className="insight"),
            ], className="tarjeta"), lg=6),
        ]),
        html.Div([
            html.H2("Metodología"),
            html.P("De los datos crudos al pronóstico del mes siguiente. El naranja marca las piezas originales del proyecto.", className="sub"),
            dbc.Row([
                _paso("1 · ETL", "10 archivos del SGC integrados, duplicados y errores depurados, filtro M ≥ 2,5."),
                _paso("2 · EDA", "Objetivo, univariado, bivariado, temporal y espacial; auditoría de fuga."),
                _paso("3 · Panel", "Celda × mes; variables solo con lo ocurrido hasta el cierre del mes t."),
                _paso("4 · Modelos", "7 familias × 4 optimizadores (Grid, Random, Bayesiano, Genético)."),
                _paso("5 · Validación", "Validación anidada (2005–2020) y origen móvil 2021–2026 (69 meses)."),
                _paso("6 · Comparación", "MCS → Giacomini-White / Clark-West → Diebold-Mariano, SPA; Holm."),
                _paso("7 · Modelo original", "A+B: boosting ZINB sobre CART propio + combinador contextual + calibración.", propio=True),
            ], className="g-2"),
        ], className="tarjeta"),
        dbc.Row([
            dbc.Col(html.Div([
                html.H2("Datos y variables"),
                html.P([f"Catálogo del SGC, {cat.time.min():%Y-%m} a {cat.time.max():%Y-%m}; {len(cat):,} sismos tras la depuración. ".replace(",", " "),
                        f"Panel de {P.celda.nunique()} celdas × meses; tarea: regresión de conteos (métrica principal RMSLE)."], className="sub"),
                dash_table.DataTable(dicc, [{"name": n, "id": n} for n in ("variable", "significado", "unidad")],
                                     style_table={"overflowX": "auto"}, page_size=10, sort_action="native",
                                     style_cell={"fontSize": 13, "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                                     style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"}),
            ], className="tarjeta"), lg=7),
            dbc.Col(html.Div([
                html.H2("Limitaciones, ética y sesgos"),
                html.Ul([
                    html.Li("Pronostica tasas de actividad, no sismos individuales: no debe usarse para alertas ni decisiones de evacuación."),
                    html.Li("El catálogo depende de la red de estaciones: zonas con menos cobertura pueden tener sismos pequeños sin registrar (sesgo de detección)."),
                    html.Li("Las réplicas de un sismo grande inflan los conteos de los meses siguientes; ningún modelo anticipa el sismo principal."),
                    html.Li("Celdas con poca actividad histórica tienen pronósticos menos estables (más incertidumbre relativa)."),
                    html.Li("Comunicar estos resultados exige cuidado: una celda con pronóstico alto no implica mayor peligro de un sismo dañino."),
                ]),
                html.H3("Autores, versión y fecha"),
                html.P("Juan David Polo y Gabriella Pallares · Machine Learning (Universidad del Norte) · versión 3.0 · actualizado "
                       "en octubre de 2026 con datos hasta septiembre de 2026."),
            ], className="tarjeta"), lg=5),
        ]),
    ])
