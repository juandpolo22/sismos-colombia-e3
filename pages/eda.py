"""Sección 2 · EDA interactivo. Los filtros globales actualizan todos los gráficos de la página."""
import warnings

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src import data as D
from src import plots as pl
from src.datos import VARIABLES_BASE

dash.register_page(__name__, path="/eda", name="EDA", order=1)

CFG = {"displayModeBar": False, "responsive": True}


def _grafico(id_, alto=380):
    return dcc.Loading(dcc.Graph(id=id_, config=CFG, style={"height": f"{alto}px"}), type="dot", color=pl.AZUL)


def _tarjeta(titulo, sub, cuerpo, insight_id=None):
    hijos = [html.H2(titulo), html.P(sub, className="sub"), cuerpo]
    if insight_id:
        hijos.append(html.Div(id=insight_id, className="insight"))
    return html.Div(hijos, className="tarjeta")


def layout():
    cat = D.catalogo()
    deptos = sorted(cat.depto.dropna().unique())
    opciones = [{"label": f"{c} — {D.DESCRIPCION[c]}", "value": c} for c in VARIABLES_BASE] + [{"label": "y — sismos del mes objetivo", "value": "y"}]
    return html.Div([
        html.Div([
            html.H2("Filtros globales"),
            html.P("Todo lo de esta página responde a estos filtros. La magnitud y la profundidad filtran el catálogo; el periodo y "
                   "los departamentos filtran también el panel celda × mes.", className="sub"),
            dbc.Row([
                dbc.Col([html.Label("Periodo"), dcc.DatePickerRange(id="f-fechas", min_date_allowed=cat.time.min().date(),
                        max_date_allowed=cat.time.max().date(), start_date="2005-01-01", end_date=cat.time.max().date(),
                        display_format="YYYY-MM", className="d-block")], md=4),
                dbc.Col([html.Label("Magnitud"), dcc.RangeSlider(id="f-mag", min=2.5, max=7.5, step=0.1, value=[2.5, 7.5],
                        marks={m: str(m) for m in (2.5, 3.5, 4.5, 5.5, 6.5, 7.5)}, tooltip={"placement": "bottom"})], md=4),
                dbc.Col([html.Label("Profundidad (km)"), dcc.RangeSlider(id="f-prof", min=0, max=300, step=5, value=[0, 300],
                        marks={p: str(p) for p in (0, 50, 100, 150, 200, 300)}, tooltip={"placement": "bottom"})], md=4),
            ], className="g-3"),
            dbc.Row([dbc.Col([html.Label("Departamentos (vacío = todos)"),
                              dcc.Dropdown(id="f-depto", options=deptos, multi=True, placeholder="Todos los departamentos")], md=12)],
                    className="mt-2"),
            html.Div(id="f-resumen", className="insight mt-2"),
        ], className="tarjeta"),
        dbc.Row([
            dbc.Col(_tarjeta("Variable objetivo", "Sismos M ≥ 2,5 por celda y mes (eje vertical logarítmico).", _grafico("g-objetivo"), "i-objetivo"), lg=6),
            dbc.Col(_tarjeta("Mapa de sismos", "Muestra de hasta 4 000 sismos; tamaño ∝ magnitud, color = profundidad.", _grafico("g-mapa", 520), "i-mapa"), lg=6),
        ]),
        dbc.Row([
            dbc.Col(_tarjeta("Distribución univariada", "Elija una variable del panel.", html.Div([
                dcc.Dropdown(id="u-var", options=opciones, value="n_12m", clearable=False), _grafico("g-univ")]), "i-univ"), lg=6),
            dbc.Col(_tarjeta("Explorador bivariado", "Elija las variables de los ejes y del color (muestra de 4 000 filas; ejes en log1p si son conteos).", html.Div([
                dbc.Row([dbc.Col(dcc.Dropdown(id="b-x", options=opciones, value="n_12m", clearable=False), xs=4),
                         dbc.Col(dcc.Dropdown(id="b-y", options=opciones, value="y", clearable=False), xs=4),
                         dbc.Col(dcc.Dropdown(id="b-c", options=opciones, value="prof_media_hist", clearable=False), xs=4)], className="g-1"),
                _grafico("g-biv")]), "i-biv"), lg=6),
        ]),
        dbc.Row([
            dbc.Col(_tarjeta("Matriz de correlación", "Spearman entre las variables del panel y el objetivo.", _grafico("g-corr", 620), "i-corr"), lg=7),
            dbc.Col(_tarjeta("Valores faltantes y atípicos", "Porcentaje por variable en las filas filtradas (atípico: fuera de Q1 − 1,5·RIC, Q3 + 1,5·RIC).",
                             html.Div(id="t-faltantes"), "i-falt"), lg=5),
        ]),
        _tarjeta("Serie mensual y descomposición", "Sismos por mes (filtros aplicados) y descomposición STL en tendencia, estacionalidad y residuo.",
                 _grafico("g-stl", 560), "i-stl"),
        dbc.Row([
            dbc.Col(_tarjeta("ACF y PACF", "Autocorrelación simple y parcial de la serie mensual (banda: ±1,96/√n).", _grafico("g-acf", 420)), lg=7),
            dbc.Col(_tarjeta("Pruebas de estacionariedad", "ADF (H0: raíz unitaria) y KPSS (H0: estacionaria en nivel).", html.Div(id="t-estac"), "i-estac"), lg=5),
        ]),
    ])


def _filtrar(f0, f1, mag, prof, deptos):
    c = D.catalogo()
    c = c[(c.time >= pd.Timestamp(f0)) & (c.time <= pd.Timestamp(f1) + pd.Timedelta(days=1)) & c.mag.between(*mag)]
    c = c[c.depth.between(prof[0], prof[1] if prof[1] < 300 else 1e4)]
    P = D.panel()
    P = P[P.y.notna() & (P.fecha_obj >= pd.Timestamp(f0)) & (P.fecha_obj <= pd.Timestamp(f1))]
    if deptos:
        c, P = c[c.depto.isin(deptos)], P[P.depto.isin(deptos)]
    return c, P


@callback(
    [Output("f-resumen", "children"), Output("g-objetivo", "figure"), Output("i-objetivo", "children"), Output("g-mapa", "figure"),
     Output("i-mapa", "children"), Output("g-corr", "figure"), Output("i-corr", "children"), Output("t-faltantes", "children"),
     Output("i-falt", "children"), Output("g-stl", "figure"), Output("i-stl", "children"), Output("g-acf", "figure"),
     Output("t-estac", "children"), Output("i-estac", "children")],
    [Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-mag", "value"), Input("f-prof", "value"), Input("f-depto", "value")],
)
def actualizar(f0, f1, mag, prof, deptos):
    c, P = _filtrar(f0, f1, mag, prof, deptos)
    resumen = [html.B(f"{len(c):,} sismos".replace(",", " ")), f" del catálogo y ", html.B(f"{len(P):,} filas celda × mes".replace(",", " ")),
               f" ({P.celda.nunique()} celdas) cumplen los filtros."]
    if len(c) < 30 or len(P) < 50:
        v = pl.vacia()
        aviso = "Muy pocos datos: amplíe los filtros."
        return resumen, v, aviso, v, aviso, v, aviso, html.Div(), aviso, v, aviso, v, html.Div(), aviso
    # objetivo
    f_obj = pl.histograma(P.y, "", "sismos en el mes (y)", log_y=True, nbins=60)
    ceros = 100 * (P.y == 0).mean()
    i_obj = [html.B("Lectura: "), f"{ceros:.1f} % de las celdas-mes no tiene sismos y la media es {P.y.mean():.2f} con varianza "
             f"{P.y.var():.1f}: un conteo con exceso de ceros y sobredispersión (varianza ≫ media). Por eso se modela en escala "
             f"log1p (métrica RMSLE) y el modelo B usa una binomial negativa inflada en ceros."]
    # mapa
    m = c.sample(min(len(c), 4000), random_state=0)
    f_mapa = pl.mapa_puntos(m, "")
    top = c.depto.value_counts().head(3)
    i_mapa = [html.B("Lectura: "), f"los departamentos con más sismos en el filtro son " +
              ", ".join(f"{d.title()} ({100 * n / len(c):.0f} %)" for d, n in top.items()) +
              f"; el {100 * (c.depth > 70).mean():.0f} % son de profundidad intermedia (> 70 km), sobre todo el nido de Bucaramanga."]
    # correlación
    cols = VARIABLES_BASE + ["y"]
    C = P[cols].corr(method="spearman")
    f_corr = pl.correlacion(C.round(2), "")
    cy = C["y"].drop("y").abs().sort_values(ascending=False)
    i_corr = [html.B("Lectura: "), f"las variables más asociadas con el objetivo son {', '.join(cy.index[:3])} (|ρ| hasta {cy.iloc[0]:.2f}); "
              f"los conteos de distintas ventanas están muy correlacionados entre sí, por eso se regularizan los modelos lineales."]
    # faltantes y atípicos
    filas = []
    for v in cols:
        x = P[v]
        q1, q3 = x.quantile([.25, .75])
        r = q3 - q1
        filas.append({"variable": v, "faltantes %": round(100 * x.isna().mean(), 1),
                      "atípicos %": round(100 * ((x < q1 - 1.5 * r) | (x > q3 + 1.5 * r)).mean(), 1)})
    T = pd.DataFrame(filas)
    tabla = dash_table.DataTable(T.to_dict("records"), [{"name": k, "id": k} for k in T.columns], sort_action="native",
                                 style_cell={"fontSize": 12.5, "padding": "4px"}, style_table={"maxHeight": "560px", "overflowY": "auto"},
                                 style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"})
    mf = T.sort_values("faltantes %").iloc[-1]
    i_falt = [html.B("Lectura: "), f"los faltantes se concentran en las magnitudes ({mf.variable}: {mf['faltantes %']} %), porque no hay "
              "magnitud media ni máxima en un periodo sin sismos; se imputan con 2,5 (el umbral) dentro del pipeline. Los «atípicos» de "
              "los conteos son la cola real de la distribución (enjambres y réplicas), no errores, y se conservan."]
    # serie, STL, ACF/PACF, estacionariedad
    s = c.groupby("mes").size().reindex(pd.date_range(c.mes.min(), c.mes.max(), freq="MS"), fill_value=0)
    if len(s) < 36:
        v = pl.vacia("Se necesitan al menos 36 meses para la descomposición")
        return resumen, f_obj, i_obj, f_mapa, i_mapa, f_corr, i_corr, tabla, i_falt, v, "", v, html.Div(), ""
    ls = np.log1p(s)
    st = STL(ls, period=12, robust=True).fit()
    from plotly.subplots import make_subplots
    f_stl = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                          subplot_titles=("log(1 + sismos por mes)", "tendencia", "estacionalidad", "residuo"))
    for i, (yv, col) in enumerate([(ls, pl.AZUL), (st.trend, pl.NARANJA), (st.seasonal, pl.AZUL), (st.resid, pl.GRIS)], 1):
        f_stl.add_trace(go.Scatter(x=s.index, y=yv, mode="lines", line=dict(color=col, width=1.4), showlegend=False), row=i, col=1)
    f_stl.update_layout(margin=dict(l=50, r=10, t=40, b=30))
    fuerza_est = max(0, 1 - np.var(st.resid) / np.var(st.seasonal + st.resid))
    pico = s.idxmax()
    i_stl = [html.B("Lectura: "), f"el mes con más sismos es {pico:%Y-%m} ({s.max():,} sismos, típicamente un enjambre o una secuencia "
             f"de réplicas). La fuerza de la estacionalidad es {fuerza_est:.2f} (0 = ninguna, 1 = total): ".replace(",", " ") +
             ("hay un patrón anual débil" if fuerza_est > 0.3 else "no hay un ciclo anual relevante") +
             "; la variación la dominan la tendencia y los episodios puntuales."]
    nl = min(24, len(s) // 3)
    a, p_ = acf(ls, nlags=nl)[1:], pacf(ls, nlags=nl, method="ywm")[1:]
    f_acf = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    banda = 1.96 / np.sqrt(len(s))
    for j, vals in enumerate((a, p_), 1):
        f_acf.add_trace(go.Bar(x=np.arange(1, nl + 1), y=vals, marker_color=pl.AZUL, showlegend=False), row=1, col=j)
        f_acf.add_hrect(y0=-banda, y1=banda, fillcolor=pl.ARENA, opacity=0.8, line_width=0, layer="below", row=1, col=j)
    f_acf.update_xaxes(title_text="rezago (meses)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(ls, autolag="AIC")[1]
        kpss_p = kpss(ls, regression="c", nlags="auto")[1]
        kpss_d = kpss(ls.diff().dropna(), regression="c", nlags="auto")[1]
    E = pd.DataFrame([{"prueba": "ADF (nivel)", "p": round(adf_p, 4), "conclusión": "estacionaria" if adf_p < 0.05 else "raíz unitaria"},
                      {"prueba": "KPSS (nivel)", "p": round(kpss_p, 4), "conclusión": "no estacionaria" if kpss_p < 0.05 else "estacionaria"},
                      {"prueba": "KPSS (diferencia)", "p": round(kpss_d, 4), "conclusión": "no estacionaria" if kpss_d < 0.05 else "estacionaria"}])
    t_est = dash_table.DataTable(E.to_dict("records"), [{"name": k, "id": k} for k in E.columns],
                                 style_cell={"fontSize": 13}, style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"})
    i_est = [html.B("Lectura: "), f"la ACF decae lentamente (ρ₁ = {a[0]:.2f}): la actividad de un mes anticipa la del siguiente, lo que "
             "justifica las variables de rezago. " + ("ADF y KPSS no coinciden en el nivel (tendencia o cambios de régimen); "
                                                      if (adf_p < 0.05) == (kpss_p < 0.05) else "")
             + "Los modelos se reentrenan cada año para adaptarse a esa deriva."]
    return resumen, f_obj, i_obj, f_mapa, i_mapa, f_corr, i_corr, tabla, i_falt, f_stl, i_stl, f_acf, t_est, i_est


def _transf(v, x):
    return np.log1p(x) if v in ("y",) or v.startswith(("n_", "n35", "vec")) else x


@callback([Output("g-univ", "figure"), Output("i-univ", "children")],
          [Input("u-var", "value"), Input("f-fechas", "start_date"), Input("f-fechas", "end_date"), Input("f-mag", "value"),
           Input("f-prof", "value"), Input("f-depto", "value")])
def univariado(v, f0, f1, mag, prof, deptos):
    _, P = _filtrar(f0, f1, mag, prof, deptos)
    if len(P) < 50:
        return pl.vacia(), "Muy pocos datos."
    x = P[v].dropna()
    f = pl.histograma(x, "", f"{v} ({D.UNIDADES.get(v, 'sismos')})", nbins=50, log_y=v in ("y",) or v.startswith(("n_", "n35", "vec")))
    sesgo = x.skew()
    return f, [html.B("Lectura: "), f"{v}: media {x.mean():.2f}, mediana {x.median():.2f}, asimetría {sesgo:.1f}. " +
               ("Distribución muy sesgada a la derecha: muchos valores bajos y pocos muy altos (por eso los conteos entran en log1p)."
                if sesgo > 2 else "Distribución sin una cola extrema.")]


@callback([Output("g-biv", "figure"), Output("i-biv", "children")],
          [Input("b-x", "value"), Input("b-y", "value"), Input("b-c", "value"), Input("f-fechas", "start_date"), Input("f-fechas", "end_date"),
           Input("f-mag", "value"), Input("f-prof", "value"), Input("f-depto", "value")])
def bivariado(vx, vy, vc, f0, f1, mag, prof, deptos):
    _, P = _filtrar(f0, f1, mag, prof, deptos)
    if len(P) < 50:
        return pl.vacia(), "Muy pocos datos."
    S = P.sample(min(4000, len(P)), random_state=0)
    x, y = _transf(vx, S[vx]), _transf(vy, S[vy])
    f = go.Figure(go.Scattergl(x=x + np.random.default_rng(0).normal(0, 0.02, len(S)), y=y, mode="markers",
                               marker=dict(size=5, color=S[vc], colorscale=pl.ESCALA, opacity=0.6, colorbar=dict(title=vc, thickness=10)),
                               hovertemplate=f"{vx}: %{{x:.2f}}<br>{vy}: %{{y:.2f}}<extra></extra>"))
    lx = f"log1p({vx})" if x is not S[vx] else vx
    ly = f"log1p({vy})" if y is not S[vy] else vy
    f.update_layout(xaxis_title=lx, yaxis_title=ly)
    rho = S[[vx, vy]].corr(method="spearman").iloc[0, 1]
    return f, [html.B("Lectura: "), f"correlación de Spearman entre {vx} y {vy}: ρ = {rho:.2f} "
               f"({'fuerte' if abs(rho) > 0.5 else 'moderada' if abs(rho) > 0.3 else 'débil'}). El color muestra {vc}."]
