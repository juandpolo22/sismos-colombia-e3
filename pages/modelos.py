"""Sección 3 · Modelos: tabla de líderes, predicción en vivo con explicación local, diagnósticos, interpretabilidad,
comparación estadística y el modelo original."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

from src import artefactos as A
from src import data as D
from src import plots as pl
from src.datos import VARIABLES_BASE
from src import presentacion_contenido as P
from src.original_model import COLUMNAS, PSEUDOCODIGO

dash.register_page(__name__, path="/modelos", name="Modelos", order=2)

CFG = {"displayModeBar": False, "responsive": True}
AB = "A+B v2"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

PSEUDO = PSEUDOCODIGO


def _t(titulo, sub, *cuerpo):
    return html.Div([html.H2(titulo), html.P(sub, className="sub"), *cuerpo], className="tarjeta")


def _g(id_, alto=380):
    return dcc.Loading(dcc.Graph(id=id_, config=CFG, style={"height": f"{alto}px"}), type="dot", color=pl.AZUL)


def _celdas_opciones():
    F = D.panel()
    F = F[F.y.isna()].sort_values("n_12m", ascending=False)
    return [{"label": f"({r.lat:.2f}, {r.lon:.2f}) · {str(r.depto).title()} · {int(r.n_12m)} sismos en 12 meses", "value": int(r.celda)}
            for r in F.itertuples()]


def _campo(e):
    paso = 1 if e["entero"] else 0.01
    return dbc.Col([
        dbc.Label(e["columna"], html_for=f"in-{e['columna']}", className="mb-0 small fw-bold", style={"wordBreak": "break-all"}),
        dbc.Input(id=f"in-{e['columna']}", type="number", min=e["min_valido"], max=e["max_valido"], step=paso, size="sm",
                  placeholder="vacío = sin sismos" if e["admite_faltante"] else None),
        html.Div(f"{e['descripcion']} · observado [{e['min']:g} – {e['max']:g}]", className="text-muted", style={"fontSize": "11px"}),
    ], xs=6, md=4, lg=3, className="mb-2")


def bloque_en_vivo():
    """Tarjeta de predicción en vivo (formulario, predicción y explicación). Se reutiliza en la página de presentación."""
    esquema = D.esquema_formulario()
    return _t("Predicción en vivo", "Elija una celda: el formulario se llena con sus valores actuales; puede modificarlos y predecir. "
                       "Cada campo valida tipo y límites físicos; si un valor supera lo observado en el entrenamiento, se avisa que el pronóstico extrapola.",
                       dcc.Dropdown(id="p-celda", options=_celdas_opciones(), placeholder="Elija una celda", className="mb-2"),
                       dbc.Row([_campo(e) for e in esquema], className="g-1"),
                       dbc.Button("Predecir", id="p-boton", color="primary", className="mt-1"),
                       html.Div(id="p-alerta", className="mt-2"),
                       dcc.Loading(html.Div(id="p-salida", className="mt-2"), type="dot", color=pl.AZUL))


def _lideres_card():
    L = D.lideres()
    vis = ["modelo", "balanceo", "optimizador", "RMSLE bucle externo (media ± d.e.)", "RMSLE validación < 2016",
           "RMSLE origen móvil 2021–2026", "inferencia (ms/fila)"]
    Lv = L[vis].copy()
    for c in ("RMSLE validación < 2016", "RMSLE origen móvil 2021–2026"):
        Lv[c] = Lv[c].round(4)
    return _t("Tabla de líderes", "Mejor combinación de cada modelo y el modelo original. Ordene con los encabezados y filtre escribiendo en la "
           "fila superior (por ejemplo «XGB» o «< 0.33»).",
           dash_table.DataTable(Lv.to_dict("records"), [{"name": c, "id": c} for c in vis], sort_action="native", filter_action="native",
                                style_table={"overflowX": "auto"}, style_cell={"fontSize": 13, "padding": "6px", "whiteSpace": "normal"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"},
                                style_data_conditional=[{"if": {"filter_query": '{modelo} contains "A+B"'}, "backgroundColor": "#fbeee6", "fontWeight": "600"}]),
           html.Div([html.B("Criterio de selección: "), "en cada familia gana la combinación con menor RMSLE en la validación interna con datos "
                     "anteriores a 2016 (la misma que fija sus hiperparámetros); con ella se evalúa el origen móvil 2021–2026 y se hace la "
                     "comparación estadística. Clasificación no se incluye: la tarea del tablero es la regresión del número de sismos."], className="insight"))


def layout():
    """Sección única «Modelos»: A+B, pruebas, B, líderes, modelo original y, al final, la simulación en vivo."""
    dm, gw, spa, mcs, ab, reg = P._datos()
    return dbc.Container([
        html.H2("Modelos: A+B y las pruebas que lo respaldan", className="mt-3"),
        html.P("Pronóstico mensual de sismos M ≥ 2,5 por celda de 0,5° en Colombia · origen móvil 2021–2026 (69 meses).",
               className="text-muted"),
        P._explicacion_ab(),
        html.H4("2. Pruebas estadísticas: qué comprueba cada una", className="mt-4"),
        html.P("Cada prueba responde a la pregunta «¿la diferencia de error entre A+B y el campeón es real o es azar?». "
               "p menor a 0,05 significa que, si no hubiera diferencia, sería muy raro observar lo que observamos."),
        *P._pruebas(dm, gw, spa, mcs),
        html.H4("3. Comportamiento de B y comparación con combinaciones", className="mt-4"),
        *P._modelo_b_y_combinaciones(ab, reg),
        html.H4("4. Líderes y modelo original", className="mt-4"),
        _lideres_card(),
        _original(D.original()),
        html.H4("5. Simulación en vivo", className="mt-4"),
        html.P("Elija el modelo y una celda, revise o cambie sus valores, y pulse «Predecir»."),
        P._tabla_entradas_salidas(),
        dbc.Row([dbc.Col(html.Div([
            html.Label("Modelo para la predicción", className="fw-bold"),
            dcc.Dropdown(id="m-sel", options=[{"label": "A+B v2 (modelo original)" if m == AB else m, "value": m} for m in D.modelos()],
                         value=AB, clearable=False, className="mb-3"),
        ]), md=6)]),
        dbc.Row([dbc.Col(bloque_en_vivo(), lg=12)], className="mt-2"),
    ], fluid=True, className="pagina-pres pb-4")


def _comparacion():
    M = D.tabla("e2_stats_mcs_origen_movil.csv").rename(columns={"perdida_media": "pérdida media", "rango_perdida": "rango", "en_MCS": "¿en el conjunto?"})
    M["¿en el conjunto?"] = M["¿en el conjunto?"].map({True: "sí", False: "no"})
    M["pérdida media"], M["p_MCS"] = M["pérdida media"].round(5), M["p_MCS"].round(4)
    return _t("Comparación estadística: Model Confidence Set", "Universo: los 7 campeones, el modelo original y las líneas base; pérdida = error "
              "cuadrático en log1p promediado por mes (69 meses); α = 0,10; bootstrap estacionario.",
              dash_table.DataTable(M.to_dict("records"), [{"name": c, "id": c} for c in M.columns], sort_action="native",
                                   style_cell={"fontSize": 13, "padding": "5px"}, style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"},
                                   style_data_conditional=[{"if": {"filter_query": '{¿en el conjunto?} = "sí"'}, "backgroundColor": "#fbeee6", "fontWeight": "600"}]),
              html.Div([html.B("Lectura: "), "el conjunto de confianza al 90 % contiene solo al modelo original: los demás son significativamente "
                        "peores tras controlar todas las comparaciones a la vez."], className="insight"))


def _tabla_pruebas():
    dm, gc = D.tabla("e2_stats_dm.csv").set_index("rival"), D.tabla("e2_stats_gw_cw.csv").set_index("rival")
    mcs = D.tabla("e2_stats_mcs_origen_movil.csv")
    peor = dm["p Holm (7)"].idxmax()
    fmt = lambda p: "< 0,001" if p < 0.001 else f"{p:.4f}".replace(".", ",")  # noqa: E731
    filas = [
        {"prueba": "Model Confidence Set (Hansen et al., 2011)", "hipótesis nula": "todos los modelos del conjunto tienen igual pérdida esperada",
         "supuestos verificados": "pérdidas mensuales estacionarias y débilmente dependientes (bootstrap estacionario)",
         "estadístico": "Tmax", "p crudo": fmt(mcs.p_MCS.iloc[1]), "p ajustado": "no requiere (controla la familia)",
         "efecto": "—", "conclusión": "solo A+B en el conjunto al 90 %"},
        {"prueba": "Diebold-Mariano con corrección HLN", "hipótesis nula": "igual pérdida esperada que el campeón",
         "supuestos verificados": "d_t sin autocorrelación fuerte (EL, BG); varianza Newey-West",
         "estadístico": f"DM = {dm.loc[peor, 'DM-HLN']:.2f} (vs {peor})", "p crudo": fmt(dm.loc[peor, "p crudo"]),
         "p ajustado": f"Holm (7): {fmt(dm['p Holm (7)'].max())} (máx.)", "efecto": f"d = {dm.loc[peor, 'd de Cohen']:.2f}; mín. {dm['d de Cohen'].min():.2f}",
         "conclusión": "A+B mejor que los 7 campeones"},
        {"prueba": "Giacomini-White (habilidad condicional)", "hipótesis nula": "E[d_t | constante, d_{t−1}] = 0",
         "supuestos verificados": "ventana de reestimación finita (origen móvil anual)",
         "estadístico": f"χ²(2) = {gc.loc[peor, 'GW χ²(2)']:.1f} (vs {peor})", "p crudo": fmt(gc.loc[peor, "p GW"]),
         "p ajustado": f"Holm (7): {fmt(gc['p GW Holm'].max())} (máx.)", "efecto": "—", "conclusión": "A+B mejor en los 7"},
    ]
    T = pd.DataFrame(filas)
    return dash_table.DataTable(T.to_dict("records"), [{"name": c, "id": c} for c in T.columns], style_table={"overflowX": "auto"},
                                style_cell={"fontSize": 12.5, "padding": "6px", "whiteSpace": "normal", "height": "auto", "textAlign": "left",
                                            "minWidth": "110px"},
                                style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"})


def _tabla_reglas():
    R = D.tabla("reglas_combinacion.csv").copy()
    R = R[~R["regla"].str.contains("stacking|sin calibración", case=False)]
    R["veredicto"] = np.where(R["p Holm"].isna(), "—", np.where(R["p Holm"] < 0.05, np.where(R["dif. pérdida vs A+B"] > 0, "A+B mejor", "A+B peor"),
                                                                  "sin diferencia significativa"))
    for c in ("RMSLE 2021–2026", "dif. pérdida vs A+B", "p DM", "p Holm"):
        R[c] = R[c].round(4)
    return html.Div([
        dash_table.DataTable(R.to_dict("records"), [{"name": c, "id": c} for c in R.columns], sort_action="native",
                             style_table={"overflowX": "auto"}, style_cell={"fontSize": 12.5, "padding": "5px"},
                             style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"},
                             style_data_conditional=[{"if": {"filter_query": '{regla} contains "A+B"'}, "backgroundColor": "#fbeee6", "fontWeight": "600"}]),
        html.Div([html.B("Lectura: "), "con los mismos miembros, la regla propia supera al mejor miembro, a los pesos por error global y al "
                  "promedio simple; la ventaja frente a los campeones viene sobre todo de los miembros (en particular del componente "
                  "propio B) y de combinarlos con pesos que cambian según el contexto."], className="insight")])


def _original(o):
    curva = np.array(o["curva_validacion_B"])
    f = go.Figure(go.Scatter(x=np.arange(1, len(curva) + 1), y=curva, mode="lines", line=dict(color=pl.NARANJA)))
    f.add_vline(x=o["n_iter_B"], line_dash="dot", line_color=pl.GRIS, annotation_text=f"parada: {o['n_iter_B']} iteraciones")
    f.update_layout(xaxis_title="iteración del boosting B", yaxis_title="NLL ZINB en validación (20 % final)", margin=dict(t=20))
    pesos = pd.Series(o["pesos_medios"]).sort_values(ascending=False)
    fw = pl.barras_h(pesos.index, pesos.values, "", "peso medio w_j", resaltar=lambda e: "propio" in e)
    dif = pd.DataFrame([
        {"método afín": "FFORMA (Montero-Manso et al., 2020)", "qué conserva A+B": "pesos por meta-características",
         "qué es nuevo": "pérdida en log1p para conteos, miembros ML + ZINB, OOF cronológico y calibración"},
        {"método afín": "Gradient boosting (Friedman, 2001)", "qué conserva A+B": "aditivo en árboles con pasos de Newton",
         "qué es nuevo": "pérdida ZINB con dos predictores (media y ceros) y dispersión re-estimada"},
        {"método afín": "Regresión ZINB / ETAS", "qué conserva A+B": "verosimilitud de conteos con exceso de ceros; excitación",
         "qué es nuevo": "estimada por boosting no paramétrico y combinada con ML"},
    ])
    return html.Div([
        html.H2("Modelo original A+B v2"),
        html.P("Combinador contextual (A) de pronosticadores que incluye un boosting propio de la verosimilitud binomial negativa inflada "
               "en ceros (B), con calibración final. Clase compatible con scikit-learn en src/original_model.py.", className="sub"),
        dbc.Row([
            dbc.Col([html.H3("Pseudocódigo"), html.Pre(PSEUDO, className="pseudo")], lg=7),
            dbc.Col([html.H3("Diferencias con la literatura"), dash_table.DataTable(dif.to_dict("records"), [{"name": c, "id": c} for c in dif.columns],
                     style_cell={"fontSize": 12.5, "whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                     style_header={"fontWeight": "700", "backgroundColor": "#f4f1ec"}),
                     html.P("Tabla preliminar: la matriz de novedad completa está en el capítulo 4 del libro.", className="sub mt-1")], lg=5),
        ]),
        dbc.Row([
            dbc.Col([html.H3("Curva de convergencia de B"), dcc.Graph(figure=f, config=CFG, style={"height": "320px"})], lg=6),
            dbc.Col([html.H3("Peso medio de cada miembro en A"), dcc.Graph(figure=fw.update_layout(margin=dict(l=10, r=50, t=10, b=40)),
                                                                            config=CFG, style={"height": "320px"})], lg=6),
        ]),
        html.H3("Regla de combinación propia frente a reglas estándar (mismos miembros)"),
        _tabla_reglas(),
        html.H3("Pruebas de hipótesis contra los campeones (69 meses, 2021–2026)", className="mt-3"),
        html.P("Para cada prueba se muestra el campeón más difícil (el de mayor p ajustado) y el máximo de los p ajustados entre los 7.", className="sub"),
        _tabla_pruebas(),
    ], className="tarjeta")


@callback([Output(f"in-{c}", "value") for c in VARIABLES_BASE], Input("p-celda", "value"), prevent_initial_call=True)
def llenar(celda):
    if celda is None:
        return [no_update] * len(VARIABLES_BASE)
    F = D.panel()
    r = F[(F.celda == celda) & F.y.isna()].iloc[0]
    return [None if pd.isna(r[c]) else round(float(r[c]), 4) for c in VARIABLES_BASE]


@callback([Output("p-alerta", "children"), Output("p-salida", "children")], Input("p-boton", "n_clicks"),
          [State("p-celda", "value"), State("m-sel", "value")] + [State(f"in-{c}", "value") for c in VARIABLES_BASE], prevent_initial_call=True)
def predecir(_, celda, m, *valores):
    if celda is None:
        return dbc.Alert("Elija primero una celda: sus variables de contexto (excitación e historia) se toman de ella.", color="warning"), no_update
    errores, avisos = [], []
    for e, v in zip(D.esquema_formulario(), valores):
        if v is None and not e["admite_faltante"]:
            errores.append(f"{e['columna']}: falta el valor o no es válido (límites [{e['min_valido']:g}, {e['max_valido']:g}])")
        elif v is not None and not (e["min_valido"] - 1e-9 <= v <= e["max_valido"] + 1e-9):
            errores.append(f"{e['columna']}: {v} está fuera de los límites válidos [{e['min_valido']:g}, {e['max_valido']:g}]")
        elif v is not None and e["entero"] and abs(v - round(v)) > 1e-9:
            errores.append(f"{e['columna']}: debe ser un número entero")
        elif v is not None and not (e["min"] - 1e-9 <= v <= e["max"] + 1e-9):
            avisos.append(f"{e['columna']} = {v:g} (observado en el entrenamiento: [{e['min']:g}, {e['max']:g}])")
    if errores:
        return dbc.Alert([html.B("Revise las entradas:"), html.Ul([html.Li(x) for x in errores])], color="danger"), no_update
    aviso = (dbc.Alert([html.B("Extrapolación: "), "estos valores superan lo observado en el entrenamiento; el pronóstico es menos confiable: ",
                        "; ".join(avisos)], color="warning") if avisos else None)
    try:
        F = D.panel()
        fila = F[(F.celda == celda) & F.y.isna()].iloc[[0]].copy()
        for c, v in zip(VARIABLES_BASE, valores):
            fila[c] = np.nan if v is None else float(v)
        fila["h0"], fila["h11"] = fila["n_1m"], fila["n_mismo_mes_anio_prev"]       # coherencia con la historia
        mod = A.modelo(m)
        X = fila[COLUMNAS] if m == AB else fila[VARIABLES_BASE]
        p = float(np.clip(mod.predict(X), 0, None)[0])
        lo, hi = D.intervalo(m, p)
        if m == AB:
            E = mod.explicar(X).iloc[0].drop("calibración (c0)")
            base_txt = (f"cada barra es el aporte c1·w·f del miembro; las barras y c0 = {mod.c0_:.3f} suman el pronóstico en escala "
                        f"log1p ({np.log1p(p):.2f})")
            nombres, val = E.index, E.values
            eje = "aporte al pronóstico log1p (c1·w·f)"
        else:
            from src.explicacion import shap_filas, valor_base
            S = shap_filas(mod, X, D.fondo_shap()[VARIABLES_BASE], max_evals=300, n_fondo=50)[0]
            orden = np.argsort(-np.abs(S))[:10]
            nombres, val = np.array(VARIABLES_BASE)[orden], S[orden]
            base_txt = f"valor base (pronóstico log1p medio del fondo) = {valor_base(mod, D.fondo_shap()[VARIABLES_BASE].iloc[:100]):.3f}"
            eje = "SHAP (escala log1p)"
        fig = go.Figure(go.Bar(y=list(nombres)[::-1], x=list(val)[::-1], orientation="h",
                               marker_color=[pl.NARANJA if v > 0 else pl.AZUL for v in list(val)[::-1]],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(xaxis_title=eje, height=360, margin=dict(l=10, r=10, t=10, b=40))
        salida = html.Div([
            html.Div([html.Span("Pronóstico: ", className="fw-bold"), html.Span(f"{p:.2f} sismos M ≥ 2,5", className="cifra"),
                      html.Div(f"Intervalo de predicción al 80 %: {lo[0]:.1f} – {hi[0]:.1f} sismos (cuantiles de los errores fuera de muestra "
                               f"2021–2026 del mismo tramo de pronóstico)", className="sub")]),
            html.H3("Explicación local", className="mt-2"),
            html.P((f"{base_txt}." if m == AB else f"Barras naranjas suben el pronóstico y azules lo bajan respecto del {base_txt}."), className="sub"),
            dcc.Graph(figure=fig, config=CFG),
        ])
        return aviso, salida
    except Exception as e:  # el error se muestra; la aplicación sigue funcionando
        return dbc.Alert(f"No se pudo calcular la predicción: {e}", color="danger"), no_update


