"""Contenido de la sección «Modelos»: primero la explicación de A+B, luego cada prueba
estadística con sus resultados y su conclusión, después el comportamiento de B y las combinaciones, y al final la
simulación en vivo. Todos los números salen de data/app/."""
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from src import data as D
from src.datos import VARIABLES_BASE


COL_AB, COL_OTRO, COL_NEUTRO = "#c4561b", "#1f6f78", "#7a8394"
AB_RMSLE = 0.32596947199891413  # RMSLE de A+B v2 en el origen móvil 2021–2026 (lideres)
CFG = {"displayModeBar": False, "responsive": True}


def _n(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def _p(valor):
    return "< 0,0001" if valor < 0.0001 else _n(valor)


def _layout_fig(fig, alto, margen_izq=170, color="#000000"):
    """Formato común de las gráficas. color: letra de ejes y etiquetas (blanco en las pruebas, negro en el resto)."""
    fig.update_layout(height=alto, margin=dict(l=margen_izq, r=30, t=10, b=45), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=color))
    fig.update_xaxes(tickfont=dict(color=color), title_font=dict(color=color))
    fig.update_yaxes(tickfont=dict(color=color), title_font=dict(color=color))
    return fig


def _conclusion(texto):
    return html.Div([html.B("Conclusión: "), texto], className="conclusion")


def _caja_prueba(numero, nombre, quien, para_que, h0, h1, contenido):
    return html.Div([
        html.Span(f"Prueba {numero}", className="etiqueta etiqueta-a"),
        html.H3(nombre, className="h5 mt-2"),
        html.P([html.B("Quién la propone: "), quien], className="mb-1"),
        html.P([html.B("Para qué sirve: "), para_que], className="mb-1"),
        html.Div([
            html.Div([html.B("H0 (hipótesis nula): "), h0], className="hipo"),
            html.Div([html.B("H1 (alternativa): "), h1], className="hipo"),
        ], className="mb-2"),
        *contenido,
    ], className="tarjeta-prueba")


def _datos():
    dm = D.tabla("e2_stats_dm.csv").set_index("rival")
    gw = D.tabla("e2_stats_gw_cw.csv").set_index("rival")
    spa = D.tabla("e2_stats_spa.csv").set_index("benchmark")
    mcs = D.tabla("e2_stats_mcs_origen_movil.csv")
    ab = D.tabla("e2_ab_ablacion.csv")
    reg = D.tabla("reglas_combinacion.csv")
    return dm, gw, spa, mcs, ab, reg


def _explicacion_ab():
    return html.Div([
        html.H4("1. Qué hicimos: el modelo A+B en palabras sencillas", className="mt-4"),
        html.P("Tenemos siete pronosticadores conocidos (los benchmarks) y dos piezas propias: B, que aprende a contar "
               "sismos, y A, que decide cuánto creerle a cada pronosticador según la situación de cada celda."),
        html.Div([html.Span(t, className="paso-flujo") for t in [
            "Historia de la celda y sus vecinas", "B: estima si la celda está activa y cuántos sismos",
            "Los 7 benchmarks + B dan su opinión", "A: pondera cada opinión según el contexto",
            "Calibración: corrige la escala", "Sismos esperados + intervalo"]], className="flujo-pres"),
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("Modelo A", className="etiqueta etiqueta-a"),
                html.H3("El coordinador: decide a quién creerle", className="h5 mt-1"),
                html.Ul([
                    html.Li("Recibe las predicciones de los benchmarks, de B y de un núcleo de conteos."),
                    html.Li("Decide cuánto peso darle a cada uno en cada celda y mes, según el contexto (actividad de la celda, "
                            "réplicas recientes, magnitud máxima, época del año)."),
                    html.Li("Los pesos son positivos y suman 1: el resultado es un promedio ponderado."),
                    html.Li(f"Aporte medido: sin B, el error sube de {_n(AB_RMSLE)} a {_n(0.32665327723946325)}."),
                ], className="pres-lista"),
            ], className="tarjeta-modelo"), xs=12, md=6),
            dbc.Col(html.Div([
                html.Span("Modelo B", className="etiqueta etiqueta-b"),
                html.H3("El experto que aprende a contar", className="h5 mt-1"),
                html.Ul([
                    html.Li("Un boosting de árboles propios (programados desde cero) que aprende, por celda, la probabilidad "
                            "de que no pase nada y, si hay actividad, cuántos sismos se esperan."),
                    html.Li("Distribución: binomial negativa inflada en ceros. Admite que los conteos varían más de lo que "
                            "predice la Poisson."),
                    html.Li("Salida: la esperanza de log(1 + sismos), que es exactamente lo que mide el RMSLE."),
                    html.Li(f"Aporte medido: solo, B tiene un error de {_n(0.33423956767416074)}, peor que el conjunto."),
                ], className="pres-lista"),
            ], className="tarjeta-modelo"), xs=12, md=6),
        ], className="g-3"),
        html.P([html.B("Por qué los unimos: "), "los benchmarks son buenos en promedio, pero ninguno gana siempre; B modela muy "
                "bien los ceros y la dispersión, pero solo pierde. A aprende cuándo conviene cada uno."], className="mt-3"),
        html.P([html.B("Versiones: "), f"v1 combina los miembros con A y entrega la predicción directamente; v2 agrega la "
                "calibración final, que corrige el aplastamiento de los promedios. v2 es la final porque mejora la validación "
                "previa a 2021 (RMSLE de 0,3152 a 0,3144). v1 no se descartó por fallar: se reporta junto a v2."]),
        _conclusion(f"A+B v2 tiene un RMSLE de {_n(AB_RMSLE)} en el origen móvil 2021–2026, frente a 0,3270 del mejor "
                    "benchmark (XGBoost). Es una mejora pequeña, pero consistente, y la confirmamos con las pruebas siguientes."),
    ])


def _pruebas(dm, gw, spa, mcs):
    rivales = list(dm.index)
    # Prueba 1: Diebold–Mariano
    dif = [float(dm.loc[r, "dif. media pérdida"]) for r in rivales]
    ic = [[float(x) for x in dm.loc[r, "IC95 bootstrap estacionario"].strip("[]").split(",")] for r in rivales]
    fig1 = go.Figure(go.Scatter(
        x=dif, y=rivales, mode="markers", marker=dict(color=COL_AB, size=10),
        error_x=dict(type="data", symmetric=False, array=[h - d for d, (_, h) in zip(dif, ic)],
                     arrayminus=[d - lo for d, (lo, _) in zip(dif, ic)], color=COL_OTRO),
        hovertemplate="%{y}<br>diferencia %{x:.4f}<extra></extra>"))
    fig1.add_vline(x=0, line_dash="dot", line_color="#333")
    fig1.update_layout(xaxis=dict(title="Pérdida del campeón menos pérdida de A+B (positivo = A+B gana)"), yaxis=dict(autorange="reversed"))
    prueba1 = _caja_prueba(
        1, "Diebold–Mariano con corrección de Harvey–Leybourne–Newbold",
        "Diebold y Mariano (1995); corrección de Harvey, Leybourne y Newbold (1997).",
        "compara la precisión de dos pronósticos usando la diferencia de pérdida mes a mes.",
        "E[d_t] = 0: los dos modelos tienen la misma pérdida esperada.",
        "E[d_t] ≠ 0: uno de los dos tiene menor pérdida.",
        [dcc.Graph(figure=_layout_fig(fig1, 340, 170, color="#ffffff"), config=CFG),
         _conclusion(f"En los siete casos la diferencia es positiva: A+B pierde menos error que cada campeón. Todos los "
                     f"intervalos de confianza al 95 % quedan por encima de cero, y el p ajustado por Holm es menor a "
                     f"{_n(float(dm['p Holm (7)'].max()), 3)} en el peor caso. Se rechaza H0 en todos.")])

    # Prueba 2: Giacomini–White
    fig2 = go.Figure(go.Bar(x=[float(gw.loc[r, "p GW Holm"]) for r in rivales], y=rivales, orientation="h",
                            marker_color=COL_OTRO, text=[_p(float(gw.loc[r, "p GW Holm"])) for r in rivales],
                            textposition="outside"))
    fig2.add_vline(x=0.05, line_dash="dash", line_color=COL_AB)
    fig2.update_layout(xaxis=dict(title="p de Holm (menor es mejor). Línea naranja = 0,05", type="log"), yaxis=dict(autorange="reversed"))
    prueba2 = _caja_prueba(
        2, "Giacomini–White (capacidad predictiva condicional)",
        "Giacomini y White (2006).",
        "igual que Diebold–Mariano, pero permite que la superioridad dependa de la información disponible en cada momento. "
        "Es más realista cuando el modelo se reentrena cada año.",
        "E[d_t | información pasada] = 0: la capacidad predictiva es igual dado lo que se sabía.",
        "E[d_t | información pasada] ≠ 0: la capacidad predictiva difiere.",
        [dcc.Graph(figure=_layout_fig(fig2, 340, 170, color="#ffffff"), config=CFG),
         _conclusion("Los siete p ajustados están por debajo de 0,05. La ventaja de A+B se mantiene aun cuando se condiciona "
                     "a la información del momento, así que no es un efecto de una ventana particular de meses.")])

    # Prueba 5: Model Confidence Set
    M = mcs.copy()
    filas = [html.Tr([html.Td(r.modelo), html.Td(_n(float(r.perdida_media), 5)), html.Td(_p(float(r.p_MCS))),
                      html.Td(html.Span("sí", className="si") if bool(r.en_MCS) else html.Span("no", className="no"))])
             for r in M.itertuples()]
    tabla5 = html.Table([html.Thead(html.Tr([html.Th("Modelo"), html.Th("pérdida media (log²)"), html.Th("p del MCS"), html.Th("¿En el conjunto?")])),
                         html.Tbody(filas)], className="tabla-pres")
    prueba5 = _caja_prueba(
        3, "Model Confidence Set (MCS)",
        "Hansen, Lunde y Nason (2011).",
        "no compara un modelo contra otro, sino todo el conjunto a la vez: elimina iterativamente el peor modelo hasta que "
        "no se pueda rechazar que los restantes son iguales. Lo que queda es el conjunto de los mejores.",
        "Todos los modelos del conjunto tienen la misma pérdida esperada.",
        "Al menos un modelo del conjunto es peor que otro.",
        [html.Div(tabla5, className="wrap-tabla"),
         _conclusion("El conjunto de confianza al 90 % contiene solo A+B v2. Los demás quedan fuera porque son significativamente "
                     "peores, tras controlar todas las comparaciones a la vez.")])

    return [prueba1, prueba2, prueba5]


def _modelo_b_y_combinaciones(ab, reg):
    nombres = list(ab["variante"])
    vals = [float(v) for v in ab["RMSLE (principal)"]]
    cols = [COL_AB if n.startswith("A+B") else COL_NEUTRO for n in nombres]
    fig = go.Figure(go.Bar(x=vals, y=nombres, orientation="h", marker_color=cols,
                           text=[_n(v) for v in vals], textposition="outside"))
    fig.update_layout(xaxis=dict(title="RMSLE (menor es mejor)", range=[0.318, 0.345]), yaxis=dict(autorange="reversed"))
    b_solo = [float(v) for n, v in zip(nombres, vals) if n.startswith("B solo")][0]
    bloque_b = html.Div([
        html.H3("Comportamiento de B dentro del modelo", className="h5 mt-4"),
        html.P("Se quita una pieza y se mide cuánto empeora el error. Mismo origen móvil y mismos miembros."),
        dcc.Graph(figure=_layout_fig(fig, 340, 300), config=CFG),
        _conclusion(f"B por sí solo (RMSLE {_n(b_solo)}) queda por detrás del conjunto (0,3260), aunque mejora al núcleo ZINB solo. Su valor está en "
                    "combinarse con los benchmarks: sin B, A+B pasa de 0,3260 a 0,3267, una diferencia pequeña pero significativa "
                    "(p = 0,001)."),
    ])
    reg = reg[~reg["regla"].str.contains("stacking|sin calibración", case=False)]
    nombres3 = list(reg["regla"])
    vals3 = [float(v) for v in reg["RMSLE 2021–2026"]]
    cols3 = [COL_AB if n.startswith("A+B") else COL_NEUTRO for n in nombres3]
    fig3 = go.Figure(go.Bar(x=vals3, y=nombres3, orientation="h", marker_color=cols3,
                            text=[_n(v) for v in vals3], textposition="outside"))
    fig3.update_layout(xaxis=dict(title="RMSLE (menor es mejor)", range=[0.32, 0.331]), yaxis=dict(autorange="reversed"))
    bloque_c = html.Div([
        html.H3("¿Aporta la combinación frente a reglas simples?", className="h5 mt-4"),
        html.P("Con los mismos miembros, comparamos A+B con reglas más sencillas."),
        dcc.Graph(figure=_layout_fig(fig3, 420, 300), config=CFG),
        _conclusion("A+B supera de forma significativa al mejor miembro individual, a los pesos por error global y al promedio "
                    "simple (p de Holm entre 0,00004 y 0,0006 sin calibración)."),
    ])
    return [bloque_b, bloque_c]


def _tabla_entradas_salidas():
    filas = [html.Tr([html.Td(c), html.Td(D.DESCRIPCION[c]), html.Td(D.UNIDADES[c])]) for c in VARIABLES_BASE]
    tabla = html.Table([html.Thead(html.Tr([html.Th("Variable"), html.Th("Qué significa"), html.Th("Unidad")])),
                        html.Tbody(filas)], className="tabla-pres")
    return html.Div([
        html.H3("Qué parámetros le entregamos y qué nos da", className="h5 mt-2"),
        html.P([html.B("Entradas: "), f"{len(VARIABLES_BASE)} variables de una celda y un mes. Se llenan solas con los "
                "valores históricos de la celda elegida, y se pueden cambiar para simular un escenario."]),
        html.Div(tabla, className="wrap-tabla"),
        html.P([html.B("Salidas: "), "(1) el número esperado de sismos M ≥ 2,5 en la celda el mes siguiente; (2) un intervalo "
                "al 80 % para ese número; (3) el aporte de cada miembro del modelo a la predicción; (4) una advertencia si algún "
                "valor queda fuera de lo observado en el entrenamiento (extrapolación)."], className="mt-2"),
        html.P([html.B("Qué no entrega: "), "un sí o no a que ocurra un sismo, ni la hora, ni la magnitud exacta."]),
    ])
