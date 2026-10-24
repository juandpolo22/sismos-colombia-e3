"""Punto de entrada del dashboard (Dash multipágina). En Render se sirve con ``gunicorn app:server``.

Páginas: pages/contexto.py («/»), pages/eda.py («/eda»), pages/modelos.py («/modelos»).
"""
import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(
    __name__,
    use_pages=True,
    # Tema Bootstrap «Flatly» (Bootswatch 5.3.3, licencia MIT) servido desde assets/: no depende de una CDN externa.
    title="Pronóstico de sismos por celda",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([html.Span("●", className="marca-punto"), " Sismos Colombia · pronóstico mensual"], href="/"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Contexto", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("EDA", href="/eda", active="exact")),
            dbc.NavItem(dbc.NavLink("Modelos", href="/modelos", active="exact")),
        ], navbar=True, className="ms-auto"),
    ], fluid=True),
    color="dark", dark=True, sticky="top", className="barra",
)

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid=True, className="contenido"),
    html.Footer(dbc.Container([
        html.Span("Juan David Polo y Gabriella Pallares · Machine Learning, Universidad del Norte · "),
        html.Span("Datos: Servicio Geológico Colombiano (catálogo sísmico 2004–2026) · versión 3.0 · octubre de 2026"),
    ], fluid=True), className="pie"),
])

if __name__ == "__main__":
    app.run(debug=True)
