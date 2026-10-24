# %% [markdown]
# # Pronóstico mensual de sismos por celda en Colombia: dashboard predictivo, modelo original e informe reproducible
#
# **Juan David Polo y Gabriella Pallares** · Machine Learning · Universidad del Norte · Entregable 3 · octubre de 2026
#
# **Dashboard desplegado:** `https://<servicio>.onrender.com` *(se completa al publicar en Render; ver capítulo 7)*
#
# **Repositorio:** `https://github.com/<usuario>/<repositorio>` *(ver capítulo 9)*

# %% [cabecera]

# %% [markdown]
# ## Resumen ejecutivo
#
# El proyecto pronostica el **número de sismos de magnitud M ≥ 2,5 que ocurrirán el mes siguiente en cada celda de
# 0,5° × 0,5° de Colombia**, a partir del catálogo sísmico del Servicio Geológico Colombiano (SGC) 2004–2026. Cada fila
# del panel es una celda activa y un mes; las variables usan solo lo ocurrido hasta el cierre del mes.
#
# - **Benchmarks (Entregable 2).** 7 familias de modelos × 4 métodos de optimización (Grid, Random, Bayesiano y Genético),
#   con validación anidada temporal y evaluación por origen móvil 2021–2026 (69 meses, reentrenando cada año).
# - **Modelo original A+B v2.** Un combinador contextual (A) que aprende, según el estado de cada celda, cuánto confiar en
#   cada pronosticador, entre ellos un **boosting de la verosimilitud binomial negativa inflada en ceros (B)** construido
#   sobre árboles CART programados desde cero, con una calibración final de Mincer-Zarnowitz.
# - **Resultado.** A+B v2 tiene el menor error del estudio y supera de forma estadísticamente significativa al campeón de
#   cada una de las 7 familias (Diebold-Mariano, Giacomini-White, Clark-West y SPA, con corrección de Holm), con tamaño del
#   efecto no despreciable.
# - **Producto.** Un dashboard en Dash Plotly, desplegado en Render, con contexto, EDA interactivo, pronóstico del mes
#   siguiente, predicción en vivo con explicación local y la validación estadística del modelo original.

# %%
from src import data as D
P = D.panel()
obs = P[P.y.notna()]
md(f"**Datos en cifras:** {len(D.catalogo()):,} sismos M ≥ 2,5; {P.celda.nunique()} celdas activas; {len(obs):,} filas celda × mes "
   f"con objetivo conocido ({obs.fecha_obj.min():%Y-%m} a {obs.fecha_obj.max():%Y-%m}); {100 * (obs.y == 0).mean():.1f} % de las "
   f"celdas-mes sin sismos. Mes pronosticado en el dashboard: {P[P.y.isna()].fecha_obj.max():%Y-%m}.".replace(",", " "))

# %% [markdown]
# ## Problema y objetivos
#
# **Pregunta de investigación.** ¿Puede un modelo que combina pronosticadores según el contexto de cada celda, con un
# componente propio de conteo inflado en ceros, pronosticar el número mensual de sismos M ≥ 2,5 por celda con un error
# menor, de forma estadísticamente significativa, que los mejores modelos de machine learning de cada familia?
#
# **Objetivos.** (1) Corregir y consolidar el Entregable 2; (2) seleccionar con criterio estadístico la mejor combinación
# de cada modelo y serializarla; (3) diseñar, implementar y validar el modelo original; (4) publicar un dashboard
# reproducible; (5) documentar todo en este libro.
#
# ## Mapa del libro
#
# | Cap. | Contenido |
# |---|---|
# | 1 | Trazabilidad de las correcciones del Entregable 2 |
# | 2 | Datos: ETL y EDA |
# | 3 | Modelos benchmark y mejores combinaciones |
# | 4 | Revisión de literatura y novedad |
# | 5 | Modelo original: formulación, pseudocódigo, complejidad, propiedades, código, pruebas y ablación |
# | 6 | Validación estadística |
# | 7 | Dashboard y despliegue |
# | 8 | Conclusiones, limitaciones y declaración de uso de herramientas |
# | 9 | Reproducibilidad y bibliografía |
