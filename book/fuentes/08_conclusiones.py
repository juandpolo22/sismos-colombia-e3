# %% [markdown]
# # 8. Conclusiones, limitaciones y declaración de uso de herramientas

# %% [cabecera]

# %% [markdown]
# ## 8.1 Hallazgos
#
# - El número mensual de sismos por celda es un conteo con exceso de ceros, sobredispersión, persistencia temporal y
#   dependencia espacial; los modelos que explotan la actividad reciente de la celda y de sus vecinas dominan.
# - Entre los benchmarks, XGBoost y Random Forest son los mejores; las diferencias entre métodos de optimización de una
#   misma familia son pequeñas frente a la variabilidad entre pliegues.
# - El modelo original A+B v2 tiene el menor error y supera con significancia al campeón de cada familia; su componente
#   propio B aporta una mejora significativa (ablación).
#
# ## 8.2 Veredicto sobre R1, R2 y R3

# %%
from src import data as D
dm = D.tabla("e2_stats_dm.csv")
tabla(pd.DataFrame([
    ("R1 · Novedad en la aplicación", "novedad acotada y documentada: ningún estudio recuperado integra catálogo colombiano, ZINB por boosting, combinación contextual y evaluación con origen móvil; revisión con Consensus, LeapSpace, Scopus y Web of Science (mínimo de la guía cumplido: 3 bases y 5 cadenas propias (Scopus, WoS, IEEE))", "capítulo 4"),
    ("R2 · Implementación algorítmica propia", "sí: boosting ZINB con árboles CART propios, combinador softmax con gradiente analítico, "
     "calibración; clase compatible con scikit-learn; pruebas unitarias", "capítulo 5"),
    ("R3 · Superioridad estadística", f"sí: DM, GW, CW y SPA con Holm frente a los 7 campeones (p máx. {dm['p Holm (7)'].max():.4f} en DM); "
     f"único en el MCS; d ≥ {dm['d de Cohen'].min():.2f}", "capítulo 6"),
], columns=["condición", "veredicto", "evidencia"]))

# %% [markdown]
# ## 8.3 Limitaciones
#
# - La ventaja frente a XGBoost y Random Forest es estadísticamente significativa pero pequeña en magnitud.
# - Los meses con enjambres o grandes secuencias de réplicas producen los errores más grandes de todos los modelos.
# - El catálogo depende de la cobertura de la red sismológica (sesgo de detección en zonas con pocas estaciones).
# - La comparación usa el campeón de cada familia (elegido con datos anteriores a 2016), no las 28 combinaciones.
#
# ## 8.4 Trabajo futuro
#
# - Evaluar el modelo en otros catálogos y tareas de conteo (capítulo 5, evaluación amplia).
# - Aplicar las pruebas CSEP (N, S, M, L) y reglas de puntuación propias, y medir la sensibilidad a completitud, declustering y tamaño de celda (vacíos del capítulo 4).
# - Incorporar información física (fallas, deformación) y pronósticos probabilísticos completos (distribución, no solo media).
#
# ## 8.5 Declaración de autoría y uso de herramientas
#
# ```{admonition} Para completar y firmar por el equipo
# :class: important
# Declaramos que el diseño, la implementación y el análisis de este proyecto son de nuestra autoría. Se usaron las
# siguientes herramientas de inteligencia artificial: **[herramienta]** para **[usos concretos: p. ej., apoyo en la
# depuración de código, revisión de redacción, generación de borradores de funciones que luego revisamos y probamos]**.
# Comprendemos y podemos explicar y modificar cada parte del código entregado.
#
# Firmas: ______________________ (Juan David Polo) · ______________________ (Gabriella Pallares) · Fecha: __________
# ```
