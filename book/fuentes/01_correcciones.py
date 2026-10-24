# %% [markdown]
# # 1. Trazabilidad de correcciones del Entregable 2
#
# Cada observación del docente sobre el Entregable 2 tiene una fila con la acción realizada, su ubicación en este libro y
# la evidencia. Si una corrección cambia un resultado, todas las tablas, figuras y pruebas que dependen de él se
# recalculan (los notebooks se ejecutan de nuevo desde los datos).

# %% [cabecera]

# %% [markdown]
# ```{admonition} Pendiente
# :class: warning
# Esta tabla se completa cuando llegue la retroalimentación del Entregable 2. Las filas de abajo son las observaciones de
# la revisión previa del docente (comparación estadística) y su estado actual.
# ```

# %%
filas = [
    ("Validez de Diebold-Mariano con 69 meses", "Pérdida agregada por mes, varianza Newey-West (4 rezagos) y corrección HLN; pruebas de independencia de d_t",
     "cap. 6 (y 2.ª parte, cap. 21)", "tabla de DM y de independencia de d_t"),
    ("Giacomini-White y SPA/Reality Check", "GW y Clark-West contra los 7 campeones; SPA y Reality Check por campeón con Holm",
     "cap. 6", "tablas de GW/CW y SPA"),
    ("Dependencia espacial y temporal de las celdas", "Unidad de análisis = mes; Escanciano-Lobato, Breusch-Godfrey, BDS y Moran por mes",
     "cap. 6", "tabla de independencia"),
    ("Holm en la misma familia de pruebas", "Holm sobre 7 comparaciones por prueba y sobre la familia de 21 contrastes",
     "cap. 6", "tabla de Holm (21)"),
    ("Datos usados en cada etapa respecto a 2021–2026", "Tabla de qué se estimó con qué datos; scripts reproducibles",
     "cap. 5 y 9", "tabla de protocolo"),
    ("Trazabilidad: diferencias por periodo, p, efecto, IC", "Diferencias mensuales d_t, p crudos y ajustados, d de Cohen, IC bootstrap",
     "cap. 6", "tabla y figura de d_t"),
    ("Ridge y Lasso con resultados idénticos", "Verificación: distinto α y predicciones distintas (correlación 0,99999999)",
     "cap. 6", "tabla de verificación"),
]
tabla(pd.DataFrame(filas, columns=["observación del docente", "acción realizada", "ubicación en el libro", "evidencia"]))
