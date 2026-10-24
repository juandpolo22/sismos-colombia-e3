# %% [markdown]
# # 2. Datos: ETL y EDA
#
# Este capítulo reúne el ETL y el análisis exploratorio de los entregables anteriores (secciones 2.1 a 2.8, que se
# muestran ejecutadas con sus figuras e interpretación) y resume los hallazgos que guiaron el diseño del modelo original.
# El EDA interactivo está en la página **EDA** del dashboard.

# %% [cabecera]

# %% [markdown]
# ## Fuente y ETL
#
# - **Fuente:** catálogo sísmico de la Red Sismológica Nacional de Colombia, Servicio Geológico Colombiano (SGC),
#   portales histórico y actual (10 archivos), 2004-01 a 2026-09.
# - **Transformaciones:** unificación de esquemas, eliminación de duplicados entre portales, depuración de coordenadas y
#   profundidades imposibles, filtro de magnitud M ≥ 2,5 (completitud del catálogo) y asignación a celdas de 0,5°.
# - **Carga:** `data/sismos_sgc_2004_2026.csv` (catálogo limpio) y el panel celda × mes que construye `src/panel.py`.

# %%
from src import data as D
cat, P = D.catalogo(), D.panel()
obs = P[P.y.notna()]
tabla(pd.DataFrame([
    ("Sismos (M ≥ 2,5)", f"{len(cat):,}".replace(",", " ")),
    ("Periodo", f"{cat.time.min():%Y-%m} a {cat.time.max():%Y-%m}"),
    ("Celdas activas (≥ 20 sismos antes de 2021)", P.celda.nunique()),
    ("Filas celda × mes con objetivo", f"{len(obs):,}".replace(",", " ")),
    ("Variables predictoras (benchmarks)", 18),
    ("Variables adicionales del modelo original", "13 de excitación + 12 de historia"),
    ("Celdas-mes sin sismos", f"{100 * (obs.y == 0).mean():.1f} %"),
], columns=["concepto", "valor"]))

# %% [markdown]
# ## Hallazgos del EDA que guían el modelado
#
# 1. **Exceso de ceros y sobredispersión.** La mayoría de las celdas-mes no tiene sismos y la varianza supera con creces a
#    la media: motiva la pérdida binomial negativa inflada en ceros del modelo B y la métrica RMSLE.
# 2. **Persistencia temporal.** La actividad de un mes anticipa la del siguiente (ACF alta en rezagos cortos): motiva las
#    variables de rezago, la persistencia como miembro y la excitación tipo ETAS.
# 3. **Dependencia espacial.** Moran y Knox muestran agrupamiento y difusión entre celdas vecinas: motiva las variables de
#    vecindad y la excitación espacial.
# 4. **Heterogeneidad entre celdas y deriva.** Celdas muy activas (nido de Bucaramanga) conviven con celdas casi
#    inactivas, y hay cambios de régimen: motiva un combinador cuyos pesos dependen del contexto de cada celda (A) y el
#    reentrenamiento anual.
# 5. **Sin fuga.** La auditoría de fuga (sección 2.8) confirma que ninguna variable usa información del mes objetivo.
