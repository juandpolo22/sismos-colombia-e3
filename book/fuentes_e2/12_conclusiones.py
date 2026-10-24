# %% [markdown]
# # 12. Conclusiones del Entregable 1, limitaciones y checklist

# %% [cabecera]

# %%
# Resumen ejecutivo a partir de las tablas guardadas por los capítulos anteriores
res = pd.read_csv(cfg.TABLAS / "resultados_prueba.csv").set_index("modelo")
ic = pd.read_csv(cfg.TABLAS / "intervalos_bootstrap.csv")
ic = ic[ic["métrica"] == "RMSLE"].set_index("modelo")
comp = pd.read_csv(cfg.TABLAS / "diferencias_vs_triviales.csv")
resumen = pd.read_csv(cfg.TABLAS / "resumen_proyecto.csv")
md(f"""## Resumen ejecutivo

- **Pregunta.** Con el catálogo del SGC (2004–2026, {resumen.valor[0]} sismos M ≥ 2,5), ¿cuántos sismos habrá el próximo mes en cada una de las
  {resumen.valor[1]} celdas activas de 0,5°?
- **Diseño.**
  - Panel celda × mes de {resumen.valor[3]} filas.
  - Partición cronológica con gap: entrenamiento hasta noviembre de 2020; prueba de enero de 2021 a septiembre de 2026.
  - Validación cruzada temporal de ventana creciente.
  - Métrica principal: **RMSLE**.
- **Resultado del modelo base.** El SVR lineal (Pipeline con log1p, imputación y escalado) obtiene **RMSLE = {es(res.loc['SVR lineal', 'RMSLE'], 3)}**
  (IC 95 % BCa {es(ic.loc['SVR lineal', 'IC_BCa_inf'], 3)}–{es(ic.loc['SVR lineal', 'IC_BCa_sup'], 3)}).
  - Mejora a la mejor línea base trivial (media por celda, {es(res.loc['Media por celda', 'RMSLE'], 3)}) y al Dummy (mediana 0,
    {es(res.loc['Dummy (mediana = 0)', 'RMSLE'], 3)}).
  - La mejora es significativa frente a las {int((comp.veredicto == 'SVR mejor').sum())} líneas base (Diebold-Mariano HLN con Holm).
- **R² en log = {es(res.loc['SVR lineal', 'R2_log'], 3)}.** Desempeño moderado y creíble, sin señales de fuga.""")

# %% [markdown]
# ## Métrica elegida: RMSLE, y por qué
#
# - **Naturaleza del objetivo.** Es un conteo con 76 % de ceros y enjambres de cientos de sismos. El RMSLE
#   mide el error **relativo**, que es lo relevante para una tasa de actividad, y no se deja dominar por
#   unos pocos meses extremos, como sí le pasa al RMSE.
# - **Por qué no MAPE.** Se indefine con ceros.
# - **Cuándo se fijó.** Antes de evaluar, y no se cambió.
# - **Métricas complementarias.** RMSE, MAE, MAPE (solo y > 0), sMAPE, R² y R² en log se reportan siempre.
#   El RMSE cuenta otra historia, centrada en los meses extremos, y se discute en el capítulo 11.

# %% [markdown]
# ## Respuestas a las preguntas del enunciado
#
# | Requisito | Respuesta breve |
# |---|---|
# | Problema y justificación | Pronóstico mensual de tasa sísmica por zona; insumo para gestión del riesgo; datos oficiales y abiertos a consulta |
# | Tamaño | 42.380 filas celda-mes (≥ 20.000); 18 predictoras; n/p ≈ 2.354; 163 celdas × 260 meses (tamaño efectivo menor) |
# | Estructura | Proceso puntual agregado en panel espacio-temporal regular (ruta D) |
# | Faltantes | `depto` MAR por ubicación (mar y frontera, excluidos por homogeneidad entre portales); magnitudes de ventana: faltante estructural (Little rechaza MCAR) |
# | Duplicados | 11 exactos eliminados; casi-duplicados < 0,1 %, conservados (sismos reales muy seguidos) |
# | Atípicos | Reales (enjambres, nido de Bucaramanga, cola de Gutenberg-Richter): se conservan y se usa escala log |
# | Imposibles | Ninguno; profundidades negativas plausibles; profundidades fijadas en 0 km documentadas |
# | Sesgos | Completitud por debajo de M 2,9 (estable en el tiempo); sin mar ni fronteras; menor detección en zonas con pocas estaciones |
# | Fuga | Auditoría limpia (cap. 9); demostración con fuga deliberada |
# | Validación | Cronológica con gap; CV temporal; bloques espaciales con buffer como diagnóstico |

# %% [markdown]
# ## Hallazgos del EDA que guiaron el modelado
#
# 1. **Objetivo inflado en ceros y con cola pesada** (cap. 3) → log1p, RMSLE y bootstrap por meses.
# 2. **Sobredispersión, sobre todo por heterogeneidad entre celdas** (cap. 3 y 6) → variables de historia
#    propia de la celda y, en el Entregable 2, modelos de conteo (Poisson, ZINB).
# 3. **Persistencia moderada hasta 12 meses y sin estacionalidad** (cap. 7) → ventanas de 1, 3, 6 y 12
#    meses; el calendario pesa ≈ 0.
# 4. **Autocorrelación espacial y agrupamiento espacio-temporal** (cap. 8: Moran, LISA, Knox) → rezago
#    espacial y, en el Entregable 2, intensidades de excitación tipo ETAS.
# 5. **Relaciones no lineales con la ubicación y heterogeneidad regional** (cap. 5 y 8), confirmadas por
#    los residuos del SVR (cap. 11) → modelos no lineales en el Entregable 2.
# 6. **Deriva moderada en la actividad acumulada** (cap. 7) → evaluación en varios cortes temporales.

# %% [markdown]
# ## Limitaciones
#
# - **Completitud.** El catálogo no está completo entre M 2,5 y ≈ 2,9. Los conteos subestiman la
#   actividad real, aunque de forma estable en el tiempo.
# - **Cobertura.** Los sismos en el mar y en las fronteras quedan fuera, y las zonas con pocas estaciones
#   están subrepresentadas.
# - **Escalas de magnitud.** Cambian entre portales. No se detectó un salto en la distribución, pero no
#   se puede descartar un efecto menor alrededor del umbral.
# - **Lo que el modelo no anticipa.** Las secuencias de réplicas que empiezan de golpe (por ejemplo,
#   diciembre de 2019 en el Meta) dominan los errores grandes.
# - **Alcance del modelo base.** Es lineal: deja heterocedasticidad, sesgos regionales (Moran de los
#   residuos medios por celda) y estructura no lineal (BDS) sin capturar.
# - **Prueba única.** La evaluación final es una sola prueba cronológica (2021–2026). En el Entregable 2
#   se añaden cortes de 2018 y 2023.

# %% [markdown]
# ## Usos apropiados y consideraciones éticas
#
# - **Para qué sirve.** El pronóstico es una **tasa esperada de actividad**: sirve para priorizar
#   monitoreo y comunicar niveles de actividad.
# - **Para qué no sirve.** No predice terremotos individuales ni su hora, lugar o magnitud. No sustituye
#   los análisis oficiales del SGC.
# - **Incertidumbre.** Toda comunicación del resultado debe incluir su incertidumbre. Debe advertir que
#   las zonas con menos estaciones están peor representadas.

# %% [markdown]
# ## Checklist del Entregable 1

# %%
# Lista de chequeo de la guía del Entregable 1 y dónde se cumple cada punto
check = pd.DataFrame([
    ("1", "Definición del problema, justificación, fuente, licencia", "Cap. 1 y 2"),
    ("1", "Diccionario de variables (tipo, unidad, significado)", "Cap. 2 · tables/diccionario_*.csv"),
    ("1", "Estructura: unidad, agregación, tipo", "Cap. 2"),
    ("1", "Tamaño ≥ 20.000; n/p; rango del objetivo; entidades independientes", "Cap. 1 y 2"),
    ("1", "Faltantes: patrón (matriz de nulos) y mecanismo (Little, comparación de distribuciones, AUC)", "Cap. 2"),
    ("1", "Duplicados exactos y casi-duplicados", "Cap. 2"),
    ("1", "Outliers: detección y tratamiento", "Cap. 2, 4 y 6"),
    ("1", "Valores imposibles o inconsistentes", "Cap. 2"),
    ("1", "Sesgos de muestreo y representatividad; ética", "Cap. 2, 8 y 12"),
    ("2", "Reserva del conjunto de prueba antes de decidir", "Cap. 1 (partición cronológica con gap)"),
    ("2.1", "Variable objetivo: distribución, asimetría, colas, transformación, tiempo y espacio", "Cap. 3"),
    ("2.2", "Univariado: tipos, cardinalidad, estadísticos, histogramas, asimetría, curtosis, normalidad, boxplots, categorías raras", "Cap. 4"),
    ("2.3", "Bivariado: hexbin, Pearson/Spearman, pruebas entre grupos con tamaño de efecto, χ²/Cramér, información mutua, heatmap, VIF, Holm/FDR", "Cap. 5"),
    ("2.4", "Multivariado: PCA, t-SNE (visual), Mahalanobis robusta, Isolation Forest, clustering", "Cap. 6"),
    ("2.5", "Auditoría de fuga: disponibilidad, proxies, R² univariado, entidades entre particiones", "Cap. 9"),
    ("2.6", "Temporal: validación, agregaciones, STL, ADF/KPSS, ACF/PACF/CCF, regímenes, calendario, deriva, panel", "Cap. 7"),
    ("2.7", "Espacial: CRS, mapa base, densidad, coropletas, Clark-Evans, Ripley, DBSCAN haversine, Moran, LISA, Gi*, MAUP, heterogeneidad", "Cap. 8"),
    ("2.8", "Espacio-temporal: mapas por periodo, series por zona, Knox, persistencia de hotspots, Moran por año", "Cap. 8"),
    ("2.9", "Preprocesamiento en Pipeline ajustado solo con entrenamiento; decisiones trazadas al EDA", "Cap. 10"),
    ("3", "Línea base trivial (Dummy, persistencia, ingenuo estacional, media por celda) y SVR lineal", "Cap. 11"),
    ("3", "Validación cruzada del mismo tipo que la partición (temporal) y espacial por separado", "Cap. 11"),
    ("3", "Métricas RMSE, MAPE, R² (+ RMSLE, MAE, sMAPE) e IC bootstrap (percentil y BCa)", "Cap. 11"),
    ("3", "Residuos: normalidad, heterocedasticidad (White/BP), ACF + Ljung-Box, BDS, I de Moran", "Cap. 11"),
    ("3", "Curva de aprendizaje; interpretación de coeficientes; nota crítica", "Cap. 11"),
    ("Entregables", "Jupyter Book, datos (data/), requirements.txt, semilla 42", "Este repositorio"),
], columns=["sección", "requisito", "dónde"])
eda.guardar_tabla(check, "checklist_entregable1.csv")
check

# %% [markdown]
# ## Reproducibilidad: semillas y versiones
#
# - **Semilla global:** `SEMILLA = 42` en `src/config.py`. Se propaga con `cfg.fijar_semillas()` (random,
#   numpy, `PYTHONHASHSEED`) y como `random_state` en cada estimador.
# - **Versiones exactas:** `requirements.txt`, verificadas con **Python 3.9.25**.
# - **Ejecución completa desde los datos crudos:** `python herramientas/construir_notebooks.py`.
# - **Datos:** `data/crudos/` (archivos originales del SGC) y `data/sismos_sgc_2004_2026.csv` (salida
#   de la ETL).

# %%
# Versiones del entorno y semilla usadas
import platform, sklearn, statsmodels, scipy
pd.DataFrame([("Python", platform.python_version()), ("numpy", np.__version__), ("pandas", pd.__version__),
              ("scikit-learn", sklearn.__version__), ("statsmodels", statsmodels.__version__), ("scipy", scipy.__version__),
              ("semilla", cfg.SEMILLA)], columns=["componente", "versión"])

# %% [markdown]
# ## Referencias
#
# ```{bibliography}
# :all:
# ```
