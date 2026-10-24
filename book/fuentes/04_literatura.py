# %% [markdown]
# # 4. Revisión de literatura y novedad
#
# La novedad del modelo original no se declara: se sustenta con una revisión documentada (guía, sección 5.2). Este capítulo
# reporta la búsqueda hecha el 23 de octubre de 2026 con dos motores asistidos por IA (Consensus y Elsevier LeapSpace) y con
# consultas directas en Scopus y Web of Science, la comparación
# con los trabajos más cercanos, la matriz de novedad, los vacíos encontrados, el estado de verificación de las referencias
# citadas y una declaración de novedad acotada a lo que la evidencia respalda.

# %% [cabecera]

# %% [markdown]
# ## 4.1 Protocolo de búsqueda
#
# | | Motor 1: Consensus | Motor 2: Elsevier LeapSpace |
# |---|---|---|
# | Fuentes que indexa | más de 220 millones de artículos (Semantic Scholar, PubMed y otras fuentes académicas) | contenido de ScienceDirect y Scopus |
# | Fecha de consulta | 23 de octubre de 2026 | 23 de octubre de 2026 |
# | Búsquedas | 26 búsquedas y 1 uso del grafo de citas (a partir de Marzocchi et al., 2012) | 1 consulta estructurada, con 73 referencias en el informe |
# | Embudo | 601,8 mil recuperados → 401 elegibles (tras eliminar duplicados) → 100 incluidos | no reporta embudo; devuelve síntesis con citas |
# | Criterio de relevancia | relevancia y calidad según el motor | verificación fuente por fuente; marca como «no verificable» lo que no puede confirmar |
#
# **Cuatro frentes de búsqueda** (los mismos en ambos motores):
#
# 1. aprendizaje automático para conteos o tasas de sismicidad en catálogos regionales;
# 2. ensambles y combinación de pronósticos sísmicos (CSEP; Marzocchi et al., 2012);
# 3. modelos de conteo con exceso de ceros y sobredispersión (ZIP, ZINB, *hurdle*) aplicados a sismicidad;
# 4. pronóstico y amenaza sísmica en Colombia (Servicio Geológico Colombiano y universidades).
#
# **Consulta enviada a LeapSpace** (texto literal): *«Conduct a systematic literature review on ML-based regional earthquake
# count/rate forecasting, seismic forecast ensembles (CSEP; Marzocchi et al., 2012), ZINB/zero-inflated seismicity models,
# and Colombian seismic forecasting (SGC/universities). Identify the 3–5 closest studies, compare them with A+B, and
# establish research gaps. Verify the 15 cited references, providing exact titles, authors, years, DOIs/full-text links.
# Never invent sources; flag unverifiable references.»*
#
# Ninguno de los dos informes describe a A+B más allá de su nombre, de modo que la comparación se hace aquí, con la
# descripción de los capítulos 5 y 6.

# %%
embudo = pd.DataFrame([
    ("Recuperados (26 búsquedas + 1 uso del grafo de citas)", 601800),
    ("Elegibles (relevancia y calidad, sin duplicados)", 401),
    ("Incluidos en el análisis final", 100),
], columns=["etapa (Consensus)", "artículos"])
tabla(embudo, dec=0)

# %% [markdown]
# ### Consultas directas en Scopus y Web of Science
#
# Además de los dos motores asistidos, se corrieron cadenas propias en Scopus y Web of Science y se exportaron los
# resultados. Web of Science limita cada exportación a 50 registros; el total de resultados de cada cadena se anotó desde
# la plataforma (136 y 80). La cadena que corresponde a cada archivo se dedujo de su contenido y debe
# confirmarse. La cadena 8 (IEEE Xplore) ya se ejecutó. Con las cinco cadenas de Scopus, Web of Science e IEEE Xplore se cumple el mínimo de la guía (tres bases y cinco cadenas); las de Google Scholar y arXiv son complementarias.

# %%
cadenas = [
    ("Scopus", 'TITLE-ABS-KEY("zero-inflated negative binomial" AND (earthquake OR seismicity))', "23-oct-2026", "5", "5", "0",
     "ninguno trata sismicidad; el único con sismos (Newton et al., 2015) predice víctimas"),
    ("Scopus", 'TITLE-ABS-KEY("forecast combination" AND (earthquake OR seismicity) AND (count* OR rate))', "23-oct-2026", "0", "0", "0",
     "sin documentos"),
    ("Web of Science", 'TS=(("gradient boosting" OR XGBoost) AND "earthquake forecast*" AND (count OR rate))', "23-oct-2026", "136", "50 exportados", "4 de 13 afines",
     "13 tratan sismicidad; 4 pronostican sismos con ML; ninguno usa ZINB ni combinación"),
    ("Web of Science", 'TS=(("meta-learning" OR "stacking") AND (earthquake OR seismic*) AND forecast*)', "23-oct-2026", "80", "50 exportados", "4",
     "casi todo es exploración sísmica (inversión, interpolación); 4 de sismología"),
    ("Google Scholar", 'earthquake count forecasting machine learning Colombia "Servicio Geológico Colombiano"', "", "", "", "", "por ejecutar"),
    ("Google Scholar", '"combinación de pronósticos" sismos Colombia', "", "", "", "", "por ejecutar"),
    ("arXiv", 'abs:(earthquake AND forecast AND (boosting OR "zero-inflated" OR ensemble))', "", "", "", "", "por ejecutar"),
    ("IEEE Xplore", '("earthquake forecasting") AND ("ensemble" OR "combination") AND ("count")', "23-oct-2026", "2", "2", "0",
     "ninguno: uno es de precursores electromagnéticos satelitales (2004) y otro de puntos de control con ICESat-2; las palabras coinciden sin relación con el tema"),
]
tabla(pd.DataFrame(cadenas, columns=["base de datos", "cadena de búsqueda", "fecha", "resultados", "revisados", "relevantes", "observación"]))

# %% [markdown]
# **Qué aportaron las consultas directas.**
#
# - **Ceros.** La única cadena con ZINB y sismos en Scopus devolvió cinco documentos de otros campos (lluvia, agricultura,
#   deslizamientos, mezclas de binomiales negativas) y ninguno de sismicidad. Es coherente con el vacío que reportaron los
#   dos motores asistidos.
# - **Combinación de pronósticos.** La cadena con *forecast combination* no devolvió documentos en Scopus.
# - **Pronóstico de sismos con aprendizaje automático (Web of Science).** Los cuatro trabajos más afines son Mukherjee et
#   al. (2025, cinturón del Himalaya), Kaushal et al. (2025, RNN-LSTM), Hu et al. (2025, redes multimodales; ya aparecía en
#   Consensus) y **Cuvier et al. (2026, pronóstico de sismicidad inducida por fluidos con series de tiempo)**, este último
#   nuevo y por leer, porque podría pronosticar tasas o conteos. Hay además dos revisiones críticas (Mignan y Broccardo,
#   2020; Mousavi y Beroza, 2023), un híbrido de meta-aprendizaje para magnitud (Maya et al., 2022), un modelo de
#   pronóstico de corto plazo con autoaprendizaje (Jiang et al., 2024) y una inferencia sin verosimilitud para ETAS
#   incompleto (Bountzis et al., 2026).
# - **Nota de Mignan y Broccardo (2020).** Su meta-análisis advierte que muchos estudios de redes neuronales para sismos
#   tienen límites estadísticos de evaluación, lo que respalda el énfasis de este proyecto en origen móvil, pruebas formales
#   y comparación contra el campeón de cada familia.
# - **Qué no se vio.** Con solo títulos no se puede juzgar el objetivo de cada trabajo; ninguno anuncia en el título conteos
#   por celda, distribución de ceros ni combinación de pronósticos, pero eso debe confirmarse en los resúmenes.

# %% [markdown]
# ## 4.2 Qué encontró la búsqueda
#
# **El panorama, en una frase:** las reglas de evaluación y los ensambles sísmicos están maduros; el pronóstico regional de
# conteos con aprendizaje automático, con distribución completa y con tratamiento explícito del exceso de ceros, casi no
# existe, y en Colombia lo publicado es sobre amenaza, recurrencia y monitoreo más que sobre pronóstico.
#
# - **Aprendizaje automático para tasas y conteos sísmicos.** Hay estudios en California, Japón, China, Asia Central y
#   Colombia (Zlydenko et al., 2023; Dascher-Cousineau et al., 2023; Zhang et al., 2023; Hou, 2026; García-Arias et al.,
#   2026), pero la mayoría pronostica magnitud, réplicas fuertes o clases, no un conteo por celda con distribución. LeapSpace
#   concluye que ningún estudio posterior a 2010 recuperado expone la combinación completa de variables, grillas, ventanas
#   y métricas fuera de muestra para conteos regionales.
# - **Ensambles.** Están bien establecidos en sismología: pesos bayesianos, híbridos aditivos y multiplicativos, ganancias
#   diferenciales de probabilidad y combinaciones por regresión logística (Marzocchi et al., 2012; Shebalin et al., 2014;
#   Herrmann y Marzocchi, 2023; Mizrahi et al., 2022). En el experimento CSEP de Italia a un día, el ensamble igualó al
#   mejor modelo individual (Taroni et al., 2018). Son ensambles de modelos de tasas (ETAS y similares), no de modelos de
#   aprendizaje automático.
# - **Poisson frente a sobredispersión.** Varios trabajos muestran que la variabilidad de los conteos viola el supuesto de
#   Poisson y que conviene pronosticar la distribución completa (Lombardi y Marzocchi, 2010; Nandan et al., 2019; Brehmer
#   et al., 2024; Igilik, 2026).
# - **Exceso de ceros.** Solo se encontraron aplicaciones vecinas: un modelo de dos partes para series con muchos ceros
#   (Wang et al., 2018) y un ZINAR para frecuencia de terremotos mayores en Sumatra (Wahyuliani et al., 2026). LeapSpace no
#   verifica ningún estudio de ZIP, ZINB o *hurdle* aplicado directamente a catálogos de sismicidad.
# - **Colombia.** Lo más cercano es García-Arias et al. (2026) en el nido de Bucaramanga con datos del SGC, que usa una
#   alerta binaria. El resto es amenaza, recurrencia o monitoreo (Mariniere et al., 2021; Arteta et al., 2023; Acevedo et
#   al., 2020).
#
# **Fuerza de la evidencia** (según Consensus; la ausencia de un estudio no se puede demostrar):

# %%
tabla(pd.DataFrame([
    ("El aprendizaje automático se ha usado para pronosticar tasas o conteos regionales", "fuerte", "varios estudios en distintas regiones"),
    ("El pronóstico por ensamble está establecido en sismología", "fuerte", "CSEP, ensambles bayesianos, híbridos y sistemas operativos"),
    ("Poisson suele ser inadecuado para la variabilidad de los conteos", "fuerte", "varios trabajos muestran sobredispersión o favorecen la distribución completa"),
    ("El pronóstico sísmico basado en ZINB está poco desarrollado", "débil", "hay modelos vecinos con ceros, pero casi ningún precedente directo en pronóstico"),
    ("Un ensamble de aprendizaje automático con distribución de conteos para Colombia sería novedoso", "moderada", "literatura colombiana de pronóstico escasa; la ausencia no se puede probar"),
], columns=["afirmación", "fuerza de la evidencia", "razón"]))

# %% [markdown]
# ## 4.3 Los trabajos más cercanos y en qué se diferencian de A+B
#
# Se comparan seis trabajos: los cuatro más cercanos por objetivo y método, el estudio colombiano y el ensamble de
# referencia. Los datos de la tabla salen de los informes de búsqueda; la lectura del texto completo de cada uno queda como
# verificación pendiente (sección 4.7).

# %%
tabla(pd.DataFrame([
    ("Zhang et al. (2023)", "California del Sur", "pronóstico de conteos, magnitudes y espacio", "ETAS híbrido con ConvLSTM",
     "pruebas N, M, S y PL; supera a ETAS y a ConvLSTM", "malla gruesa de 2°; no modela exceso de ceros"),
    ("Zlydenko et al. (2023)", "catálogo JMA (Japón)", "tasas de sismos", "codificador neuronal de proceso puntual",
     "predicción de tasa en prueba; mejor ganancia de información y más rápido", "referencia principal basada en ETAS; no ensambla modelos"),
    ("Igilik (2026)", "no detallada en los informes", "pronóstico semanal por celda", "regresión binomial negativa neuronal, dispersión por celda",
     "no detallada en los informes", "binomial negativa, sin componente de ceros ni combinación de modelos (por verificar en el texto)"),
    ("Hou (2026)", "regiones de Indonesia", "frecuencia (conteo) de sismos, varios pasos", "híbrido ARIMA y aprendizaje automático",
     "RMSE y MAE", "sin evaluación probabilística tipo CSEP; sin celdas"),
    ("García-Arias et al. (2026)", "nido de Bucaramanga, catálogo SGC 1994–2024", "alerta binaria M ≥ 4,5 a seis días", "red neuronal sobre conteos diarios rezagados",
     "ROC-AUC y TPR; mejora modesta (ROC-AUC apenas sobre 0,60)", "objetivo binario, no conteos ni distribución"),
    ("Marzocchi et al. (2012)", "California (RELM)", "ensamble de pronósticos de tasas", "pesos bayesianos con factores de Bayes",
     "evaluación bayesiana de pronósticos", "no usa aprendizaje automático ni modela ceros"),
], columns=["trabajo", "datos", "objetivo", "método central", "evaluación", "diferencia con A+B"]))

# %% [markdown]
# Los trabajos añadidos por Web of Science (Cuvier et al., 2026, entre otros) no están en esta matriz porque solo se tienen sus
# títulos; entran cuando se lean sus resúmenes.
#
# **Lectura.** Ningún trabajo recuperado reúne a la vez catálogo regional colombiano, conteo mensual por celda con
# verosimilitud de ZINB, combinación de pronósticos y una evaluación con origen móvil y pruebas estadísticas formales
# (García-Arias et al., 2026; Wahyuliani et al., 2026; Marzocchi et al., 2012). El trabajo colombiano usa clasificación y los
# híbridos ETAS con aprendizaje profundo no modelan ceros de forma explícita. **El vecino distribucional más cercano es
# Igilik (2026)**, que también usa binomial negativa por celda: hay que leerlo completo antes de afirmar diferencias finas.
#
# ## 4.4 Matriz de novedad
#
# Cada columna es un ingrediente de A+B. «Sí» indica que el trabajo lo tiene según los informes; «—», que los informes no
# lo reportan o lo descartan; «parcial», que lo tiene de forma aproximada.

# %%
tabla(pd.DataFrame([
    ("Zhang et al. (2023)", "parcial", "sí", "—", "—", "híbrido", "—", "pruebas CSEP"),
    ("Zlydenko et al. (2023)", "sí", "sí", "—", "—", "—", "—", "información ganada"),
    ("Igilik (2026)", "sí", "sí", "sí", "—", "—", "—", "por verificar"),
    ("Hou (2026)", "sí", "sí", "—", "—", "híbrido", "—", "RMSE, MAE"),
    ("García-Arias et al. (2026)", "sí", "alerta binaria", "—", "—", "—", "sí", "ROC-AUC"),
    ("Marzocchi et al. (2012)", "—", "tasas", "—", "—", "sí", "—", "bayesiana"),
    ("**A+B v2 (este trabajo)**", "sí", "sí", "sí", "sí", "sí", "sí", "origen móvil, DM, GW, CW, SPA, MCS"),
], columns=["trabajo", "aprendizaje automático", "conteo o tasa regional", "distribución de conteos", "exceso de ceros (ZINB)",
            "combinación de pronósticos", "Colombia", "evaluación"]))

# %% [markdown]
# **Cobertura por tema** (Consensus, número de artículos por intersección):

# %%
tabla(pd.DataFrame([
    ("Tasas con aprendizaje automático", "13", "3", "5", "ninguno"),
    ("Ensambles de pronósticos", "12", "4", "10", "2"),
    ("Conteos con exceso de ceros", "5", "1", "ninguno", "ninguno"),
    ("Pronósticos para Colombia", "3", "1", "ninguno", "ninguno"),
    ("Referencias ETAS", "13", "4", "14", "1"),
], columns=["tema", "catálogos regionales", "distribución completa", "pruebas prospectivas", "uso operativo"]))

# %% [markdown]
# La intersección de **ceros + Colombia + pruebas prospectivas** está vacía en el corpus. LeapSpace llega a la misma
# conclusión: la evidencia cubre mucho mejor la parte de evaluación y ensambles que la de aprendizaje automático con
# incertidumbre calibrada y tratamiento de ceros, y no hay un estudio colombiano de pronóstico prospectivo verificado.
#
# ## 4.5 Métodos afines por familia algorítmica
#
# | Método afín | Familia | Qué conserva A+B | Qué es nuevo en A+B | Limitación que A+B aborda |
# |---|---|---|---|---|
# | Stacking (Wolpert, 1992) | combinación de pronósticos | combina predicciones fuera de pliegue | pesos que dependen del contexto de cada celda-mes | pesos fijos para todas las celdas |
# | Combinación de pronósticos (Timmermann, 2006) | combinación | promedio ponderado de pronósticos | pesos por contexto sísmico y calibración final | pesos constantes en el tiempo y el espacio |
# | FFORMA (Montero-Manso et al., 2020) | meta-aprendizaje | pesos por meta-características | pérdida log1p para conteos, miembros ML y ZINB, OOF cronológico, calibración | diseñado para series univariadas, no paneles de conteos |
# | Mezcla de expertos (Jacobs et al., 1991) | combinación con compuerta | compuerta softmax | expertos heterogéneos ya entrenados; compuerta con contexto sísmico | expertos y compuerta se entrenan juntos, costoso |
# | Gradient boosting (Friedman, 2001) | boosting | aditivo en árboles, pasos de Newton | pérdida ZINB con dos predictores (media y ceros) | pérdidas estándar no modelan exceso de ceros |
# | GAMLSS por boosting (Mayr et al., 2012) | boosting distribucional | varios parámetros por boosting | ZINB con dispersión re-estimada y parada cronológica | — |
# | ETAS (Ogata, 1988) y CL-ETAS (Zhang et al., 2023) | modelo puntual sísmico | excitación por sismos previos | excitación como variable dentro de un modelo de ML | supuestos paramétricos rígidos |
# | Ensamble bayesiano (Marzocchi et al., 2012) | ensamble sísmico | pesos por desempeño | miembros de aprendizaje automático y pesos que dependen del contexto | pesos globales por modelo |
# | Regresión ZIP y ZINB (Lambert, 1992) | conteos | verosimilitud ZINB | estimada por boosting no paramétrico | relación lineal en el predictor |
#
# **Un hallazgo de este proyecto que la literatura ayuda a leer.** El capítulo 5 muestra que A+B no se distingue del
# stacking lineal con los mismos miembros. Eso es coherente con la literatura de combinación de pronósticos (Timmermann,
# 2006), donde una combinación sencilla suele ser muy competitiva. Por eso la novedad que se reclama no está en la regla de
# combinación, sino en el conjunto.
#
# ## 4.6 Vacíos de investigación y cómo los aborda este trabajo
#
# | Vacío (según los informes) | Qué hace este trabajo |
# |---|---|
# | Ningún estudio de ZIP, ZINB o *hurdle* aplicado a catálogos sísmicos está verificado | boosting de la verosimilitud ZINB con CART propio, sobre el catálogo del SGC |
# | Sin evaluación colombiana de pronóstico de conteos | origen móvil 2021–2026, 69 meses, con reentrenamiento anual |
# | Pocos estudios con salida de conteos por celda y evaluación formal | DM con Newey-West, GW, CW, SPA, MCS y Holm frente al campeón de cada familia |
# | Calibración y nitidez poco reportadas en trabajos de aprendizaje automático | calibración final de Mincer–Zarnowitz y prueba de ida y vuelta |
# | Sensibilidad a completitud, declustering y tamaño de celda | **no abordado** (trabajo futuro) |
# | Pruebas CSEP (N, S, M, L) y evaluación con reglas de puntuación propias | **no abordado**: se usa RMSLE y pruebas de comparación de pérdidas |
# | Comparación Poisson, NB, ZIP, ZINB y *hurdle* bajo un mismo objetivo | solo parcialmente: el capítulo 5 compara A+B con los campeones, no con cada distribución |
#
# ## 4.7 Estado de verificación de las referencias citadas en el proyecto
#
# Los informes verifican las referencias que siguen. Las demás fueron aportadas por el equipo con su DOI o enlace oficial;
# los identificadores no se resolvieron desde el entorno de redacción y deben comprobarse al abrir cada enlace. Marzocchi et al. (2012) sí queda verificada: aparece con título, autores y DOI en ambos
# informes (el segundo la marca como «no verificable» solo porque el texto de la consulta no traía metadatos).

# %%
tabla(pd.DataFrame([
    ("Timmermann (2006)", "verificada (informes)", "10.1016/s1574-0706(05)01004-9"),
    ("Gneiting, Balabdaoui y Raftery (2007)", "verificada (informes)", "10.1111/j.1467-9868.2007.00587.x"),
    ("Wolpert (1992)", "verificada (informes)", "10.1016/s0893-6080(05)80023-1"),
    ("Mayr et al. (2012)", "verificada (informes; GAMLSS por boosting)", "10.1111/j.1467-9876.2011.01033.x"),
    ("Ogata (1988)", "verificada (informes)", "10.1080/01621459.1988.10478560"),
    ("Marzocchi, Zechar y Jordan (2012)", "verificada (informes)", "10.1785/0120110327"),
    ("Mincer y Zarnowitz (1969)", "capítulo NBER (enlace oficial, no es artículo)", "https://www.nber.org/system/files/chapters/c1214/c1214.pdf"),
    ("Breiman (1996), stacked regressions", "aportado por el equipo", "10.1023/A:1018046112532"),
    ("Breiman, Friedman, Olshen y Stone (1984), CART", "libro (editorial Routledge; Wadsworth, 1984)", "—"),
    ("Friedman (2001)", "aportado por el equipo", "10.1214/aos/1013203451"),
    ("Montero-Manso et al. (2020), FFORMA", "aportado por el equipo", "10.1016/j.ijforecast.2019.02.011"),
    ("Jacobs et al. (1991)", "aportado por el equipo", "10.1162/neco.1991.3.1.79"),
    ("Krogh y Vedelsby (1995)", "aportado por el equipo (NIPS 7, publicado en 1995)", "enlace NIPS"),
    ("Marsaglia (2003)", "aportado por el equipo", "10.18637/jss.v008.i14"),
    ("Ke et al. (2017), LightGBM", "aportado por el equipo (NeurIPS 30)", "enlace NeurIPS"),
    ("Prokhorenkova et al. (2018), CatBoost", "aportado por el equipo (NeurIPS 31)", "enlace NeurIPS"),
    ("Lambert (1992)", "aportado por el equipo", "10.1080/00401706.1992.10485228"),
], columns=["referencia", "estado", "DOI"]))

# %% [markdown]
# ## 4.8 Declaración de novedad
#
# > Según la búsqueda descrita (Consensus: 26 búsquedas y un uso del grafo de citas, con 100 artículos incluidos; Elsevier
# > LeapSpace: una consulta estructurada con 73 referencias; y cinco cadenas propias en Scopus, Web of Science e IEEE Xplore, todo el 23 de
# > octubre de 2026), no se encontró un estudio que
# > reúna en un mismo marco: (i) el catálogo regional colombiano del SGC, (ii) el pronóstico mensual del número de sismos por
# > celda con una verosimilitud binomial negativa inflada en ceros optimizada por boosting con árboles propios, (iii) la
# > combinación contextual de pronósticos con calibración final y (iv) una evaluación con origen móvil y pruebas formales de
# > comparación frente al campeón de cada familia. Cada ingrediente existe por separado en la literatura; la contribución es
# > su integración y su evaluación para este problema.
#
# **Lo que no se afirma.** No se afirma ser el primer trabajo de aprendizaje automático para pronosticar sismos, el primer
# ensamble sísmico ni el primer pronóstico colombiano; la propia búsqueda encontró los tres. Tampoco se afirma novedad de la
# regla de combinación (empata con el stacking lineal, capítulo 5) ni de los modelos de conteo con ceros en general.
#
# **Límites de la revisión.**
#
# - Consensus y LeapSpace son motores asistidos por IA y no listan cada cadena con su número de resultados. Se cumple el mínimo de la guía con cinco cadenas en tres bases
#   (Scopus, Web of Science e IEEE Xplore). Google Scholar y arXiv quedan como complementarias; el total de resultados de
#   Web of Science (136 y 80) sí se anotaron.
# - Las exportaciones de Web of Science traen solo título y metadatos, sin resumen: la relevancia se juzgó por el título.
# - La ausencia de un estudio no se puede demostrar; Consensus califica de *moderada* la evidencia de que el enfoque
#   colombiano sea novedoso, y de *débil* la de que el pronóstico con ZINB esté poco desarrollado.
# - Los dos informes no tienen el texto completo de los trabajos cercanos, y Igilik (2026) y Hou (2026) son preimpresiones
#   o publicaciones de 2026 que deben leerse completas.
# - Este proyecto evalúa con origen móvil (pseudoprospectivo) y con RMSLE, no con las pruebas CSEP N, S, M y L.
