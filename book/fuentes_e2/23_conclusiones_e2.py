# %% [markdown]
# # 23. Conclusiones del Entregable 2 y matriz de trazabilidad

# %% [cabecera]

# %%
# Resumen final a partir de las tablas guardadas por los capítulos del Entregable 2
from src import resultados_e2 as r2
M = r2.maestro()
TR = pd.read_csv(cfg.TABLAS / "e2_reg_prueba.csv")
CL = pd.read_csv(cfg.TABLAS / "e2_clf_mejores_por_modelo.csv")
DM = pd.read_csv(cfg.TABLAS / "e2_stats_dm.csv")
CMP = pd.read_csv(cfg.TABLAS / "e2_ab_vs_campeones.csv")
ABL = pd.read_csv(cfg.TABLAS / "e2_ab_ablacion.csv").set_index("variante")
MC = pd.read_csv(cfg.TABLAS / "e2_stats_mcs_origen_movil.csv")
SPA = pd.read_csv(cfg.TABLAS / "e2_stats_spa.csv")
GC = pd.read_csv(cfg.TABLAS / "e2_stats_gw_cw.csv")
FAM = pd.read_csv(cfg.TABLAS / "e2_stats_familia_holm.csv")
ab = TR[TR.modelo == "A+B v2 (propuesto)"].iloc[0]; b = TR[TR.modelo != "A+B v2 (propuesto)"].iloc[0]
emp_c = CMP.loc[CMP.veredicto == "empate", "campeón"].tolist()
xg = CMP[CMP.campeón.str.startswith("XGBoost")].iloc[0]
md(f"""## Resumen ejecutivo

- **Diseño experimental.** Se ejecutaron las **{int((M.estado == 'completado').sum())} combinaciones** del enunciado con validación anidada
  temporal completa (3 bloques externos de 12 meses × 3 pliegues internos, 15 evaluaciones por optimizador). Las reglas de reducción del
  cap. 13 (equivalencia y tamizaje) se mantuvieron como experimento y se verificaron con los resultados completos (cap. 14.7).
- **Evaluación final** (cap. 13.11). **Origen móvil 2021–2026**: cada año los modelos se reentrenan con todo el pasado y pronostican
  ese año (69 meses); hiperparámetros y campeones de cada familia fijados con datos anteriores a 2016. Se evaluaron las 28
  combinaciones de regresión y 26 de clasificación (7 modelos × 4 balanceos con TPE, sin los pares equivalentes; cap. 13.12).
- **Clasificación** (¿habrá al menos un sismo el mes siguiente?). El mejor campeón es **{CL.modelo.iloc[0]} ({CL.balanceo.iloc[0]})**,
  AUC {CL.AUC.iloc[0]:.3f}; marcando cada mes la proporción histórica de celdas activas alcanza recall {CL.Recall.iloc[0]:.2f}. El balanceo
  no mejora el AUC de los modelos fuertes y descalibra las probabilidades; Platt e isotónica, ajustadas con años anteriores, lo corrigen
  (cap. 17).
- **Regresión** (¿cuántos sismos?). XGBoost y Random Forest encabezan las 28 combinaciones; el campeón de XGBoost (TPE) obtiene RMSLE {xg['RMSLE campeón']:.4f} (cap. 18).
- **Modelo propuesto A+B v2** (boosting ZINB, combinador contextual y calibración final propios), con predicciones fuera de pliegue
  cronológicas: **RMSLE {ab.RMSLE:.4f}**, el menor de todos.
  - **Frente al campeón de cada familia** (Holm, 7 comparaciones): gana a **{int((CMP.veredicto == 'A+B v2 gana').sum())} de 7**
    {('(empates: ' + ', '.join(emp_c) + ')') if emp_c else f'(frente a XGBoost: p Holm = {xg["p Holm"]:.3f}, d de Cohen = {xg["d de Cohen"]:.2f})'}.
  - **Pruebas alternativas** frente a los mismos 7 campeones: Giacomini-White significativo en {int((GC['p GW Holm'] < 0.05).sum())} de 7 y
    Clark-West en {int((GC['p CW Holm'] < 0.05).sum())} de 7; con Holm sobre la familia completa de 21 contrastes (DM, GW y CW) el mayor
    p ajustado es {FAM['p Holm (21)'].max():.4f}.
  - **MCS al 90 %** (7 campeones, A+B v2 y líneas base): {', '.join(MC[MC.en_MCS].modelo)}.
  - **Ablación:** sin el componente propio B el RMSLE sube a {ABL.loc['A sin B', 'RMSLE (principal)']:.4f} (p DM = {ABL.loc['A sin B', 'p DM']:.3f}).
  - **Lectura:** superioridad clara sobre Random Forest, KNN, árbol, SVR, Ridge y Lasso; frente a XGBoost la ventaja es significativa
    pero pequeña (≈ {xg['mejora %']:.1f} % de RMSLE frente a su campeón) (cap. 21 y 22).
- **Optimización.** Bayesiano (TPE) es el más eficiente por unidad de presupuesto, aunque solo se distingue con significancia de Grid. En la
  métrica externa los cuatro optimizadores quedan prácticamente empatados. La multi-fidelidad ahorra cómputo y no mejora la métrica
  (cap. 14 y 15).
- **Cómputo.** Las versiones optimizadas (XGBoost hist con early stopping, LinearSVC/SGD, FAISS) reducen el tiempo hasta un orden de
  magnitud o más sin perder desempeño. El cuello de botella real es el ajuste de los modelos, no el preprocesamiento (cap. 16).""")

# %% [markdown]
# ## Limitaciones
#
# - **Enjambres.** Ningún modelo anticipa el inicio de un enjambre; con información solo hasta el cierre del
#   mes t eso parece un límite del problema, no de los modelos (cap. 18 y 19).
# - **Tamaño del efecto.** Las diferencias significativas entre los mejores modelos son pequeñas (d de Cohen y
#   δ de Cliff pequeños o despreciables); la significancia se debe a que son **sistemáticas**, no grandes.
# - **Costo.** A+B v2 cuesta decenas de veces más que XGBoost.
# - **Tiempos.** Los tiempos de la validación anidada se midieron con varias combinaciones en paralelo y tienen
#   ruido de contención; las comparaciones de tiempo finas se hicieron aparte (cap. 16).

# %% [markdown]
# ## Matriz de trazabilidad: requisito del enunciado → dónde se cumple
#
# | Sección del enunciado | Requisito | Dónde | Evidencia |
# |---|---|---|---|
# | 1 | Rigor estadístico, eficiencia, interpretabilidad, reproducibilidad (semillas, versionado, pipelines) | todo el libro; `src/config.py`, `src/modelos.py` | semilla 42 en todas las funciones; Pipeline de scikit-learn/imblearn en cada pliegue; versiones exactas en `requirements.txt` |
# | 2 | Pipeline combinatorio 7×4×4 + 7×4 = 140; tabla maestra; criterio para reducir | cap. 13, 14.7 | `experimentos_maestro.csv`: 140 combinaciones completadas, con la decisión previa de las reglas de reducción |
# | 3.1–3.2 | Validación anidada; métrica del bucle externo | cap. 13.2, 14.2 | `correr_combinacion` en `src/benchmark.py` |
# | 3.3 | Grid, Random, Bayesiano (sustituto y adquisición), Genético (selección, cruce, mutación, elitismo, población, diversidad) | cap. 14.1, 14.5 | tabla de configuración; curva de diversidad |
# | 3.3 | Multi-fidelidad (Successive Halving / Hyperband), métrica de fidelidad y factor de reducción | cap. 15; cap. 13.7 | η = 3, fidelidad = número de árboles |
# | 3.4 | Espacios de búsqueda justificados (escala log) | cap. 13.3 | `ESPACIOS` en `src/benchmark.py` |
# | 3.4 | Comparar métodos por métrica, tiempo total y por evaluación, presupuesto | cap. 14.2 | tabla de optimizadores |
# | 3.4 | Curvas anytime; evidencia cuantitativa de eficiencia | cap. 14.3–14.4 | área de regret + Wilcoxon con Holm |
# | 3.4 | Estabilidad ante semillas | cap. 14.6 | 5 semillas × 4 optimizadores |
# | 4.2 | Complejidad O(·) de entrenamiento e inferencia vs tiempo empírico | cap. 13.4, 16.1–16.2 | exponentes empíricos log–log |
# | 4.2 | KNN brute/KD/Ball/FAISS; SAGA; partial_fit; XGBoost hist vs exact + early stopping; SVM SGD/LinearSVC | cap. 16.2–16.3 | tabla estándar vs optimizado |
# | 4.3 | joblib (loky, threading); Dask/Ray (justificación); GPU con factor de aceleración | cap. 16.4 | tabla joblib; tabla CPU vs GPU |
# | 4.4 | Perfilamiento (cProfile, memory_profiler); tabla complejidad–tiempo–memoria–desempeño | cap. 16.3, 16.5 | perfil y memoria |
# | 5.1.1 | Matriz de confusión, ROC, Accuracy/Precision/Recall/F1/AUC; justificar métrica | cap. 17.1–17.3 | 26 combinaciones en el origen móvil 2021–2026 (cap. 13.12); las 112 en la validación anidada (cap. 14) |
# | 5.1.2 | SMOTE, ADASYN, class_weight vs sin balanceo | cap. 17.4 | mapas de calor por modelo y técnica |
# | 5.1.3 | Diagrama de confiabilidad, Brier, ECE; Platt e isotónica | cap. 17.5 | antes/después |
# | 5.2.1 | RMSE, MAE; gráfico entrenamiento/validación/prueba | cap. 18.1–18.2 | 28 combinaciones + A+B v2 en el origen móvil |
# | 5.2.2 | White, BDS, ACF + Ljung-Box, histograma y normalidad | cap. 18.1, 18.3 | |
# | 5.3 | SHAP global y local (dos observaciones) del mejor modelo de cada tarea; SHAP vs LIME en XGBoost | cap. 19 | summary, waterfall, comparación cuantitativa |
# | 5.4 | Media ± desviación del bucle externo; semillas; variabilidad vs diferencias entre modelos | cap. 20 | |
# | 6.1 | Friedman (ómnibus) | cap. 21.1 | |
# | 6.2 | Nemenyi, diagrama CD, DeLong (subconjunto), DM-HLN, Holm/BH (p crudo y ajustado), bootstrap BCa | cap. 21.2–21.3, 21.6; cap. 20.3 | |
# | 6.3 | MCS → Giacomini-White / Clark-West → DM-HLN con bootstrap estacionario; SPA / Reality Check | cap. 21.4–21.7 | |
# | 6.4 | δ de Cliff, d de Cohen; interpretar p-valores; IC | cap. 21.3, 21.6, 21.9 | |
# | Análisis adicional | Pruebas de independencia de residuos (Escanciano-Lobato, Breusch-Godfrey, BDS, Moran) | cap. 21.8 | `scripts/independencia_residuos.py` |
# | Análisis adicional | DM sobre diferencias pareadas por periodo, con dependencia temporal | cap. 21.6 | pérdidas mensuales pareadas; Newey-West (4 rezagos) + HLN |
# | Análisis adicional | Pruebas alternativas: Giacomini-White y SPA / Reality Check, con hipótesis, pertinencia y correcciones | cap. 21.5, 21.7 | GW y Clark-West frente a los 7 campeones; SPA y RC por campeón con Holm |
# | Análisis adicional | Unidad de análisis y dependencia espacial y temporal | cap. 21 (introducción), 21.8 | el mes como unidad; Moran, Escanciano-Lobato, Breusch-Godfrey |
# | Análisis adicional | Holm sobre las 7 comparaciones y la familia de pruebas | cap. 21.6 | Holm (7) por prueba y Holm (21) sobre DM + GW + CW |
# | Análisis adicional | Datos usados en cada etapa respecto al periodo 2021–2026 | cap. 22.2–22.3 | tabla de qué se fijó con qué datos; scripts reproducibles |
# | Análisis adicional | Trazabilidad: diferencias por periodo, p, tamaños del efecto, IC, implementación | cap. 21.6 | `tables/e2_stats_diferencias_mensuales.csv`, IC bootstrap, d de Cohen |
# | Análisis adicional | Ridge y Lasso con resultados casi idénticos | cap. 21.6 | α distintos, pronósticos distintos (no duplicados), ambos ≈ regresión lineal |
# | 7 | Jupyter Book reproducible con código, figuras, tablas e interpretación | todo el libro | `herramientas/construir_notebooks.py` |
# | 7.1 | EDA mejorado | cap. 2–9 | |
# | 7.2 | Modelos benchmark: KNN, Naive Bayes, logística L1/L2, Ridge, Lasso, árbol, RF, XGBoost (LIME y SHAP), SVM/SVR | cap. 13–21 | 140 combinaciones ejecutadas |
# | Rúbrica | Modelo propio original y su comparación estadística | cap. 22, 21 | A+B v2; comparación con el campeón de cada familia; ablación |
# | Diseño | Evaluación final en origen móvil, justificada con literatura | cap. 13.11–13.12 | Tashman (2000); Hyndman y Athanasopoulos (2021); Cerqueira et al. (2020); CSEP |

# %% [markdown]
# ## Cómo reproducir el Entregable 2
#
# ```text
# python scripts/correr_experimentos.py --tarea reg --fase todo      # 28 combinaciones de regresión
# python scripts/correr_experimentos.py --tarea clf --fase todo      # tamizaje + 84 de clasificación
# python scripts/correr_experimentos.py --tarea clf --fase 2 --todas # las 28 restantes (140 en total)
# python scripts/xgboost_gpu.py                                      # CPU vs GPU (requiere CUDA)
# python scripts/miembros_fuera_de_pliegue.py                       # miembros: predicciones fuera de pliegue (< 2021)
# python scripts/origen_movil.py                                     # origen móvil: hiperparámetros < 2016 y miembros
# python scripts/origen_movil_benchmark.py                           # origen móvil: 28 combinaciones
# python scripts/origen_movil_final.py                               # origen móvil: modelo propuesto final y su ablación
# python scripts/validacion_previa_final.py                          # validación previa del modelo final (< 2021)
# python scripts/origen_movil_clasificacion.py                       # origen móvil: 26 combinaciones de clasificación
# python scripts/semillas_origen_movil.py                            # sensibilidad a la semilla (cap. 20.2)
# python scripts/experimentos_e2.py                                  # experimentos de los cap. 14, 15, 16 y 20
# python scripts/independencia_residuos.py                           # pruebas de independencia (cap. 21.8)
# python herramientas/construir_notebooks.py                         # ejecuta y convierte todos los capítulos
# jupyter-book build .
# ```
#
# Todos los scripts guardan puntos de control en `resultados/` y se reanudan si se interrumpen.
