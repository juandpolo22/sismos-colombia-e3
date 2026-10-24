# %% [markdown]
# # 3. Modelos benchmark y mejores combinaciones
#
# Los 7 modelos benchmark de regresión (KNN, Ridge, Lasso, árbol de decisión, Random Forest, XGBoost y SVR) se
# optimizaron con 4 métodos (Grid, Random, Bayesiano TPE y Genético) dentro de una validación anidada temporal
# (secciones 3.1 a 3.11, del Entregable 2). Este capítulo documenta la selección de la mejor combinación de cada modelo,
# su reentrenamiento final y su serialización para el dashboard.

# %% [cabecera]

# %%
import json
from src import artefactos as A, data as D
L = D.lideres()
tabla(L[["modelo", "optimizador", "RMSLE bucle externo (media ± d.e.)", "RMSLE validación < 2016", "RMSLE origen móvil 2021–2026",
         "mejor optimizador por bucle externo", "inferencia (ms/fila)"]].round(4))

# %% [markdown]
# ## Criterio de selección
#
# - **Mejor combinación de cada modelo.** En cada familia se elige la combinación con menor RMSLE en la validación
#   interna con datos anteriores a 2016, la misma que fija sus hiperparámetros para el origen móvil. Así, ningún mes de
#   2021–2026 interviene en la elección, y los modelos del dashboard son exactamente los que se compararon
#   estadísticamente (capítulo 6). La columna «mejor optimizador por bucle externo» muestra la combinación que habría
#   ganado con la media del bucle externo de la validación anidada; las diferencias entre optimizadores de una misma
#   familia son menores que la variabilidad entre pliegues (sección 3.8).
# - **Mejor combinación global.** A+B v2 tiene el menor RMSLE en el origen móvil y es el único modelo en el Model
#   Confidence Set (capítulo 6). Entre los benchmarks, el campeón de XGBoost es el mejor y quedó fuera del MCS por poco.

# %% [markdown]
# ## Reentrenamiento final y serialización
#
# Cada pipeline completo (preprocesamiento + modelo, con el objetivo en escala log1p) se reentrena con **todos los meses
# con objetivo conocido** y se guarda con `joblib.dump(..., compress=3)` en `artifacts/`, junto con un `metadata.json`
# (versiones de librerías, semilla, hiperparámetros, esquema de columnas con tipos y rangos válidos y métricas). El
# dashboard solo carga estos archivos (`scripts/entrenar_artefactos.py`).
#
# **Prueba de ida y vuelta.** Antes de guardar, cada modelo predice los últimos 12 meses del entrenamiento; el archivo se
# carga de nuevo y debe producir exactamente las mismas predicciones. Las predicciones de referencia quedan en
# `artifacts/*.ida_vuelta.npy` y `tests/test_artefactos.py` repite la verificación. En otra máquina, el álgebra lineal
# puede cambiar el último dígito de los modelos lineales y de A+B (diferencias del orden de 1e-14), por eso la prueba
# unitaria compara con un error relativo máximo de 1e-9.

# %%
filas = []
for etq in D.modelos():
    m = A.metadata(etq)
    filas.append({"modelo": etq, "archivo": m["archivo"], "tamaño (MB)": m["tamano_MB"],
                  "filas de entrenamiento": m["entrenamiento"]["filas"], "último mes objetivo": m["entrenamiento"]["ultimo_mes_objetivo"],
                  "ida y vuelta idéntica": "sí" if m["prueba_ida_y_vuelta"]["identicas"] else "NO",
                  "carga (s)": m["tiempo_carga_s"], "inferencia (ms/fila)": m["tiempo_inferencia_ms_por_fila"]})
tabla(pd.DataFrame(filas))
