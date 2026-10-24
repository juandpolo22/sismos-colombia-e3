# %% [markdown]
# # 9. Reproducibilidad y bibliografía

# %% [cabecera]

# %% [markdown]
# ## 9.1 Entorno y semillas
#
# - Python 3.9.25; dependencias del dashboard con versiones exactas en `requirements.txt` y las de experimentos y libro en
#   `requirements-dev.txt`.
# - Semilla global única `SEMILLA = 42` (`src/config.py`), propagada a numpy, random, scikit-learn, XGBoost, Optuna, DEAP,
#   al árbol CART propio y al modelo original.
#
# ## 9.2 Cómo reproducir todo
#
# ```text
# pip install -r requirements-dev.txt
# python scripts/correr_experimentos.py --tarea reg --fase todo   # 28 combinaciones de regresión (Entregable 2)
# python scripts/correr_experimentos.py --tarea clf --fase todo   # clasificación (Entregable 2)
# python scripts/origen_movil.py                                  # hiperparámetros < 2016 y miembros
# python scripts/origen_movil_benchmark.py                        # 28 combinaciones en el origen móvil
# python scripts/miembros_fuera_de_pliegue.py                     # predicciones fuera de pliegue (< 2021)
# python scripts/origen_movil_final.py                            # modelo original en el origen móvil 2021–2026
# python scripts/independencia_residuos.py                        # pruebas de independencia
# python scripts/verificar_modelo_original.py                     # la clase reproduce el modelo evaluado
# python scripts/optimizar_B_tpe.py                              # TPE de los hiperparámetros de B (datos < 2016)
# python scripts/comparar_B_validacion.py                        # decisión sobre B con la validación previa (< 2021)
# python scripts/entrenar_artefactos.py                           # reentrenamiento final y serialización
# python scripts/preparar_datos_app.py                            # datos del dashboard
# python -m pytest tests                                          # pruebas unitarias
# cd book && python herramientas/construir_notebooks.py --e2 && python herramientas/construir_notebooks.py
# jupyter-book build book/
# ```
#
# ## 9.3 Tabla maestra de experimentos
#
# Las 140 combinaciones del Entregable 2 y el modelo original, con métricas, tiempos y semilla
# (`tables/tabla_maestra_e3.csv`).

# %%
M = pd.read_csv(cfg.RESULTADOS / "benchmark" / "experimentos_maestro.csv")
cols = ["tarea", "modelo", "balanceo", "optimizador", "metrica_principal", "externo_media", "externo_desv", "t_total_s", "semilla"]
M = M[cols].copy()
from src import data as D
import json
meta = json.load(open(cfg.RAIZ / "artifacts" / "A+B_v2.metadata.json", encoding="utf-8"))
ab = D.lideres().set_index("etiqueta").loc["A+B v2"]
M = pd.concat([M, pd.DataFrame([{"tarea": "reg", "modelo": "A+B v2 (original)", "balanceo": "—", "optimizador": "rejilla interna de λ",
                                  "metrica_principal": "RMSLE (origen móvil 2021–2026)", "externo_media": ab["RMSLE origen móvil 2021–2026"],
                                  "externo_desv": np.nan, "t_total_s": meta["tiempo_entrenamiento_s"], "semilla": cfg.SEMILLA}])], ignore_index=True)
M.to_csv(cfg.TABLAS / "tabla_maestra_e3.csv", index=False)
md(f"La tabla maestra tiene **{len(M)} filas** ({(M.tarea == 'reg').sum()} de regresión y {(M.tarea == 'clf').sum()} de clasificación).")
tabla(M.tail(8), dec=4)

# %% [markdown]
# ## 9.4 Bibliografía
#
# ```{bibliography}
# :all:
# ```
