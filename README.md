# Pronóstico mensual de sismos por celda en Colombia

Proyecto final de Machine Learning (Universidad del Norte) · Juan David Polo y Gabriella Pallares · Entregable 3.

Pronostica el **número de sismos M ≥ 2,5 del mes siguiente en cada celda de 0,5° × 0,5°** de Colombia con el catálogo
del Servicio Geológico Colombiano (2004–2026). Incluye 7 familias de modelos benchmark optimizados con 4 métodos, un
**modelo original (A+B v2)** —combinador contextual de pronosticadores con un boosting propio de la verosimilitud
binomial negativa inflada en ceros sobre árboles CART programados desde cero— validado estadísticamente, un dashboard en
Dash Plotly y un informe en Jupyter Book.

- **Dashboard:** `https://<servicio>.onrender.com` (ver Despliegue)
- **Informe:** `book/` (versión compilada en `book/_build/html/index.html`)

## Estructura

```text
app.py                  punto de entrada de Dash (expone server para gunicorn)
pages/                  contexto.py, eda.py, modelos.py (una página por sección)
src/                    original_model.py (clase del modelo original), data.py, plots.py, artefactos.py,
                        explicacion.py, arbol_cart.py, propuesto.py y los módulos de datos y experimentos
artifacts/              modelos serializados (.joblib), metadata.json y predicciones de ida y vuelta
data/                   catálogo limpio y data/app/ (datos livianos del dashboard)
assets/                 estilo.css
tests/                  pruebas unitarias (pytest)
book/                   fuentes y notebooks del Jupyter Book
scripts/                experimentos, entrenamiento de artefactos y preparación de datos
resultados/, tables/    resultados precalculados de los experimentos
requirements.txt        dependencias del dashboard (versiones exactas)
requirements-dev.txt    dependencias de experimentos, pruebas y libro
render.yaml             despliegue en Render
```

## Instalación y uso local

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                            # http://127.0.0.1:8050
```

Semilla global: `SEMILLA = 42` en `src/config.py` (se propaga a numpy, scikit-learn, XGBoost, Optuna, DEAP, el árbol
CART propio y el modelo original). Versión de Python: 3.9.25.

## Reproducir resultados, pruebas y libro

```bash
pip install -r requirements-dev.txt
python scripts/entrenar_artefactos.py        # reentrena y serializa los 8 modelos del dashboard (verifica ida y vuelta)
python scripts/preparar_datos_app.py         # datos livianos del dashboard (pronóstico, intervalos, SHAP)
python -m pytest tests                       # pruebas unitarias (52)
python scripts/origen_movil_final.py --matrices && python scripts/reglas_combinacion.py   # A frente a reglas estándar
python scripts/estado_del_arte.py            # LightGBM, CatBoost y boosting de Poisson con el mismo protocolo
python scripts/complejidad_original.py       # complejidad empírica del modelo original
python scripts/optimizar_B_tpe.py            # TPE de los hiperparámetros de B (datos < 2016)
python scripts/comparar_B_validacion.py      # decisión sobre B con la validación previa (< 2021)
cd book && python herramientas/construir_notebooks.py && cd ..
jupyter-book build book/
```

Los comandos de los experimentos del Entregable 2 y del origen móvil están en el capítulo 9 del libro.

## Despliegue en Render

1. Subir este repositorio a GitHub (incluye `artifacts/` y `data/app/`, que pesan pocos MB).
2. En Render: **New → Blueprint** y elegir el repositorio (usa `render.yaml`), o **New → Web Service** con
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:server --workers 1 --threads 2 --timeout 120 --preload`
   - Variable de entorno: `PYTHON_VERSION = 3.9.25`
3. Revisar los registros de construcción y abrir la URL pública. El plan gratuito se suspende por inactividad: abrir la
   URL unos minutos antes de la sustentación.

## Datos

Catálogo sísmico de la Red Sismológica Nacional de Colombia, Servicio Geológico Colombiano
(<https://www.sgc.gov.co/sismos>). Uso académico, sin fines comerciales y con atribución.
