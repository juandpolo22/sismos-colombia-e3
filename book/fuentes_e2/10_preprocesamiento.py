# %% [markdown]
# # 10. Preprocesamiento y Pipeline (2.9)
#
# Todo el preprocesamiento vive en un `Pipeline` de scikit-learn (`src/modelos.py`) que se ajusta **solo
# con el entrenamiento**, o con la parte de ajuste de cada pliegue. Este capítulo documenta cada decisión
# y el hallazgo del EDA que la motiva, y elige con validación cruzada temporal las dos decisiones que no
# se pueden fijar a priori:
#
# 1. cómo imputar las magnitudes faltantes;
# 2. qué conjunto de variables usar.

# %% [cabecera]

# %%
# Datos y módulo de modelos
from sklearn import set_config
from sklearn.utils import estimator_html_repr
from IPython.display import HTML
from src import modelos as mdl
ev_, X, info, tr, gp, te = datos.panel_completo()
y = info["y"]; tm = info["tm"]
pl = datos.pliegues_temporales(tm[tr])

# %% [markdown]
# ## 10.1 Decisiones y hallazgo que las motiva

# %%
# Decisiones de preprocesamiento y el hallazgo del EDA que justifica cada una
decisiones = pd.DataFrame([
    ("Objetivo en log(1 + y) dentro de TransformedTargetRegressor", "Cap. 3: 76 % de ceros, asimetría 13,2, cola de ley de potencia; métrica RMSLE"),
    ("log1p + estandarización de conteos (sin recortar)", "Cap. 4 y 6: asimetría > 10 y atípicos que son las celdas más activas (información, no error)"),
    ("Imputación de magnitudes faltantes (constante 2,5 o mediana; elegida por CV)", "Cap. 2: faltante estructural «no hubo sismos» (Little rechaza MCAR)"),
    ("Mediana de entrenamiento para prof_media_hist", "Cap. 2: 1,4 % faltante en celdas aún sin sismos"),
    ("Sin codificación categórica", "Cap. 4: depto se descarta (lo sustituye la celda); no quedan categóricas en el panel"),
    ("Componentes cíclicos sen/cos del mes objetivo", "Cap. 7: sin estacionalidad, pero se incluyen por completitud (su exclusión se evalúa por CV)"),
    ("Rezagos y ventanas 1, 3, 6, 12 meses solo con el pasado", "Cap. 7: ACF y correlación cruzada positivas hasta 12 meses"),
    ("Rezago espacial (8 vecinas) en el mes t", "Cap. 8: I de Moran > 0; Knox: agrupamiento espacio-temporal"),
    ("lat/lon del centro de la celda", "Cap. 5 y 8: relación no monótona con la ubicación; heterogeneidad regional"),
    ("Estandarización de todas las entradas", "Modelos lineales con regularización (SVR lineal) y KNN son sensibles a la escala"),
    ("Sin selección agresiva por VIF", "Cap. 5: VIF máximo 8,4 (< 10); la regularización controla la colinealidad"),
], columns=["decisión", "hallazgo del EDA que la motiva"])
eda.guardar_tabla(decisiones, "decisiones_preprocesamiento.csv")
decisiones

# %% [markdown]
# ## 10.2 El Pipeline

# %%
# Diagrama del Pipeline completo (preprocesamiento + SVR lineal)
modelo = mdl.svr_lineal(X.columns)
set_config(display="diagram")
HTML(estimator_html_repr(modelo))

# %% [markdown]
# ## 10.3 Elección por validación cruzada temporal (solo entrenamiento)
#
# Se usa la **validación cruzada temporal de ventana creciente con gap** del capítulo 7: 5 pliegues sobre
# el último 60 % de los meses de entrenamiento. El modelo es el SVR lineal base con C = 0,1; su
# hiperparámetro se ajusta en el capítulo 11. Se compara el RMSLE medio de los pliegues.

# %%
# Selección de la imputación y del conjunto de variables por CV temporal (RMSLE)
def cv_rmsle(cols, imput="constante", C=0.1):
    r = []
    for a, v in pl:
        m_ = mdl.svr_lineal(cols, C=C, imputacion_magnitud=imput).fit(X.iloc[tr[a]][cols], y[tr[a]])
        r.append(ev.rmsle(y[tr[v]], m_.predict(X.iloc[tr[v]][cols])))
    return np.mean(r), np.std(r), r

conjuntos = {
    "todas (18)": list(X.columns),
    "sin calendario (16)": [c for c in X.columns if c not in mdl.CICLICAS],
    "sin lat/lon (16)": [c for c in X.columns if c not in mdl.UBICACION],
    "sin vecindad (16)": [c for c in X.columns if not c.startswith("vec")],
    "sin magnitudes (14)": [c for c in X.columns if c not in mdl.MAGNITUDES],
    "solo conteos propios (7)": ["n_1m", "n_3m", "n_6m", "n_12m", "n35_3m", "n35_12m", "n_mismo_mes_anio_prev"],
}
filas = []
for imput in ("constante", "mediana"):
    mu_, sd_, _ = cv_rmsle(list(X.columns), imput)
    filas.append({"prueba": f"imputación de magnitud: {imput}", "RMSLE_CV_media": mu_, "RMSLE_CV_desv": sd_})
sel_imp = pd.DataFrame(filas)
imput_ok = "constante" if sel_imp.RMSLE_CV_media[0] <= sel_imp.RMSLE_CV_media[1] else "mediana"
filas = []
for nombre, cols in conjuntos.items():
    mu_, sd_, r = cv_rmsle(cols, imput_ok)
    filas.append({"conjunto": nombre, "n_variables": len(cols), "RMSLE_CV_media": mu_, "RMSLE_CV_desv": sd_,
                  **{f"pliegue_{i + 1}": v for i, v in enumerate(r)}})
sel = pd.DataFrame(filas).sort_values("RMSLE_CV_media")
eda.guardar_tabla(sel_imp, "seleccion_imputacion.csv"); eda.guardar_tabla(sel, "seleccion_conjuntos.csv")
display(sel_imp.round(4))
sel.round(4)

# %%
# Regla de un error estándar para elegir el conjunto más simple
mejor = sel.iloc[0]
umbral_1se = mejor.RMSLE_CV_media + mejor.RMSLE_CV_desv / np.sqrt(len(pl))
candidatos = sel[sel.RMSLE_CV_media <= umbral_1se]
md(f"""**Interpretación.**
- **Imputación.** La **{imput_ok}** da el menor RMSLE medio ({es(sel_imp.RMSLE_CV_media.min(), 4)} frente a {es(sel_imp.RMSLE_CV_media.max(), 4)}).
  La diferencia es {('mínima' if abs(sel_imp.RMSLE_CV_media.diff().iloc[-1]) < 0.002 else 'apreciable')}.
- **Conjunto de variables.** El menor RMSLE de validación lo da **«{mejor.conjunto}»** ({es(mejor.RMSLE_CV_media, 4)} ± {es(mejor.RMSLE_CV_desv, 4)}).
- **Regla de un error estándar.** Los conjuntos a menos de un error estándar del mejor (umbral {es(umbral_1se, 4)}) son:
  {', '.join('«' + c + '»' for c in candidatos.conjunto)}.
- **Decisión.** Se adopta **{'«todas (18)»' if 'todas (18)' in set(candidatos.conjunto) else '«' + mejor.conjunto + '»'}**.
  {'Las 18 variables están dentro del margen de un error estándar: se conservan por coherencia con el EDA, que justifica cada una, y para que los modelos no lineales del Entregable 2 partan del mismo conjunto.' if 'todas (18)' in set(candidatos.conjunto) else 'El conjunto completo queda fuera del margen de un error estándar, así que se usa el mejor.'}
- **Qué variables importan.** Quitar las **magnitudes** (o quedarse solo con los conteos propios) empeora el RMSLE en
  {es(sel.set_index('conjunto').loc['sin magnitudes (14)', 'RMSLE_CV_media'] - mejor.RMSLE_CV_media, 3)}, y esa diferencia se repite en los 5 pliegues. Quitar calendario,
  ubicación o vecindad no cambia nada para un modelo **lineal**. En el caso de la ubicación y la vecindad es esperable: su relación con el
  objetivo es no lineal (capítulos 5 y 8).
- **Variabilidad entre pliegues.** La desviación entre pliegues ({es(sel.RMSLE_CV_desv.mean(), 3)} en promedio) es comparable a la mayor diferencia entre
  conjuntos: el error depende mucho del **periodo evaluado** (el pliegue 2 es el más difícil en todos los conjuntos). Es una señal de los
  cambios de régimen del capítulo 7.""")

# %% [markdown]
# ## 10.4 Comprobación: el Pipeline se ajusta solo con entrenamiento

# %%
# Ajuste final del Pipeline y verificación de que el escalador usa solo el entrenamiento
final_cols = list(X.columns) if "todas (18)" in set(candidatos.conjunto) else conjuntos[mejor.conjunto]
m_ = mdl.svr_lineal(final_cols, imputacion_magnitud=imput_ok).fit(X.iloc[tr][final_cols], y[tr])
prep = m_.regressor_.named_steps["prep"]
esc = prep.named_transformers_["conteos"].named_steps["escala"]
media_tr = np.log1p(X.iloc[tr][[c for c in mdl.CONTEOS if c in final_cols]]).mean().to_numpy()
media_todo = np.log1p(X[[c for c in mdl.CONTEOS if c in final_cols]]).mean().to_numpy()
md(f"""- Medias del escalador de conteos = medias de log1p en **entrenamiento**: **{np.allclose(esc.mean_, media_tr)}**. Con todo el panel no
  coinciden (diferencia máxima {es(np.abs(media_todo - esc.mean_).max(), 3)}).
- Columnas de salida del preprocesador: {len(prep.get_feature_names_out())} (`{', '.join(prep.get_feature_names_out()[:6])}, …`).
- Decisiones guardadas en `tables/parametros_preprocesamiento.csv` para el capítulo 11.""")
_ = eda.guardar_tabla(pd.DataFrame({"parametro": ["imputacion_magnitud", "variables"], "valor": [imput_ok, ";".join(final_cols)]}),
                      "parametros_preprocesamiento.csv")
