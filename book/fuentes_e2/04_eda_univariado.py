# %% [markdown]
# # 4. Análisis univariado (2.2)
#
# Se analiza cada variable en dos niveles, siempre con el periodo de **entrenamiento**:
#
# - **Catálogo** (una fila por sismo): magnitud, profundidad, coordenadas, departamento, escala de
#   magnitud y calidad de la localización.
# - **Panel** (celda × mes): las 18 predictoras del modelo base.

# %% [cabecera]

# %%
# Datos de entrenamiento: panel y catálogo limpio hasta el fin del entrenamiento
ev_, X, info, tr, gp, te = datos.panel_completo()
P = datos.panel_dataframe(X, info, tr)
fin_tr = datos.fecha_de_mes(info["m_test"] - cfg.GAP_MESES)[0]
crudo = datos.leer_crudos()
cat_tr, _ = datos.limpiar(crudo)
cat_tr = cat_tr[cat_tr.time < fin_tr]
md(f"Sismos del catálogo en el periodo de entrenamiento (antes de {fin_tr:%Y-%m-%d}): **{es(len(cat_tr), 0)}**; filas del panel de entrenamiento: **{es(len(P), 0)}**.")

# %% [markdown]
# ## 4.1 Variables numéricas del catálogo

# %%
# Resumen numérico (media, cuantiles, asimetría, curtosis, atípicos) de las variables del catálogo
num_cat = ["mag", "depth", "latitude", "longitude", "fases", "rms", "gap_az", "err_lat", "err_lon", "err_prof"]
r_cat = eda.resumen_numerico(cat_tr[num_cat])
eda.guardar_tabla(r_cat.reset_index().rename(columns={"index": "variable"}), "univariado_catalogo.csv")
r_cat.round(2)

# %%
# Histogramas con densidad de las variables del catálogo
fig, axes = plt.subplots(2, 5, figsize=(17, 6))
for ax, c in zip(axes.ravel(), num_cat):
    sns.histplot(cat_tr[c], bins=60, kde=True, ax=ax, color="#4299e1")
    ax.set_title(f"{c} (asim. {es(r_cat.loc[c, 'asimetría'], 2)})"); ax.set_xlabel("")
plt.tight_layout(); eda.guardar_figura(fig, "04_histogramas_catalogo.png"); plt.show()

# %% [markdown]
# ## 4.2 Variables numéricas del panel (predictoras del modelo base)

# %%
# Resumen numérico de las 18 predictoras del panel
r_pan = eda.resumen_numerico(P[X.columns])
eda.guardar_tabla(r_pan.reset_index().rename(columns={"index": "variable"}), "univariado_panel.csv")
r_pan.round(2)

# %%
# Histogramas de las predictoras; los conteos se muestran en log1p por su asimetría
fig, axes = plt.subplots(3, 6, figsize=(19, 8.5))
for ax, c in zip(axes.ravel(), X.columns):
    x = P[c].dropna()
    es_conteo = c.startswith(("n_", "n35", "vec", "n_mismo"))
    sns.histplot(np.log1p(x) if es_conteo else x, bins=40, ax=ax, color="#ed8936" if es_conteo else "#4299e1")
    ax.set_title(("log1p " if es_conteo else "") + c, fontsize=9); ax.set_xlabel("")
plt.tight_layout(); eda.guardar_figura(fig, "04_histogramas_panel.png"); plt.show()

# %%
# Diagramas de caja de conteos (log1p) y magnitudes
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
conteos = [c for c in X.columns if c.startswith(("n_", "n35", "vec", "n_mismo"))]
sns.boxplot(data=np.log1p(P[conteos]), orient="h", ax=axes[0], color="#fbd38d"); axes[0].set_title("Conteos (escala log1p)")
otras = ["mag_media_3m", "mag_media_12m", "mag_max_3m", "mag_max_12m"]
sns.boxplot(data=P[otras], orient="h", ax=axes[1], color="#90cdf4"); axes[1].set_title("Magnitudes de la ventana")
plt.tight_layout(); eda.guardar_figura(fig, "04_boxplots_panel.png"); plt.show()

# %% [markdown]
# ## 4.3 Pruebas de normalidad (con cautela)
#
# Con decenas de miles de filas, cualquier desviación mínima rechaza la normalidad, así que el p-valor
# por sí solo no informa. Se reportan la prueba de **D'Agostino-Pearson** (asimetría y curtosis
# conjuntas) y la de **Shapiro-Wilk** sobre una submuestra de 5.000, junto con la asimetría y la
# curtosis, que son las que sí guían decisiones.

# %%
# Pruebas de normalidad (D'Agostino y Shapiro sobre submuestra) para justificar transformaciones
filas = []
rng = np.random.default_rng(cfg.SEMILLA)
for c in ["mag", "depth"]:
    x = cat_tr[c].to_numpy()
    filas.append(("catálogo", c, *stats.normaltest(x), stats.shapiro(rng.choice(x, 5000, replace=False))[1], stats.skew(x), stats.kurtosis(x)))
for c in X.columns:
    x = P[c].dropna().to_numpy()
    if np.std(x) == 0:
        continue
    xx = np.log1p(x) if c.startswith(("n_", "n35", "vec", "n_mismo")) else x
    filas.append(("panel", c + (" (log1p)" if xx is not x else ""), *stats.normaltest(xx),
                  stats.shapiro(rng.choice(xx, min(5000, len(xx)), replace=False))[1], stats.skew(xx), stats.kurtosis(xx)))
norm = pd.DataFrame(filas, columns=["nivel", "variable", "K2_DAgostino", "p_DAgostino", "p_Shapiro_5000", "asimetría", "curtosis"])
norm["p_DAgostino_Holm"] = eda.holm(norm.p_DAgostino)
eda.guardar_tabla(norm, "normalidad.csv")
norm.round(4)

# %%
md(f"""**Interpretación (numéricas).**
- **Ninguna variable es normal.** La prueba rechaza en {int((norm.p_DAgostino_Holm < 0.05).sum())} de {len(norm)} variables tras Holm. Lo útil es la forma:
  - **Conteos (`n_*`, `vec_*`).** Muy asimétricos a la derecha incluso en log1p, con la mayor parte de la masa en 0 y una cola
    hacia el nido de Bucaramanga. Mediana de `n_1m` = {es(r_pan.loc['n_1m', 'mediana'], 0)}, p99 = {es(r_pan.loc['n_1m', 'p99'], 0)}, máximo = {es(r_pan.loc['n_1m', 'máx'], 0)}.
    Los atípicos por IQR ({es(r_pan.loc['n_1m', 'atípicos_IQR_%'], 0)} % en `n_1m`) son celdas muy activas reales.
    **Decisión (cap. 10):** transformar con log1p en los modelos sensibles a la escala, como SVR lineal y KNN, en lugar de recortar.
  - **Magnitudes de ventana.** Acotadas por abajo en 2,5 y asimétricas por la ley de Gutenberg-Richter. La magnitud media de
    3 meses falta en {es(r_pan.loc['mag_media_3m', 'faltantes'] / len(P), 0, pct=True)} de las filas (celdas sin sismos; capítulo 2).
  - **Profundidad media histórica.** Bimodal: celdas superficiales y celdas del nido (≈ 150 km).
  - **`sen_mes`, `cos_mes`, `lat`, `lon`.** Uniformes o discretas por construcción; la prueba de normalidad no tiene sentido
    para ellas y se reporta solo por completitud.
- **Implicación.** Los modelos lineales (SVR lineal, Ridge, Lasso) no requieren normalidad de las predictoras, pero sí se
  benefician de escalas comparables y de reducir la influencia de los extremos: de ahí log1p y estandarización.""")

# %% [markdown]
# ## 4.4 Variables categóricas: cardinalidad, frecuencias y categorías raras

# %%
# Variables categóricas: departamento, tipo de magnitud y mes calendario; categorías raras (< 1 %)
dep = cat_tr.depto.value_counts().to_frame("sismos")
dep["%"] = 100 * dep.sismos / dep.sismos.sum()
dep["rara (< 1 %)"] = dep["%"] < 1
tipo = cat_tr.tipo_mag.value_counts().to_frame("sismos")
tipo["%"] = 100 * tipo.sismos / tipo.sismos.sum()
mes_cal = cat_tr.time.dt.month.value_counts().sort_index().to_frame("sismos")
eda.guardar_tabla(dep.reset_index(), "frecuencias_departamento.csv")
eda.guardar_tabla(tipo.reset_index(), "frecuencias_tipo_magnitud.csv")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1.4, 1, 1]})
sns.barplot(x=dep["%"], y=dep.index, ax=axes[0], color="#4299e1"); axes[0].set_xscale("log")
axes[0].set_title(f"Departamento (cardinalidad {len(dep)}, escala log)"); axes[0].tick_params(axis="y", labelsize=7)
sns.barplot(x=tipo["%"].head(10), y=tipo.index[:10], ax=axes[1], color="#ed8936"); axes[1].set_title("Escala de magnitud (top 10)")
sns.barplot(x=mes_cal.index, y=mes_cal.sismos, ax=axes[2], color="#48bb78"); axes[2].set_title("Mes calendario del sismo")
plt.tight_layout(); eda.guardar_figura(fig, "04_categoricas.png"); plt.show()
dep.head(12).round(2)

# %%
md(f"""**Interpretación (categóricas).**
- **Departamento.** Cardinalidad {len(dep)}, muy concentrada: **{dep.index[0].title()}** reúne el {es(dep['%'].iloc[0], 1)} % de los sismos del entrenamiento y
  los cinco primeros el {es(dep['%'].head(5).sum(), 1)} %. Hay **{int(dep['rara (< 1 %)'].sum())} departamentos raros** (< 1 % cada uno), que juntos suman
  el {es(dep.loc[dep['rara (< 1 %)'], '%'].sum(), 1)} %.
  - **Tratamiento:** el departamento **no entra al modelo**. La unidad espacial es la celda de 0,5°, más fina y homogénea, y
    las coordenadas de la celda ya contienen esa información. Se usa solo para describir.
  - Si se usara, las categorías raras se agruparían en «OTROS», con el umbral decidido en entrenamiento.
- **Escala de magnitud.** El periodo de entrenamiento mezcla Ml (portal histórico, {es(tipo.loc['Ml', '%'] if 'Ml' in tipo.index else 0, 1)} %) con
  variantes MLr del portal actual. Las variantes con menos del 1 % (Mw, M, …) se agruparían como «otras». La escala no entra al
  modelo; su posible efecto se discute como deriva (capítulo 7).
- **Mes calendario.** Las frecuencias por mes son parecidas, sin un patrón estacional evidente (la prueba formal está en el
  capítulo 7).""")
