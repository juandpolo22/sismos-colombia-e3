# %% [markdown]
# # 3. Análisis de la variable objetivo (2.1)
#
# **Objetivo:** número de sismos M ≥ 2,5 en la celda durante el mes siguiente.
#
# Todo el capítulo usa **solo el entrenamiento**, es decir, los meses objetivo hasta noviembre de 2020.
# Las conclusiones de aquí deciden la métrica, la transformación del objetivo y la validación.

# %% [cabecera]

# %%
# Distribución del objetivo (sismos por celda-mes) en el entrenamiento: frecuencias y acumulado
ev_, X, info, tr, gp, te = datos.panel_completo()
P = datos.panel_dataframe(X, info, tr)
y = P["y"].to_numpy()
frec = (pd.Series(y).value_counts().sort_index().rename_axis("sismos_en_el_mes").to_frame("celdas_mes"))
frec["%"] = 100 * frec.celdas_mes / len(y)
frec["% acumulado"] = frec["%"].cumsum()
desc = pd.DataFrame({
    "n": [len(y)], "media": [y.mean()], "varianza": [y.var()], "índice de dispersión (var/media)": [y.var() / y.mean()],
    "mediana": [np.median(y)], "p90": [np.percentile(y, 90)], "p99": [np.percentile(y, 99)], "máx": [y.max()],
    "% ceros": [100 * np.mean(y == 0)], "asimetría": [stats.skew(y)], "curtosis": [stats.kurtosis(y)],
    "asimetría log1p": [stats.skew(np.log1p(y))],
})
eda.guardar_tabla(desc, "objetivo_resumen.csv")
eda.guardar_tabla(frec.reset_index(), "objetivo_frecuencias.csv")
display(desc.round(2))
frec.head(12).round(2)

# %%
# Gráficos del objetivo: zona 0–20, cola en escala log-log y log(1 + y)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].bar(frec.index[:21], frec["%"].iloc[:21], color="#2b6cb0")
axes[0].set_title("Distribución del objetivo (0 a 20)"); axes[0].set_xlabel("sismos en el mes siguiente"); axes[0].set_ylabel("% de celdas-mes")
axes[1].hist(y[y > 0], bins=np.logspace(0, np.log10(y.max() + 1), 40), color="#dd6b20")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_title("Cola (y > 0, escalas log-log)")
axes[2].hist(np.log1p(y), bins=40, color="#38a169"); axes[2].set_title("log(1 + y)")
plt.tight_layout(); eda.guardar_figura(fig, "03_distribucion_objetivo.png"); plt.show()

# %% [markdown]
# ## ¿Poisson, binomial negativa o inflada en ceros?
#
# Se compara la proporción de ceros observada con la que predicen una Poisson y una binomial negativa
# de la misma media. Primero se hace globalmente y después **dentro de cada celda**, porque la mezcla
# de celdas activas e inactivas por sí sola ya produce sobredispersión.

# %%
# Exceso de ceros y sobredispersión: ceros observados frente a Poisson y binomial negativa, global y por celda
mu, v = y.mean(), y.var()
r_nb = mu ** 2 / max(v - mu, 1e-9)
global_ = {"observado": np.mean(y == 0), "Poisson": np.exp(-mu), "Binomial negativa": (r_nb / (r_nb + mu)) ** r_nb}
por_celda = P.groupby("celda").y.agg(["mean", "var", lambda s: np.mean(s == 0)]).rename(columns={"<lambda_0>": "ceros"})
por_celda["ceros_Poisson"] = np.exp(-por_celda["mean"])
por_celda["dispersion"] = por_celda["var"] / por_celda["mean"].replace(0, np.nan)
ceros = pd.DataFrame({"proporción de ceros": global_}).T
eda.guardar_tabla(por_celda.reset_index(), "objetivo_por_celda.csv")
display(ceros.round(3))
md(f"""**Interpretación.**
- **Global.** Una Poisson con la media del entrenamiento ({es(mu, 2)}) esperaría {es(global_['Poisson'], 1, pct=True)} de ceros; se observan
  {es(global_['observado'], 1, pct=True)}. El índice de dispersión var/media = {es(v / mu, 1)} (1 en una Poisson) indica **sobredispersión
  extrema**. Una binomial negativa con la misma media y varianza se pasa al otro extremo ({es(global_['Binomial negativa'], 1, pct=True)} de ceros):
  **ninguna distribución única** describe bien la mezcla de celdas tan distintas.
- **Dentro de cada celda.** La sobredispersión es **moderada en la celda típica** (índice var/media mediano de
  {es(por_celda.dispersion.median(), 1)}), pero fuerte en las celdas activas: en {es((por_celda.dispersion > 1.5).mean(), 0, pct=True)} de las celdas pasa de 1,5.
  La mediana del exceso de ceros sobre una Poisson con la media de la propia celda es de solo {es((por_celda.ceros - por_celda.ceros_Poisson).median(), 1, pct=True)}.
  Conclusión: la sobredispersión global se debe **sobre todo a la heterogeneidad entre celdas** y, en las celdas activas,
  al agrupamiento temporal.
- **Por qué importa.** El modelo debe distinguir las celdas (heterogeneidad) y, en las activas, capturar que los sismos llegan
  **en grupos** (réplicas, enjambres). Esto es coherente con los modelos de
  autoexcitación tipo ETAS (Ogata, 1988), en los que un sismo aumenta la probabilidad de otros cercanos en tiempo y espacio.""")

# %% [markdown]
# ## Valores extremos: los enjambres

# %%
# Las 10 celdas-mes con más sismos y la magnitud máxima de ese mes
top = P.nlargest(10, "y")[["fecha_obj", "celda", "lat", "lon", "y", "n_1m", "n_12m"]]
mmax = ev_.groupby(["celda", "mes"]).mag.max()
top["mag_max_en_el_mes"] = [mmax.get((c, m), np.nan) for c, m in zip(top.celda, P.loc[top.index, "mes_obj"])]
top["fecha_obj"] = top.fecha_obj.dt.strftime("%Y-%m")
eda.guardar_tabla(top, "objetivo_extremos.csv")
top

# %%
# Separa los extremos del nido de Bucaramanga de los demás para interpretarlos
nido = top[(top.lat == 6.75) & (top.lon == -73.25)]
otra = top.drop(nido.index)
md(f"""**Interpretación.** Las 10 celdas-mes con más sismos tienen entre {int(top.y.min())} y {int(top.y.max())} sismos. Son de dos tipos:
- **{len(nido)} están en el nido de Bucaramanga** (lat 6,75, lon -73,25). Ahí el mes anterior ya era muy activo (`n_1m` entre
  {int(nido.n_1m.min())} y {int(nido.n_1m.max())}): son extremos **persistentes** y en buena parte predecibles con el pasado reciente.
- **{len(otra)} es una secuencia que empieza de golpe**: celda ({', '.join(f'{a}, {b}' for a, b in zip(otra.lat, otra.lon))}) en {', '.join(otra.fecha_obj)}, con
  `n_1m` = {', '.join(str(int(v)) for v in otra.n_1m)} el mes anterior y un sismo de magnitud máxima {', '.join(es(v, 1) for v in otra.mag_max_en_el_mes)}
  en el mes. Es la típica secuencia de réplicas tras un sismo fuerte: **no se puede anticipar con el pasado de la celda** y será
  un error grande de cualquier modelo.

Ninguno es un error de registro. Se conservan y se tratan con la escala logarítmica, sin recortarlos.""")

# %% [markdown]
# ## Relación con las predictoras principales (resumen; detalle en el capítulo 5)

# %%
# Relación monótona (Spearman) de cada predictora con el objetivo
rel = []
for c in X.columns:
    x = P[c].to_numpy()
    ok = ~np.isnan(x)
    rho, p = stats.spearmanr(x[ok], y[ok])
    rel.append({"variable": c, "Spearman_rho": rho, "p": p})
rel = pd.DataFrame(rel)
rel["p_FDR"] = eda.fdr_bh(rel.p)
rel = rel.sort_values("Spearman_rho", key=np.abs, ascending=False)
eda.guardar_tabla(rel, "objetivo_spearman.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=rel, y="variable", x="Spearman_rho", color="#4299e1", ax=ax)
ax.set_title("Correlación de Spearman de cada predictora con el objetivo (entrenamiento)")
eda.guardar_figura(fig, "03_spearman_objetivo.png"); plt.show()
md(f"""**Interpretación.** Las predictoras más asociadas al conteo del mes siguiente son los **conteos recientes de la propia celda**
({', '.join(rel.variable.head(4))}, ρ entre {es(rel.Spearman_rho.head(4).min(), 2)} y {es(rel.Spearman_rho.head(4).max(), 2)}). Las variables de
calendario (`sen_mes`, `cos_mes`) casi no se asocian (|ρ| < {es(rel.loc[rel.variable.isin(['sen_mes', 'cos_mes']), 'Spearman_rho'].abs().max() + 0.005, 2)}).
La sismicidad no tiene estacionalidad apreciable, algo que el capítulo 7 confirma formalmente.""")

# %% [markdown]
# ## Comportamiento en el tiempo y en el espacio

# %%
# Evolución mensual del total de sismos y de la proporción de ceros, y media por celda en el mapa
mensual = P.groupby("fecha_obj").y.agg(["sum", "mean", lambda s: np.mean(s == 0)]).rename(columns={"<lambda_0>": "ceros"})
celda = P.groupby(["celda", "lat", "lon"]).y.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
axes[0].plot(mensual.index, mensual["sum"], lw=0.8, color="#2b6cb0")
axes[0].set_title("Sismos en las celdas activas por mes objetivo (entrenamiento)")
ax2 = axes[0].twinx(); ax2.plot(mensual.index, mensual["ceros"] * 100, lw=0.8, color="#a0aec0"); ax2.set_ylabel("% celdas en cero", color="#718096")
sc = axes[1].scatter(celda.lon, celda.lat, c=np.log1p(celda.y), s=40, marker="s", cmap="magma_r")
plt.colorbar(sc, ax=axes[1], label="log(1 + media mensual)"); axes[1].set_title("Media del objetivo por celda")
axes[1].set_xlabel("longitud"); axes[1].set_ylabel("latitud"); axes[1].set_aspect("equal")
plt.tight_layout(); eda.guardar_figura(fig, "03_objetivo_tiempo_espacio.png"); plt.show()
md(f"""**Interpretación.**
- **En el tiempo.** El total mensual oscila alrededor de {es(mensual['sum'].median(), 0)} sismos (mediana), con picos aislados de
  enjambres. El porcentaje de celdas en cero se mantiene estable (entre {es(mensual.ceros.quantile(0.05), 0, pct=True)} y
  {es(mensual.ceros.quantile(0.95), 0, pct=True)} en el 90 % de los meses).
- **En el espacio.** La media por celda varía en **más de tres órdenes de magnitud**: de {es(celda.y.min(), 2)} a {es(celda.y.max(), 1)} sismos/mes.
  Una sola celda (Bucaramanga) concentra buena parte del total. Esta heterogeneidad explica la sobredispersión global y exige
  que el modelo use la **historia propia de cada celda**.""")

# %% [markdown]
# ## Implicaciones para la métrica y la validación
#
# 1. **Transformación del objetivo.**
#    - Con 76 % de ceros y cola de ley de potencia, Box-Cox no es aplicable (requiere y > 0).
#    - Se modela **log(1 + y)**: reduce la asimetría de 13,2 a 3,9 (tabla de arriba) y convierte el
#      error relativo en error absoluto. La asimetría restante se debe a la masa en cero, que ninguna
#      transformación monótona elimina.
#    - Los modelos se entrenan sobre log1p y se evalúan en la escala original con `expm1`.
# 2. **Métrica principal: RMSLE.** Coincide con la pérdida en escala log y no se deja dominar por los
#    enjambres, a diferencia del RMSE. El MAPE no se puede usar como métrica principal porque se
#    indefine con ceros.
# 3. **Validación.**
#    - La dependencia temporal (enjambres que duran varios meses) obliga a una **partición
#      cronológica con gap**.
#    - La heterogeneidad entre celdas obliga a reportar también un esquema **por bloques espaciales**,
#      que mide cuánto empeora el modelo en zonas que no vio (capítulo 11).
#    - La incertidumbre se estima **remuestreando meses**.
# 4. **Modelos de conteo.** La sobredispersión y el exceso de ceros dentro de cada celda justifican,
#    en el Entregable 2, probar pérdidas de Poisson y modelos binomiales negativos inflados en ceros
#    (ZINB).
