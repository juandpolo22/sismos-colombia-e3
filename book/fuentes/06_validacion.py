# %% [markdown]
# # 6. Validación estadística de la superioridad
#
# **Protocolo común.** Todos los modelos (benchmarks y A+B) se evalúan en el mismo origen móvil 2021–2026: seis orígenes
# anuales; en cada uno se reentrena con todos los meses anteriores (gap de 1 mes) y se pronostica el año, mes a mes. Las
# particiones, la semilla global y el preprocesamiento son los mismos para todos. Los hiperparámetros de los benchmarks y
# la elección del campeón de cada familia usan solo datos anteriores a 2016.
#
# **Unidad de análisis.** La pérdida mensual L_t = media sobre celdas de (log(1+ŷ) − log(1+y))². Las 163 celdas de un
# mismo mes no son independientes (dependencia espacial), así que las pruebas comparan 69 pérdidas mensuales, no 11 754
# filas.
#
# **Superioridad (guía, sección 6.1).** (1) Menor RMSLE que el campeón de cada familia; (2) al menos tres pruebas de
# hipótesis distintas rechazan H0 con p ajustado por Holm < 0,05 (< 0,10 en el MCS); (3) tamaño del efecto no
# despreciable (|d de Cohen| ≥ 0,2).

# %% [cabecera]

# %%
from src import data as D
dm = D.tabla("e2_stats_dm.csv"); gc = D.tabla("e2_stats_gw_cw.csv").set_index("rival"); sp = D.tabla("e2_stats_spa.csv").set_index("benchmark")
fam = D.tabla("e2_stats_familia_holm.csv"); mcs = D.tabla("e2_stats_mcs_origen_movil.csv")
R = dm.set_index("rival")
res = pd.DataFrame({
    "RMSLE campeón": R["RMSLE rival"], "dif. media pérdida": R["dif. media pérdida"], "IC 95 % (bootstrap estacionario)": R["IC95 bootstrap estacionario"],
    "p DM Holm": R["p Holm (7)"], "p GW Holm": gc["p GW Holm"], "p CW Holm": gc["p CW Holm"], "p SPA Holm": sp["p SPA Holm (7)"],
    "d de Cohen": R["d de Cohen"], "meses con A+B mejor": R["meses con A+B mejor"]})
tabla(res, dec=4)

# %%
ab_rmsle = float(np.sqrt(mcs.loc[mcs.modelo.str.startswith("A+B"), "perdida_media"].iloc[0]))
cond1 = (R["RMSLE rival"] > D.lideres().set_index("etiqueta").loc["A+B v2", "RMSLE origen móvil 2021–2026"]).all()
cond2 = (res[["p DM Holm", "p GW Holm", "p CW Holm", "p SPA Holm"]] < 0.05).all().all() and mcs.en_MCS.sum() == 1
cond3 = (R["d de Cohen"].abs() >= 0.2).all()
md(f"**Verificación de la definición operativa:** (1) menor RMSLE que los 7 campeones: **{'sí' if cond1 else 'no'}**; "
   f"(2) DM, GW, CW y SPA con p de Holm < 0,05 frente a los 7 y A+B único en el MCS (p de Holm máximo sobre la familia de 21 "
   f"contrastes: {fam['p Holm (21)'].max():.4f}): **{'sí' if cond2 else 'no'}**; (3) |d| ≥ 0,2 en todos (mínimo "
   f"{R['d de Cohen'].abs().min():.2f}): **{'sí' if cond3 else 'no'}**. Comparaciones realizadas: 7 por prueba, 21 en la familia "
   f"DM + GW + CW, 7 en SPA/Reality Check y un MCS sobre 10 modelos.")

# %% [markdown]
# ## Tabla de pruebas de hipótesis
#
# | Prueba | Hipótesis nula | Supuestos y cómo se verificaron | Ajuste |
# |---|---|---|---|
# | Model Confidence Set | igual pérdida esperada entre los modelos del conjunto | pérdidas débilmente dependientes: bootstrap estacionario (Politis y Romano) | controla la familia por construcción |
# | Diebold-Mariano (HLN) | E[d_t] = 0 | d_t estacionaria y sin autocorrelación fuerte: Escanciano-Lobato y Breusch-Godfrey; varianza Newey-West | Holm (7) |
# | Giacomini-White | E[d_t \| 1, d_{t−1}] = 0 | reestimación con ventana finita (origen móvil anual) | Holm (7) |
# | Clark-West | igual precisión poblacional en modelos anidados | cada campeón es miembro de A+B (anidamiento) | Holm (7) |
# | SPA / Reality Check | ninguna versión del modelo propio supera al campeón | 4 versiones (final, sin calibración, sin B, sin B ni ZINB); bootstrap estacionario | Holm (7) |
#
# La sección 6.1 (capítulo 21 del Entregable 2, ejecutado de nuevo con el modelo final) contiene el detalle de cada
# prueba, las diferencias mensuales d_t, los intervalos, las pruebas de independencia y la verificación de Ridge y Lasso.
