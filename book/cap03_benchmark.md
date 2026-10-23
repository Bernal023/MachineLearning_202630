# Capítulo 3 · Modelos benchmark y mejores combinaciones

Este capítulo compara **HAR-Ridge + XGBoost** con los modelos de referencia, primero con el diseño original (Grid Search, validación cruzada temporal) y
después con la ampliación que reoptimiza a los rivales con otras tres búsquedas y validación anidada.

```{code-cell} ipython3
:tags: [remove-cell]
import json, tempfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from IPython.display import display
from jbook_utils import (R, REPO, SEED, HAR, FAMILIES, SEARCHES, read_csv, glue_fig, glue_tbl)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
```

## 3.1 HAR-Ridge + XGBoost frente a seis modelos de referencia

La pregunta es cuánto mejora HAR-Ridge + XGBoost frente a cada modelo, **con los mismos objetivos y fechas**.

| Modelo comparado | Cómo representa la información histórica |
|---|---|
| k-NN | Promedia ejemplos históricos próximos en las características. |
| Ridge | Regresión lineal con penalización L2 {cite}`hoerl1970ridge`. |
| Lasso | Regresión lineal con penalización L1; puede anular coeficientes {cite}`tibshirani1996lasso`. |
| Random Forest | Promedia árboles para representar relaciones no lineales {cite}`breiman2001random`. |
| XGBoost | Añade árboles sucesivos para corregir errores del conjunto {cite}`chen2016xgboost`. |
| SVR Lineal | Usa un margen $\epsilon$ y regularización {cite}`drucker1997svr`. |
| **HAR-Ridge + XGBoost** | Reúne estructura temporal regularizada y relaciones no lineales. |

En la ampliación se añade un **Decision Tree** (marcado con asterisco en las tablas del Grid) y se reoptimizan los rivales con tres búsquedas adicionales:
Random Search {cite}`bergstra2012random`, bayesiana con Optuna TPE {cite}`akiba2019optuna` y genética con DEAP {cite}`fortin2012deap`.
Los modelos guardados no se reentrenan al cambiar los filtros del dashboard.

## 3.2 Validación cruzada temporal del Grid Search

El Grid Search usa **seis cortes con entrenamiento creciente** (*expanding window*) y validación bimestral en 2024; la configuración elegida se fija y se evalúa en
2025, que no interviene en la selección. La selección es por RMSE de validación y el escalado, cuando corresponde, se ajusta solo con el entrenamiento.
Las seis familias originales y HAR-Ridge + XGBoost usan los mismos seis cortes.

```{code-cell} ipython3
# Calendario de los seis cortes (results/improved_classical_2023_2025/calendar.csv)
cal = read_csv(REPO / "results" / "improved_classical_2023_2025" / "calendar.csv", parse_dates=["origin", "target_end"])
filas = []
for f, g in cal.groupby("fold"):
    tr, va = g[g.split == "train"], g[g.split == "validation"]
    filas.append({"corte": f, "entrenamiento": len(tr), "validación": len(va),
                  "último_origen_train": tr.origin.max().date(), "primer_origen_val": va.origin.min().date(),
                  "separación_días": (va.origin.min() - tr.origin.max()).days,
                  "etiqueta_train_termina_antes_días": (va.origin.min() - tr.target_end.max()).days})
cortes = pd.DataFrame(filas).set_index("corte")
assert cortes["entrenamiento"].tolist() == [274, 334, 395, 456, 518, 579] and cortes["validación"].tolist() == [60, 61, 61, 62, 61, 54]
assert cortes["validación"].sum() == 359 and (cortes["separación_días"] == 28).all() and (cortes["etiqueta_train_termina_antes_días"] == 21).all()
assert 660 in read_csv(R / "selected_inputs.csv")["n_train"].unique() and len(read_csv(R / "window_review_original_vs_optimized_2025" / "daily_losses.csv").query("volatility_window == 21")) == 358

fig, ax = plt.subplots(figsize=(10, 3.2))
for f, g in cal.groupby("fold"):
    tr, va = g[g.split == "train"], g[g.split == "validation"]
    ax.barh(f, (tr.origin.max() - tr.origin.min()).days, left=tr.origin.min().toordinal(), color="#8fb4d9", height=.6)
    ax.barh(f, (va.origin.max() - va.origin.min()).days, left=va.origin.min().toordinal(), color="#1f4e79", height=.6)
ticks = [pd.Timestamp(d).toordinal() for d in ["2023-02-01", "2023-07-01", "2024-01-01", "2024-07-01", "2024-12-01"]]
ax.set_xticks(ticks); ax.set_xticklabels(["feb. 2023", "jul. 2023", "ene. 2024", "jul. 2024", "dic. 2024"])
ax.set_yticks(range(1, 7)); ax.invert_yaxis(); ax.set_ylabel("Corte"); ax.set_xlabel("Fecha del origen")
ax.set_title("Cortes de validación cruzada temporal (claro: entrenamiento; oscuro: validación)")
fig.tight_layout(); glue_fig("fig-cortes", fig)
glue_tbl("tbl-cortes", cortes)
cortes
```

```{glue:figure} tbl-cortes
:name: tbl-cortes
:figwidth: 100%

Cortes de la validación cruzada temporal: orígenes de entrenamiento y validación, y separación entre ambos (días).
```

```{glue:figure} fig-cortes
:name: fig-cortes
:figwidth: 90%

Orígenes de entrenamiento (claro) y validación (oscuro) de cada corte; eje x: fecha del origen.
```

**Interpretación de la {numref}`tbl-cortes` y la {numref}`fig-cortes`.** Los seis cortes tienen entre 274 y 579 orígenes de entrenamiento y entre 54 y 62 de
validación (359 en total). Entre el último origen de entrenamiento y el primero de validación hay **28 días**, y la última etiqueta de entrenamiento termina **21 días antes**
de la validación, de modo que ninguna etiqueta de entrenamiento se solapa con la validación (control contra la fuga de información). Después de elegir la
configuración, el modelo se **reajusta con 660 orígenes** y se evalúa en **358 orígenes de 2025**.

## 3.3 Resultados de 2025 con la ventana de 21 días

La ventana $w=21$ es la **elección histórica** de la presentación (mayor $R^2$ entre las ventanas con mayoría favorable bajo DM-Holm). Se comparan cuatro activos y siete horizontes
con 358 orígenes comunes, del 1 de enero al 24 de diciembre de 2025. RMSE y MAE están en puntos porcentuales de volatilidad; el MSE, en puntos porcentuales al cuadrado; el $R^2$, sin
unidades. Son promedios por activo y horizonte, por lo que **el RMSE medio es distinto de la raíz del MSE medio**.

```{code-cell} ipython3
# Métricas 2025, w = 21, por procedimiento de búsqueda
macro = read_csv(R / "evaluation" / "macro_by_window.csv")
macro = macro[(macro.year == 2025) & (macro.volatility_window == 21)].set_index("model")
orig = read_csv(REPO / "results" / "professor_feedback_20261018" / "metrics_by_window.csv")
orig = orig[(orig.year == 2025) & (orig.volatility_window.astype(str) == "21")].set_index("model")

def tabla_busqueda(search):
    filas = {HAR: orig.loc[HAR, ["mse", "rmse", "mae", "r2"]]}
    for f, n in FAMILIES.items():
        if search == "grid":
            r = macro.loc["DecisionTree [grid]"] if f == "DecisionTree" else orig.loc[n]
            n = n + ("*" if f == "DecisionTree" else "")
        else:
            r = macro.loc[f"{f} [{search}]"]
        filas[n] = r[["mse", "rmse", "mae", "r2"]]
    t = pd.DataFrame(filas).T.astype(float)
    return pd.concat([t.loc[[HAR]], t.drop(HAR).sort_values("mse")]).round(5)

tablas = {s: tabla_busqueda(s) for s in SEARCHES}
def victorias(t):   # el original frente a los 7 rivales: R² mayor, RMSE menor, MAE menor
    h, r = t.loc[HAR], t.drop(HAR)
    return (int((h.r2 > r.r2).sum()), int((h.rmse < r.rmse).sum()), int((h.mae < r.mae).sum()))
resumen = pd.DataFrame({SEARCHES[s]: dict(zip(["R2 mayor", "RMSE menor", "MAE menor"], victorias(t))) for s, t in tablas.items()}).T
assert resumen.loc["Grid Search*"].tolist() == [7, 7, 7] and resumen.loc["Random Search"].tolist() == [7, 7, 6]
assert resumen.loc["Genética (DEAP)"].tolist() == [7, 6, 5]
assert abs(tablas["grid"].loc[HAR, "mse"] - .18444) < 1e-5 and abs(tablas["bayesian"].loc["Lasso", "r2"] - .85) < 1e-4
for s, t in tablas.items():
    glue_tbl(f"tbl-w21-{s}", t)
glue_tbl("tbl-w21-resumen", resumen)
tablas["grid"]
```

```{glue:figure} tbl-w21-grid
:name: tbl-w21-grid
:figwidth: 90%

Grid Search, 2025, $w=21$: MSE, RMSE, MAE y $R^2$. *Decision Tree añadido (ocho candidatos y validación temporal anidada).
```

```{code-cell} ipython3
tablas["random"]
```

```{glue:figure} tbl-w21-random
:name: tbl-w21-random
:figwidth: 90%

Random Search, 2025, $w=21$: métricas de HAR-Ridge + XGBoost y de los siete rivales reoptimizados.
```

```{code-cell} ipython3
tablas["bayesian"]
```

```{glue:figure} tbl-w21-bayesian
:name: tbl-w21-bayesian
:figwidth: 90%

Bayesiana (Optuna TPE), 2025, $w=21$: métricas de HAR-Ridge + XGBoost y de los siete rivales reoptimizados.
```

```{code-cell} ipython3
tablas["genetic"]
```

```{glue:figure} tbl-w21-genetic
:name: tbl-w21-genetic
:figwidth: 90%

Genética (DEAP), 2025, $w=21$: métricas de HAR-Ridge + XGBoost y de los siete rivales reoptimizados.
```

```{code-cell} ipython3
resumen
```

```{glue:figure} tbl-w21-resumen
:name: tbl-w21-resumen
:figwidth: 80%

Número de rivales (de 7) frente a los que HAR-Ridge + XGBoost tiene mayor $R^2$, menor RMSE y menor MAE, por búsqueda.
```

**Interpretación de las {numref}`tbl-w21-grid`–{numref}`tbl-w21-resumen`.**

- **Grid Search:** HAR-Ridge + XGBoost tiene el menor error y el mayor $R^2$ de la tabla (MSE 0,18444; RMSE 0,38601; MAE 0,25372; $R^2$ 0,8445), mejor que los siete rivales en las tres métricas.
  Para comparar, la persistencia en $w=21$ tiene MSE 0,37183 ($R^2$ 0,6885).
- **Random Search:** el modelo original sigue teniendo el mayor $R^2$ frente a 7/7 rivales y el menor RMSE frente a 7/7, pero el menor MAE solo frente a 6/7: **SVR lineal
  tiene un MAE ligeramente menor** (0,25002 frente a 0,25372).
- **Bayesiana (Optuna TPE):** **Lasso supera al original en las cuatro métricas observadas** (MSE 0,17825; RMSE 0,37874; MAE 0,23904; $R^2$ 0,8500). Que sea mejor en el promedio no
  implica ventaja estadística: eso se evalúa en el Capítulo 6.
- **Genética (DEAP):** el original mantiene el mayor $R^2$ frente a 7/7, el menor RMSE frente a 6/7 y el menor MAE frente a 5/7 (Lasso tiene menor RMSE; Lasso y SVR lineal tienen menor MAE).

Es decir, el **tamaño de la ventaja del original depende de cómo se optimizó al rival**: con el Grid original es amplia, y con rivales reoptimizados se estrecha o desaparece frente a Lasso.

## 3.4 Ampliación: espacio de modelos evaluado

La plantilla de la rúbrica menciona «140 combinaciones». En este proyecto, de regresión, el espacio
real es el siguiente y es el que se reporta:

- **7 familias de regresión base:** k-NN, Ridge {cite}`hoerl1970ridge`, Lasso {cite}`tibshirani1996lasso`,
  árbol de decisión, *random forest* {cite}`breiman2001random`, XGBoost {cite}`chen2016xgboost` y SVR
  {cite}`drucker1997svr`; más la familia **híbrida HAR-XG** (Ridge sobre entradas HAR + XGBoost).
- **4 procedimientos de búsqueda de hiperparámetros:** *grid* (prefijo cartesiano determinista),
  *random* {cite}`bergstra2012random`, bayesiano TPE con Optuna {cite}`akiba2019optuna` y algoritmo
  genético con DEAP {cite}`fortin2012deap`, todos con presupuesto de **8 evaluaciones únicas** por búsqueda.
- **16 configuraciones** activo × ventana de volatilidad (4 × 4), cada una con 7 salidas (horizontes).

```{code-cell} ipython3
# Conteo del espacio de modelos a partir del protocolo congelado (protocol.json)
proto = json.loads((R / "protocol.json").read_text(encoding="utf-8"))
n_conf = len(proto["assets"]) * len(proto["target_windows"])
base = len(proto["regression_families"]) * len(proto["methods"]) * n_conf
hibr = len(proto["methods"]) * n_conf
espacio = pd.DataFrame({
    "concepto": ["Familias base", "Optimizadores", "Configuraciones activo×ventana",
                 "Benchmark (7×4×16)", "Híbridos HAR-XG (4×16)", "Pipelines finales"],
    "valor": [len(proto["regression_families"]), len(proto["methods"]), n_conf,
              base, hibr, base + hibr]}).set_index("concepto")
assert (base, hibr, base + hibr) == (448, 64, 512)
assert proto["benchmark_combinations"] == 448 and proto["final_models"] == 512
glue_tbl("tbl-espacio", espacio)
espacio
```

```{glue:figure} tbl-espacio
:name: tbl-espacio
:figwidth: 70%

Tamaño del espacio de modelos evaluado (cada pipeline predice 7 horizontes).
```

La {numref}`tbl-espacio` fija las cantidades: **448 modelos benchmark y 64 híbridos, 512 pipelines
finales**. Se evaluaron además 28 procedimientos «familia × optimizador» como unidades de comparación
(7 × 4), más los 4 de HAR-XG.

## 3.5 Protocolo de validación anidada con purga

Para cada una de las seis particiones cronológicas externas de 2024 se **repite toda la optimización
de hiperparámetros dentro del entrenamiento de ese pliegue**; el pliegue externo solo se usa para
puntuar. Es el diseño que evita el sesgo optimista de seleccionar y evaluar sobre los mismos datos
{cite}`varma2006bias`. Con $\mathcal{D}_k^{tr}$ y $\mathcal{D}_k^{te}$ el entrenamiento y la prueba del
pliegue externo $k$, y $\theta_k^\star$ los hiperparámetros elegidos solo con $\mathcal{D}_k^{tr}$:

```{math}
:label: eq-nested
\hat{\theta}_k^{\star}=\argmin_{\theta\in\Theta}\ \frac{1}{J}\sum_{j=1}^{J}
\mathrm{MSE}\!\left(f_\theta;\ \mathcal{D}_{k,j}^{val}\right),
\qquad
\widehat{\mathrm{Err}}=\frac{1}{K}\sum_{k=1}^{K}\mathrm{MSE}\!\left(f_{\hat\theta_k^\star};\ \mathcal{D}_k^{te}\right),
```

con $K=6$ pliegues externos, $J=2$ pliegues internos expansivos y $\mathrm{MSE}$ en unidades naturales
(puntos porcentuales al cuadrado) agrupado sobre los siete horizontes. **Purga:** como cada etiqueta
mira hasta 7 días adelante, la última etiqueta de entrenamiento termina estrictamente antes del
primer origen de validación; así se evita la fuga de información entre particiones.

```{code-cell} ipython3
# Calendario anidado: verificación de la purga estricta en TODAS las particiones
cal = read_csv(R / "nested_calendar.csv")
assert cal["strict_purge"].all() and cal["inner_validation_inside_outer_training"].all()
cal["validation_first"] = pd.to_datetime(cal["validation_first"]).dt.date
cal["validation_last"] = pd.to_datetime(cal["validation_last"]).dt.date
res = (cal[cal.outer_fold > 0].groupby("outer_fold")
       .agg(origenes_entrenamiento_min=("n_train", "min"), origenes_entrenamiento_max=("n_train", "max"),
            primer_origen_validacion=("validation_first", "min"),
            ultimo_origen_validacion=("validation_last", "max"),
            purga_estricta=("strict_purge", "all")))
glue_tbl("tbl-calendario", res)
res
```

```{glue:figure} tbl-calendario
:name: tbl-calendario
:figwidth: 100%

Calendario de los seis pliegues externos (fila 0 omitida: reajuste final con 660 orígenes): origen
de validación y verificación de purga estricta.
```

La {numref}`tbl-calendario` muestra que el entrenamiento crece de pliegue en pliegue (los
orígenes de entrenamiento mínimos van de 130 a 282) y que **las 1.920 particiones internas satisfacen la
purga estricta** (`strict_purge` verdadero en todas). Este es el control directo contra la fuga de datos
que reclamaba la corrección del docente (Capítulo 1).

## 3.6 Desempeño en validación anidada: media ± desviación del bucle externo

```{code-cell} ipython3
# Media ± desviación (ddof=1) del bucle externo: promedio por pliegue sobre las 16 configuraciones
outer = read_csv(R / "outer_metrics.csv")
por_pliegue = outer.groupby(["model", "outer_fold"])[["rmse", "mae", "mse"]].mean()
tabla = por_pliegue.groupby("model").agg(["mean", "std"])          # std de pandas: ddof=1
tabla.columns = [f"{m}_{s}" for m, s in tabla.columns]
tabla = tabla.sort_values("rmse_mean")

# Coincidencia con el archivo oficial del proyecto
oficial = read_csv(R / "outer_fold_mean_sd.csv").set_index("model")
assert np.allclose(tabla.loc[oficial.index, "rmse_mean"], oficial["rmse_mean"])
assert np.allclose(tabla.loc[oficial.index, "rmse_std"], oficial["rmse_sd"])
assert tabla.shape[0] == 32 and outer["outer_fold"].nunique() == 6

top = tabla.head(10)[["rmse_mean", "rmse_std", "mae_mean", "mse_mean"]].round(4)
glue_tbl("tbl-top-externo", top)

fig, ax = plt.subplots(figsize=(8, 8.5))
t = tabla.sort_values("rmse_mean", ascending=False)
y = np.arange(len(t))
ax.errorbar(t["rmse_mean"], y, xerr=t["rmse_std"], fmt="o", ms=4, lw=1, color="#1f4e79", ecolor="#8fb4d9")
ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7)
ax.set_xlabel("RMSE externo (puntos porcentuales de volatilidad), media ± 1 desv. entre 6 pliegues")
ax.set_title("Validación anidada: 32 procedimientos (familia × optimizador)")
fig.tight_layout(); glue_fig("fig-externo", fig)
top
```

```{glue:figure} tbl-top-externo
:name: tbl-top-externo
:figwidth: 90%

Diez mejores procedimientos por RMSE externo (media ± desviación entre 6 pliegues; unidades: puntos porcentuales).
```

```{glue:figure} fig-externo
:name: fig-externo
:figwidth: 80%

RMSE externo (eje x, puntos porcentuales) con barras de ±1 desviación entre pliegues, por procedimiento (eje y).
```

**Interpretación de la {numref}`tbl-top-externo` y la {numref}`fig-externo`.** Por RMSE medio, los mejores son
Lasso con búsqueda genética (0,4920 ± 0,1541) y bayesiana (0,4927 ± 0,1494), seguidos de SVR (≈0,495–0,497).
Pero **la desviación entre pliegues (≈0,13–0,15) es unas 30 veces mayor que la diferencia entre el primero y el
quinto** (≈0,0045): estadísticamente, los líderes son indistinguibles con solo seis pliegues
externos. En el extremo opuesto, el árbol de decisión es claramente el peor (RMSE ≈ 0,65), con una brecha
que sí supera ampliamente la dispersión. Conclusión operativa: el benchmark discrimina **grupos** de modelos
(lineales/SVR/HAR-XG vs. árboles individuales), no posiciones individuales.

## 3.7 Criterio de selección de la mejor combinación por familia

El criterio fijado de antemano en el protocolo es el **MSE externo medio de 2024** (media sobre los
seis pliegues y las 16 configuraciones), sin usar 2025. Dentro de cada familia se elige el
optimizador con menor MSE externo; se informa además cuántas desviaciones separan al ganador del segundo.

```{code-cell} ipython3
# Mejor optimizador por familia según MSE externo de 2024 (criterio predeclarado)
sel = read_csv(R / "explain_best" / "selection_2024_outer_mse.csv")
filas = []
for fam, g in sel.sort_values("outer_mse").groupby("family"):
    g = g.sort_values("outer_mse").reset_index(drop=True)
    filas.append({"familia": fam, "mejor_optimizador": g.loc[0, "method"],
                  "MSE_externo": g.loc[0, "outer_mse"], "desv_externa": g.loc[0, "outer_fold_mse_sd"],
                  "segundo": g.loc[1, "method"],
                  "brecha_vs_segundo_en_desv": (g.loc[1, "outer_mse"] - g.loc[0, "outer_mse"]) / g.loc[0, "outer_fold_mse_sd"]})
mejor = pd.DataFrame(filas).sort_values("MSE_externo").set_index("familia").round(4)
assert mejor.index[0] == "HAR-XG" and mejor.loc["HAR-XG", "mejor_optimizador"] == "bayesian"
assert mejor.loc["SVR", "mejor_optimizador"] == "genetic"
glue_tbl("tbl-mejor", mejor)
mejor
```

```{glue:figure} tbl-mejor
:name: tbl-mejor
:figwidth: 100%

Mejor optimizador de cada familia por MSE externo 2024, su desviación y la brecha con el segundo en unidades de desviación.
```

**Interpretación de la {numref}`tbl-mejor`.** El mejor procedimiento global por MSE externo es
**HAR-XG con búsqueda bayesiana (0,3795)**, seguido de SVR con búsqueda genética (0,3817). La brecha entre el mejor y el
segundo optimizador de cada familia es, en general, una fracción pequeña de la desviación entre
pliegues: **la elección del optimizador cambia poco el resultado** y no justifica, por sí sola, un
costo computacional mayor. Nótese además que el ranking por *MSE medio* (HAR-XG bayesiano primero) y por *RMSE
medio* (Lasso genético primero, {numref}`tbl-top-externo`) no coinciden, porque el promedio de raíces no es
la raíz del promedio: el criterio predeclarado es el MSE.

## 3.8 Estabilidad del ranking entre validación y evaluación (honestidad sobre cambios)

```{code-cell} ipython3
# ¿Se conserva el ranking al pasar de la validación anidada de 2024 a la evaluación de 2025? (32 procedimientos)
mm = read_csv(R / "evaluation" / "macro_metrics.csv")
mse24 = read_csv(R / "outer_fold_mean_sd.csv").set_index("model")["mse_mean"]
mse25 = mm[mm.year == 2025].set_index("model")["mse"]
comun = [m for m in mse24.index if m in mse25.index]
rk = pd.DataFrame({"rank_2024_anidado": mse24[comun].rank(), "rank_2025": mse25[comun].rank()})
rho = stats.spearmanr(rk.rank_2024_anidado, rk.rank_2025)
rho_df = pd.DataFrame({"2024 anidado vs 2025": {"rho_Spearman": rho.statistic, "p": rho.pvalue}}).T.round(4)
lider = pd.Series({"2024 (anidado)": rk.rank_2024_anidado.idxmin(), "2025": rk.rank_2025.idxmin()},
                  name="procedimiento líder por MSE")
top5 = pd.DataFrame({"2024 anidado": rk.rank_2024_anidado.nsmallest(5).index,
                     "2025": rk.rank_2025.nsmallest(5).index})
assert len(comun) == 32 and abs(rho.statistic - .922) < .002
glue_tbl("tbl-rho", rho_df); glue_tbl("tbl-lider", top5)
display(rho_df); top5
```

```{glue:figure} tbl-rho
:name: tbl-rho
:figwidth: 70%

Correlación de rangos de Spearman del MSE de los 32 procedimientos entre la validación anidada 2024 y la evaluación 2025.
```

```{glue:figure} tbl-lider
:name: tbl-lider
:figwidth: 80%

Cinco mejores procedimientos por MSE en la validación anidada de 2024 y en 2025.
```

**Interpretación de las {numref}`tbl-rho` y {numref}`tbl-lider`.** El ordenamiento global es muy estable (ρ de Spearman = 0,92 entre la validación
anidada de 2024 y 2025), pero **el líder cambia**: HAR-XG con búsqueda bayesiana encabeza 2024 y HAR-XG con *grid* encabeza 2025 (ambos de la misma
familia híbrida). Hay un grupo de cabeza estable y un puesto 1 inestable entre optimizadores. Se reporta así, sin reescribir el ranking a posteriori:
el cambio de líder es evidencia de que las diferencias entre los mejores no son robustas, lo que se confirma con las pruebas del Capítulo 6.

## 3.9 Reentrenamiento, serialización y prueba de ida y vuelta

**Protocolo de reentrenamiento.** Tras la selección, cada uno de los 512 pipelines se reajusta con los
660 orígenes de 2023–2024 (etiquetas que terminan antes de 2025) y queda **congelado** durante la
evaluación de 2025. El manifiesto `artifacts/model_manifest.csv` registra, por modelo, semilla,
hiperparámetros, MSE interno y SHA-256 de la serialización original; `artifacts/metadata.json` consolida
librerías, semilla global, esquema de entrada y métricas del bucle externo (se regenera con
`python src/build_artifacts_metadata.py`).

```{code-cell} ipython3
# Verificación de artifacts/: 512 modelos, semilla 42 y hashes únicos por serialización
art = REPO / "artifacts"
meta = json.loads((art / "metadata.json").read_text(encoding="utf-8"))
man = pd.read_csv(art / "model_manifest.csv")
assert meta["modelos_finales"] == 512 == len(man) and meta["semilla_global"] == SEED
assert man["model_sha256"].str.len().eq(64).all()
print("Librerías registradas:", meta["librerias"])
print("Hashes SHA-256 distintos:", man["model_sha256"].nunique(), "de", len(man),
      "(los pipelines idénticos entre optimizadores comparten hash)")
```

Que haya menos hashes distintos que modelos es esperable: cuando dos optimizadores eligen los mismos
hiperparámetros, el pipeline serializado es byte a byte idéntico.

```{code-cell} ipython3
# Prueba de ida y vuelta (round-trip): cargar -> predecir -> serializar -> recargar -> predecir
# Se usa un pipeline Ridge ya serializado en el repositorio (no requiere reentrenar).
ruta = sorted((REPO / "results" / "optimized_minute_ridge_2023_2025" / "models").glob("BTCUSDT_v*.joblib"))[0]
obj = joblib.load(ruta)
pipe = obj["model"]
rng = np.random.default_rng(SEED)
X = rng.normal(size=(5, pipe.n_features_in_))
pred_original = pipe.predict(X)
with tempfile.TemporaryDirectory() as tmp:
    destino = Path(tmp) / "roundtrip.joblib"
    joblib.dump(obj, destino)
    pred_recargado = joblib.load(destino)["model"].predict(X)
assert np.array_equal(pred_original, pred_recargado), "La ida y vuelta alteró las predicciones"
print(f"{ruta.name}: {pipe.n_features_in_} entradas, {pred_original.shape[1]} salidas; "
      f"diferencia máxima tras ida y vuelta = {np.abs(pred_original - pred_recargado).max():.1e}")
```

La prueba de ida y vuelta confirma que serializar y recargar el pipeline deja las predicciones
**idénticas bit a bit** (diferencia máxima 0). Limitación honesta: los `.joblib` de los 512 pipelines
finales no están versionados por peso, así que la prueba se ejecuta sobre un pipeline equivalente que sí lo está;
el SHA-256 de los originales permite verificarlos cuando se dispone de ellos.
