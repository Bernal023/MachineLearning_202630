# Capítulo 3 · Modelos benchmark y mejores combinaciones

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
from jbook_utils import R, REPO, SEED, read_csv, glue_fig, glue_tbl
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
```

## 3.1 Espacio de modelos evaluado

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

## 3.2 Protocolo de validación anidada con purga

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

## 3.3 Desempeño en validación anidada: media ± desviación del bucle externo

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

## 3.4 Criterio de selección de la mejor combinación por familia

El criterio fijado de antemano en el protocolo es el **MSE externo medio de 2024** (media sobre los
seis pliegues y las 16 configuraciones), sin usar 2025 ni 2026. Dentro de cada familia se elige el
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

## 3.5 Estabilidad del ranking entre periodos (honestidad sobre cambios)

```{code-cell} ipython3
# ¿Se conserva el ranking al pasar de validación 2024 a 2025 y 2026? (32 procedimientos)
mm = read_csv(R / "evaluation" / "macro_metrics.csv")
mse24 = read_csv(R / "outer_fold_mean_sd.csv").set_index("model")["mse_mean"]
mse25 = mm[mm.year == 2025].set_index("model")["mse"]
mse26 = mm[mm.year == 2026].set_index("model")["mse"]
comun = [m for m in mse24.index if m in mse25.index and m in mse26.index]
rk = pd.DataFrame({"rank_2024_anidado": mse24[comun].rank(), "rank_2025": mse25[comun].rank(),
                   "rank_2026": mse26[comun].rank()})
rho = {"2024 vs 2025": stats.spearmanr(rk.rank_2024_anidado, rk.rank_2025),
       "2024 vs 2026": stats.spearmanr(rk.rank_2024_anidado, rk.rank_2026),
       "2025 vs 2026": stats.spearmanr(rk.rank_2025, rk.rank_2026)}
rho_df = pd.DataFrame({k: {"rho_Spearman": v.statistic, "p": v.pvalue} for k, v in rho.items()}).T.round(3)
lider = pd.Series({"2024 (anidado)": rk.rank_2024_anidado.idxmin(), "2025": rk.rank_2025.idxmin(),
                   "2026": rk.rank_2026.idxmin()}, name="procedimiento líder por MSE")
assert len(comun) == 32
assert abs(rho["2024 vs 2025"].statistic - .922) < .002 and abs(rho["2025 vs 2026"].statistic - .745) < .002
glue_tbl("tbl-rho", rho_df); glue_tbl("tbl-lider", lider.to_frame())
display(rho_df); lider.to_frame()
```

```{glue:figure} tbl-rho
:name: tbl-rho
:figwidth: 70%

Correlación de rangos de Spearman del MSE de los 32 procedimientos entre periodos.
```

```{glue:figure} tbl-lider
:name: tbl-lider
:figwidth: 80%

Procedimiento líder por MSE en cada periodo de evaluación.
```

**Interpretación de las {numref}`tbl-rho` y {numref}`tbl-lider`.** El ordenamiento global es
estable (ρ = 0,92 entre la validación anidada de 2024 y 2025; 0,79 con 2026; 0,75 entre 2025 y 2026), pero **el líder cambia
en cada periodo**: HAR-XG bayesiano en 2024, HAR-XG con *grid* en 2025 y Lasso genético en 2026. Es decir, hay un
grupo de cabeza estable y un puesto 1 inestable. Se reporta así, sin reescribir el ranking a posteriori: el
cambio de líder es evidencia de que las diferencias entre los mejores no son robustas, lo que se confirma
con las pruebas del Capítulo 6.

## 3.6 Reentrenamiento, serialización y prueba de ida y vuelta

**Protocolo de reentrenamiento.** Tras la selección, cada uno de los 512 pipelines se reajusta con los
660 orígenes de 2023–2024 (etiquetas que terminan antes de 2025) y queda **congelado** durante la
evaluación de 2025 y 2026. El manifiesto `artifacts/model_manifest.csv` registra, por modelo, semilla,
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
