# Capítulo 5 · Diseño e implementación del modelo propuesto

```{admonition} Estado respecto de R2 (implementación algorítmica propia)
:class: warning
La rúbrica exige que el modelo **no se obtenga con una línea `from libreria import Modelo`** y que exista un **proceso de implementación
algorítmica programado por el estudiante**. Veredicto: **cumplimiento parcial**.

- **Programado por los autores** (`src/improve_classical_forecast.py`): la construcción de las entradas HAR (`har_features`), el objetivo
  de corrección relativa, la búsqueda del peso de mezcla y de la penalización (`choose_blend`) y el pronóstico combinado (`predict_bundle`).
- **Importado, no programado:** los aprendices base, `Ridge` de `scikit-learn` y `xgboost`. Por eso **no** se cumple la formulación más estricta del
  Capítulo 5 (solo `numpy`/`scipy`, clase propia con `BaseEstimator` y pruebas `pytest` de esa clase).

No se usan redes neuronales ni *deep learning*. Este capítulo documenta lo que existe y declara lo que falta (§5.6).
```

```{code-cell} ipython3
:tags: [remove-cell]
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from jbook_utils import R, PF, read_csv, glue_fig, glue_tbl
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
```

## 5.1 Justificación teórica y formulación

La volatilidad realizada es persistente y se explica bien por componentes de memoria a **varias escalas**
(día, semana, mes); esa es la idea del modelo HAR {cite}`corsi2009har`. Los árboles potenciados capturan, en cambio,
no linealidades y asimetrías (por ejemplo, el efecto de rendimientos negativos) que un modelo lineal no recoge
{cite}`chen2016xgboost`. La propuesta combina ambos: un componente lineal regularizado y uno no lineal.

Sea $\mathbf{x}_t$ el vector de entradas HAR en el origen $t$ (resúmenes de volatilidad de 1, 3, 7, 14 y 28 días,
medida de riesgo a la baja y los retornos conocidos que abandonan la ventana) y $b_t$ la volatilidad actual de
referencia. El modelo predice la **corrección relativa** $r_{t+h}=\sigma_{t+h}/b_t-1$ para cada horizonte $h=1,\dots,7$:

```{math}
:label: eq-ridge
\hat{\boldsymbol\beta}_h=\argmin_{\boldsymbol\beta}\ \sum_{t}\left(r_{t+h}-\mathbf{x}_t^{\top}\boldsymbol\beta\right)^2+\alpha\lVert\boldsymbol\beta\rVert_2^2
\;=\;(\mathbf{X}^{\top}\mathbf{X}+\alpha\mathbf{I})^{-1}\mathbf{X}^{\top}\mathbf{r}_h ,
```

```{math}
:label: eq-blend
\hat\sigma_{t+h}=b_t\Big(1+w\,\mathbf{x}_t^{\top}\hat{\boldsymbol\beta}_h+(1-w)\,\hat g_h(\mathbf{x}_t)\Big),\qquad w\in[0,1],
```

donde $\hat g_h$ es el ensamble de árboles de XGBoost (pérdida cuadrática, profundidad pequeña) y $w$ el peso del
componente Ridge, elegido en validación temporal junto con $\alpha$. Con $w=0$ el modelo se reduce a XGBoost, de modo que
la combinación **no puede ser peor que su componente no lineal en validación**.

## 5.2 Pseudocódigo

```{code-block} text
:caption: Algoritmo HAR-Ridge + XGBoost (entrenamiento y pronóstico)
Entrada : series de cierre diario; ventana w_v ∈ {7,14,21,28}; horizontes h = 1..7;
          rejilla A de penalizaciones α; rejilla W de pesos w; semilla s = 42
Salida  : pronósticos σ̂_{t+h} para h = 1..7

1.  Construir x_t (HAR) y b_t; objetivos r_{t+h} = σ_{t+h}/b_t − 1
2.  Definir bloques de validación expansivos y purgados (etiqueta máx. 7 días antes del origen de validación)
3.  Para cada (α, w) ∈ A × W:
4.      Para cada bloque: ajustar escalador SOLO con el entrenamiento; ajustar β_h (eq. Ridge)
        y g_h (XGBoost, semilla s); evaluar MSE en unidades naturales
5.  (α*, w*) ← argmin del MSE medio de validación (w = 0 permite conservar XGBoost)
6.  Reajustar con todos los orígenes de 2023–2024 (etiquetas < 2025); congelar
7.  Pronóstico: σ̂_{t+h} = b_t (1 + w* xᵀβ_h + (1−w*) g_h(x))
Criterios de parada: Ridge es de forma cerrada (sin iteración); XGBoost usa un número fijo de árboles
(n_estimators ∈ {64,128}); las búsquedas de hiperparámetros se detienen tras 8 evaluaciones únicas.
```

## 5.3 Complejidad algorítmica

Con $n$ orígenes de entrenamiento, $p$ entradas y $T$ árboles de profundidad $d$:

| Etapa | Entrenamiento | Inferencia (por pronóstico) |
|---|---|---|
| Ridge (forma cerrada) | $O(np^{2}+p^{3})$ | $O(p)$ |
| XGBoost (histogramas) | $O\big(T\,n\,p\big)$ aprox. por nivel de árbol | $O(T\,d)$ |
| HAR-Ridge + XGBoost | suma de ambos, por horizonte ($\times 7$) | $O(p+T\,d)$ por horizonte |

```{code-cell} ipython3
# Tiempo empírico de entrenamiento e inferencia al variar n (200, 400, 600 orígenes), 1 hilo, 3 repeticiones
ef = read_csv(R / "efficiency" / "standard_vs_optimized_summary.csv")
sel = ef[ef.model.isin(["Ridge", "XGBoost", "HAR-XG", "RandomForest"])]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
for m, g in sel.groupby("model"):
    axes[0].plot(g.train_rows, g.fit_wall_seconds_standard_mean, marker="o", label=m)
    axes[1].plot(g.train_rows, g.predict_wall_seconds_standard_mean * 1e3, marker="o", label=m)
axes[0].set_xlabel("Orígenes de entrenamiento n"); axes[0].set_ylabel("Tiempo de ajuste (s)")
axes[1].set_xlabel("Orígenes de entrenamiento n"); axes[1].set_ylabel("Tiempo de inferencia, 32 pronósticos (ms)")
axes[0].legend(frameon=False, fontsize=8); fig.suptitle("Costo computacional empírico (220 entradas, 7 salidas)", y=1.02)
fig.tight_layout(); glue_fig("fig-costo", fig)

tc = (ef[ef.model.isin(["Ridge", "XGBoost", "HAR-XG"])]
      .pivot(index="train_rows", columns="model", values="fit_wall_seconds_standard_mean").round(4))
assert tc.shape == (3, 3) and (tc["HAR-XG"] > tc["XGBoost"]).all() and (tc["XGBoost"] > tc["Ridge"]).all()
glue_tbl("tbl-costo", tc)
tc
```

```{glue:figure} tbl-costo
:name: tbl-costo
:figwidth: 70%

Tiempo de ajuste (s) por número de orígenes de entrenamiento.
```

```{glue:figure} fig-costo
:name: fig-costo
:figwidth: 95%

Tiempo de ajuste (izq.) y de inferencia (der.) frente al número de orígenes de entrenamiento.
```

**Interpretación de la {numref}`tbl-costo` y la {numref}`fig-costo`.** El orden de costo es el esperado,
Ridge < XGBoost < HAR-XG (la combinación suma ambos componentes y es la más cara, 0,49–0,66 s por ajuste frente a
0,27–0,37 s de XGBoost y ≈0,01 s de Ridge), y la inferencia es del orden de milisegundos. Con solo tres tamaños (200–600) y
$p$ fijo en 220, **no se puede estimar empíricamente el exponente asintótico**: en estos tamaños dominan costos fijos y el
tiempo crece menos que lo teórico. La variación de $p$ no se midió, por lo que la contrastación completa de la
complejidad $O(\cdot)$ con $n$ y $p$ queda como trabajo pendiente.

## 5.4 Ablación de componentes

```{code-cell} ipython3
# Ablación: componente no lineal solo (XGBoost), componente lineal solo (HAR diario OLS) y combinación
m = read_csv(PF / "macro_metrics.csv")
nombres = {"XGBoost": "Solo XGBoost (w=0)", "HAR diario OLS (1/7/28)": "Solo HAR lineal (OLS, 3 escalas)",
           "HAR-Ridge + XGBoost": "HAR-Ridge + XGBoost", "Persistencia": "Persistencia"}
ab = (m[m.model.isin(nombres)].pivot(index="model", columns="year", values="mse").rename(index=nombres).round(4))
ab["mejora_vs_XGBoost_2025_%"] = (100 * (ab[2025] / ab.loc["Solo XGBoost (w=0)", 2025] - 1)).round(1)
ab["mejora_vs_XGBoost_2026_%"] = (100 * (ab[2026] / ab.loc["Solo XGBoost (w=0)", 2026] - 1)).round(1)
assert ab.loc["HAR-Ridge + XGBoost", 2025] < ab.loc["Solo XGBoost (w=0)", 2025] < ab.loc["Solo HAR lineal (OLS, 3 escalas)", 2025]
glue_tbl("tbl-ablacion", ab)
ab
```

```{glue:figure} tbl-ablacion
:name: tbl-ablacion
:figwidth: 100%

MSE 2025–2026 de cada componente y de la combinación; las dos últimas columnas dan el cambio porcentual frente a XGBoost solo.
```

**Interpretación de la {numref}`tbl-ablacion`.** Cada componente aporta: el HAR lineal solo (0,6247 en 2025) y XGBoost solo
(0,4965) quedan por detrás de la combinación (0,4647), que mejora a XGBoost un 6,4 % en 2025 y un 3,4 % en 2026. Es una
ablación de dos componentes con evaluación retrospectiva; no aísla por separado el aporte de la corrección relativa ni de la
ponderación por escala, que se tratan como decisiones de diseño fijadas antes de ver 2025.

## 5.5 Interpretabilidad con SHAP

```{code-cell} ipython3
# Importancia global |SHAP| media (valores de Shapley, 26.432 pronósticos de 2026)
g = read_csv(R / "explain_global" / "global_importance.csv").sort_values("mean_absolute_shap", ascending=False)
top = g.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top.feature.str.replace("HAR::", ""), top.mean_absolute_shap, color="#1f4e79")
ax.set_xlabel("Importancia global: media de |SHAP| (puntos porcentuales de volatilidad)")
ax.set_title("Diez entradas más influyentes (HAR-XG bayesiano, 2026)"); fig.tight_layout()
glue_fig("fig-shap", fig)
share = g.mean_absolute_shap / g.mean_absolute_shap.sum()
assert g.iloc[0].feature == "current_volatility" and .55 < share.iloc[0] < .57
print(f"La volatilidad actual concentra {100 * share.iloc[0]:.1f} % de la importancia total ({len(g)} entradas).")
```

```{glue:figure} fig-shap
:name: fig-shap
:figwidth: 90%

Importancia global SHAP (eje x: media de |SHAP|, puntos porcentuales) de las diez entradas principales.
```

**Interpretación de la {numref}`fig-shap`.** La volatilidad actual domina (|SHAP| medio 0,91, ≈56 % de la importancia total
entre las 220 entradas), seguida a gran distancia por el logaritmo de la volatilidad actual (0,05) y por promedios de 28 días de
retornos absolutos y a la baja (≈0,03). El modelo, por tanto, es en esencia **persistencia corregida por memoria de largo plazo**, lo
que explica por qué las ganancias sobre ella son sustanciales pero las diferencias entre variantes, pequeñas. Los valores SHAP
{cite}`lundberg2017shap` son aproximados (enmascaramiento fuera de la variedad de datos) y **no son causales**; la interpretación local con
LIME {cite}`ribeiro2016lime` se calculó para un caso representativo (BTC, ventana 21, horizonte 1) en el informe de explicabilidad
del repositorio.

## 5.6 Pruebas unitarias y lo que falta para R2

El repositorio incluye 28 archivos de pruebas (`tests/`, `unittest`) con 116 funciones de prueba que verifican fronteras
temporales, cálculos de métricas, DM, calendario, rutas del dashboard y consistencia de resultados, y la revisión final ejecutó 71
pruebas sin fallos ({doc}`cap01_trazabilidad`). **No existen** pruebas de aprendizaje sobre datos sintéticos, de consistencia ante
semillas, de entradas inválidas ni de `sklearn.base.clone` para una clase propia, porque esa clase no existe.

Para cerrar R2 se requeriría: (i) una clase `BaseEstimator`/`RegressorMixin` en `numpy`/`scipy` (por ejemplo,
Ridge en forma cerrada + *boosting* propio), (ii) su comparación pareada bajo las mismas particiones, y (iii) la revisión de
novedad del Capítulo 4. Eso no existe hoy en el repositorio; por eso R2 se declara solo como **parcial**.
