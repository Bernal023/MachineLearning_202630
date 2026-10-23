# Capítulo 5 · Diseño e implementación del modelo propuesto

```{admonition} Estado respecto de R2 (implementación algorítmica propia)
:class: warning
La rúbrica exige que el modelo **no se obtenga con una línea `from libreria import Modelo`** y que exista un **proceso de implementación
algorítmica programado por el estudiante**. Veredicto: **cumplimiento parcial**.

- **Programado por los autores** (`src/improve_classical_forecast.py`): la construcción de las entradas HAR (`har_features`), el objetivo
  de corrección relativa, la búsqueda del peso de mezcla y de la penalización (`choose_blend`) y el pronóstico combinado (`predict_bundle`).
- **Importado, no programado:** los aprendices base, `Ridge` de `scikit-learn` y `xgboost`. Por eso **no** se cumple la formulación más estricta del
  Capítulo 5 (solo `numpy`/`scipy`, clase propia con `BaseEstimator` y pruebas `pytest` de esa clase).

No se usan redes neuronales ni *deep learning*. Este capítulo documenta lo que existe y declara lo que falta (§5.7).
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

## 5.1 Componentes y combinación

La volatilidad es persistente y se explica bien por memoria a **varias escalas** (día, semana, mes), idea del modelo HAR {cite}`corsi2009har`; los árboles potenciados
capturan relaciones no lineales {cite}`chen2016xgboost`. HAR-Ridge + XGBoost reúne ambos:

- **Componente HAR-Ridge.** Resúmenes de 1, 3, 7, 14 y 28 días de retornos, variación intradía y riesgo negativo. Representa escalas temporales y regulariza los coeficientes; el escalador se
  ajusta solo con el entrenamiento.
- **Componente XGBoost.** Árboles potenciados sobre características históricas: relaciones no lineales e interacciones que complementan la estructura temporal. Conserva la selección de entrada
  y los hiperparámetros elegidos con 2024.

La predicción combina los dos pronósticos de volatilidad con un peso $\lambda$:

```{math}
:label: eq-blend
\hat\sigma_{t+h}=\lambda\,\hat\sigma^{\text{HAR-Ridge}}_{t+h}+(1-\lambda)\,\hat\sigma^{\text{XGBoost}}_{t+h},\qquad h=1,\dots,7 .
```

En el código, cada componente predice una **corrección relativa** sobre la volatilidad actual $b_t$, es decir,
$\hat\sigma=\max\{b_t(1+\hat r),0\}$, y un único peso se comparte entre horizontes. Se exploran **cinco penalizaciones Ridge** $\alpha$ y **cinco pesos**
$\lambda\in\{0;\,0{,}25;\,0{,}5;\,0{,}75;\,1\}$, y se elige la pareja de menor RMSE de validación en los seis bloques de 2024 (ante empate, el menor $\lambda$ y luego el menor $\alpha$).
Con $\lambda=0$ se conserva XGBoost sin cambios, de modo que la combinación **no puede ser peor que su componente no lineal en validación**. Como combina dos pronósticos con errores
potencialmente distintos, los pesos se seleccionan **antes de 2025**.

## 5.2 De HAR a HAR-Ridge

**El modelo HAR.** Expresa la volatilidad futura con una regresión lineal con componentes de tres escalas:

```{math}
:label: eq-har
RV_{t+1}=\beta_0+\beta_d\,RV^{(d)}_t+\beta_w\,RV^{(w)}_t+\beta_m\,RV^{(m)}_t+u_{t+1}.
```

$RV_{t+1}$ es la variable respuesta; $RV^{(d)}_t$ (diario) recoge información de corto plazo, $RV^{(w)}_t$ (semanal) de escala intermedia y $RV^{(m)}_t$ (mensual) la de mayor persistencia.
Es un esquema ilustrativo de tres escalas: el estudio amplía las entradas a resúmenes de 1, 3, 7, 14 y 28 días y otras características históricas.

**Ridge restringe el tamaño de los coeficientes del HAR:**

```{math}
:label: eq-ridge-restr
\min_{\beta_0,\beta_d,\beta_w,\beta_m}\ \sum_{t=1}^{n}\Big[RV_{t+1}-\beta_0-\beta_dRV^{(d)}_t-\beta_wRV^{(w)}_t-\beta_mRV^{(m)}_t\Big]^2
\quad\text{sujeto a}\quad \beta_d^2+\beta_w^2+\beta_m^2\le c .
```

Con el multiplicador $\alpha\ge0$, el Lagrangiano es

```{math}
:label: eq-lagrangiano
\mathcal L(\boldsymbol\beta,\alpha)=\sum_{t=1}^{n}\Big[RV_{t+1}-\beta_0-\beta_dRV^{(d)}_t-\beta_wRV^{(w)}_t-\beta_mRV^{(m)}_t\Big]^2+\alpha\big(\beta_d^2+\beta_w^2+\beta_m^2-c\big).
```

Para $\alpha$ fijo, $-\alpha c$ es constante respecto de $\boldsymbol\beta$: minimizar equivale a **mínimos cuadrados más penalización L2**. El intercepto $\beta_0$ no se penaliza;
$\alpha$ es la penalización Ridge y $\lambda$ se reserva para el peso de combinación con XGBoost.

**Forma matricial y derivada.** Con $\mathbf y=(RV_2,\dots,RV_{n+1})^\top$, $\boldsymbol\beta=(\beta_0,\beta_d,\beta_w,\beta_m)^\top$ y $\mathbf P=\operatorname{diag}(0,1,1,1)$ (que excluye el intercepto de la penalización),
omitiendo la constante $-\alpha c$:

```{math}
:label: eq-ridge-matriz
\mathcal L(\boldsymbol\beta)=(\mathbf y-\mathbf X\boldsymbol\beta)^\top(\mathbf y-\mathbf X\boldsymbol\beta)+\alpha\,\boldsymbol\beta^\top\mathbf P\boldsymbol\beta,
\qquad
\nabla_{\boldsymbol\beta}\mathcal L=-2\mathbf X^\top\mathbf y+2\mathbf X^\top\mathbf X\boldsymbol\beta+2\alpha\mathbf P\boldsymbol\beta .
```

Igualando a cero se obtiene el sistema lineal regularizado y su solución:

```{math}
:label: eq-ridge-sol
(\mathbf X^\top\mathbf X+\alpha\mathbf P)\,\hat{\boldsymbol\beta}=\mathbf X^\top\mathbf y
\quad\Longrightarrow\quad
\hat{\boldsymbol\beta}_{\text{HAR-Ridge}}=(\mathbf X^\top\mathbf X+\alpha\mathbf P)^{-1}\mathbf X^\top\mathbf y,
\qquad
\mathbf X=\begin{pmatrix}1&RV^{(d)}_1&RV^{(w)}_1&RV^{(m)}_1\\\vdots&\vdots&\vdots&\vdots\\1&RV^{(d)}_n&RV^{(w)}_n&RV^{(m)}_n\end{pmatrix}.
```

Para $\alpha>0$ la matriz es invertible; en el cálculo se **resuelve el sistema lineal sin formar la inversa**. Ridge obtiene conjuntamente el intercepto y los coeficientes diario, semanal y mensual, y el pronóstico
del componente temporal es

```{math}
:label: eq-har-ridge-pred
\widehat{RV}^{\text{HAR-Ridge}}_{t+1}=\hat\beta_0+\hat\beta_d\,RV^{(d)}_t+\hat\beta_w\,RV^{(w)}_t+\hat\beta_m\,RV^{(m)}_t .
```

Ese pronóstico se combina con el de los árboles mediante $\lambda$ ({eq}`eq-blend`).

```{code-cell} ipython3
# Verificación numérica de la solución cerrada (eq-ridge-sol) frente a scikit-learn, con datos reales de BTC
from sklearn.linear_model import Ridge
from jbook_utils import load_daily_panel
b = load_daily_panel("2023-01-01", "2024-12-24").query("symbol == 'BTCUSDT'").set_index("date")
rv = b["ret"] ** 2                                          # varianza diaria realizada (proxy)
X_df = pd.DataFrame({"d": rv, "w": rv.rolling(7).mean(), "m": rv.rolling(28).mean()})   # escalas 1, 7 y 28 días
y_s = rv.shift(-1)
ok = X_df.notna().all(axis=1) & y_s.notna()
Xm = np.column_stack([np.ones(ok.sum()), X_df[ok].values]); yv = y_s[ok].values
P = np.diag([0.0, 1.0, 1.0, 1.0]); alpha = 5.0
beta = np.linalg.solve(Xm.T @ Xm + alpha * P, Xm.T @ yv)     # sin formar la inversa
sk = Ridge(alpha=alpha, fit_intercept=True).fit(X_df[ok].values, yv)
assert np.allclose(beta[1:], sk.coef_, rtol=1e-6) and np.isclose(beta[0], sk.intercept_, rtol=1e-6)
tabla_beta = pd.DataFrame({"solución cerrada": beta, "scikit-learn": np.r_[sk.intercept_, sk.coef_]},
                          index=["β0", "βd", "βw", "βm"]).round(5)
glue_tbl("tbl-beta", tabla_beta)
tabla_beta
```

```{glue:figure} tbl-beta
:name: tbl-beta
:figwidth: 60%

Coeficientes HAR-Ridge ($\alpha=5$) de BTC con la solución cerrada y con `Ridge` de scikit-learn (datos 2023–2024; $n=$ muestras completas).
```

La {numref}`tbl-beta` confirma que resolver $(\mathbf X^\top\mathbf X+\alpha\mathbf P)\hat{\boldsymbol\beta}=\mathbf X^\top\mathbf y$ reproduce los coeficientes de `Ridge`, con el intercepto sin penalizar. *(Ejemplo ilustrativo
con tres escalas, 1, 7 y 28 días, sobre el cuadrado del retorno; no son los coeficientes del modelo final, que usa 44 entradas estandarizadas.)*

## 5.3 Pseudocódigo

```{code-block} text
:caption: Algoritmo HAR-Ridge + XGBoost (entrenamiento y pronóstico)
Entrada : series de cierre diario; ventana w_v ∈ {7,14,21,28}; horizontes h = 1..7;
          rejilla A de penalizaciones α (5 valores); rejilla Λ = {0; 0,25; 0,5; 0,75; 1}; semilla s = 42
Salida  : pronósticos σ̂_{t+h} para h = 1..7

1.  Construir x_t (HAR) y la volatilidad actual b_t; objetivos r_{t+h} = σ_{t+h}/b_t − 1
2.  Definir seis bloques de validación expansivos y separados (etiqueta máx. 21 días antes de la validación)
3.  Ajustar XGBoost (semilla s) con los hiperparámetros elegidos con 2024
4.  Para cada α ∈ A: ajustar β(α) resolviendo (XᵀX + αP) β = Xᵀr con el escalador ajustado solo en entrenamiento
5.  Para cada (α, λ) ∈ A × Λ: evaluar el RMSE de validación de λ·σ̂_Ridge + (1−λ)·σ̂_XGB
6.  (α*, λ*) ← mínimo RMSE (desempate: menor λ, luego menor α); λ = 0 conserva XGBoost
7.  Reajustar con todos los orígenes de 2023–2024 (660); congelar
8.  Pronóstico: σ̂_{t+h} = λ*·σ̂^{Ridge}_{t+h} + (1−λ*)·σ̂^{XGB}_{t+h}   (eq-blend)
Criterios de parada: Ridge es de forma cerrada (sin iteración); XGBoost usa un número fijo de árboles;
las búsquedas de hiperparámetros se detienen tras 8 evaluaciones únicas.
```

## 5.4 Complejidad algorítmica

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

## 5.5 Ablación de componentes

```{code-cell} ipython3
# Ablación: componente no lineal solo (XGBoost), componente lineal solo (HAR diario OLS) y combinación
m = read_csv(PF / "macro_metrics.csv")
nombres = {"XGBoost": "Solo XGBoost (w=0)", "HAR diario OLS (1/7/28)": "Solo HAR lineal (OLS, 3 escalas)",
           "HAR-Ridge + XGBoost": "HAR-Ridge + XGBoost", "Persistencia": "Persistencia"}
ab = (m[m.model.isin(nombres)].pivot(index="model", columns="year", values="mse").rename(index=nombres).round(4))
ab = ab[[2025]].rename(columns={2025: "MSE_2025"})
ab["cambio_vs_XGBoost_%"] = (100 * (ab["MSE_2025"] / ab.loc["Solo XGBoost (w=0)", "MSE_2025"] - 1)).round(1)
assert ab.loc["HAR-Ridge + XGBoost", "MSE_2025"] < ab.loc["Solo XGBoost (w=0)", "MSE_2025"] < ab.loc["Solo HAR lineal (OLS, 3 escalas)", "MSE_2025"]
glue_tbl("tbl-ablacion", ab)
ab
```

```{glue:figure} tbl-ablacion
:name: tbl-ablacion
:figwidth: 100%

MSE 2025 de cada componente y de la combinación; la última columna es el cambio porcentual frente a XGBoost solo.
```

**Interpretación de la {numref}`tbl-ablacion`.** Cada componente aporta: el HAR lineal solo (0,6247 en 2025) y XGBoost solo
(0,4965) quedan por detrás de la combinación (0,4647), que mejora a XGBoost un 6,4 % en 2025. Es una
ablación de dos componentes con evaluación retrospectiva de un solo año; no aísla por separado el aporte de la corrección relativa ni de la
ponderación por escala, que se tratan como decisiones de diseño fijadas antes de ver 2025.

## 5.6 Interpretabilidad con SHAP

```{admonition} Nota sobre el periodo
:class: note
Las explicaciones SHAP disponibles en el repositorio se calcularon sobre pronósticos de 2026. Se muestran **solo para describir el mecanismo del modelo**; no
entran en ninguna conclusión de desempeño, que se basa solo en 2025.
```

```{code-cell} ipython3
# Importancia global |SHAP| media (valores de Shapley, 26.432 pronósticos de 2026)
g = read_csv(R / "explain_global" / "global_importance.csv").sort_values("mean_absolute_shap", ascending=False)
top = g.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top.feature.str.replace("HAR::", ""), top.mean_absolute_shap, color="#1f4e79")
ax.set_xlabel("Importancia global: media de |SHAP| (puntos porcentuales de volatilidad)")
ax.set_title("Diez entradas más influyentes (explicación sobre pronósticos de 2026)"); fig.tight_layout()
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
retornos absolutos y a la baja (≈0,03). El modelo, por tanto, descansa sobre todo en la persistencia de la volatilidad reciente, con correcciones menores de otras escalas. Los valores SHAP
{cite}`lundberg2017shap` son aproximados (enmascaramiento fuera de la variedad de datos) y **no son causales**; la interpretación local con
LIME {cite}`ribeiro2016lime` se calculó para un caso representativo (BTC, ventana 21, horizonte 1) en el informe de explicabilidad
del repositorio.

## 5.7 Pruebas unitarias y lo que falta para R2

El repositorio incluye 28 archivos de pruebas (`tests/`, `unittest`) con 116 funciones de prueba que verifican fronteras
temporales, cálculos de métricas, DM, calendario, rutas del dashboard y consistencia de resultados, y la revisión final ejecutó 71
pruebas sin fallos ({doc}`cap01_trazabilidad`). **No existen** pruebas de aprendizaje sobre datos sintéticos, de consistencia ante
semillas, de entradas inválidas ni de `sklearn.base.clone` para una clase propia, porque esa clase no existe.

Para cerrar R2 se requeriría: (i) una clase `BaseEstimator`/`RegressorMixin` en `numpy`/`scipy` (por ejemplo,
Ridge en forma cerrada + *boosting* propio), (ii) su comparación pareada bajo las mismas particiones, y (iii) la revisión de
novedad del Capítulo 4. Eso no existe hoy en el repositorio; por eso R2 se declara solo como **parcial**.
