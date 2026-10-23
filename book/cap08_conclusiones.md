# Capítulo 8 · Conclusiones, limitaciones e integridad

```{code-cell} ipython3
:tags: [remove-cell]
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from IPython.display import display
from jbook_utils import R, REPO, HAR, SEARCHES, read_csv, inference_table, glue_tbl
```

## 8.1 Síntesis de hallazgos

Resultados de **2025, $w=21$**, cuatro activos, siete horizontes y 358 orígenes comunes; son los de la presentación y se leen de los artefactos del repositorio.

```{code-cell} ipython3
# Cifras que sustentan las conclusiones (no se copian a mano)
met = read_csv(REPO / "results" / "professor_feedback_20261018" / "metrics_by_window.csv")
met = met[(met.year == 2025) & (met.volatility_window.astype(str) == "21")].set_index("model")
macro = read_csv(R / "evaluation" / "macro_by_window.csv")
macro = macro[(macro.year == 2025) & (macro.volatility_window == 21)].set_index("model")
lasso_opt = macro.loc["Lasso [bayesian]"]
W21 = {s: inference_table(21, s) for s in SEARCHES}

hallazgos = pd.DataFrame({
    "MSE": [met.loc[HAR, "mse"], met.loc["Persistencia", "mse"], lasso_opt["mse"]],
    "RMSE": [met.loc[HAR, "rmse"], met.loc["Persistencia", "rmse"], lasso_opt["rmse"]],
    "MAE": [met.loc[HAR, "mae"], met.loc["Persistencia", "mae"], lasso_opt["mae"]],
    "R²": [met.loc[HAR, "r2"], met.loc["Persistencia", "r2"], lasso_opt["r2"]],
}, index=["HAR-Ridge + XGBoost (original)", "Persistencia", "Lasso con Optuna TPE"]).round(5)
assert abs(hallazgos.loc["HAR-Ridge + XGBoost (original)", "RMSE"] - .38601) < 1e-5 and abs(hallazgos.loc["Lasso con Optuna TPE", "R²"] - .85) < 1e-4

respaldo = pd.DataFrame({"DM-Holm7 (de 7)": {SEARCHES[s]: int(W21[s].rechaza_H0.sum()) for s in SEARCHES},
                         "IC positivos (de 7)": {SEARCHES[s]: int(W21[s].IC_positivo.sum()) for s in SEARCHES}})
assert respaldo["DM-Holm7 (de 7)"].tolist() == [7, 3, 2, 4] and respaldo["IC positivos (de 7)"].tolist() == [7, 3, 3, 4]
glue_tbl("tbl-hallazgos", hallazgos); glue_tbl("tbl-respaldo", respaldo)
display(hallazgos); respaldo
```

```{glue:figure} tbl-hallazgos
:name: tbl-hallazgos
:figwidth: 85%

Desempeño en 2025, $w=21$ (MSE en puntos porcentuales²; RMSE y MAE en puntos porcentuales).
```

```{glue:figure} tbl-respaldo
:name: tbl-respaldo
:figwidth: 60%

Respaldo estadístico al HAR original por búsqueda (2025, $w=21$): rechazos DM–Holm7 e intervalos positivos.
```

**Conclusiones.**

1. HAR-Ridge + XGBoost obtiene RMSE 0,38601 y $R^2=0{,}8445$, frente a RMSE 0,54614 y $R^2=0{,}6885$ de la persistencia ({numref}`tbl-hallazgos`); la persistencia queda además fuera del MCS al 90 % ({numref}`tbl-spa`).
2. **HAR-Ridge + XGBoost es superior a los siete rivales del Grid original**, con respaldo estadístico total (7/7 en DM-Holm7 y 7/7 en IC); con los rivales reoptimizados la ventaja significativa se mantiene en 2–4 de 7 contrastes ({numref}`tbl-respaldo`).
3. **Lasso con Optuna TPE** mejora al original en las cuatro métricas observadas (RMSE 0,37874; $R^2=0{,}8500$), pero **sin ventaja significativa en MSE** bajo DM-Holm7.
4. **La superioridad del HAR se sostiene con evidencia estadística ajustada en el protocolo principal (Grid); su respaldo depende de la búsqueda.** Ambos conteos (DM-Holm7 e IC) indican evidencia favorable al HAR original en las cuatro búsquedas, total con el Grid y parcial con los rivales reoptimizados.
5. El diagnóstico de Bitcoin muestra que precio y logaritmo del precio no son estacionarios, que la magnitud del retorno tiene patrón semanal y que la estacionariedad de la volatilidad es dudosa: **no corresponde declarar a todo Bitcoin estacionario o sin estacionalidad**.

**Fortalezas.** (1) El modelo reduce el error de la persistencia de forma amplia. (2) Validación temporal con separación de 28 días y verificación automática de la purga ({numref}`tbl-cortes`, {numref}`tbl-calendario`).
(3) Inferencia con DM–HLN, HAC y Holm, intervalos y tamaño del efecto, todo recalculado desde pérdidas diarias versionadas (Capítulo 6). (4) Las tablas de la presentación se reproducen con celdas `assert`.

**Debilidades.** (1) 2025 es retrospectivo: ya había sido examinado y no hay un periodo confirmatorio intacto. (2) La ventana $w=21$ se eligió retrospectivamente. (3) Los presupuestos de búsqueda son pequeños (8 evaluaciones
únicas por búsqueda), por lo que la comparación es acotada. (4) Las familias de Holm7 son exploratorias y los IC son individuales, sin ajuste simultáneo.

## 8.2 Veredicto sobre las condiciones del modelo original

| Condición | Veredicto | Justificación |
|---|---|---|
| **R1 — Novedad** | **Declarada de forma acotada y condicionada** | Se declara la novedad de la *combinación* HAR-Ridge + XGBoost con ese protocolo, no del enfoque (existen antecedentes ML/HAR en cripto); condicionada a completar la matriz de búsqueda (Cap. 4). |
| **R2 — Implementación algorítmica propia** | **Parcial** | Los autores programaron las entradas HAR, el objetivo relativo y la selección de la mezcla; los aprendices base (`Ridge`, `xgboost`) son importados y no hay clase propia con `pytest` (Cap. 5). |
| **R3 — Superioridad estadística** | **Cumplida en el protocolo principal; respaldo parcial ante la reoptimización** | Superior y significativo frente a los siete rivales del Grid (DM-Holm7 7/7 e IC 7/7) y de forma consistente frente a k-NN y Decision Tree reoptimizados; no distinguible de Ridge, Lasso ni SVR lineal reoptimizados (con Optuna, Lasso tiene mejores métricas puntuales); SPA no rechaza; 2025 es retrospectivo (Cap. 6). |

```{admonition} Tope de calificación
:class: important
Con R2 solo parcial, R1 declarada de forma acotada y condicionada, y R3 cumplida en el protocolo principal pero con respaldo parcial ante la reoptimización, el tope de calificación aplicable depende de cómo la rúbrica pondere cada condición
(la rúbrica menciona topes de 5,0, 4,0 y 3,0). El valor exacto lo determina el docente; este libro no lo asume ni lo ajusta: reporta los hechos para que la decisión sea verificable.
```

## 8.3 Limitaciones

- **Sin periodo confirmatorio:** 2025 fue explorado repetidamente; el periodo de 2026 del repositorio se excluyó de las conclusiones por decisión de los autores. Un solo año de evaluación (358 orígenes) limita la potencia.
- **Diagnóstico de Bitcoin:** el contraste de Wald se recalcula en el libro; **ADF, KPSS y STL se transcriben de la presentación** porque sus archivos de resultados no están en el repositorio entregado.
- **Matriz de Spearman:** las celdas con resúmenes intradía se transcriben de la presentación (requieren los datos por minuto).
- **Sin benchmark GARCH/GJR/Realized GARCH** (el comparador natural según la literatura del Cap. 4 §4.5) ni comparación con redes profundas.
- **Matriz de búsqueda bibliográfica sin recuentos** (Cap. 4): la declaración de novedad queda condicionada a completarla.
- **Modelos serializados:** los `.joblib` de los 512 pipelines de la ampliación no están versionados; solo sus hashes.
- **Desviaciones de la rúbrica** (Cap. 6): IC normales HAC y por percentiles, no BCa; $d_z$ de Cohen y no $\delta$ de Cliff; Holm y no Benjamini–Hochberg.
- **Rúbrica de clasificación vs. problema de regresión:** no aplican balanceo de clases ni DeLong/McNemar.

## 8.4 Trabajo futuro

1. Evaluar sobre un periodo **realmente nuevo** con reglas fijadas antes de ver los datos.
2. Ejecutar y registrar la revisión sistemática del Capítulo 4.
3. Implementar un modelo propio en `numpy`/`scipy` con clase compatible con `scikit-learn` y sus pruebas, y compararlo bajo el mismo protocolo.
4. Añadir un benchmark GARCH/GJR, intervalos BCa y $\delta$ de Cliff; ampliar presupuestos de búsqueda.
5. Incorporar inferencia en vivo y capturas al dashboard sin comprometer la memoria del plan gratuito.

## 8.5 Consideraciones éticas

El estudio no formula recomendaciones de inversión: pronostica **volatilidad**, no precios ni rentabilidad, y sus resultados son retrospectivos. Los datos son públicos (Binance Spot; condiciones de uso en `DATA_LICENSE.md`)
y no se usan datos personales. Un pronóstico de volatilidad puede inducir sobreconfianza si se interpreta como garantía; por eso se reportan intervalos y se evita declarar superioridad sin evidencia ajustada.

## 8.6 Declaración de autoría y uso de herramientas de IA

Autores: **Jassan Arteta** y **Mateo Bernal** (Universidad del Norte, Maestría en Matemáticas). La estructura y la redacción de este libro, que sigue la presentación del proyecto, y parte del código de análisis se
elaboraron con asistencia de un modelo de lenguaje (**Claude, de Anthropic**), que no sustituye la responsabilidad de los autores. Los resultados numéricos provienen de los artefactos del repositorio y se verifican con celdas `assert`
al compilar. *Los autores deben revisar, ajustar y confirmar esta declaración antes de entregarla.*
