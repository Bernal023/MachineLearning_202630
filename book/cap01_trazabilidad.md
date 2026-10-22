# Capítulo 1 · Trazabilidad de correcciones (Entregable 2)

```{code-cell} ipython3
:tags: [remove-cell]
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from IPython.display import display
from jbook_utils import R, PF, REPO, read_csv, glue_tbl
```

## 1.1 Tabla de trazabilidad

| # | Observación del docente | Acción realizada | Ubicación en el Jupyter Book | Evidencia |
|---|---|---|---|---|
| 1 | Riesgo de fuga de datos: particiones temporales intercaladas y ventanas que cruzan discontinuidades. | Particiones cronológicas expansivas con purga estricta (la última etiqueta de entrenamiento termina antes del primer origen de validación) y ventanas dentro de un mismo segmento continuo. | Cap. 3 §3.2, {numref}`tbl-calendario` | `nested_calendar.csv`: `strict_purge` verdadero en las 1.920 particiones. |
| 2 | Selección y evaluación sobre los mismos datos. | Validación anidada: toda la búsqueda se repite dentro de cada pliegue externo; la selección usa solo 2024. | Cap. 3 §3.2–3.4, {eq}`eq-nested` | `outer_metrics.csv` (3.072 filas), `selection_freeze.json`. |
| 3 | Comparar solo contra un modelo; faltan referencias simples. | Se añadieron persistencia, HAR diario OLS (1/7/28) y HAR simple (OLS y Ridge) como referencias. | Cap. 1 §1.3, Cap. 6 §6.2 | `professor_feedback_20261018/macro_metrics.csv`. |
| 4 | Examinar información mutua y colinealidad de las entradas. | MI y VIF de las 44 entradas HAR y de las 6 diarias, solo con 2023–2024; no se usaron para elegir variables. | Cap. 2 §2.6, {numref}`tbl-vif`, {numref}`tbl-mi` | `diagnostics/mutual_information.csv`, `variance_inflation.csv`. |
| 5 | Estacionalidad intradía. | Perfil horario de retornos² normalizados, 2023 vs. 2024, con bandas por bloques. | Cap. 2 §2.7, {numref}`fig-intradia` | `diagnostics/intraday_*.csv`. |
| 6 | Diagnóstico de residuos y posible autocorrelación. | Corrección autorregresiva causal por horizonte (solo errores ya madurados) con controles de último error maduro y sin etiquetas nuevas. | Cap. 1 §1.3, {numref}`tbl-ar` | `professor_feedback_20261018/dm_ar_holm4.csv`. |
| 7 | Pruebas de significancia con comparaciones múltiples. | DM–HLN con Holm, MCS 90 %, SPA, IC *bootstrap* estacionario y tamaño del efecto. | Cap. 6 §6.3–6.4 | `evaluation/*.csv`, `spa_vs_original.json`. |
| 8 | Reportar con honestidad cambios de ranking al recalcular. | Se documenta el cambio de líder entre 2024, 2025 y 2026 y la correlación de rangos. | Cap. 3 §3.5, {numref}`tbl-rho`, {numref}`tbl-lider` | `macro_metrics.csv`, `outer_fold_mean_sd.csv`. |
| 9 | Reproducibilidad y modelos serializados. | `artifacts/` con `metadata.json`, manifiesto SHA-256 y prueba de ida y vuelta. | Cap. 3 §3.6, Cap. 9 | `artifacts/metadata.json`, `model_manifest.csv`. |

## 1.2 Verificación automática de la evidencia

Cada fila de la tabla se respalda con archivos versionados; la celda siguiente comprueba que existen y que
las cifras citadas en los capítulos coinciden. Si un archivo falta o una cifra cambia, **la compilación del libro falla**.

```{code-cell} ipython3
# Verificación de evidencias de la tabla de trazabilidad
evidencias = {
    1: R / "nested_calendar.csv", 2: R / "outer_metrics.csv", 3: PF / "macro_metrics.csv",
    4: PF / "diagnostics" / "mutual_information.csv", 5: PF / "diagnostics" / "intraday_summary.csv",
    6: PF / "dm_ar_holm4.csv", 7: R / "evaluation" / "directed_dm_all_years.csv",
    8: R / "outer_fold_mean_sd.csv", 9: REPO / "artifacts" / "metadata.json",
}
chequeo = pd.DataFrame({"fila": list(evidencias), "archivo": [str(p.relative_to(REPO)) for p in evidencias.values()],
                        "existe": [p.exists() for p in evidencias.values()]}).set_index("fila")
assert chequeo["existe"].all(), chequeo[~chequeo["existe"]]

val = json.loads((R / "validation.json").read_text(encoding="utf-8"))
assert val["status"] == "passed" and val["tests_run"] == 71 and val["failures"] == 0 and val["errors"] == 0
print(f"Pruebas automáticas de la revisión final: {val['tests_run']} ejecutadas, "
      f"{val['failures']} fallos, {val['errors']} errores.")
glue_tbl("tbl-evidencias", chequeo)
chequeo
```

```{glue:figure} tbl-evidencias
:name: tbl-evidencias
:figwidth: 100%

Existencia de cada archivo de evidencia citado en la tabla de trazabilidad.
```

La {numref}`tbl-evidencias` confirma que las nueve evidencias están presentes; además, la revisión final
pasó 71 pruebas automáticas sin fallos ni errores (`validation.json`).

## 1.3 Resultados recalculados tras las correcciones

Las referencias añadidas (fila 3) y la corrección autorregresiva (fila 6) **no cambian la conclusión** y sí
ordenan los modelos de forma informativa.

```{code-cell} ipython3
# Ranking 2025/2026 incluyendo las referencias nuevas (MSE macro)
m = read_csv(PF / "macro_metrics.csv")
rank = (m.pivot(index="model", columns="year", values="mse").sort_values(2026).round(4))
rank["pos_2025"] = rank[2025].rank().astype(int); rank["pos_2026"] = rank[2026].rank().astype(int)
assert rank.loc["Persistencia", 2026] > rank.loc["HAR diario OLS (1/7/28)", 2026] > rank.loc["HAR-Ridge + XGBoost", 2026]
glue_tbl("tbl-rank-corr", rank)
rank
```

```{glue:figure} tbl-rank-corr
:name: tbl-rank-corr
:figwidth: 95%

MSE macro (puntos porcentuales al cuadrado) de 2025 y 2026 con las referencias añadidas y posición en cada año.
```

```{code-cell} ipython3
# Corrección AR causal vs. modelo vigente: sensibilidad al rezago HAC del DM
d = read_csv(PF / "dm_ar_holm4.csv")
ar = d[(d.reference == "HAR-Ridge + XGBoost") & (d.scope == "global")]
ar = (ar[["year", "hac", "mse_candidate", "mse_reference", "reduction_percent", "p", "p_holm"]]
      .rename(columns={"hac": "rezagos_HAC", "mse_candidate": "MSE_con_AR", "mse_reference": "MSE_vigente",
                       "reduction_percent": "reduccion_%", "p": "p_crudo", "p_holm": "p_Holm4"})
      .round(4).set_index(["year", "rezagos_HAC"]))
p26 = ar.xs(2026, level="year")["p_Holm4"]
assert (p26 > .05).all() and abs(ar.loc[(2026, 33), "p_Holm4"] - .5851) < 1e-3
glue_tbl("tbl-ar", ar)
ar
```

```{glue:figure} tbl-ar
:name: tbl-ar
:figwidth: 100%

Corrección AR causal frente al vigente: MSE, reducción porcentual y p-valores DM (crudo y Holm4) según el rezago HAC.
```

**Interpretación de las {numref}`tbl-rank-corr` y {numref}`tbl-ar`.** (i) Las nuevas referencias simples quedan claramente
detrás: en 2026 el HAR diario OLS tiene MSE 0,6086 frente a 0,4560 del vigente y la persistencia 0,7951, de modo que
la complejidad de ML sí aporta. (ii) La corrección AR reduce el MSE solo un 1,1 % en 2025 y un 0,7 % en 2026. En 2025 su
p ajustado depende del rezago HAC (0,027–0,065) y solo cruza 0,05 con 33 y 56 rezagos; en 2026, la
prueba primaria predeclarada, el p ajustado es 0,58–0,73 para cualquier rezago. Como 2025 y 2026 ya habían sido examinados
repetidamente, **no se considera evidencia suficiente para sustituir al modelo vigente**, y
así se conservó. El cambio de ranking que sí ocurre (el HAR-XG ponderado y la corrección de residuos tienen MSE puntual menor que el
vigente, {numref}`tbl-rank-corr`) tampoco está respaldado estadísticamente: las candidatas HAR-XG contrastadas con DM–Holm en el
Capítulo 6 no resultaron significativas. El ranking entre 2024, 2025 y 2026 se analiza en el Capítulo 3 §3.5.
