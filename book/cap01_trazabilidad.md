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
| 1 | Riesgo de fuga de datos: particiones temporales intercaladas y ventanas que cruzan discontinuidades. | Particiones cronológicas expansivas con purga estricta (la última etiqueta de entrenamiento termina antes del primer origen de validación) y ventanas dentro de un mismo segmento continuo. | Cap. 3 §3.2 y §3.5, {numref}`tbl-cortes`, {numref}`tbl-calendario` | `nested_calendar.csv`: `strict_purge` verdadero en las 1.920 particiones. |
| 2 | Selección y evaluación sobre los mismos datos. | Validación anidada: toda la búsqueda se repite dentro de cada pliegue externo; la selección usa solo 2024. | Cap. 3 §3.5–3.7, {eq}`eq-nested` | `outer_metrics.csv` (3.072 filas), `selection_freeze.json`. |
| 3 | Comparar solo contra un modelo; faltan referencias simples. | Se añadieron persistencia, HAR diario OLS (1/7/28) y HAR simple (OLS y Ridge) como referencias. | Cap. 1 §1.3, Cap. 8 §8.1 | `professor_feedback_20261018/macro_metrics.csv`. |
| 4 | Examinar información mutua y colinealidad de las entradas. | MI y VIF de las 44 entradas HAR y de las 6 diarias, solo con 2023–2024; no se usaron para elegir variables. | Cap. 2 §2.10, {numref}`tbl-vif`, {numref}`tbl-mi` | `diagnostics/mutual_information.csv`, `variance_inflation.csv`. |
| 5 | Estacionalidad intradía. | Perfil horario de retornos² normalizados, 2023 vs. 2024, con bandas por bloques. | Cap. 2 §2.10, {numref}`fig-intradia` | `diagnostics/intraday_*.csv`. |
| 6 | Diagnóstico de residuos y posible autocorrelación. | Corrección autorregresiva causal por horizonte (solo errores ya madurados) con controles de último error maduro y sin etiquetas nuevas. | Cap. 1 §1.3, {numref}`tbl-ar` | `professor_feedback_20261018/dm_ar_holm4.csv`. |
| 7 | Pruebas de significancia con comparaciones múltiples. | DM–HLN con HAC y Holm, IC de la diferencia de MSE, tamaño del efecto (*bootstrap* estacionario), MCS 90 % y SPA. | Cap. 6 §6.3–6.7 | `evaluation/*.csv`, `spa_vs_original.json`. |
| 8 | Reportar con honestidad cambios de ranking al recalcular. | Se documenta el cambio de líder entre la validación 2024 y la evaluación 2025 y la correlación de rangos. | Cap. 3 §3.8, {numref}`tbl-rho`, {numref}`tbl-lider` | `macro_metrics.csv`, `outer_fold_mean_sd.csv`. |
| 9 | Reproducibilidad y modelos serializados. | `artifacts/` con `metadata.json`, manifiesto SHA-256 y prueba de ida y vuelta. | Cap. 3 §3.9, Cap. 9 | `artifacts/metadata.json`, `model_manifest.csv`. |

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
# Ranking 2025 incluyendo las referencias nuevas (MSE macro)
m = read_csv(PF / "macro_metrics.csv")
m = m[m.year == 2025]
rank = m.set_index("model")[["mse"]].sort_values("mse").round(4)
rank["posicion_2025"] = rank["mse"].rank().astype(int)
assert rank.loc["Persistencia", "mse"] > rank.loc["HAR diario OLS (1/7/28)", "mse"] > rank.loc["HAR-Ridge + XGBoost", "mse"]
glue_tbl("tbl-rank-corr", rank)
rank
```

```{glue:figure} tbl-rank-corr
:name: tbl-rank-corr
:figwidth: 80%

MSE macro (puntos porcentuales al cuadrado) de 2025 con las referencias añadidas y posición.
```

```{code-cell} ipython3
# Corrección AR causal vs. modelo vigente en 2025: sensibilidad al rezago HAC del DM
d = read_csv(PF / "dm_ar_holm4.csv")
ar = d[(d.reference == "HAR-Ridge + XGBoost") & (d.scope == "global") & (d.year == 2025)]
ar = (ar[["hac", "mse_candidate", "mse_reference", "reduction_percent", "p", "p_holm"]]
      .rename(columns={"hac": "rezagos_HAC", "mse_candidate": "MSE_con_AR", "mse_reference": "MSE_vigente",
                       "reduction_percent": "reduccion_%", "p": "p_crudo", "p_holm": "p_Holm4"})
      .round(4).set_index("rezagos_HAC"))
assert len(ar) == 3 and abs(ar.loc[14, "p_Holm4"] - .0645) < 1e-3 and (ar.loc[[33, 56], "p_Holm4"] < .05).all()
glue_tbl("tbl-ar", ar)
ar
```

```{glue:figure} tbl-ar
:name: tbl-ar
:figwidth: 100%

Corrección AR causal frente al vigente en 2025: MSE, reducción porcentual y p-valores DM (crudo y Holm4) según el rezago HAC.
```

**Interpretación de las {numref}`tbl-rank-corr` y {numref}`tbl-ar`.** (i) Las nuevas referencias simples quedan claramente detrás: en 2025 el
HAR diario OLS tiene MSE 0,6247 frente a 0,4647 del vigente y la persistencia 0,8282, de modo que la complejidad de ML sí aporta. (ii) La corrección
AR reduce el MSE solo un 1,1 %. Su p ajustado por Holm depende del rezago HAC: 0,065 con 14 rezagos y ≈ 0,027 con 33 y 56. Es decir, la evidencia es
**sensible a una elección del analista** (el rezago) y se obtuvo sobre un año ya examinado y sin periodo confirmatorio; por eso **no se considera
suficiente para sustituir al modelo vigente**, y así se conservó. Ni esa corrección ni el HAR-XG ponderado (con MSE puntual menor que el vigente, {numref}`tbl-rank-corr`) se adoptaron como reemplazo: no hay un periodo confirmatorio intacto. El ranking entre la validación 2024 y 2025 se analiza en el Capítulo 3 §3.8.
