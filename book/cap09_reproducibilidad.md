# Capítulo 9 · Reproducibilidad y referencias

```{code-cell} ipython3
:tags: [remove-cell]
import json
import pandas as pd
from IPython.display import display
from jbook_utils import R, REPO, SEED, read_csv, glue_tbl
```

## 9.1 Entorno y semilla

- **Semilla global:** `42` (estimadores y búsquedas).
- **Python:** 3.11 para compilar el libro (CI); los resultados originales se calcularon con las versiones registradas abajo.
- **Entorno del libro:** `requirements-book.txt` (versiones fijadas). **Entorno de entrenamiento:** `requirements.txt`,
  `requirements-xgboost.txt` y `dashboard_data/final_review_20261018/files/results/final_regression_20261018/requirements-lock.txt`.

```{code-cell} ipython3
# Versiones de las librerías con que se entrenaron los 512 pipelines (artifacts/metadata.json)
meta = json.loads((REPO / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
vers = pd.Series(meta["librerias"], name="versión").to_frame()
assert meta["semilla_global"] == SEED == 42
glue_tbl("tbl-versiones", vers)
vers
```

```{glue:figure} tbl-versiones
:name: tbl-versiones
:figwidth: 55%

Librerías y versiones usadas en el entrenamiento de los pipelines finales.
```

## 9.2 Cómo reproducir

```{code-block} bash
:caption: Compilar el libro localmente
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-book.txt
python src/build_artifacts_metadata.py        # regenera artifacts/
python src/check_jbook_cells.py               # (opcional) ejecuta todas las celdas sin Sphinx
jupyter-book build book/                      # HTML en book/_build/html
```

Los cómputos pesados (512 pipelines, 28.672 evaluaciones de búsqueda) **no se re-ejecutan** al compilar: se leen sus
resultados versionados y el código que los generó está en `src/` (`run_final_regression.py`, `final_regression_core.py`,
`final_regression_search.py`) con sus *hashes* en `protocol.json`.

## 9.3 Tabla maestra de experimentos

```{code-cell} ipython3
# Tabla maestra: una fila por procedimiento (32), con métricas externas, hiperparámetros más frecuentes y tiempos
outer = read_csv(R / "outer_metrics.csv")
trials = read_csv(R / "master_trials.csv")
maestra = (outer.groupby(["family", "method"])
           .agg(filas_externas=("rmse", "size"), rmse_medio=("rmse", "mean"), mse_medio=("mse", "mean"),
                mae_medio=("mae", "mean"), r2_medio=("r2", "mean"),
                busqueda_s_total=("search_seconds", "sum"), ajuste_s_total=("fit_seconds", "sum"))
           .join(trials.groupby(["family", "method"]).agg(evaluaciones=("trial", "size")))
           .round(4).reset_index())
assert len(maestra) == 32 and maestra["filas_externas"].eq(96).all() and maestra["evaluaciones"].sum() == 28672
maestra.to_csv(REPO / "artifacts" / "master_experiments_summary.csv", index=False)
glue_tbl("tbl-maestra", maestra.sort_values("mse_medio").head(10).set_index(["family", "method"]))
maestra.sort_values("mse_medio").head(10)
```

```{glue:figure} tbl-maestra
:name: tbl-maestra
:figwidth: 100%

Tabla maestra (10 mejores de 32 procedimientos por MSE externo medio); tiempos en segundos.
```

La {numref}`tbl-maestra` resume la tabla completa, que se exporta a `artifacts/master_experiments_summary.csv`. El detalle por
modelo, pliegue, hiperparámetros y métricas está en `outer_metrics.csv` (3.072 filas) y el de cada evaluación de búsqueda en
`master_trials.csv` (28.672 filas), ambos en `dashboard_data/final_review_20261018/files/results/final_regression_20261018/`.
Nota: **el modelo original (HAR-Ridge + XGBoost) no figura como control del bucle externo** por política del protocolo (se
seleccionó globalmente en 2024); solo sus pronósticos congelados de 2025 entran en la comparación retrospectiva de las conclusiones.

## 9.4 Bibliografía

La bibliografía completa en BibTeX está en `references.bib` y se imprime en la sección {doc}`referencias`.
