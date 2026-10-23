# Predicción de la volatilidad futura de criptomonedas con Machine Learning

**Autores:** Jassan Alberto Arteta Chavarro · Mateo Bernal — Curso Machine Learning (2026-30)

| Recurso | Enlace |
|---|---|
| 📖 Informe técnico final (Jupyter Book) | <https://jassan-bit.github.io/MachineLearning_202630/> |
| 📊 Dashboard en producción (Render) | <https://machinelearning-202630.onrender.com/> |

## Qué hace el proyecto

Pronostica la **volatilidad diaria futura** (desviación estándar móvil poblacional de los retornos logarítmicos diarios, en %, sin anualizar) de **BTC, ETH, BNB y XRP** (pares USDT, Binance Spot,
cierres de un minuto, 2023-2025, 1.096 fechas diarias) para **7 horizontes** (1 a 7 días) y **4 ventanas** de volatilidad (7, 14, 21 y 28 días). **No** pronostica precios ni rentabilidad y no es una recomendación de inversión.

El modelo propuesto es **HAR-Ridge + XGBoost**: una regresión Ridge sobre entradas tipo HAR (Corsi, 2009) combinada con XGBoost mediante un peso λ elegido en validación temporal. Se compara con k-NN, Ridge, Lasso,
Random Forest, XGBoost y SVR lineal (más un Decision Tree añadido) sobre los mismos objetivos y fechas. No usa redes neuronales.

## Resultados principales (2025, ventana w = 21, 358 orígenes)

| Modelo | MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| **HAR-Ridge + XGBoost** | **0,18444** | **0,38601** | **0,25372** | **0,8445** |
| Persistencia | 0,37183 | 0,54614 | 0,34405 | 0,6885 |
| Lasso (Optuna TPE) | 0,17825 | 0,37874 | 0,23904 | 0,8500 |

| Búsqueda de los rivales | DM-Holm7 favorables al HAR | IC95 % positivos |
|---|---:|---:|
| Grid (original) | 7/7 | 7/7 |
| Random Search | 3/7 | 3/7 |
| Optuna TPE | 2/7 | 3/7 |
| DEAP | 4/7 | 4/7 |

- **El respaldo al HAR depende de la búsqueda; la superioridad requiere evidencia estadística ajustada.** Con Optuna, Lasso mejora al original en las cuatro métricas, sin ventaja significativa bajo DM-Holm7.
- La persistencia queda fuera del MCS al 90 %. El SPA global no rechaza (p = 0,106).
- Las conclusiones usan **solo 2025**, evaluación **retrospectiva** (periodo ya examinado); el periodo de 2026 del repositorio no se usa para concluir.

## Estado frente a las condiciones del modelo original

| Condición | Estado |
|---|---|
| R1 — Novedad | Declarada de forma **acotada y condicionada**: novedad de la *combinación* y del protocolo, no del enfoque (hay antecedentes de ML/HAR en cripto). Falta completar los recuentos de la matriz de búsqueda (Cap. 4). |
| R2 — Implementación algorítmica propia | **Parcial**: entradas HAR, objetivo relativo y selección de la mezcla son propios; `Ridge` y `xgboost` son importados. |
| R3 — Superioridad estadística | **No verificada de forma general** (Cap. 6). |

## Estructura del repositorio

```
book/                 Jupyter Book (_config.yml, _toc.yml, references.bib, capítulos cap01–cap09)
artifacts/            metadata.json, model_manifest.csv (512 pipelines, SHA-256), tabla maestra
src/                  scripts de experimentos, informes y utilidades del libro
tests/                pruebas automáticas (unittest)
volatility_dashboard/ aplicación Dash (pestañas) desplegada en Render
dashboard_data/       resultados precalculados que lee el dashboard y el libro
data/processed/       cierres diarios 2020-2025 y datos por minuto procesados
results/              resultados de cada estudio y modelos serializados
```

## Reproducir el libro

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-book.txt
python src/build_artifacts_metadata.py                 # regenera artifacts/
jupyter-book build book/                               # HTML en book/_build/html
```

Verificación rápida sin Sphinx: `python src/check_jbook_cells.py`. Los cómputos pesados (512 pipelines) **no se re-ejecutan**: el libro lee sus
resultados versionados; el código que los generó está en `src/` (`run_final_regression.py`, `final_regression_core.py`, `final_regression_search.py`).
Semilla global: `42`. Versiones de entrenamiento: `artifacts/metadata.json`.

## Publicación

El workflow `.github/workflows/jupyter-book-pages.yml` compila y publica el libro en GitHub Pages en cada *push* a `main`
(*Settings → Pages → Source: GitHub Actions*). El dashboard se despliega en Render con `render.yaml` (`gunicorn`, 1 worker).

## Datos y licencia

Datos públicos de Binance Spot; condiciones de uso en [`DATA_LICENSE.md`](DATA_LICENSE.md). No se usan datos personales.

## Uso de herramientas de IA

La estructura y la redacción del Jupyter Book se elaboraron con asistencia de un modelo de lenguaje (Claude, de Anthropic). Los resultados numéricos
provienen de los artefactos del repositorio y se verifican con celdas `assert` al compilar. Los autores son responsables del contenido.

> El historial de experimentos con MLP y SVR del README anterior se conserva en `README_MLP_legacy.md`.
