# Predicción de la volatilidad futura de criptomonedas con Machine Learning

**Autores:** Jassan Alberto Arteta Chavarro · Mateo Bernal — Curso Machine Learning (2026-30)

| Recurso | Enlace |
|---|---|
| 📖 Informe técnico final (Jupyter Book) | <https://jassan-bit.github.io/MachineLearning_202630/> |
| 📊 Dashboard en producción (Render) | <https://machinelearning-202630.onrender.com/> |

## Qué hace el proyecto

Pronostica la **volatilidad diaria futura** (desviación estándar móvil poblacional de los retornos logarítmicos diarios, en %, sin anualizar)
de **BTC, ETH, BNB y XRP** (pares USDT, Binance Spot, 2020-2025, velas de 1 minuto) para **7 horizontes** (1 a 7 días) y **4 ventanas**
de volatilidad (7, 14, 21 y 28 días): 16 configuraciones y 112 series. **No** pronostica precio ni rentabilidad y no es una recomendación de inversión.

El modelo vigente es **HAR-Ridge + XGBoost**: una regresión Ridge sobre entradas tipo HAR (Corsi, 2009) combinada linealmente con XGBoost mediante
un peso y una penalización elegidos por validación temporal. No usa redes neuronales.

## Resultados principales

| Indicador (MSE macro, puntos porcentuales²) | 2025 (358 orígenes) | 2026 ene–ago (236 orígenes) |
|---|---:|---:|
| Persistencia | 0,8282 | 0,7951 |
| HAR diario OLS (1/7/28) | 0,6247 | 0,6086 |
| **HAR-Ridge + XGBoost (vigente)** | **0,4647** | **0,4560** |

- Se evaluaron **512 pipelines** (7 familias × 4 optimizadores × 16 configuraciones, más 64 híbridos HAR-XG) con **validación anidada y purga**.
- El modelo vigente reduce el MSE de la persistencia ≈ 44 % (2025) y ≈ 43 % (2026).
- **Ninguna variante lo supera de forma estadísticamente sólida** (DM–HLN con Holm, SPA, MCS 90 %, IC bootstrap estacionario).
- Las evaluaciones de 2025 y 2026 son **retrospectivas**: esos periodos ya habían sido examinados.

## Estado frente a las condiciones del modelo original

| Condición | Estado |
|---|---|
| R1 — Novedad | Declarada de forma **acotada y condicionada**: novedad de la *combinación* y del protocolo, no del enfoque (hay antecedentes de ML/HAR en cripto). Falta completar los recuentos de la matriz de búsqueda (Cap. 4). |
| R2 — Implementación propia (numpy/scipy) | **No cumplida**: usa `xgboost` y `scikit-learn`. |
| R3 — Superioridad estadística | **Refutada** (Cap. 6). |

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
