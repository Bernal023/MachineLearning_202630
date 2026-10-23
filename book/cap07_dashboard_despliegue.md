# Capítulo 7 · Dashboard predictivo y despliegue

```{code-cell} ipython3
:tags: [remove-cell]
import re
import pandas as pd
from jbook_utils import REPO, DASHBOARD_URL
```

El dashboard público está en <https://machinelearning-202630.onrender.com/> (Dash + Plotly, servido con `gunicorn` en Render).

## 7.1 Arquitectura técnica

| Componente | Archivo | Función |
|---|---|---|
| Aplicación | `volatility_dashboard/app.py` | Crea la app Dash, el `server` Flask, rutas `/healthz`, `/review-release.json`, `/final-review/…` |
| Vistas | `volatility_dashboard/tabs/{contexto,eda,modelos,revision_final}.py` | Cada vista es un módulo con `layout()` y `register(app)` |
| Datos | `volatility_dashboard/data_loader.py`, `dashboard_data/` | Lee solo artefactos precalculados (no entrena) |
| Despliegue | `render.yaml`, `requirements-render.txt` | Servicio web gratuito en Render |

## 7.2 Las vistas

1. **Contexto del problema:** definición del objetivo, datos y modelos comparados.
2. **EDA:** gráficos interactivos por activo (distribuciones, series, correlaciones, temporalidad).
3. **Comparación de modelos:** *leaderboard* de métricas (R², RMSE, MAE) con filtros por activo, método y ventana, frente a la persistencia, y diagnósticos.
4. **Optimización y revisión final:** los 512 modelos, validación anidada, pruebas DM con Holm7/Holm28/Holm112, MCS, SHAP/LIME y eficiencia.

## 7.3 Guía de despliegue en Render

```{code-cell} ipython3
# Lectura del render.yaml real para documentar la configuración de producción
txt = (REPO / "render.yaml").read_text(encoding="utf-8")
assert "gunicorn volatility_dashboard.app:server" in txt and "--workers 1" in txt and "healthCheckPath: /healthz" in txt
claves = {k: re.search(pat, txt).group(1).strip() for k, pat in {
    "plan": r"plan:\s*(\S+)", "runtime": r"runtime:\s*(\S+)", "build": r"buildCommand:\s*(.+)",
    "health": r"healthCheckPath:\s*(\S+)", "python": r'PYTHON_VERSION\s*\n\s*value:\s*(\S+)'}.items()}
workers = re.search(r"--workers\s+(\d+)\s+--threads\s+(\d+)", txt).groups()
print(claves, "| workers/threads:", workers)
```

Puntos clave de la configuración (`render.yaml`):

- **Servidor:** `gunicorn volatility_dashboard.app:server --bind 0.0.0.0:$PORT`, con **1 *worker* y 2 hilos** y `--timeout 180`
  para absorber el arranque en frío del plan gratuito.
- **Salud:** `healthCheckPath: /healthz` (devuelve `{"status": "ok"}`).
- **Memoria:** el plan gratuito de Render ofrece unos 512 MB de RAM (verificar el límite vigente); por eso (i) se fijan `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y
  `MKL_NUM_THREADS` en 1, (ii) un solo *worker*, (iii) la aplicación **solo lee datos precalculados** y no importa SHAP, Optuna ni DEAP,
  y (iv) las dependencias de runtime son mínimas (`requirements-render.txt`).
- **Entorno:** `PYTHON_VERSION=3.11.11`.
- **Trazabilidad:** `/review-release.json` publica el identificador del paquete y el *commit* desplegado.
