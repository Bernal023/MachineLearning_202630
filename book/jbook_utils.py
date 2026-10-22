"""Utilidades compartidas del Jupyter Book (rutas, carga de datos y figuras).

Todas las celdas del libro leen artefactos versionados en el repositorio
(``dashboard_data/``, ``data/processed/``). No se entrena ningún modelo pesado
al compilar el libro: los cómputos costosos se precalcularon con los scripts de
``src/`` y aquí solo se leen y analizan sus salidas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BOOK = Path(__file__).resolve().parent
REPO = BOOK.parent
_FR = REPO / "dashboard_data" / "final_review_20261018" / "files" / "results"
R = _FR / "final_regression_20261018"      # revisión final: 512 modelos, nested CV, inferencia
PF = _FR / "professor_feedback_20261018"   # correcciones atendidas (MI/VIF, HAR, AR)
DAILY = REPO / "data" / "processed" / "daily_2020_2025.csv"

SEED = 42
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
WINDOWS = [7, 14, 21, 28]
HORIZONS = list(range(1, 8))
DASHBOARD_URL = "https://machinelearning-202630.onrender.com/"


def read_csv(path, **kw):
    """Lee un CSV del repositorio con mensaje claro si falta el artefacto."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Falta el artefacto {path.relative_to(REPO)}; verifique que "
            "dashboard_data/ y data/processed/ estén versionados en el repositorio."
        )
    return pd.read_csv(path, **kw)


def load_daily_panel() -> pd.DataFrame:
    """ETL diario: cierres UTC -> retorno log (%) -> volatilidad móvil (ddof=0).

    Un retorno solo existe si ambos cierres consecutivos son válidos; una ventana
    de volatilidad solo existe si contiene ``w`` retornos válidos consecutivos
    (nunca cruza un hueco). Es la misma convención del estudio principal.
    """
    d = read_csv(DAILY, parse_dates=["date"])
    out = []
    for s, g in d.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        g["ret"] = 100.0 * np.log(g["close"]).diff()
        for w in WINDOWS:
            g[f"vol{w}"] = g["ret"].rolling(w, min_periods=w).std(ddof=0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def acf(x, nlags):
    """Autocorrelación muestral (numpy) de una serie con NaN eliminados."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    x = x - x.mean()
    den = float(np.dot(x, x))
    return np.array([np.dot(x[: len(x) - k], x[k:]) / den for k in range(nlags + 1)])


def glue_fig(name, fig):
    """Publica la figura con ``myst_nb.glue`` para citarla con {numref}."""
    import matplotlib.pyplot as plt
    from myst_nb import glue

    glue(name, fig, display=False)
    plt.close(fig)


def glue_tbl(name, df):
    from myst_nb import glue

    glue(name, df, display=False)
