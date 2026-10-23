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


def load_daily_panel(start=None, end=None) -> pd.DataFrame:
    """ETL diario: cierres UTC -> retorno log (%) -> volatilidad móvil (ddof=0).

    Un retorno solo existe si ambos cierres consecutivos son válidos; una ventana
    de volatilidad solo existe si contiene ``w`` retornos válidos consecutivos
    (nunca cruza un hueco). Es la misma convención del estudio principal.
    """
    d = read_csv(DAILY, parse_dates=["date"])
    if start is not None:
        d = d[d["date"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        d = d[d["date"] <= pd.Timestamp(end, tz="UTC")]
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


# ---------------------------------------------------------------------------
# Inferencia predictiva (Diebold-Mariano, Holm, IC HAC, bootstrap estacionario)
# ---------------------------------------------------------------------------
from scipy import stats as _st


def hac_lrv(x, lag):
    """Varianza de largo plazo con núcleo de Bartlett (Newey-West), x ya centrada o no."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    g = [float(np.dot(xc[k:], xc[: n - k]) / n) for k in range(lag + 1)]
    return g[0] + 2.0 * sum((1.0 - k / (lag + 1.0)) * g[k] for k in range(1, lag + 1))


def dm_hln(loss_a, loss_b, lag=26, h=7):
    """DM bilateral con ajuste HLN y p de una t(n-1). d_t = L_A,t - L_B,t (DM<0 favorece a A)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    stat = d.mean() / np.sqrt(hac_lrv(d, lag) / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"dm": float(stat), "p": float(2 * (1 - _st.t.cdf(abs(stat), n - 1))),
            "dbar": float(d.mean()), "n": n}


def gain_ci(loss_rival, loss_ref, lag=26, z=1.96):
    """G = MSE_rival - MSE_ref con IC normal HAC (Bartlett) individual al 95 %."""
    d = np.asarray(loss_rival, dtype=float) - np.asarray(loss_ref, dtype=float)
    n = len(d)
    se = np.sqrt(hac_lrv(d, lag) / n)
    return float(d.mean()), float(d.mean() - z * se), float(d.mean() + z * se)


def holm(p):
    """p ajustados de Holm: (m-j+1)p_(j), máximo acumulado, truncado a 1."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    out = np.empty(m)
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (m - r) * p[i])
        out[i] = min(1.0, run)
    return out


def stationary_bootstrap_indices(n, mean_block, reps, rng):
    """Índices del bootstrap estacionario de Politis y Romano (bloques geométricos)."""
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=reps)
    for t in range(1, n):
        jump = rng.random(reps) < 1.0 / mean_block
        idx[:, t] = np.where(jump, rng.integers(n, size=reps), (idx[:, t - 1] + 1) % n)
    return idx


def acf_gaps(x, nlags):
    """ACF 'conservadora': conserva el calendario; usa solo pares de fechas ambos observadas."""
    x = np.asarray(x, dtype=float)
    xo = x - np.nanmean(x)
    den = float(np.nansum(xo ** 2))
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float(np.nansum(xo[:-k] * xo[k:]) / den))
    return np.array(out)


def longest_segment(s):
    """(n, primera fecha, última fecha) del mayor tramo continuo de valores finitos."""
    ok = np.isfinite(s.values.astype(float))
    best, i = (0, 0, 0), 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j - 1)
            i = j
        else:
            i += 1
    return best[0], s.index[best[1]].date(), s.index[best[2]].date()


def weekday_wald(y, lags=28):
    """Wald conjunto (chi2 con 6 g.l.) de igualdad de medias por día de la semana.

    OLS con constante y seis indicadores; covarianza HAC (Newey-West, Bartlett) con la
    corrección de muestra pequeña n/(n-k), equivalente a statsmodels (HAC, use_correction=True).
    """
    wd = y.index.dayofweek.values
    X = np.column_stack([np.ones(len(y))] + [(wd == k).astype(float) for k in range(1, 7)])
    n, k = X.shape
    b = np.linalg.solve(X.T @ X, X.T @ y.values)
    Xe = X * (y.values - X @ b)[:, None]
    S = Xe.T @ Xe
    for j in range(1, lags + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lags + 1)) * (G + G.T)
    A = np.linalg.inv(X.T @ X)
    V = A @ S @ A * n / (n - k)
    Rm = np.zeros((6, 7)); Rm[:, 1:] = np.eye(6)
    rb = Rm @ b
    W = float(rb @ np.linalg.solve(Rm @ V @ Rm.T, rb))
    return W, float(_st.chi2.sf(W, 6))


# ---------------------------------------------------------------------------
# Pérdidas diarias de 2025 por ventana y procedimiento de búsqueda
# ---------------------------------------------------------------------------
HAR = "HAR-Ridge + XGBoost"
FAMILIES = {"kNN": "k-NN", "Ridge": "Ridge", "Lasso": "Lasso", "RandomForest": "Random Forest",
            "XGBoost": "XGBoost", "SVR": "SVR Lineal", "DecisionTree": "Decision Tree"}
SEARCHES = {"grid": "Grid Search*", "random": "Random Search", "bayesian": "Bayesiana (Optuna TPE)", "genetic": "Genética (DEAP)"}
WIN_LOSSES = R / "window_review_original_vs_optimized_2025" / "daily_losses.csv"
ORIG_DIR = REPO / "results" / "har_original_ml_confidence_20261006" / "2025"


def hac_lag(window):
    """Rezagos HAC del estudio: w + 5 (12, 19, 26 y 33 para w = 7, 14, 21 y 28)."""
    return int(window) + 5


def search_losses(window, search):
    """(pérdida diaria del HAR, DataFrame de pérdidas de los 7 rivales) en 2025.

    'grid' usa los seis rivales del estudio original más el Decision Tree añadido (*); las demás
    búsquedas usan los ocho candidatos reoptimizados con validación anidada.
    """
    W = read_csv(WIN_LOSSES)
    x = W[W.volatility_window == window].reset_index(drop=True)
    cols = {}
    if search == "grid":
        o = read_csv(ORIG_DIR / str(window) / "daily_loss.csv")
        assert np.allclose(o[HAR].values, x[HAR].values)
        for f, n in FAMILIES.items():
            cols[n + ("*" if f == "DecisionTree" else "")] = (x["DecisionTree [grid]"] if f == "DecisionTree" else o[n]).values
    else:
        for f, n in FAMILIES.items():
            cols[n] = x[f"{f} [{search}]"].values
    return x[HAR].values, pd.DataFrame(cols)


def inference_table(window, search):
    """DM-HLN (HAC w+5), Holm7 e IC95 % HAC de G = MSE_rival - MSE_HAR frente a cada rival."""
    har, riv = search_losses(window, search)
    lag = hac_lag(window)
    rows = []
    for name in riv.columns:
        d = dm_hln(har, riv[name].values, lag=lag)
        g, lo, hi = gain_ci(riv[name].values, har, lag=lag)
        rows.append({"rival": name, "DM_HLN": d["dm"], "p_bruto": d["p"], "G": g, "IC_inf": lo, "IC_sup": hi})
    t = pd.DataFrame(rows).set_index("rival")
    t["p_Holm7"] = holm(t["p_bruto"].values)
    t["rechaza_H0"] = (t["p_Holm7"] < .05) & (t["DM_HLN"] < 0)
    t["IC_positivo"] = t["IC_inf"] > 0
    return t


def glue_fig(name, fig):
    """Publica la figura con ``myst_nb.glue`` para citarla con {numref}."""
    import matplotlib.pyplot as plt
    from myst_nb import glue

    glue(name, fig, display=False)
    plt.close(fig)


def glue_tbl(name, df):
    from myst_nb import glue

    glue(name, df, display=False)
