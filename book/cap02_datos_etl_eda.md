# Capítulo 2 · Datos: ETL y EDA

```{code-cell} ipython3
:tags: [remove-cell]
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from jbook_utils import (SYMBOLS, WINDOWS, PF, DAILY, load_daily_panel, read_csv,
                         acf, glue_fig, glue_tbl)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = dict(zip(SYMBOLS, ["#f7931a", "#627eea", "#c9a227", "#23292f"]))
```

## 2.1 Pipeline de extracción, limpieza y transformación

El pipeline completo se ejecutó una vez sobre los archivos mensuales de velas de un minuto de
Binance Spot (`src/download_three_years.py`, `src/minute_experiment.py`; procedencia y *hashes* en
`results/minute_2023_2025/data_manifest.json`) y produjo el cierre diario UTC por activo, que
es el insumo de este libro (`data/processed/daily_2020_2025.csv`, 8.768 filas). Las etapas son:

1. **Extracción.** Velas de 1 min de BTCUSDT, ETHUSDT, BNBUSDT y XRPUSDT, 1-ene-2020 a 31-dic-2025.
2. **Limpieza.** Se marca un día como *completo* si tiene 1.440 minutos válidos (`complete`) y se
   cuentan cierres inválidos (`invalid_close_times`). Un cierre de un día incompleto no se imputa:
   queda como valor faltante.
3. **Transformación.** Retorno logarítmico diario en porcentaje, $r_t = 100\,[\ln P_t - \ln P_{t-1}]$,
   y volatilidad móvil poblacional de ventana $w \in \{7,14,21,28\}$:

```{math}
:label: eq-vol
\sigma_t^{(w)} = \sqrt{\frac{1}{w}\sum_{j=0}^{w-1}\left(r_{t-j}-\bar r_t^{(w)}\right)^2},
\qquad \bar r_t^{(w)}=\frac{1}{w}\sum_{j=0}^{w-1} r_{t-j}.
```

El objetivo supervisado en el origen $t$ son los siete valores $\big(\sigma_{t+1}^{(w)},\dots,\sigma_{t+7}^{(w)}\big)$.
Un retorno solo existe si ambos cierres consecutivos son válidos y una ventana solo existe si
contiene $w$ retornos válidos consecutivos, de modo que **ninguna ventana cruza un hueco**. La función
`load_daily_panel` (en `jbook_utils.py`) implementa exactamente esta transformación.

```{code-cell} ipython3
# ETL: lectura del cierre diario y construcción de retornos y volatilidades (ecuación eq-vol)
panel = load_daily_panel()

calidad = panel.groupby("symbol").agg(
    dias=("date", "size"),
    primer_dia=("date", lambda s: s.min().date()),
    ultimo_dia=("date", lambda s: s.max().date()),
    dias_incompletos=("complete", lambda s: int((~s).sum())),
    cierres_faltantes=("close", lambda s: int(s.isna().sum())),
    retornos_validos=("ret", "count"),
).loc[SYMBOLS]
calidad["incompletos_%"] = (100 * calidad["dias_incompletos"] / calidad["dias"]).round(2)
glue_tbl("tbl-calidad", calidad)
calidad
```

```{glue:figure} tbl-calidad
:name: tbl-calidad
:figwidth: 95%

Calidad del cierre diario por activo (2020-2025): días, días incompletos y retornos válidos.
```

**Interpretación de la {numref}`tbl-calidad`.** Cada activo tiene 2.192 días calendario consecutivos
y solo 15–16 días incompletos (≈0,7 %), por lo que la cobertura es casi total. Los faltantes no
son aleatorios en el tiempo: se concentran en 2020 y 2021 (59 de las 63 observaciones
incompletas del panel; las 4 restantes caen en 2023), antes de las evaluaciones de 2025–2026 y casi
todas antes de la ventana de desarrollo 2023–2024. Como los faltantes no se imputan y las ventanas no cruzan huecos, la
decisión de no imputar no altera el periodo de evaluación; sí reduce unas pocas ventanas de
volatilidad en 2020–2021.

```{code-cell} ipython3
# Verificaciones de coherencia con los números citados en el texto
assert calidad["dias"].eq(2192).all()
assert calidad["dias_incompletos"].between(15, 16).all()
inc_year = panel.assign(anio=panel["date"].dt.year).groupby("anio")["complete"].apply(lambda s: int((~s).sum()))
assert inc_year.loc[[2020, 2021]].sum() == 59 and inc_year.loc[2023] == 4 and inc_year.sum() == 63  # 59 + 4 = 63
```

## 2.2 Distribución de las características

```{code-cell} ipython3
# Estadísticos descriptivos de los retornos diarios (%) y detección de atípicos por regla IQR
filas = []
for s in SYMBOLS:
    r = panel.loc[panel.symbol == s, "ret"].dropna()
    q1, q3 = r.quantile([.25, .75])
    iqr = q3 - q1
    filas.append({
        "activo": s, "n": len(r), "media_%": r.mean(), "desv_%": r.std(),
        "asimetria": stats.skew(r), "curtosis_exceso": stats.kurtosis(r),
        "min_%": r.min(), "max_%": r.max(),
        "atipicos_IQR_%": 100 * ((r < q1 - 1.5 * iqr) | (r > q3 + 1.5 * iqr)).mean(),
    })
desc = pd.DataFrame(filas).set_index("activo").round(3)
glue_tbl("tbl-desc", desc)
desc
```

```{glue:figure} tbl-desc
:name: tbl-desc
:figwidth: 100%

Estadísticos de los retornos logarítmicos diarios (%), 2020-2025, y proporción de atípicos por la regla del IQR.
```

```{code-cell} ipython3
# Figura: histogramas de retornos diarios y comparación con la normal ajustada
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
for ax, s in zip(axes, SYMBOLS):
    r = panel.loc[panel.symbol == s, "ret"].dropna()
    xs = np.linspace(-20, 20, 400)
    ax.hist(r.clip(-20, 20), bins=80, density=True, color=COL[s], alpha=.75)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "k--", lw=1, label="Normal ajustada")
    ax.set_title(s); ax.set_xlabel("Retorno logarítmico diario (%)")
axes[0].set_ylabel("Densidad"); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Distribución de retornos diarios (recortados a ±20 % solo para visualizar)", y=1.03)
fig.tight_layout()
glue_fig("fig-ret-hist", fig)
```

```{glue:figure} fig-ret-hist
:name: fig-ret-hist
:figwidth: 100%

Histogramas de los retornos diarios (eje x: retorno logarítmico en %; eje y: densidad) frente a la normal ajustada.
```

**Interpretación de la {numref}`tbl-desc` y la {numref}`fig-ret-hist`.** Los retornos tienen colas
muy pesadas: el exceso de curtosis va de 18,8 (ETH) a 30,8 (BNB) —una normal tendría 0—, y existen
caídas diarias de entre 50 % y 59 % en logaritmos (el 12-mar-2020 en BTC, ETH y BNB; el 23-dic-2020 en XRP). La asimetría es negativa en BTC,
ETH y BNB y positiva en XRP. Entre 6,6 % y 8,2 % de los días son atípicos por la regla del IQR, una
proporción alta que **no corresponde a errores de medición sino a la dinámica del activo**; por eso no se
eliminan observaciones: eliminar los extremos borraría justo los episodios de volatilidad que se
quieren pronosticar. La sensibilidad a extremos se evalúa con errores en escala original (MSE y MAE) y
con intervalos *bootstrap* en el Capítulo 6. La normalidad queda descartada, lo que justifica
usar métodos no paramétricos y errores robustos a heterocedasticidad en las pruebas.

## 2.3 La variable objetivo: forma y «balance»

Al ser un problema de regresión no hay clases que balancear; el análogo es la **asimetría y la
escala** del objetivo.

```{code-cell} ipython3
# Objetivo vol7: nivel, asimetría y efecto de la transformación logarítmica
filas = []
for s in SYMBOLS:
    v = panel.loc[panel.symbol == s, "vol7"].dropna()
    filas.append({"activo": s, "n": len(v), "media": v.mean(), "mediana": v.median(),
                  "p95": v.quantile(.95), "asimetria": stats.skew(v),
                  "asimetria_log": stats.skew(np.log(v))})
obj = pd.DataFrame(filas).set_index("activo").round(3)
glue_tbl("tbl-objetivo", obj)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for s in SYMBOLS:
    v = panel.loc[panel.symbol == s, "vol7"].dropna()
    axes[0].hist(v, bins=60, density=True, histtype="step", lw=1.6, color=COL[s], label=s)
    axes[1].hist(np.log(v), bins=60, density=True, histtype="step", lw=1.6, color=COL[s], label=s)
axes[0].set_xlabel("Volatilidad a 7 días (% de retorno diario)"); axes[0].set_ylabel("Densidad")
axes[1].set_xlabel("ln(volatilidad a 7 días)"); axes[1].set_ylabel("Densidad")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Objetivo σ⁽⁷⁾: escala original (izq.) y logarítmica (der.)", y=1.03)
fig.tight_layout()
glue_fig("fig-objetivo", fig)
obj
```

```{glue:figure} tbl-objetivo
:name: tbl-objetivo
:figwidth: 90%

Resumen del objetivo σ⁽⁷⁾ (% de retorno diario): media, mediana, percentil 95 y asimetría en nivel y en logaritmo.
```

```{glue:figure} fig-objetivo
:name: fig-objetivo
:figwidth: 100%

Densidad del objetivo σ⁽⁷⁾ en escala original (izq.) y en logaritmo (der.).
```

**Interpretación de la {numref}`tbl-objetivo` y la {numref}`fig-objetivo`.** En nivel, el objetivo
es fuertemente asimétrico a la derecha (asimetría entre 2,8 y 4,1) y su media supera a su mediana
(p. ej. BTC: 2,57 vs. 2,23). En logaritmo la asimetría cae a valores entre −0,12 y 0,40, es decir, la
distribución es aproximadamente log-normal. Consecuencia metodológica: los errores grandes
dominan el MSE en nivel, por lo que el estudio reporta MSE, RMSE y MAE **en escala original** y
compara modelos con pérdidas pareadas por fecha, no con una sola métrica.

## 2.4 Análisis temporal: dependencia y estacionariedad

```{code-cell} ipython3
# Figura: volatilidad a 7 días en el tiempo y ACF muestral (con bandas ±1.96/sqrt(n))
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
for s in SYMBOLS:
    g = panel[panel.symbol == s]
    axes[0].plot(g["date"], g["vol7"], lw=.8, color=COL[s], label=s)
axes[0].set_xlabel("Fecha (UTC)"); axes[0].set_ylabel("Volatilidad a 7 días (%)")
axes[0].set_title("Volatilidad σ⁽⁷⁾ diaria, 2020–2025"); axes[0].legend(frameon=False, ncol=4, fontsize=8)
lags = np.arange(0, 61)
for s in SYMBOLS:
    v = panel.loc[panel.symbol == s, "vol7"].dropna().values
    axes[1].plot(lags[1:], acf(v, 60)[1:], lw=1.4, color=COL[s], label=s)
band = 1.96 / np.sqrt(panel.loc[panel.symbol == "BTCUSDT", "vol7"].notna().sum())
axes[1].axhspan(-band, band, color="gray", alpha=.25)
axes[1].set_xlabel("Rezago (días)"); axes[1].set_ylabel("Autocorrelación")
axes[1].set_title("ACF de σ⁽⁷⁾ (banda gris: ±1,96/√n)")
fig.tight_layout()
glue_fig("fig-temporal", fig)

acf_tab = pd.DataFrame({
    s: {f"ACF({k})": acf(panel.loc[panel.symbol == s, "vol7"].dropna().values, 28)[k] for k in (1, 7, 14, 28)}
    for s in SYMBOLS}).T.round(3)
glue_tbl("tbl-acf", acf_tab)
acf_tab
```

```{glue:figure} fig-temporal
:name: fig-temporal
:figwidth: 100%

Izquierda: σ⁽⁷⁾ diaria por activo (eje y: % de retorno diario). Derecha: ACF muestral de σ⁽⁷⁾ hasta 60 días.
```

```{glue:figure} tbl-acf
:name: tbl-acf
:figwidth: 80%

Autocorrelaciones de σ⁽⁷⁾ a 1, 7, 14 y 28 días.
```

**Interpretación de la {numref}`fig-temporal` y la {numref}`tbl-acf`.** La ACF a un día supera 0,91 en
los cuatro activos y cae a 0,34–0,46 a siete días y a 0,15–0,26 a 28 días. Dos lecturas: (i) una parte
de esa persistencia es **mecánica**, pues ventanas consecutivas de 7 días comparten 6 de sus 7 retornos,
y (ii) aun así, la persistencia a 28 días sigue fuera de la banda de confianza, lo que justifica un
modelo que use varias escalas de memoria (la idea del HAR de {cite}`corsi2009har`). La serie
no es estacionaria en media: la volatilidad promedio de BTC pasó de 3,06 (2020) a 3,75 (2021),
2,79 (2022), 1,85 (2023), 2,37 (2024) y 1,85 (2025). Los regímenes (2021 alto, 2023 bajo)
implican que el error de un modelo estimado en 2023 puede no trasladarse a otros años; por eso
la evaluación se hace con orígenes expansivos y por periodos (2025 y 2026) y no con una partición
aleatoria. *No se reportó una prueba ADF/KPSS*: la no estacionariedad se documenta aquí
descriptivamente por promedios anuales (ver Capítulo 8, limitaciones).

```{code-cell} ipython3
# Promedio anual de volatilidad por activo (evidencia descriptiva de cambio de régimen)
anual = (panel.assign(anio=panel["date"].dt.year)
         .groupby(["symbol", "anio"])["vol7"].mean().unstack().loc[SYMBOLS].round(2))
glue_tbl("tbl-anual", anual)
assert abs(anual.loc["BTCUSDT", 2021] - 3.75) < .01 and abs(anual.loc["BTCUSDT", 2023] - 1.85) < .01
anual
```

```{glue:figure} tbl-anual
:name: tbl-anual
:figwidth: 85%

Volatilidad σ⁽⁷⁾ media anual por activo (% de retorno diario).
```

La {numref}`tbl-anual` confirma que 2021 fue el año de mayor volatilidad en los cuatro activos y que 2023 fue el de
menor en BNB, ETH y XRP (en BTC empata con 2025 en 1,85); 2025 queda por debajo de 2024 en BTC y BNB pero por
encima en ETH y XRP.

## 2.5 Correlaciones entre activos

```{code-cell} ipython3
# Matrices de correlación de Pearson: retornos diarios y σ⁽⁷⁾
wide_r = panel.pivot(index="date", columns="symbol", values="ret")[SYMBOLS]
wide_v = panel.pivot(index="date", columns="symbol", values="vol7")[SYMBOLS]
cr, cv = wide_r.corr(), wide_v.corr()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for ax, m, t in zip(axes, [cr, cv], ["Retornos diarios (%)", "Volatilidad σ⁽⁷⁾ (%)"]):
    im = ax.imshow(m.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(4)); ax.set_xticklabels([s[:3] for s in SYMBOLS])
    ax.set_yticks(range(4)); ax.set_yticklabels([s[:3] for s in SYMBOLS]); ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                    color="w" if m.values[i, j] < .75 else "k", fontsize=9)
    ax.set_title(t)
fig.colorbar(im, ax=axes, shrink=.8, label="Correlación de Pearson")
glue_fig("fig-corr", fig)
assert abs(cr.loc["BTCUSDT", "ETHUSDT"] - .832) < .002 and abs(cv.loc["BTCUSDT", "ETHUSDT"] - .853) < .002
```

```{glue:figure} fig-corr
:name: fig-corr
:figwidth: 85%

Correlaciones de Pearson entre activos: retornos (izq.) y σ⁽⁷⁾ (der.); sin unidades, rango 0–1.
```

**Interpretación de la {numref}`fig-corr`.** BTC y ETH son los más sincronizados (0,83 en retornos y 0,85 en
volatilidad) y XRP el más independiente (0,55–0,64). La alta correlación de volatilidades indica un
factor común de riesgo de mercado, pero cada activo se modela por separado (como el estudio original) y las
pruebas del Capítulo 6 promedian pérdidas sobre las 16 configuraciones, por lo que
**no son independientes entre sí**: el cálculo de errores estándar usa HAC por fecha y no trata a los cuatro
activos como cuatro muestras independientes.

## 2.6 Información mutua y colinealidad de las entradas

Este diagnóstico atiende una observación del docente (Capítulo 1). Se calculó únicamente con el
prefijo de desarrollo 2023–2024 (660 orígenes con etiquetas anteriores a 2025) y no se usó para
elegir variables.

```{code-cell} ipython3
# Lectura de los diagnósticos precalculados (src/professor_feedback_diagnostics.py)
vif = read_csv(PF / "diagnostics" / "variance_inflation.csv")
vif = vif[vif.period == "development_2023_2024"]
resumen_vif = (vif.assign(finito=np.isfinite(vif["vif"]))
    .groupby("feature_family")
    .agg(columnas=("feature", "size"), vif_mayor_10=("high_vif_over_10", "sum"),
         vif_infinito=("vif", lambda s: int(np.isinf(s).sum())),
         mediana_vif_finito=("vif", lambda s: s[np.isfinite(s)].median())).round(2))
glue_tbl("tbl-vif", resumen_vif)

mi = read_csv(PF / "diagnostics" / "mutual_information.csv")
mi = mi[(mi.period == "development_2023_2024") & (mi.target_kind == "relative_correction")]
top_mi = (mi.groupby("feature")["mi_nats"].mean().sort_values(ascending=False).head(6).round(3)
          .rename("MI media (nats)").to_frame())
glue_tbl("tbl-mi", top_mi)
display(resumen_vif); top_mi
```

```{glue:figure} tbl-vif
:name: tbl-vif
:figwidth: 90%

Colinealidad (VIF) de las entradas HAR (44 columnas) y de las seis variables diarias contemporáneas, 2023–2024.
```

```{glue:figure} tbl-mi
:name: tbl-mi
:figwidth: 70%

Seis entradas con mayor información mutua media (nats) con el objetivo de corrección relativa, 2023–2024.
```

**Interpretación de las {numref}`tbl-vif` y {numref}`tbl-mi`.** Ambos bloques son colineales, el HAR
más que el diario: en el bloque HAR (44 entradas × 16 diseños = 704 posiciones) 581 tienen VIF > 10, 48 son
infinitas por una dependencia lineal exacta y la mediana finita es 32,5; en el bloque diario de seis
variables (96 posiciones) 53 superan 10, ninguna es infinita y la mediana es 16,9. Las entradas con mayor
información mutua son los términos de decaimiento conocido a 4–7 días (≈0,25–0,31 nats), que comparten
información con el objetivo móvil por construcción. Un VIF alto
**no demuestra** que retirar una columna mejore el modelo: Ridge está diseñado precisamente para
estabilizar diseños colineales {cite}`hoerl1970ridge`, y por eso la regularización es parte del modelo vigente.
La información mutua es una asociación marginal (no un p-valor ni una importancia condicional).

## 2.7 Estacionalidad intradía

```{code-cell} ipython3
# Perfil intradía de retornos al cuadrado normalizados (2023-2024), por hora UTC
prof = read_csv(PF / "diagnostics" / "intraday_minute_profile.csv")
prof = prof[prof.period == "development_2023_2024"]
horaria = (prof.groupby(["symbol", "hour_utc"])["mean_daily_normalized_squared_return"].mean()
           .unstack(0)[SYMBOLS])

fig, ax = plt.subplots(figsize=(9, 3.6))
for s in SYMBOLS:
    ax.plot(horaria.index, horaria[s], marker="o", ms=3, lw=1.4, color=COL[s], label=s)
ax.axhline(1, color="gray", lw=.8, ls="--")
ax.set_xlabel("Hora del día (UTC)"); ax.set_ylabel("Retorno² normalizado (media diaria = 1)")
ax.set_title("Estacionalidad intradía de la volatilidad, 2023–2024"); ax.legend(frameon=False, ncol=4)
fig.tight_layout(); glue_fig("fig-intradia", fig)

resumen_h = read_csv(PF / "diagnostics" / "intraday_summary.csv")
resumen_h = (resumen_h[resumen_h.period == "development_2023_2024"]
             .set_index("symbol")[["hourly_peak_utc", "hourly_trough_utc", "hourly_peak_to_trough_ratio"]]
             .loc[SYMBOLS].round(2))
glue_tbl("tbl-intradia", resumen_h)
resumen_h
```

```{glue:figure} fig-intradia
:name: fig-intradia
:figwidth: 85%

Perfil horario (UTC) del retorno al cuadrado normalizado por su media diaria; la línea punteada marca 1.
```

```{glue:figure} tbl-intradia
:name: tbl-intradia
:figwidth: 80%

Hora pico, hora valle y razón pico/valle de la volatilidad intradía.
```

**Interpretación de la {numref}`fig-intradia` y la {numref}`tbl-intradia`.** La actividad es más intensa
hacia las 14–15 h UTC y mínima a las 4 h UTC, con razones pico/valle de 3,67 (BTC), 3,35 (ETH) y
cercanas a 2,1 (BNB, XRP). Hay, por tanto, un patrón intradía marcado que es coherente con la literatura
de periodicidad intradía {cite}`andersen1997intraday`. Como el objetivo del estudio es diario, el
patrón se promedia dentro de cada día y su efecto sobre los pronósticos se documenta como diagnóstico; **no se
demostró** que desestacionalizar mejore el pronóstico.
