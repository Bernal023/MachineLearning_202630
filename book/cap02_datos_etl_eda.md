# Capítulo 2 · Datos: ETL y EDA

Este capítulo sigue el orden de la presentación del proyecto: fuente y cobertura, ETL, exploración de Bitcoin (precios, retornos y
diagnóstico temporal), distribución de retornos y del objetivo, correlaciones, dependencia temporal y decisiones metodológicas.

```{code-cell} ipython3
:tags: [remove-cell]
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from jbook_utils import (SYMBOLS, PF, load_daily_panel, read_csv, acf_gaps, longest_segment,
                         weekday_wald, glue_fig, glue_tbl)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = dict(zip(SYMBOLS, ["#f7931a", "#627eea", "#c9a227", "#23292f"]))
```

## 2.1 Fuente, cobertura y ETL

- **Fuente:** Binance Spot, BTC, ETH, BNB y XRP frente a USDT.
- **Cobertura:** cierres de **un minuto**, 2023–2025, **1.096 fechas diarias** por activo. Los huecos se conservan: **no se interpolan precios**.
- **Tres vistas de la evidencia:** (i) el EDA, del 01/01/2023 al 24/12/2024; (ii) la comparación principal, del 01/01 al 24/12/2025 (358 orígenes
  diarios comunes); (iii) el informe complementario de intervalos de diferencias de MSE.
- **Frecuencias:** adquisición cada minuto; emisión del pronóstico y objetivo, diarios.

El pipeline (`src/minute_experiment.py`, `src/optimize_minute_svr.py`) resume cada día de 1.440 minutos en un cierre diario UTC
y en características diarias. Este libro parte del cierre diario (`data/processed/daily_2020_2025.csv`, restringido a 2023–2025).

```{code-cell} ipython3
# Cobertura del cierre diario 2023-2025 (ETL: sin interpolación de precios)
panel = load_daily_panel("2023-01-01", "2025-12-31")
calidad = panel.groupby("symbol").agg(
    fechas=("date", "size"), primera=("date", lambda s: s.min().date()), ultima=("date", lambda s: s.max().date()),
    dias_incompletos=("complete", lambda s: int((~s).sum())), cierres_faltantes=("close", lambda s: int(s.isna().sum())),
).loc[SYMBOLS]
assert (calidad["fechas"] == 1096).all()
glue_tbl("tbl-calidad", calidad)
calidad
```

```{glue:figure} tbl-calidad
:name: tbl-calidad
:figwidth: 95%

Cobertura del cierre diario por activo, 2023-2025: fechas, días incompletos y cierres faltantes.
```

**Interpretación de la {numref}`tbl-calidad`.** Cada activo tiene las 1.096 fechas de calendario (365 + 366 + 365). Un cierre faltante
corresponde a un día incompleto (menos de 1.440 minutos válidos): queda como valor faltante y las muestras que lo contienen se excluyen, sin imputar.

### Frecuencia, ventanas y horizonte

- **Ventana de entrada** $L\in\{7,14,21,28\}$ días, hasta el día $t$ inclusive. Para el SVR hay seis variables por día y siete referencias
  de salida conocida, es decir, $6L+7$ entradas.
- **Ventana objetivo** $w\in\{7,14,21,28\}$ retornos diarios. Con $r_t=100\,[\ln P_t-\ln P_{t-1}]$ y $P_t$ el cierre diario, la volatilidad es
  poblacional, en porcentaje y sin anualizar:

```{math}
:label: eq-vol
\sigma_t^{(w)}=\operatorname{std}_{\text{pob}}\!\left(r_{t-w+1},\dots,r_t\right)
=\sqrt{\frac{1}{w}\sum_{j=0}^{w-1}\left(r_{t-j}-\bar r_t^{(w)}\right)^2}.
```

- **Horizonte:** siete días futuros, $\mathbf y_t^{(w)}=\big(\sigma_{t+1}^{(w)},\dots,\sigma_{t+7}^{(w)}\big)$; cada salida conserva su ventana $w$.

Ejemplo: con $L=14$ se usan 14 días de características; con $w=7$ la volatilidad se mide con siete retornos diarios; se pronostican $t+1,\dots,t+7$.
Una ventana solo existe si contiene $w$ retornos consecutivos válidos, de modo que **ninguna ventana cruza un hueco**.

## 2.2 Precios y retornos del periodo de desarrollo (BTC)

El filtro inicial del dashboard es BTC con 724 fechas de calendario del 01/01/2023 al 24/12/2024.

```{code-cell} ipython3
# Vista del EDA: BTC, 01/01/2023-24/12/2024 (w = 7, h = 1)
eda = load_daily_panel("2023-01-01", "2024-12-24")
btc = eda[eda.symbol == "BTCUSDT"].set_index("date")
assert (len(btc), btc.close.notna().sum(), btc.ret.notna().sum()) == (724, 723, 721)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
axes[0].plot(btc.index, btc.close / 1000, color=COL["BTCUSDT"], lw=1)
axes[0].set_xlabel("Fecha UTC"); axes[0].set_ylabel("Miles de USDT"); axes[0].set_title("Cierre diario de BTC")
axes[1].plot(btc.index, btc.ret, color="#1f4e79", lw=.7)
axes[1].set_xlabel("Fecha UTC"); axes[1].set_ylabel("Retorno (%)"); axes[1].set_title("Retorno logarítmico diario")
fig.tight_layout(); glue_fig("fig-precio-ret", fig)
```

```{glue:figure} fig-precio-ret
:name: fig-precio-ret
:figwidth: 100%

Cierre diario de BTC (izq., miles de USDT) y retorno logarítmico diario (der., %), 01/01/2023–24/12/2024.
```

**Interpretación de la {numref}`fig-precio-ret`.** Hay 724 fechas, 723 cierres y 721 retornos disponibles (un cierre falta y los retornos
que lo usan no existen). El precio describe el **nivel**; los retornos permiten comparar la **magnitud** de los cambios entre fechas, y por eso el
modelado trabaja con retornos y no con precios.

## 2.3 Bitcoin: pasos del diagnóstico temporal

El diagnóstico se hizo sobre BTC/USDT diario del 01/01/2023 al 24/12/2024, sin interpolar precios, en cuatro pasos.

**Paso 1. Preparar la muestra.** Se toma el mayor tramo continuo de cada serie.

```{code-cell} ipython3
# Paso 1: mayor tramo continuo de cada serie examinada
btc = btc.assign(abs_ret=btc.ret.abs(), logprice=np.log(btc.close))
filas = {"Precio y logaritmo del precio": longest_segment(btc.close), "Retorno r_t y magnitud |r_t|": longest_segment(btc.ret),
         "Volatilidad σ(21)_t": longest_segment(btc.vol21)}
tramo = pd.DataFrame(filas, index=["n", "inicio", "fin"]).T
assert tramo.loc["Precio y logaritmo del precio", "n"] == 641 and tramo.loc["Retorno r_t y magnitud |r_t|", "n"] == 640
assert tramo.loc["Volatilidad σ(21)_t", "n"] == 620
glue_tbl("tbl-tramo", tramo)
tramo
```

```{glue:figure} tbl-tramo
:name: tbl-tramo
:figwidth: 85%

Series examinadas, número de observaciones y mayor tramo continuo.
```

La {numref}`tbl-tramo` muestra que el diagnóstico usa 641 precios, 640 retornos y 620 volatilidades de 21 días. El periodo semanal (7 días)
de la estacionalidad es distinto de la ventana de volatilidad de 21: son dos preguntas diferentes, y **un patrón semanal no es una raíz unitaria**.

**Paso 2. Separar tendencia, componente semanal y residuo.** Se aplicó una descomposición STL robusta (*Seasonal–Trend decomposition using LOESS*,
con LOESS = suavizado por regresión local) a $|r_t|=T_t+S_t+R_t$, con periodo de 7 días y 640 observaciones (26/03/2023–24/12/2024).
$T_t$ es la tendencia, $S_t$ el componente semanal y $R_t$ el residuo. La descomposición **estima** un patrón semanal; su existencia se contrasta por separado
(Paso 3). *El código de la descomposición se ejecutó con `statsmodels` fuera de este libro (carpeta `outputs/presentation_btc_diagnostics_20261019/`, no incluida en el repositorio entregado).*

**Paso 3. Contrastar diferencias entre los siete días de la semana** en $r_t$ y en $|r_t|$ con un contraste de Wald conjunto, $H_0$: las siete
medias por día de la semana son iguales. La covarianza es HAC (Newey–West, núcleo de Bartlett, 28 rezagos), que es robusta a heterocedasticidad y
autocorrelación, y el nivel se ajusta por Holm solo sobre estos dos contrastes (Holm2).

```{code-cell} ipython3
# Paso 3: Wald conjunto de igualdad de medias por día de la semana (640 fechas comunes)
r = btc.ret.loc["2023-03-26":"2024-12-24"]
res = []
for nombre, y in [("Retorno", r), ("Magnitud absoluta", r.abs())]:
    W, p = weekday_wald(y, lags=28)
    res.append({"serie": nombre, "W (chi2, 6 g.l.)": W, "p_bruto": p})
wald = pd.DataFrame(res).set_index("serie")
p = wald["p_bruto"].to_numpy(); order = np.argsort(p)
wald["p_Holm2"] = [min(1, max(p[order[0]] * 2, 0)) if i == order[0] else min(1, max(p[order[0]] * 2, p[i])) for i in range(2)]
wald["decisión_5%"] = np.where(wald["p_Holm2"] < .05, "Rechazar igualdad", "No rechazar igualdad")
assert len(r) == 640 and abs(wald.loc["Retorno", "W (chi2, 6 g.l.)"] - 3.973) < .01
assert abs(wald.loc["Magnitud absoluta", "W (chi2, 6 g.l.)"] - 143.061) < .05 and abs(wald.loc["Retorno", "p_bruto"] - .6803) < 1e-3

dias = ["lun.", "mar.", "mié.", "jue.", "vie.", "sáb.", "dom."]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, y, t, u in zip(axes, [r, r.abs()], ["Retorno medio por día", "Magnitud absoluta media por día"], ["Retorno (%)", "Magnitud (%)"]):
    ax.bar(dias, y.groupby(y.index.dayofweek).mean().values, color="#1f4e79"); ax.set_title(t); ax.set_ylabel(u)
    ax.set_xlabel("Día de la semana")
fig.tight_layout(); glue_fig("fig-semana", fig)
glue_tbl("tbl-wald", wald.round(6))
wald
```

```{glue:figure} tbl-wald
:name: tbl-wald
:figwidth: 95%

Contraste de Wald conjunto (HAC Newey–West, Bartlett, 28 rezagos) de igualdad de las siete medias por día de la semana.
```

```{glue:figure} fig-semana
:name: fig-semana
:figwidth: 95%

Retorno medio (izq., %) y magnitud absoluta media (der., %) por día de la semana, BTC, 26/03/2023–24/12/2024.
```

**Interpretación de la {numref}`tbl-wald` y la {numref}`fig-semana`.** El calendario semanal aparece en la **magnitud** media
($W\approx143{,}06$, $p\approx2{,}3\times10^{-28}$, rechazo al 5 % incluso tras Holm2) pero **no se detecta en el retorno medio** del tramo completo
($W\approx3{,}97$, $p\approx0{,}68$). No rechazar no demuestra igualdad. El contraste es aproximado (seis indicadores semanales y constante).

**Paso 4. Contrastar raíz unitaria y estacionariedad** en precio, logaritmo del precio, retornos y volatilidad con dos pruebas de hipótesis distintas:
ADF (*Augmented Dickey–Fuller*; $H_0$: raíz unitaria) y KPSS (*Kwiatkowski–Phillips–Schmidt–Shin*; $H_0$: estacionariedad alrededor de un nivel);
en ambas, $p<0{,}05$ rechaza su hipótesis nula. Los resultados se leen conjuntamente.

```{code-cell} ipython3
# Paso 4: resultados ADF/KPSS TRANSCRITOS de la presentación (constante sin tendencia; rezagos ADF por AIC; KPSS con rezagos automáticos)
estac = pd.DataFrame({
    "n": [641, 641, 640, 620],
    "p_ADF": ["0,9896", "0,9407", "< 0,001", "0,0026"],
    "p_KPSS": ["< 0,01", "< 0,01", "> 0,10", "< 0,01"],
    "lectura_al_5%": ["No estacionario en nivel", "No estacionario en nivel", "ADF rechaza; KPSS no rechaza", "Ambas rechazan: discordancia"],
}, index=["Precio", "Logaritmo del precio", "Retorno", "Volatilidad w = 21"])
glue_tbl("tbl-estac", estac)
estac
```

```{glue:figure} tbl-estac
:name: tbl-estac
:figwidth: 95%

ADF y KPSS por serie (valores de la presentación; las cotas «<» y «>» no son valores exactos).
```

```{admonition} Procedencia de la {numref}`tbl-estac`
:class: note
Estos valores (y la descomposición STL) se **transcriben de la presentación**; no se recalculan en el libro porque requieren `statsmodels` y los resultados
guardados (`outputs/presentation_btc_diagnostics_20261019/`) no están en el repositorio entregado. El contraste de Wald de la {numref}`tbl-wald` sí se recalcula aquí
y reproduce los valores de la presentación.
```

**Lectura conjunta y decisión metodológica.** El precio y el logaritmo del precio **no deben tratarse como estacionarios** en este tramo. En los retornos se
rechaza raíz unitaria (ADF) y KPSS no rechaza estacionariedad, pero eso **no demuestra estabilidad de toda su distribución**. La magnitud del retorno muestra diferencias
semanales en su media, así que conviene evaluar el calendario al modelar riesgo. En la **volatilidad de 21 días** los resultados son discordantes (ambas pruebas rechazan), y con
tendencia lineal la discrepancia persiste al 5 % (p ADF = 0,0050; p KPSS = 0,0385); por eso hay que estudiar la estabilidad por periodos y la especificación antes de
asumir estacionariedad. Se hicieron comprobaciones de sensibilidad (variar rezagos, añadir tendencia lineal y controlar por mes en el contraste semanal). La evidencia distingue nivel, retorno
y magnitud: **no corresponde declarar a todo Bitcoin estacionario o sin estacionalidad**. Es un diagnóstico retrospectivo que no demuestra predictibilidad ni selecciona un modelo.

## 2.4 Distribución de retornos y de la característica relativa

```{code-cell} ipython3
# Retornos y la característica r_t / b_t, con b_t = σ(7)_t (volatilidad actual)
ret = btc.ret.dropna()
feat = (btc.ret / btc.vol7).replace([np.inf, -np.inf], np.nan).dropna()
est = pd.Series({"n": len(ret), "media_%": ret.mean(), "desv_%": ret.std(), "asimetría": stats.skew(ret, bias=False),
                 "exceso_curtosis": stats.kurtosis(ret, bias=False)}, name="retornos BTC").round(4).to_frame()
assert abs(ret.mean() - .2512) < 5e-4 and abs(ret.std() - 2.5365) < 5e-4 and abs(est.loc["asimetría"].iloc[0] - .454) < 1e-3
assert abs(est.loc["exceso_curtosis"].iloc[0] - 2.344) < 1e-3

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].hist(ret, bins=55, color="#1f4e79"); axes[0].set_xlabel("Retorno diario (%)"); axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribución de retornos")
axes[1].hist(feat, bins=45, color="#8fb4d9"); axes[1].set_xlabel("r_t / b_t"); axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Característica: retorno diario relativo")
fig.tight_layout(); glue_fig("fig-hist-ret", fig)
glue_tbl("tbl-ret", est)
est
```

```{glue:figure} tbl-ret
:name: tbl-ret
:figwidth: 60%

Estadísticos de los retornos diarios de BTC, 01/01/2023–24/12/2024.
```

```{glue:figure} fig-hist-ret
:name: fig-hist-ret
:figwidth: 95%

Histograma de retornos diarios (izq., %) y de la característica $r_t/b_t$ (der., sin unidades; $b_t=\sigma^{(7)}_t$).
```

**Interpretación de la {numref}`tbl-ret` y la {numref}`fig-hist-ret`.** Los retornos tienen media 0,2512 %, desviación 2,5365 %, asimetría 0,454 y exceso de curtosis
2,344 (colas más pesadas que la normal). Los **valores extremos permanecen en los datos** y en el cálculo de las estadísticas: eliminarlos borraría los episodios de volatilidad que se
quieren pronosticar. La característica $r_t/b_t$ expresa el retorno del día en unidades de la volatilidad actual.

## 2.5 Volatilidad histórica y objetivo futuro

```{code-cell} ipython3
# σ(7)_t histórica frente a σ(7)_{t+1} futura (ventana de 7 retornos, horizonte de 1 día)
hist, fut = btc.vol7, btc.vol7.shift(-1)
assert hist.notna().sum() == 709 and fut.notna().sum() == 709
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.plot(hist.index, hist, lw=1, label="Histórica σ(7)_t"); ax.plot(fut.index, fut, lw=1, label="Futura σ(7)_{t+1}")
ax.set_xlabel("Fecha UTC (origen)"); ax.set_ylabel("Volatilidad (puntos porcentuales)")
ax.set_title("Volatilidad de BTC: histórica y futura"); ax.legend(frameon=False)
fig.tight_layout(); glue_fig("fig-hist-fut", fig)
```

```{glue:figure} fig-hist-fut
:name: fig-hist-fut
:figwidth: 90%

Volatilidad histórica $\sigma^{(7)}_t$ y futura $\sigma^{(7)}_{t+1}$ de BTC (puntos porcentuales); cada serie tiene 709 valores disponibles.
```

La {numref}`fig-hist-fut` muestra que las dos líneas casi se superponen desplazadas un día: **las ventanas sucesivas comparten retornos**, por lo que la comparación exige
respetar el orden temporal y no mezclar fechas en la validación.

## 2.6 Distribución del objetivo entre activos

```{code-cell} ipython3
# Objetivo σ(7)_{t+1} para los cuatro activos, mismo periodo (709 etiquetas por activo)
objetivo = {s: eda[eda.symbol == s].set_index("date").vol7.shift(-1).dropna() for s in SYMBOLS}
tabla = pd.DataFrame({s: {"n": len(v), "media": v.mean(), "mediana": v.median(), "desv": v.std()} for s, v in objetivo.items()}).T.round(2)
assert (tabla["n"] == 709).all() and tabla.loc["BTCUSDT", "media"] == 2.13 and tabla.loc["XRPUSDT", "desv"] == 2.48

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for s, v in objetivo.items():
    axes[0].hist(v.clip(upper=20), bins=45, histtype="step", lw=1.5, color=COL[s], label=s)
axes[0].set_xlabel("Volatilidad futura (puntos porcentuales)"); axes[0].set_ylabel("Frecuencia"); axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("Distribución de volatilidad futura")
axes[1].boxplot([objetivo[s] for s in SYMBOLS], labels=[s[:3] for s in SYMBOLS], showfliers=False)
axes[1].set_ylabel("Volatilidad futura (puntos porcentuales)"); axes[1].set_title("Objetivo: comparación entre activos")
fig.tight_layout(); glue_fig("fig-objetivo", fig)
glue_tbl("tbl-objetivo", tabla)
tabla
```

```{glue:figure} tbl-objetivo
:name: tbl-objetivo
:figwidth: 70%

Objetivo $\sigma^{(7)}_{t+1}$ por activo: observaciones, media, mediana y desviación (puntos porcentuales, sin anualizar).
```

```{glue:figure} fig-objetivo
:name: fig-objetivo
:figwidth: 95%

Histograma (izq.) y diagrama de cajas de Tukey sin atípicos (der.) del objetivo por activo.
```

**Interpretación de la {numref}`tbl-objetivo` y la {numref}`fig-objetivo`.** El nivel y la dispersión de la volatilidad varían entre activos del mismo periodo: BTC es el más
estable (media 2,13; desviación 0,97) y XRP el más volátil y disperso (media 2,93; desviación 2,48, con una mediana de 2,23 muy inferior a la media por su cola
derecha). Las cajas no muestran los puntos atípicos, pero **todas las muestras intervienen en el cálculo**.

## 2.7 Correlaciones entre entradas y volatilidad

```{code-cell} ipython3
# Matriz de Spearman de nueve variables (BTC, w = 7, h = 1). Valores TRANSCRITOS de la presentación (dashboard, pestaña EDA).
v = ["r/b", "r²/b²", "RV/b", "abs/b", "down/b", "max/b", "dec1", "σt", "σt+1"]
M = np.array([
 [1.00, .12, -.04, -.03, -.14, -.03, .03, .02, .02], [.12, 1.00, .46, .42, .43, .42, .10, -.04, .02],
 [-.04, .46, 1.00, .96, .99, .79, .11, -.49, -.36], [-.03, .42, .96, 1.00, .96, .66, .10, -.43, -.31],
 [-.14, .43, .99, .96, 1.00, .78, .10, -.48, -.35], [-.03, .42, .79, .66, .78, 1.00, .12, -.40, -.29],
 [.03, .10, .11, .10, .10, .12, 1.00, -.02, .23], [.02, -.04, -.49, -.43, -.48, -.40, -.02, 1.00, .88],
 [.02, .02, -.36, -.31, -.35, -.29, .23, .88, 1.00]])
# Verificación: las celdas que solo dependen de retornos y volatilidad se recalculan con los datos
b_t = btc.vol7; x = (btc.ret / b_t).replace([np.inf, -np.inf], np.nan)
chk = {("r/b", "r²/b²"): (x, x ** 2), ("r/b", "σt"): (x, b_t), ("r/b", "σt+1"): (x, b_t.shift(-1)),
       ("r²/b²", "σt"): (x ** 2, b_t), ("r²/b²", "σt+1"): (x ** 2, b_t.shift(-1)), ("σt", "σt+1"): (b_t, b_t.shift(-1))}
for (a, c), (u, w) in chk.items():
    ok = u.notna() & w.notna()
    assert abs(stats.spearmanr(u[ok], w[ok]).statistic - M[v.index(a), v.index(c)]) < .015, (a, c)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r"); ax.grid(False)
ax.set_xticks(range(9)); ax.set_xticklabels(v, rotation=45, ha="right"); ax.set_yticks(range(9)); ax.set_yticklabels(v)
for i in range(9):
    for j in range(9):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if abs(M[i, j]) > .6 else "k")
fig.colorbar(im, ax=ax, label="Correlación de Spearman"); ax.set_title("Spearman de nueve variables (BTC, w = 7, h = 1)")
fig.tight_layout(); glue_fig("fig-spearman", fig)
```

```{glue:figure} fig-spearman
:name: fig-spearman
:figwidth: 70%

Matriz de Spearman de nueve variables (BTC, $w=7$, $h=1$; sin unidades). `dec1`: decaimiento conocido; $b_t=\sigma^{(7)}_t$; RV, abs, down y max: resúmenes intradía.
```

**Interpretación de la {numref}`fig-spearman`.** Los resúmenes intradía (RV, abs, down y max normalizados) están **muy correlacionados entre sí** (0,66–0,99) y se asocian de forma
**negativa** con la volatilidad futura (−0,29 a −0,36); la volatilidad actual y la futura tienen $\rho=0{,}88$, y `dec1` es la entrada con mayor asociación positiva con
$\sigma_{t+1}$ (0,23). La matriz **describe asociación**; no demuestra causalidad ni aporte predictivo incremental. Cada par usa sus observaciones disponibles. *(Nota: las seis celdas
que solo usan retornos y volatilidad se recalculan en el libro y coinciden con la presentación; las que usan resúmenes intradía se transcriben porque requieren los datos por minuto.)*

```{code-cell} ipython3
# Característica seleccionada frente al objetivo: r_t/b_t vs σ(7)_{t+1}
ok = x.notna() & b_t.shift(-1).notna()
rho = stats.spearmanr(x[ok], b_t.shift(-1)[ok])
assert ok.sum() == 707 and abs(rho.statistic - .016) < 1e-3
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(x[ok], b_t.shift(-1)[ok], s=8, alpha=.6, color="#1f4e79")
ax.set_xlabel("r_t / b_t"); ax.set_ylabel("Volatilidad futura (puntos porcentuales)")
ax.set_title(f"Retorno relativo frente a volatilidad futura (ρ = {rho.statistic:.3f}, n = {ok.sum()})")
fig.tight_layout(); glue_fig("fig-scatter", fig)
```

```{glue:figure} fig-scatter
:name: fig-scatter
:figwidth: 70%

Retorno diario relativo $r_t/b_t$ (eje x, sin unidades) frente a la volatilidad futura (eje y, puntos porcentuales); 707 pares completos de BTC.
```

La {numref}`fig-scatter` muestra una relación **débil** en el filtro inicial: Spearman $\rho=0{,}016$ con 707 pares. No se ajusta aquí una regresión; la débil asociación marginal
de esta característica no impide que aporte en combinación con otras.

## 2.8 Dependencia temporal: retornos, retornos al cuadrado y volatilidad

```{code-cell} ipython3
# ACF con calendario preservado y tratamiento conservador de huecos (rezagos 0-30)
a_ret, a_r2, a_vol = acf_gaps(btc.ret, 30), acf_gaps(btc.ret ** 2, 30), acf_gaps(btc.vol7, 30)
assert abs(a_ret[1] + .024) < 1e-3 and abs(a_r2[1] - .113) < 1e-3 and abs(a_vol[1] - .885) < 1e-3
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, a, t in zip(axes, [a_ret, a_r2, a_vol], ["ACF de retornos", "ACF de retornos al cuadrado", "ACF de volatilidad histórica σ(7)"]):
    ax.bar(range(31), a, color="#1f4e79"); ax.set_xlabel("Rezago (días)"); ax.set_title(t)
axes[0].set_ylabel("ACF")
fig.tight_layout(); glue_fig("fig-acf", fig)
acf_tab = pd.DataFrame({"rezago 1": [a_ret[1], a_r2[1], a_vol[1]], "rezago 7": [a_ret[7], a_r2[7], a_vol[7]],
                        "rezago 28": [a_ret[28], a_r2[28], a_vol[28]]}, index=["retornos", "retornos²", "σ(7)"]).round(3)
glue_tbl("tbl-acf", acf_tab)
acf_tab
```

```{glue:figure} tbl-acf
:name: tbl-acf
:figwidth: 70%

Autocorrelaciones de retornos, retornos al cuadrado y volatilidad histórica (BTC).
```

```{glue:figure} fig-acf
:name: fig-acf
:figwidth: 100%

ACF (eje y, sin unidades) de retornos, retornos al cuadrado y $\sigma^{(7)}$ de BTC hasta 30 rezagos (eje x, días).
```

**Interpretación de la {numref}`tbl-acf` y la {numref}`fig-acf`.** En el rezago 1, los retornos casi no tienen autocorrelación (−0,024), los retornos al cuadrado sí (0,113) y la
volatilidad histórica es muy persistente (0,885). Estudiar retornos y magnitudes permite **distinguir formas de dependencia**: el nivel de los retornos es casi impredecible, pero su magnitud
no. Una parte de la persistencia de $\sigma^{(7)}$ es mecánica (ventanas sucesivas comparten seis de siete retornos).

## 2.9 Del diagnóstico a la decisión metodológica

| Aspecto explorado | Decisión metodológica |
|---|---|
| Retornos y variación intradía | Construir características diarias a partir de los cierres de un minuto. |
| Asimetría y movimientos extremos | Reportar RMSE y MAE; conservar movimientos reales y comparar sus distribuciones. |
| Datos faltantes | Conservar el calendario y excluir muestras afectadas, sin interpolar precios. |
| Dependencia en la ACF | Usar validación creciente y varianzas HAC para las diferencias de pérdida. |

La exploración orienta las decisiones; por sí sola no demuestra capacidad predictiva ni causalidad.

## 2.10 Diagnósticos adicionales atendidos en las correcciones

Estos dos diagnósticos responden a observaciones del docente (Capítulo 1) y se calcularon solo con el prefijo de desarrollo 2023–2024.

```{code-cell} ipython3
# Colinealidad (VIF) e información mutua de las entradas HAR (src/professor_feedback_diagnostics.py)
vif = read_csv(PF / "diagnostics" / "variance_inflation.csv")
vif = vif[vif.period == "development_2023_2024"]
resumen_vif = (vif.groupby("feature_family")
    .agg(posiciones=("feature", "size"), vif_mayor_10=("high_vif_over_10", "sum"),
         vif_infinito=("vif", lambda s: int(np.isinf(s).sum())),
         mediana_vif_finito=("vif", lambda s: s[np.isfinite(s)].median())).round(2))
assert resumen_vif.loc["HAR44", "vif_mayor_10"] == 581 and resumen_vif.loc["daily6", "vif_mayor_10"] == 53
glue_tbl("tbl-vif", resumen_vif)
resumen_vif
```

```{glue:figure} tbl-vif
:name: tbl-vif
:figwidth: 90%

Colinealidad (VIF) del bloque HAR (44 entradas × 16 diseños) y del bloque diario de seis variables, 2023–2024.
```

La {numref}`tbl-vif` muestra que ambos bloques son colineales, el HAR más que el diario: 581 de 704 posiciones tienen VIF > 10 (48 infinitas por una dependencia lineal exacta;
mediana finita 32,5) frente a 53 de 96 en el bloque diario (mediana 16,9). Un VIF alto **no demuestra** que retirar una columna mejore el modelo: Ridge está diseñado para
estabilizar diseños colineales {cite}`hoerl1970ridge`, y por eso la regularización es parte del modelo.

```{code-cell} ipython3
# Información mutua (nats) con el objetivo de corrección relativa, 2023-2024
mi = read_csv(PF / "diagnostics" / "mutual_information.csv")
mi = mi[(mi.period == "development_2023_2024") & (mi.target_kind == "relative_correction")]
top_mi = mi.groupby("feature")["mi_nats"].mean().sort_values(ascending=False).head(6).round(3).rename("MI media (nats)").to_frame()
assert top_mi.index[0] == "known_decay_h4" and abs(top_mi.iloc[0, 0] - .308) < 1e-3
glue_tbl("tbl-mi", top_mi)
top_mi
```

```{glue:figure} tbl-mi
:name: tbl-mi
:figwidth: 65%

Seis entradas con mayor información mutua media (nats) con el objetivo de corrección relativa, 2023–2024.
```

La {numref}`tbl-mi` muestra que las entradas con mayor información mutua {cite}`cover2006information` son los términos de decaimiento conocido a 4–7 días (≈0,25–0,31 nats), que comparten información con el objetivo móvil por
construcción. La información mutua es una asociación marginal, no una importancia condicional.

```{code-cell} ipython3
# Estacionalidad intradía: perfil horario del retorno² normalizado (2023-2024)
prof = read_csv(PF / "diagnostics" / "intraday_minute_profile.csv")
prof = prof[prof.period == "development_2023_2024"]
horaria = prof.groupby(["symbol", "hour_utc"])["mean_daily_normalized_squared_return"].mean().unstack(0)[SYMBOLS]
res_h = (read_csv(PF / "diagnostics" / "intraday_summary.csv").query("period == 'development_2023_2024'")
         .set_index("symbol")[["hourly_peak_utc", "hourly_trough_utc", "hourly_peak_to_trough_ratio"]].loc[SYMBOLS].round(2))
assert res_h.loc["BTCUSDT", "hourly_peak_to_trough_ratio"] == 3.67
fig, ax = plt.subplots(figsize=(8, 3.3))
for s in SYMBOLS:
    ax.plot(horaria.index, horaria[s], marker="o", ms=3, lw=1.3, color=COL[s], label=s)
ax.axhline(1, color="gray", ls="--", lw=.8); ax.set_xlabel("Hora del día (UTC)")
ax.set_ylabel("Retorno² normalizado (media diaria = 1)"); ax.set_title("Estacionalidad intradía, 2023–2024"); ax.legend(frameon=False, ncol=4)
fig.tight_layout(); glue_fig("fig-intradia", fig); glue_tbl("tbl-intradia", res_h)
res_h
```

```{glue:figure} fig-intradia
:name: fig-intradia
:figwidth: 80%

Perfil horario (UTC) del retorno al cuadrado normalizado por su media diaria; la línea punteada marca 1.
```

```{glue:figure} tbl-intradia
:name: tbl-intradia
:figwidth: 80%

Hora pico, hora valle y razón pico/valle de la volatilidad intradía.
```

La {numref}`fig-intradia` y la {numref}`tbl-intradia` muestran un patrón intradía marcado: pico hacia las 14–15 h UTC, valle a las 4 h UTC y razones pico/valle de 3,67 (BTC), 3,35 (ETH) y
≈ 2,1 (BNB y XRP), coherente con la literatura de periodicidad intradía {cite}`andersen1997intraday`. Como el objetivo es diario, el patrón se promedia dentro de cada día.
