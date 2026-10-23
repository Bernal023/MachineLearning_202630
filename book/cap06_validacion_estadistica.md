# Capítulo 6 · Validación estadística de superioridad

Este capítulo responde si la ventaja de HAR-Ridge + XGBoost sobre sus rivales es estadísticamente sólida. **Resultado principal:** bajo el protocolo principal (Grid original, 2025, $w=21$) el modelo es **superior** a los siete rivales: 7/7 contrastes DM–Holm7 favorables y 7/7 intervalos de ganancia positivos; cuando los rivales se reoptimizan, el respaldo se mantiene frente a algunos rivales y no frente a otros (§6.8). Sigue el orden de la presentación: contraste de Diebold–Mariano,
ajuste de Holm, resultados por búsqueda, resumen por ventana, tamaño de la mejora con intervalos y conclusión; se añaden el tamaño del efecto estandarizado y las pruebas MCS y SPA.

```{code-cell} ipython3
:tags: [remove-cell]
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from jbook_utils import (R, SEED, HAR, FAMILIES, SEARCHES, read_csv, dm_hln, hac_lag, holm, gain_ci, search_losses,
                         inference_table, stationary_bootstrap_indices, glue_fig, glue_tbl)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
EV = R / "evaluation"
W21 = {s: inference_table(21, s) for s in SEARCHES}      # w = 21: DM-HLN, Holm7 e IC95 % por búsqueda
```

## 6.1 Diebold–Mariano: hipótesis y estadístico

Sean A = HAR-Ridge + XGBoost y B un rival; se comparan **las mismas fechas y objetivos**. Con $\ell_{j,t}$ el error cuadrado medio del modelo $j$ en la fecha $t$ sobre cuatro activos y siete horizontes,

```{math}
:label: eq-dm
d_t=\ell_{A,t}-\ell_{B,t},\qquad \bar d=\frac1n\sum_{t=1}^{n}d_t,\qquad n=358 .
```

El contraste es **bilateral**, $H_0:\ E[d_t]=0$ (igual pérdida esperada) frente a $H_1:\ E[d_t]\neq0$; $\mathrm{DM}<0$ favorece a A y $\mathrm{DM}>0$ a B {cite}`diebold1995comparing`:

```{math}
:label: eq-dm-hln
\mathrm{DM}=\frac{\bar d}{\widehat{\mathrm{se}}_{\mathrm{HAC}}(\bar d)},\qquad
\mathrm{DM}_{\mathrm{HLN}}\simeq0{,}98184\,\mathrm{DM},\qquad
p_{\text{bruto}}=2\big[1-F_{t_{357}}(|\mathrm{DM}_{\mathrm{HLN}}|)\big].
```

El error estándar es HAC (Newey–West {cite}`newey1987simple`, núcleo de Bartlett) para incorporar la dependencia temporal, con $w+5$ rezagos (26 para $w=21$). El ajuste HLN
{cite}`harvey1997testing` con horizonte $h=7$ multiplica por $\sqrt{[n+1-2h+h(h-1)/n]/n}$; $F_{t_{357}}$ es la distribución acumulada de Student. El valor $p$ mide cuán extremo es el
estadístico bajo $H_0$; **no** es la probabilidad de que $H_0$ sea cierta.

```{code-cell} ipython3
# Factor HLN y verificación de la implementación contra el archivo del repositorio (Random Search, w = 21)
n, h = 358, 7
factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
assert abs(factor - 0.98184) < 5e-6
ref = read_csv(R / "dm_original_vs_optimized_w21_2025" / "dm_random.csv")
ref["rival"] = ref["model_b"].str.replace(r" \[random\]", "", regex=True).map(FAMILIES)
mio = W21["random"]
comp = ref.set_index("rival").loc[mio.index, ["dm", "p", "p_holm7"]]
assert np.allclose(comp["dm"], mio["DM_HLN"]) and np.allclose(comp["p"], mio["p_bruto"]) and np.allclose(comp["p_holm7"], mio["p_Holm7"])
print(f"Factor HLN = {factor:.5f}. La implementación del libro reproduce dm_random.csv (DM, p y p Holm7) para los 7 rivales.")
```

## 6.2 Holm: qué significa $p_{\text{Holm}}$ y cómo se calcula

$p_{\text{Holm}}$ es el valor $p$ ajustado por comparaciones múltiples {cite}`holm1979simple`. Con valores $p$ individuales válidos, controla al 5 % el riesgo de rechazar **al menos una** hipótesis nula
verdadera en la familia declarada. Se ordenan los $m$ valores, $p_{(1)}\le\dots\le p_{(m)}$, se multiplica, se toma el máximo acumulado y se limita a 1:

```{math}
:label: eq-holm
p^{\text{Holm}}_{(i)}=\min\Big\{1,\ \max_{1\le j\le i}\big[(m-j+1)\,p_{(j)}\big]\Big\}.
```

```{code-cell} ipython3
# Ejemplo real: Holm6 con los seis rivales históricos (Grid original), w = 21
g = W21["grid"].drop(index="Decision Tree*")
orden = g.sort_values("p_bruto")
m = len(orden)
ejemplo = pd.DataFrame({"p_bruto": orden["p_bruto"],
                        "multiplicador": [m - j for j in range(m)],
                        "producto": [(m - j) * p for j, p in enumerate(orden["p_bruto"])],
                        "p_Holm6": holm(orden["p_bruto"].values)})
assert abs(ejemplo["producto"].iloc[0] - 0.000147086) < 2e-9 and abs(ejemplo["p_Holm6"].iloc[1] - 0.001160081) < 2e-9
assert abs(ejemplo["p_Holm6"].iloc[2] - 0.001160081) < 2e-9
glue_tbl("tbl-holm", ejemplo)
ejemplo
```

```{glue:figure} tbl-holm
:name: tbl-holm
:figwidth: 85%

Cálculo de $p_{\text{Holm6}}$: seis contrastes del HAR original frente a sus rivales históricos (Grid, $w=21$, 2025).
```

**Interpretación de la {numref}`tbl-holm`.** Los tres menores $p$ brutos (Random Forest, Ridge y XGBoost) se multiplican por 6, 5 y 4; el máximo acumulado hace que Ridge y XGBoost compartan
$p_{\text{Holm6}}=0{,}001160$. Se rechaza $H_0$ si $p_{\text{Holm}}<0{,}05$; el signo de DM identifica al favorecido y **no rechazar no demuestra equivalencia**. Las familias usadas son: Holm6 (seis
rivales históricos), Holm7 (seis más Decision Tree, por búsqueda) y Holm21 (los 21 pares entre los siete modelos históricos).

## 6.3 DM con Holm7 por búsqueda, 2025, $w=21$

Cuatro activos, siete horizontes, $n=358$. En cada búsqueda el modelo A conserva su selección original; los rivales usan la búsqueda indicada (Grid: seis rivales históricos más Decision Tree
añadido con validación anidada). Las familias de siete contrastes son **exploratorias**.

```{code-cell} ipython3
# Tablas DM-HLN con Holm7 por procedimiento de búsqueda
def tabla_dm(s):
    t = W21[s][["DM_HLN", "p_bruto", "p_Holm7", "rechaza_H0"]].copy()
    t.insert(0, "Modelo A", HAR)
    t["Decisión al 5 %"] = np.where(t["rechaza_H0"], "Rechaza H0 (favorece al HAR)", "No rechaza H0")
    return t.drop(columns="rechaza_H0").round(4)
rechazos = {s: [r for r in W21[s].index if W21[s].loc[r, "rechaza_H0"]] for s in SEARCHES}
assert [len(rechazos[s]) for s in SEARCHES] == [7, 3, 2, 4]
assert set(rechazos["random"]) == {"k-NN", "Random Forest", "Decision Tree"} and set(rechazos["bayesian"]) == {"k-NN", "Decision Tree"}
assert set(rechazos["genetic"]) == {"XGBoost", "Random Forest", "k-NN", "Decision Tree"}
for s in SEARCHES:
    glue_tbl(f"tbl-dm-{s}", tabla_dm(s))
tabla_dm("grid")
```

```{glue:figure} tbl-dm-grid
:name: tbl-dm-grid
:figwidth: 100%

DM–HLN (HAC de 26 rezagos, $t_{357}$) y Holm7: HAR-Ridge + XGBoost frente a los rivales del Grid; *Decision Tree añadido.
```

```{code-cell} ipython3
tabla_dm("random")
```

```{glue:figure} tbl-dm-random
:name: tbl-dm-random
:figwidth: 100%

DM–HLN y Holm7 con rivales reoptimizados por Random Search (validación temporal anidada).
```

```{code-cell} ipython3
tabla_dm("bayesian")
```

```{glue:figure} tbl-dm-bayesian
:name: tbl-dm-bayesian
:figwidth: 100%

DM–HLN y Holm7 con rivales reoptimizados por búsqueda bayesiana (Optuna TPE; no se usó Hyperopt).
```

```{code-cell} ipython3
tabla_dm("genetic")
```

```{glue:figure} tbl-dm-genetic
:name: tbl-dm-genetic
:figwidth: 100%

DM–HLN y Holm7 con rivales reoptimizados por búsqueda genética (DEAP).
```

**Interpretación de las {numref}`tbl-dm-grid`–{numref}`tbl-dm-genetic`.**

- **Grid:** los siete contrastes favorecen a HAR-Ridge + XGBoost (7/7 rechazos; el menor $p_{\text{Holm7}}$ es 0,0002 frente a Random Forest y el mayor 0,0443 frente a k-NN y Lasso).
- **Random Search:** 3/7 rechazos (k-NN, Random Forest y Decision Tree). Frente a XGBoost, Ridge, Lasso y SVR lineal **no se rechaza** (p Holm7 entre 0,34 y 0,78).
- **Optuna TPE:** 2/7 (k-NN y Decision Tree). Lasso no se distingue: su DM es positivo (0,869), es decir, apunta a favor de Lasso, sin significancia ($p_{\text{Holm7}}=1$).
- **DEAP:** 4/7 (XGBoost, Random Forest, k-NN y Decision Tree).

## 6.4 Resumen por ventana y búsqueda

Las ventanas $w\in\{7,14,21,28\}$ se evaluaron con la misma lógica (rezagos HAC $12,19,26,33$ y siete rivales por búsqueda y ventana, Holm7 por separado, sin ajuste conjunto).

```{code-cell} ipython3
# Rechazos DM-Holm7 favorables al HAR por ventana y búsqueda (de 7) y R² del HAR por ventana
res = read_csv(R / "window_review_original_vs_optimized_2025" / "window_summary.csv").set_index("volatility_window")
filas = {}
for w in (7, 14, 21, 28):
    fila = {"R² del HAR": round(res.loc[w, "reference_r2"], 4)}
    for s in SEARCHES:
        t = inference_table(w, s)
        fila[SEARCHES[s]] = f"{int(t.rechaza_H0.sum())}/7"
    filas[f"w = {w}"] = fila
por_ventana = pd.DataFrame(filas).T
esperado = {"w = 7": ["6/7", "6/7", "4/7", "4/7"], "w = 14": ["7/7", "1/7", "1/7", "4/7"],
            "w = 21": ["7/7", "3/7", "2/7", "4/7"], "w = 28": ["2/7", "1/7", "1/7", "0/7"]}
for k, v in esperado.items():
    assert por_ventana.loc[k, list(SEARCHES.values())].tolist() == v, k
glue_tbl("tbl-ventanas", por_ventana)
por_ventana
```

```{glue:figure} tbl-ventanas
:name: tbl-ventanas
:figwidth: 100%

$R^2$ del modelo original y rechazos DM–Holm7 favorables a HAR-Ridge + XGBoost (de 7) por ventana y búsqueda, 2025.
```

**Interpretación de la {numref}`tbl-ventanas`.** La elección histórica de $w=21$ fue la de mayor $R^2$ entre las ventanas con mayoría favorable bajo DM-Holm6. La ampliación muestra
que **el respaldo estadístico depende de la búsqueda**: con el Grid, el original gana 6/7 en $w=7$ y 7/7 en $w=14$ y $w=21$ (solo 2/7 en $w=28$), pero con rivales reoptimizados
los rechazos caen a 1–4 de 7 en $w=14$, $w=21$ y $w=28$ y solo son altos en $w=7$ con Random Search (6/7). La ventana $w=28$ tiene el mayor $R^2$ (0,8755) pero casi ningún rechazo.
La selección de la ventana es retrospectiva.

## 6.5 Tamaño de la mejora e intervalos de confianza

Cada intervalo compara el MSE de un rival con el de HAR-Ridge + XGBoost:

```{math}
:label: eq-ic
\hat G=\mathrm{MSE}_{\text{rival}}-\mathrm{MSE}_{\text{HAR-Ridge + XGBoost}},\qquad \mathrm{IC}_{95\%}\simeq\hat G\pm1{,}96\,\widehat{\mathrm{se}}_{\mathrm{HAC}}(\hat G).
```

Si todo el intervalo es $>0$, favorece a HAR-Ridge + XGBoost (menor MSE a nivel individual); si incluye cero, la diferencia media no se distingue de cero con este procedimiento; si es $<0$, favorece al rival.
Los IC son normales HAC al 95 %, **individuales y sin ajuste simultáneo**; el DM–Holm se interpreta por separado. Unidades: puntos porcentuales de volatilidad al cuadrado.

```{code-cell} ipython3
# IC95 % de G por búsqueda (w = 21, HAC de 26 rezagos, sin HLN ni ajuste múltiple)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, s in zip(axes.ravel(), SEARCHES):
    t = W21[s].iloc[::-1]
    y = np.arange(len(t))
    color = np.where(t["IC_positivo"], "#2e7d32", "#888888")
    for yi, g, lo, hi, c in zip(y, t["G"], t["IC_inf"], t["IC_sup"], color):
        ax.plot([lo, hi], [yi, yi], color=c, lw=2); ax.plot(g, yi, "o", color=c)
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=8)
    ax.set_title(f"{SEARCHES[s]}: {int(t.IC_positivo.sum())}/7 IC positivos")
for ax in axes[1]:
    ax.set_xlabel("Reducción de MSE (puntos porcentuales²; >0 favorece al HAR)")
fig.suptitle("Ventaja de HAR-Ridge + XGBoost en 2025, w = 21 (verde: IC > 0; gris: incluye cero)", y=1.0)
fig.tight_layout(); glue_fig("fig-ic", fig)

ic = pd.concat({SEARCHES[s]: W21[s][["G", "IC_inf", "IC_sup"]] for s in SEARCHES}, axis=1).round(5)
assert abs(W21["random"].loc["XGBoost", "G"] - .01972) < 2e-5 and abs(W21["bayesian"].loc["Lasso", "G"] + .00619) < 2e-5
assert abs(W21["genetic"].loc["XGBoost", "IC_inf"] - .00392) < 2e-5 and abs(W21["grid"].loc["Decision Tree*", "G"] - .07535) < 2e-5
glue_tbl("tbl-ic", ic)
ic
```

```{glue:figure} fig-ic
:name: fig-ic
:figwidth: 100%

Reducción de MSE $\hat G$ (eje x, puntos porcentuales al cuadrado) con IC del 95 % para cada rival (eje y), por búsqueda; $w=21$, 2025.
```

```{glue:figure} tbl-ic
:name: tbl-ic
:figwidth: 100%

$\hat G$ e intervalo del 95 % por rival y búsqueda ($w=21$, 2025; puntos porcentuales al cuadrado).
```

**Interpretación de la {numref}`fig-ic` y la {numref}`tbl-ic`.** Con el Grid los siete intervalos son positivos (de 0,024 frente a XGBoost a 0,075 frente a Decision Tree). Con rivales reoptimizados, el número
de intervalos positivos baja a 3 (Random Search), 3 (Optuna) y 4 (DEAP), y la mejora estimada y su precisión cambian según el rival. Con Optuna, **Lasso tiene un MSE puntual menor** ($\hat G=-0{,}0062$, IC
$[-0{,}0199;\ 0{,}0075]$), pero el intervalo incluye cero.

## 6.6 Tamaño del efecto estandarizado

Además de la reducción de MSE, se calcula la $d$ de Cohen pareada, $d_z=\bar d_{\text{rival−HAR}}/s_d$ {cite}`cohen1988power`, con intervalo por *bootstrap* estacionario {cite}`politis1994stationary`
(2.000 réplicas, bloque medio 34, semilla 42). El umbral convencional de efecto pequeño es $|d_z|\ge0{,}2$.

```{code-cell} ipython3
# d de Cohen pareada (rival - HAR) con IC por bootstrap estacionario de percentiles
rng = np.random.default_rng(SEED)
idx = stationary_bootstrap_indices(358, 34, 2000, rng)
filas = []
for s in SEARCHES:
    har, riv = search_losses(21, s)
    for nombre in riv.columns:
        d = riv[nombre].values - har
        dz = d.mean() / d.std(ddof=1)
        bs = d[idx]; boot = bs.mean(axis=1) / bs.std(axis=1, ddof=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        filas.append({"búsqueda": SEARCHES[s], "rival": nombre, "d_z": dz, "IC_inf": lo, "IC_sup": hi, "|d_z|>=0.2": abs(dz) >= .2})
efecto = pd.DataFrame(filas).set_index(["búsqueda", "rival"])
resumen_efecto = efecto.groupby(level=0, sort=False)["|d_z|>=0.2"].sum().astype(int).rename("rivales con |d_z| ≥ 0,2 (de 7)")
assert len(efecto) == 28 and resumen_efecto.tolist() == [7, 3, 3, 3]
for s_ in ["random", "bayesian", "genetic"]:   # en las búsquedas reoptimizadas, solo estos tres rivales alcanzan 0,2
    e_ = efecto.loc[SEARCHES[s_]]
    assert set(e_.index[e_["|d_z|>=0.2"]]) == {"k-NN", "Random Forest", "Decision Tree"}
glue_tbl("tbl-dz", efecto.round(3)); glue_tbl("tbl-dz-res", resumen_efecto.to_frame())
efecto.round(3)
```

```{glue:figure} tbl-dz
:name: tbl-dz
:figwidth: 100%

$d_z$ de Cohen pareada (rival − HAR) con IC bootstrap estacionario del 95 %; positivo favorece al HAR.
```

```{code-cell} ipython3
resumen_efecto.to_frame()
```

```{glue:figure} tbl-dz-res
:name: tbl-dz-res
:figwidth: 70%

Número de rivales con efecto de al menos 0,2 a favor o en contra del HAR, por búsqueda.
```

**Interpretación de las {numref}`tbl-dz` y {numref}`tbl-dz-res`.** Con el Grid, los siete rivales tienen $|d_z|\ge0{,}2$ a favor del HAR (efecto al menos pequeño). Con rivales reoptimizados solo **tres**
(k-NN, Random Forest y Decision Tree, con $d_z$ entre 0,2 y 0,4 aproximadamente) alcanzan ese umbral en las tres búsquedas; frente a Ridge, Lasso, SVR lineal y XGBoost el efecto es trivial ($|d_z|<0{,}2$), y con Lasso
y Optuna es ligeramente **negativo** ($d_z\approx-0{,}07$), a favor del rival. Es coherente con los intervalos de la {numref}`tbl-ic`.

## 6.7 Pruebas complementarias: MCS y SPA

Con las pérdidas de **todas las ventanas** de 2025 (112 series, 34 modelos incluidos HAR-Ridge + XGBoost y la persistencia), se aplicaron el *Model Confidence Set* al 90 % {cite}`hansen2011mcs` y la prueba de habilidad predictiva superior (SPA)
de Hansen {cite}`hansen2005spa` con *bootstrap* estacionario de 2.000 réplicas y bloque medio 34.

```{code-cell} ipython3
spa = json.loads((EV / "2025" / "spa_vs_original.json").read_text(encoding="utf-8"))
mcs = read_csv(EV / "2025" / "mcs_membership.csv").set_index("model")
resumen = pd.Series({
    "p_SPA (consistente)": round(spa["primary_p_value"], 4), "SPA rechaza H0 (α = 0,05)": spa["reject_null"],
    "modelos en el MCS 90 %": int(mcs["mcs_90"].sum()), "modelos evaluados": len(mcs),
    "HAR-Ridge + XGBoost en el MCS": bool(mcs.loc[HAR, "mcs_90"]), "persistencia en el MCS": bool(mcs.loc["Persistencia", "mcs_90"]),
}, name="2025").to_frame()
assert spa["n_models"] == 34 and not spa["reject_null"] and not mcs.loc["Persistencia", "mcs_90"] and mcs.loc[HAR, "mcs_90"]
glue_tbl("tbl-spa", resumen)
resumen
```

```{glue:figure} tbl-spa
:name: tbl-spa
:figwidth: 80%

SPA de Hansen (referencia: HAR-Ridge + XGBoost) y MCS al 90 % con las pérdidas globales de 2025.
```

La {numref}`tbl-spa` muestra que el SPA no rechaza ($p=0{,}106$): no hay evidencia de que **algún** candidato supere al original en el conjunto global. El original permanece en el MCS al 90 % junto con muchos otros
modelos, mientras que la **persistencia queda excluida**.

## 6.8 Veredicto de superioridad (R3)

```{code-cell} ipython3
# Resumen final: rechazos DM-Holm7 e IC positivos por búsqueda y matriz rival x búsqueda
final = pd.DataFrame({SEARCHES[s]: {"DM-Holm7 (de 7)": int(W21[s].rechaza_H0.sum()), "IC positivos (de 7)": int(W21[s].IC_positivo.sum())}
                      for s in SEARCHES}).T
assert final.loc["Grid Search*"].tolist() == [7, 7] and final.loc["Random Search"].tolist() == [3, 3]
assert final.loc["Bayesiana (Optuna TPE)"].tolist() == [2, 3] and final.loc["Genética (DEAP)"].tolist() == [4, 4]
matriz = pd.DataFrame({SEARCHES[s]: np.where(W21[s].rechaza_H0, "✔", "—") for s in SEARCHES}, index=W21["grid"].index.str.replace("*", "", regex=False))
glue_tbl("tbl-final", final); glue_tbl("tbl-matriz", matriz)
display(final); matriz
```

```{glue:figure} tbl-final
:name: tbl-final
:figwidth: 70%

Conteo final por búsqueda ($w=21$, 2025): rechazos DM–Holm7 e intervalos de ganancia positivos.
```

```{glue:figure} tbl-matriz
:name: tbl-matriz
:figwidth: 80%

Rechazos DM–Holm7 favorables a HAR-Ridge + XGBoost (✔) por rival y búsqueda.
```

La {numref}`tbl-final` y la {numref}`tbl-matriz` resumen la evidencia. Ambos conteos (DM-Holm7 e IC) indican respaldo favorable al HAR original en las cuatro búsquedas; el respaldo es **total con el Grid** y **parcial con los rivales reoptimizados** (3/7, 2/7 y 4/7 en DM-Holm7).

```{admonition} Veredicto R3
:class: important
**R3 (superioridad estadística): CUMPLIDA EN EL PROTOCOLO PRINCIPAL y con respaldo parcial ante la reoptimización de los rivales.**

- **Superioridad respaldada.** Con el Grid original ($w=21$, 2025, 358 orígenes comunes) HAR-Ridge + XGBoost tiene el menor MSE, RMSE y MAE y el mayor $R^2$ de la tabla, y supera de forma estadísticamente significativa a los **siete** rivales
  (DM–HLN con HAC y Holm7: 7/7; intervalos de ganancia positivos: 7/7). Es además muy superior a la persistencia, que queda excluida del MCS al 90 %.
- **Respaldo dependiente de la búsqueda.** Con Random Search, Optuna TPE y DEAP los rivales se reoptimizan con validación anidada y la ventaja significativa se sostiene en 3/7, 2/7 y 4/7 contrastes; de forma consistente frente a **k-NN y Decision Tree**.
- **Lo que no se afirma.** Frente a Ridge, Lasso y SVR lineal reoptimizados la diferencia no es distinguible de cero; con Optuna **Lasso obtiene mejores MSE, RMSE, MAE y $R^2$ puntuales** (sin ventaja significativa en MSE bajo DM-Holm7), el SPA global no rechaza
  ($p=0{,}106$) y 2025 es un único año ya examinado, sin periodo confirmatorio intacto. Por eso la superioridad se declara **bajo el protocolo principal y con evidencia ajustada**, no como superioridad general sobre cualquier configuración de los rivales.
```
