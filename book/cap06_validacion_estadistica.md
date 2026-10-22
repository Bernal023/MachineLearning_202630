# Capítulo 6 · Validación estadística de superioridad

```{code-cell} ipython3
:tags: [remove-cell]
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from jbook_utils import R, read_csv, glue_fig, glue_tbl
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
EV = R / "evaluation"
REF = "HAR-Ridge + XGBoost"
```

## 6.1 Protocolo de comparación pareada

Los datos son **series temporales con dependencia**, por lo que la rama aplicable es la de
series (no Friedman/Nemenyi, que suponen conjuntos de datos independientes). El protocolo,
declarado antes de generar los pronósticos de 2026 (`inference_plan.json`), fue:

- **Mismas particiones y misma semilla:** todos los modelos usan los mismos orígenes diarios (358 en 2025;
  236 en 2026, del 1-ene al 24-ago), el mismo preprocesamiento y semilla global 42. Cada fecha aporta una
  pérdida media sobre las 112 series (16 configuraciones × 7 horizontes).
- **Contraste primario:** cuatro candidatas HAR-XG (una por optimizador) frente al modelo vigente
  {math}`\text{HAR-Ridge + XGBoost}`, en 2026.
- **Pruebas formales (tres):** (1) Diebold–Mariano {cite}`diebold1995comparing` con corrección de Harvey–Leybourne–Newbold
  {cite}`harvey1997testing` y varianza HAC de Bartlett (33 rezagos), con ajuste de Holm {cite}`holm1979simple` sobre las
  4 comparaciones; (2) *Model Confidence Set* al 90 % {cite}`hansen2011mcs`; (3) prueba de habilidad predictiva superior (SPA)
  de Hansen {cite}`hansen2005spa`, con *bootstrap* estacionario {cite}`politis1994stationary` de 2.000 réplicas y bloque medio 34.
- **Tamaño del efecto e intervalos:** ganancia absoluta de MSE y *d* de Cohen pareado ($d_z$) con intervalos *bootstrap* estacionario del 95 %.

La diferencia de pérdida para la fecha $t$ entre referencia y candidata es

```{math}
:label: eq-dm
d_t=L_t^{\text{ref}}-L_t^{\text{cand}},\qquad
\mathrm{DM}=\frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}(d_t)/n}}\times
\sqrt{\frac{n+1-2h+h(h-1)/n}{n}},
```

con $h=7$ y $\widehat{\mathrm{LRV}}$ la varianza de largo plazo HAC; un $\bar d>0$ favorece a la candidata.

## 6.2 Métricas fuera de muestra (retrospectivas)

```{code-cell} ipython3
# MSE macro 2025 y 2026 de los modelos relevantes frente a persistencia
mm = read_csv(EV / "macro_metrics.csv")
claves = [REF, "HAR-XG [grid]", "HAR-XG [bayesian]", "Lasso [genetic]", "SVR [genetic]", "XGBoost [grid]", "Persistencia"]
met = mm[mm.model.isin(claves)].pivot(index="model", columns="year", values=["mse", "r2"]).loc[claves]
met.columns = [f"{a}_{b}" for a, b in met.columns]
met = met.round(4)
glue_tbl("tbl-metricas", met)
assert abs(met.loc[REF, "mse_2025"] - .4647) < 1e-4 and abs(met.loc["Persistencia", "mse_2026"] - .7951) < 1e-4
met
```

```{glue:figure} tbl-metricas
:name: tbl-metricas
:figwidth: 90%

MSE (puntos porcentuales al cuadrado) y R² macro de 2025 y 2026 para los modelos clave.
```

**Interpretación de la {numref}`tbl-metricas`.** Todos los modelos de ML reducen el MSE frente a la persistencia
en una magnitud grande (el modelo vigente: de 0,8282 a 0,4647 en 2025 y de 0,7951 a 0,4560 en 2026, es decir
−43,9 % y −42,6 %). Entre los mejores modelos las diferencias son pequeñas: en 2025 el mejor es HAR-XG *grid*
(0,4517, −2,8 % frente al vigente) y en 2026 Lasso genético (0,4275, −6,3 %). Que el mejor cambie de
año es la señal que las pruebas siguientes cuantifican.

## 6.3 Diebold–Mariano con Holm, SPA y MCS

```{code-cell} ipython3
# Prueba primaria: DM-HLN (Bartlett HAC 33) con Holm sobre 4 contrastes, por año
dm = read_csv(EV / "directed_dm_all_years.csv")
tabla_dm = (dm[["year", "model_a", "n", "diferencia_mse", "dm", "p", "p_holm", "candidate_lower_mse"]]
            .rename(columns={"model_a": "candidata", "diferencia_mse": "MSE_cand−MSE_ref",
                             "p": "p_crudo", "p_holm": "p_Holm4", "candidate_lower_mse": "MSE_menor"})
            .round(4).set_index(["year", "candidata"]))
assert (dm.loc[dm.year == 2026, "p_holm"] > .05).all() and (dm.loc[dm.year == 2025, "p_holm"] > .05).all()
glue_tbl("tbl-dm", tabla_dm)
tabla_dm
```

```{glue:figure} tbl-dm
:name: tbl-dm
:figwidth: 100%

DM–HLN con HAC de Bartlett: candidatas HAR-XG frente al modelo vigente; p-valores crudos y ajustados por Holm (4 contrastes).
```

```{code-cell} ipython3
# SPA de Hansen y pertenencia al MCS 90 % del modelo vigente y de las candidatas
spa = {a: json.loads((EV / str(a) / "spa_vs_original.json").read_text(encoding="utf-8")) for a in (2025, 2026)}
mcs = {a: read_csv(EV / str(a) / "mcs_membership.csv") for a in (2025, 2026)}
resumen = pd.DataFrame({
    "p_SPA (consistente)": {a: spa[a]["primary_p_value"] for a in spa},
    "rechaza H0 (α=0,05)": {a: spa[a]["reject_null"] for a in spa},
    "modelos en MCS 90 %": {a: int(mcs[a]["mcs_90"].sum()) for a in mcs},
    "modelos totales": {a: len(mcs[a]) for a in mcs},
    "vigente en MCS 90 %": {a: bool(mcs[a].set_index("model").loc[REF, "mcs_90"]) for a in mcs},
}).round(4)
assert not any(spa[a]["reject_null"] for a in spa) and resumen["vigente en MCS 90 %"].all()
glue_tbl("tbl-spa", resumen)
resumen
```

```{glue:figure} tbl-spa
:name: tbl-spa
:figwidth: 95%

SPA de Hansen (referencia: modelo vigente) y tamaño del MCS al 90 % por año de evaluación.
```

**Interpretación de las {numref}`tbl-dm` y {numref}`tbl-spa`.** Ninguna de las cuatro candidatas supera al vigente
tras el ajuste de Holm: los p ajustados de 2026 van de 0,657 a 0,759 y los de 2025 de 0,230 a 0,931. El SPA global
tampoco rechaza (p = 0,106 en 2025 y 0,227 en 2026), y el modelo vigente **permanece dentro del MCS al 90 %** en
ambos años: el conjunto de modelos indistinguibles es amplio y lo incluye. Dicho de otro modo, hay candidatas con
MSE puntual menor, pero esa ventaja es compatible con ruido muestral. Las 
pruebas son *bilaterales* y con 236–358 fechas dependientes tienen potencia limitada para ganancias de 2–4 %; no
rechazar **no prueba** equivalencia, solo falta de evidencia de superioridad.

## 6.4 Tamaño del efecto e intervalos

```{code-cell} ipython3
# Ganancia de MSE pareada, d de Cohen pareado e intervalos bootstrap estacionario 95 %
ef = read_csv(EV / "paired_effects_vs_original.csv")
ef_t = (ef[["year", "model", "mse_gain", "gain_ci_lower", "gain_ci_upper", "cohen_dz", "dz_ci_lower", "dz_ci_upper"]]
        .round(4).set_index(["year", "model"]))
ef["ic_incluye_cero"] = (ef.gain_ci_lower <= 0) & (ef.gain_ci_upper >= 0)
ef["efecto_>=0.2"] = ef.cohen_dz.abs() >= 0.2
assert ef["ic_incluye_cero"].all() and ef["efecto_>=0.2"].sum() == 2  # grid y genetic de 2026

fig, ax = plt.subplots(figsize=(8, 3.8))
for i, (y, g) in enumerate(ef.groupby("year")):
    ypos = np.arange(len(g)) + i * .35
    ax.errorbar(g.mse_gain, ypos, xerr=[g.mse_gain - g.gain_ci_lower, g.gain_ci_upper - g.mse_gain],
                fmt="o", capsize=3, label=str(y))
ax.set_yticks(np.arange(4) + .175); ax.set_yticklabels(ef[ef.year == 2025].model)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Ganancia de MSE frente al vigente (puntos porc.²; >0 favorece a la candidata)")
ax.set_title("Ganancia pareada con IC bootstrap estacionario 95 %"); ax.legend(frameon=False)
fig.tight_layout(); glue_fig("fig-efecto", fig)
glue_tbl("tbl-efecto", ef_t)
ef_t
```

```{glue:figure} tbl-efecto
:name: tbl-efecto
:figwidth: 100%

Ganancia de MSE, $d_z$ de Cohen pareado y sus intervalos del 95 %.
```

```{glue:figure} fig-efecto
:name: fig-efecto
:figwidth: 85%

Ganancia de MSE (eje x, puntos porcentuales al cuadrado) de cada candidata frente al vigente con IC del 95 %, 2025 y 2026.
```

**Interpretación de la {numref}`tbl-efecto` y la {numref}`fig-efecto`.** Los ocho intervalos de ganancia de MSE
contienen el cero. En 2025 ningún $|d_z|$ alcanza el umbral convencional de 0,2 (el mayor es 0,14 en valor absoluto). En 2026
dos candidatas lo superan por poco (*grid*: 0,22; genética: 0,21), un efecto «pequeño» en la convención de Cohen, pero los IC
de $d_z$ incluyen valores nulos (p. ej. [−0,010; 0,475] para *grid*). Un tamaño de efecto pequeño con intervalo que cruza
cero es coherente con la ausencia de significancia de la {numref}`tbl-dm`: la ventaja puntual existe, pero no es distinguible
del ruido con la muestra disponible.

```{admonition} Desviaciones respecto de la rúbrica
:class: warning
Los intervalos son *bootstrap* estacionario por **percentiles**, no BCa {cite}`efron1987bca`; el efecto se mide con
$d_z$ de Cohen y **no** con el $\delta$ de Cliff; no se aplicaron Friedman–Nemenyi (aplican a corte transversal), Clark–West
(solo a pares anidados lineales documentados) ni se corrigió por Benjamini–Hochberg (se usó Holm, que controla el error
por familia). La prueba de Giacomini–White se reportó como exploratoria en el análisis original (`inference_plan.json`).
```

## 6.5 Veredicto honesto de superioridad (R3)

```{code-cell} ipython3
# Regla fuerte predeclarada: ninguna candidata la cumple
rs = json.loads((R / "review_summary.json").read_text(encoding="utf-8"))
prim = pd.DataFrame(rs["primary_comparisons"])[["model", "mse2025", "mse2026", "reduction2026_pct",
                                              "p_dm_holm4", "gain_ci_low", "gain_ci_high", "strong_retrospective_gate"]].round(4)
assert rs["strong_retrospective_rule_passed"] is False and not prim["strong_retrospective_gate"].any()
assert rs["untouched_confirmatory_test"] is False
glue_tbl("tbl-gate", prim.set_index("model"))
prim.set_index("model")
```

```{glue:figure} tbl-gate
:name: tbl-gate
:figwidth: 100%

Comparaciones primarias y resultado de la regla fuerte retrospectiva.
```

La regla exigía, simultáneamente: menor MSE global en 2026, DM favorable con $p_{\text{Holm4}}<0{,}05$, IC 95 % de ganancia
positivo, pertenencia al MCS 90 %, SPA con $p<0{,}05$ y menor MSE global que el original en 2025. En la
{numref}`tbl-gate` las cuatro candidatas reducen el MSE de 2026 entre 2,6 % y 4,1 %, pero ninguna pasa DM con Holm, ninguna tiene IC
estrictamente positivo (todos incluyen 0) y el SPA no rechaza.

```{admonition} Veredicto R3
:class: important
**R3 (superioridad estadística verificada): REFUTADO.** No hay evidencia de que alguna variante supere al modelo vigente, y
tampoco existe un periodo no explorado que permita una prueba confirmatoria. Lo que sí está sólidamente demostrado es
que los modelos de ML mejoran de forma amplia y robusta a la persistencia y al HAR diario OLS.
```
