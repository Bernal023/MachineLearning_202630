# Capítulo 4 · Revisión sistemática de literatura y declaración de novedad

```{code-cell} ipython3
:tags: [remove-cell]
import re
import pandas as pd
from IPython.display import display
from jbook_utils import BOOK, REPO, glue_tbl
```

## 4.1 Protocolo de revisión

- **Bases de datos:** Scopus y Web of science.
- **Periodo:** 2009–2026 (desde la formulación del HAR {cite}`corsi2009har`).
- **Criterios de inclusión:** pronóstico de volatilidad (realizada o rodante) de criptoactivos; modelos HAR, ML o híbridos;
  evaluación fuera de muestra con pérdida explícita. **Exclusión:** pronóstico de precio/rentabilidad; solo redes neuronales; sin evaluación fuera de muestra.
- **Cribado:** título/resumen → texto completo; dos revisores; acuerdo medido con el coeficiente $\kappa$ de Cohen {cite}`cohen1960kappa`; discrepancias resueltas por consenso.
- **Cadenas de búsqueda (5, en inglés y español):**

```{code-cell} ipython3
# Matriz de novedad: columnas exigidas; recuentos pendientes de ejecución por los autores
cadenas = [
    ("S1", 'cryptocurrency AND volatility AND forecasting AND ("HAR" OR "heterogeneous autoregressive")'),
    ("S2", 'cryptocurrency AND volatility AND "machine learning" AND (XGBoost OR "gradient boosting")'),
    ("S3", '"realized volatility" AND bitcoin AND (ridge OR lasso OR "regularized regression")'),
    ("S4", '("model confidence set" OR "superior predictive ability") AND cryptocurrency AND volatility'),
    ("S5", 'volatilidad AND criptomonedas AND (pronóstico OR predicción) AND "aprendizaje automático"'),
]
bases = ["Scopus", "IEEE Xplore", "Google Scholar", "arXiv"]
filas = [{"base_de_datos": b, "cadena": f"{c}: {q}", "resultados": "pendiente", "revisados": "pendiente",
          "relevantes": "pendiente", "observaciones": ""} for b in bases for c, q in cadenas]
matriz = pd.DataFrame(filas)
assert len(matriz) == 20 and matriz["base_de_datos"].nunique() == 4 and len(cadenas) == 5
glue_tbl("tbl-novedad", matriz)
matriz.head(8)
```

```{glue:figure} tbl-novedad
:name: tbl-novedad
:figwidth: 100%

Matriz de novedad (muestra de 8 de 20 filas; 4 bases × 5 cadenas). Los recuentos están pendientes de ejecución.
```

## 4.2 Métodos afines (diferenciación)

| Método | Fuente | Qué hace | Diferencia frente a HAR-Ridge + XGBoost |
|---|---|---|---|
| GARCH | {cite}`bollerslev1986garch` | Varianza condicional paramétrica | No usa entradas de múltiples escalas ni ML. |
| HAR-RV | {cite}`corsi2009har` | Regresión lineal sobre volatilidad diaria/semanal/mensual | Es el componente lineal; aquí se regulariza y se combina con árboles. |
| HAR con Lasso | {cite}`audrino2016lasso` | Selección de rezagos HAR con Lasso | Misma idea de regularización; sin componente no lineal. |
| Bosques aleatorios | {cite}`breiman2001random` | Ensamble de árboles | Se evalúa como benchmark independiente. |
| XGBoost | {cite}`chen2016xgboost` | *Boosting* de árboles | Es el componente no lineal. |
| SVR | {cite}`drucker1997svr` | Regresión por vectores soporte | Benchmark del estudio. |

## 4.3 Antecedentes localizados (búsqueda web preliminar, 21-oct-2026)

Dos búsquedas web generales (no sistemáticas; no sustituyen la matriz de §4.1) mostraron que **no es nuevo** usar ML, XGBoost o HAR para
la volatilidad de criptomonedas:

| Antecedente | Qué hace | Relación con este proyecto |
|---|---|---|
| {cite}`brauneis2024horserace` | Compara HAR con cinco métodos de ML (incluye XGBoost, LightGBM y LSTM) para volatilidad cripto, con y sin sentimiento. | Mismo enfoque comparativo; no combina HAR-Ridge y XGBoost en una sola predicción. |
| {cite}`papi2025xgboostbtc` | Compara XGBoost con HAR para la varianza realizada de BTC/USDT en Binance. | Mismo activo y fuente de datos; compara los modelos, no los mezcla. |
| {cite}`xgblstm2026` | Apilamiento híbrido XGBoost + LSTM para volatilidad de Bitcoin. | Híbrido con redes neuronales (excluidas aquí). |
| {cite}`garchxgb` | Híbrido GARCH + XGBoost para el índice JSE Top40. | Híbrido estadístico + XGBoost, pero con GARCH y en otro mercado. |

En ninguno de los resultados vistos aparece la mezcla **Ridge sobre entradas HAR + XGBoost con un peso elegido por validación
temporal**, pero dos búsquedas generales no bastan para afirmarlo: debe confirmarse con la matriz completa de §4.1.

## 4.4 Declaración formal de novedad (R1)

```{admonition} Declaración formal de novedad
:class: important
Los autores declaran que, en la búsqueda bibliográfica que realizaron y en la verificación preliminar de este capítulo, **no
encontraron estudios que apliquen la siguiente combinación al pronóstico de la volatilidad futura de criptomonedas**:

1. Una regresión **Ridge sobre entradas tipo HAR** (volatilidad a 1, 3, 7, 14 y 28 días, medida a la baja y retornos que salen de la ventana),
   que predice la **corrección relativa** respecto de la volatilidad actual;
2. **combinada linealmente con XGBoost** mediante un peso $w\in[0,1]$ y una penalización $\alpha$ elegidos por validación temporal
   (con $w=0$ se conserva XGBoost), ecuación {eq}`eq-blend`;
3. para **siete horizontes diarios** y cuatro ventanas de volatilidad en **BTC, ETH, BNB y XRP**, evaluada con validación anidada
   purgada y contrastes DM–Holm, SPA y MCS.

**Alcance y límites.** La declaración se refiere únicamente a esa combinación. **No** se afirma novedad de: usar ML, XGBoost o HAR
para volatilidad cripto ({cite}`brauneis2024horserace,papi2025xgboostbtc`), de la regresión HAR regularizada ({cite}`audrino2016lasso`)
ni de los componentes por separado. La novedad es de **combinación y de protocolo de evaluación**, no de un algoritmo nuevo, y **no implica
superioridad**.
```

**Diferenciación frente a métodos afines.** La tabla de §4.2 (GARCH, HAR-RV, HAR-Lasso, bosques aleatorios, XGBoost, SVR) y la de §4.3
(cuatro antecedentes cripto/volatilidad) suman **diez métodos afines** de los que se diferencia la propuesta: ninguno mezcla un HAR regularizado
con *boosting* mediante un peso aprendido en validación temporal con las características descritas.

## 4.5 Literatura de base utilizada por los autores

Se trabajo con una lista de **50 referencias**, organizadas aquí en ocho bloques temáticos. La
{numref}`tbl-literatura` indica, para cada bloque, qué papel cumple en el proyecto; el criterio es explícito: **«aplicada»** significa que
la técnica se ejecutó en este repositorio; **«contexto»** que fundamenta el problema o la elección de métodos; **«no aplicada»** que se cita
como alternativa o trabajo futuro y no se implementó.

```{code-cell} ipython3
# Clasificación de las 50 referencias por bloque y verificación contra references.bib
bloques = {
    "A. Riesgo financiero y medidas de riesgo": (["jorion2007var", "mcneil2015qrm", "artzner1999coherent", "acerbi2002coherence"], "Contexto"),
    "B. Criptoactivos: economía, riesgo y factores": (["nakamoto2008bitcoin", "bohme2015bitcoin", "liu2021risks", "liu2022common", "bouri2017hedge", "baur2018bitcoin"], "Contexto"),
    "C. Modelos de volatilidad condicional": (["engle1982arch", "bollerslev1986garch", "glosten1993relation", "hansen2012realized"], "Contexto (no comparada)"),
    "D. Volatilidad de criptomonedas con GARCH y afines": (["dyhrberg2016garch", "katsiampa2017volatility", "chu2017garch", "caporale2019markov", "troster2019bitcoin", "trucios2019forecasting", "tiwari2019garchsv", "charles2019replication", "walther2019exogenous"], "Contexto (no comparada)"),
    "E. Aprendizaje profundo y transformers": (["hochreiter1997lstm", "ozbayoglu2020deep", "vaswani2017attention", "nie2023patchtst", "wen2023transformers", "liu2024itransformer"], "No aplicada"),
    "F. Texto, sentimiento y modelos de lenguaje": (["antweiler2004talk", "tetlock2007giving", "loughran2011liability", "devlin2019bert", "araci2019finbert", "wu2023bloomberggpt", "gruver2023llmtime"], "No aplicada"),
    "G. Evaluación de pronósticos de riesgo (VaR/ES)": (["kupiec1995techniques", "christoffersen1998evaluating", "engle2004caviar", "patton2019dynamic"], "No aplicada"),
    "H. Estadística, optimización y diagnóstico": (["cohen1960kappa", "shahriari2016human", "freire2025wind", "rasmussen2006gp", "jones1998ego", "broock1996bds", "cover2006information", "diebold1995comparing", "newey1987simple", "holm1979simple"], "Aplicada (parcial)"),
}
bib = (BOOK / "references.bib").read_text(encoding="utf-8")
claves = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
todas = [k for ks, _ in bloques.values() for k in ks]
assert len(todas) == 50 and len(set(todas)) == 50, "la lista debe tener 50 referencias únicas"
faltan = [k for k in todas if k not in claves]
assert not faltan, f"claves sin entrada en references.bib: {faltan}"
assert (REPO / "results" / "bds.csv").exists()   # el test BDS se ejecutó en una fase previa del proyecto
tabla = pd.DataFrame({"referencias": {b: len(ks) for b, (ks, _) in bloques.items()},
                      "papel en el proyecto": {b: u for b, (_, u) in bloques.items()}})
glue_tbl("tbl-literatura", tabla)
tabla
```

```{glue:figure} tbl-literatura
:name: tbl-literatura
:figwidth: 100%

Bloques temáticos de las 50 referencias de los autores y su papel en el proyecto.
```

### A. Riesgo financiero y medidas de riesgo (4)

Los textos de Jorion {cite}`jorion2007var` y McNeil, Frey y Embrechts {cite}`mcneil2015qrm`, junto con los resultados axiomáticos de
Artzner et al. {cite}`artzner1999coherent` y Acerbi y Tasche {cite}`acerbi2002coherence` sobre medidas coherentes y *expected shortfall*,
establecen **por qué importa pronosticar la volatilidad**: es el insumo de las medidas de riesgo de mercado (VaR y ES). Este bloque justifica
el problema (Capítulo 0), pero el proyecto **no calcula VaR ni ES**: se limita al pronóstico de la volatilidad.

### B. Criptoactivos (6)

Nakamoto {cite}`nakamoto2008bitcoin` define el sistema que origina el mercado; Böhme et al. {cite}`bohme2015bitcoin` describen su economía,
tecnología y gobernanza. Liu y Tsyvinski {cite}`liu2021risks` y Liu, Tsyvinski y Wu {cite}`liu2022common` documentan los riesgos, rendimientos y
factores comunes de los criptoactivos, lo que apoya tratar a BTC, ETH, BNB y XRP como un mercado con dinámica propia y factor común (la alta
correlación de volatilidades del Capítulo 2). Bouri et al. {cite}`bouri2017hedge` y Baur et al. {cite}`baur2018bitcoin` discuten el papel de
Bitcoin como cobertura o activo especulativo, un debate sobre su naturaleza que ayuda a interpretar por qué su volatilidad es alta y variable.

### C. Modelos de volatilidad condicional (4)

ARCH {cite}`engle1982arch`, GARCH {cite}`bollerslev1986garch`, el modelo asimétrico GJR {cite}`glosten1993relation` y Realized GARCH
{cite}`hansen2012realized` forman la familia de referencia de la volatilidad condicional. Son el punto de comparación natural de un modelo
HAR. **Limitación reconocida:** el proyecto **no incluye un GARCH** entre sus 512 modelos; sus referencias simples son la persistencia y el HAR
diario OLS. Un benchmark GARCH/GJR es la ausencia más evidente frente a esta literatura y queda como trabajo futuro.

### D. Volatilidad de criptomonedas con GARCH y afines (9)

Nueve trabajos aplican y comparan modelos de la familia GARCH a Bitcoin y otros criptoactivos: Dyhrberg {cite}`dyhrberg2016garch`,
Katsiampa {cite}`katsiampa2017volatility`, Chu et al. {cite}`chu2017garch`, Caporale y Zekokh {cite}`caporale2019markov` (cambio de régimen),
Tröster et al. {cite}`troster2019bitcoin` (GARCH y GAS), Trucíos {cite}`trucios2019forecasting` (medidas de riesgo robustas), Tiwari et al.
{cite}`tiwari2019garchsv` (GARCH frente a volatilidad estocástica) y Charles y Darné {cite}`charles2019replication` (réplica y robustez), más
Walther et al. {cite}`walther2019exogenous`, que pronostican volatilidad con datos mixtos (MIDAS) y determinantes exógenos. El patrón común es
**enfoque econométrico paramétrico**; frente a él, este proyecto usa **entradas tipo HAR y aprendizaje automático regularizado** con validación
temporal y pruebas de comparación (posicionamiento que se retoma en §4.3–4.4). Esta literatura confirma que modelar la volatilidad cripto con
modelos de varianza condicional es un enfoque establecido, no nuevo.

### E. Aprendizaje profundo y transformers (6)

LSTM {cite}`hochreiter1997lstm`, la revisión de aprendizaje profundo en finanzas {cite}`ozbayoglu2020deep`, el *Transformer*
{cite}`vaswani2017attention` y sus variantes para series temporales (PatchTST {cite}`nie2023patchtst`, la revisión de Wen et al.
{cite}`wen2023transformers` e iTransformer {cite}`liu2024itransformer`) describen la frontera de la predicción con redes neuronales. **No se
aplican**: el curso excluye redes neuronales y *deep learning* en el modelo original, de modo que este bloque delimita el alcance y señala la
comparación pendiente como trabajo futuro.

### F. Texto, sentimiento y modelos de lenguaje (7)

Antweiler y Frank {cite}`antweiler2004talk`, Tetlock {cite}`tetlock2007giving` y Loughran y McDonald {cite}`loughran2011liability` muestran
que el texto financiero contiene información útil; BERT {cite}`devlin2019bert`, FinBERT {cite}`araci2019finbert`, BloombergGPT
{cite}`wu2023bloomberggpt` y los pronosticadores de series con modelos de lenguaje {cite}`gruver2023llmtime` extienden esa idea con modelos
preentrenados. **No se aplican**: el estudio usa únicamente precios de Binance Spot, sin variables de sentimiento ni de texto. Se citan como
fuentes de variables exógenas para extensiones futuras.

### G. Evaluación de pronósticos de riesgo (4)

La prueba de Kupiec {cite}`kupiec1995techniques`, la evaluación de pronósticos de intervalo de Christoffersen
{cite}`christoffersen1998evaluating`, CAViaR {cite}`engle2004caviar` y los modelos semiparamétricos para ES y VaR de Patton, Ziegel y Chen
{cite}`patton2019dynamic` son el estándar para **validar medidas de riesgo** (cobertura y condicionalidad de las excedencias). **No se aplican**,
porque el objeto pronosticado es la volatilidad y no un cuantil; si el trabajo se extiende a VaR/ES, estas pruebas son el paso siguiente.

### H. Estadística, optimización y diagnóstico (10)

Este bloque sí tiene uso directo:

- **Diebold–Mariano** {cite}`diebold1995comparing`, con varianza HAC de **Newey–West** {cite}`newey1987simple` (núcleo de Bartlett) y corrección de
  **Holm** {cite}`holm1979simple`: son las pruebas del Capítulo 6 (el ajuste HLN adicional es de {cite}`harvey1997testing`).
- **Información mutua** {cite}`cover2006information`: el diagnóstico de entradas del Capítulo 2 §2.10.
- **Prueba BDS** {cite}`broock1996bds`: se ejecutó en una fase previa del proyecto (`results/bds.csv`), **no** forma parte de los análisis de este libro.
- **Optimización bayesiana:** la revisión de Shahriari et al. {cite}`shahriari2016human`, los procesos gaussianos {cite}`rasmussen2006gp` y la
  optimización global eficiente {cite}`jones1998ego` fundamentan la familia de métodos secuenciales sobre modelo sustituto. El optimizador
  «bayesiano» del proyecto es TPE de Optuna {cite}`akiba2019optuna`, que **no** usa procesos gaussianos: estas fuentes enmarcan la idea general, no
  el algoritmo exacto.
- **Pronóstico híbrido con optimización en otro dominio:** Freire et al. {cite}`freire2025wind` (según su título, pronóstico de energía eólica con redes
  híbridas en Brasil) ilustra el uso de hibridación y optimización de hiperparámetros fuera de las finanzas.
- **Acuerdo entre revisores:** el $\kappa$ de Cohen {cite}`cohen1960kappa` se propone para el cribado de la revisión (§4.1). No debe confundirse
  con la *d* de Cohen del Capítulo 6, que proviene de {cite}`cohen1988power`.

### Síntesis y brecha respecto de esta literatura

La literatura cripto de volatilidad recopilada es mayoritariamente **econométrica (GARCH y variantes)** o de **aprendizaje
profundo/texto**. Este proyecto ocupa un punto intermedio: **regresión regularizada con entradas HAR y *boosting* de árboles**, evaluada con
inferencia predictiva formal. Tres brechas deben declararse: (i) **no hay benchmark GARCH/GJR/Realized GARCH**; (ii) **no se compara con redes
profundas ni transformers**, por restricción del curso; (iii) **no se evalúan cuantiles de riesgo** (VaR/ES).
