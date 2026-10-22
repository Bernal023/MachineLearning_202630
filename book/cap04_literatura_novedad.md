# Capítulo 4 · Revisión sistemática de literatura y declaración de novedad

```{admonition} Estado de esta sección
:class: warning
Una revisión sistemática exige consultar bases académicas (Scopus, Google Scholar, arXiv, IEEE Xplore…) y registrar
resultados reales. Los recuentos de esa consulta **no están registrados en el repositorio** y este libro no los inventa. Se entrega
(a) el protocolo, (b) la matriz de novedad con las columnas exigidas (recuentos **pendientes de completar por los autores**),
(c) los antecedentes que sí se localizaron en una búsqueda web preliminar del 21-oct-2026 y (d) una declaración de novedad
**acotada**: el enfoque general (ML/XGBoost/HAR para volatilidad cripto) **ya existe en la literatura**; lo que se declara como
no encontrado es la combinación específica descrita en §4.4.
```

```{code-cell} ipython3
:tags: [remove-cell]
import pandas as pd
from IPython.display import display
from jbook_utils import glue_tbl
```

## 4.1 Protocolo de revisión

- **Bases de datos (≥3):** Scopus, IEEE Xplore, Google Scholar y arXiv (q-fin.ST / stat.ML).
- **Periodo:** 2009–2026 (desde la formulación del HAR {cite}`corsi2009har`).
- **Criterios de inclusión:** pronóstico de volatilidad (realizada o rodante) de criptoactivos; modelos HAR, ML o híbridos;
  evaluación fuera de muestra con pérdida explícita. **Exclusión:** pronóstico de precio/rentabilidad; solo redes neuronales; sin evaluación fuera de muestra.
- **Cribado:** título/resumen → texto completo; dos revisores; discrepancias resueltas por consenso.
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

La {numref}`tbl-novedad` fija el diseño de la búsqueda; **no contiene resultados**. Los autores deben ejecutar las 20 búsquedas,
registrar fecha y recuentos y completar el cribado antes de afirmar cualquier ausencia en la literatura.

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
superioridad**: el Capítulo 6 refuta R3. Tampoco cumple R2 (usa `xgboost` y `scikit-learn`). La declaración es válida solo mientras la
matriz de §4.1 se complete y no aparezca un antecedente exacto.
```

**Diferenciación frente a métodos afines.** La tabla de §4.2 (GARCH, HAR-RV, HAR-Lasso, bosques aleatorios, XGBoost, SVR) y la de §4.3
(cuatro antecedentes cripto/volatilidad) suman **diez métodos afines** de los que se diferencia la propuesta: ninguno mezcla un HAR regularizado
con *boosting* mediante un peso aprendido en validación temporal con las características descritas.
