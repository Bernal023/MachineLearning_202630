# Capítulo 4 · Revisión sistemática de literatura y declaración de novedad

```{admonition} Estado de esta sección
:class: warning
Una revisión sistemática exige consultar bases académicas (Scopus, Google Scholar, arXiv, IEEE Xplore…) y registrar
resultados reales. **Esa consulta no se ejecutó ni está registrada en el repositorio**, y este libro no inventa recuentos. Se entrega
(a) el protocolo listo para ejecutar, (b) la matriz de novedad con las columnas exigidas y las celdas de recuentos **pendientes**, y
(c) el posicionamiento frente a métodos afines con fuentes primarias que sí se citan. La declaración de novedad resultante es **negativa**.
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

## 4.3 Declaración de novedad

```{admonition} Declaración formal (R1)
:class: important
**R1 (novedad): NO SE DECLARA.** HAR-Ridge + XGBoost combina componentes ampliamente conocidos (regresión HAR regularizada
{cite}`corsi2009har,audrino2016lasso,hoerl1970ridge` y *boosting* de árboles {cite}`chen2016xgboost`) y no hay evidencia
registrada de ausencia en la literatura: la revisión sistemática no se ha ejecutado. Hasta completarla, el proyecto debe
presentarse como **aplicación y comparación rigurosa de métodos existentes**, no como un algoritmo nuevo.
```
