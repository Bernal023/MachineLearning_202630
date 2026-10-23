# Capítulo 0 · Introducción

**Autores:** Jassan Arteta y Mateo Bernal · Universidad del Norte, Maestría en Matemáticas · Machine Learning (2026-30)

```{admonition} Dashboard público
:class: tip
La solución desplegada que este libro sustenta técnicamente está en
**<https://machinelearning-202630.onrender.com/>** (Dash + Plotly sobre Render).
Código y datos: <https://github.com/jassan-bit/MachineLearning_202630>.
```

## 0.1 Resumen ejecutivo

Este libro documenta, con criterios de reproducibilidad y honestidad científica, el proyecto **«Predicción de la volatilidad de criptomonedas con Machine Learning: HAR-Ridge + XGBoost frente a seis modelos de
referencia»** (Jassan Arteta y Mateo Bernal, Universidad del Norte, Maestría en Matemáticas). Se pronostica la **volatilidad diaria futura** de **BTC, ETH, BNB y XRP** (pares contra USDT en Binance Spot; cierres
de un minuto, 2023–2025, 1.096 fechas diarias) en **siete horizontes** (1 a 7 días) y con **cuatro ventanas** de volatilidad (7, 14, 21 y 28 días). No se predicen precios ni se demuestra rentabilidad.

El modelo propuesto, **HAR-Ridge + XGBoost**, combina una regresión Ridge sobre resúmenes de volatilidad de 1, 3, 7, 14 y 28 días (estructura heterogénea autorregresiva, {cite}`corsi2009har`) con XGBoost
{cite}`chen2016xgboost`, mediante un peso $\lambda$ elegido en validación temporal. Se compara con k-NN, Ridge, Lasso, Random Forest, XGBoost y SVR lineal (más un Decision Tree añadido) sobre los **mismos objetivos y
fechas**: 358 orígenes diarios de 2025. Con la ventana $w=21$ alcanza RMSE 0,38601 y $R^2=0{,}8445$ y es el mejor modelo frente a los rivales del Grid original (7/7 contrastes DM–Holm7 favorables). Cuando los
rivales se **reoptimizan** con Random Search, Optuna TPE y DEAP, el respaldo estadístico cae a 2–4 de 7 y, con Optuna, **Lasso mejora al original en las cuatro métricas sin ventaja significativa**.
La conclusión es que **el respaldo al HAR depende de la búsqueda y que la superioridad requiere evidencia estadística ajustada** (Capítulos 6 y 8).

```{admonition} Alcance y estado de la evidencia
:class: warning
El periodo 2025 ya fue examinado repetidamente durante el desarrollo; por eso la evaluación fuera de muestra (solo 2025; el periodo de 2026 del repositorio no se usa para concluir) es **retrospectiva** y no constituye una
prueba confirmatoria. Además, el modelo usa `xgboost` y `scikit-learn` como aprendices base, por lo que el requisito de implementación propia (R2) se cumple solo de forma **parcial**
(Capítulos 5 y 8). La novedad (R1) se declara solo para la combinación específica del modelo, no para el enfoque (Capítulo 4).
```

## 0.2 Planteamiento del problema y relevancia

La volatilidad es la magnitud de riesgo que alimenta la gestión de posiciones, el dimensionamiento de apalancamiento y la valoración de derivados, y es el insumo de medidas como VaR y *expected shortfall*
{cite}`jorion2007var,mcneil2015qrm`. En cripto es, además, inusualmente alta e inestable: según el resumen de Binance Research, el valor total del mercado osciló en 2025 entre unos 2,4 y 4,2 billones de
dólares {cite}`binanceresearch2025`. La liquidez que hace observable esa dinámica está muy concentrada: CryptoQuant estima en 18,6 billones de dólares el volumen *spot* de 2025, con Binance cerca de 7 billones
(≈41 % del grupo de las diez mayores plataformas centralizadas) {cite}`cryptoquant2025forklog`. Pronosticar bien la volatilidad a una semana vista, y saber **cuánto mejor es realmente** un modelo complejo
que referencias simples, tiene valor económico y metodológico.

El problema técnico es de **regresión multisalida sobre series temporales dependientes**: el objetivo es la desviación estándar móvil poblacional de los retornos logarítmicos diarios en porcentaje, sin anualizar.
Las ventanas solapadas inducen autocorrelación fuerte (Capítulo 2), de modo que la validación debe respetar el orden temporal y separar las etiquetas de las fronteras de partición.

## 0.3 Pregunta de investigación y objetivos

```{admonition} Pregunta de investigación
HAR-Ridge + XGBoost frente a seis modelos: **¿cuánto mejora y qué explica su ventaja observada?** ¿Es esa ventaja estadísticamente sólida cuando los rivales se optimizan con distintas búsquedas?
```

**Objetivo general.** Anticipar la volatilidad de los siguientes siete días y medir cuánto mejora HAR-Ridge + XGBoost frente a seis modelos de referencia.

**Objetivos específicos.**

1. Construir un pipeline de datos reproducible (ETL y EDA) desde cierres de un minuto, con tratamiento explícito de huecos y diagnóstico temporal de Bitcoin.
2. Comparar los modelos con validación cruzada temporal (Grid Search) y, en una ampliación, con tres búsquedas adicionales y validación anidada.
3. Interpretar el aporte de los componentes temporal (HAR-Ridge) y no lineal (XGBoost).
4. Evaluar errores, diferencias de MSE y dependencia temporal sobre los mismos objetivos y fechas (DM con HAC y HLN, ajuste de Holm, intervalos e IC de la diferencia de MSE).
5. Atender las observaciones del docente y declarar con honestidad el cumplimiento de R1 (novedad), R2 (implementación propia) y R3 (superioridad verificada).
6. Documentar el dashboard desplegado y el entorno exacto de reproducción.

## 0.4 Mapa de navegación

| Capítulo | Contenido | Dónde |
|---|---|---|
| 1 | Trazabilidad de correcciones | {doc}`cap01_trazabilidad` |
| 2 | Datos: ETL y EDA | {doc}`cap02_datos_etl_eda` |
| 3 | Modelos benchmark, validación cruzada temporal y ampliación anidada | {doc}`cap03_benchmark` |
| 4 | Revisión de literatura y novedad | {doc}`cap04_literatura_novedad` |
| 5 | HAR-Ridge + XGBoost: formulación, derivación y complejidad | {doc}`cap05_modelo_original` |
| 6 | Validación estadística: DM, Holm, IC y veredicto | {doc}`cap06_validacion_estadistica` |
| 7 | Dashboard y despliegue | {doc}`cap07_dashboard_despliegue` |
| 8 | Conclusiones, limitaciones e integridad | {doc}`cap08_conclusiones` |
| 9 | Reproducibilidad y referencias | {doc}`cap09_reproducibilidad` |

```{note}
**Diferencia con la plantilla de la rúbrica.** La rúbrica habla de «140 combinaciones» y de
siete modelos *base* con balanceo y optimizador (formulación de clasificación). Este proyecto es de
**regresión**, no hay balanceo de clases y el espacio real evaluado es de **512 modelos**
(Capítulo 3 §3.4). Se reporta lo que efectivamente se ejecutó, sin reetiquetarlo para encajar en la plantilla.
```
