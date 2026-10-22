# Capítulo 0 · Introducción

**Autores:** Jassan Alberto Arteta Chavarro y Mateo Bernal · **Curso:** Machine Learning (2026-30)

```{admonition} Dashboard público
:class: tip
La solución desplegada que este libro sustenta técnicamente está en
**<https://machinelearning-202630.onrender.com/>** (Dash + Plotly sobre Render).
Código y datos: <https://github.com/jassan-bit/MachineLearning_202630>.
```

## 0.1 Resumen ejecutivo

Este libro documenta, con criterios de reproducibilidad y honestidad científica, el
proyecto **«Predicción de la volatilidad futura de criptomonedas mediante Machine
Learning»**. Se pronostica, para **BTC, ETH, BNB y XRP** (pares contra USDT en Binance
Spot, 2020-2025 a partir de velas de un minuto), la **volatilidad diaria futura** en
**siete horizontes** (1 a 7 días) y con **cuatro definiciones de ventana** de volatilidad
(7, 14, 21 y 28 días). Cada combinación activo–ventana produce 7 salidas, lo que da
16 configuraciones y 112 series pronosticadas.

El modelo vigente es **HAR-Ridge + XGBoost**: una regresión Ridge sobre resúmenes
de volatilidad de 1, 3, 7, 14 y 28 días (estructura heterogénea autorregresiva,
{cite}`corsi2009har`) combinada con XGBoost {cite}`chen2016xgboost`. En 2025 (358 orígenes
fuera de muestra) alcanza un MSE macro de 0,4647 frente a 0,8282 de la persistencia
(−43,9 %). La revisión final comparó **512 modelos** (7 familias de regresión × 4
procedimientos de búsqueda × 16 configuraciones, más 64 híbridos HAR-XG) con
**validación anidada** y concluyó que **ninguna variante supera de forma estadísticamente
sólida al modelo vigente**. Ese resultado negativo se reporta tal cual en los capítulos
6 y 8.

```{admonition} Alcance y estado de la evidencia
:class: warning
Los periodos 2025 y enero–agosto de 2026 ya fueron examinados repetidamente durante el
desarrollo; por eso todas las evaluaciones fuera de muestra son **retrospectivas** y no
constituyen una prueba confirmatoria. Además, el modelo vigente usa `xgboost` y `scikit-learn`,
por lo que **no cumple** el requisito de un modelo original implementado solo con
`numpy`/`scipy` (ver Capítulos 5 y 8).
```

## 0.2 Planteamiento del problema y relevancia

La volatilidad es la magnitud de riesgo que alimenta la gestión de posiciones, el
dimensionamiento de apalancamiento y la valoración de derivados. En cripto es,
además, inusualmente alta e inestable: según el resumen de Binance Research, el valor
total del mercado osciló en 2025 entre unos 2,4 y 4,2 billones de dólares
{cite}`binanceresearch2025`. La liquidez que hace observable esa dinámica está muy
concentrada: CryptoQuant estima en 18,6 billones de dólares el volumen *spot* de 2025,
con Binance cerca de 7 billones (≈41 % del grupo de las diez mayores plataformas
centralizadas) {cite}`cryptoquant2025forklog`. Pronosticar bien la volatilidad a unos
días vista, y saber **cuánto mejor es realmente** un modelo complejo que una referencia
simple, tiene por tanto valor económico y metodológico.

El problema técnico es de **regresión multisalida sobre series temporales dependientes**:
el objetivo es la desviación estándar móvil poblacional (`ddof=0`) de los retornos
logarítmicos diarios en porcentaje, sin anualizar. Las ventanas solapadas inducen
autocorrelación fuerte (Capítulo 2), de modo que la validación debe respetar el orden
temporal y purgar las etiquetas que atraviesan las fronteras de partición.

## 0.3 Pregunta de investigación y objetivos

```{admonition} Pregunta de investigación
¿Reduce un modelo de Machine Learning con entradas tipo HAR el error de pronóstico de la
volatilidad diaria futura de BTC, ETH, BNB y XRP frente a la persistencia y a un HAR
clásico, y existe evidencia estadística, bajo validación temporal anidada y
comparaciones pareadas, de que alguna variante optimizada supera al modelo vigente?
```

**Objetivo general.** Evaluar con rigor estadístico si el modelo HAR-Ridge + XGBoost y sus
variantes optimizadas mejoran el pronóstico de volatilidad diaria de criptomonedas.

**Objetivos específicos.**

1. Construir un pipeline de datos reproducible (ETL y EDA) desde cierres diarios UTC
   derivados de velas de un minuto, con tratamiento explícito de huecos.
2. Comparar 7 familias de regresión con 4 procedimientos de optimización de
   hiperparámetros mediante validación anidada con purga (media ± desviación del bucle
   externo).
3. Atender las observaciones del docente y documentar la trazabilidad de cada corrección.
4. Contrastar formalmente el modelo vigente frente a las variantes (Diebold–Mariano con
   Holm, MCS 90 %, SPA, intervalos *bootstrap* estacionario y tamaño del efecto).
5. Declarar con honestidad el cumplimiento de las condiciones R1 (novedad), R2
   (implementación propia) y R3 (superioridad verificada).
6. Desplegar un dashboard multipágina y documentar el entorno exacto de reproducción.

## 0.4 Mapa de navegación

| Capítulo | Contenido | Dónde |
|---|---|---|
| 1 | Trazabilidad de correcciones | {doc}`cap01_trazabilidad` |
| 2 | Datos: ETL y EDA | {doc}`cap02_datos_etl_eda` |
| 3 | Benchmark: 512 modelos y mejores combinaciones | {doc}`cap03_benchmark` |
| 4 | Revisión de literatura y novedad | {doc}`cap04_literatura_novedad` |
| 5 | Modelo original: diseño e implementación | {doc}`cap05_modelo_original` |
| 6 | Validación estadística de superioridad | {doc}`cap06_validacion_estadistica` |
| 7 | Dashboard y despliegue | {doc}`cap07_dashboard_despliegue` |
| 8 | Conclusiones, limitaciones e integridad | {doc}`cap08_conclusiones` |
| 9 | Reproducibilidad y referencias | {doc}`cap09_reproducibilidad` |

```{note}
**Diferencia con la plantilla de la rúbrica.** La rúbrica habla de «140 combinaciones» y de
siete modelos *base* con balanceo y optimizador (formulación de clasificación). Este proyecto es de
**regresión**, no hay balanceo de clases y el espacio real evaluado es de **512 modelos**
(Capítulo 3). Se reporta lo que efectivamente se ejecutó, sin reetiquetarlo para encajar en la plantilla.
```
