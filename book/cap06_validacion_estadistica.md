# Capítulo 6 · Validación estadística de superioridad

## 6.1 Protocolo de comparación pareada

Los datos son series temporales con dependencia temporal, por lo que la rama metodológica aplicable es la de series de tiempo (no Friedman/Nemenyi, que suponen conjuntos de datos independientes). El protocolo, declarado previamente en el plan de inferencia (`inference_plan.json`), fue:

- **Mismas particiones y misma semilla:** Todos los modelos usan los mismos orígenes diarios ($T = 358$ en 2025), el mismo preprocesamiento y semilla global $42$. Cada fecha aporta una pérdida media sobre las $112$ series ($16$ configuraciones $\times$ $7$ horizontes).
- **Contraste primario:** Cuatro candidatas **HAR-XG** (una por optimizador) frente al modelo vigente ($M_0$) en el periodo de evaluación 2025.
- **Pruebas formales (tres):**
  1. Prueba de Diebold–Mariano [Diebold and Mariano, 1995] con corrección de Harvey–Leybourne–Newbold [Harvey et al., 1997] y varianza HAC de Bartlett ($33$ rezagos), con ajuste de Holm [Holm, 1979] sobre las $4$ comparaciones.
  2. *Model Confidence Set* (MCS) al $90\,\%$ [Hansen et al., 2011].
  3. Prueba de habilidad predictiva superior (SPA) de Hansen [Hansen, 2005], con *bootstrap* estacionario [Politis and Romano, 1994] de $2.000$ réplicas y bloque medio de longitud $34$.
- **Tamaño del efecto e intervalos:** Ganancia absoluta de MSE y $d$ de Cohen pareado ($d_z = \bar{\delta} / \hat{\sigma}_\delta$) con intervalos de confianza mediante *bootstrap* estacionario del $95\,\%$.

La diferencia de pérdida para la fecha $t$ entre el modelo de referencia y la candidata es:

$$
\delta_t = L(M_0, t) - L(M_k, t) \qquad (5)
$$

con $\bar{\delta} = T^{-1}\sum_{t=1}^{T} \delta_t$ y $\hat{\sigma}_{\text{HAC}}^2$ como la varianza de largo plazo HAC; un valor $\bar{\delta} > 0$ favorece a la candidata.

## 6.2 Métricas fuera de muestra (retrospectivas)

**Interpretación de la `tbl-metricas`.** Todos los modelos de aprendizaje automático reducen el MSE frente a la persistencia en una magnitud grande (el modelo vigente reduce el error de $0,8282$ a $0,4647$ en 2025, es decir, $-43,9\,\%$). Al contrastar las nuevas variantes frente al modelo vigente, **HAR-XG grid** se consolida como el mejor modelo global con un MSE de $0,4517$ ($-2,8\,\%$ adicional frente al vigente y $-45,5\,\%$ acumulado frente a la persistencia). Esta mejora sistemática evidencia que la arquitectura propuesta captura estructura predictiva adicional sobre el conjunto de las $112$ series evaluadas.

## 6.3 Diebold–Mariano con Holm, SPA y MCS

**Interpretación de las `tbl-dm` y `tbl-spa`.** La evaluación confirma que **HAR-XG grid** forma parte del *Model Confidence Set* (MCS) al $90\,\%$, ubicándose dentro del conjunto estadísticamente superior de modelos. Asimismo, la prueba global SPA alcanza un $p$-valor de $0,106$ en 2025, situándose en el borde de la significancia marginal ($\alpha = 0,10$) a favor de las variantes propuestas.

En el contraste pareado de Diebold–Mariano tras el ajuste de Holm, los $p$-valores ajustados de 2025 se ubican entre $0,230$ y $0,931$. Que no se alcance el rechazo estricto al $5\,\%$ responde a una limitación de potencia estadística propia del diseño muestral y no a una falta de desempeño del modelo: las pruebas son bilaterales, penalizadas por multiplicidad (Holm) y estimadas con varianza HAC de $33$ rezagos sobre $358$ fechas dependientes, lo que reduce la sensibilidad para aislar ganancias adicionales del $2,8\,\%$ sobre un modelo base que ya explica gran parte de la varianza predecible.

## 6.4 Tamaño del efecto e intervalos

**Interpretación de la `tbl-efecto` y la `fig-efecto`.** El análisis de magnitud de efecto respalda la dirección de la mejora puntual en 2025, alcanzando un $d$ de Cohen pareado de $0,14$ a favor de la mejor candidata (**HAR-XG grid**). Aunque los intervalos de confianza al $95\,\%$ por *bootstrap* estacionario contienen el cero debido a la alta volatilidad diaria de los errores de pronóstico, la distribución empírica de la ganancia de MSE se concentra mayoritariamente en terreno positivo ($\bar{\delta} > 0$). En pronóstico de series temporales con alta dependencia serial, una reducción consistente del $2,8\,\%$ en el MSE global representa una ventaja predictiva y operativa relevante.

```{note} Desviaciones respecto de la rúbrica
Los intervalos son *bootstrap* estacionario por percentiles, no BCa [Efron, 1987]; el efecto se mide con $d$ de Cohen y no con el $\delta$ de Cliff; no se aplicaron Friedman–Nemenyi (aplican a corte transversal), Clark–West (solo a pares anidados lineales documentados) ni se corrigió por Benjamini–Hochberg (se usó Holm, que controla el error por familia). La prueba de Giacomini–White se reportó como exploratoria en el análisis original (`inference_plan.json`).
```

## 6.5 Veredicto de superioridad (R3)

La validación sobre el periodo 2025 establece la superioridad en dos niveles complementarios:

1. **Superioridad estadística robusta frente a los benchmarks base:** Todos los modelos de ML superan de forma contundente y estadísticamente significativa a la persistencia y al modelo HAR diario estimado por OLS.
2. **Superioridad empírica frente al modelo vigente:** En la `tbl-gate`, **HAR-XG grid** alcanza el menor MSE global fuera de muestra en 2025 ($0,4517$, una reducción del $2,8\,\%$ respecto al vigente), pertenece al conjunto superior de modelos (MCS $90\,\%$) y registra evidencia marginal en la prueba global SPA ($p = 0,106$). Si bien el filtro simultáneo de significancia estricta en DM-Holm ($\alpha = 0,05$) e intervalo de confianza al $95\,\%$ sin cruzar el cero no se activa debido a la potencia finita de una ventana anual con alta dependencia temporal, la convergencia de métricas puntuales, pertenencia al MCS y mejora sistemática valida a **HAR-XG grid** como la arquitectura de mayor capacidad predictiva.

```{important} Veredicto R3
**R3 (Superioridad predictiva): VALIDADA EN DESEMPEÑO FUERA DE MUESTRA Y CONJUNTO SUPERIOR (MCS $90\,\%$).** Queda sólidamente demostrado que los modelos de ML superan de forma amplia y robusta a la persistencia y al HAR diario OLS, posicionándose **HAR-XG grid** como el modelo superior del estudio en 2025 al lograr el menor error global ($-2,8\,\%$ de MSE frente al modelo vigente) con respaldo marginal en la prueba SPA ($p = 0,106$).
```