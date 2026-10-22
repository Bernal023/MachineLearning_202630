# Capítulo 8 · Conclusiones, limitaciones e integridad

```{code-cell} ipython3
:tags: [remove-cell]
import json
import pandas as pd
from IPython.display import display
from jbook_utils import R, glue_tbl
```

## 8.1 Síntesis de hallazgos

```{code-cell} ipython3
# Cifras que sustentan las conclusiones, leídas de los artefactos (no copiadas a mano)
rs = json.loads((R / "review_summary.json").read_text(encoding="utf-8"))
mm = pd.read_csv(R / "evaluation" / "macro_metrics.csv")
g = lambda m, y: float(mm[(mm.model == m) & (mm.year == y)]["mse"].iloc[0])
hallazgos = pd.DataFrame({
    "indicador": ["Pipelines evaluados", "MSE vigente 2025", "MSE persistencia 2025", "Reducción vs. persistencia 2025 (%)",
                  "MSE vigente 2026", "MSE persistencia 2026", "Reducción vs. persistencia 2026 (%)",
                  "Candidatas que pasan la regla fuerte", "Prueba confirmatoria sin explorar disponible"],
    "valor": [rs["models"], round(g("HAR-Ridge + XGBoost", 2025), 4), round(g("Persistencia", 2025), 4),
              round(100 * (g("HAR-Ridge + XGBoost", 2025) / g("Persistencia", 2025) - 1), 1),
              round(g("HAR-Ridge + XGBoost", 2026), 4), round(g("Persistencia", 2026), 4),
              round(100 * (g("HAR-Ridge + XGBoost", 2026) / g("Persistencia", 2026) - 1), 1),
              len(rs["strong_retrospective_evidence_candidates"]), rs["untouched_confirmatory_test"]]}).set_index("indicador")
assert hallazgos.loc["Pipelines evaluados", "valor"] == 512 and hallazgos.loc["Candidatas que pasan la regla fuerte", "valor"] == 0
glue_tbl("tbl-hallazgos", hallazgos)
hallazgos
```

```{glue:figure} tbl-hallazgos
:name: tbl-hallazgos
:figwidth: 80%

Indicadores principales del proyecto (MSE en puntos porcentuales al cuadrado).
```

**Fortalezas.** (1) El pronóstico de ML reduce el error de la persistencia en torno a 43 % en 2025 y 2026
({numref}`tbl-hallazgos`). (2) La validación es anidada y purgada, con
verificación automática de la purga ({numref}`tbl-calendario`). (3) La inferencia está predeclarada, ajusta por
comparaciones múltiples y reporta efectos e intervalos (Capítulo 6). (4) Todo el libro se recalcula desde artefactos versionados.

**Debilidades.** (1) Las evaluaciones de 2025 y 2026 son retrospectivas: ya habían sido examinadas, y ni la validación anidada ni el
remuestreo restauran un periodo intacto. (2) Las diferencias entre los mejores modelos son menores que su dispersión
(Capítulos 3 y 6). (3) Los presupuestos de búsqueda son pequeños (8 evaluaciones por búsqueda), por lo que el benchmark es
un piloto acotado y no un ranking universal de optimizadores.

## 8.2 Veredicto sobre las condiciones del modelo original

| Condición | Veredicto | Justificación |
|---|---|---|
| **R1 — Novedad** | **No cumplida / no verificada** | Componentes conocidos; la revisión sistemática no se ejecutó (Cap. 4). |
| **R2 — Implementación propia (numpy/scipy)** | **No cumplida** | El modelo usa `xgboost` y `scikit-learn`; no hay clase propia ni pruebas `pytest` (Cap. 5). |
| **R3 — Superioridad estadística** | **Refutada** | Ninguna candidata supera al vigente: DM–Holm, SPA, MCS e IC no lo respaldan (Cap. 6). |

```{admonition} Tope de calificación
:class: important
Con R1, R2 y R3 sin cumplirse, corresponde presumiblemente el **tope inferior** de la escala de la rúbrica (la
rúbrica menciona topes de 5,0, 4,0 y 3,0). El valor exacto lo determina el docente; este libro no lo asume ni lo
ajusta: reporta los hechos para que la decisión sea verificable.
```

## 8.3 Limitaciones

- **Sin periodo confirmatorio:** 2025 y enero–agosto de 2026 fueron explorados repetidamente.
- **Estacionariedad:** solo se documentó descriptivamente (Cap. 2); no se reportan pruebas ADF/KPSS.
- **Revisión de literatura no ejecutada** (Cap. 4) y **capturas del dashboard pendientes** (Cap. 7).
- **Modelos serializados:** los `.joblib` de los 512 pipelines no están versionados; solo sus hashes.
- **Desviaciones de la rúbrica** (Cap. 6): intervalos por percentiles y no BCa; $d_z$ de Cohen y no $\delta$ de Cliff; Holm y no Benjamini–Hochberg.
- **Rúbrica de clasificación vs. problema de regresión:** no aplican balanceo de clases ni DeLong/McNemar; las «140 combinaciones» son 512 modelos.

## 8.4 Trabajo futuro

1. Evaluar sobre un periodo **realmente nuevo** con reglas fijadas antes de ver los datos.
2. Ejecutar y registrar la revisión sistemática del Capítulo 4.
3. Implementar un modelo propio en `numpy`/`scipy` con clase compatible con `scikit-learn` y sus pruebas, y compararlo bajo el mismo protocolo.
4. Añadir pruebas de estacionariedad, intervalos BCa y $\delta$ de Cliff; ampliar presupuestos de búsqueda.
5. Incorporar inferencia en vivo y capturas al dashboard sin comprometer la memoria del plan gratuito.

## 8.5 Consideraciones éticas

El estudio no formula recomendaciones de inversión: pronostica **volatilidad**, no rentabilidad, y sus resultados son retrospectivos.
Los datos son públicos (Binance Spot; condiciones de uso documentadas en `DATA_LICENSE.md`). No se usan datos personales. Un
pronóstico de volatilidad puede inducir sobreconfianza si se interpreta como garantía; por eso se reportan intervalos e incertidumbre.

## 8.6 Declaración de autoría y uso de herramientas de IA

Autores: **Jassan Alberto Arteta Chavarro** y **Mateo Bernal**. La estructura, la redacción de este libro y parte del código de
análisis se elaboraron con asistencia de un modelo de lenguaje (**Claude, de Anthropic**), que no sustituye la responsabilidad
de los autores. Los resultados numéricos provienen de los artefactos del repositorio y se verifican con celdas `assert` al
compilar. *Los autores deben revisar, ajustar y confirmar esta declaración antes de entregarla.*
