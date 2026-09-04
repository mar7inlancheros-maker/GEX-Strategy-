# Plan de desarrollo — Backtest de la estrategia de Net Gamma Exposure

> **Nota (documento histórico).** Éste es el plan amplio y aspiracional del
> 2026-09-01. Describe una visión de universo dinámico de ~1000 nombres, ~13
> módulos y un árbol de carpetas que **no coincide con lo implementado**. Lo que
> se construyó de verdad es el **piloto** de [`WORKFLOW_PILOTO.md`](WORKFLOW_PILOTO.md)
> (universo fijo de 30 nombres, quintiles, foco en el mecanismo). El mapa del
> código real está en [`ARQUITECTURA.md`](ARQUITECTURA.md). Se conserva este
> plan porque fija el contrato de replicación del paper y el presupuesto de
> riesgo, que siguen vigentes.

**Paper base:** Amar Soebhag (2023), *"Option gamma and stock returns"*, Journal of Empirical Finance 74, 101442 (open access, CC BY).
**Objetivo del software:** replicar y extender, con precisión auditable, la señal de *net gamma exposure* (Γ) y su backtest cross-seccional sobre acciones individuales de EE.UU., usando Databento como fuente de datos.
**Fecha del plan:** 2026-09-01 · Estado: borrador para aprobación

---

## 1. El contrato de replicación (qué dice exactamente el paper)

### 1.1 La señal

Para cada contrato de opción `j` sobre la acción `i` en el día `t`, con gamma por acción `γ`, open interest `OI` y multiplicador `m` (100 estándar):

- Call: contribución `+ γ · OI · m`
- Put: contribución `− γ · OI · m` (el market maker está corto gamma en los puts que vendió)

Agregando sobre todos los strikes y vencimientos, y normalizando (Eq. 1 del paper):

```
Γ_i,t = 0.01 · S_t² · Σ_j ( sign_j · γ_j · OI_j · m_j )  /  ADV$_i,t-1
```

donde `ADV$` es el volumen promedio en dólares del subyacente en los últimos 21 días hábiles.
Con `m = 100` esto se simplifica a:

```
Γ_i,t = S_t² · ( Σ_calls γ·OI − Σ_puts γ·OI ) / ADV$_i,t-1
```

**Interpretación:** fracción del volumen diario promedio del subyacente que los market makers deben negociar para re-cubrirse ante un movimiento de 1% del precio. Adimensional, comparable entre acciones.

> Nota de precisión: el paper aplica `×100` en la definición de Γ^c/Γ^p y `/100` en Eq. 1; ambos se cancelan y queda un `S²`. Esto **no** es un error del paper: el `×S` inicial convierte acciones a dólares y el `×S/100` posterior convierte "movimiento de $1" en "movimiento de 1%". Implementarlo mal (un solo `S`) cambia el ordenamiento cross-seccional porque introduce un sesgo por nivel de precio.

### 1.2 Muestra y filtros del paper

| Elemento | Especificación del paper |
|---|---|
| Universo | NYSE / AMEX / NASDAQ, CRSP `shrcd ∈ {10,11}` (acción común), `exchcd ∈ {1,2,3}` |
| Filtro de precio | precio > $5 al cierre del mes t |
| Filtro de tamaño | market cap ≥ percentil 20 de NYSE (excluye microcaps) |
| Datos de opciones | OptionMetrics: IV, volumen, OI y greeks por contrato, diario |
| Periodo | 1996-01-01 → 2021-12-31 (311 meses) |
| Winsorización | Γ recortado al 1% y 99% **cada mes** |

### 1.3 Tests que hay que reproducir

1. **Sorts univariados en deciles** de Γ al cierre del mes t, retorno value-weighted del mes t+1. Portfolio L−H autofinanciado.
2. **Alfas ajustados por riesgo:** FF3+MOM, FF5, FF5+MOM, q-factor (HXZ), q-factor+MOM. `t` de Newey–West.
3. **Persistencia / matriz de transición** de deciles (Tabla 3) → determina turnover y costos.
4. **Sorts bivariados condicionales** contra ~20 predictores conocidos.
5. **Regresiones Fama–MacBeth y panel** a nivel de acción.
6. **Factor gamma 2×3** (mediana de tamaño NYSE × percentiles 30/70 NYSE de Γ) y spanning regressions.
7. **Frecuencias diaria y semanal** (§4.3) — Γ predice retorno del día siguiente, coef. ≈ −3.13.
8. **Descomposición** por moneyness (ATM/OTM/ITM) y por vencimiento (`fast` = expira el mes próximo vs `slow`).

**Resultados de referencia (benchmarks de validación):**
- Retorno excedente decil L: 1.45%/mes → decil H: 0.58%/mes, monótono.
- H−L: alfa 6 factores **−0.93%/mes**, `t = −5.40`. Diferencial anual ≈ 10.44%.
- Γ promedio: 0.92 · P25 = 0.05 · mediana = 0.41 · P75 = 1.23 · **21.8% de las observaciones acción-mes con Γ < 0**.
- Correlaciones cross-seccionales de Γ: IV −0.11 · Call OI +0.27 · log(Size) +0.15 · RVOL −0.10.
- El alfa viene **de la pata larga** (Γ bajo): 2.18%–3.31% anual, `t` 2.51–3.78. La pata corta está *spanned* por los factores conocidos (alfa no significativo).

---

## 2. Brecha entre lo que pide el paper y lo que da Databento

Esto es lo que determina el alcance real. Verificado en la documentación de Databento (ver §8 Fuentes).

| Requisito | Databento | Decisión de diseño |
|---|---|---|
| Greeks por contrato (γ) | **No los provee.** Declaración explícita: *"We don't currently provide pre-calculated implied volatility (IV) or greeks."* Es un *feature request* abierto en su roadmap. | Construir motor propio de valuación. **Este es el mayor riesgo de precisión del proyecto** (§4.2). |
| Open interest por contrato | **Sí.** Schema `statistics`, `stat_type = 9` ("open interest"), soportado en `OPRA.PILLAR`. | Fuente primaria de OI. Ver caveat de desfase abajo. |
| Definiciones de contrato (strike, vencimiento, tipo, multiplicador) | Sí, schema `definition`. | Usar el multiplicador real, no `100` fijo (existen contratos ajustados de 10 / 1000 tras splits y spinoffs). |
| Precios de opciones para invertir IV | Sí (`mbp-1`, `tbbo`, `bbo-1m`, `ohlcv-1d`). | **Usar mid del NBBO en la última ventana del día, no el último trade.** El último trade en opciones ilíquidas puede ser de horas antes y contamina la IV. |
| Precio y volumen del subyacente | Sí (equities: `EQUS.SUMMARY` / `XNAS.ITCH` / `XNYS.PILLAR`). | Cierre consolidado + volumen consolidado en dólares para el ADV$ de 21 días. |
| Splits, dividendos, factores de ajuste | Sí — API de corporate actions, 61 tipos de evento, ~6 años de historia point-in-time. | Dividendos discretos para el árbol; factores de ajuste para retornos. |
| **Historia de OPRA** | **Conflicto en las fuentes.** El blog de lanzamiento dice *"starting from March 28, 2023"*; la página comercial de opciones dice *"Since 2013"*. | **No comprometer ninguna fecha hasta ejecutar `metadata.get_dataset_range("OPRA.PILLAR")` con la API key real.** Es el primer entregable (Fase 0). |
| Shares outstanding / market cap | **No.** No aparece en corporate actions ni en reference data. | Sin market cap no hay percentil 20 de NYSE ni ponderación value-weighted. Ver §3.1. |
| Book equity, ROE, IA, NSI, CSI, OP, CP (Compustat) | **No.** | ~7 de los ~20 controles del paper quedan fuera de alcance. Los sorts bivariados se limitan a controles de precio/opciones. |
| Factores FF5, MOM, q-factor | No (no es su negocio) | Ken French Data Library + global-q.org. Gratuitos y suficientes. |
| Curva libre de riesgo | No | FRED (H.15 / curva cero del Tesoro). |

### Caveat estructural del open interest
OPRA disemina el OI como resumen de **inicio de sesión**, es decir refleja el cierre de la sesión anterior. La Γ del paper usa el OI del día `t`. Esto introduce un **desfase de un día que no es opcional** — hay que documentarlo como diferencia metodológica declarada, no ocultarlo. Mitigante: el propio paper (Tabla A.4, panel A) muestra que imponer un lag de un día en Γ **no destruye el resultado**.

---

## 3. Decisiones de diseño forzadas por la brecha

### 3.1 Universo: usar las sub-muestras de robustez del propio paper
No podemos construir el percentil 20 de NYSE sin market cap. Pero el paper **ya validó** su resultado en sub-muestras que no requieren CRSP (§4.1):

- (B) top 1000 acciones más líquidas por la medida de Amihud
- (C) top 1000 acciones con mayor volumen de opciones

**Decisión:** el universo primario del backtest será **top 1000 por volumen en dólares a 21 días** (proxy directo y limpio de liquidez), recalculado mensualmente, con filtro de precio > $5 y exclusión de ETFs/ETNs/ADRs/fondos cerrados por tipo de instrumento. Esto es defendible ante la mesa: *no es una desviación del paper, es una de sus propias especificaciones de robustez.*

**Consecuencia honesta:** perdemos la comparabilidad directa con la Tabla 2 (que es value-weighted con breakpoints NYSE). Reportaremos equal-weighted y **liquidity-weighted** como ponderaciones primarias, y value-weighted solo si conseguimos shares outstanding de una fuente externa.

### 3.2 Definición de la estrategia para asset management
El paper es un long–short cross-seccional market-neutral. Para una mesa de asset management se implementarán **tres variantes**, todas desde el mismo motor:

| Variante | Construcción | Por qué |
|---|---|---|
| **A. L−S puro (replicación)** | Largo decil Γ bajo, corto decil Γ alto | Es la prueba científica. Es el número que valida o refuta el paper. |
| **B. Long-only tilt** | Sobreponderar Γ bajo dentro de un benchmark (p.ej. top 500 líquidas), tracking error objetivo | El paper muestra que el alfa vive en la pata larga; la pata corta está *spanned*. Esta es la variante realmente implementable y la que un mandato de asset management puede comprar. |
| **C. Overlay de señal** | Γ como filtro/tilt sobre una estrategia existente | Encaja con el trabajo previo de rotación sectorial. |

### 3.3 Frecuencia
La ventana de datos disponible es corta (§6). Por eso el backtest será **multi-frecuencia desde el diseño**: mensual (replicación), semanal y **diaria** (§4.3 del paper confirma predictibilidad diaria con `t = −3.02`). La frecuencia diaria es lo que nos da potencia estadística suficiente para concluir algo.

---

## 4. Arquitectura del software

### 4.1 Stack
- Python 3.12 · `databento` (cliente oficial) · `polars` + `pyarrow` · `DuckDB` para consultas analíticas
- `numba` para el motor de valuación (árbol binomial vectorizado)
- `statsmodels` (Newey–West, Fama–MacBeth) · `pytest` para la suite de validación
- Configuración declarativa (`pydantic-settings` + YAML), todo parametrizado, nada hardcodeado
- Lakehouse en Parquet particionado por fecha; snapshots inmutables con hash para reproducibilidad
- Dashboard: **al final y opcional** (Streamlit). La prioridad son los números, no la presentación.

### 4.2 Módulos

```
gex/
├─ config/            # YAML: universo, filtros, fechas, parámetros del modelo
├─ ingest/
│  ├─ databento_opra.py      # definitions, statistics(OI), quotes EOD
│  ├─ databento_equities.py  # cierre + volumen consolidado
│  ├─ corp_actions.py        # splits, dividendos, factores de ajuste
│  └─ external.py            # FF factors, q-factors, curva del Tesoro
├─ pricing/
│  ├─ curves.py              # r(T) interpolada, dividendos discretos proyectados
│  ├─ crr.py                 # árbol Cox-Ross-Rubinstein americano (numba)
│  ├─ bsm.py                 # Black-Scholes (fallback y control de convergencia)
│  └─ iv.py                  # inversión de IV (Brent con bracketing robusto)
├─ signal/
│  ├─ gamma_exposure.py      # Eq.1: Γ total + descomposiciones ATM/OTM/ITM, fast/slow
│  └─ controls.py            # IV, IV skew, CPIV, VoV, O/S, net Δ, net $OI, RVOL, MOM, SREV, MAX, ILQ
├─ universe/                 # construcción PIT del universo, filtros, exclusiones
├─ backtest/
│  ├─ portfolios.py          # sorts univariados/bivariados, factor 2×3
│  ├─ costs.py               # spread, comisión, impacto, costo de préstamo (pata corta)
│  └─ engine.py              # bucle PIT, rebalanceo, contabilidad
├─ stats/                    # Newey–West, Fama–MacBeth, panel, spanning regressions
└─ validate/                 # la suite que define el "90-100% de precisión" (§5)
```

### 4.3 El motor de valuación (el punto crítico)

OptionMetrics — la fuente del paper — calcula greeks de opciones **americanas** sobre acciones individuales con un **árbol binomial (CRR)** que incorpora dividendos discretos y ejercicio anticipado. Usar Black-Scholes europeo introduce error sistemático en la gamma, concentrado en contratos ITM y en subyacentes con dividendo alto.

**Decisión:** implementar CRR americano (200–400 pasos), IV invertida del mid del NBBO, gamma por diferencias finitas centradas sobre el mismo árbol. BSM se conserva solo como control de convergencia (caso sin dividendos → CRR debe converger a BSM con error < 1e-4).

**Mitigante importante:** el paper (§4.4) muestra que el poder predictivo viene de **ATM y OTM**, y de contratos **`slow`** (vencimiento > 1 mes). Justo donde la prima de ejercicio anticipado es pequeña. Es decir: el error del modelo se concentra donde la señal *no* vive. Esto es lo que hace creíble el objetivo de precisión.

**Filtros de calidad de cotización** (antes de invertir IV): descartar bid = 0, spreads cruzados o bloqueados, spread relativo > 50%, precio por debajo del valor intrínseco, IV fuera de [1%, 500%], `OI = 0`. Registrar la tasa de descarte por día — es un indicador de salud del pipeline.

**Detalles de calendario que hay que hacer bien:** todo en horario del Este (ET), snapshot a las 15:59–16:00 ET, y **manejo de medias sesiones** (cierre 13:00 ET) — hay ~3 por año y romperían el snapshot silenciosamente.

---

## 5. Fases, entregables y criterios de aceptación

### Fase 0 — Verificación de datos (1–2 días) · **GO / NO-GO**
No se escribe una línea del motor antes de esto.
- `metadata.get_dataset_range()` para `OPRA.PILLAR` y para el dataset de equities → **fecha real de inicio de la historia**.
- `metadata.get_cost()` para estimar el costo en USD de: definitions + statistics + snapshot EOD de quotes, para 1000 subyacentes, sobre todo el rango disponible.
- Descargar **un día** completo y verificar: ¿llega el OI para todos los contratos? ¿en qué timestamp? ¿cubre los 1000 subyacentes?
- **Entregable:** informe de una página con fecha de inicio, costo estimado, número de contratos/día, y recomendación GO/NO-GO.
- **Criterio de aceptación:** el costo cabe en el presupuesto y la historia es ≥ 24 meses.

### Fase 1 — Ingesta y lakehouse (3–5 días)
- Descarga incremental, reintentos, caché local, registro de linaje.
- Tabla maestra: `(date, underlying, contract_id, strike, expiry, cp, multiplier, bid, ask, mid, oi, volume)`.
- **Aceptación:** reconciliación de conteo de contratos contra las definitions del día; cero fechas faltantes en el calendario de sesiones; test de idempotencia (re-ejecutar no cambia el output).

### Fase 2 — Motor de IV y greeks (5–7 días)
- **Aceptación:**
  - CRR → BSM sin dividendos, error < 1e-4 en precio y < 1e-5 en gamma.
  - Round-trip: precio → IV → precio, error < 1e-6.
  - Greeks contra valores analíticos conocidos en una batería de casos de referencia.
  - Paridad put-call sobre la IV implícita de pares ATM: desviación mediana < 1 punto de vol.
  - Rendimiento: ≥ 1M de contratos/minuto.

### Fase 3 — Constructor de Γ (3–4 días)
- Eq. 1 completa + descomposiciones (ATM/OTM/ITM con umbral `|ln(S/K)| < 0.1`; fast/slow).
- Winsorización mensual 1%/99%.
- **Aceptación — esta es la validación externa más fuerte que existe.** Sobre el periodo solapado, la distribución cross-seccional de nuestra Γ debe parecerse a la Tabla 1 del paper: mediana ≈ 0.4, P25 ≈ 0.05, P75 ≈ 1.2, y **~20–25% de observaciones con Γ < 0**. Además, las correlaciones con IV (−0.11), Call OI (+0.27) y log(Size) (+0.15) deben reproducir el **signo y el orden de magnitud**. Si no, el pipeline está mal y no se avanza.

### Fase 4 — Motor de backtest y estadística (5–7 días)
- Sorts univariados/bivariados, factor 2×3, Fama–MacBeth, panel, spanning regressions, Newey–West.
- **Auditoría anti-look-ahead automatizada:** un test que verifique, para cada feature, que su timestamp de disponibilidad ≤ timestamp de decisión. Falla el build si no.
- **Aceptación:** replicación de la mecánica (monotonía de deciles, signo del H−L, `t` de Newey–West) en el periodo disponible. Ver §6 sobre qué se puede y qué no se puede concluir.

### Fase 5 — Costos, robustez y variantes (4–6 días)
- Costos: medio spread del subyacente (de los datos, no supuesto), comisión, impacto lineal en participación de volumen, **costo de préstamo de la pata corta**.
- Turnover real desde la matriz de transición.
- Robustez: lag de 1 día, Γ promedio del mes, escalado por market cap en vez de volumen, deciles vs quintiles, con/sin filtro de precio.
- Variantes B (long-only tilt) y C (overlay).
- **Aceptación:** todo resultado se reporta **bruto y neto de costos**. Un resultado que solo funciona bruto se reporta como fallido.

### Fase 6 (opcional, al final) — Dashboard

**Total estimado: 21–31 días hábiles de desarrollo**, condicionado al GO de Fase 0.

---

## 6. Lo que este backtest NO va a poder demostrar

Esto tiene que estar en la primera página de cualquier presentación a la mesa.

1. **La historia es demasiado corta.** El paper usa 311 meses. Si OPRA arranca en marzo de 2023, tenemos ~41 meses. Con ~41 observaciones mensuales, un alfa verdadero de 0.93%/mes con la volatilidad reportada **no alcanza significancia estadística ni en el mejor de los casos**. La conclusión mensual será, casi con certeza, "consistente con el paper pero no concluyente".
   → Por eso la frecuencia **diaria** no es un extra: es la única vía a potencia estadística real (≈ 850 días vs 41 meses).
2. **Es puramente out-of-sample y en un régimen distinto.** 2023–2026 es el periodo de explosión de 0DTE y flujo retail en opciones. La estructura de gamma del mercado cambió materialmente respecto a 1996–2021. Un resultado más débil no refuta el paper, y un resultado más fuerte no lo confirma.
3. **~7 de los 20 controles del paper no son replicables** sin Compustat. No podremos afirmar "es distinto de todos los predictores conocidos", solo "es distinto de los predictores de precio y de opciones".
4. **Sin market cap no hay value-weighting ni breakpoints NYSE.** Los números no serán directamente comparables con la Tabla 2.
5. **Riesgo de p-hacking.** Con 41 meses, probar muchas especificaciones garantiza encontrar una que funcione. Mitigante obligatorio: **pre-registrar la especificación primaria en este documento antes de mirar un solo resultado**, y reportar todas las especificaciones probadas, no solo la mejor.

---

## 7. Presupuesto de riesgo del proyecto

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Historia de OPRA < 24 meses | Mata la variante mensual | Fase 0 antes de cualquier compromiso; pivote a diaria/semanal |
| Costo de datos de quotes de OPRA | OPRA es uno de los feeds más voluminosos del mundo | `metadata.get_cost()` en Fase 0; usar snapshot de 1 minuto en vez de tick completo; limitar a 1000 subyacentes |
| Error del motor de greeks | Contamina toda la señal | Suite de aceptación de Fase 2 + validación contra Tabla 1 en Fase 3 |
| Desfase del OI | Diferencia metodológica | Declararlo; el paper ya lo testeó (Tabla A.4A) |
| Sin fuente de market cap | Pierde value-weighting | Evaluar fuente externa (Sharadar/Nasdaq Data Link) en Fase 0 |
| Look-ahead accidental | Invalida todo | Auditoría automatizada en el build |

---

## 8. Fuentes

- Soebhag, A. (2023). *Option gamma and stock returns.* Journal of Empirical Finance 74, 101442. (PDF en esta carpeta)
- Databento — dataset OPRA.PILLAR, blog de lanzamiento de OPRA, página de opciones, schema `statistics`, corporate actions (ver enlaces en el mensaje que acompaña este plan)
- Referencias metodológicas del paper: Barbon & Buraschi (2020); Baltussen, Da, Lammers & Van Bekkum (2021); Ni, Pearson, Poteshman & White (2021); Fama & French (1993, 2015); Hou, Xue & Zhang (2015); Bali & Hovakimian (2009); Xing, Zhang & Zhao (2010)
