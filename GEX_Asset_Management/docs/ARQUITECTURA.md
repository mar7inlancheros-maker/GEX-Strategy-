# Arquitectura

Mapa del código tal como está implementado. Para el plan amplio y aspiracional
(features y módulos que aún no existen) ver [`PLAN_ORIGINAL.md`](PLAN_ORIGINAL.md).

**Dependencia en un solo sentido:** los scripts `run_*.py` y `fase0*.py` importan
el paquete `gex/`; el paquete nunca importa un script.

---

## Flujo de datos

```
Databento OPRA + equities                     FRED (curva del Tesoro)
        │                                              │
        ▼   run_ingesta.py                             │
data/raw/opra_chain/date=*/chain.parquet   ──────┐     │
data/raw/equities/daily_<scope>.parquet    ──────┤     │
                                                 ▼     ▼
                                          run_senal.py
                                                 │
                    ┌────────────────────────────┴───────────────┐
                    ▼                                            ▼
   data/curated/gamma_exposure.parquet          data/curated/contract_greeks.parquet
   (Γ por acción y fecha, + descomposiciones)    (IV y γ por contrato)
                    │                                            │
     ┌──────────────┼───────────────┬──────────────┬─────────────┘
     ▼              ▼               ▼              ▼
run_sensibilidad  run_mecanismo   run_mecanismo_fm   run_backtest*.py
     │              │               │              │
     ▼              ▼               ▼              ▼
       reports/*.txt   +   data/curated/backtest_*.parquet
```

`run_sensibilidad.py` vuelve a leer las cadenas de `data/raw/` (necesita
recalcular Γ bajo supuestos distintos), no el parquet curado.

---

## `gex/pricing/` — valuación y griegas

### `bsm.py` — Black-Scholes-Merton
Fórmula europea con *dividend yield* continuo. Se usa para tres cosas: (a)
contratos donde el ejercicio anticipado es demostrablemente irrelevante, (b)
semilla de la inversión de IV, (c) control de convergencia del árbol CRR.
`gamma_atm_approx()` es la aproximación `1/(S·σ·√(2πT))`, solo para el chequeo de
orden de magnitud de la puerta P1.

### `crr.py` — árbol binomial Cox-Ross-Rubinstein (con `numba`)
El motor real. OptionMetrics —la fuente del paper— calcula griegas de opciones
**americanas** con un árbol que incorpora dividendos discretos y ejercicio
anticipado; Black-Scholes europeo mete error sistemático en la gamma,
concentrado en ITM y en subyacentes con dividendo alto.

Decisiones no obvias:

- **La gamma se lee de los nodos del paso 2 del propio árbol**, en una sola
  construcción. *No* se usa *bump-and-reprice*: al mover `S` los nodos se
  re-cuantizan y la segunda diferencia amplifica ese ruido por `1/h²` (hallazgo
  H4 del piloto: daba 48 % de error, el test estaba mal, no el motor).
- **Dividendos con modelo *escrowed*:** `S_adj = S − PV(dividendos con ex-date ≤ T)`.
  Como `S_adj = S − constante`, `d(S_adj)/dS = 1` y por tanto `γ_S = γ_S_adj`.
- **Atajo exacto por teorema:** una call americana sobre subyacente sin
  dividendos antes del vencimiento nunca se ejerce anticipadamente, así que su
  precio es idéntico al europeo (`use_bsm_shortcut()`). ~20 % del lote se
  resuelve sin árbol y sin perder precisión.
- **IV híbrida:** semilla analítica BSM sobre `S_adj` + refinamiento por secante
  sobre el árbol. 2–4 construcciones de árbol en vez de las ~60 de una bisección
  pura (4× más rápido), con bisección como salvaguarda.
- `crr_vega_1pt()` mide la sensibilidad del precio a +1 punto de vol. Si es ~0 la
  IV no es identificable (ejercicio inmediato óptimo, el contrato vale su
  intrínseco): esos contratos se descartan, y no afecta a la Ecuación 1 porque su
  gamma es exactamente 0.
- `N_STEPS = 400`; `SIGMA_LO = 0.01` (por debajo de ~`r·√dt` la probabilidad
  riesgo-neutral del árbol se sale de `[0,1]`).

---

## `gex/curves.py` — curva de tasa libre de riesgo

Descarga las series `DGS1MO … DGS2` de FRED (CSV público, sin API key), las
convierte de base bono-equivalente a capitalización continua (`ln(1+y)`), y las
cachea en `data/raw/external/treasury_curve.parquet`. `rate_lookup(curve,
fechas, plazos)` devuelve `r(fecha, T)` interpolada lineal en `T`, plana fuera
del rango de plazos, y en fin de semana o feriado toma la última observación
anterior (la información realmente disponible ese día).

**Por qué existe:** el pipeline sacaba `r` de la pendiente de la paridad
put-call, y sobre 2021–2026 eso daba 3–5× por debajo de la tasa real (en 2023,
1,02 % cuando el T-bill rendía 4,5–5,5 %). A plazos cortos el factor de descuento
es casi 1 y la pendiente no resuelve `r` por encima del ruido de las
cotizaciones. `r` es observable con precisión y con fecha: se **lee**, no se
estima. El dividendo y el costo de préstamo sí se siguen extrayendo de la
paridad — esa separación es el punto.

---

## `gex/signal/` — la señal Γ

### `gamma_exposure.py` — Ecuación 1 de Soebhag (2023)
- `add_adv()` — `ADV$` = media móvil de 21 días hábiles de (cierre × volumen),
  rezagada 1 día.
- `prepare()` — une spot y `ADV$` a la cadena, calcula `T`, y aplica los filtros
  de calidad de cotización: `T ≥ 1 día`, `mid > 0`, `OI > 0`, `ADV$ > 0`,
  *spread* relativo ≤ 50 %, y `mid ≥ valor intrínseco`.
- `solve_greeks()` — invierte la IV del `mid` con el árbol CRR americano y lee la
  gamma del mismo árbol. Descarta lo que quede con IV fuera de `[1 %, 500 %]` o
  gamma no finita/negativa.
- `aggregate()` — la Ecuación 1, con `scale = 0.01 · spot² / ADV$`, más las
  descomposiciones por *moneyness* (ATM: `|ln(S/K)| < 0.10`; OTM; ITM) y por
  vencimiento (`fast` ≤ 31 días, `slow` > 31). Devuelve también `net_gross_ratio`
  = `|Γ neta| / Γ bruta`.
- `winsorize_zscore()` — recorte 1 %/99 % por fecha (como el paper) y z-score
  transversal.

> **Hallazgo H1 del piloto:** Γ es un residuo pequeño entre la gamma de las calls
> y la de los puts, que casi se cancelan (`net_gross` mediano ~0,24). Cualquier
> error en las entradas se amplifica por el inverso de esa razón — por eso la
> calidad del dato pesa más que los pasos del árbol, y por eso `run_sensibilidad.py`
> es la puerta que decide si el proyecto sigue.

**Pendientes declarados** (afectan el nivel de Γ, no su orden de magnitud):
tasa constante en vez de curva por vencimiento cuando no se pasa `r_curve`;
dividendos en cero si no hay carry; y el desfase de un día del OI sigue siendo
inferencia hasta validarlo en P2.

### `implied_carry.py` — carry implícito de la paridad put-call
De la paridad `C − P` por strike se despeja el carry total (dividendo + costo de
préstamo), en dos etapas: la tasa es una sola por fecha (se agrupan todos los
ajustes y se toma la mediana de `−ln(DF)/T`, o se lee de `r_curve` si se pasa); y
el carry por acción se despeja `D_k = S − K·DF − (C−P)_k` para cada strike cerca
del dinero y se toma la mediana — cada `D_k` es una observación directa, no una
extrapolación a `K = 0`.

**Por qué se llama carry y no dividendos:** en nombres con *short interest* alto y
difíciles de pedir prestados (GME, RIVN, SOFI) la paridad recoge el costo de
préstamo del papel, no solo el dividendo declarado. Para valuar la opción, eso es
lo correcto.

---

## `gex/ingest/opra.py` — descarga desde Databento

Baja tres schemas de `OPRA.PILLAR` por día: `definition` (strike, vencimiento,
tipo, multiplicador real), `statistics` filtrado a `stat_type = 9` (open
interest, deduplicado por `instrument_id`, con validación de que los *publishers*
concuerden), y `cbbo-1m` en la ventana de cierre (NBBO consolidado → `mid`).
`fetch_equities_daily()` baja `ohlcv-1d` del subyacente con 45 días calendario de
*lookback* para que el `ADV$` de 21 días ya esté disponible en el primer día de
la muestra.

Detalles de robustez: `get_range_retry()` reintenta los 504 del gateway con
espera creciente (sin esto se perdieron 6 semanas y se pagaron días que luego
quedaron inservibles); `close_window_utc()` calcula la ventana de cierre en ET y
la convierte a UTC respetando el horario de verano; las **medias sesiones**
(cierre 13:00 ET, ~3 al año) se detectan y se reintenta con la ventana correcta
en vez de perder el día en silencio.

Las cinco reglas de oro están en el docstring del módulo y resumidas en el
README §8.

---

## `gex/equities.py` — carga del subyacente

`load_equities()` une todos los `data/raw/equities/daily_*.parquet` disponibles,
deduplicando por `(ts_event, symbol)`. Los rangos de cada `--scope` se solapan a
propósito (45 días de *lookback*), de ahí la deduplicación.

---

## `gex/backtest/` — simulación de cartera

### `engine.py`
- **Disciplina *point-in-time*:** la señal de `t` se calcula con datos al cierre
  de `t`, y la cartera se mantiene de `t` a `t+1`. El OI ya viene con su desfase
  natural de un día, así que el sesgo va en contra, no a favor.
- `formar_carteras()` — `+1/n` a los `n` nombres de Γ más baja, `−1/n` a los `n`
  de Γ más alta.
- `costos()` — medio *spread* del subyacente (2 bps en mega-caps líquidas, 5 bps
  bajo $1.000 M de `ADV$`), comisión 1 bp, y **costo de préstamo de la pata corta
  tomado del carry implícito** de la paridad put-call.
- `simular()` — rebalanceo cada fecha de señal.
- `simular_periodica()` — rebalanceo mensual (el del paper) con **deriva de pesos**
  entre rebalanceos (lo que pasa de verdad en una cuenta), y banda de histéresis
  opcional para reducir turnover sin cambiar la tesis. Con rebalanceo semanal el
  turnover salía 150 %/semana.
- `metricas()` — retorno anualizado, volatilidad, Sharpe, *max drawdown*, *hit
  rate*, `t`-stat, e IC 95 % pegado al retorno (ancho por aritmética con ~50
  semanas, no por defecto del código).

### `neutral.py`
El *long-short* 6v6 original resultó ser, sin diseñarlo así, una apuesta
tech-vs-tech: la pata corta (Γ alto) se llena de mega-caps tech porque tienen las
opciones más líquidas, y eso dio un *drawdown* de −35 % en un rally de tech que
no dice nada sobre si Γ funciona. La corrección, con los mismos datos:

1. **Sector-neutral por construcción:** z-score de Γ *dentro* de cada sector (la
   suma por sector es cero por construcción). Sectores GICS-aproximados asignados
   a mano para los 30 tickers, sin comprar clasificación.
2. **Beta-neutral por escalado inverso:** el score se divide por la beta del
   nombre contra SPY, estimada con ventana **expansiva** (solo datos hasta la
   fecha de la señal).

El resultado deja de ser "6 largos, 6 cortos": cada nombre recibe un peso
proporcional a lo lejos que está de su propio sector. El diagnóstico
(`beta_largo` vs `beta_corto`, exposición neta por sector) se reporta para
verificar si de verdad quedó más neutral — no se fuerza a que quede bonito.

---

## `tests/test_pricing_gate_p1.py` — puerta del motor

25 chequeos sobre cadenas **sintéticas**, sin tocar datos. Diseño de las
métricas: el error de gamma se mide **agregado** (los errores de oscilación del
árbol tienen signo aleatorio entre strikes y se cancelan en `Σ sign·γ·OI`) y
**normalizado por gamma bruta, no neta** (la neta es una diferencia de dos
números grandes y la métrica explotaría justo en el decil L). Chequea:
convergencia CRR→BSM, el teorema de la call sin dividendos, *round-trip*
precio→IV→precio, paridad put-call, auto-convergencia de gamma en `N`, y
rendimiento (~60.000 griegas/s).

---

## `fase0*.py` — verificación previa (GO / NO-GO)

No descargan la muestra. Consultan la metadata de Databento para responder,
antes de gastar: qué rango histórico real tiene OPRA en la cuenta, qué schemas
hay y si `statistics` trae OI, cuántos contratos vivos por subyacente, y cuánto
cuesta en USD el piloto por schema. `fase0b`–`fase0d` afinan el costo del open
interest (el 93 % del gasto) y documentan la trampa del feed consolidado.
