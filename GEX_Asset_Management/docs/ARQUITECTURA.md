# Arquitectura

Mapa del paquete `gex/` y de los scripts. Verificado contra el código el 2026-09-04:
los 37 ficheros `.py` de este repo son idénticos a los que produjeron los resultados
de [`RESULTADOS.md`](RESULTADOS.md).

---

## El paquete `gex/`

Librería importable. No se ejecuta directamente.

### `gex/pricing/` — valuación y griegas

| Módulo | Qué hace |
|---|---|
| `bsm.py` | Black-Scholes europeo. Sirve de control de convergencia del árbol y de semilla analítica para la inversión de IV. |
| `crr.py` | Árbol Cox-Ross-Rubinstein **americano**, vectorizado con numba. 400 pasos en producción. Calcula precio, IV invertida y gamma por diferencias finitas centradas sobre el mismo árbol. |

**Por qué CRR y no Black-Scholes:** OptionMetrics —la fuente del paper— usa árbol
binomial para opciones americanas sobre acciones individuales. Usar BSM europeo sobre
puts con dividendos introduce 4,9 % (ITM), 9,7 % (ATM) y 9,6 % (OTM) de error en Σγ.
Para calls sin dividendos antes del vencimiento el atajo analítico es **exacto por
teorema**, así que ~20 % del lote se resuelve sin árbol.

**Trampa documentada en el código:** el *bump-and-reprice* no sirve como referencia de
gamma sobre un árbol — los nodos se re-cuantizan al mover S y la segunda diferencia
amplifica ese ruido por 1/h². Las referencias válidas son la gamma analítica de BSM en
el caso europeo y la auto-convergencia en N para el americano.

### `gex/curves.py` — la tasa libre de riesgo

Descarga la curva diaria del Tesoro de FRED (series `DGS1MO`, `DGS3MO`, `DGS6MO`,
`DGS1`, `DGS2`; sin API key), la cachea en `data/raw/external/treasury_curve.parquet`,
convierte de base bono a capitalización continua con `ln(1+y)` e interpola `r(T)` al
plazo de cada contrato.

Existe porque el método anterior —estimar r de la pendiente de la paridad put-call—
daba 3-5× por debajo de la tasa real. El detalle está en `RESULTADOS.md` §7.

### `gex/signal/` — la señal

| Módulo | Qué hace |
|---|---|
| `gamma_exposure.py` | Ecuación 1 de Soebhag y sus descomposiciones (ATM/OTM/ITM, fast/slow). Filtros de calidad de cotización, inversión de IV, agregación y winsorización. |
| `implied_carry.py` | Dividendo y costo de préstamo extraídos de la paridad put-call, por (fecha, acción, vencimiento). Acepta `r_curve=` para leer la tasa en vez de estimarla. |

**Regla de oro del open interest:** OPRA es un feed consolidado y hasta 18 publishers
diseminan el *mismo* valor de OI por contrato. Se **deduplica, nunca se suma** — sumar
inflaría Gamma 18×. El pipeline valida cada día que los publishers concuerden y avisa
si discrepan, en vez de promediar en silencio.

**Hallazgo que gobierna el diseño (H1):** Γ es una diferencia de dos números grandes.
La razón `|Γ neta| / Γ bruta` va de 2 % a 26 %, así que cualquier error en los datos de
entrada se amplifica por el inverso de esa razón. Por eso el presupuesto de esfuerzo va
a la calidad de la cotización (mid del NBBO, nunca el último trade) y no a subir los
pasos del árbol.

### `gex/ingest/opra.py` — descarga

Cliente de Databento para `definition`, `statistics` (open interest) y `cbbo-1m`
(cotizaciones). Reintentos con espera creciente ante los 504 del gateway, que aparecen
en los días grandes cuando varios workers piden a la vez.

**Ventanas horarias**, ambas calibradas empíricamente:
- `OI_WIN = (10:00, 15:00)` UTC — OPRA movió la ráfaga de OI de las 13:31/14:31 a las
  10:30 alrededor de abril 2023; la ventana cubre ambos regímenes.
- `QUOTE_WIN = (19:55, 20:00)` UTC — los últimos 5 minutos de sesión. Pedir la sesión
  entera cuesta 38× más por el mismo dato útil.

**`instrument_id` se recicla entre días.** Todo join es intra-día, y el emparejamiento
entre fechas se hace por `raw_symbol` (el símbolo OSI).

### `gex/equities.py` — precio y volumen del subyacente

`load_equities()` concatena todos los `data/raw/equities/daily_*.parquet` y deduplica
por `(ts_event, symbol)`. Los ficheros se solapan 45 días a propósito, para que el
promedio móvil de 21 días ya esté disponible en el primer día de cada tramo.

### `gex/backtest/` — simulación

| Módulo | Qué hace |
|---|---|
| `engine.py` | Formación de carteras por quintiles, rebalanceo periódico con deriva de pesos, costos (spread, comisión, préstamo), métricas. |
| `neutral.py` | Construcción sector-neutral y beta-escalada: z-score de Γ dentro del sector, dividido por beta. Betas point-in-time con ventana expansiva. |

---

## Los scripts

Se ejecutan en este orden. Los marcados con 💰 gastan dinero en Databento; el resto
cuesta 0 porque solo lee lo que ya está en disco.

### Verificación previa a gastar

| Script | Qué responde |
|---|---|
| `fase0_verificacion.py` | 💰(gratis) Rango histórico real, schemas disponibles, contratos/día, costo por schema |
| `fase0b_optimizacion.py` | En qué franja horaria llega el open interest |
| `fase0c_open_interest.py` | Dónde vive el OI y cuánto cuesta pedir solo esa franja |
| `fase0d_publishers.py` | La trampa del feed consolidado: 18 publishers, el mismo OI |

### Pipeline principal

| Script | Etapa | Salida |
|---|---|---|
| `run_ingesta.py` | 💰 Descarga OPRA + equities | `data/raw/` |
| `run_senal.py` | Puerta P2 — calcula Γ y valida magnitudes | `data/curated/`, `reports/p2_senal.txt` |
| `run_sensibilidad.py` | Puerta P2b — estabilidad del ranking | `reports/p2b_sensibilidad.txt` |
| `run_mecanismo.py` | Puerta P3 — panel con efectos fijos | `reports/p3_mecanismo.txt` |
| `run_mecanismo_fm.py` | Puerta P3b — Fama-MacBeth, spec del paper | `reports/p3b_mecanismo_fm.txt` |

### Tests de la señal (todos pre-registrados)

| Script | Hipótesis |
|---|---|
| `run_ortogonal.py` | ¿Queda señal al quitarle a Γ su parte de volatilidad? |
| `run_vrp.py` | ¿Predice Γ la prima de varianza? Decide si vale la versión con opciones |
| `run_regimen.py` | GEX de índice como switch momentum/reversal |
| `run_semaforo.py` | GEX de índice como semáforo de exposición |
| `run_intradia.py` | Momentum intradía condicionado por gamma (Baltussen 2021) |

Los parámetros de cada uno se congelaron **antes** de correrlo, en
[`preregistros/`](preregistros/).

### Backtest y presentación

| Script | Qué produce |
|---|---|
| `run_backtest.py` | Long-short por quintiles, rebalanceo semanal |
| `run_backtest_freq.py` | Comparación de cuatro frecuencias de rebalanceo |
| `run_backtest_neutral.py` | **La estrategia final**: neutral por beta y sector |
| `run_metricas.py` | Cuadro de métricas de presentación (618 trades) |
| `run_riesgo.py` | VaR, CVaR, bootstrap por bloques, estrés 2022 |

### Mantenimiento

`diagnostico_llamadas.py`, `limpiar_dias_est.py`, `reparar_y_completar.py` — utilidades
puntuales. Cada una explica su propósito en la cabecera.

### Tests

`tests/test_pricing_gate_p1.py` — la puerta del motor de valuación. No necesita datos y
sale con código 1 si algún chequeo falla.

---

## Flujo de datos

```
Databento OPRA          FRED                 yfinance / EQUS.SUMMARY
      │                   │                            │
      ▼                   ▼                            ▼
run_ingesta.py      gex/curves.py              data/raw/equities/
      │              (curva r(T))                      │
      ▼                   │                            ▼
data/raw/opra_chain/      │                    gex/equities.py
  date=YYYY-MM-DD/        │                            │
      │                   │                            │
      └───────────────────┴────────────┬───────────────┘
                                       ▼
                               run_senal.py
                     (filtros → IV → CRR → gamma → Ecuación 1)
                                       │
                                       ▼
                          data/curated/gamma_exposure.parquet
                          data/curated/contract_greeks.parquet
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              puertas P2b/P3/P3b   tests de señal     backtests
                    │                  │                  │
                    └──────────────────┴──────────────────┘
                                       ▼
                                  reports/*.txt
```

---

## Reglas de datos

| Ruta | Regla |
|---|---|
| `data/raw/opra_chain/` | **Irreemplazable** si se pierde el acceso pagado. Fuera de git, con copia propia |
| `data/{interim,curated}/` | Se regeneran desde `raw/`. Fuera de git |
| `reports/` | Texto plano, versionado: es el registro de qué salió |
| `.env` | Secretos (`DATABENTO_API_KEY`). **Nunca** se comitea; copia de `.env.example` |
