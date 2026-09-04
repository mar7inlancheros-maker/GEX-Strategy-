# Ficha de estrategia — GEX Neutral

**Periodo:** 2021-09-03 → 2026-08-28 · 266 semanas · 60 rebalanceos mensuales
**Universo:** 30 mega-caps de EE.UU. (SPY y QQQ quedan fuera del ranking)
**Vehículo:** acciones al contado. **Nunca se operan opciones** — solo alimentan el indicador.
**Capital de referencia:** 100 000 USD
**Generado por:** `run_metricas.py` · última ejecución 2026-09-04

> **Advertencia.** El mecanismo económico que justifica esta señal **no se confirmó**
> (ver [`RESULTADOS.md`](RESULTADOS.md) §4). Las cifras de abajo son un registro
> histórico, no una expectativa de rendimiento.

---

## 1. Cómo funciona

Cinco pasos, el último día hábil de cada mes. Toda la información usada está disponible
antes de decidir — no hay look-ahead.

| Paso | Qué se hace |
|---|---|
| **1 · Señal** | Para cada acción se suma la gamma de todos sus contratos vivos, positiva en calls y negativa en puts, ponderada por open interest y normalizada por el volumen en dólares de 21 días:<br>`Γ = 0,01 · S² · Σ(signo · γ · OI · m) / ADV$` |
| **2 · Valuación** | IV invertida del punto medio del NBBO (ventana 15:55–16:00 ET) con árbol **Cox-Ross-Rubinstein americano de 400 pasos**. Tasa leída de la curva del Tesoro; dividendo y costo de préstamo extraídos de la paridad put-call. |
| **3 · Ranking** | z-score de Γ **dentro de cada sector**, dividido por la beta:<br>`score = −z(Γ) / β`<br>El signo negativo implementa la tesis: **gamma baja → largo**. |
| **4 · Entrada** | Peso proporcional al score, normalizado a 100 % largo y −100 % corto. Exposición bruta 200 %, neta 0 %. Sin apalancamiento adicional. |
| **5 · Salida** | Se mantiene un mes; los pesos derivan con el mercado. Cada mes se recalcula desde cero. **Sin stop-loss, sin objetivo de beneficio, sin gatillos intramensuales.** |

**Parámetros congelados:** rebalanceo mensual · 400 pasos en el árbol · IV ∈ [1 %, 500 %] ·
descarte de cotizaciones con spread relativo > 50 %, bid = 0, OI = 0 o precio bajo el
intrínseco · banda ATM `|ln(S/K)| < 0,10` · corte fast/slow en 31 días.

---

## 2. Resultados frente al benchmark

> Todos los Sharpe de esta ficha son **exceso sobre la tasa libre de riesgo** (curva del
> Tesoro a 3 meses, media del periodo 3,76 %). Los ficheros `reports/backtest*.txt`
> imprimen `retorno/volatilidad` sin restarla y por eso muestran un número mayor.

### Muestra y actividad

| Métrica | GEX Neutral |
|---|---|
| Trades totales | **618** (311 largos / 307 cortos) |
| Tiempo en mercado | 100 % (exposición bruta 200 %, neta 0 %) |
| Duración media por trade | 12,7 sem (ganadoras 12,1 · perdedoras 13,6) |
| Rebalanceos | 60 · turnover medio 192 % |

### Rendimiento

| Métrica | GEX Neutral | SPY |
|---|---|---|
| Retorno total neto | **72,42 %** | 69,30 % |
| Capital final | **172 425 USD** | 169 297 USD |
| CAGR | **11,24 %** | 10,84 % |
| Retorno bruto (antes de costos) | 94,06 % | — |
| Expectativa por trade | 0,063 % | — |

### Riesgo y dispersión

| Métrica | GEX Neutral | SPY |
|---|---|---|
| Máximo drawdown | **−18,17 %** (−18 167 USD) | −24,80 % (−24 798 USD) |
| Duración del drawdown | **35 sem** | 106 sem |
| Semanas bajo el agua (racha) | **85** | 105 |
| Volatilidad anual | 15,73 % | 16,07 % |
| Racha perdedora máxima | 7 sem | 7 sem |

### Eficiencia y calidad

| Métrica | GEX Neutral | SPY |
|---|---|---|
| Win rate por trade | 59,1 % | — |
| Win rate semanal | 53,4 % | 54,5 % |
| Profit factor | 1,17 | — |
| Payoff ratio | 0,80 | — |
| **Ratio de Sharpe** | **0,52** | 0,49 |
| **Ratio de Sortino** | **0,90** | 0,75 |
| **Ratio de Calmar** | **0,62** | 0,44 |

### Exposición al mercado

| Métrica | GEX Neutral |
|---|---|
| Alfa de Jensen anual | +6,35 % · **t = 0,93 (no significativo)** |
| Beta | 0,25 |
| R² contra SPY | 0,06 |

### Fricción operativa

| Concepto | % del capital | USD |
|---|---|---|
| Comisión y spread | 4,43 % | 4 432 |
| Préstamo de la pata corta | 7,40 % | 7 402 |
| **Total** | **11,83 %** | **11 834** |

El préstamo es el costo dominante y es intrínseco a la pata corta.

---

## 3. Lectura

La estrategia **iguala al SPY en retorno con un tercio de su beta, menos drawdown y
recuperación tres veces más rápida**. Sortino y Calmar la favorecen con claridad.

Pero el alfa de +6,35 % tiene **t = 0,93**: con 266 semanas no se puede rechazar que el
alfa verdadero sea cero. Lo demostrable es el **perfil de diversificación** (beta 0,25,
R² 0,06), no una ventaja de retorno.

Un detalle estructural que conviene conocer: el ranking por Γ separa los nombres por
volatilidad implícita, porque la gamma es proporcional a 1/σ por construcción
(corr(Γ, 1/IV) = +0,573 frente a corr(Γ, log ADV$) = +0,062). La pata larga acaba
concentrada en nombres volátiles (RIVN, MU, AMD, con IV media de 48,3 %) y la corta en
defensivos (KO, PG, WMT, IV media 25,8 %). Esa exposición es un efecto lateral del
diseño, no una decisión.

### Comportamiento por régimen

| Periodo | GEX Neutral | SPY | 30 nombres EW |
|---|---|---|---|
| 2022 completo | **+15,4 %** | −16,4 % | −42,4 % |
| Caída ene–oct 2022 | **+25,1 %** | −23,8 % | −46,8 % |
| Resto (2023–2026) | 9,1 % | 20,1 % | 39,8 % |

Toda la ventaja de retorno viene de 2022. Lo que **sí** es consistente en todos los años
es la reducción de drawdown. Y no es una cobertura: en las 15 semanas con SPY por debajo
de −3 %, la neutral también perdió (−1,18 % en media) y su beta en caídas es +0,20.

---

## 4. Reproducir

```bash
python3 run_ingesta.py --scope pilot --freq weekly    # descarga (cuesta dinero)
python3 run_senal.py                                  # Γ por acción y día
python3 run_backtest_neutral.py                       # cartera neutral
python3 run_metricas.py                               # este cuadro
```

Versión visual con gráficos interactivos: [`ficha_estrategia.html`](ficha_estrategia.html)
