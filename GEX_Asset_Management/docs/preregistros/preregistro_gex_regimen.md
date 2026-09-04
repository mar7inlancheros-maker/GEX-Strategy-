# Pre-registro — GEX de índice como switch momentum/reversal

**Fecha:** 2026-09-03
**Autor del test:** sesión de análisis GEX PROJECT
**Estado:** congelado ANTES de correr

## Hipótesis

El GEX agregado de índice mide si los dealers están largos o cortos gamma.
- Dealers **largos gamma** (GEX alto) → cubren contra tendencia → mercado **revierte** a 1 semana.
- Dealers **cortos gamma** (GEX bajo) → cubren a favor → mercado **trend-ea** a 1 semana.

Predicción falsable: una estrategia que hace **reversal** en semanas de gamma largo y
**momentum** en semanas de gamma corto debe superar tanto al reversal incondicional como
al momentum incondicional, en Sharpe neto.

## Definiciones — CONGELADAS

**Indicador de régimen (`G_t`):** suma del `gamma_exposure` de SPY + QQQ en la fecha t.
(Ambos son negativos en media; el indicador es su nivel relativo, no el signo.)

**Clasificación de régimen:** z-score de `G_t` contra su media/desvío de las **26 semanas
previas** (expansivo hasta tener 26).
- `z <= -0.5` → **régimen CORTO gamma** (dealers amplifican) → señal = momentum
- `z >= +0.5` → **régimen LARGO gamma** (dealers amortiguan) → señal = reversal
- `-0.5 < z < +0.5` → **neutral** → sin posición (flat)

**Señal de selección (sobre los 30 nombres, nunca SPY/QQQ):**
- Momentum: ranking por retorno de la semana t-1 a t. Largo el quintil top (6 nombres),
  corto el bottom (6). Equal-weight.
- Reversal: lo mismo, invertido.

**Ejecución:** entrar al cierre de t, mantener hasta el cierre de t+1. Spot, equal-weight,
sin apalancamiento. Costo: 5 bps por lado de turnover (consistente con `run_backtest.py`).

**Horizonte de retorno:** 1 semana.

## Estrategias comparadas

| # | nombre | qué hace |
|---|---|---|
| A | **GEX-switch** | momentum en régimen corto, reversal en largo, flat en neutral |
| B | momentum incondicional | siempre momentum |
| C | reversal incondicional | siempre reversal |
| D | GEX-switch invertido (placebo) | reversal en corto, momentum en largo — DEBE perder |
| E | switch aleatorio (placebo) | régimen barajado — DEBE dar ~0 |

## Criterio de éxito — CONGELADO

La hipótesis se considera **respaldada** solo si TODAS se cumplen:

1. Sharpe neto de A > Sharpe neto de B **y** > Sharpe neto de C, ambos por un margen ≥ 0.15.
2. El retorno medio de A tiene **t de Newey-West (4 lags) ≥ 2.0**.
3. El placebo D tiene Sharpe **menor** que A (el signo del switch importa).
4. El placebo E (aleatorio, 1000 barajados) sitúa a A **por encima del percentil 95**
   de la distribución de Sharpe aleatorios.
5. El resultado **no** depende de un solo año: A es positivo en ≥ 3 de los 5 años.

Si falla cualquiera → hipótesis **no respaldada**, se reporta y se cierra.

## Qué NO se hace

- No se optimiza el umbral z (queda en ±0.5), ni la ventana (26 sem), ni el tamaño de
  pata (6), ni el horizonte (1 sem). Si el test pasa, ESO se valida después en
  train/test partido, no ahora.
- No se prueban variantes hasta ver el resultado de esta.
