# Pre-registro — Momentum intradía condicionado por gamma

**Fecha:** 2026-09-03 · congelado ANTES de correr
**Costo:** $0 — solo parquets locales, sin llamadas a Databento

## Base teórica

Baltussen, Da, Lammers & Martens (2021), *JFE*, "Hedging Demand and Market
Intraday Momentum": el retorno del último tramo del día es predicho por el
retorno del resto del día, y el efecto se explica por la cobertura de gamma.
Revierte en los días siguientes — por eso no aparece a horizonte semanal.

## Hipótesis

Dealers **cortos** gamma → cubren a favor del movimiento → el gap de apertura
**continúa** durante el día.
Dealers **largos** gamma → cubren en contra → el gap **revierte**.

Predicción falsable: el coeficiente de `intradía ~ gap` es **más positivo** en
régimen de gamma corto que en gamma largo, y la diferencia es significativa.

## Definiciones — CONGELADAS

**Tramos del día:**
- `gap_t` = apertura_t / cierre_{t−1} − 1
- `intra_t` = cierre_t / apertura_t − 1

**Gamma (índice):** GEX(SPY)+GEX(QQQ) del snapshot semanal, z-score contra las
26 semanas previas. El valor de la semana se aplica a los días hábiles
**posteriores** a esa fecha (nunca al mismo día ni a días previos — sin look-ahead).

**Gamma (acción):** `gamma_exposure` de cada nombre, z-score **cross-seccional**
por fecha. Mismo esquema de propagación.

**Régimen:** `z <= -0.5` → CORTO · `z >= +0.5` → LARGO · resto → NEUTRAL

**Estrategia (entrada en apertura, salida en cierre, solo spot):**
- CORTO gamma → posición = signo(gap)  [continuación]
- LARGO gamma → posición = −signo(gap)  [reversión]
- NEUTRAL → sin posición

**Costos:** 5 pb ida y vuelta en SPY · 10 pb ida y vuelta en acciones individuales.

## Estrategias comparadas

| # | nombre |
|---|---|
| A | Condicionada por gamma (la hipótesis) |
| B | Continuación incondicional (siempre signo(gap)) |
| C | Reversión incondicional (siempre −signo(gap)) |
| D | A invertida (placebo) — DEBE perder |
| E | Régimen barajado, 1000 veces (placebo) |

## Criterio de éxito — CONGELADO

Respaldada solo si TODAS se cumplen:

1. β(gap) en régimen CORTO − β(gap) en régimen LARGO > 0, con **t ≥ 2.0**
2. Sharpe neto de A > B **y** > C, por ≥ 0.15
3. Sharpe de D < Sharpe de A
4. A por encima del percentil 95 de los 1000 barajados (E)
5. A positivo **sin 2022**
6. Se cumple a nivel índice (SPY) **o** a nivel acción — basta uno, pero se
   reportan ambos

## Qué NO se hace

No se optimiza el umbral (±0.5), la ventana (26 sem), ni los costos. Si pasa,
se valida en train/test partido después, no aquí.
