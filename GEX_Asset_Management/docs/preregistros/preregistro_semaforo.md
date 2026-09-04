# Pre-registro — GEX de índice como semáforo de exposición

**Fecha:** 2026-09-03 · congelado ANTES de correr

## Idea

GEX = indicador (de datos de opciones). Acciones = vehículo. NO se operan opciones.
Una sola decisión por semana: cuánta exposición larga al mercado tener.

## Hipótesis

Dealers **largos** gamma → cubren contra tendencia → mercado estable → estar invertido paga.
Dealers **cortos** gamma → cubren a favor → mercado frágil, caídas amplificadas → estar fuera protege.

Predicción falsable: un semáforo guiado por GEX debe superar a comprar y sostener,
en Sharpe neto, y debe reducir el drawdown.

## Definiciones — CONGELADAS

**Indicador `G_t`:** `gamma_exposure` de SPY + QQQ sumado, en la fecha t.
**Normalización:** z-score de `G_t` contra las **26 semanas previas** (expansivo hasta 26).
Solo usa información disponible en t — sin look-ahead.

**Reglas del semáforo (3 estados):**
- `z >= +0.5` → **VERDE**: 100% largo
- `-0.5 < z < +0.5` → **AMARILLO**: 50% largo
- `z <= -0.5` → **ROJO**: 0% (cash, gana la tasa libre de riesgo)

**Activo operado:** dos versiones, ambas reportadas.
- V1: SPY
- V2: las 30 acciones equiponderadas

**Ejecución:** decidir al cierre del viernes t con datos de t, mantener hasta t+1.
Costo: 5 bps por lado sobre el CAMBIO de exposición (no sobre la posición entera).
Cash rinde la tasa del Tesoro a 3M.

## Estrategias comparadas

| # | nombre | qué hace |
|---|---|---|
| A | **Semáforo GEX** | 100/50/0% según z |
| B | Buy & hold | siempre 100% |
| C | Semáforo INVERTIDO (placebo) | 0% en verde, 100% en rojo — DEBE perder |
| D | Semáforo aleatorio (placebo) | 1000 barajados del vector de estados |
| E | Exposición fija equivalente | % largo constante = exposición media de A |

E es clave: descarta que la mejora venga solo de tener menos beta.

## Criterio de éxito — CONGELADO

Respaldada solo si TODAS se cumplen:

1. Sharpe neto de A > Sharpe de B por ≥ 0.15
2. Sharpe neto de A > Sharpe de E por ≥ 0.10 (supera a bajar beta a secas)
3. Max drawdown de A **menor** (en valor absoluto) que el de B
4. Placebo C con Sharpe menor que A
5. A por encima del **percentil 95** de los 1000 semáforos aleatorios (D)
6. Se cumple en **ambas versiones** (SPY y 30 nombres), no en una sola

Si falla cualquiera → no respaldada, se reporta y se cierra.

## Qué NO se hace

No se optimiza el umbral (±0.5), ni la ventana (26), ni los pesos (100/50/0),
ni el horizonte (1 semana). Si pasa, se valida en train/test partido después.
