# GEX_Asset_Management

**Aparato de medida para la señal de *net gamma exposure* (Γ) de Soebhag (2023), y un
piloto de validación de su mecanismo sobre acciones de EE.UU.**

Subcarpeta independiente dentro del repo `GEX-Strategy-`. No comparte código con
`gamma_quant/` — es otra línea de trabajo, con otra fuente de datos (Databento OPRA) y
otro paper de referencia.

- Paper base: Amar Soebhag (2023), *"Option gamma and stock returns"*,
  *Journal of Empirical Finance* 74, 101442 (open access, CC BY).
- **Muestra:** 2021-09-03 → 2026-08-31 · 267 fechas semanales · 15 617 846 contrato-día
  sobre 30 mega-caps más SPY y QQQ.
- **Última ejecución completa:** 2026-09-03. Los 37 ficheros `.py` de este repo son los
  que produjeron los resultados de [`docs/RESULTADOS.md`](docs/RESULTADOS.md).

---

## Empieza aquí

| Documento | Qué contiene |
|---|---|
| [`docs/FICHA_ESTRATEGIA.md`](docs/FICHA_ESTRATEGIA.md) | **Lee esto primero.** Reglas de la estrategia y cuadro completo de métricas contra el SPY |
| [`docs/RESULTADOS.md`](docs/RESULTADOS.md) | Resultado de las puertas de validación y de los siete tests de la señal |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Mapa módulo por módulo y flujo de datos |
| [`docs/preregistros/`](docs/preregistros/) | Parámetros de cada test, congelados antes de correrlo |
| [`docs/PLAN_ORIGINAL.md`](docs/PLAN_ORIGINAL.md) · [`docs/WORKFLOW_PILOTO.md`](docs/WORKFLOW_PILOTO.md) | El plan del que salió el piloto |
| [`reports/`](reports/) | Salida cruda de cada corrida, en texto plano |

---

## 1. Qué mide y por qué importa

Un *market maker* que ha vendido opciones queda con una posición de gamma cuyo signo
depende de la mezcla de calls y puts en circulación. Para mantenerse cubierto tiene que
comprar y vender el subyacente según se mueva el precio. Cuando su **gamma agregada es
negativa**, ese *hedging* va **a favor** del movimiento y amplifica la volatilidad;
cuando es positiva, la amortigua.

Soebhag (2023) convierte esa idea en una señal transversal por acción y día:

```
Γ_i,t  =  0.01 · S_t²  ·  Σ_j ( sign_j · γ_j · OI_j · m_j )  /  ADV$_i,t-1
```

con `sign = +1` para calls y `−1` para puts. El `S` aparece dos veces y es intencional:
el primero convierte acciones a dólares, el segundo convierte "movimiento de 1 dólar" en
"movimiento de 1 %".

**La tesis a falsar:** si el canal de cobertura opera, gamma alta hoy debería anticipar
menos volatilidad la semana siguiente. Coeficiente negativo.

---

## 2. Encuadre honesto — qué es y qué no es este piloto

**Es** un aparato de medida validado y un test del mecanismo sobre 5 años de datos reales.

**No es** una estrategia lista para asignar capital. El resultado principal es que
**el mecanismo no se confirma**: la relación entre Γ y volatilidad futura no sobrevive
al control por volatilidad implícita, porque la gamma es proporcional a `1/σ` por
construcción. El detalle está en [`docs/RESULTADOS.md`](docs/RESULTADOS.md) §4.

Con 30 acciones y quintiles quedan **6 nombres por pata**: ningún test cross-seccional
tiene potencia real. Los papers de referencia usan miles de acciones. Eso es una
limitación estructural del piloto, no un defecto del código.

---

## 3. Qué está construido

- Motor de valuación CRR americano validado **25/25** en su puerta.
- Ingesta reanudable de OPRA con las cinco reglas de oro de la Fase 0 aplicadas.
- Señal completa con sus descomposiciones (ATM/OTM/ITM, fast/slow).
- Siete tests de la señal, **todos pre-registrados** con placebos.
- Cartera neutral por beta y sector, con costos reales.

Lo que **no** está construido (roadmap del plan original, no implementado): módulos de
configuración YAML, universo point-in-time, suite estadística separada y validación
automatizada. Esos directorios no existen en el repo en vez de existir vacíos.

---

## 4. Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # y pon tu DATABENTO_API_KEY dentro
```

`requirements.txt` fija las versiones exactas del entorno donde se produjeron los
resultados, verificadas contra los imports reales del código.

---

## 5. Ejecución

El repo se sube **sin datos** — los parquets pesan más de 1 GB y se regeneran. Un clon
nuevo trae la estructura de carpetas vacía.

```bash
# 0. Verificar la cuenta y el costo antes de gastar (gratis, no descarga la muestra)
python3 fase0_verificacion.py       # rango histórico, schemas, contratos/día, costo
python3 fase0b_optimizacion.py      # en qué franja horaria llega el open interest
python3 fase0c_open_interest.py     # dónde vive el OI y su costo real
python3 fase0d_publishers.py        # la trampa del feed consolidado

# 1. Ingesta — pide confirmación explícita del costo, es reanudable
python3 run_ingesta.py --dry-run                    # solo muestra el costo
python3 run_ingesta.py --scope pilot --freq weekly  # 1 año  (~$72 en weekly)
#   otros scopes: extension / extension2..4 (años hacia atrás), medium, full

# 2. Señal y sus controles de calidad  (cuestan $0)
python3 run_senal.py                # Puerta P2 — magnitudes de Γ
python3 run_sensibilidad.py         # Puerta P2b — estabilidad del ranking

# 3. Test de mecanismo (el objetivo real del piloto)
python3 run_mecanismo.py            # Puerta P3 — panel con efectos fijos
python3 run_mecanismo_fm.py         # Puerta P3b — Fama-MacBeth, spec del paper

# 4. Tests adicionales de la señal (pre-registrados)
python3 run_ortogonal.py            # ¿queda señal sin la parte de volatilidad?
python3 run_vrp.py                  # ¿predice Γ la prima de varianza?
python3 run_regimen.py              # GEX índice → switch momentum/reversal
python3 run_semaforo.py             # GEX índice → semáforo de exposición
python3 run_intradia.py             # momentum intradía condicionado por gamma

# 5. Backtest y presentación
python3 run_backtest.py             # long-short semanal por quintiles
python3 run_backtest_freq.py        # comparación de frecuencias de rebalanceo
python3 run_backtest_neutral.py     # la estrategia final: neutral beta+sector
python3 run_metricas.py             # cuadro de métricas (618 trades)
python3 run_riesgo.py               # VaR, CVaR, bootstrap, estrés 2022

# Test del motor de valuación (sin datos)
python3 tests/test_pricing_gate_p1.py
```

---

## 6. Estructura

```
GEX_Asset_Management/
├── gex/                        el paquete importable
│   ├── pricing/  bsm.py, crr.py            valuación y griegas
│   ├── curves.py                            curva del Tesoro para r(T)
│   ├── signal/   gamma_exposure.py          Ecuación 1 y descomposiciones
│   │             implied_carry.py           dividendo + préstamo de la paridad
│   ├── ingest/   opra.py                    descarga desde Databento
│   ├── equities.py                          carga de precio/volumen del subyacente
│   └── backtest/ engine.py, neutral.py      simulación de cartera
├── run_*.py                    15 scripts, uno por etapa del pipeline
├── fase0*.py                   verificación previa a gastar en Databento
├── tests/                      puerta del motor de valuación
├── docs/                       ficha, resultados, arquitectura, pre-registros
├── reports/                    salida de cada corrida, texto plano
│   └── fase0/                  histórico de la fase de adquisición
└── data/{raw,interim,curated}/ vacías en git; las llenan los scripts
```

---

## 7. Las cinco reglas de oro de la ingesta

Establecidas en la Fase 0 y verificadas empíricamente. No cambiar sin re-verificar.

1. **El open interest se DEDUPLICA, nunca se suma.** OPRA es consolidado: hasta 18
   publishers diseminan el mismo valor por contrato. Verificado: 18/18 publishers y
   3 650/3 650 contratos con valor idéntico, factor de inflación exactamente 18,00×.
2. **Ventana del OI: 10:00–15:00 UTC.** Cubre los dos regímenes de publicación (ráfaga
   a las 13:31/14:31 antes de abril 2023, a las 10:30 después).
3. **Cotizaciones: mid del NBBO de `cbbo-1m` en 19:55–20:00 UTC.** Nunca el último trade:
   en opciones ilíquidas es rancio, y por el hallazgo H1 el error de precio se amplifica.
4. **Multiplicador: el real de `definition`, no 100 fijo.** Hay contratos ajustados de
   10 y 1 000 tras splits y spin-offs.
5. **`instrument_id` se recicla entre días.** Todo join es intra-día; entre fechas se
   empareja por `raw_symbol`.

---

## 8. La regla que gobierna el proyecto

Un resultado negativo bien medido vale más que uno positivo mal medido. Cada test de
este repo se pre-registró con sus parámetros congelados y sus placebos **antes** de
correrlo, y se reporta como salió. Los siete salieron negativos o no concluyentes, y
así están documentados.
