# GEX_Asset_Management

**Aparato de medida para la señal de *net gamma exposure* (Γ) de Soebhag (2023), y
un piloto de validación de su mecanismo sobre acciones de EE.UU.**

Subcarpeta independiente dentro del repo `GEX-Strategy-`. No comparte código con
`gamma_quant/` — es otra línea de trabajo, con otra fuente de datos (Databento
OPRA) y otro paper de referencia.

- Paper base: Amar Soebhag (2023), *"Option gamma and stock returns"*,
  *Journal of Empirical Finance* 74, 101442 (open access, CC BY).
- El PDF del paper y el plan amplio original están en [`docs/`](docs/).

---

## 1. Qué mide y por qué importa

Un *market maker* que ha vendido opciones queda con una posición de gamma cuyo
signo depende de la mezcla de calls y puts en circulación. Para mantenerse
cubierto tiene que comprar y vender el subyacente según se mueva el precio.
Cuando su **gamma agregada es negativa**, ese *hedging* va **a favor** del
movimiento (compra cuando sube, vende cuando baja) y amplifica la volatilidad;
cuando es positiva, la amortigua.

Soebhag (2023) convierte esa idea en una señal transversal por acción y día:

```
Γ_i,t  =  0.01 · S_t²  ·  Σ_j ( sign_j · γ_j · OI_j · m_j )  /  ADV$_i,t-1
```

con `sign = +1` para calls y `−1` para puts, `γ` la gamma por acción del
contrato, `OI` el open interest, `m` el multiplicador y `ADV$` el volumen medio
en dólares del subyacente a 21 días hábiles. Con `m = 100` se reduce a

```
Γ_i,t  =  S_t²  ·  ( Σ_calls γ·OI  −  Σ_puts γ·OI )  /  ADV$_i,t-1
```

**Interpretación:** la fracción del volumen diario típico del subyacente que los
*market makers* tendrían que operar para re-cubrirse ante un movimiento del 1 %.
Es adimensional y comparable entre acciones. El paper documenta que las acciones
con Γ más bajo (o negativo) rinden más y tienen más volatilidad realizada
después — un diferencial anualizado de ~10 % en el *long-short* 1996–2021.

> **El `S` aparece dos veces a propósito.** El primero pasa de acciones a
> dólares; el segundo pasa de "movimiento de $1" a "movimiento de 1 %".
> Implementarlo con un solo `S` sesga el ranking hacia las acciones caras.

---

## 2. Encuadre honesto — qué es y qué no es este piloto

> Esto **no es un backtest concluyente de una estrategia.** Es un **piloto de
> validación de mecanismo e infraestructura.** Con ~1 año de datos y ~30 nombres
> no existe forma estadística de confirmar o refutar un alfa de 0,93 %/mes. Lo
> que sí se puede probar con potencia real es: (a) que el pipeline calcula Γ con
> las magnitudes de la literatura, y (b) que el **eslabón de volatilidad** del
> paper (Γ bajo → más volatilidad realizada futura) opera en estos datos.

Detalle completo del alcance, los tres errores del plan original que se
corrigieron, y qué se puede afirmar al final: [`docs/WORKFLOW_PILOTO.md`](docs/WORKFLOW_PILOTO.md).

---

## 3. Qué está construido

| Componente | Estado |
|---|---|
| `gex/pricing/` — BSM + árbol CRR americano, IV invertida, gamma del árbol | Hecho. Puerta **P1: 25/25 chequeos** |
| `gex/curves.py` — curva del Tesoro (FRED) para `r(T)` | Hecho |
| `gex/signal/` — Ecuación 1, descomposiciones ATM/OTM/ITM y *fast/slow*, carry implícito de la paridad put-call | Hecho |
| `gex/ingest/opra.py` — descarga OPRA (definitions + OI + NBBO de cierre) y equities desde Databento | Hecho, con las 5 reglas de oro de la Fase 0 |
| `gex/backtest/` — motor *point-in-time*, rebalanceo periódico, costos reales, versión neutralizada por beta y sector | Hecho |
| Puertas P2 (magnitudes de Γ), P2b (sensibilidad del ranking), P3/P3b (mecanismo) | Hecho, como scripts `run_*.py` |
| features adicionales, selección dinámica de universo, más tests automatizados | Pendiente. Ver `docs/PLAN_ORIGINAL.md` |

---

## 4. El pipeline

```
  fase0_*.py ........... verifican en Databento qué datos hay y cuánto cuestan
        │                (no descargan la muestra; son GO/NO-GO)
        ▼
  run_ingesta.py ....... descarga la muestra  →  data/raw/opra_chain/date=*/chain.parquet
        │                                        data/raw/equities/daily_<scope>.parquet
        ▼
  run_senal.py ......... Ecuación 1 sobre lo descargado (cuesta $0)
        │                →  data/curated/gamma_exposure.parquet     (Γ por acción-fecha)
        │                →  data/curated/contract_greeks.parquet    (IV + γ por contrato)
        ▼
  run_sensibilidad.py .. ¿el ranking de Γ sobrevive a la incertidumbre de tasa/dividendos/IV?
  run_mecanismo.py ..... panel  RV_{t+1} ~ Γ_t  con efectos fijos por acción
  run_mecanismo_fm.py .. la especificación real del paper (Fama-MacBeth + descomposición de identificación)
  run_backtest*.py ..... cuánto habría rendido el portafolio (3 diseños de rebalanceo, con costos)
```

Cada `run_*.py` deja además un informe legible en `reports/`.
`tests/test_pricing_gate_p1.py` valida el motor de valuación sin datos (cadenas
sintéticas).

Mapa módulo por módulo, con las decisiones no obvias: [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## 5. Instalación

Requiere **Python 3.9 o superior**.

```bash
cd GEX_Asset_Management
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # y pon tu DATABENTO_API_KEY
```

`.env` y `.venv/` están en `.gitignore` y nunca se comitean. La API key se lee
de `.env` o de la variable de entorno `DATABENTO_API_KEY`; nunca se pasa por
línea de comandos.

---

## 6. Ejecución

El repo se sube **sin datos** — los parquets de OPRA/equities pesan >1 GB y se
regeneran. Un clon nuevo trae la estructura de carpetas vacía.

```bash
# 0. Verificar la cuenta de Databento y el costo del piloto (gratis, no descarga la muestra)
python3 fase0_verificacion.py       # rango histórico, schemas, contratos/día, costo por schema
python3 fase0b_optimizacion.py      # en qué franja horaria llega el open interest
python3 fase0c_open_interest.py     # dónde vive el OI y su costo real
python3 fase0d_publishers.py        # la trampa del feed consolidado (18 publishers, mismo OI)

# 1. Ingesta — pide confirmación explícita del costo, es reanudable
python3 run_ingesta.py --dry-run           # solo muestra el costo, no descarga
python3 run_ingesta.py --scope pilot       # 1 año   (~$450 en Databento)
#   otros scopes: extension / extension2..4 (años hacia atrás), medium (26m), full (100m)

# 2. Señal y sus controles de calidad  (cuestan $0: usan lo ya descargado)
python3 run_senal.py                       # Puerta P2 — magnitudes de Γ vs la Tabla 1 del paper
python3 run_sensibilidad.py                # Puerta P2b — estabilidad del ranking

# 3. Test de mecanismo  (el objetivo real del piloto)
python3 run_mecanismo.py                   # Puerta P3 — panel con efectos fijos
python3 run_mecanismo_fm.py                # Puerta P3b — Fama-MacBeth, la spec del paper

# 4. Backtest
python3 run_backtest.py                    # long-short por quintiles, rebalanceo semanal
python3 run_backtest_freq.py               # comparación de frecuencias de rebalanceo
python3 run_backtest_neutral.py            # neutralizado por beta y sector

# Test del motor de valuación (sin datos)
python3 tests/test_pricing_gate_p1.py      # sale con código 1 si algún chequeo falla
```

Utilidades de mantenimiento de datos (ver la cabecera de cada una):
`diagnostico_llamadas.py`, `limpiar_dias_est.py`, `reparar_y_completar.py`.

---

## 7. Estructura

```
GEX_Asset_Management/
├── gex/                        el paquete importable (no se ejecuta directo)
│   ├── pricing/  bsm.py, crr.py            valuación y griegas
│   ├── curves.py                            curva del Tesoro para r(T)
│   ├── signal/   gamma_exposure.py          Ecuación 1 y descomposiciones
│   │             implied_carry.py           dividendo + costo de préstamo de la paridad put-call
│   ├── ingest/   opra.py                    descarga desde Databento
│   ├── equities.py                          carga de precio/volumen del subyacente
│   └── backtest/ engine.py, neutral.py      simulación de cartera
├── run_*.py                    scripts de ejecución (uno por etapa del pipeline)
├── fase0*.py                   verificación de datos y costo, previa a gastar en Databento
├── tests/                      test_pricing_gate_p1.py — puerta del motor de valuación
├── docs/                       ARQUITECTURA.md, WORKFLOW_PILOTO.md, PLAN_ORIGINAL.md
└── data/{raw,interim,curated}/ vacías en git; las llenan los scripts
```

| Ruta | Regla |
|---|---|
| `data/raw/opra_chain/` | Irreemplazable si se pierde el acceso pagado. Fuera de git, con copia propia |
| `data/{interim,curated}/` | Se regeneran desde `raw/`. Fuera de git |
| `reports/` | Lo escribe una máquina. Fuera de git |
| `.env` | Secretos (`DATABENTO_API_KEY`). Nunca se comitea; copia de `.env.example` |

---

## 8. Las cinco reglas de oro de la ingesta (Fase 0)

No cambiar sin re-verificar contra Databento. Detalle en `gex/ingest/opra.py`.

1. **Open interest: deduplicar, nunca sumar.** OPRA es un feed consolidado —
   hasta 18 *publishers* diseminan el **mismo** valor de OI por contrato. Sumar
   inflaría Γ hasta 18×. El pipeline además valida cada día que concuerden.
2. **Ventana del OI: 10:00–12:00 UTC.** Única franja con `stat_type = 9`. Pedir
   el día entero cuesta 4,4× por el mismo dato.
3. **Cotización: mid del NBBO en 15:55–16:00 ET**, nunca el último trade — en
   opciones ilíquidas es rancio y el error se amplifica en Γ (ver hallazgo H1 en
   `docs/WORKFLOW_PILOTO.md`).
4. **Multiplicador real de `definition`**, no 100 fijo — hay contratos ajustados
   de 10 y 1000 tras splits y spinoffs.
5. **`instrument_id` se recicla entre días.** Todo *join* es intradía.

---

## 9. La regla que gobierna el proyecto

Un umbral escondido en un `if` es un grado de libertad que nadie audita y que
acaba sobreajustado. Toda decisión de modelado — convención de signo, definición
de Γ, lag del OI, escalón de costos, tasa — vive como constante nombrada en la
cabecera de su módulo, no enterrada en el código. El desfase estructural de un
día en el open interest de OPRA se **declara** como diferencia metodológica, no
se oculta (el propio paper, Tabla A.4A, muestra que ese lag no destruye el
resultado).
