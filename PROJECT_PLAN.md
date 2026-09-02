# PLAN DE PROYECTO — Estrategia de Gamma Exposure de Dealers (GEX)

**Proyecto:** `gamma_quant`
**Inicio:** 31 de agosto de 2026
**Estado:** Fase 1 cerrada (auditoría de entorno y datos). Fases 2–6 construibles ya. Fases 7–15 **bloqueadas por datos**.

---

## 0. LO PRIMERO, Y LO INCÓMODO

> **Hoy no podemos responder a la pregunta de investigación, y ninguna cantidad de código lo cambia.**

La hipótesis a contrastar — *el posicionamiento gamma de los dealers predice retornos y
volatilidad realizada futuros* — exige un **panel histórico** de cadenas de opciones: para
cada fecha `t`, la rejilla completa `(strike × vencimiento × tipo)` con **open interest**,
volatilidad implícita y el spot sincronizado. El OI no es negociable: el GEX es un *stock*
de posicionamiento, y sin OI no hay GEX, sino un proxy de volumen que mide otra cosa.

**Tenemos cero días de ese panel.** Verificado en vivo el 2026-08-31:

| Fuente | ¿Cadena? | ¿OI? | ¿Griegas? | **¿Histórico?** | Coste |
|---|---|---|---|---|---|
| CBOE delayed quotes (JSON) | Sí (SPY 13.514 / SPX 28.648 contratos) | Sí | Sí | **No — sólo snapshot** | Gratis |
| yfinance option chain | Sí (más pobre) | Sí | No | **No — sólo snapshot** | Gratis |
| yfinance subyacente | n/a | n/a | n/a | Sí (SPY 1993–, SPX 1927–, VIX 1990–) | Gratis |

El lado del subyacente es real y profundo. **El lado de opciones no tiene histórico a
ningún precio que paguemos hoy.** Por tanto:

1. Las fases 2–6 (arquitectura, griegas, motor GEX, validación sintética, interfaces de
   proveedor) se construyen **ya** y son útiles de verdad: son el aparato de medida.
2. Las fases 7–15 (análisis exploratorio, contrastes predictivos, estrategias, backtest,
   walk-forward, placebo, ML, veredicto) **no producen ni un número publicable** hasta que
   exista histórico real.
3. Cualquier backtest sobre cadenas sintéticas es una **prueba del código**, jamás evidencia
   sobre el mercado. Todo artefacto así lleva el sello `SINTÉTICO — NO ES EVIDENCIA`.

**Una conclusión terminal legítima de este proyecto es "datos insuficientes para contrastar
la hipótesis".** Eso es un hallazgo, no un fracaso, y no se tapará con una curva de equity
sintética.

### 0.1 Decisión tomada (31-08-2026)

De las tres vías planteadas — (A) comprar histórico, (B) archivar hacia delante, (C) sólo
sintético — se elige:

> **(B) ARCHIVAR HACIA DELANTE, GRATIS.** Snapshot diario del endpoint de CBOE desde hoy.
> Sin coste, ~250 observaciones al año.

**Consecuencia asumida explícitamente:** la primera lectura honesta del contraste no llega
antes de **finales de 2027**, y una muestra 2026–2027 no puede hablar de dependencia de
régimen (no contiene ni un mercado bajista serio). No se ha autorizado proveedor de pago
(A) ni serie GEX derivada de terceros (C). Si esa decisión cambia, el §4.3 queda listo para
retomarse y las fases 2–6 hacen que la compra sea utilizable el mismo día.

**El reloj corre desde el primer snapshot.** Ésa es la razón de que el archivador sea
prioridad de la Fase 6 y no del final.

---

## 1. HALLAZGOS DE LA FASE 1 — AUDITORÍA

### 1.1 Repositorio

- El directorio `GEX Strategy/` estaba **vacío**. Terreno virgen.
- La raíz de git está un nivel por encima: `Python - AM/`, compartida con
  `Global Value Investing Strategy/` (60 ficheros versionados, ~12.200 LOC). Rama `main`,
  árbol limpio.
- No hay `CLAUDE.md`. No hay código de opciones, ni de GEX, ni datos de opciones cacheados.

### 1.2 Entorno Python

- **Python 3.14.2**, intérprete global, **sin entorno virtual**.
  `C:\Users\usuario\AppData\Local\Programs\Python\Python314\python.exe`

| Instalado | Versión | | Ausente | Para qué |
|---|---|---|---|---|
| numpy | 2.4.0 | | `pytest` | no hace falta — la convención del repo son scripts de test autónomos |
| pandas | 2.3.3 | | `pyarrow` | **instalado ya** (25.0.1): las cadenas son anchas |
| scipy | 1.17.0 | | `xgboost`/`lightgbm` | sólo Fase 14, opcional |
| scikit-learn | 1.8.0 | | `arch` | baselines GARCH, opcional |
| statsmodels | 0.14.6 | | `ruff`/`mypy` | higiene, opcional |
| matplotlib | 3.10.9 | | | |
| seaborn | 0.13.2 | | | |
| yfinance | 1.0 | | | |
| joblib, tqdm, requests | | | | |

Un snapshot de la cadena de SPX son ~28.600 filas; en Parquet diario se conservan los tipos
y comprime en torno a un orden de magnitud frente a CSV. Todo el núcleo corre con
numpy + pandas + scipy, en línea con el minimalismo deliberado del proyecto hermano.

### 1.3 Credenciales

Sólo existe `SEC_EDGAR_USER_AGENT` (en el `.env` del proyecto hermano). **No hay ningún
proveedor de datos de opciones configurado.** Los secretos se leen sólo de variables de
entorno; `.env` en gitignore, `.env.example` documenta los nombres de clave.

### 1.4 Convenciones heredadas del proyecto hermano

Se replican a propósito, para que ambas estrategias se lean como un solo repositorio:

- Paquete con layout `src`, `pyproject.toml`, docstrings narrativos que explican **el porqué**.
- **Los tests son scripts autónomos** (`python tests/test_greeks.py`), encadenados por
  `run_tests.py`. Sin dependencia de pytest.
- Seguridad de consola en Windows: los tests reconfiguran stdout a UTF-8 (cp1252 revienta
  con la primera σ).
- **Pre-registro** (`docs/PREREGISTRO.md` en el hermano): criterios de aceptación fijados por
  escrito antes de ver resultados. Reproducido aquí en el §7.
- **Documentación en español.** Decidido el 31-08-2026.

---

## 2. ARQUITECTURA

```
GEX Strategy/
├── archive_chains.py       ENTRADA OPERATIVA — archivador diario de cadenas de CBOE
├── run_tests.py            ENTRADA — encadena las suites de `tests/`
├── gamma_quant/            el motor. Importable, puro, testeado
│   ├── config.py           carga del TOML, rutas, secretos
│   ├── logging_setup.py    logging con fichero para procesos desatendidos
│   ├── registry.py         registro de experimentos append-only
│   ├── data/
│   │   ├── ingestion/      base.py (ABC + esquema canónico), cboe.py, bloomberg_omon.py, synthetic.py
│   │   ├── cleaning/       VACÍO — hoy la normalización vive en `ingestion/base.py::ensure_canonical`
│   │   ├── validation/     quality.py — controles de calidad -> informe; nada se descarta en silencio
│   │   └── storage/        panel.py — almacén Parquet, particionado por (símbolo, año)
│   ├── options/
│   │   ├── pricing.py      Black-Scholes-Merton europeo; binomial americano para medir el sesgo
│   │   ├── greeks.py       delta/gamma/vega/theta, forma cerrada + manejo de límites
│   │   ├── gex.py          motor GEX: contrato/strike/vencimiento/total, convenciones enchufables
│   │   ├── gamma_flip.py   raíz de GEX(S*) = 0
│   │   └── gamma_walls.py  detección de concentración, por percentil y estadística
│   ├── features/           VACÍO (Fase 7) — opciones / precio / volatilidad / régimen
│   ├── strategies/         VACÍO (Fase 9) — mean_reversion, momentum, volatility
│   ├── backtest/           VACÍO (Fases 10-12) — engine, execution, costs, portfolio
│   ├── research/           VACÍO (Fases 7-13) — contrastes; NADIE del motor importa de aquí
│   ├── models/             VACÍO (Fase 14, condicionada a que el baseline tenga señal)
│   └── visualization/      VACÍO — gex_plots, performance, diagnostics
├── research/               ORQUESTACIÓN, no motor. Scripts que leen `gamma_quant/` y escriben en `reports/`
│   ├── probe_data_sources.py        qué vende de verdad cada fuente
│   └── calibrate_from_bloomberg.py  reproduce el spot y la q de §4.1.d
├── tests/                  suites autónomas, con verdad analítica. `python run_tests.py`
├── configs/                TOML; ningún número mágico en el código
├── reports/                generado por máquina, FUERA DE GIT (calidad, bitácora, registro)
├── data/
│   ├── archive/            IRREEMPLAZABLE, fuera de git, necesita copia de seguridad propia
│   ├── external/           ENTRADAS MANUALES, sí versionadas: los exports OMON del Terminal
│   └── {raw,processed}/    fuera de git
└── notebooks/
```

**Código de investigación y código de producción separados:** `gamma_quant/` es importable,
puro y testeado; `research/` orquesta y escribe en `reports/`. El motor nunca importa nada
de `research/`.

**Dos carpetas se llaman `research/` y no son lo mismo.** `gamma_quant/research/` será
código de contraste importable (hoy vacío); `research/` en la raíz son *scripts* que se
ejecutan. La regla de dependencia va en un solo sentido: los scripts importan el paquete,
nunca al revés.

**Los directorios marcados VACÍO tienen sólo un `__init__.py`.** Están creados para que la
estructura no se improvise a mitad de camino, pero hoy no contienen nada: no los cuentes
como hechos al leer el estado del proyecto.

### 2.1 Reglas de diseño

1. **Toda decisión de modelado es configuración, no un literal.** Convención de signo,
   multiplicador, tipo libre de riesgo, lag de OI, escalón de costes: todo en TOML y todo
   registrado en el experimento.
2. **Point-in-time por construcción.** El almacén expone `as_of(t)`; una función de features
   no puede ver `t+1` porque nunca se le entrega.
3. **Los supuestos son objetos.** La convención de signo del dealer es una clase
   intercambiable, de modo que el placebo de "signo invertido" (§26 del encargo) es un
   cambio de configuración, no una edición de código.

---

## 3. ESQUEMA CANÓNICO DE DATOS

Todo adaptador debe emitir exactamente esto. Los campos marcados **R** son obligatorios para
calcular GEX; sin ellos la fila se pone en cuarentena, no se descarta en silencio.

### 3.1 Cadena de opciones (una fila por contrato y timestamp)

| Campo | Tipo | R | Nota |
|---|---|---|---|
| `timestamp` | UTC con tz | R | instante de **observación**, no de negociación |
| `symbol` | str | R | SPY, SPX |
| `underlying_price` | float | R | spot en el mismo instante |
| `expiration` | date | R | |
| `strike` | float | R | |
| `option_type` | {C,P} | R | |
| `open_interest` | int | R | **el campo crítico**; reglas de lag en §5 (A5) |
| `implied_volatility` | float | R | necesaria si calculamos gamma nosotros |
| `gamma` | float | | gamma del proveedor, si viene — se contrasta contra la nuestra |
| `bid`, `ask`, `mid` | float | | costes y controles de calidad |
| `volume` | int | | proxy de flujo, distinto del OI |
| `delta`, `vega`, `theta` | float | | |
| `multiplier` | int | R | 100 en SPY y SPX |
| `risk_free_rate` | float | R | emparejado al plazo |
| `dividend_yield` | float | | SPY reparte; las opciones sobre el índice SPX no |
| `source`, `ingested_at` | str, ts | R | trazabilidad |

### 3.2 Barras del subyacente

`timestamp, symbol, open, high, low, close, volume, vwap?` — con tz y calendario de mercado.

### 3.3 SPY vs SPX — diferencias estructurales que prohíben mezclarlos

| | SPY | SPX |
|---|---|---|
| Estilo | **Americano** | Europeo |
| Liquidación | física (acciones) | efectivo |
| Dividendos | reparte; ejercicio anticipado posible | índice, la opción no cobra dividendo |
| Nocional | ~767 $ × 100 | ~7.686 $ × 100 (~10×) |
| Black-Scholes | **sesgado** (no recoge la prima de ejercicio anticipado) | apropiado |
| Liquidación AM/PM | PM | los mensuales liquidan en **AM (SET)** — trampa real el día de vencimiento |

Consecuencia: la gamma de SPY con fórmula europea es una **aproximación**; el error se
concentra en puts muy ITM, que es justo donde el OI puede ser grande. Queda como supuesto
A4 y se **mide**, no se supone despreciable.

---

## 4. FUENTES DE DATOS

### 4.1 Verificadas y funcionando hoy (gratis)

- **CBOE delayed quotes** `cdn.cboe.com/api/global/delayed_quotes/options/{SPY,_SPX}.json`
  Cadena completa con `open_interest`, `iv`, `gamma`, `delta`, `bid/ask`, `volume`, `theo`.
  Diferida y **sólo snapshot**. Es la fuente del archivador.
- **yfinance** — subyacente diario (profundo), intradía 5m (60d) / 1h (~2a), VIX. Snapshot de
  cadena como contraste cruzado del de CBOE.

### 4.1.b Alpha Vantage — PROBADO Y DESCARTADO EN GRATUITO (2026-09-01)

Se consiguió clave gratuita y se probó contra el API real. Resultado:

| Endpoint | Clave gratuita | Nota |
|---|---|---|
| `HISTORICAL_OPTIONS` | **NO** — *"This is a premium endpoint"* | Es exactamente lo que necesita el proyecto: cadena completa de una fecha pasada con OI, IV y griegas. Verificado con la clave `demo` sobre IBM: 998 contratos y el esquema canónico entero. Pero de pago |
| `REALTIME_OPTIONS` | **NO, y engaña** | Devuelve HTTP 200 con `data` no vacío… de **contratos falsos**: `XXYYZZ999999C00020000`, vencimiento `2099-99-99`, más un aviso de que el esquema es artificial |
| `TIME_SERIES_DAILY` | Sí | Subyacente. Redundante con yfinance |

**La trampa de `REALTIME_OPTIONS` es la lección de diseño del día.** Un adaptador
que valide con `if data:` ingiere contratos de mentira sin lanzar un solo error.
`research/probe_data_sources.py` incorpora ahora `_looks_like_placeholder()` y
aborta ante fechas imposibles o tickers de relleno. **Ninguna fuente se considera
válida por devolver 200 con una lista dentro.**

**Antes de pagar hay que verificar en qué plan entra `HISTORICAL_OPTIONS`.** El
mensaje de `REALTIME_OPTIONS` menciona el plan de 600 req/min (199,99 $/mes), así
que no está confirmado que el de 49,99 $ incluya opciones. Preguntar a soporte
antes de suscribir, no después.

### 4.1.c Bloomberg OMON — PROBADO (2026-09-01)

Se recibieron tres exports XLSX de la pantalla OMON del Terminal. Resultado:

**Lo bueno, y es más de lo esperado:**

| Hallazgo | Detalle |
|---|---|
| **El histórico llega a 95 días** | El export "as of 29may" se **autoverifica**: sus grupos dicen `18-Jun-26 (20d)`, luego la fecha es 2026-05-29. La ventana documentada de Bloomberg son 90 días; sirvió 95 |
| Trae **open interest y griegas** | `OInt`, `DL`, `GL`, `VL`, `TL` por contrato |
| Publica **r y dividendo por vencimiento** | La cabecera de grupo trae `R 4.11` e `IDiv .71`. Deja de hacer falta *suponer* A6 |
| Distingue SPX de SPXW | Recuperable de la raíz del ticker |

**Lo malo, y es decisivo:** OMON exporta **lo que se ve en pantalla**, unos 150
contratos de ~15 strikes. Una cadena completa de SPX son ~28.000. **Estos exports
no son un panel de investigación** y una fecha por export manual no escala a las
~250 sesiones que exige un backtest.

**Para qué SÍ sirven, y es valioso:** calibrar y contrastar el motor.

### 4.1.d CALIBRACIÓN CONTRA BLOOMBERG — el motor está validado

Contrastando nuestra gamma BSM contra la `GL` de Bloomberg sobre 146 contratos
reales de SPX:

1. **`GL` de Bloomberg es gamma por movimiento del 1%, no por dólar.**
   Ratio `GL/γ_BS` = 75,81 frente a `S×0,01` = 76,44. Usar `GL` cruda en el GEX
   lo multiplica por ~76 **sin producir ningún error visible**.
2. **Convertida, nuestra gamma coincide con la de Bloomberg: error mediano 1,25%**
   (p90 8,8%). El motor está calibrado contra la referencia institucional.
3. **Spot y dividendo recuperados de la propia cadena** por paridad put-call:
   `F = K + (C−P)e^{rT}`, y `ln F = ln S + (r−q)T` sobre seis vencimientos.
   Resultado: spot 7.643,58 y **q = 0,30%**.

El punto 3 corrigió un error propio: yo había puesto `dividend_yield = 0.013`
"porque es el dividendo típico del S&P". Estaba **cuatro veces alto**. La lección
no es el número — es que existía un método para **medirlo** con datos ya
disponibles en vez de suponerlo.

Todo ello queda fijado en `tests/test_bloomberg_omon.py` (22 comprobaciones), que
falla si alguien rompe las griegas o cambia r/q/tau sin darse cuenta.

### 4.2 Limitación intradía

Las barras de 5 minutos sólo llegan 60 días atrás. **La investigación de 0DTE (§21 del
encargo) es intradía por naturaleza** y es por tanto la parte con menos datos de todo el
proyecto. Se señala ahora y no al final.

### 4.3 Proveedores históricos candidatos — **sin verificar, en reserva**

No se ha confirmado precio, cobertura ni licencia actuales de ninguno, y no se afirman como
hecho. Quedan como pistas por evaluar si algún día se autoriza la vía (A):

| Proveedor | Relevancia reputada | A verificar |
|---|---|---|
| CBOE DataShop | cadenas EOD con OI, autoritativas | precio, fecha más antigua, momento de publicación del OI |
| ThetaData | histórico SPX/SPY accesible a retail | qué plan incluye OI + griegas |
| Polygon.io | agregados/snapshots de opciones, flat files | si el histórico de OI está incluido |
| ORATS | cadenas curadas + superficies de IV | licencia, profundidad |
| Databento | OPRA, pago por uso | coste de reconstruir varios años de cadena |
| OptionMetrics IvyDB | estándar académico | acceso sólo institucional |
| Índices GEX derivados (series DIX/GEX publicadas) | serie GEX-like **histórica** | licencia; y ojo: es una **caja negra** — contrasta la hipótesis económica, no nuestro motor |

La última fila es la más interesante estratégicamente: una serie GEX histórica de terceros
permitiría contrastar la **hipótesis económica** sobre historia real de inmediato, a cambio
de no saber cómo está construida. Complementa a las cadenas crudas; no las sustituye.

---

## 5. REGISTRO DE SUPUESTOS

Cada uno explícito, configurable y, cuando se puede, contrastado. **Ninguno se trata como verdad.**

| ID | Supuesto | Por defecto | Cómo se cuestiona |
|---|---|---|---|
| **A1** | Los dealers están netos **cortos de calls y cortos de puts** frente al cliente, lo que da el libro convencional `calls +gamma / puts −gamma` | `convention="conventional"` | Convenciones alternativas implementadas; el **placebo de inversión de signo** es un cambio de config |
| **A2** | El posicionamiento del dealer es **inferible del OI** | — | **No lo es.** El OI dice cuántos contratos vivos hay, jamás quién está largo. Es la debilidad más profunda de todo el GEX público y se declara como tal en el informe |
| **A3** | `GEX = gamma × OI × mult × S² × signo` (escalado por spot, $ por movimiento del 1%) | una entre varias | Se implementan varias definiciones y se **comparan por poder predictivo**, no por gusto |
| **A4** | La gamma europea de Black-Scholes sirve para SPY (americana) | activo | Se mide el error contra un binomial americano; no se supone pequeño |
| **A5** | El OI utilizable hoy es el **publicado ayer** | `oi_lag_days=1` | El OCC publica el OI antes de la apertura siguiente. Usar el OI del mismo día es look-ahead. El lag es configurable y el caso sin lag se corre **sólo** para cuantificar cuánto infla |
| **A6** | Tipo libre de riesgo plano por plazo | config | Se barre la sensibilidad; la gamma depende poco del tipo |
| **A7** | La IV del proveedor es fiable | verificar | Se contrasta re-resolviendo la IV desde el mid |
| **A8** | Al mover el spot para buscar el flip, cada strike conserva su IV | `sticky_strike` | `flip_sensitivity()` calcula el flip bajo las tres reglas |
| **A9** | `tau` se mide en **horas reales** hasta las 16:00 ET, no en días de calendario | horas | **Medido 2026-09-01: cambiar a días de calendario mueve el GEX total un 22,5% (SPY) y un 17,2% (SPX)** |

### A9 — la convención de `tau` es un supuesto de primer orden

Descubierto al contrastar nuestra gamma contra la de CBOE sobre datos reales. La
discrepancia mediana ponderada por GEX es del 3,3% (SPY) y 2,5% (SPX) — aceptable —
**pero en 0DTE sube al 49% (SPY) y 15% (SPX)**.

La causa no es un error: es que `gamma ~ 1/√T` diverge, así que en el último día
de vida el valor de `tau` domina el resultado. Nosotros contamos horas reales
hasta el cierre, que es lo económicamente correcto; CBOE usa una convención que
no publica.

Consecuencia para la investigación: **el 0DTE aporta el 6-9% del |GEX| bruto pero
hasta el 50% del GEX neto de SPX**, y es justo donde el número es menos robusto.
Cualquier resultado sobre 0DTE debe reportarse con la sensibilidad a A9 al lado.

### El GEX neto es un residuo, y eso lo hace frágil

Medido sobre SPX real el 2026-09-01:

```
calls   +261.893.991.358
puts    -305.154.291.323
neto     -43.260.299.965      <- el 14% del bruto
```

Las dos patas casi se cancelan, así que **un error del 3% en cualquiera de ellas
mueve el neto un 20%**. Esto no es un problema de implementación: es una
propiedad de la magnitud. Implica que:

1. La precisión de la gamma importa mucho más de lo que sugiere su efecto sobre
   una sola opción.
2. Comparar niveles de GEX entre días exige que la convención no cambie **nunca**.
3. Cualquier estrategia basada en el signo del GEX neto está operando sobre la
   diferencia de dos números grandes, con todo lo que eso implica para la
   relación señal/ruido.

**A2 merece énfasis.** Toda construcción retail de GEX asume una convención de signo porque
el inventario del dealer es privado. Si la convención está mal, el signo de la señal está
mal. Por eso el placebo de inversión es un test de primera clase y no una nota al pie.

---

## 6. METODOLOGÍA DE INVESTIGACIÓN

**Orden estricto. Ninguna etapa empieza sin que pase la anterior.**

1. **Corrección del motor** (sintético, verdad analítica). ¿El código calcula lo que dice?
2. **Descriptivo** — ¿qué aspecto tiene el GEX real? Distribuciones, persistencia, estructura temporal.
3. **Predictivo, antes de cualquier estrategia.** `GEX_t -> r_{t+1,5,30}` y `GEX_t -> RV_{t+1,5}`.
   IC de Spearman con t de Newey-West (las ventanas solapadas inflan la t ordinaria ~√h);
   tablas por quintil y condicionales; intervalos por bootstrap de bloques.
4. **Regímenes** — ¿difieren los cinco en media, volatilidad, autocorrelación y riesgo de cola?
   Contrastes con corrección por comparaciones múltiples.
5. **Valor incremental** — Modelo A (precio) / B (+vol) / C (+posicionamiento) / D (+GEX).
   *Si D no bate a C, el proyecto informa "sin valor incremental" y para.*
6. **Estrategias** sólo si 3–5 muestran señal.
7. **Costes** — optimista / base / conservador. Ganar sólo antes de costes es fracasar.
8. **Walk-forward**, con parámetros elegidos únicamente dentro de train/validación.
9. **Placebo** — GEX barajado, GEX retardado, gamma aleatoria, signo invertido, entradas
   aleatorias, baseline sólo-precio. Si la estrategia sobrevive a aleatorizar el GEX, el GEX
   no es lo que la mueve.
10. **Informe**, incluido el caso negativo.

**El contador de intentos arranca en cero y se incrementa en
`reports/experiment_registry.jsonl` por cada configuración evaluada.** El Sharpe deflactado
y el PBO usan ese número. Es la defensa contra el hecho de que un sistema con tantos mandos
puede producir cualquier resultado.

---

## 7. CRITERIOS DE ACEPTACIÓN PRE-REGISTRADOS

**Escritos ahora, antes de que exista ningún resultado. No se renegocian después.**

### Puerta 1 — ¿El GEX predice algo? (obligatoria para justificar una estrategia)

| # | Criterio | Umbral | Por qué ése |
|---|---|---|---|
| 1 | IC de Spearman, `GEX_t` vs retorno futuro | \|IC\| > 0,03 | por debajo, los costes a nivel índice se lo comen |
| 2 | t de Newey-West de la serie de ICs | \|t\| > 2,5 | 2,0 es laxo dado el número de horizontes probados |
| 3 | Monotonía entre quintiles de GEX | corr. de rangos > 0,60 | un Q5−Q1 grande con el medio desordenado es una cola, y las colas no se repiten |
| 4 | Predicción de RV: R² incremental sobre HAR-RV | > 0,01, t > 2,5 | la volatilidad es la afirmación *fácil*; fallar aquí es evidencia fuerte en contra |

### Puerta 2 — ¿Es una estrategia? (sólo si pasa la Puerta 1)

| # | Criterio | Umbral |
|---|---|---|
| 5 | Sharpe OOS con costes **base** | > 0,50 |
| 6 | Sharpe OOS con costes **conservadores** | > 0,00 |
| 7 | Sharpe deflactado (por nº de intentos, asimetría y curtosis) | > 0,95 |
| 8 | PBO | < 0,50 |
| 9 | Incremental sobre el mejor baseline sin GEX | positivo, t > 2,0 |
| 10 | Suite de placebos | **todos** deben rendir peor que la señal real |
| 11 | Meseta de parámetros | Sharpe estable en un entorno contiguo, no un pico |

**Fallar la Puerta 1 cierra el proyecto con un informe negativo.** Ese desenlace es
explícitamente aceptable y se reportará con la misma claridad que uno positivo.

---

## 8. METODOLOGÍA DE PRUEBAS

- **Primero la verdad analítica.** La gamma se verifica contra valores en forma cerrada,
  contra la paridad put-call (bajo BSM la gamma de call y put es idéntica a igual
  strike/vencimiento), contra la segunda derivada por diferencias finitas del precio BSM, y
  contra los límites conocidos `T→0`, `σ→0`, muy ITM y muy OTM.
- **Cadenas sintéticas con respuesta conocida.** Una cadena de un solo strike y OI conocido
  tiene GEX total calculable a mano; el motor debe reproducirlo exactamente.
- **Invariantes**, no inspección visual: el GEX total debe igualar la suma por strikes y la
  suma por vencimientos; el gamma flip debe caer donde la curva de GEX neto cambia de signo.
- **Pruebas de propiedad** sobre la numérica: monotonía, simetría, ausencia de NaN propagados.
- Seguridad de consola en Windows (reconfiguración a UTF-8) en todas las suites.

---

## 9. METODOLOGÍA DE VALIDACIÓN

- **Walk-forward**, con fechas fijadas cuando existan datos; nunca un único train/test.
- Selección de parámetros **sólo** dentro de train+validación. El periodo de test final se
  toca una vez.
- **Bootstrap por bloques** (no i.i.d.) para intervalos: los retornos agrupan volatilidad.
- **Corrección por comparaciones múltiples** entre horizontes, umbrales y definiciones.
- **Cortes por régimen**: alcista/bajista, alta/baja volatilidad, vencimiento/no, 0DTE/no.
- **Entre activos**: SPY vs SPX, y luego QQQ/IWM si los datos lo permiten.

---

## 10. HITOS

| Fase | Entregable | Estado |
|---|---|---|
| 1 | Auditoría de entorno y datos, este plan | **Hecho** |
| 2 | Esqueleto del paquete, config, logging, registro de experimentos | Siguiente |
| 3 | `pricing.py`, `greeks.py` + suite analítica | Siguiente |
| 4 | `gex.py`, `gamma_flip.py`, `gamma_walls.py` + tests de invariantes | Siguiente |
| 5 | Generador de cadenas sintéticas, extremo a extremo con respuesta conocida | Siguiente |
| 6 | ABC de proveedor, adaptador CBOE, **archivador corriendo a diario** | Siguiente |
| 7–9 | Exploratorio, contrastes predictivos, estrategias baseline | **Bloqueado por datos** |
| 10–12 | Motor de backtest, costes, walk-forward | Bloqueado |
| 13 | Robustez y placebo | Bloqueado |
| 14 | ML (condicionado a señal en el baseline) | Bloqueado |
| 15 | Informe final de investigación | Bloqueado |

Las fases 2–6 merecen la pena pase lo que pase con los datos: son agnósticas al proveedor y
son lo que haría inmediatamente utilizable una compra posterior de histórico.

---

## 11. RIESGOS

| Riesgo | Severidad | Mitigación |
|---|---|---|
| **Sin histórico de opciones** | **Crítico — bloqueante hoy** | Vía (B) elegida; el archivador arranca ya. Veredicto no antes de finales de 2027 |
| Posicionamiento del dealer no observable (A2) | Alta, irreducible | Declararlo; contrastar inversión de signo; nunca afirmar que medimos inventario del dealer |
| Look-ahead por OI | Alta | `oi_lag_days=1` por defecto; el caso sin lag sólo para cuantificar el sesgo |
| Sobreajuste con tantos mandos | Alta | Pre-registro §7, registro de intentos, Sharpe deflactado, PBO, exigencia de meseta |
| Masificación — el GEX se publica por todas partes | Media | Cualquier ventaja puede estar ya arbitrada; contrastar subperiodos recientes por separado |
| Dependencia de régimen (el 0DTE explota tras 2022) | Media | Nunca mezclar pre/post-2022 sin test de ruptura |
| Los costes dominan a nivel índice | Media | Tres escalones; el conservador debe superar cero |
| Sesgo de gamma americana en SPY (A4) | Media | Medir contra binomial |
| Confundir resultados sintéticos con reales | Media | Sello `SINTÉTICO — NO ES EVIDENCIA` en todo artefacto |
| Supervivencia en el universo de contratos | Baja en índice | El archivo debe conservar los contratos vencidos |
| **El archivo se interrumpe** | **Media-alta, nueva con la vía (B)** | Un hueco de semanas en una muestra de un año es grave. El archivador registra los fallos y el informe de calidad lista los días ausentes |

---

## 12. LIMITACIONES CONOCIDAS (permanentes, se repetirán en el informe final)

1. **El posicionamiento del dealer no es observable.** El GEX es un modelo de él, construido
   sobre una convención de signo que no se puede verificar con datos públicos. (A1, A2)
2. **El OI es de cierre.** Cualquier GEX intradía construido con OI diario es rancio por
   construcción; el GEX intradía real exigiría datos de posicionamiento intradía que no
   existen públicamente.
3. **Black-Scholes sobre opciones americanas de SPY** arrastra sesgo de ejercicio anticipado.
4. **Los datos gratuitos de cadena son diferidos y no archivados** — el archivo que
   empezamos hoy tiene fecha de inicio conocida y nada anterior.
5. **El intradía llega 60 días a 5m**, lo que limita materialmente el trabajo en 0DTE.
6. **El mecanismo es público.** Las señales muy publicadas se operan mucho.
7. **Las opciones sobre índice son un mercado.** Las conclusiones no se extrapolan a acciones
   individuales sin contrastarlo aparte.
8. **La muestra que produzca la vía (B) empieza en 2026-08-31.** No contiene 2018, ni el COVID,
   ni 2022. Cualquier conclusión estará condicionada al régimen vivido desde esa fecha, y así
   se dirá.

---

## 13. PRÓXIMAS ACCIONES

1. ~~Instalar `pyarrow`~~ — hecho (25.0.1).
2. Construir las fases 2–6.
3. **Arrancar el archivador de CBOE hoy** — sin coste, y la muestra sólo crece.
4. Programar la ejecución diaria y vigilar los huecos.
