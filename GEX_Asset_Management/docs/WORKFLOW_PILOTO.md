# Workflow — Piloto de validación: Net Gamma Exposure (Soebhag 2023)

**Versión corregida del workflow. Alcance: 1 año de datos (2025-09 → 2026-09), universo reducido, Databento.**

> **Encuadre honesto y no negociable:** esto **no es un backtest concluyente de la estrategia**. Es un **piloto de validación de mecanismo e infraestructura**. Con 12 meses y ~20 acciones no existe forma matemática de validar o refutar un alfa de 0.93%/mes. Lo que sí se puede probar, y con potencia real, es (a) que el pipeline calcula Γ correctamente y (b) que el **mecanismo** del paper existe en estos datos. Si eso pasa, se escala. Presentarlo como "backtest de la estrategia" ante la mesa es donde te van a desarmar.

---

## 0. Los tres errores del plan original que hay que corregir

### Error 1 — 20 acciones no se pueden dividir en 10 deciles
2 acciones por decil. El portafolio largo sería 2 nombres y el corto 2 nombres. Eso no es una estrategia cross-seccional: es una apuesta binaria entre 4 acciones, con volatilidad de portafolio 10–20× mayor que su alfa esperado. El resultado sería ruido puro, en cualquier dirección.

**Corrección:** con N ≈ 30 nombres → **quintiles (6 por grupo)** o **terciles (10 por grupo)**. Largo el quintil inferior de Γ, corto el superior. Y reportar el *spread continuo* (regresión) además del spread de grupos, porque la regresión usa los 30 nombres y no solo 12.

### Error 2 — SPY y QQQ no pueden estar en el ranking
Tres razones, todas eliminatorias:
1. El paper excluye explícitamente cualquier cosa que no sea acción común (CRSP `shrcd` 10/11). Los ETFs quedan fuera por definición.
2. Su Γ no es comparable: en SPY/QQQ el open interest está dominado por hedgers institucionales y 0DTE, con un perfil de posicionamiento de market maker estructuralmente distinto al de una acción individual. Meterlos en el mismo ranking contamina los puntos de corte.
3. **Redundancia**: SPY y QQQ *contienen* a los otros 18 nombres. Si SPY cae en el quintil largo y NVDA en el corto, estás largo NVDA y corto NVDA a la vez.

**Corrección:** SPY y QQQ salen del cross-section y pasan a un **track paralelo separado** (§6). Ahí sí son valiosos — son exactamente tu pregunta abierta sobre GEX de índice.

### Error 3 — Value-weighting entre mega-caps es una apuesta a 3 acciones
NVDA + AAPL + MSFT concentran la mayor parte de la capitalización de tu lista. Ponderar por cap. de mercado convierte el portafolio en un trade de esos tres nombres, independientemente de lo que diga Γ.

**Corrección:** **equal-weight como ponderación primaria** en el piloto. Value-weight se reserva para cuando el universo tenga cientos de nombres.

---

## 1. Universo corregido

### 1.A Cross-section (el ranking) — 30 acciones comunes

**Bloque núcleo (18 — tus nombres, sin los ETFs):**
AAPL · MSFT · NVDA · AMD · AMZN · GOOGL · META · TSLA · NFLX · JPM · BAC · XOM · CVX · DIS · BA · WMT · KO · PG

**Bloque de dispersión (12 — APROBADO 2026-09-01):**
PLTR · COIN · MSTR · GME · RIVN · SOFI · UBER · SHOP · MU · CRM · GS · CAT

**Por qué añadir el segundo bloque.** El edge del paper vive en la **dispersión cross-seccional de Γ**. Tus 18 nombres son todos mega-caps ultra-líquidas con estructuras de opciones parecidas → la dispersión de Γ entre ellas es comprimida y el ranking se vuelve casi aleatorio. Los 12 añadidos tienen perfiles de flujo de opciones deliberadamente distintos (retail/meme intenso, OI concentrado en strikes redondos, ADV$ menor relativo al OI) → generan la varianza en Γ que la señal necesita para ordenar algo.

Costo de datos de pasar de 18 a 30 nombres: marginal (§3). Ganancia estadística: grupos de 6 en vez de 2, y varianza real en el regresor. **Es la mejora de mejor relación costo-beneficio de todo el plan.**

### 1.B Filtros (aplicados mensualmente, point-in-time)
| Filtro | Valor | Nota |
|---|---|---|
| Tipo de instrumento | Acción común únicamente | Sin ETFs, ADRs, fondos cerrados |
| Precio | > $5 | Ninguno de los 30 lo incumple, pero el código lo verifica igual |
| Cadena de opciones | ≥ 200 contratos vivos con OI > 0 | Gate de calidad de datos |
| Contratos válidos ese día | ≥ 60% de la cadena pasa filtros de cotización | Si no, la observación acción-día se marca `NaN`, no se imputa |

---

## 2. Fuentes de datos — Databento (reemplaza la columna de Bloomberg)

| Dato | Databento | Detalle |
|---|---|---|
| Definiciones de contrato | `OPRA.PILLAR`, schema `definition` | strike, vencimiento, call/put, **multiplicador real** (no 100 fijo) |
| Open interest | `OPRA.PILLAR`, schema `statistics`, `stat_type = 9` | OI de inicio de sesión → desfase estructural de 1 día, declararlo |
| Precio de la opción | `OPRA.PILLAR`, quotes (`bbo-1m` / `mbp-1` / `tbbo`) | **Mid del NBBO en la ventana 15:55–16:00 ET.** Nunca el último trade |
| **Gamma** | **No existe en Databento** | Se calcula: IV invertida del mid → árbol CRR americano → γ por diferencias finitas |
| Precio y volumen del subyacente | Equities (`EQUS.SUMMARY`) | Cierre y volumen consolidados; ADV$ de 21 días hábiles |
| Dividendos y splits | API de corporate actions | Dividendos discretos para el árbol; factores de ajuste para retornos |
| Tasa libre de riesgo | FRED / curva cero del Tesoro | Interpolada al vencimiento de cada contrato |
| Factores FF / q | Ken French Data Library, global-q.org | Ver §5 sobre por qué en el piloto casi no sirven |

**Nota sobre Bloomberg (por qué Databento gana aquí):** recuperar la **cadena histórica completa con greeks y OI para contratos ya expirados** en Bloomberg es el punto débil de esa terminal — OMON es point-in-time, y reconstruir la membresía histórica de la cadena contrato por contrato choca con los límites diarios de datos. Verifícalo con tu acceso antes de descartarlo, pero para un pipeline sistemático Databento es la ruta correcta.

---

## 3. Volumen y costo de datos (la buena noticia de tu simplificación)

Estimación gruesa: 30 subyacentes × ~1.500–4.000 contratos vivos × ~250 días hábiles ≈ **20–30 millones de filas contrato-día**. Eso es un dataset de decenas de GB en Parquet — perfectamente manejable en un portátil, y un pedido de datos acotado en vez de un pedido de OPRA completo.

**Además tu recorte a 1 año elimina el mayor riesgo del plan grande:** el conflicto sobre cuándo empieza la historia de OPRA (blog dice mar-2023, página comercial dice 2013) deja de importar — 12 meses están disponibles con certeza bajo cualquiera de las dos versiones.

Sigue siendo obligatorio correr `metadata.get_cost()` antes de descargar. Los quotes de OPRA son el componente caro; los `statistics` y `definition` son baratos.

---

## 4. Cálculo de Γ — tu fórmula está correcta, con una trampa

Tus pasos 1–4 reproducen la Ecuación 1 correctamente. La trampa está en el paso 4:

```
Paso 1  Γ_call = γ × OI × (+multiplicador) × S
Paso 2  Γ_put  = γ × OI × (−multiplicador) × S
Paso 3  Γ_bruto = Σ todos los calls + Σ todos los puts
Paso 4  Γ_final = Γ_bruto × S / (100 × ADV$_21d)      ← aquí
Paso 5  Winsorizar 1% / 99% cada mes (o cada semana, si la frecuencia es semanal)
```

El `S` aparece **dos veces** (una en el paso 1–2, otra en el paso 4). No es un error del paper: el primero convierte acciones a dólares, el segundo convierte "movimiento de $1" en "movimiento de 1%". Si lo implementas con un solo `S`, Γ queda contaminada por el nivel de precio y el ranking se sesga hacia las acciones caras. Con multiplicador = 100 la fórmula colapsa a:

```
Γ = S² × (Σ_calls γ·OI − Σ_puts γ·OI) / ADV$_21d
```

**Añadir al paso 5:** además de winsorizar, calcular el **z-score cross-seccional de Γ** cada fecha. El z-score es lo que entra en las regresiones; los grupos se forman sobre el Γ winsorizado.

### Filtros de calidad de cotización (antes de invertir IV) — faltaban en tu plan
Descartar contratos con: bid = 0 · mercado cruzado o bloqueado · spread relativo > 50% · precio bajo valor intrínseco · IV fuera de [1%, 500%] · OI = 0. **Registrar la tasa de descarte por día** — es el mejor indicador temprano de que el pipeline se rompió.

---

## 5. Frecuencia y tests — reordenados por potencia estadística disponible

Este es el cambio conceptual más importante. Con 1 año, la frecuencia mensual tiene **12 observaciones**. Es inútil. La jerarquía correcta:

| # | Test | Observaciones | ¿Concluyente? |
|---|---|---|---|
| **1** | **Panel: RV_{t+1} ~ Γ_t** con efectos fijos por acción, errores clusterizados | 30 × 250 ≈ **7.500** | **Sí.** Este es el test principal del piloto |
| **2** | Fama–MacBeth diaria: r_{i,t+1} ~ Γ_i,t | ~250 cross-sections | Marginal, pero real |
| **3** | Spread de quintiles semanal | ~52 | Indicativo |
| **4** | Spread de quintiles mensual (la "estrategia") | **12** | **No. Solo descriptivo** |

### Por qué el test #1 es el correcto para el piloto
El mecanismo económico del paper es: **Γ negativo → market makers amplifican movimientos → mayor volatilidad realizada futura → los inversionistas exigen prima de riesgo.** El paper verifica cada eslabón. El eslabón de volatilidad (§5) es un panel con miles de observaciones y un efecto grande — es testeable con 30 nombres y 1 año. El eslabón de retorno es una prima de riesgo pequeña que necesita décadas.

**Si solo puedes probar una cosa con este piloto, prueba el mecanismo, no el alfa.** Y es un resultado presentable: "confirmamos que el canal de hedging de gamma opera en estos 30 nombres en 2025–26, con el signo y la magnitud del paper" es una conclusión defendible. "El long-short dio +X% en 12 meses" no lo es.

### Lo que hay que **eliminar** del plan original
- **Los 5 alfas de modelos de factores.** Regresión de 12 observaciones contra 4–6 factores → ~6 grados de libertad. Los `t` no significan nada. Sustituir por: retorno del spread ajustado por beta de mercado, y exposición sectorial reportada explícitamente.
- **Los sorts bivariados contra ~20 características.** Imposible: al condicionar por un control, cada celda tendría 1–3 acciones. Sustituir por: **regresión panel con controles** (tamaño, momentum, reversión de corto plazo, volatilidad, iliquidez, IV, skew de IV, O/S, net Δ) todos a la vez, que sí cabe con 7.500 observaciones.
- **Matriz de transición de 12 meses.** No hay 12 meses de historia adicional para medirla. Mantener solo la de 1 mes, para estimar turnover.

### Lo que hay que **añadir**
- **Neutralización de beta y sector.** Con 6 nombres por pata en un universo cargado de tecnología, el long-short es en gran parte una apuesta tech-vs-tech disfrazada. Reportar: beta del spread, exposición sectorial neta, y una versión con Γ residualizado contra sector.
- **Descomposición ATM/OTM/ITM y fast/slow** (§4.4 del paper). Es casi gratis una vez que tienes los greeks por contrato, y es una **prueba de falsificación fuerte**: el paper predice que la señal viene de ATM/OTM y de vencimientos > 1 mes. Si en nuestros datos el poder predictivo apareciera en ITM y en `fast`, eso es evidencia de que tenemos un bug, no un descubrimiento.
- **Costos de transacción** desde el primer día. Medio spread real del subyacente (medido, no supuesto) + comisión + costo de préstamo de la pata corta. Todo resultado se reporta bruto **y** neto.

---

## 6. Track paralelo: GEX de índice (SPY / QQQ) — APROBADO 2026-09-01

Aquí es donde van SPY y QQQ, y responde tu §9. Mismo pipeline, pregunta distinta:

1. Calcular Γ agregado de SPY y QQQ (misma Ecuación 1, normalizado por ADV$ del ETF).
2. Test A: ¿Γ del índice predice la **volatilidad realizada** del índice al día/semana siguiente? (Esta es la predicción robusta y la que la literatura respalda.)
3. Test B: ¿Γ del índice predice la **dirección**? (Esta es tu pregunta abierta. Advertencia: la literatura de gamma de índice apoya efectos de *volatilidad* y de *reversión intradía*, no direccionalidad de baja frecuencia. Espera un resultado nulo y diséñalo para poder reportar un nulo limpio.)
4. Añadir el desglose por vencimiento, aislando 0DTE — es el cambio de régimen dominante en 2023–2026 y no existía en la muestra del paper.

**Mantener los dos tracks separados en el código y en el reporte.** Son dos estrategias distintas con dos mecanismos distintos.

---

## 7. Ciclo del backtest (corregido)

```
Para cada día hábil t en [2025-09-01, 2026-09-01]:
    1. Cargar definitions → cadena viva de los 30 nombres
    2. Cargar statistics → OI por contrato (con su timestamp real)
    3. Cargar quotes 15:55–16:00 ET → mid del NBBO
    4. Filtros de calidad de cotización  → registrar tasa de descarte
    5. Invertir IV (CRR americano, dividendos discretos, r interpolada)
    6. Calcular γ por diferencias finitas sobre el mismo árbol
    7. Agregar Ecuación 1 → Γ_bruto
    8. Normalizar por ADV$_21d → Γ_final ; winsorizar ; z-score
    9. Guardar Γ, sus componentes (ATM/OTM/ITM, fast/slow) y los controles

Formación de portafolio (3 frecuencias en paralelo):
    Diaria   → Fama-MacBeth sobre r_{t+1}
    Semanal  → quintiles, rebalanceo viernes al cierre
    Mensual  → quintiles, rebalanceo último día hábil  [solo descriptivo]

Ponderación: equal-weight (primaria) · inverso de volatilidad (robustez)
Instrumento: spot del subyacente. Nunca opciones.
Sin gatillos intramensuales.
```

---

## 8. Puertas de validación — esto es lo que define "precisión"

Ninguna fase avanza sin pasar su puerta.

**P1 — Motor de valuación**
- CRR converge a Black-Scholes sin dividendos: error < 1e-4 en precio, < 1e-5 en gamma
- Round-trip precio → IV → precio: error < 1e-6
- Paridad put-call en IV de pares ATM: desviación mediana < 1 punto de vol
- Gamma ATM contra la aproximación analítica `1/(S·σ·√(2πT))`: mismo orden de magnitud

**P2 — Γ tiene magnitudes correctas**
Contra la Tabla 1 del paper, **con una corrección de expectativa importante**: el paper reporta correlación de +0.15 entre Γ y tamaño, así que un universo de 30 mega-caps debe dar **Γ mediana por encima de 0.41 y menos del 21.8% de observaciones negativas**. Si nuestro Γ saliera con mediana 0.4 y 22% de negativos, eso sería sospechoso, no tranquilizador. La puerta es: Γ en el rango [0.1, 5] para la gran mayoría de observaciones, mediana entre 0.4 y 3, y correlaciones con IV (−), Call OI (+) y tamaño (+) con **el signo correcto**.

**P3 — Sin look-ahead**
Test automatizado que verifica, para cada variable, que su timestamp de disponibilidad ≤ timestamp de decisión. Falla el build si no. Atención especial al desfase real del OI, que es el punto más fácil de arruinar.

**P4 — Falsificación**
La señal debe venir de ATM/OTM y de `slow`. Si viene de ITM o de `fast`, es un bug.

---

## 9. Qué vas a poder afirmar al final del piloto, y qué no

**Podrás afirmar:**
- Que el pipeline calcula Γ con magnitudes consistentes con la literatura
- Si el canal de hedging (Γ → volatilidad futura) opera en estos nombres en 2025–26, con signo, magnitud y significancia
- El signo y la magnitud aproximada del spread de retornos, con intervalos de confianza honestos (y serán anchos)
- Si el GEX de índice predice volatilidad del índice
- El turnover y los costos reales de implementación

**No podrás afirmar:**
- Que la estrategia genera alfa. 12 meses, 30 nombres, 6 por pata: el intervalo de confianza del spread mensual será varias veces su punto central.
- Que es distinta de los predictores conocidos (no hay Compustat, ni grados de libertad)
- Nada sobre el régimen de 1996–2021. Este es un test out-of-sample en un régimen dominado por 0DTE y flujo retail que no existía en la muestra del paper.

**Criterio de promoción al build completo:** si P1–P4 pasan **y** el test de mecanismo sale significativo con el signo correcto → se escala a 500–1.000 nombres y a toda la historia disponible de OPRA. Si el mecanismo no aparece, el hallazgo del piloto es que el canal no opera en mega-caps líquidas en este régimen — que también es un resultado, y ahorra el gasto del build grande.


---

## 10. Resultados de la puerta P1 — motor de valuación (ejecutado 2026-09-01)

`python3 tests/test_pricing_gate_p1.py` → **25/25 chequeos superados.** Código en
`gex/pricing/{bsm,crr}.py`, suite en `tests/test_pricing_gate_p1.py`.

### Lo que quedó validado
| Chequeo | Resultado |
|---|---|
| CRR converge a Black-Scholes (europeo, sin dividendos) | error agregado de Σγ = **1.2e-3** |
| Teorema: call americana sin dividendos == europea | error **exactamente 0** |
| Prima de ejercicio anticipado del put ≥ 0 siempre | 238 casos con prima > 0, ninguno negativo |
| Round-trip precio → IV → precio | IV recuperada con error **1.3e-10** en contratos bien condicionados |
| Paridad put-call | 3.0e-11 |
| Auto-convergencia de gamma N=400 vs N=1600 | mediana 6.7e-4, p95 1.1e-2 |
| Gamma ATM vs aproximación analítica | ratio en [0.90, 1.00] |
| Rendimiento | 60.700 greeks/s · 10.700 inversiones de IV/s → **el piloto completo se procesa en ~30 min** |

### Cuatro hallazgos que cambian el diseño

**H1 — Γ es una diferencia de dos números grandes, y eso domina todo lo demás.**
Sobre cadenas sintéticas realistas, la razón `|Γ neta| / Γ bruta` va de **2% a 26%**. Es decir: la señal es un residuo pequeño entre el gamma de las calls y el de los puts, que casi se cancelan.

Consecuencias directas:
- El error numérico **debe normalizarse por gamma bruta, no por la neta**. Normalizado por la bruta el error del árbol a N=400 es 2.5e-4; normalizado por la neta llega al 8.6% en el peor caso. La segunda cifra no mide el motor, mide la cancelación.
- **Cualquier error en los datos de entrada se amplifica en Γ por el inverso de esa razón.** Medido: un ruido de ±0.5 puntos de vol en la IV mueve Γ un **1.2% en promedio y 2.8% en el peor caso**; un ruido de ±1% en el open interest la mueve **1.7%**.
- **Por lo tanto la calidad del dato importa mucho más que el método numérico.** Subir el árbol de 400 a 1600 pasos mejora 4× algo que ya vale 2.5e-4; usar el último trade en vez del mid del NBBO puede meter varios puntos de vol de error. El presupuesto de esfuerzo va a la calidad de la cotización, no a los pasos del árbol.
- Y lo más relevante para la estrategia: **la región de cancelación casi total es exactamente el decil L del paper** (Γ cerca de cero o negativa). Es decir, la pata donde vive el alfa es también la pata donde la señal es numéricamente más frágil. Esto **hay que testearlo explícitamente**: perturbar la IV y el OI con ruido realista y medir cuánto se reordenan los quintiles. Si el ranking se reshufflea, la estrategia no es implementable por más que el paper la respalde. **Nuevo requisito, añadido a la puerta P2.**

**H2 — Filtrar por vega baja habría borrado los contratos más importantes.**
Vega ∝ √T y gamma ∝ 1/√T: las opciones **muy cortas ATM tienen vega baja y gamma alta**. Mi primer filtro de "IV no identificable" usaba vega y habría descartado justo los contratos que más aportan a Γ (se midió gamma·S hasta 1.67 entre los descartados). La clasificación correcta separa dos causas distintas de vega baja:
- **(a) frontera de ejercicio inmediato** (put americano ITM, precio = intrínseco): IV genuinamente no identificable → se descarta. Medido: descartarlos cuesta solo **7.1e-4 de la gamma bruta**. Inocuo.
- **(b) vencimiento muy corto**: IV mal condicionada pero gamma grande → **se conserva**, y su riesgo se mide, no se esconde.

Esto importa especialmente en 2023–2026, un régimen dominado por vencimientos cortos.

**H3 — El árbol CRR sí es necesario; queda cuantificado cuánto.**
Usar Black-Scholes europeo en vez de CRR americano sobre puts con dividendos introduce un error de **4.9% (ITM), 9.7% (ATM), 9.6% (OTM)** en Σγ. Con la razón de cancelación de H1, eso se amplifica a decenas de por ciento en Γ. No es un detalle.
En cambio, para **calls sin dividendos antes del vencimiento el atajo analítico es EXACTO por teorema** (una call americana así nunca se ejerce anticipadamente): ~20% del lote se resuelve sin árbol, sin perder nada de precisión.

**H4 — El bump-and-reprice no sirve como referencia de gamma sobre un árbol.**
Los nodos se re-cuantizan al mover S y la segunda diferencia amplifica ese ruido por 1/h². Mi primer test lo usaba como referencia y daba 48% de error mediano — el test estaba mal, no el motor. Las referencias válidas son: gamma analítica de BSM en el caso europeo, y auto-convergencia en N para el americano. Queda documentado en el código para que nadie lo reintroduzca.

### Decisión de parámetros de producción
- `N_STEPS = 400` (error de Γ a nivel de cadena: 2.5e-4 normalizado por gamma bruta)
- `SIGMA_LO = 0.01` — por debajo de ~r·√dt la probabilidad riesgo-neutral del árbol sale de [0,1] y el modelo deja de estar definido. Coincide con el filtro de calidad de IV ∈ [1%, 500%]
- IV por semilla analítica BSM + refinamiento por secante sobre el árbol (2–4 árboles en vez de ~60 de una bisección pura: **4× más rápido**)
- Atajo exacto activo para calls sin dividendos
- Descarte de contratos en frontera de ejercicio inmediato; los cortos mal condicionados se conservan y se marcan

### Siguiente paso
**Fase 0 (GO/NO-GO): necesito la API key de Databento** para correr `metadata.get_dataset_range()` y `metadata.get_cost()` sobre los 32 subyacentes (30 + SPY/QQQ) para el rango 2025-09 → 2026-09.

---

## 11. Resultados de la Fase 0.A — verificación de datos (2026-09-01) · **GO**

### El conflicto de la historia de OPRA queda resuelto
`OPRA.PILLAR: 2013-04-01 → 2026-09-01` = **161 meses**. La página comercial de Databento tenía razón; el blog de lanzamiento estaba desactualizado. **El límite de 12 meses del piloto ya no es una restricción de los datos: es una elección nuestra.**

### Pero el cuello de botella se movió a los datos de equities
La Ecuación 1 necesita precio y volumen del subyacente para el denominador (ADV$ de 21 días). Rangos disponibles:

| Dataset | Desde | Uso |
|---|---|---|
| OPRA.PILLAR | 2013-04-01 | opciones |
| XNAS.ITCH | 2018-05-01 | subyacentes listados en Nasdaq |
| XNYS.PILLAR | 2018-05-01 | subyacentes listados en NYSE |
| EQUS.SUMMARY | 2024-07-01 | consolidado (el más cómodo, pero el más corto) |
| DBEQ.BASIC / EQUS.MINI | 2023-03-28 | — |

**El build completo máximo realista es 2018-05 → hoy ≈ 100 meses**, combinando XNAS.ITCH y XNYS.PILLAR. Un tercio de la muestra del paper (311 meses), pero suficiente para que un efecto real con t = −5.4 sea detectable.

### Schemas confirmados
- `definition` + `statistics` presentes en OPRA. Open interest disponible.
- Cotizaciones: **`cbbo-1s`, `cbbo-1m`, `tcbbo`** — no hay `mbp-1` ni `bbo-1m` para OPRA. No es un problema: `cbbo` es el **consolidated BBO = NBBO**, que es exactamente lo que el plan pedía para el mid.

### Costo del piloto de 1 año (32 subyacentes)
| Schema | Alcance | Costo |
|---|---|---|
| `definition` | año completo | $44.28 |
| `statistics` (open interest) | año completo, día entero | **$1,360.24** |
| `cbbo-1m` | año completo, sesión entera | $2,127.79 (referencia, no se usa) |
| `cbbo-1m` | **ventana 15:55–16:00 ET × 250 días** | **$56.42** |
| `ohlcv-1d` equities | año completo | $0.01 |
| | **TOTAL** | **$1,460.93** |

**El 93% del costo está en un solo schema.** Y ya está demostrado que se puede optimizar: en cotizaciones, pedir solo la ventana de cierre costó **38× menos** por el mismo dato útil ($2,128 → $56). La Fase 0.B verifica si el mismo truco aplica al open interest.

### Bug corregido
`get_record_count` exige `end > start`; no admite `start == end`. Por eso falló el bloque 3 de la Fase 0.A. Corregido en `fase0b_optimizacion.py`.
(El timeout 504 de `cbbo-1m` a un año es irrelevante: nunca vamos a pedir la sesión entera.)

### Siguiente: Fase 0.B
`python3 fase0b_optimizacion.py` — mide en qué franja horaria llega el open interest, cuánto cuesta pedir solo esa franja, confirma `stat_type=9` con una descarga mínima, y proyecta el costo de los tres alcances posibles (1 año / 2024-07 / 2018-05).
