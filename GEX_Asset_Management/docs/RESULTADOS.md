# Resultados

**Muestra:** 2021-09-03 → 2026-08-31 · 267 fechas semanales · 30 acciones + SPY y QQQ
**Volumen:** 15 617 846 contrato-día de cadena OPRA
**Última ejecución completa:** 2026-09-03

Todas las cifras de este documento salen de los ficheros de `reports/`, regenerados
en esa corrida. Cuando una cifra aparece también en `FICHA_ESTRATEGIA.md`, es la misma.

> **Nota sobre el Sharpe.** Aquí se reporta siempre el **exceso sobre la tasa libre de
> riesgo** (curva del Tesoro a 3 meses, media del periodo 3,76 %). Los ficheros
> `reports/backtest*.txt` imprimen `retorno/volatilidad` sin restar la tasa, así que
> muestran un número mayor. La cifra correcta para comparar es la de este documento.

---

## 1. Motor de valuación — Puerta P1

`python3 tests/test_pricing_gate_p1.py` → **25/25 chequeos superados.**

| Chequeo | Resultado |
|---|---|
| CRR converge a Black-Scholes (europeo, sin dividendos) | error agregado de Σγ = 1,2e-3 |
| Call americana sin dividendos == europea (teorema) | error exactamente 0 |
| Prima de ejercicio anticipado del put ≥ 0 | 238 casos, ninguno negativo |
| Round-trip precio → IV → precio | 1,3e-10 |
| Paridad put-call | 3,0e-11 |
| Auto-convergencia de gamma N=400 vs N=1600 | mediana 6,7e-4 · p95 1,1e-2 |
| Rendimiento | 60 700 greeks/s · 10 700 inversiones de IV/s |

**Parámetros de producción:** `N_STEPS = 400`, IV ∈ [1 %, 500 %], semilla analítica BSM
con refinamiento por secante, atajo exacto para calls sin dividendos.

---

## 2. Magnitudes de Γ — Puerta P2

`run_senal.py` → `reports/p2_senal.txt`

- 15 617 846 contrato-día ingestados → **13 605 587 tras filtros de calidad** (87,1 %)
- **13 268 353 con IV válida** (97,5 %) · 1 057 s de cómputo
- r implícito mediano: **4,10 %** (leído de la curva del Tesoro, ver §7)

Benchmark: la **Tabla 2** del paper (value-weighted), no la Tabla 1. Razón: Γ lleva el
ADV$ en el denominador, y un universo de mega-caps no es comparable con el CRSP completo.

| Estadístico | Nuestro | Tabla 2 (VW) | Veredicto |
|---|---|---|---|
| P10 (~decil L) | −0,010 | −0,01 | ✅ |
| Mediana (~decil 9) | 0,016 | 0,02 | ✅ |
| P90 (~decil H) | 0,068 | 0,04 | ✅ |
| Rango P90−P10 (~H−L) | 0,078 | 0,05 | ✅ |
| % observaciones negativas | 22,59 % | 21,8 % | ⚠️ REVISAR |

**Validaciones externas:** Γ de índice negativa (SPY −0,070, QQQ −0,037) sin habérselo
indicado al pipeline — valida la convención de signos. 77 % de acciones con Γ positiva
(paper: 78,2 %).

**El único REVISAR se explica por régimen:**

| Año | % Γ negativa |
|---|---|
| 2021 | 17,3 % |
| **2022** | **48,5 %** |
| 2023 | 21,1 % |
| 2024 | 12,2 % |
| 2025 | 12,7 % |
| 2026 | 19,8 % |

Todos los años cumplen < 21,8 % salvo 2022, el año bajista. Es coherente: con el mercado
cayendo, los inversores se cargan de puts y los dealers quedan cortos gamma.
**P2 se da por superada.**

---

## 3. Estabilidad del ranking — Puerta P2b

`run_sensibilidad.py` → `reports/p2b_sensibilidad.txt` · 67 fechas · 3 403 437 contratos

| Escenario | Spearman | Quintil L | Quintil H |
|---|---|---|---|
| Método viejo (r de la paridad) | 0,9938 | 96 % | 94 % |
| r = 3,0 % + dividendos manuales | 0,9942 | 94 % | 96 % |
| r = 5,0 % + dividendos ×2 | 0,9857 | 92 % | 94 % |
| + ruido IV 0,5 pt vol | 0,9994 | 100 % | 100 % |
| + ruido OI 1 % | 0,9996 | 100 % | 99 % |
| + ruido OI 5 % | 0,9979 | 97 % | 98 % |

**Spearman mínimo 0,9857 > 0,98 → pasa.** El temor del hallazgo H1 (que la cancelación
net/gross amplificara errores en el quintil L) no se materializa en el *ranking*: afecta
al nivel de Γ, no al orden.

---

## 4. Test de mecanismo — Puertas P3 y P3b · **el resultado principal**

### P3 — panel con efectos fijos (`reports/p3_mecanismo.txt`)

| Especificación | coef | t |
|---|---|---|
| 1. Γ sola (con FE de acción) | −0,0066 | −0,84 |
| 2. + volatilidad realizada previa | −0,0053 | −0,68 |
| 3. + IV mediana | +0,0047 | 0,58 |
| 4. + log(ADV$) y log(precio) | **+0,0004** | **0,06** |
| 5. + razón de cancelación | +0,0296 | 2,26 |

### P3b — Fama-MacBeth, la especificación del paper (`reports/p3b_mecanismo_fm.txt`)

| Especificación | coef | t (NW) |
|---|---|---|
| Γ sola (col. 1 del paper) | −1,430 | **−2,37** |
| + volatilidad realizada previa | −0,381 | −0,66 |
| **+ IV mediana** | **+0,039** | **+0,08** |
| + log(OI), log(ADV$), log(precio) | −0,167 | −0,54 |

### La descomposición de identificación del paper — **esto sí replica**

| Componente | coef | t (NW) | El paper |
|---|---|---|---|
| Γ_viejo (re-balanceo de cobertura) | −1,864 | **−3,86** | t = −4,57 |
| Γ_info (información privada) | +2,205 | 0,49 | no significativo |
| Γ_viejo controlando por IV | −0,344 | −1,07 | — |

Γ_viejo usa el open interest de la semana anterior con el precio de hoy: son posiciones
que ya existían y no pueden venir de información posterior. Que el componente
significativo sea ese y no el de información **es la firma que el paper usa** para
atribuir el efecto a cobertura. Se reproduce.

### Veredicto

**MECANISMO NO CONFIRMADO.** La relación negativa entre Γ y volatilidad futura es real
y significativa sin controles, replica la estructura del paper, y **desaparece al
controlar por volatilidad implícita**.

La causa es aritmética antes que económica: la gamma de Black-Scholes lleva
`1/(S·σ·√T)` en su definición, así que Γ es una función decreciente de σ **por
construcción**. Como σ es persistente, Γ "predice" la volatilidad futura porque *es*
una función de la volatilidad presente.

Evidencia directa, sobre el promedio de 5 años de las 30 acciones:

| Correlación | Valor |
|---|---|
| corr(Γ, 1/IV) | **+0,573** |
| corr(Γ, IV) | −0,442 |
| corr(Γ, log ADV$) | +0,062 |

No es tamaño ni liquidez: es volatilidad, con la forma funcional exacta que impone la
fórmula. La pata larga acaba con IV media de 48,3 % y la corta con 25,8 %.

---

## 5. Test de ortogonalización (`reports/ortogonal.txt`) · 259 semanas

Se quita de Γ, fecha por fecha, la parte explicada por la IV, y se testea el residuo.

| Especificación | coef | t (NW) |
|---|---|---|
| Γ cruda (referencia) | −1,4103 | −2,48 |
| **Γ ortogonal a IV** | **+0,0249** | **+0,04** |
| Γ ortogonal a IV y vol previa | +0,0400 | 0,08 |
| *control: IV sola* | 0,8664 | *22,56* |
| *control: vol previa sola* | 0,5343 | *22,88* |

La Γ agregada pierde todo su poder. Los controles muestran quién predecía de verdad.

**Excepción, y es la única:** el componente `slow` (>31 d) —donde el paper dice que vive
la señal— **sobrevive**: t = −6,52 crudo, **t = −2,66 ortogonalizado**, con `fast` nulo
(t = −0,22), justo como predice el paper. Pero el componente OTM sale con t = +4,24 y el
**signo contrario** al predicho, lo que sugiere un problema en la descomposición.

---

## 6. Tests adicionales — todos pre-registrados

Parámetros congelados **antes** de correr, en [`preregistros/`](preregistros/).

| # | Hipótesis | Script | Resultado |
|---|---|---|---|
| 1 | Γ → prima de varianza (RV − IV) | `run_vrp.py` | ❌ t = +0,89 · spread Q0−Q4 con **signo invertido** |
| 2 | GEX índice → switch momentum/reversal | `run_regimen.py` | ❌ 4 de 5 criterios fallan |
| 3 | GEX índice → semáforo de exposición | `run_semaforo.py` | ⚠️ pasa en 30 nombres, falla en SPY |
| 4 | Γ → momentum intradía (Baltussen 2021) | `run_intradia.py` | ❌ 0 de 5; test parcial |

### El test de la prima de varianza cierra el hilo abierto

`run_vrp.py` pregunta si Γ predice **RV − IV**, que es lo que cobra un straddle. Restar
la IV elimina el artefacto por construcción, así que es la prueba más limpia.

| Quintil de Γ | IV ATM | RV futura | Prima RV−IV |
|---|---|---|---|
| Q0 (Γ baja → comprar vol) | 44,1 % | 41,2 % | −0,0284 |
| Q4 (Γ alta → vender vol) | 31,1 % | 31,2 % | +0,0011 |

**Spread Q0−Q4 = −0,0296 · t = −0,98.** El signo está al revés del que la estrategia
necesitaría. Y el componente `slow`, que sobrevivía en §5, aquí muere (t = −1,06).

Conclusión operativa: **no construir la versión con opciones.** Como Γ baja == IV alta,
comprar volatilidad donde Γ es baja es comprar sistemáticamente las opciones más caras
del universo y pagar la prima de riesgo de varianza en cada rotación.

### El hallazgo transversal

En los cuatro tests **la única ventaja aparece en 2022**. Fuera de ese año, todo empata
o pierde. Es coherente con la teoría: el flujo de cobertura de los dealers domina el
precio bajo estrés y es ruido frente al flujo fundamental cuando no lo hay.

**Γ se comporta como indicador de fragilidad, no de dirección.** Lo único consistente en
todos los periodos es la reducción de drawdown.

### Advertencia sobre comparaciones múltiples

Se probaron siete hipótesis sobre la misma muestra de 267 semanas. Con siete pruebas al
5 %, la probabilidad de al menos un falso positivo ronda el 30 %. El semáforo "pasando"
en una de sus dos versiones encaja con ese perfil, y el desglose anual confirmó que era
2022. **No se debe tratar como hallazgo.**

---

## 7. Dos defectos del pipeline encontrados y corregidos

Ninguno era visible en los resultados finales — solo al contrastar contra fuentes externas.

### La ventana del open interest estaba calibrada al régimen equivocado

OPRA movió la ráfaga de OI de las 13:31/14:31 UTC a las 10:30 UTC alrededor de abril 2023.
La ventana fija `OI_WIN = (10:00, 12:00)` devolvía **cero OI** para todo dato anterior.
Ampliada a `(10:00, 15:00)`, que captura ambos regímenes.

### La tasa libre de riesgo se estimaba, y salía 3-5× por debajo

`implied_carry` la sacaba de la pendiente de la paridad put-call. Medido contra la realidad:

| Año | r estimada | r real |
|---|---|---|
| 2021 | 0,22 % | ~0,05 % |
| 2023 | 1,02 % | 4,5–5,5 % |
| 2025 | 1,24 % | ~4,3 % |

Causa: con `T_MIN_FIT = 0,05` entran muchos vencimientos cortos, y a T = 0,05 con r = 5 %
el factor de descuento vale 0,9975 — distinguir r = 5 % de r = 1 % exige resolver la
pendiente a 0,002, por debajo del ruido de cotización.

En cadena: r baja → DF alto → `D = S − K·DF − (c−p)` negativo → el piso `max(D, 0)` lo
topaba → **dividendo cero en 28 de 32 tickers**.

**Corregido** con `gex/curves.py`: la curva diaria del Tesoro se lee de FRED (series
DGS1MO/3MO/6MO/1/2, sin API key), se convierte a capitalización continua con `ln(1+y)`
y se interpola `r(T)` al plazo de cada contrato. La paridad put-call se reserva para
dividendo + costo de préstamo, que es donde sí vive.

Resultado: r mediana 4,10 %, dividendos correctos (CVX 3,64 %, XOM 3,12 %, KO 2,55 %,
no-pagadores ~0 %, GME 1,23 % por hard-to-borrow).

---

## 8. Un límite del proveedor que obligó a cambiar de fuente

**Databento no tiene volumen consolidado antes de 2024-07-01** (inicio de `EQUS.SUMMARY`).
Medido sobre el solape, como fracción del volumen consolidado real:

| Fuente | % del consolidado |
|---|---|
| `XNAS.ITCH` + `XNYS.PILLAR` + `ARCX.PILLAR` sumados | 27–55 %, heterogéneo por ticker |
| `DBEQ.BASIC` | 1,3–6,4 % |
| `EQUS.MINI` | 3,2–5,4 % |

Cada feed trae solo lo ejecutado en su bolsa; el resto vive en Cboe, MEMX, IEX y dark
pools. Ningún factor constante lo corrige.

**Solución:** yfinance para 2021-08 → 2024-07, des-ajustado por split. Reproduce
`EQUS.SUMMARY` con error mediano de **0,0000 % en cierre y 0,0001 % en volumen** sobre
17 024 filas del solape.

**Trampa crítica:** yfinance ajusta por split *siempre*, incluso con `auto_adjust=False`
(ese flag solo controla dividendos). Hay que des-ajustar multiplicando el precio por el
producto de los ratios de splits posteriores a cada fecha. Sin eso el spot no casa con
los strikes históricos de OPRA — AMZN habría dado spot 122 contra strikes en 2 450.
Verificado contra la escalera de strikes en 5 tickers pre-split: todos casan.

---

## 9. Backtest — cartera neutralizada por beta y sector

Cuadro completo en [`FICHA_ESTRATEGIA.md`](FICHA_ESTRATEGIA.md). Resumen:

| | GEX Neutral | SPY |
|---|---|---|
| CAGR | 11,24 % | 10,84 % |
| Volatilidad | 15,73 % | 16,07 % |
| **Sharpe** (exceso sobre rf) | **0,52** | 0,49 |
| Sortino | 0,90 | 0,75 |
| Calmar | 0,62 | 0,44 |
| Max drawdown | **−18,17 %** | −24,80 % |
| Duración del drawdown | **35 sem** | 106 sem |
| Alfa de Jensen anual | +6,35 % · **t = 0,93** | — |
| Beta | 0,25 | 1,00 |
| R² contra SPY | 0,06 | 1,00 |

Comparación de diseños de rebalanceo (`reports/backtest_freq.txt`), sobre la misma señal:

| Diseño | Ret. anual neto | Sharpe (ret/vol) | Max DD |
|---|---|---|---|
| Semanal 6v6 | 7,25 % | 0,21 | −53,1 % |
| Mensual (el del paper) | 15,16 % | 0,43 | −49,7 % |
| **Neutral beta+sector** | **12,59 %** | **0,75** | **−18,2 %** |

La neutralización es lo que hace el trabajo: beta neta de 0,37 → 0,10, volatilidad a la
mitad, drawdown de −49,7 % a −18,2 % **sobre la misma señal**.

> Una variante que **se descarta**: añadir una banda de tolerancia de 6 posiciones al
> rebalanceo mensual sube el retorno de 15,2 % a 39,6 %. Con 60 rebalanceos y la
> volatilidad subiendo en paralelo, eso es la muestra siendo torturada por un parámetro
> de implementación. No se reporta como resultado.

---

## 10. Análisis de riesgo (`reports/riesgo.txt`) · 265 semanas

| Cartera | VaR 95 % | CVaR 95 % | Peor semana | Vol anual |
|---|---|---|---|---|
| GEX Neutral | −2,78 % | **−4,09 %** | −7,41 % | 15,7 % |
| SPY | −3,06 % | −4,60 % | −9,07 % | 16,1 % |
| 30 nombres EW | −5,46 % | −7,46 % | −10,97 % | 25,3 % |
| L-S semanal | −6,39 % | −9,34 % | −18,58 % | 33,3 % |

**Bootstrap por bloques** (10 000 simulaciones, bloques geométricos de 8 semanas):

| Cartera | Ret. medio | IC 90 % | P(pérdida) | P(gana a SPY) |
|---|---|---|---|---|
| GEX Neutral | 11,01 % | −0,2 % · +23,2 % | 5,4 % | **48,8 %** |
| SPY | 11,16 % | +0,1 % · +22,3 % | 4,9 % | — |

**2022, el único episodio de estrés real de la muestra:**

| Periodo | GEX Neutral | SPY | 30 nombres EW |
|---|---|---|---|
| 2022 completo | **+15,4 %** | −16,4 % | −42,4 % |
| Caída ene–oct 2022 | **+25,1 %** | −23,8 % | −46,8 % |
| Resto (2023–2026) | 9,1 % | 20,1 % | 39,8 % |

**Pero no es una cobertura.** En las 15 semanas con SPY por debajo de −3 %, la neutral
también perdió (−1,18 % en media), y su beta en semanas de caída es **+0,20**, positiva.
No protege en los golpes: tiene beta baja. La ganancia de 2022 vino del spread
cross-seccional acumulado a lo largo del año.

---

## 11. Qué se puede afirmar, y qué no

**Se puede afirmar:**

- El pipeline calcula Γ con magnitudes consistentes con la literatura, validado contra
  la Tabla 2 del paper y con la convención de signos confirmada dos veces.
- La estructura de identificación del paper (cobertura significativa, información no)
  **se reproduce**.
- El efecto **no sobrevive al control por volatilidad implícita**, y la razón es
  mecánica: γ ∝ 1/σ por construcción.
- La neutralización por beta y sector mejora el perfil de riesgo de forma sustancial y
  transferible.

**No se puede afirmar:**

- Que el paper esté equivocado. Su muestra es 1996–2021 con miles de acciones; esta es
  2021–2026 con 30 mega-caps. Otro universo, otro régimen.
- Que la crítica del control por IV le aplique al paper. Habría que verificar si su
  Tabla 13 lo incluye — no se asumió.
- Que la estrategia genere alfa. 60 meses no dan potencia: el IC del retorno anual va de
  −0,2 % a +23,2 %, y el alfa tiene t = 0,93.
- Que el comportamiento de 2022 se repita. Una observación no distingue mecanismo de
  casualidad.

**Criterio de promoción del plan original:** escalar a 500–1 000 nombres solo si P1–P4
pasan **y** el mecanismo sale significativo. **No se cumple → NO-GO.**

### Limitación estructural del piloto

Con 30 acciones y quintiles quedan **6 nombres por pata**: el ruido idiosincrático domina
y ningún test cross-seccional tiene potencia real. Los papers de referencia usan miles de
acciones. Coste medido de ampliar a ~200 nombres: **255 USD/año** en weekly, y el costo
escala sublinealmente porque las mega-caps tienen las cadenas más grandes.

---

## 12. Reproducir

```bash
python3 run_ingesta.py --dry-run                  # costo, sin descargar
python3 run_ingesta.py --scope pilot --freq weekly # descarga (cuesta dinero)
python3 run_senal.py                              # P2
python3 run_sensibilidad.py                       # P2b
python3 run_mecanismo.py && python3 run_mecanismo_fm.py   # P3 y P3b
python3 run_ortogonal.py && python3 run_vrp.py    # tests de la señal
python3 run_backtest_neutral.py                   # la estrategia
python3 run_metricas.py && python3 run_riesgo.py  # cuadros finales
```

`tests/test_pricing_gate_p1.py` no necesita datos y sale con código 1 si algo falla.
