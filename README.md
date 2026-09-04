# gamma-quant — contraste de la hipótesis de Gamma Exposure (GEX)

> Este README describe el proyecto de la **raíz**. El repositorio aloja además
> un segundo proyecto independiente, `GEX_Asset_Management/` — ver
> [Dos líneas de trabajo](#este-repositorio-contiene-dos-líneas-de-trabajo).

Un aparato de medida, **no una estrategia**. Construye GEX de dealers de forma
auditable a partir de cadenas de opciones de SPY/SPX y después intenta **tumbar**
la hipótesis de que ese GEX predice algo.

- El plan completo, los supuestos y los criterios de aceptación: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
- Cómo trabajar en el repo (ramas, PRs): **[CONTRIBUTING.md](CONTRIBUTING.md)**

---

## Este repositorio contiene DOS líneas de trabajo

No son fases de lo mismo, y tampoco son independientes: **el de la derecha ya
respondió, con datos reales, la pregunta que el de la izquierda se disponía a
investigar** — y la respuesta fue que no. Distinto paper, distinta fuente, distinto
universo. **No comparten una sola línea de código.** Antes de tocar nada, mira en
cuál de los dos estás.

| | **`gamma_quant/`** (raíz) | **`GEX_Asset_Management/`** |
|---|---|---|
| **Pregunta** | ¿El GEX de dealers sobre índice predice algo? | ¿La señal Γ de Soebhag (2023) predice volatilidad en acciones? |
| **Universo** | SPY y SPX | 30 mega-caps + SPY + QQQ |
| **Fuente** | CBOE diferido, archivado hacia delante | Databento OPRA |
| **Muestra** | Empieza 2026-08-31. Un puñado de días | 2021-09-03 → 2026-08-31 · 267 fechas · 15,6 M contrato-día |
| **Stack** | numpy + pandas, `pyproject.toml` | polars + numba, `requirements.txt` |
| **Estado** | Motor hecho y calibrado. ⛔ **DETENIDO** (§0.0) | Piloto **terminado**. El mecanismo **no se confirma** |
| **Empieza por** | este README y [PROJECT_PLAN.md](PROJECT_PLAN.md) | [su README](GEX_Asset_Management/README.md) y [docs/RESULTADOS.md](GEX_Asset_Management/docs/RESULTADOS.md) |

Sólo hay **un** punto de contacto, y es deliberado:
[research/gamma_iv_mechanics.py](research/gamma_iv_mechanics.py) usa el motor de
`gamma_quant/` para explicar en forma cerrada una anomalía de signo que apareció en
`GEX_Asset_Management/reports/ortogonal.txt`. La dependencia va en un solo sentido.

**Dos trampas al trabajar en los dos a la vez:**

1. **Hay dos carpetas `reports/` con políticas OPUESTAS.** La de la raíz la escribe
   una máquina y está fuera de git. La de `GEX_Asset_Management/` **es la evidencia**
   de las corridas y sí se versiona. No unifiques esas reglas de `.gitignore`.
2. **Hay dos entornos de dependencias.** `pip install -e ".[all]"` no instala
   `databento`, `polars` ni `numba`; `requirements.txt` no instala este paquete.
   Si vas a trabajar en los dos, considera dos entornos virtuales.

---

## Lo primero que hay que saber

> ⛔ **Este proyecto está DETENIDO desde el 2026-09-04**, y su premisa de partida
> resultó ser falsa. No empieces a construir sobre él sin leer
> [PROJECT_PLAN.md §0.0](PROJECT_PLAN.md) y la decisión pendiente de **§14**.

Dos cosas que este README afirmaba y que no eran ciertas:

1. **"No hay histórico de opciones."** Sí lo hay, en el propio repositorio:
   `GEX_Asset_Management/` tiene 15,6 M de contrato-día de cadena OPRA (2021-09 →
   2026-08). El error fue auditar el entorno una sola vez, el 31 de agosto, y no
   volver a mirar cuando el repo cambió.
2. **"La pregunta sigue abierta."** No lo está. Ese proyecto ya contrastó estas
   hipótesis con pre-registro y placebos sobre datos reales, y **no se sostienen**:
   régimen no respaldado (percentil 58 de barajados aleatorios), VRP con signo
   invertido, intradía 5 fallos de 5 con el placebo rindiendo *mejor* que la señal.

**Y el hallazgo que afecta directamente a este motor:** la gamma de Black-Scholes es
`φ(d₁)/(S·σ·√T)`, y lleva `1/σ` dentro. Cualquier GEX construido con ella decrece con
la volatilidad implícita **por construcción, no por economía** — medido,
`corr(Γ, 1/IV) = +0,573`. Al ortogonalizar contra la IV, el poder predictivo pasa de
`t = −2,48` a `t = +0,04`. Este repositorio montó cuatro definiciones de GEX y un
marco de placebos para la *convención de signo*; el confundidor real estaba un nivel
más abajo, dentro de la gamma, y no está en el registro de supuestos.

Lo que sigue siendo cierto y no depende de nada de lo anterior:

- **El posicionamiento real de los dealers no es observable.** Todo GEX público lo
  infiere del open interest mediante una convención de signo que no se puede
  verificar con datos públicos.
- **Nada marcado `SINTÉTICO — NO ES EVIDENCIA` cuenta como evidencia.**

## Qué está hecho, y para qué sirve ahora

El instrumental es correcto y está testeado (234 comprobaciones). Lo que ya no tiene
es una pregunta abierta que responder.

| Parte | Estado |
|---|---|
| `options/`: pricing, griegas, GEX, gamma flip, muros | Hecho. **Calibrado contra Bloomberg**, error mediano 1,25% sobre SPX real |
| Lector de exports OMON (spot y dividendo por paridad put-call) | Hecho |
| Config TOML, logging, registro de experimentos | Hecho |
| `data/`: ABC de proveedor, CBOE, sintético, validación, almacén Parquet | Hecho |
| Archivador diario de CBOE | Funciona. **Si sigue corriendo o no depende de §14** |
| features, strategies, backtest, research, models, visualization | Vacíos, y ya no está claro que deban llenarse |

Las cuatro piezas que **no** existen en `GEX_Asset_Management/` —calibración contra
Bloomberg, gamma flip por revaloración de la cadena entera, muros de gamma y lector
OMON— son las candidatas a sobrevivir. Es la opción (A) de §14.

Los directorios vacíos tienen sólo un `__init__.py`. No los cuentes como hechos.

## Instalación

Requiere **Python 3.11+** (`tomllib` es de la biblioteca estándar desde 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate en Unix
pip install -e ".[all]"
cp .env.example .env            # y pon tu contacto en GAMMA_QUANT_USER_AGENT
```

`pip install -e .` a secas instala sólo el núcleo (numpy, pandas, scipy,
requests). Los extras importan: sin `pyarrow` el almacén cae a CSV comprimido, y
sin `openpyxl` la suite de calibración contra Bloomberg **se salta en silencio**.

## El archivador

```bash
python archive_chains.py              # archiva SPY y SPX de hoy
python archive_chains.py --coverage   # qué hay archivado y qué huecos hay
python archive_chains.py --dry-run    # descarga y valida, no guarda
```

**Ya no es "lo urgente".** Lo era mientras se creía que sin este archivo no había
datos con los que contrastar nada; esa premisa era falsa (§0.0). Sigue funcionando y
sigue siendo barato, y la muestra sólo crece si corre — pero **si debe seguir
programado es parte de la decisión de §14**, no algo que este README pueda dar por
sentado.

Si se deja corriendo, la regla no cambia: programarlo de lunes a viernes hacia las
16:30 ET (22:30 hora peninsular) y **revisar `--coverage` de vez en cuando**. Un
archivador que lleva tres semanas fallando en silencio es peor que no tenerlo, porque
produce una muestra sesgada en lugar de una muestra corta. Las instrucciones de
`schtasks` están en la cabecera de [archive_chains.py](archive_chains.py).

## Pruebas

```bash
python run_tests.py          # todas
python run_tests.py -v       # con la salida completa de cada suite
python run_tests.py gex      # sólo las que casen con ese texto
```

Sin pytest: cada `tests/test_*.py` es un script autónomo que imprime su informe y
sale con 1 si falla. Hoy son 3 suites y 234 comprobaciones.

`tests/test_bloomberg_omon.py` es distinta de las otras: no comprueba matemáticas
conocidas, comprueba que reproducimos **la referencia institucional** sobre datos
reales de SPX. Si empieza a fallar, o hemos roto las griegas o alguien ha cambiado
`r`, `q` o `tau` sin darse cuenta.

## Investigación

```bash
python research/calibrate_from_bloomberg.py   # de dónde sale dividend_yield = 0,30%
python research/probe_data_sources.py         # qué vende de verdad cada fuente
```

`research/` son scripts que se ejecutan; `gamma_quant/` es el paquete que
importan. La dependencia va en un solo sentido.

## Dónde vive cada cosa

El árbol completo y comentado está en [PROJECT_PLAN.md §2](PROJECT_PLAN.md). Lo
que hay que saber antes de mover un fichero:

| Ruta | Regla |
|---|---|
| `data/archive/` | **Irreemplazable.** Fuera de git y necesita copia de seguridad propia |
| `data/external/` | Entradas manuales (exports OMON). **Sí se versionan**, son ~30 KB y nadie puede regenerarlas |
| `data/{raw,processed}/` | Fuera de git |
| `reports/` | Lo escribe una máquina. Fuera de git |
| `configs/default.toml` | **Ningún número que sea una decisión de modelado puede vivir en el código.** Si alguien puede discrepar de un valor, va aquí |
| `.env` | Secretos. Nunca se comitea; copia de `.env.example` |

## La regla que gobierna el proyecto

Un umbral escondido en un `if` es un grado de libertad que nadie audita y que
acaba sobreajustado. Toda decisión de modelado —convención de signo, definición
de GEX, lag del OI, escalón de costes— es configuración, y toda evaluación se
anota en `reports/experiment_registry.jsonl`, que es append-only. El Sharpe
deflactado necesita saber cuántas veces lo intentamos, y ese número sólo es
creíble si nadie puede borrar los intentos que salieron mal.
