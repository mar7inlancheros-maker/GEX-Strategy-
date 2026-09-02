# gamma-quant — contraste de la hipótesis de Gamma Exposure (GEX)

Un aparato de medida, **no una estrategia**. Construye GEX de dealers de forma
auditable a partir de cadenas de opciones de SPY/SPX y después intenta **tumbar**
la hipótesis de que ese GEX predice algo.

- El plan completo, los supuestos y los criterios de aceptación: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
- Cómo trabajar en el repo (ramas, PRs): **[CONTRIBUTING.md](CONTRIBUTING.md)**

---

## Lo primero que hay que saber

**No hay histórico de cadenas de opciones.** Las fuentes gratuitas sirven el
presente y nada más; el pasado no se puede comprar después. El proyecto archiva
**hacia delante** desde el 2026-08-31.

Consecuencias, y no son negociables:

1. **El archivador es lo urgente.** Un día que no se guarde hoy no existirá nunca.
   Ver [Uso diario](#uso-diario).
2. **Hoy no hay resultados de estrategia, sólo pruebas del motor.** Cualquier
   número que salga de datos sintéticos va marcado `SINTÉTICO — NO ES EVIDENCIA`
   y no cuenta como evidencia de nada.
3. **El posicionamiento real de los dealers no es observable.** Todo GEX público
   —este incluido— lo infiere del open interest mediante una convención de signo
   que no se puede verificar. Por eso la convención es configuración
   intercambiable y por eso invertirla es un placebo de primera clase.

## Qué está hecho y qué no

| Parte | Estado |
|---|---|
| Config TOML, logging, registro de experimentos | Hecho |
| `options/`: pricing, griegas, GEX, gamma flip, muros | Hecho, **calibrado contra Bloomberg** (error mediano 1,25%) |
| `data/`: ABC de proveedor, CBOE, OMON, sintético, validación, almacén Parquet | Hecho |
| Archivador diario | Hecho, **debe estar corriendo** |
| features, strategies, backtest, research, models, visualization | Vacíos. Fases 7-14 |

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

## Uso diario

```bash
python archive_chains.py              # archiva SPY y SPX de hoy  <- LO IMPORTANTE
python archive_chains.py --coverage   # qué hay archivado y qué huecos hay
python archive_chains.py --dry-run    # descarga y valida, no guarda
```

Programarlo de lunes a viernes hacia las 16:30 ET (22:30 hora peninsular) y
**revisar `--coverage` de vez en cuando**: un archivador que lleva tres semanas
fallando en silencio es peor que no tenerlo, porque produce una muestra sesgada
en lugar de una muestra corta. Las instrucciones de `schtasks` están en la
cabecera de [archive_chains.py](archive_chains.py).

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
