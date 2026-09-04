#!/usr/bin/env python3
"""FASE 0 -- Verificacion de datos en Databento.  GO / NO-GO del proyecto.

Responde cuatro preguntas antes de gastar un dolar en datos:
  1. Que rango historico real tiene OPRA en esta cuenta (el dato en conflicto:
     el blog de Databento dice 28-mar-2023, la pagina comercial dice 2013).
  2. Que schemas hay, y si `statistics` trae open interest para OPRA.
  3. Cuantos contratos vivos por subyacente hay realmente.
  4. Cuanto cuesta en USD el piloto completo, por schema.

USO
    cd "<esta carpeta>"
    python3 fase0_verificacion.py

La key se lee de .env (ya creado, chmod 600) o de la variable de entorno
DATABENTO_API_KEY. Nunca se pasa por linea de comandos.
Requiere:  pip install databento
La salida se guarda en reports/fase0_verificacion.txt
"""
from __future__ import annotations

import os
import pathlib
import sys
from datetime import date, datetime

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "reports" / "fase0_verificacion.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ parametros
NUCLEO = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA",
          "NFLX", "JPM", "BAC", "XOM", "CVX", "DIS", "BA", "WMT", "KO", "PG"]
DISPERSION = ["PLTR", "COIN", "MSTR", "GME", "RIVN", "SOFI", "UBER", "SHOP",
              "MU", "CRM", "GS", "CAT"]
INDICES = ["SPY", "QQQ"]                       # track paralelo, no van al ranking
CROSS_SECTION = NUCLEO + DISPERSION
TODOS = CROSS_SECTION + INDICES

START = "2025-09-02"
END = "2026-08-31"
TRADING_DAYS = 250
SAMPLE_DAY = "2026-06-10"                      # miercoles normal, sin vencimiento mensual

OPT = "OPRA.PILLAR"
EQ_CANDIDATES = ["EQUS.SUMMARY", "XNAS.ITCH", "XNYS.PILLAR", "EQUS.MINI", "DBEQ.BASIC"]

_lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _lines.append(s)


def load_key() -> str:
    k = os.environ.get("DATABENTO_API_KEY")
    if k:
        return k.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("DATABENTO_API_KEY"):
                return line.split("=", 1)[1].strip()
    sys.exit("No encuentro la API key. Pon DATABENTO_API_KEY en .env o en el entorno.")


def fmt_usd(x) -> str:
    return f"${x:,.2f}" if isinstance(x, (int, float)) else str(x)


def main() -> int:
    try:
        import databento as db
    except ImportError:
        sys.exit("Falta el cliente:  pip install databento")

    c = db.Historical(load_key())

    say("=" * 84)
    say("FASE 0 -- VERIFICACION DE DATOS EN DATABENTO".center(84))
    say(f"{datetime.now():%Y-%m-%d %H:%M}  ·  piloto {START} -> {END}".center(84))
    say("=" * 84)

    # ---------------------------------------------------- 1. datasets y rangos
    try:
        datasets = set(c.metadata.list_datasets())
    except Exception as e:
        say(f"\nFALLO al listar datasets: {type(e).__name__}: {e}")
        return 1
    say(f"\nDatasets visibles para esta key ({len(datasets)}):")
    for d in sorted(datasets):
        say(f"    {d}")

    say("\n" + "-" * 84)
    say("1. RANGO HISTORICO  <-- la pregunta GO/NO-GO")
    say("-" * 84)
    ranges: dict[str, tuple] = {}
    for d in [OPT] + EQ_CANDIDATES:
        if d not in datasets:
            say(f"    {d:<16} no disponible para esta key")
            continue
        try:
            r = c.metadata.get_dataset_range(dataset=d)
            s = str(r.get("start") or r.get("start_date"))[:19]
            e = str(r.get("end") or r.get("end_date"))[:19]
            ranges[d] = (s, e)
            say(f"    {d:<16} {s}  ->  {e}")
        except Exception as ex:
            say(f"    {d:<16} ERROR {type(ex).__name__}: {ex}")

    if OPT in ranges:
        s = ranges[OPT][0][:10]
        say("")
        say(f"    >>> OPRA arranca en {s}.")
        try:
            meses = (date.today() - date.fromisoformat(s)).days / 30.44
            say(f"    >>> Historia total disponible: ~{meses:.0f} meses.")
            say(f"    >>> Piloto de 1 ano: {'VIABLE' if date.fromisoformat(s) <= date.fromisoformat(START) else 'NO VIABLE, ajustar START'}")
            say(f"    >>> Build completo (>= 24 meses): {'VIABLE' if meses >= 24 else 'NO VIABLE'}")
        except Exception:
            pass

    # ------------------------------------------------------------- 2. schemas
    say("\n" + "-" * 84)
    say("2. SCHEMAS DISPONIBLES")
    say("-" * 84)
    schemas: dict[str, list] = {}
    for d in [OPT] + [x for x in EQ_CANDIDATES if x in datasets][:2]:
        if d not in datasets:
            continue
        try:
            sch = list(c.metadata.list_schemas(dataset=d))
            schemas[d] = sch
            say(f"    {d}:")
            say("        " + ", ".join(sch))
        except Exception as ex:
            say(f"    {d}: ERROR {ex}")
    if OPT in schemas:
        need = ["definition", "statistics"]
        falta = [s for s in need if s not in schemas[OPT]]
        say("")
        say(f"    >>> definition + statistics (open interest): "
            f"{'AMBOS PRESENTES' if not falta else 'FALTA ' + ', '.join(falta)}")
        quote_opts = [s for s in schemas[OPT]
                      if s in ("bbo-1m", "bbo-1s", "cbbo-1m", "cbbo-1s", "mbp-1", "tbbo", "tcbbo")]
        say(f"    >>> schemas de cotizacion utiles para el mid del NBBO: {', '.join(quote_opts) or 'NINGUNO'}")

    # ------------------------- 3. cuantos contratos vivos por subyacente
    say("\n" + "-" * 84)
    say(f"3. TAMANO REAL DE LAS CADENAS  (dia de muestra {SAMPLE_DAY})")
    say("-" * 84)
    total_contratos = 0
    try:
        cnt = c.metadata.get_record_count(
            dataset=OPT, symbols=[f"{s}.OPT" for s in TODOS], stype_in="parent",
            schema="definition", start=SAMPLE_DAY, end=SAMPLE_DAY)
        say(f"    Registros de `definition` para los {len(TODOS)} subyacentes en 1 dia: {cnt:,}")
        total_contratos = cnt
    except Exception as ex:
        say(f"    ERROR en get_record_count(definition): {type(ex).__name__}: {ex}")

    for grupo, nombre in ((CROSS_SECTION[:3], "muestra cross-section"), (INDICES, "indices")):
        for s in grupo:
            try:
                n = c.metadata.get_record_count(
                    dataset=OPT, symbols=[f"{s}.OPT"], stype_in="parent",
                    schema="definition", start=SAMPLE_DAY, end=SAMPLE_DAY)
                say(f"        {s:<7} {n:>8,} contratos ({nombre})")
            except Exception as ex:
                say(f"        {s:<7} ERROR {ex}")

    # --------------------------------------------------------- 4. costo en USD
    say("\n" + "-" * 84)
    say("4. COSTO DEL PILOTO EN USD")
    say("-" * 84)
    say("    (get_cost es gratuito: solo estima, no descarga)")
    say("")
    say(f"    {'schema':<14}{'alcance':<34}{'costo':>14}")
    say(f"    {'-'*14}{'-'*34}{'-'*14}")
    parents = [f"{s}.OPT" for s in TODOS]
    total = 0.0

    def cost(schema, start, end, symbols, stype="parent", dataset=OPT, label=""):
        nonlocal total
        try:
            v = c.metadata.get_cost(dataset=dataset, symbols=symbols, stype_in=stype,
                                    schema=schema, start=start, end=end,
                                    mode="historical-streaming")
            say(f"    {schema:<14}{label:<34}{fmt_usd(v):>14}")
            return v
        except Exception as ex:
            say(f"    {schema:<14}{label:<34}{('ERROR ' + type(ex).__name__):>14}")
            say(f"        -> {ex}")
            return None

    v = cost("definition", START, END, parents, label=f"{len(TODOS)} subyacentes, ano completo")
    if v: total += v
    v = cost("statistics", START, END, parents, label="open interest, ano completo")
    if v: total += v

    say("")
    say("    Cotizaciones -- comparacion de estrategias:")
    for sch in ("ohlcv-1d", "bbo-1m", "cbbo-1m", "mbp-1", "tbbo"):
        if OPT in schemas and sch not in schemas[OPT]:
            continue
        cost(sch, START, END, parents, label="ano completo, sesion entera (referencia)")
    say("")
    say("    Cotizaciones -- ventana de 5 min al cierre (la estrategia del plan):")
    win_cost = None
    for sch in ("bbo-1m", "cbbo-1m", "mbp-1"):
        if OPT in schemas and sch not in schemas[OPT]:
            continue
        v = cost(sch, f"{SAMPLE_DAY}T19:55:00", f"{SAMPLE_DAY}T20:00:00", parents,
                 label="1 dia, 15:55-16:00 ET")
        if v is not None and win_cost is None:
            win_cost = v
            say(f"    {'':14}{'x ' + str(TRADING_DAYS) + ' dias habiles':<34}{fmt_usd(v * TRADING_DAYS):>14}")

    say("")
    say(f"    Equities (cierre y volumen del subyacente):")
    eq = next((d for d in EQ_CANDIDATES if d in datasets), None)
    if eq:
        cost("ohlcv-1d", START, END, TODOS, stype="raw_symbol", dataset=eq,
             label=f"{eq}, ano completo")

    say("")
    say("=" * 84)
    say("RESUMEN PARA LA DECISION GO / NO-GO")
    say("=" * 84)
    if OPT in ranges:
        say(f"  Historia de OPRA:        {ranges[OPT][0][:10]} -> {ranges[OPT][1][:10]}")
    say(f"  Contratos en 1 dia:      {total_contratos:,} para {len(TODOS)} subyacentes")
    if total_contratos:
        say(f"  Estimado ano completo:   {total_contratos * TRADING_DAYS:,} filas contrato-dia")
    say(f"  Costo definition+stats:  {fmt_usd(total)}")
    if win_cost is not None:
        say(f"  Costo cotizaciones EOD:  {fmt_usd(win_cost * TRADING_DAYS)}  (ventana de cierre x {TRADING_DAYS} dias)")
        say(f"  COSTO TOTAL ESTIMADO:    {fmt_usd(total + win_cost * TRADING_DAYS)}")
    say("")
    say(f"  Salida guardada en: {OUT}")
    say("=" * 84)

    OUT.write_text("\n".join(_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
