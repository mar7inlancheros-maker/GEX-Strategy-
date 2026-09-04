"""Ingesta de OPRA + equities desde Databento hacia el lakehouse local.

REGLAS DE ORO ESTABLECIDAS EN LA FASE 0 (no cambiar sin re-verificar):

1. OPEN INTEREST -- DEDUPLICAR, NUNCA SUMAR.
   OPRA es un feed consolidado: 18 publishers diseminan el MISMO valor de OI por
   contrato. Verificado en Fase 0.D: 18/18 publishers, 3.650/3.650 contratos con
   valor identico, factor de inflacion exactamente 18.00x. Sumar inflaria Gamma 18x.
   El pipeline ADEMAS valida cada dia que los publishers concuerden: si algun dia
   discrepan, avisa en vez de promediar en silencio.

2. VENTANA DEL OI: 10:00-12:00 UTC (rafaga a las 10:30 = 06:30 ET). Unica franja
   del dia con stat_type=9. El dia completo cuesta 4.4x mas por el mismo dato.

3. COTIZACIONES: mid del NBBO de `cbbo-1m` en 19:55-20:00 UTC (15:55-16:00 ET).
   Nunca el ultimo trade: en opciones iliquidas es rancio, y por el hallazgo H1
   (Gamma es un residuo entre calls y puts que casi se cancelan) el error de
   precio se amplifica en Gamma.

4. MULTIPLICADOR: el real de `definition`, no 100 fijo. Hay contratos ajustados
   de 10 y 1000 tras splits y spinoffs.

5. instrument_id SE RECICLA entre dias. Todo join es intra-dia.

PENDIENTE ABIERTO: `ts_ref` viene vacio (NaT), asi que el desfase de un dia del
OI sigue siendo inferencia. Se resuelve empiricamente en la puerta P2.
"""
from __future__ import annotations

import logging
import warnings
import pathlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

import polars as pl

log = logging.getLogger("gex.ingest")

OPT_DATASET = "OPRA.PILLAR"
OI_WIN = ("10:00:00", "15:00:00")
QUOTE_WIN = ("19:55:00", "20:00:00")
STAT_TYPE_OPEN_INTEREST = 9
MAX_REL_SPREAD = 0.50


@dataclass
class DayResult:
    day: date
    n_contracts: int = 0
    n_oi: int = 0
    n_quotes: int = 0
    n_joined: int = 0
    publishers_per_contract: int = 0
    publishers_disagree: int = 0
    quote_discard_rate: float = 0.0
    t_defs: float = 0.0
    t_oi: float = 0.0
    t_quotes: float = 0.0
    degraded: bool = False
    half_day: bool = False
    skipped: bool = False
    error: str | None = None
    notes: list = field(default_factory=list)


def dataset_condition(client, dataset: str, start, end) -> dict:
    """Dias con calidad reducida segun Databento. Se marcan, no se descartan."""
    try:
        rows = client.metadata.get_dataset_condition(
            dataset=dataset, start_date=str(start), end_date=str(end))
    except Exception:
        return {}
    out = {}
    for r in rows:
        d = r.get("date") if isinstance(r, dict) else getattr(r, "date", None)
        cond = r.get("condition") if isinstance(r, dict) else getattr(r, "condition", None)
        if d and cond and str(cond) != "available":
            out[str(d)] = str(cond)
    return out


def _to_df(store):
    try:
        return store.to_df(price_type="float")
    except TypeError:
        return store.to_df()


def _fix_scale(col: pl.Series) -> pl.Series:
    """Databento entrega precios fixed-point 1e-9 si to_df no los convirtio."""
    nn = col.drop_nulls()
    if col.dtype in (pl.Int64, pl.Int32) or (nn.len() and abs(nn.median()) > 1e6):
        return col.cast(pl.Float64) / 1e9
    return col.cast(pl.Float64)


def fetch_definitions(client, parents: list, day: date) -> pl.DataFrame:
    """Especificaciones de todos los contratos vivos ese dia."""
    store = get_range_retry(
        client, dataset=OPT_DATASET, symbols=parents, stype_in="parent",
        schema="definition",
        start=day.isoformat(), end=(day + timedelta(days=1)).isoformat())
    pdf = _to_df(store)
    if pdf.empty:
        return pl.DataFrame()
    df = pl.from_pandas(pdf.reset_index())
    cols = set(df.columns)

    strike_c = next((c for c in ("strike_price", "strike") if c in cols), None)
    mult_c = next((c for c in ("unit_of_measure_qty", "contract_multiplier") if c in cols), None)
    klass_c = next((c for c in ("instrument_class", "security_type") if c in cols), None)

    sel = [pl.col("instrument_id").cast(pl.Int64),
           pl.col("raw_symbol").cast(pl.Utf8)]
    if "expiration" in cols:
        sel.append(pl.col("expiration"))
    sel.append((_fix_scale(df[strike_c]) if strike_c else pl.lit(None, pl.Float64)).alias("strike"))
    sel.append((pl.col(klass_c).cast(pl.Utf8) if klass_c else pl.lit(None, pl.Utf8)).alias("klass"))
    sel.append((pl.col(mult_c).cast(pl.Float64) if mult_c else pl.lit(100.0)).alias("multiplier"))

    out = df.select(sel).unique(subset=["instrument_id"], keep="first")
    # OSI: "AAPL  260619C00235000" -> posicion 12 = C/P, posiciones 0:6 = raiz
    out = out.with_columns([
        pl.when(pl.col("klass").is_in(["C", "Call", "call"])).then(True)
          .when(pl.col("klass").is_in(["P", "Put", "put"])).then(False)
          .otherwise(pl.col("raw_symbol").str.slice(12, 1) == "C").alias("is_call"),
        pl.col("raw_symbol").str.slice(0, 6).str.strip_chars().alias("underlying"),
        pl.when((pl.col("multiplier") <= 0) | pl.col("multiplier").is_null())
          .then(100.0).otherwise(pl.col("multiplier")).alias("multiplier"),
    ])
    return out


def fetch_open_interest(client, parents: list, day: date, res: DayResult) -> pl.DataFrame:
    """Open interest DEDUPLICADO por instrument_id, con validacion de publishers."""
    d = day.isoformat()
    store = get_range_retry(
        client, dataset=OPT_DATASET, symbols=parents, stype_in="parent",
        schema="statistics", start=f"{d}T{OI_WIN[0]}", end=f"{d}T{OI_WIN[1]}")
    pdf = _to_df(store)
    if pdf.empty:
        return pl.DataFrame()
    df = pl.from_pandas(pdf.reset_index())
    df = df.filter(pl.col("stat_type").cast(pl.Int64) == STAT_TYPE_OPEN_INTEREST)
    if df.is_empty():
        res.notes.append("sin registros stat_type=9")
        return pl.DataFrame()

    qc = next((c for c in ("quantity", "price", "value") if c in df.columns), None)
    df = df.select([pl.col("instrument_id").cast(pl.Int64),
                    pl.col("publisher_id").cast(pl.Int64),
                    pl.col(qc).cast(pl.Float64).alias("open_interest")])
    res.n_oi = df.height

    per = df.group_by("instrument_id").agg(pl.col("publisher_id").n_unique().alias("n"))
    res.publishers_per_contract = int(per["n"].median())
    if per["n"].min() != per["n"].max():
        res.notes.append(f"publishers/contrato varia: {per['n'].min()}-{per['n'].max()}")

    dis = (df.group_by("instrument_id")
             .agg(pl.col("open_interest").n_unique().alias("nv"))
             .filter(pl.col("nv") > 1))
    res.publishers_disagree = dis.height
    if dis.height:
        res.notes.append(f"ATENCION: {dis.height} contratos con OI discrepante entre publishers")

    # REGLA DE ORO: deduplicar, jamas sumar
    return (df.sort(["instrument_id", "publisher_id"])
              .unique(subset=["instrument_id"], keep="first")
              .select(["instrument_id", "open_interest"]))


def get_range_retry(client, attempts: int = 4, waits=(5, 15, 40), **kw):
    """get_range con reintentos y espera creciente.

    Los 504 del gateway de Databento aparecen en los dias grandes cuando varios
    workers piden a la vez. No son un fallo del dato: reintentar funciona. Sin
    esto se perdieron 6 semanas enteras en la primera corrida, y peor: se pago
    definition y statistics de esos dias y luego cbbo fallo, dejando el dia
    inservible pero cobrado.
    """
    import time as _time
    last = None
    for i in range(attempts):
        try:
            return client.timeseries.get_range(**kw)
        except Exception as ex:
            msg = str(ex)
            transitorio = ("504" in msg or "gateway" in msg.lower()
                           or "prematurely" in msg.lower() or "timed out" in msg.lower()
                           or "503" in msg or "429" in msg)
            last = ex
            if not transitorio or i == attempts - 1:
                raise
            _time.sleep(waits[min(i, len(waits) - 1)])
    raise last


def close_window_utc(day: date, half_day: bool = False):
    """Ventana de 5 minutos antes del cierre, en UTC, respetando horario de verano.

    Sesion normal: cierre 16:00 ET. Media sesion: 13:00 ET.
    Calcular esto en ET y convertir a UTC es obligatorio: el desfase con UTC
    cambia con el horario de verano (EST = UTC-5, EDT = UTC-4), asi que una
    ventana UTC fija se desalinea media año.
    """
    hour = 13 if half_day else 16
    close_et = datetime(day.year, day.month, day.day, hour, 0, tzinfo=NY)
    close_utc = close_et.astimezone(timezone.utc)
    start_utc = close_utc - timedelta(minutes=5)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return start_utc.strftime(fmt), close_utc.strftime(fmt)


def fetch_eod_quotes(client, parents: list, day: date, res: DayResult) -> pl.DataFrame:
    """Ultimo NBBO de la ventana de cierre + filtros de calidad de cotizacion.

    MEDIAS SESIONES: el mercado cierra a las 13:00 ET el viernes despues de
    Accion de Gracias, el 24 de diciembre y la vispera del 4 de julio. En esos
    dias la ventana de las 16:00 ET cae DESPUES del cierre y devuelve cero
    cotizaciones -- el dia entero se perderia en silencio. Si la ventana normal
    viene vacia se reintenta con la de media sesion.
    """
    def pedir(start, end):
        store = get_range_retry(
            client, dataset=OPT_DATASET, symbols=parents, stype_in="parent",
            schema="cbbo-1m", start=start, end=end)
        return _to_df(store)

    s_, e_ = close_window_utc(day, half_day=False)
    pdf = pedir(s_, e_)
    if pdf.empty:
        s_, e_ = close_window_utc(day, half_day=True)
        pdf = pedir(s_, e_)
        if not pdf.empty:
            res.notes.append("MEDIA SESION: cotizaciones tomadas a las 13:00 ET")
            res.half_day = True
    if pdf.empty:
        return pl.DataFrame()
    df = pl.from_pandas(pdf.reset_index())
    bid_c = next((c for c in ("bid_px_00", "bid_px") if c in df.columns), None)
    ask_c = next((c for c in ("ask_px_00", "ask_px") if c in df.columns), None)
    if not bid_c or not ask_c:
        res.notes.append(f"sin columnas bid/ask; vi: {list(df.columns)[:12]}")
        return pl.DataFrame()

    df = df.with_columns([_fix_scale(df[bid_c]).alias("bid"),
                          _fix_scale(df[ask_c]).alias("ask")]).sort("ts_event")
    last = (df.unique(subset=["instrument_id"], keep="last")
              .select([pl.col("instrument_id").cast(pl.Int64), "bid", "ask"]))
    n_raw = last.height

    clean = (last.filter((pl.col("bid") > 0) & (pl.col("ask") > 0)
                         & (pl.col("ask") >= pl.col("bid")))
                 .with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("mid")))
    clean = clean.with_columns(
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid")).alias("rel_spread"))
    res.n_quotes = clean.height
    res.quote_discard_rate = 1.0 - (clean.height / n_raw) if n_raw else 0.0
    # NOTA: rel_spread se guarda pero NO se filtra aqui. El filtro
    # rel_spread <= MAX_REL_SPREAD se aplica en la etapa de senal.
    return clean


def ingest_day(client, parents: list, day: date, out_dir: pathlib.Path,
               overwrite: bool = False) -> DayResult:
    res = DayResult(day=day)
    dest = out_dir / f"date={day.isoformat()}" / "chain.parquet"
    if dest.exists() and not overwrite:
        res.skipped = True
        return res
    import time as _t
    try:
        _a = _t.time()
        defs = fetch_definitions(client, parents, day)
        res.t_defs = _t.time() - _a
        if defs.is_empty():
            res.notes.append("sin definitions (probable dia no habil)")
            return res
        res.n_contracts = defs.height
        _a = _t.time()
        oi = fetch_open_interest(client, parents, day, res)
        res.t_oi = _t.time() - _a
        _a = _t.time()
        qt = fetch_eod_quotes(client, parents, day, res)
        res.t_quotes = _t.time() - _a
        if oi.is_empty() or qt.is_empty():
            res.notes.append("falta OI o cotizaciones")
            return res
        chain = (defs.join(oi, on="instrument_id", how="inner")      # join intra-dia
                     .join(qt, on="instrument_id", how="inner")
                     .filter(pl.col("open_interest") > 0)
                     .with_columns(pl.lit(day).alias("date")))
        res.n_joined = chain.height
        dest.parent.mkdir(parents=True, exist_ok=True)
        chain.write_parquet(dest)
    except Exception as ex:
        res.error = f"{type(ex).__name__}: {ex}"
    return res


def fetch_equities_daily(client, tickers: list, start: date, end: date,
                         dataset: str = "EQUS.SUMMARY",
                         lookback_days: int = 45) -> pl.DataFrame:
    """Cierre y volumen del subyacente -- denominador ADV$ de la Ecuacion 1.

    lookback_days: dias calendario ANTES de `start` que tambien se bajan, para
    que el promedio movil de 21 dias habiles ya este disponible en el primer dia
    de la muestra. Sin esto el denominador de la Eq.1 es nulo al arrancar.
    """
    store = get_range_retry(
        client, dataset=dataset, symbols=tickers, stype_in="raw_symbol",
        schema="ohlcv-1d",
        start=(start - timedelta(days=lookback_days)).isoformat(),
        end=(end + timedelta(days=1)).isoformat())
    pdf = _to_df(store)
    if pdf.empty:
        return pl.DataFrame()
    df = pl.from_pandas(pdf.reset_index())
    fixes = [_fix_scale(df[c]).alias(c) for c in ("open", "high", "low", "close")
             if c in df.columns]
    df = df.with_columns(fixes) if fixes else df
    keep = [c for c in ("ts_event", "symbol", "open", "high", "low", "close", "volume")
            if c in df.columns]
    return df.select(keep)
