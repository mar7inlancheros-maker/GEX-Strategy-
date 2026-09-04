"""Carga del historico de equities repartido en un parquet por scope.

Cada corrida de `run_ingesta.py --scope X` deja `daily_X.parquet`, y el backfill
de yfinance (2021-08 -> 2024-07, donde Databento no tiene volumen consolidado)
dejo `daily_extension{2,3,4}.parquet`. Los scripts necesitan la union de todos,
no uno solo: leer `daily_pilot.parquet` a secas limita la muestra a un ano.

Los rangos se solapan a proposito -- cada scope baja 45 dias calendario previos
para que el ADV$ de 21 dias ya este disponible en su primer dia -- asi que la
union deduplica por (ts_event, symbol).
"""
from __future__ import annotations

import glob
import pathlib

import polars as pl


def load_equities(root: pathlib.Path, pattern: str = "data/raw/equities/daily_*.parquet",
                  verbose: bool = False) -> pl.DataFrame:
    """Une todos los parquet de equities disponibles, deduplicado y ordenado."""
    files = sorted(glob.glob(str(root / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No hay parquet de equities en {root / pattern}. "
            "Corre run_ingesta.py (o el backfill de yfinance) primero.")
    eq = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    eq = eq.unique(subset=["ts_event", "symbol"]).sort(["symbol", "ts_event"])
    if verbose:
        d = eq["ts_event"].dt.date()
        print(f"equities: {eq.height:,} filas · {len(files)} archivos · "
              f"{d.min()} -> {d.max()} · {eq['symbol'].n_unique()} tickers")
    return eq
