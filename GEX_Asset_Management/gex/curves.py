"""Curva de tasa libre de riesgo del Tesoro US, desde FRED.

POR QUE NO SE EXTRAE DE LAS OPCIONES
El pipeline sacaba r de la pendiente de la paridad put-call. Medido sobre
2021-2026, eso daba 3-5x por debajo de la tasa real (2023: 1.02% cuando el
T-bill rendia 4.5-5.5%). La causa: con T_MIN_FIT = 0.05 entran muchos
vencimientos cortos, y a T = 0.05 con r = 5% el factor de descuento vale
0.9975 -- distinguir r = 5% de r = 1% exige resolver la pendiente a 0.002,
por debajo del ruido de las cotizaciones. Como los semanales dominan en
numero, la mediana se arrastraba hacia DF ~ 1, es decir r ~ 0.

Consecuencia en cadena: r baja -> DF alto -> D = S - K*DF - (c-p) negativo ->
el piso max(D, 0) lo topaba -> dividendo cero en 28 de 32 tickers.

r es observable con precision y con fecha. Se lee, no se estima. El dividendo
y el costo de prestamo si se siguen extrayendo de la paridad, que es donde
viven: esa separacion es el punto.

Las series son rendimientos anualizados en base de inversion (bond-equivalent);
el arbol necesita capitalizacion continua, asi que se convierte con ln(1+y).
"""
from __future__ import annotations

import io
import pathlib
import urllib.request
from datetime import date

import numpy as np
import polars as pl

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
# serie -> plazo en anos
TENORS = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5,
          "DGS1": 1.0, "DGS2": 2.0}
CACHE = "data/raw/external/treasury_curve.parquet"


def fetch_treasury_curve(root: pathlib.Path, refresh: bool = False,
                         timeout: int = 60) -> pl.DataFrame:
    """Curva diaria del Tesoro, cacheada en disco. Columnas: date + un plazo por serie."""
    cache = root / CACHE
    if cache.exists() and not refresh:
        return pl.read_parquet(cache)

    url = FRED_CSV + ",".join(TENORS)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        crudo = r.read().decode("utf-8")

    df = pl.read_csv(io.StringIO(crudo), null_values=[".", ""],
                     try_parse_dates=True)
    fecha = df.columns[0]
    df = (df.rename({fecha: "date"})
            .with_columns([pl.col(c).cast(pl.Float64) for c in TENORS])
            .drop_nulls(subset=["date"])
            .sort("date"))
    # FRED deja feriados en blanco: se arrastra el ultimo dato conocido
    df = df.with_columns([pl.col(c).forward_fill() for c in TENORS])
    df = df.filter(pl.col("date") >= pl.lit(date(2000, 1, 1)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def _nodes(curve: pl.DataFrame):
    """(fechas ordenadas, matriz de tasas continuas [n_fechas x n_plazos], plazos)."""
    cols = [c for c in TENORS if c in curve.columns]
    T_nodes = np.array([TENORS[c] for c in cols], dtype=float)
    orden = np.argsort(T_nodes)
    T_nodes = T_nodes[orden]
    y = curve.select([cols[i] for i in orden]).to_numpy().astype(float) / 100.0
    # bond-equivalent -> capitalizacion continua
    r_cc = np.log1p(np.clip(y, -0.5, 1.0))
    fechas = curve["date"].to_numpy()
    return fechas, r_cc, T_nodes


def rate_lookup(curve: pl.DataFrame, fechas, plazos) -> np.ndarray:
    """r(fecha, T) continua. Interpola lineal en T; plano fuera del rango de plazos.

    En fechas sin dato (fin de semana, feriado) toma la ultima observacion
    anterior, que es la informacion realmente disponible ese dia.
    """
    nodo_f, r_cc, T_nodes = _nodes(curve)
    f = np.asarray(fechas, dtype="datetime64[D]")
    nodo_f = np.asarray(nodo_f, dtype="datetime64[D]")
    T = np.asarray(plazos, dtype=float)

    idx = np.searchsorted(nodo_f, f, side="right") - 1
    idx = np.clip(idx, 0, len(nodo_f) - 1)

    out = np.empty(len(f), dtype=float)
    for i in np.unique(idx):
        m = idx == i
        out[m] = np.interp(T[m], T_nodes, r_cc[i])
    return out


def resumen(curve: pl.DataFrame) -> pl.DataFrame:
    """Tasa a 3 meses por año -- para verificar contra la historia conocida."""
    return (curve.with_columns(pl.col("date").dt.year().alias("anio"))
                 .group_by("anio")
                 .agg([pl.col("DGS3MO").median().alias("r3m_med"),
                       pl.col("DGS3MO").min().alias("r3m_min"),
                       pl.col("DGS3MO").max().alias("r3m_max")])
                 .sort("anio"))
