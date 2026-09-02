"""Almacen de snapshots de cadena. Parquet particionado por (simbolo, fecha).

POR QUE PARQUET Y NO CSV
------------------------
Un snapshot de SPX son ~28.600 filas. En CSV se pierden los tipos (las fechas
vuelven como texto, el OI como float, y `expiration` como algo que hay que
reparsear cada vez) y ocupa un orden de magnitud mas. Parquet conserva el
esquema. Si falta pyarrow se cae a CSV comprimido con un aviso: mejor un dato
guardado en formato pobre que un dato perdido.

POR QUE UN FICHERO POR (SIMBOLO, DIA) Y NO UNA TABLA GRANDE
------------------------------------------------------------
Porque el archivo crece un fichero al dia y nunca se reescribe. Un unico Parquet
que hay que reescribir entero cada tarde es una oportunidad diaria de corromper
todo el historico con un fallo de disco. Con ficheros por dia, el peor caso es
perder el dia en curso.

IDEMPOTENCIA
------------
`write_snapshot` con `overwrite=False` no pisa lo ya guardado. Relanzar el
archivador tres veces el mismo dia no duplica ni corrompe: avisa y no hace nada.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ...config import archive_dir

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:  # pragma: no cover
    _HAS_PARQUET = False


def snapshot_path(symbol: str, day: date, *, root: Path | None = None) -> Path:
    """`<archivo>/<SIMBOLO>/<AAAA>/<SIMBOLO>_<AAAA-MM-DD>.parquet`.

    Particionado por año para que ningun directorio pase de ~250 ficheros: el
    explorador de Windows se arrastra con miles de entradas en una carpeta.
    """
    base = root or archive_dir()
    ext = "parquet" if _HAS_PARQUET else "csv.gz"
    d = base / symbol.upper() / f"{day.year:04d}"
    return d / f"{symbol.upper()}_{day.isoformat()}.{ext}"


def write_snapshot(
    chain: pd.DataFrame,
    symbol: str,
    day: date,
    *,
    root: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Guarda un snapshot. Devuelve (ruta, se_escribio).

    No pisa por defecto: el archivo es irreemplazable y una reescritura
    accidental con una cadena a medias destruye el unico ejemplar del dia.
    """
    path = snapshot_path(symbol, day, root=root)
    if path.exists() and not overwrite:
        return path, False

    path.parent.mkdir(parents=True, exist_ok=True)
    out = chain.copy()

    # `expiration` como date puro no sobrevive a Parquet de forma estable segun
    # version; se normaliza a datetime64 al escribir y se devuelve a date al leer.
    if "expiration" in out.columns:
        out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce")

    # Escritura atomica: primero a temporal, luego rename. Un corte de luz a
    # mitad deja un .tmp huerfano, no un Parquet truncado que parece valido.
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _HAS_PARQUET:
        out.to_parquet(tmp, index=False, compression="snappy")
    else:
        out.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)
    return path, True


def read_snapshot(symbol: str, day: date, *, root: Path | None = None) -> pd.DataFrame:
    path = snapshot_path(symbol, day, root=root)
    if not path.exists():
        raise FileNotFoundError(f"no hay snapshot de {symbol} el {day}: {path}")
    df = pd.read_parquet(path) if _HAS_PARQUET else pd.read_csv(path)
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce").dt.date
    return df


def available_days(symbol: str, *, root: Path | None = None) -> list[date]:
    """Dias archivados de un simbolo, ordenados."""
    base = (root or archive_dir()) / symbol.upper()
    if not base.is_dir():
        return []
    days: list[date] = []
    for f in base.rglob(f"{symbol.upper()}_*"):
        stem = f.name.split("_", 1)[-1].split(".")[0]
        try:
            days.append(date.fromisoformat(stem))
        except ValueError:
            continue
    return sorted(days)


def coverage_report(symbol: str, *, root: Path | None = None) -> str:
    """Cobertura del archivo, con los HUECOS explicitos.

    Los huecos son el riesgo numero uno de la via "archivar hacia delante": en
    una muestra de un año, dos semanas perdidas no son un 4% menos de datos, son
    un sesgo de seleccion (lo que suele romper el archivador es justo un dia
    raro de mercado).
    """
    days = available_days(symbol, root=root)
    if not days:
        return f"{symbol}: archivo VACIO"

    first, last = days[0], days[-1]
    expected = pd.bdate_range(first, last).date
    missing = sorted(set(expected) - set(days))

    lines = [
        f"{symbol}: {len(days)} snapshots, de {first} a {last}",
        f"  sesiones habiles esperadas : {len(expected)}",
        f"  ausentes                   : {len(missing)}",
    ]
    if missing:
        shown = ", ".join(d.isoformat() for d in missing[:10])
        lines.append(f"  HUECOS: {shown}{' ...' if len(missing) > 10 else ''}")
        lines.append(
            "  (algunos seran festivos de mercado; el resto son fallos del "
            "archivador y NO se pueden recuperar)"
        )
    return "\n".join(lines)
