"""Interfaz comun de proveedores de cadenas de opciones.

POR QUE UN ABC Y NO UNA FUNCION POR PROVEEDOR
----------------------------------------------
Porque el proyecto va a tocar al menos cuatro fuentes con formatos incompatibles
(CBOE JSON, exports de Bloomberg en XLSX, algun vendedor de pago, sinteticas) y
la unica forma de que el motor no se entere es que todas emitan EL MISMO
DataFrame. El esquema canonico (PROJECT_PLAN 3.1) es el contrato; este modulo lo
hace cumplir.

La consecuencia practica: cambiar de proveedor no debe tocar ni una linea de
`options/` ni de `research/`. Si hay que tocarla, la abstraccion esta mal.

LO QUE UN PROVEEDOR DEBE DECLARAR SOBRE SI MISMO
------------------------------------------------
`supports_history` no es un detalle: separa las fuentes con las que se puede
hacer investigacion de las que solo sirven para archivar hacia delante. Un
proveedor que devuelve solo el presente NO puede alimentar un backtest, y el
codigo debe poder preguntarselo en vez de que lo sepa el que lo escribio.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "timestamp", "symbol", "contract_symbol", "root",
    "underlying_price", "expiration", "tau",
    "strike", "option_type", "open_interest", "implied_volatility",
    "gamma", "gamma_vendor", "delta", "vega", "theta",
    "bid", "ask", "mid", "volume", "multiplier",
    "risk_free_rate", "dividend_yield", "source", "is_synthetic",
]

# `root` no es decorativo: SPX y SPXW comparten vencimiento, strike y tipo pero
# son clases distintas (AM vs PM, mensual vs semanal). Sin la raiz, cualquier
# deduplicacion las fusiona y se pierde el open interest de una de las dos.

# Sin estos no hay GEX que calcular.
REQUIRED_COLUMNS: set[str] = {
    "timestamp", "symbol", "underlying_price", "expiration",
    "strike", "option_type", "open_interest",
}


class ProviderError(RuntimeError):
    """Fallo al obtener datos. Se registra: un hueco silencioso es peor."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Lo que un proveedor puede y no puede hacer. Se consulta, no se supone."""

    name: str
    supports_history: bool
    supports_intraday: bool
    provides_greeks: bool
    provides_open_interest: bool
    is_delayed: bool
    requires_credentials: bool
    notes: str = ""


class OptionChainProvider(ABC):
    """Un proveedor de cadenas. Todos devuelven el esquema canonico."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def fetch_chain(self, symbol: str, *, as_of: date | None = None) -> pd.DataFrame:
        """Cadena completa. `as_of=None` es el presente.

        Debe lanzar `ProviderError` si no puede servir la fecha pedida. Devolver
        el presente cuando se pidio una fecha pasada seria el peor fallo posible:
        un backtest que cree mirar 2024 mirando 2026.
        """

    def fetch_and_normalize(self, symbol: str, *, as_of: date | None = None) -> pd.DataFrame:
        """`fetch_chain` + validacion del contrato de esquema."""
        df = self.fetch_chain(symbol, as_of=as_of)
        return ensure_canonical(df, source=self.capabilities.name)


def ensure_canonical(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Comprueba y completa el esquema canonico. Falla ruidosamente si no cuadra.

    No rellena campos obligatorios que falten: si no viene `open_interest`, no
    hay GEX posible y hay que enterarse aqui, no tres capas mas arriba con un
    total que sale cero.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ProviderError(
            f"[{source}] la cadena no cumple el esquema canonico; faltan: "
            f"{sorted(missing)}"
        )

    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["source"] = out["source"].fillna(source)
    out["is_synthetic"] = out["is_synthetic"].fillna(False).astype(bool)

    out["option_type"] = (
        out["option_type"].astype(str).str.strip().str.upper().str[0]
    )
    bad_types = set(out["option_type"].unique()) - {"C", "P"}
    if bad_types:
        raise ProviderError(f"[{source}] tipos de opcion no validos: {sorted(bad_types)}")

    for col in ("underlying_price", "strike", "open_interest", "implied_volatility",
                "gamma", "delta", "vega", "theta", "bid", "ask", "mid", "volume",
                "tau", "risk_free_rate", "dividend_yield"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.date

    if "multiplier" in out.columns:
        out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce").fillna(100).astype(int)

    return out[CANONICAL_COLUMNS]


# --------------------------------------------------------------------------- #
# Simbolos OCC
# --------------------------------------------------------------------------- #

_OCC_RE = re.compile(r"^(?P<root>[A-Z0-9\.\-]{1,6}?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiration: date
    option_type: str
    strike: float


def parse_occ_symbol(symbol: str) -> OccSymbol:
    """Descompone un simbolo OCC: 'SPY260831C00420000'.

    Formato: raiz (1-6) + YYMMDD + C/P + strike en milesimas con 8 digitos.

    Se parsea desde el FINAL porque la raiz es de longitud variable ('SPY' son
    tres, 'SPXW' cuatro) y anclar por delante rompe con la primera raiz larga.

    Un simbolo que no case lanza excepcion. Aqui no se adivina: un strike mal
    leido no da error, da un GEX plausible en el sitio equivocado, que es
    exactamente el fallo que nadie detecta.
    """
    s = str(symbol).strip().upper().replace(" ", "")
    m = _OCC_RE.match(s)
    if not m:
        raise ValueError(f"simbolo OCC no reconocido: {symbol!r}")

    ymd = m.group("ymd")
    try:
        exp = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"fecha imposible en el simbolo {symbol!r}: {ymd}") from exc

    return OccSymbol(
        root=m.group("root"),
        expiration=exp,
        option_type=m.group("cp"),
        strike=int(m.group("strike")) / 1000.0,
    )


def year_fraction(
    observation: datetime | pd.Timestamp,
    expiration: date,
    *,
    close_hour: int = 16,
    tz: str = "America/New_York",
) -> float:
    """Tiempo a vencimiento en años, contando HORAS y no solo dias.

    POR QUE HORAS
    -------------
    Porque el 0DTE es una categoria de investigacion del proyecto y con
    resolucion diaria `tau` valdria 0 el dia del vencimiento, lo que anula la
    gamma de TODOS los contratos que vencen ese dia (ver `greeks`: T=0 es una
    opcion vencida, gamma cero). Justo los que mas importan.

    Se cuenta hasta el cierre del dia de vencimiento (16:00 hora de Nueva York).
    Los mensuales de SPX liquidan en AM y esto los sobreestima; la capa que
    conozca ese detalle debe corregirlo (ver `universe.SPX.monthly_am_settlement`).
    """
    obs = pd.Timestamp(observation)
    obs = obs.tz_localize("UTC") if obs.tzinfo is None else obs
    obs = obs.tz_convert(tz)

    expiry_close = pd.Timestamp(
        year=expiration.year, month=expiration.month, day=expiration.day,
        hour=close_hour, tz=tz,
    )
    seconds = (expiry_close - obs).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24.0 * 3600.0)
