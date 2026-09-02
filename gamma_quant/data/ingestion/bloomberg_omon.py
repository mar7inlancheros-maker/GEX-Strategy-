"""Lector de exports XLSX de la pantalla OMON de Bloomberg.

QUE ES OMON Y QUE SE PUEDE ESPERAR DE EL
-----------------------------------------
OMON es el monitor de opciones del Terminal. Exporta a Excel lo que se ve EN
PANTALLA: unos pocos strikes alrededor del dinero y unos pocos vencimientos, no
la cadena entera. Los tres ficheros recibidos traen 75-77 contratos cada uno,
frente a los ~28.000 de una cadena completa de SPX.

Consecuencia, y hay que decirla clara: **esto no es un panel de investigacion**.
Sirve para tres cosas, todas valiosas, ninguna es "hacer el backtest":

  1. CONTRASTAR NUESTRO MOTOR contra la referencia institucional. Bloomberg
     calcula sus propias griegas; si las nuestras coinciden sobre datos reales de
     SPX, el motor esta bien calibrado.
  2. CALIBRAR r y q. La cabecera de cada grupo trae el tipo y el dividendo
     implicito que usa Bloomberg POR VENCIMIENTO. Deja de hacer falta suponerlos.
  3. FECHAS SUELTAS de estudio de caso.

EL FORMATO, Y SUS TRAMPAS
--------------------------
    fila 1  : "Calls" (y el bloque de Puts a la derecha, sin etiqueta propia)
    fila 2  : cabeceras, REPETIDAS para calls y puts:
              Ticker | Strike | Bid | Ask | Last | Volm | OInt | IVM | DL | GL | VL | TL
    fila 3+ : alternan filas de GRUPO y filas de CONTRATO

    grupo   : "18-Sep-26 (17d); CSize 100; ; IDiv .71; R 4.11; FF 0"
              vencimiento, dias, tamaño de contrato, dividendo implicito, tipo.

TRAMPA 1 — EL TICKER VIENE TRUNCADO. "SPX 9/18/26 C763" es en realidad el strike
7635: Excel corta el texto. Por eso el strike SE LEE DE LA COLUMNA `Strike` y
jamas del ticker. Del ticker solo se saca la RAIZ (SPX vs SPXW), que si es fiable
porque va al principio.

TRAMPA 2 — LA IV VIENE EN PORCENTAJE. `IVM 12.03` es 0,1203. Meterla sin dividir
da una gamma ~100 veces menor y un GEX perfectamente plausible.

TRAMPA 3 — LA GAMMA DE BLOOMBERG NO ES LA GAMMA DE BLACK-SCHOLES. Ver
`calibrate_against_bloomberg`: Bloomberg reporta `GL` como variacion de delta por
un movimiento del 1% del subyacente, no por un dolar. El factor entre ambas es
S/100, que para SPX son ~76. Confundirlas no da error: da un GEX 76 veces mayor.

LA FECHA DEL EXPORT SE DEDUCE DEL FICHERO
------------------------------------------
Un fichero llamado "as of 29may" podria estar mal nombrado. Pero cada grupo dice
"(17d)" junto a su vencimiento, asi que la fecha del export es
`vencimiento - dias`, y se puede comprobar que TODOS los grupos coincidan en la
misma fecha. Si no coinciden, el fichero mezcla momentos y hay que rechazarlo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ProviderError, year_fraction

# "18-Sep-26 (17d); CSize 100; ; IDiv .71; R 4.11; FF 0"
_GROUP_RE = re.compile(
    r"^(?P<exp>\d{1,2}-[A-Za-z]{3}-\d{2})\s*\((?P<days>-?\d+)d\)"
    r"(?:.*?CSize\s+(?P<csize>[\d.]+))?"
    r"(?:.*?IDiv\s+(?P<idiv>[-\d.]+))?"
    r"(?:.*?R\s+(?P<rate>[-\d.]+))?",
    re.IGNORECASE,
)
_TICKER_ROOT_RE = re.compile(r"^\s*(?P<root>[A-Z]{1,6}[A-Z]?)\s")

_CALL_COLUMNS = ["Ticker", "Strike", "Bid", "Ask", "Last", "Volm", "OInt",
                 "IVM", "DL", "GL", "VL", "TL"]


@dataclass
class OmonExport:
    """Lo leido de un fichero OMON, con su contexto."""

    chain: pd.DataFrame
    as_of: date
    source_file: str
    n_contracts: int = 0
    expirations: list[date] = field(default_factory=list)
    rates: dict[date, float] = field(default_factory=dict)
    implied_div: dict[date, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"OMON: {Path(self.source_file).name}",
            f"  fecha del export (deducida): {self.as_of}",
            f"  contratos                  : {self.n_contracts}",
            f"  vencimientos               : {len(self.expirations)}",
            f"  strikes distintos          : {self.chain['strike'].nunique()}",
            f"  raices                     : {sorted(self.chain['root'].unique())}",
            f"  OI total                   : {self.chain['open_interest'].sum():,.0f}",
        ]
        if self.rates:
            r = list(self.rates.values())
            lines.append(f"  tipo de Bloomberg          : {min(r):.2f}% a {max(r):.2f}%")
        lines.extend(f"  AVISO: {w}" for w in self.warnings)
        return "\n".join(lines)


def read_omon_xlsx(path: str | Path, *, symbol: str = "SPX") -> OmonExport:
    """Lee un export de OMON y devuelve la cadena en esquema canonico parcial."""
    import openpyxl

    path = Path(path)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 3:
        raise ProviderError(f"[omon] {path.name}: fichero demasiado corto")

    header = [str(c).strip() if c is not None else "" for c in rows[1]]
    blocks = _find_blocks(header)
    if not blocks:
        raise ProviderError(
            f"[omon] {path.name}: no encuentro bloques de cabecera. "
            f"Fila 2 leida: {header[:14]}"
        )

    warnings: list[str] = []
    if len(blocks) == 1:
        warnings.append(
            "solo hay un bloque de columnas: el export parece contener unicamente "
            "calls. El GEX de una cadena sin puts no es comparable con el de una "
            "cadena completa."
        )

    records: list[dict] = []
    rates: dict[date, float] = {}
    idivs: dict[date, float] = {}
    as_of_votes: list[date] = []
    current_exp: date | None = None
    current_csize = 100

    for raw in rows[2:]:
        first = str(raw[0]).strip() if raw[0] is not None else ""
        if not first:
            continue

        m = _GROUP_RE.match(first)
        if m:
            current_exp = _parse_exp(m.group("exp"))
            days = int(m.group("days"))
            as_of_votes.append(current_exp - timedelta(days=days))
            if m.group("csize"):
                current_csize = int(float(m.group("csize")))
            if m.group("rate"):
                rates[current_exp] = float(m.group("rate"))
            if m.group("idiv"):
                idivs[current_exp] = float(m.group("idiv"))
            continue

        if current_exp is None:
            continue

        root_m = _TICKER_ROOT_RE.match(first)
        if not root_m:
            continue
        root = root_m.group("root").upper()

        for block_start, opt_type in blocks:
            rec = _read_contract(raw, block_start, header, opt_type, root,
                                 current_exp, current_csize)
            if rec is not None:
                records.append(rec)

    if not records:
        raise ProviderError(f"[omon] {path.name}: ningun contrato legible")

    as_of = _resolve_as_of(as_of_votes, warnings)
    df = pd.DataFrame(records)

    df["symbol"] = symbol
    df["timestamp"] = pd.Timestamp(as_of, tz="UTC")
    df["source"] = f"bloomberg_omon:{path.name}"
    df["is_synthetic"] = False
    df["risk_free_rate"] = df["expiration"].map(rates).astype(float) / 100.0
    df["tau"] = [
        year_fraction(pd.Timestamp(as_of, tz="UTC"), e) for e in df["expiration"]
    ]
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    dup = df.duplicated(subset=["root", "expiration", "strike", "option_type"], keep=False)
    if dup.any():
        warnings.append(
            f"{int(dup.sum())} filas duplicadas por (raiz, vencimiento, strike, "
            f"tipo): el export solapa rangos de strikes."
        )

    return OmonExport(
        chain=df,
        as_of=as_of,
        source_file=str(path),
        n_contracts=len(df),
        expirations=sorted(df["expiration"].unique()),
        rates=rates,
        implied_div=idivs,
        warnings=warnings,
    )


def _find_blocks(header: list[str]) -> list[tuple[int, str]]:
    """Localiza los bloques repetidos. El primero es Calls, el segundo Puts."""
    starts = [i for i, h in enumerate(header) if h.lower() == "ticker"]
    types = ["C", "P"]
    return [(s, types[i]) for i, s in enumerate(starts) if i < 2]


def _read_contract(
    raw: tuple,
    start: int,
    header: list[str],
    opt_type: str,
    root: str,
    expiration: date,
    csize: int,
) -> dict | None:
    """Lee un contrato de un bloque. El strike SIEMPRE de su columna, nunca del ticker."""
    def col(name: str):
        for offset, h in enumerate(header[start:start + len(_CALL_COLUMNS) + 2]):
            if h.lower() == name.lower():
                idx = start + offset
                return raw[idx] if idx < len(raw) else None
        return None

    strike = _num(col("Strike"))
    if strike is None or strike <= 0:
        return None

    iv = _num(col("IVM"))
    return {
        "root": root,
        "contract_symbol": str(raw[start]).strip() if raw[start] else None,
        "expiration": expiration,
        "strike": strike,
        "option_type": opt_type,
        "bid": _num(col("Bid")),
        "ask": _num(col("Ask")),
        "last": _num(col("Last")),
        "volume": _num(col("Volm")),
        "open_interest": _num(col("OInt")),
        # IVM viene en PORCENTAJE. 12.03 -> 0.1203.
        "implied_volatility": (iv / 100.0) if iv is not None else np.nan,
        "delta": _num(col("DL")),
        "gamma_vendor_pct": _num(col("GL")),   # OJO: por 1% de movimiento, no por $1
        "vega": _num(col("VL")),
        "theta": _num(col("TL")),
        "multiplier": csize,
    }


def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _parse_exp(text: str) -> date:
    return datetime.strptime(text, "%d-%b-%y").date()


def _resolve_as_of(votes: list[date], warnings: list[str]) -> date:
    """La fecha del export es `vencimiento - dias`, y todos los grupos deben coincidir.

    Se comprueba en vez de creerse el nombre del fichero: un XLSX llamado
    'as of 29may' pudo guardarse cualquier otro dia.
    """
    if not votes:
        raise ProviderError("[omon] no hay ningun grupo con '(Nd)' del que deducir la fecha")
    counts = pd.Series(votes).value_counts()
    winner = counts.index[0]
    if len(counts) > 1:
        warnings.append(
            f"los grupos no coinciden en la fecha del export: {dict(counts)}. "
            f"Se toma la mayoritaria ({winner}). Una diferencia de 1 dia suele ser "
            f"el redondeo de Bloomberg; mas que eso significa que el fichero mezcla "
            f"momentos distintos y no debe usarse."
        )
    return winner


def implied_spot_and_dividend(chain: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Recupera spot y rendimiento por dividendo de la propia cadena.

    El export de OMON NO trae el precio del subyacente. Se puede recuperar sin
    pedirselo a nadie, usando dos identidades:

      1. PARIDAD PUT-CALL da el forward de cada vencimiento:
             C - P = (F - K) e^{-rT}    =>    F = K + (C - P) e^{rT}
      2. El forward crece con la deriva:
             ln F = ln S + (r - q) T

    Una regresion de ln F contra T sobre varios vencimientos da `ln S` en la
    ordenada y `(r - q)` en la pendiente. Con el `r` que publica Bloomberg en la
    cabecera de cada grupo, `q` queda despejado.

    Esto convierte dos supuestos (A6 tipo, y el dividendo) en dos medidas.

    Devuelve (spot, q, tabla_de_forwards).
    """
    need = {"root", "expiration", "strike", "tau", "risk_free_rate", "option_type", "mid"}
    missing = need - set(chain.columns)
    if missing:
        raise ProviderError(f"[omon] faltan columnas para la paridad: {sorted(missing)}")

    piv = (
        chain.pivot_table(
            index=["root", "expiration", "strike", "tau", "risk_free_rate"],
            columns="option_type", values="mid",
        )
        .dropna()
        .reset_index()
    )
    if piv.empty or "C" not in piv.columns or "P" not in piv.columns:
        raise ProviderError(
            "[omon] no hay ningun strike con call Y put: sin pares no hay paridad "
            "put-call de la que sacar el forward."
        )

    piv["forward"] = piv["strike"] + (piv["C"] - piv["P"]) * np.exp(
        piv["risk_free_rate"] * piv["tau"]
    )
    per_expiry = piv.groupby(["expiration", "tau"])["forward"].median().reset_index()

    if len(per_expiry) < 2:
        raise ProviderError(
            "[omon] hace falta mas de un vencimiento para separar spot de deriva"
        )

    slope, intercept = np.polyfit(per_expiry["tau"], np.log(per_expiry["forward"]), 1)
    spot = float(np.exp(intercept))
    r_median = float(piv["risk_free_rate"].median())
    q = r_median - float(slope)
    return spot, q, per_expiry


def calibrate_against_bloomberg(
    chain: pd.DataFrame, *, spot: float, dividend_yield: float
) -> pd.DataFrame:
    """Contrasta NUESTRA gamma BSM contra la `GL` de Bloomberg.

    LO QUE ESTE CONTRASTE ESTABLECIO (2026-09-01, SPX, 154 contratos)
    -----------------------------------------------------------------
    - `GL / gamma_BS` tiene mediana 75,8, y `S * 0,01` vale 76,4. Es decir,
      **Bloomberg reporta la gamma por movimiento del 1%, no por dolar**. No es
      una sutileza: usar GL directamente en el GEX lo multiplica por ~76 y el
      resultado sigue teniendo un aspecto razonable.
    - Una vez convertida, nuestra gamma coincide con la suya con un error
      MEDIANO DEL 1,25%. El motor esta bien calibrado contra la referencia
      institucional.
    """
    from ...options.greeks import bs_gamma

    d = chain.copy()
    d = d[
        d["implied_volatility"].notna()
        & d["gamma_vendor_pct"].notna()
        & (d["gamma_vendor_pct"] > 0)
    ]
    d["gamma_bs"] = bs_gamma(
        spot, d["strike"].to_numpy(float), d["tau"].to_numpy(float),
        d["risk_free_rate"].to_numpy(float),
        d["implied_volatility"].to_numpy(float), dividend_yield,
    )
    d["gamma_bbg_converted"] = bloomberg_gamma_to_bs(d["gamma_vendor_pct"], spot)
    d["rel_error"] = (d["gamma_bbg_converted"] - d["gamma_bs"]).abs() / d["gamma_bs"]
    d["raw_ratio"] = d["gamma_vendor_pct"] / d["gamma_bs"]
    return d


def bloomberg_gamma_to_bs(gamma_pct: pd.Series | np.ndarray, spot: float) -> np.ndarray:
    """Convierte la gamma de Bloomberg (por 1% de movimiento) a gamma BSM (por $1).

        GL = gamma_BS * S * 0.01     =>     gamma_BS = GL / (S * 0.01)

    Es la conversion que separa un GEX correcto de uno multiplicado por ~76 en
    SPX. Ver `calibrate_against_bloomberg` para la verificacion empirica.
    """
    return np.asarray(gamma_pct, dtype=float) / (spot * 0.01)
