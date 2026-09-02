"""Controles de calidad de una cadena de opciones.

EL PRINCIPIO: CUARENTENA, NUNCA DESCARTE SILENCIOSO
----------------------------------------------------
Cada control marca filas; ninguno las borra por su cuenta. La funcion devuelve la
cadena LIMPIA, la cadena EN CUARENTENA y un informe que dice cuantas filas cayo
cada control y por que. Quien llama decide.

La razon es que en datos de opciones los "errores" no son ruido: son
informacion. Un mercado cruzado (bid > ask) no es un error de tipeo, es una
cotizacion stale de un contrato que no negocia. Una IV de 4,21 en una call muy
ITM no es un fallo del proveedor, es que el mid de un contrato con spread del 5%
no determina una IV. Si se borran en silencio, se borra tambien la señal de que
esa parte de la cadena no es fiable.

LO QUE APRENDIMOS EL 2026-09-01
--------------------------------
Alpha Vantage devolvio HTTP 200, con `data` no vacio y registros bien formados...
y contratos "XXYYZZ999999C00020000" con vencimiento "2099-99-99". Una validacion
que solo mire "¿hay filas?" se lo traga. Por eso el primer control de todos es
`_check_placeholder`: antes de mirar precios, mirar si estos datos son datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    """Que encontro cada control. Va a `reports/` y se guarda con el snapshot."""

    symbol: str
    n_input: int
    n_clean: int = 0
    n_quarantined: int = 0
    checks: dict[str, int] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def is_usable(self) -> bool:
        """Sin errores fatales y con al menos la mitad de la cadena limpia.

        El 50% es un umbral defendible, no magico: por debajo, lo que queda ya no
        representa el libro de opciones y agregarlo produce un GEX de una cadena
        que no existe.
        """
        return not self.fatal and self.n_input > 0 and self.n_clean >= 0.5 * self.n_input

    def report(self) -> str:
        lines = [
            f"CALIDAD DE DATOS — {self.symbol}  ({self.timestamp_utc})",
            f"  filas de entrada : {self.n_input:,}",
            f"  limpias          : {self.n_clean:,}",
            f"  en cuarentena    : {self.n_quarantined:,}",
        ]
        if self.checks:
            lines.append("  marcadas por control (una fila puede caer en varios):")
            for name, n in sorted(self.checks.items(), key=lambda kv: -kv[1]):
                if n:
                    pct = n / self.n_input if self.n_input else 0.0
                    lines.append(f"     {name:<28} {n:>7,}  ({pct:.1%})")
        if self.stats:
            lines.append("  estadisticos:")
            for k, v in sorted(self.stats.items()):
                lines.append(f"     {k:<28} {v:,.4f}")
        lines.extend(f"  FATAL: {f}" for f in self.fatal)
        lines.extend(f"  aviso: {w}" for w in self.warnings)
        lines.append(f"  VEREDICTO: {'utilizable' if self.is_usable else 'NO UTILIZABLE'}")
        return "\n".join(lines)


def validate_chain(
    chain: pd.DataFrame,
    *,
    symbol: str = "?",
    as_of: date | None = None,
    max_spread_pct: float = 0.50,
    stale_days: int = 5,
    require_open_interest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """Aplica todas las puertas. Devuelve (limpia, cuarentena, informe)."""
    rep = QualityReport(symbol=symbol, n_input=len(chain))
    if chain.empty:
        rep.fatal.append("cadena vacia")
        return chain, chain, rep

    df = chain.copy()
    flags = pd.DataFrame(index=df.index)

    # -- 0. ¿Son datos, o son relleno de documentacion? ---------------------- #
    flags["placeholder"] = _check_placeholder(df)
    if flags["placeholder"].all():
        rep.fatal.append(
            "TODA la cadena parece relleno artificial (fechas imposibles o "
            "tickers de muestra). Esto NO es un problema de calidad: es que el "
            "proveedor no ha servido datos reales."
        )

    # -- 1. Campos obligatorios --------------------------------------------- #
    flags["falta_strike"] = ~np.isfinite(pd.to_numeric(df["strike"], errors="coerce"))
    flags["falta_expiracion"] = pd.isna(df["expiration"])
    if require_open_interest:
        oi = pd.to_numeric(df["open_interest"], errors="coerce")
        flags["falta_oi"] = ~np.isfinite(oi)
        flags["oi_negativo"] = oi < 0
    else:
        rep.warnings.append(
            "validado SIN exigir open interest: la cadena resultante NO sirve "
            "para calcular GEX (el OI es el campo critico)."
        )

    # -- 2. Duplicados ------------------------------------------------------- #
    # LA RAIZ FORMA PARTE DE LA IDENTIDAD DEL CONTRATO.
    # SPX (mensual, liquidacion AM) y SPXW (semanal, liquidacion PM) comparten
    # vencimiento, strike y tipo, y son CONTRATOS DISTINTOS con open interest
    # distinto. La primera version deduplicaba sin la raiz y tiraba 3.096
    # contratos de SPXW de una cadena de SPX -- justo donde viven los 0DTE, que
    # son una categoria central de este proyecto.
    key = ["expiration", "strike", "option_type"]
    if "root" in df.columns and df["root"].notna().any():
        key = ["root"] + key
    elif "contract_symbol" in df.columns:
        key = ["contract_symbol"]
    else:
        rep.warnings.append(
            "sin 'root' ni 'contract_symbol': la deteccion de duplicados no puede "
            "distinguir clases (SPX vs SPXW) y podria fusionar contratos distintos."
        )

    if all(k in df.columns for k in key):
        dup = df.duplicated(subset=key, keep="first")
        flags["duplicado"] = dup
        if dup.any():
            rep.warnings.append(
                f"{int(dup.sum())} contratos duplicados por {tuple(key)}. Se "
                f"conserva el primero."
            )

    # -- 3. Precios ---------------------------------------------------------- #
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    if bid is not None and ask is not None:
        flags["mercado_cruzado"] = (bid > ask) & bid.notna() & ask.notna()
        flags["precio_negativo"] = (bid < 0) | (ask < 0)

        mid = (bid + ask) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            spread_pct = (ask - bid) / mid.replace(0.0, np.nan)
        flags["spread_anomalo"] = spread_pct > max_spread_pct
        rep.stats["spread_pct_mediano"] = float(spread_pct.median(skipna=True))
        rep.stats["spread_pct_p95"] = float(spread_pct.quantile(0.95))

        # Bid a cero con OI alto: el contrato existe pero nadie lo puja. Su mid no
        # informa y su IV derivada es basura. No es error, es falta de liquidez.
        oi_num = pd.to_numeric(df.get("open_interest"), errors="coerce")
        flags["sin_puja_con_oi"] = (bid == 0) & (oi_num > 100)

    # -- 4. Volatilidad implicita -------------------------------------------- #
    iv = pd.to_numeric(df.get("implied_volatility"), errors="coerce")
    if iv is not None:
        flags["iv_ausente"] = ~np.isfinite(iv)
        flags["iv_cero"] = iv <= 0
        # Una IV por encima de 500% no es una opcion cara: es un mid que no
        # determina una IV. Aparecio literalmente en la muestra de IBM (4,21).
        flags["iv_absurda"] = iv > 5.0
        finite_iv = iv[np.isfinite(iv) & (iv > 0)]
        if len(finite_iv):
            rep.stats["iv_mediana"] = float(finite_iv.median())
            rep.stats["iv_max"] = float(finite_iv.max())

    # -- 5. Fechas y coherencia temporal ------------------------------------- #
    ref = as_of or _infer_as_of(df)
    if ref is not None:
        exp = pd.to_datetime(df["expiration"], errors="coerce").dt.date
        flags["ya_vencido"] = exp.notna() & (exp < ref)
        n_expired = int(flags["ya_vencido"].sum())
        if n_expired:
            rep.warnings.append(
                f"{n_expired} contratos con vencimiento anterior a {ref}. En un "
                f"snapshot del presente no deberian existir; en un export historico "
                f"sugiere que la fecha 'as of' no es la que se cree."
            )
        # Los LEAPS de SPX llegan de verdad a 5-6 años vista (verificado
        # 2026-09-01: vencimiento mas lejano 2031-12-19, a 5,3 años). Un corte a
        # 5 años marcaba 70 contratos legitimos. Se deja en 10, que ya no
        # corresponde a ningun producto listado.
        far = exp.notna() & (exp > ref.replace(year=ref.year + 10))
        flags["vencimiento_absurdo"] = far

    # -- 6. Cotizaciones rancias --------------------------------------------- #
    if "last_trade_time" in df.columns and ref is not None:
        last = pd.to_datetime(df["last_trade_time"], errors="coerce", utc=True)
        age_days = (pd.Timestamp(ref, tz="UTC") - last).dt.days
        flags["cotizacion_rancia"] = age_days > stale_days
        if age_days.notna().any():
            rep.stats["antiguedad_mediana_dias"] = float(age_days.median(skipna=True))

    # -- 7. Consolidacion ---------------------------------------------------- #
    flags = flags.fillna(False).astype(bool)
    rep.checks = {c: int(flags[c].sum()) for c in flags.columns}

    # Solo unos pocos controles EXCLUYEN. Los demas informan.
    # Un spread ancho o una cotizacion rancia no invalidan el open interest, que
    # es lo que alimenta el GEX; excluirlos sesgaria la cadena hacia lo liquido.
    blocking = [
        "placeholder", "falta_strike", "falta_expiracion",
        "falta_oi", "oi_negativo", "duplicado",
        "mercado_cruzado", "precio_negativo", "vencimiento_absurdo", "ya_vencido",
    ]
    blocking = [c for c in blocking if c in flags.columns]
    quarantine_mask = flags[blocking].any(axis=1)

    clean = df.loc[~quarantine_mask].copy()
    quarantined = df.loc[quarantine_mask].copy()
    # `.apply(axis=1)` sobre un DataFrame VACIO devuelve un DataFrame, no una
    # Series, y asignarlo a una columna revienta. El caso vacio es justo el bueno
    # -- nada en cuarentena -- asi que no puede ser el que rompa.
    if len(quarantined):
        quarantined["_motivos"] = flags.loc[quarantine_mask, blocking].apply(
            lambda r: ",".join(r.index[r.to_numpy()]), axis=1
        )
    else:
        quarantined["_motivos"] = pd.Series(dtype="object")

    rep.n_clean = len(clean)
    rep.n_quarantined = len(quarantined)

    if rep.n_input and rep.n_clean < 0.5 * rep.n_input:
        rep.warnings.append(
            f"solo el {rep.n_clean / rep.n_input:.0%} de la cadena pasa los "
            f"controles: el agregado ya no representa el libro real."
        )
    return clean, quarantined, rep


# Valores que significan "este campo esta vacio", no "este dato es falso".
_NULLISH = {"", "none", "nan", "nat", "null", "n/a", "na", "-", "--"}


def _check_placeholder(df: pd.DataFrame) -> pd.Series:
    """Detecta datos de relleno de documentacion disfrazados de cadena real.

    CUIDADO CON LOS FALSOS POSITIVOS, QUE AQUI CUESTAN CAROS
    --------------------------------------------------------
    Este control BLOQUEA, asi que cada falso positivo es open interest real que
    desaparece del GEX. La primera version marcaba 6.690 contratos de SPX (23,6%)
    porque comprobaba tambien `last_trade_time`, que CBOE devuelve como el texto
    'None' en los contratos QUE NUNCA HAN NEGOCIADO. Esos contratos existen, y
    entre todos sumaban 7.005 de open interest: precisamente el dato que alimenta
    el GEX. Un contrato sin negociar no es un contrato falso.

    Regla: solo `expiration` debe ser siempre parseable, porque sin vencimiento
    no hay contrato. Las demas fechas pueden faltar legitimamente y su ausencia
    se trata en `cotizacion_rancia`, que informa pero no bloquea.
    """
    flag = pd.Series(False, index=df.index)

    for col in ("symbol", "contract_symbol"):
        if col in df.columns:
            text = df[col].astype(str).str.upper()
            flag |= text.str.contains("XXYYZZ", na=False)
            flag |= text.isin(["SYMBOL", "TICKER", "NAN", ""])

    # Solo el vencimiento. Un valor presente pero imposible ('2099-99-99') es
    # relleno; un valor ausente es simplemente un campo vacio.
    if "expiration" in df.columns:
        raw = df["expiration"].astype(str).str.strip().str.lower()
        parsed = pd.to_datetime(df["expiration"], errors="coerce")
        flag |= parsed.isna() & ~raw.isin(_NULLISH)

    # Fechas imposibles en cualquier otro campo SI son señal, siempre que el
    # valor no sea uno de los "vacio".
    for col in ("date", "last_trade_time"):
        if col in df.columns:
            raw = df[col].astype(str).str.strip().str.lower()
            parsed = pd.to_datetime(df[col], errors="coerce")
            suspicious = parsed.isna() & ~raw.isin(_NULLISH)
            # '2099-99-99' no parsea y no es nullish -> marca.
            flag |= suspicious

    return flag


def _infer_as_of(df: pd.DataFrame) -> date | None:
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return ts.max().date() if ts.notna().any() else None
