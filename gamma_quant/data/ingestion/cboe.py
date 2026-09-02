"""Adaptador del endpoint publico de cotizaciones diferidas de CBOE.

QUE ES ESTA FUENTE Y QUE NO ES
------------------------------
Es la unica fuente GRATUITA y REAL de cadenas completas que se ha verificado
funcionando (2026-08-31): SPY 13.514 contratos y SPX 28.648, con open interest,
IV y griegas propias de CBOE.

Y es SOLO EL PRESENTE. No hay parametro de fecha, no hay archivo, no se puede
pedir el pasado. Por eso `supports_history = False` y por eso `fetch_chain`
LANZA si se le pide un `as_of` distinto de hoy, en vez de devolver el presente
disimuladamente: un backtest que cree mirar mayo mirando hoy es el peor fallo
que puede tener este proyecto.

De ahi que el archivador sea urgente. Cada dia que no se guarda es un dia que no
existe: no hay forma de recuperarlo despues a ningun precio.

SOBRE LA GAMMA DE CBOE
----------------------
Viene en el payload, pero no sabemos con que modelo ni con que tipo libre de
riesgo la calculan, ni si tratan SPY como americana. Por eso la config trae
`gamma_source = "both"`: se guarda la suya como `gamma_vendor`, se calcula la
nuestra desde la IV, y se reporta la discrepancia. Fiarse de una griega ajena sin
contrastarla es heredar un modelo que no se ha leido.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from ...options.greeks import bs_gamma
from .base import (
    OptionChainProvider,
    ProviderCapabilities,
    ProviderError,
    parse_occ_symbol,
    year_fraction,
)

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"

# El endpoint usa una raiz distinta para indices: '_SPX', '_VIX'...
ENDPOINT_SYMBOL = {"SPX": "_SPX", "VIX": "_VIX", "NDX": "_NDX", "RUT": "_RUT"}


class CboeDelayedProvider(OptionChainProvider):
    """Cadena completa diferida de CBOE. Gratis, real, solo el presente."""

    def __init__(
        self,
        *,
        user_agent: str = "gamma_quant/0.1",
        timeout: int = 60,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0,
        multiplier: int = 100,
        compute_own_gamma: bool = True,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.multiplier = multiplier
        self.compute_own_gamma = compute_own_gamma

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="cboe_delayed",
            supports_history=False,          # <- lo mas importante de esta ficha
            supports_intraday=True,
            provides_greeks=True,
            provides_open_interest=True,
            is_delayed=True,
            requires_credentials=False,
            notes=(
                "Snapshot del presente, diferido ~15 min. Sin archivo historico: "
                "un dia no guardado se pierde para siempre."
            ),
        )

    def fetch_raw(self, symbol: str) -> dict:
        endpoint = ENDPOINT_SYMBOL.get(symbol.upper(), symbol.upper())
        url = CBOE_URL.format(sym=endpoint)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"[cboe] HTTP {exc.code} para {symbol} ({url})") from exc
        except Exception as exc:
            raise ProviderError(f"[cboe] fallo de red para {symbol}: {exc}") from exc

    def fetch_chain(self, symbol: str, *, as_of: date | None = None) -> pd.DataFrame:
        today = datetime.now(timezone.utc).date()
        if as_of is not None and as_of != today:
            raise ProviderError(
                f"[cboe] esta fuente NO tiene historico: se pidio {as_of} y solo "
                f"puede servir {today}. Devolver el presente disfrazado de pasado "
                f"produciria un backtest con datos del futuro."
            )

        payload = self.fetch_raw(symbol)
        data = payload.get("data") or {}
        options = data.get("options") or []
        if not options:
            raise ProviderError(f"[cboe] respuesta sin contratos para {symbol}")

        spot = data.get("current_price")
        if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
            raise ProviderError(f"[cboe] spot invalido para {symbol}: {spot!r}")
        spot = float(spot)

        observed = pd.Timestamp.now(tz="UTC")
        rows = []
        unparsed = 0
        for c in options:
            raw_symbol = c.get("option")
            try:
                occ = parse_occ_symbol(raw_symbol)
            except (ValueError, TypeError):
                unparsed += 1
                continue
            rows.append({
                "contract_symbol": raw_symbol,
                "root": occ.root,          # SPX vs SPXW: clases distintas
                "expiration": occ.expiration,
                "strike": occ.strike,
                "option_type": occ.option_type,
                "open_interest": c.get("open_interest"),
                "volume": c.get("volume"),
                "implied_volatility": c.get("iv"),
                "gamma_vendor": c.get("gamma"),
                "delta": c.get("delta"),
                "vega": c.get("vega"),
                "theta": c.get("theta"),
                "bid": c.get("bid"),
                "ask": c.get("ask"),
                "last_trade_price": c.get("last_trade_price"),
                "last_trade_time": c.get("last_trade_time"),
            })

        if not rows:
            raise ProviderError(f"[cboe] ningun simbolo OCC parseable para {symbol}")

        df = pd.DataFrame(rows)
        df["symbol"] = symbol.upper()
        df["underlying_price"] = spot
        df["timestamp"] = observed
        df["multiplier"] = self.multiplier
        df["risk_free_rate"] = self.risk_free_rate
        df["dividend_yield"] = self.dividend_yield
        df["source"] = "cboe_delayed"
        df["is_synthetic"] = False

        df["tau"] = [year_fraction(observed, e) for e in df["expiration"]]
        df["mid"] = (
            pd.to_numeric(df["bid"], errors="coerce")
            + pd.to_numeric(df["ask"], errors="coerce")
        ) / 2.0

        # Gamma propia desde la IV: auditable y comparable con la del proveedor.
        if self.compute_own_gamma:
            df["gamma"] = bs_gamma(
                spot,
                df["strike"].to_numpy(dtype=float),
                df["tau"].to_numpy(dtype=float),
                self.risk_free_rate,
                pd.to_numeric(df["implied_volatility"], errors="coerce").to_numpy(dtype=float),
                self.dividend_yield,
            )
        else:
            df["gamma"] = pd.to_numeric(df["gamma_vendor"], errors="coerce")

        if unparsed:
            df.attrs["unparsed_symbols"] = unparsed
        df.attrs["spot"] = spot
        df.attrs["n_raw"] = len(options)
        return df


def gamma_cross_check(df: pd.DataFrame, *, rtol: float = 0.05) -> dict[str, float]:
    """Compara nuestra gamma con la del proveedor (supuesto A7), PESANDO POR GEX.

    POR QUE PESAR, Y NO CONTAR CONTRATOS
    -------------------------------------
    La primera version contaba contratos: "el 55% difiere mas de un 5%". El
    numero asustaba y no significaba nada. Al desglosarlo, el acuerdo entre 1 dia
    y 90 dias -- donde vive el grueso de la gamma -- era de 0,997 a 1,001, y todo
    el "55%" venia de las alas, donde la gamma es ~1e-9 en ambos y una diferencia
    relativa del 26% no mueve el GEX ni un dolar.

    Lo que importa no es cuantos contratos discrepan, sino CUANTA GAMMA
    EFECTIVA discrepa. Por eso se pondera por |gamma * OI|, que es la
    contribucion de cada contrato al agregado. `share_above_tol` pasa a ser
    "que fraccion del GEX viene de contratos en desacuerdo", que si es
    accionable.

    Se desglosa ademas por plazo, porque las dos causas conocidas de discrepancia
    tienen firmas distintas:
      - convencion de `tau` en 0DTE  -> discrepancia concentrada a plazo cero
      - `q` equivocado               -> discrepancia que CRECE con el plazo
    """
    if "gamma_vendor" not in df.columns:
        return {}
    ours = pd.to_numeric(df["gamma"], errors="coerce")
    theirs = pd.to_numeric(df["gamma_vendor"], errors="coerce")
    oi = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0.0)

    mask = ours.notna() & theirs.notna() & (ours > 1e-8) & (theirs > 1e-8)
    if not mask.any():
        return {"n_compared": 0.0}

    rel = ((ours[mask] - theirs[mask]) / theirs[mask]).abs()
    weight = (theirs[mask] * oi[mask]).abs()
    total_w = float(weight.sum())

    out = {
        "n_compared": float(mask.sum()),
        "median_rel_diff": float(rel.median()),
        "p95_rel_diff": float(rel.quantile(0.95)),
        "share_contracts_above_tol": float((rel > rtol).mean()),
    }
    if total_w > 0:
        out["weighted_median_rel_diff"] = float(
            _weighted_quantile(rel.to_numpy(), weight.to_numpy(), 0.5)
        )
        # ESTE es el numero que decide si preocuparse.
        out["share_gex_above_tol"] = float(weight[rel > rtol].sum() / total_w)

    if "tau" in df.columns:
        tau = pd.to_numeric(df["tau"], errors="coerce")[mask]
        short = tau < (2.0 / 365.0)
        if short.any():
            out["median_rel_diff_0dte"] = float(rel[short].median())
        if (~short).any():
            out["median_rel_diff_rest"] = float(rel[~short].median())
    return out


def _weighted_quantile(values, weights, q: float) -> float:
    """Cuantil ponderado. numpy no lo trae y aqui es justo lo que hace falta."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float("nan")
    return float(np.interp(q * cum[-1], cum, v))
