"""Gamma flip: el precio S* donde el GEX agregado cruza cero.

QUE ES Y POR QUE IMPORTARIA
---------------------------
Por encima de S* el GEX agregado es positivo y la hipotesis dice que la cobertura
de los dealers AMORTIGUA el movimiento (venden en subidas, compran en bajadas).
Por debajo es negativo y lo AMPLIFICA. El flip seria, si la hipotesis es cierta,
una frontera de regimen.

Es una hipotesis. Este modulo la CALCULA; no la valida. Validarla es la Fase 8.

EL ATAJO QUE CASI TODO EL MUNDO USA, Y POR QUE ESTA MAL
-------------------------------------------------------
Lo comodo es coger el perfil `strike -> GEX` que ya tienes e interpolar donde
cruza cero. Es rapido y es incorrecto: ese perfil dice cuanto GEX hay EN cada
strike con el spot ACTUAL, no cuanto habria si el spot ESTUVIERA en ese strike.
La gamma de cada contrato depende de S -- una opcion a 5% OTM tiene poca gamma
hoy y mucha si el mercado se mueve hasta ella -- asi que mover el spot recalcula
TODA la cadena, no solo desplaza un puntero sobre una curva fija.

Aqui se hace lo correcto: para cada spot candidato se revalora la cadena entera
con `greeks.bs_gamma` y se suma. Cuesta mas, y es lo que mide lo que dice medir.

LA IV AL MOVER EL SPOT: SUPUESTO A8 (NUEVO)
--------------------------------------------
Al preguntar "¿cuanto valdria el GEX si el spot fuese S*?" hay que decidir que
pasa con la superficie de volatilidad, y no hay respuesta neutra:

    sticky_strike : cada strike conserva SU IV actual. Es lo que hace todo el
                    mundo y lo que se asume por defecto. Empiricamente es
                    razonable en horizontes cortos y movimientos pequeños.

    sticky_moneyness : la IV viaja con el moneyness (K/S), de modo que la sonrisa
                    se desplaza con el mercado. Encaja mejor con lo que se observa
                    en indices tras movimientos grandes.

    flat          : una unica IV (la ATM) para toda la cadena. Diagnostico: sirve
                    para ver cuanta del la forma del perfil viene de la sonrisa y
                    no del open interest.

La eleccion MUEVE el flip, y cuanto mas lejos del spot este S*, mas lo mueve.
`GammaFlipResult` devuelve el supuesto usado, y `flip_sensitivity` calcula el
flip bajo los tres para que la sensibilidad sea un numero y no una intuicion.

MULTIPLES RAICES
----------------
El perfil de GEX no es monotono: puede cruzar cero varias veces. Devolver "el"
flip sin mirar es esconder informacion. Se localizan TODAS las raices sobre una
rejilla y se elige segun `root_selection`, dejando el resto en `all_roots`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .gex import (
    GexDefinition,
    SignConvention,
    _call_mask_from_column,
    get_convention,
    get_definition,
)
from .greeks import bs_gamma

IvRule = Literal["sticky_strike", "sticky_moneyness", "flat"]
RootSelection = Literal["nearest_spot", "lowest", "highest"]


@dataclass
class GammaFlipResult:
    """Resultado del flip, con lo que hace falta para no fiarse de el a ciegas."""

    flip: float | None
    all_roots: list[float] = field(default_factory=list)
    spot: float = float("nan")
    iv_rule: IvRule = "sticky_strike"
    root_selection: RootSelection = "nearest_spot"
    profile: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gex_at_spot: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def distance_pct(self) -> float:
        """(S - S*) / S. Positivo = el spot esta POR ENCIMA del flip.

        Es la forma normalizada que pide el encargo (seccion 9) y la unica
        comparable entre SPY y SPX, cuyos niveles difieren en un factor 10.
        """
        if self.flip is None or not np.isfinite(self.spot) or self.spot == 0:
            return float("nan")
        return (self.spot - self.flip) / self.spot

    @property
    def regime(self) -> str:
        if self.flip is None:
            return "indeterminado"
        return "gamma_positiva" if self.spot > self.flip else "gamma_negativa"

    def report(self) -> str:
        lines = [
            f"spot            : {self.spot:,.2f}",
            f"GEX en el spot  : {self.gex_at_spot:,.0f}",
            f"gamma flip      : " + (f"{self.flip:,.2f}" if self.flip is not None else "NO ENCONTRADO"),
            f"distancia       : {self.distance_pct:+.2%}" if self.flip is not None else "distancia       : n/d",
            f"regimen         : {self.regime}",
            f"raices halladas : {len(self.all_roots)}"
            + (f"  {[round(r, 2) for r in self.all_roots]}" if len(self.all_roots) > 1 else ""),
            f"supuesto de IV  : {self.iv_rule}  (A8)",
        ]
        lines.extend(f"AVISO: {w}" for w in self.warnings)
        return "\n".join(lines)


def gex_at_spot(
    chain: pd.DataFrame,
    candidate_spot: float,
    *,
    multiplier: int = 100,
    convention: SignConvention | str = "conventional",
    definition: GexDefinition | str = "spot_scaled",
    iv_rule: IvRule = "sticky_strike",
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    atm_iv: float | None = None,
    strike_column: str = "strike",
    iv_column: str = "implied_volatility",
    tau_column: str = "tau",
    oi_column: str = "open_interest",
    type_column: str = "option_type",
) -> float:
    """GEX agregado SI el subyacente estuviera en `candidate_spot`.

    Revalora la cadena entera: recalcula la gamma de cada contrato a ese spot.
    `tau` es el tiempo a vencimiento EN AÑOS y debe venir en la cadena (lo pone
    la capa de datos, que es la que conoce el calendario).
    """
    conv = get_convention(convention) if isinstance(convention, str) else convention
    defn = get_definition(definition) if isinstance(definition, str) else definition

    K = pd.to_numeric(chain[strike_column], errors="coerce").to_numpy(dtype=float)
    tau = pd.to_numeric(chain[tau_column], errors="coerce").to_numpy(dtype=float)
    oi = pd.to_numeric(chain[oi_column], errors="coerce").to_numpy(dtype=float)
    iv = pd.to_numeric(chain[iv_column], errors="coerce").to_numpy(dtype=float)
    is_call = _call_mask_from_column(chain[type_column]).to_numpy()

    sigma = _shift_iv(
        iv, K, chain, candidate_spot, iv_rule, atm_iv,
        strike_column=strike_column, iv_column=iv_column,
    )

    gamma = bs_gamma(candidate_spot, K, tau, risk_free_rate, sigma, dividend_yield)

    signs = conv.signs_for(is_call)
    spot_factor = candidate_spot ** defn.spot_power if defn.spot_power else 1.0
    contrib = gamma * oi * float(multiplier) * spot_factor * defn.scale * signs
    return float(np.nansum(contrib))


def _shift_iv(
    iv: np.ndarray,
    K: np.ndarray,
    chain: pd.DataFrame,
    candidate_spot: float,
    iv_rule: IvRule,
    atm_iv: float | None,
    *,
    strike_column: str,
    iv_column: str,
) -> np.ndarray:
    """Aplica el supuesto A8 sobre que le pasa a la IV al mover el spot."""
    if iv_rule == "sticky_strike":
        return iv

    if iv_rule == "flat":
        if atm_iv is None:
            finite = iv[np.isfinite(iv) & (iv > 0)]
            atm_iv = float(np.median(finite)) if finite.size else 0.20
        return np.full_like(iv, atm_iv)

    if iv_rule == "sticky_moneyness":
        # La sonrisa viaja con el mercado: la IV en el nuevo spot para el strike K
        # es la que hoy tiene el strike con el MISMO moneyness. Se interpola sobre
        # el perfil observado IV(K), evaluandolo en K * (S_actual / S_candidato).
        obs = (
            pd.DataFrame({"K": K, "iv": iv})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .groupby("K")["iv"]
            .median()
            .sort_index()
        )
        if len(obs) < 2:
            return iv
        current_spot = float(
            pd.to_numeric(chain.get("underlying_price", pd.Series([np.nan])), errors="coerce")
            .dropna()
            .median()
        )
        if not np.isfinite(current_spot) or current_spot <= 0:
            return iv
        equivalent_K = K * (current_spot / candidate_spot)
        return np.interp(equivalent_K, obs.index.to_numpy(), obs.to_numpy())

    raise ValueError(f"iv_rule desconocido: {iv_rule!r}")


def find_gamma_flip(
    chain: pd.DataFrame,
    *,
    spot: float | None = None,
    multiplier: int = 100,
    convention: SignConvention | str = "conventional",
    definition: GexDefinition | str = "spot_scaled",
    iv_rule: IvRule = "sticky_strike",
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    search_lower_pct: float = 0.80,
    search_upper_pct: float = 1.20,
    n_grid: int = 121,
    tolerance: float = 0.01,
    max_iterations: int = 100,
    root_selection: RootSelection = "nearest_spot",
    **columns,
) -> GammaFlipResult:
    """Localiza S* tal que GEX(S*) = 0, revalorando la cadena en cada candidato.

    Metodo: rejilla gruesa para detectar TODOS los cambios de signo, y biseccion
    dentro de cada intervalo. Se usa biseccion y no Brent porque el perfil puede
    tener escalones (contratos que entran y salen del rango de gamma relevante) y
    la robustez importa mas aqui que las iteraciones.
    """
    if spot is None:
        spot_col = columns.get("spot_column", "underlying_price")
        spot = float(pd.to_numeric(chain[spot_col], errors="coerce").dropna().median())

    col_kwargs = {k: v for k, v in columns.items() if k != "spot_column"}
    common = dict(
        multiplier=multiplier, convention=convention, definition=definition,
        iv_rule=iv_rule, risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield, **col_kwargs,
    )

    lo, hi = spot * search_lower_pct, spot * search_upper_pct
    grid = np.linspace(lo, hi, n_grid)
    values = np.array([gex_at_spot(chain, float(s), **common) for s in grid])

    profile = pd.Series(values, index=grid, name="gex")
    warnings: list[str] = []

    # Cambios de signo entre nodos consecutivos.
    roots: list[float] = []
    signs = np.sign(values)
    for i in range(len(grid) - 1):
        if signs[i] == 0.0:
            roots.append(float(grid[i]))
            continue
        if signs[i] * signs[i + 1] < 0:
            roots.append(
                _bisect(chain, float(grid[i]), float(grid[i + 1]),
                        values[i], common, tolerance, max_iterations)
            )

    roots = sorted(set(round(r, 6) for r in roots))

    at_spot = gex_at_spot(chain, float(spot), **common)

    if not roots:
        warnings.append(
            f"sin cruce por cero entre {lo:,.2f} y {hi:,.2f}: el GEX no cambia de "
            f"signo en ese rango (min {values.min():,.0f}, max {values.max():,.0f}). "
            f"El flip esta fuera del rango de busqueda o no existe hoy."
        )
        flip = None
    else:
        if len(roots) > 1:
            warnings.append(
                f"{len(roots)} raices: {[round(r, 2) for r in roots]}. El perfil de "
                f"GEX no es monotono; 'el' gamma flip es una simplificacion. Se "
                f"reporta segun root_selection='{root_selection}'."
            )
        if root_selection == "nearest_spot":
            flip = min(roots, key=lambda r: abs(r - spot))
        elif root_selection == "lowest":
            flip = roots[0]
        elif root_selection == "highest":
            flip = roots[-1]
        else:
            raise ValueError(f"root_selection desconocido: {root_selection!r}")

    return GammaFlipResult(
        flip=flip,
        all_roots=roots,
        spot=float(spot),
        iv_rule=iv_rule,
        root_selection=root_selection,
        profile=profile,
        gex_at_spot=at_spot,
        warnings=warnings,
    )


def _bisect(
    chain: pd.DataFrame,
    a: float,
    b: float,
    fa: float,
    common: dict,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Biseccion sobre [a, b] sabiendo que hay cambio de signo."""
    for _ in range(max_iterations):
        if (b - a) < tolerance:
            break
        mid = 0.5 * (a + b)
        fm = gex_at_spot(chain, mid, **common)
        if fm == 0.0:
            return mid
        if (fa < 0) != (fm < 0):
            b = mid
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def flip_sensitivity(
    chain: pd.DataFrame, *, spot: float | None = None, **kwargs
) -> pd.DataFrame:
    """El flip bajo los tres supuestos de IV (A8), para que la duda sea un numero.

    Si los tres coinciden dentro de unas decimas, el supuesto no importa ese dia.
    Si difieren en varios puntos porcentuales, cualquier estrategia que dependa
    de la distancia al flip depende en realidad de A8, y eso hay que decirlo.
    """
    rows = []
    for rule in ("sticky_strike", "sticky_moneyness", "flat"):
        res = find_gamma_flip(chain, spot=spot, iv_rule=rule, **kwargs)
        rows.append({
            "iv_rule": rule,
            "flip": res.flip,
            "distancia_pct": res.distance_pct,
            "n_raices": len(res.all_roots),
            "regimen": res.regime,
        })
    return pd.DataFrame(rows).set_index("iv_rule")
