"""Motor de opciones: precios, griegas, GEX, gamma flip y muros.

Todo lo de aqui es funcion pura sobre arrays de numpy. Ni lee ficheros, ni
descarga nada, ni conoce el calendario de mercado. Eso lo hace `data/`.

La razon es que estas funciones son las unicas del proyecto que tienen VERDAD
ANALITICA: una gamma de Black-Scholes se puede comparar contra su forma cerrada,
contra la paridad put-call y contra una segunda derivada numerica. Si estan
aisladas se pueden contrastar de verdad; si dependieran de un DataFrame con
veinte columnas, solo se podrian mirar.
"""

from __future__ import annotations

from .pricing import (
    T_FLOOR_DEFAULT,
    SIGMA_FLOOR_DEFAULT,
    bs_price,
    d1_d2,
    implied_volatility,
    american_binomial_price,
    american_binomial_gamma,
)
from .greeks import (
    GreeksResult,
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    compute_greeks,
)
from .gex import (
    CONVENTIONS,
    DEFINITIONS,
    GexDefinition,
    GexResult,
    SignConvention,
    compare_conventions,
    compare_definitions,
    compute_gex,
    describe_all,
    gex_contract,
    get_convention,
    get_definition,
)
from .gamma_flip import (
    GammaFlipResult,
    find_gamma_flip,
    flip_sensitivity,
    gex_at_spot,
)
from .gamma_walls import (
    GammaWall,
    WallsResult,
    find_gamma_walls,
    wall_features,
)

__all__ = [
    "T_FLOOR_DEFAULT",
    "SIGMA_FLOOR_DEFAULT",
    "d1_d2",
    "bs_price",
    "implied_volatility",
    "american_binomial_price",
    "american_binomial_gamma",
    "GreeksResult",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "compute_greeks",
    # GEX
    "CONVENTIONS",
    "DEFINITIONS",
    "GexDefinition",
    "GexResult",
    "SignConvention",
    "compute_gex",
    "gex_contract",
    "get_convention",
    "get_definition",
    "compare_definitions",
    "compare_conventions",
    "describe_all",
    # Gamma flip
    "GammaFlipResult",
    "find_gamma_flip",
    "flip_sensitivity",
    "gex_at_spot",
    # Muros
    "GammaWall",
    "WallsResult",
    "find_gamma_walls",
    "wall_features",
]
