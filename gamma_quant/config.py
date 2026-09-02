"""Carga de configuracion TOML y rutas del proyecto.

POR QUE TOML Y NO YAML
----------------------
`tomllib` es biblioteca estandar desde Python 3.11. PyYAML seria una dependencia
mas para hacer lo mismo peor (YAML tiene la desgracia de convertir `NO` en False
y `1.2.3` en string segun el humor del parser). Misma decision que en el proyecto
hermano.

POR QUE UNA CLASE Y NO UN DICT SUELTO
-------------------------------------
Porque cada valor leido aqui acaba en el registro del experimento. Un dict suelto
se muta desde cualquier sitio y entonces el registro miente sobre con que se
corrio el backtest. `Config` es de solo lectura: `cfg.get(...)` devuelve copias de
las secciones anidadas y `snapshot()` produce el diccionario exacto que se
archiva.

RUTAS
-----
Nada de rutas absolutas en el codigo. Todo cuelga de `project_root()`, que se
localiza subiendo desde este fichero, y puede redirigirse con la variable de
entorno GAMMA_QUANT_DATA_ROOT (util porque `data/archive/` es irreemplazable y
conviene tenerlo en un disco con copia de seguridad).
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #

def project_root() -> Path:
    """Raiz del proyecto: el directorio que contiene `gamma_quant/`."""
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Secretos
# --------------------------------------------------------------------------- #

_ENV_LOADED = False


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Carga `.env` en os.environ. Sin dependencias externas.

    POR QUE NO python-dotenv
    ------------------------
    Son treinta lineas y una dependencia menos, igual que en el proyecto hermano.

    POR QUE utf-8-sig Y NO utf-8
    ----------------------------
    Porque en Windows esto pasa de verdad: `Out-File -Encoding utf8` y
    `Set-Content` escriben un BOM al principio del fichero. Con `utf-8` el BOM se
    lee como parte del texto y la PRIMERA clave del fichero pasa a llamarse
    "\\ufeffALPHAVANTAGE_API_KEY", que no coincide con nada y ademas es invisible
    al mirarlo. El sintoma es "no encuentro la clave" con la clave delante.
    `utf-8-sig` se come el BOM si esta y no molesta si no.

    Las variables que YA existen en el entorno ganan por defecto (`override=False`):
    en un servidor el entorno real manda sobre un fichero olvidado en disco.
    """
    global _ENV_LOADED
    env_path = Path(path) if path is not None else project_root() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        _ENV_LOADED = True
        return loaded

    for raw in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")   # comillas sobrantes al pegar
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    _ENV_LOADED = True
    return loaded


def get_secret(name: str, *, required: bool = False) -> str | None:
    """Lee un secreto del entorno, cargando `.env` la primera vez.

    Nunca devuelve cadena vacia disfrazada de valor: una clave puesta como
    `ALPHAVANTAGE_API_KEY=` (sin nada detras) es lo mismo que no tenerla, y
    conviene que falle donde se pide y no dentro de una peticion HTTP.
    """
    if not _ENV_LOADED:
        load_env()
    value = (os.environ.get(name) or "").strip()
    if not value:
        if required:
            raise RuntimeError(
                f"falta el secreto '{name}'. Ponlo en {project_root() / '.env'} "
                f"(o exportalo como variable de entorno)."
            )
        return None
    return value


def config_dir() -> Path:
    return project_root() / "configs"


def data_root() -> Path:
    """Raiz de datos. `GAMMA_QUANT_DATA_ROOT` la redirige si existe."""
    env = os.environ.get("GAMMA_QUANT_DATA_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else project_root() / "data"


def reports_dir() -> Path:
    d = project_root() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive_dir() -> Path:
    """Donde viven los snapshots de cadena.

    IRREEMPLAZABLE: el endpoint de CBOE solo sirve el presente, asi que un dia
    que no se archive es un dia perdido para siempre. Merece copia de seguridad
    propia aunque este fuera de git.
    """
    d = data_root() / "archive"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Especificaciones tipadas de las secciones que mas se tocan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class UnderlyingSpec:
    """Lo que distingue a un subyacente de otro. Ver PROJECT_PLAN seccion 3.3.

    `exercise_style` no es decorativo: con "american" la gamma de Black-Scholes
    es una aproximacion (supuesto A4) y los modulos que la usen deben decirlo.
    """

    symbol: str
    multiplier: int
    exercise_style: str          # "american" | "european"
    settlement: str              # "physical" | "cash"
    pays_dividend: bool
    dividend_yield: float
    cboe_endpoint_symbol: str
    monthly_am_settlement: bool = False

    @property
    def black_scholes_is_exact(self) -> bool:
        """False para SPY. Quien pregunte esto debe propagar la advertencia."""
        return self.exercise_style == "european"


@dataclass(frozen=True)
class CostTier:
    """Un escalon de costes. Siempre se reportan los tres."""

    name: str
    spread_capture: float        # fraccion del bid-ask que pagamos
    commission_per_share: float
    slippage_bps: float
    market_impact_bps: float


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

class Config:
    """Configuracion inmutable cargada de TOML.

    Acceso por ruta con puntos:

        cfg.get("gex.definition")            -> "spot_scaled"
        cfg.get("data.oi_lag_days", 1)       -> 1
        cfg.underlying("SPY").multiplier     -> 100
        cfg.cost_tier("base").spread_capture -> 0.5
    """

    def __init__(self, data: Mapping[str, Any], source: str = "<memoria>") -> None:
        self._data = copy.deepcopy(dict(data))
        self._source = source

    # -- construccion ------------------------------------------------------- #

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Carga un TOML. Sin argumento, `configs/default.toml`."""
        p = Path(path) if path is not None else config_dir() / "default.toml"
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"no existe el fichero de configuracion: {p}")
        with p.open("rb") as fh:
            return cls(tomllib.load(fh), source=str(p))

    def with_overrides(self, **dotted: Any) -> "Config":
        """Copia con valores sustituidos. No muta el original.

            cfg.with_overrides(**{"gex.convention": "inverted"})

        Asi se corren los placebos: cambiando configuracion, nunca codigo.
        """
        data = copy.deepcopy(self._data)
        for dotted_key, value in dotted.items():
            parts = dotted_key.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise TypeError(f"'{dotted_key}': '{part}' no es una seccion")
            node[parts[-1]] = value
        return Config(data, source=f"{self._source} + overrides")

    # -- lectura ------------------------------------------------------------ #

    _MISSING = object()

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Valor por ruta con puntos. Sin `default`, la ausencia es un error.

        Que falte una clave debe romper ruidosamente: un `.get()` que devuelve
        None en silencio acaba en un backtest corrido con parametros que nadie
        puso.
        """
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is Config._MISSING:
                    raise KeyError(
                        f"falta '{dotted_key}' en la configuracion ({self._source})"
                    )
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def underlying(self, symbol: str) -> UnderlyingSpec:
        """Especificacion de un subyacente. Falla si no esta declarado.

        Deliberado: preferimos un KeyError a inventar un multiplicador de 100
        para un activo que nadie configuro.
        """
        section = self.get(f"universe.{symbol}", None)
        if section is None:
            declared = sorted(self._data.get("universe", {}))
            raise KeyError(
                f"'{symbol}' no esta en [universe]. Declarados: {declared}"
            )
        return UnderlyingSpec(
            symbol=symbol,
            multiplier=int(section["multiplier"]),
            exercise_style=str(section["exercise_style"]),
            settlement=str(section["settlement"]),
            pays_dividend=bool(section["pays_dividend"]),
            dividend_yield=float(section["dividend_yield"]),
            cboe_endpoint_symbol=str(section["cboe_endpoint_symbol"]),
            monthly_am_settlement=bool(section.get("monthly_am_settlement", False)),
        )

    def cost_tier(self, name: str) -> CostTier:
        section = self.get(f"costs.{name}", None)
        if section is None:
            declared = sorted(self._data.get("costs", {}))
            raise KeyError(f"escalon de costes '{name}' no existe. Hay: {declared}")
        return CostTier(
            name=name,
            spread_capture=float(section["spread_capture"]),
            commission_per_share=float(section["commission_per_share"]),
            slippage_bps=float(section["slippage_bps"]),
            market_impact_bps=float(section["market_impact_bps"]),
        )

    def all_cost_tiers(self) -> list[CostTier]:
        """Los tres escalones. Se reportan siempre juntos, nunca uno solo."""
        return [self.cost_tier(n) for n in sorted(self._data.get("costs", {}))]

    @property
    def seed(self) -> int:
        return int(self.get("meta.random_seed", 20260831))

    @property
    def source(self) -> str:
        return self._source

    def snapshot(self) -> dict[str, Any]:
        """Copia completa, para archivar junto al resultado del experimento."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self._source!r}, secciones={sorted(self._data)})"
