"""Muros de gamma: strikes con concentracion anomala de GEX.

"GRANDE" NO SE DEFINE A OJO
---------------------------
La tentacion es mirar el grafico de GEX por strike, señalar los tres picos que
destacan y llamarlos muros. Eso es elegir la respuesta mirando los datos, y en un
proyecto cuyo objetivo declarado es no sobreajustar (PROJECT_PLAN seccion 25) no
vale.

Aqui hay tres definiciones, las tres explicitas y comparables:

    percentile : |GEX| por encima del percentil p del propio perfil del dia.
                 Se adapta sola a dias tranquilos y a dias cargados, porque el
                 umbral es relativo a la distribucion de ESE dia. Es la mas
                 defendible por defecto.

    zscore     : |GEX| a mas de k desviaciones tipicas de la media del perfil.
                 Supone que el perfil es aproximadamente normal, y NO LO ES: es
                 muy asimetrico y con colas gordas, asi que k=2 selecciona mas
                 strikes de lo que la intuicion gaussiana sugiere. Se incluye
                 porque se usa mucho y conviene poder compararla.

    topk       : los k mayores. Sin pretension estadistica, pero tiene una
                 virtud: el numero de muros es constante, lo que hace las series
                 temporales comparables entre dias.

Ninguna es "la correcta". Cual predice mejor es una pregunta empirica de la
Fase 8, y por eso las tres devuelven el mismo tipo de objeto.

STRIKES CONTIGUOS SON UN MURO, NO TRES
---------------------------------------
En SPX hay strikes cada 5 puntos. Tres strikes seguidos con GEX alto son UNA
concentracion, no tres muros independientes. Sin agrupar, cualquier medida de
"distancia al muro mas cercano" cuenta el mismo accidente tres veces y las
estadisticas de conteo quedan infladas. Por eso `cluster_adjacent` viene activado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

WallMethod = Literal["percentile", "zscore", "topk"]


@dataclass
class GammaWall:
    """Una concentracion de gamma, posiblemente agrupando varios strikes."""

    strike: float                # centro, ponderado por |GEX|
    gex: float                   # GEX neto sumado del grupo
    abs_gex: float               # |GEX| sumado, que es lo que mide "muro"
    strikes: list[float] = field(default_factory=list)
    rank: int = 0

    @property
    def is_positive(self) -> bool:
        return self.gex > 0

    def distance_pct(self, spot: float) -> float:
        return (self.strike - spot) / spot if spot else float("nan")


@dataclass
class WallsResult:
    walls: list[GammaWall] = field(default_factory=list)
    method: WallMethod = "percentile"
    threshold_value: float = float("nan")
    spot: float = float("nan")
    n_strikes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def nearest(self) -> GammaWall | None:
        if not self.walls or not np.isfinite(self.spot):
            return None
        return min(self.walls, key=lambda w: abs(w.strike - self.spot))

    @property
    def largest(self) -> GammaWall | None:
        return max(self.walls, key=lambda w: w.abs_gex) if self.walls else None

    def nearest_above(self) -> GammaWall | None:
        above = [w for w in self.walls if w.strike > self.spot]
        return min(above, key=lambda w: w.strike) if above else None

    def nearest_below(self) -> GammaWall | None:
        below = [w for w in self.walls if w.strike < self.spot]
        return max(below, key=lambda w: w.strike) if below else None

    def to_frame(self) -> pd.DataFrame:
        if not self.walls:
            return pd.DataFrame(
                columns=["strike", "gex", "abs_gex", "n_strikes", "distancia_pct"]
            )
        return pd.DataFrame([
            {
                "strike": w.strike,
                "gex": w.gex,
                "abs_gex": w.abs_gex,
                "n_strikes": len(w.strikes),
                "distancia_pct": w.distance_pct(self.spot),
            }
            for w in self.walls
        ])

    def report(self) -> str:
        lines = [
            f"metodo          : {self.method}  (umbral {self.threshold_value:,.0f})",
            f"strikes en total: {self.n_strikes}",
            f"muros detectados: {len(self.walls)}",
        ]
        n, a, b = self.nearest, self.nearest_above(), self.nearest_below()
        if n:
            lines.append(f"mas cercano     : {n.strike:,.2f} ({n.distance_pct(self.spot):+.2%})")
        if a:
            lines.append(f"resistencia     : {a.strike:,.2f} ({a.distance_pct(self.spot):+.2%})")
        if b:
            lines.append(f"soporte         : {b.strike:,.2f} ({b.distance_pct(self.spot):+.2%})")
        lines.extend(f"AVISO: {w}" for w in self.warnings)
        return "\n".join(lines)


def find_gamma_walls(
    net_gex_by_strike: pd.Series,
    *,
    spot: float,
    method: WallMethod = "percentile",
    percentile: float = 95.0,
    zscore_threshold: float = 2.0,
    top_k: int = 5,
    cluster_adjacent: bool = True,
    cluster_max_gap: int = 2,
    use_absolute: bool = True,
) -> WallsResult:
    """Detecta muros en un perfil `strike -> GEX neto`.

    `use_absolute=True` mide concentracion por |GEX|: un muro es un sitio donde
    hay MUCHA gamma, tenga el signo que tenga. Con False solo se buscan
    concentraciones positivas, que es lo que interesa si se quiere el nivel donde
    la cobertura frena el mercado.
    """
    profile = pd.to_numeric(net_gex_by_strike, errors="coerce").dropna().sort_index()
    warnings: list[str] = []

    if profile.empty:
        return WallsResult(method=method, spot=spot, warnings=["perfil vacio"])

    magnitude = profile.abs() if use_absolute else profile.clip(lower=0.0)

    if method == "percentile":
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile debe estar en (0, 100)")
        threshold = float(np.percentile(magnitude.to_numpy(), percentile))
        selected = magnitude[magnitude >= threshold]
    elif method == "zscore":
        mu, sd = float(magnitude.mean()), float(magnitude.std(ddof=0))
        if sd == 0.0:
            return WallsResult(
                method=method, spot=spot, n_strikes=len(profile),
                warnings=["perfil constante: no hay concentracion que detectar"],
            )
        threshold = mu + zscore_threshold * sd
        selected = magnitude[magnitude >= threshold]
        warnings.append(
            "el metodo zscore supone normalidad y el perfil de GEX es asimetrico "
            "y de colas gordas: el umbral no corresponde a la probabilidad que "
            "sugiere la intuicion gaussiana."
        )
    elif method == "topk":
        if top_k <= 0:
            raise ValueError("top_k debe ser positivo")
        selected = magnitude.nlargest(min(top_k, len(magnitude)))
        threshold = float(selected.min()) if len(selected) else float("nan")
    else:
        raise ValueError(f"metodo desconocido: {method!r}")

    if selected.empty:
        return WallsResult(
            method=method, threshold_value=threshold, spot=spot,
            n_strikes=len(profile),
            warnings=["ningun strike supera el umbral"],
        )

    # UMBRAL QUE NO DISCRIMINA. Si el umbral cae sobre el valor modal del perfil
    # -- tipico cuando el perfil es casi plano con unos pocos picos -- la
    # comparacion `>=` selecciona practicamente todos los strikes. "Todos los
    # strikes son un muro" es lo mismo que "no hay muros", pero el resultado sale
    # con aspecto normal: un solo muro enorme tras agrupar los contiguos, con su
    # centro en la media del perfil entero. Conviene decirlo.
    selected_share = len(selected) / len(profile)
    if selected_share > 0.50:
        warnings.append(
            f"el umbral selecciona el {selected_share:.0%} de los strikes: no esta "
            f"discriminando nada. Suele pasar cuando el perfil es casi plano y el "
            f"percentil cae sobre el valor modal. Sube el percentil o usa 'topk'."
        )

    strikes = sorted(selected.index.tolist())
    groups = (
        _cluster(strikes, profile.index.tolist(), cluster_max_gap)
        if cluster_adjacent else [[s] for s in strikes]
    )

    walls: list[GammaWall] = []
    for group in groups:
        gex_sum = float(profile.loc[group].sum())
        abs_sum = float(profile.loc[group].abs().sum())
        weights = profile.loc[group].abs().to_numpy()
        centre = (
            float(np.average(group, weights=weights))
            if weights.sum() > 0 else float(np.mean(group))
        )
        walls.append(
            GammaWall(strike=centre, gex=gex_sum, abs_gex=abs_sum, strikes=list(group))
        )

    walls.sort(key=lambda w: w.abs_gex, reverse=True)
    for i, w in enumerate(walls, 1):
        w.rank = i

    return WallsResult(
        walls=walls, method=method, threshold_value=threshold,
        spot=float(spot), n_strikes=len(profile), warnings=warnings,
    )


def _cluster(selected: list[float], all_strikes: list[float], max_gap: int) -> list[list[float]]:
    """Agrupa strikes seleccionados separados por <= `max_gap` posiciones.

    La distancia se mide en POSICIONES de la rejilla de strikes, no en dolares:
    la rejilla de SPX es de 5 puntos y la de SPY de 1, asi que un umbral en
    dolares significaria cosas distintas en cada activo.
    """
    if not selected:
        return []
    index_of = {s: i for i, s in enumerate(all_strikes)}
    groups: list[list[float]] = [[selected[0]]]
    for prev, cur in zip(selected, selected[1:]):
        if index_of[cur] - index_of[prev] <= max_gap:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    return groups


def wall_features(result: WallsResult, spot: float) -> dict[str, float]:
    """Features derivadas de los muros, para la capa de `features/`.

    Se devuelven NaN y no ceros cuando no hay muro: un cero en "distancia al
    muro" significaria que el muro esta EN el spot, que es justo lo contrario de
    que no haya muro.
    """
    nearest = result.nearest
    above = result.nearest_above()
    below = result.nearest_below()
    largest = result.largest

    def dist(w: GammaWall | None) -> float:
        return w.distance_pct(spot) if w is not None else float("nan")

    return {
        "wall_n": float(len(result.walls)),
        "wall_nearest_dist": dist(nearest),
        "wall_above_dist": dist(above),
        "wall_below_dist": dist(below),
        "wall_largest_dist": dist(largest),
        "wall_largest_gex": largest.gex if largest else float("nan"),
        "wall_concentration": (
            largest.abs_gex / sum(w.abs_gex for w in result.walls)
            if result.walls else float("nan")
        ),
    }
