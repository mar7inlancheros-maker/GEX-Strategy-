"""Motor de Gamma Exposure. El corazon del proyecto, y su supuesto mas fragil.

LO PRIMERO, PORQUE CONDICIONA TODO LO DEMAS
-------------------------------------------
El GEX NO SE MIDE. Se MODELA.

Lo que se observa publicamente es el open interest: cuantos contratos vivos hay
en cada strike. Lo que NO se observa es quien esta a cada lado. El inventario de
un market maker es privado y no se publica en ningun sitio.

Todo el GEX que circula -- este incluido -- salva ese hueco con una CONVENCION DE
SIGNO: se supone que el cliente compra calls y compra puts, luego el dealer esta
corto de ambas, luego su gamma es positiva en calls y negativa en puts. Es una
suposicion razonable y es, literalmente, una suposicion (supuestos A1 y A2 del
PROJECT_PLAN).

La consecuencia practica hay que tenerla clara:

    SI LA CONVENCION ESTA MAL, EL SIGNO DE LA SEÑAL ESTA MAL.

Y hay una segunda consecuencia que casi nadie menciona: bajo Black-Scholes la
gamma de una call y la de una put con mismo strike y vencimiento son IDENTICAS
(ver `greeks.py`). Asi que "call GEX" y "put GEX" NO son dos curvaturas
distintas: son la MISMA curvatura repartida segun un supuesto sobre quien esta
enfrente, ponderada por OI distinto. El GEX neto es, en el fondo, una diferencia
de open interest ponderada por gamma.

Por eso la convencion vive aqui como OBJETO INTERCAMBIABLE y no como un `if`:
invertirla es un parametro de configuracion, y el placebo de inversion es un
test de primera clase (PROJECT_PLAN seccion 26).

EL OPEN INTEREST QUE ENTRA AQUI DEBE VENIR YA RETARDADO
-------------------------------------------------------
Este modulo NO aplica el lag de OI: no sabe que dia es. Espera que la capa de
datos le entregue el OI que era CONOCIDO en el momento de la decision
(`oi_lag_days=1` por defecto, supuesto A5). Pasarle el OI del mismo dia produce
un GEX perfectamente calculado sobre informacion del futuro.

`compute_gex` lo comprueba en la medida en que puede -- exige que la columna se
llame como es y acepta `oi_column` explicito -- pero la responsabilidad ultima es
de quien construye el DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


# --------------------------------------------------------------------------- #
# Convenciones de signo del dealer (SUPUESTO A1)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SignConvention:
    """Como se traduce open interest en gamma del dealer.

    `call_sign` y `put_sign` multiplican la gamma de cada contrato. Nada mas.
    Toda la carga teorica del GEX cabe en estos dos numeros, y por eso conviene
    que esten a la vista y no repartidos por el codigo.
    """

    name: str
    call_sign: float
    put_sign: float
    rationale: str
    is_placebo: bool = False

    def signs_for(self, is_call: NDArray[np.bool_]) -> NDArray[np.float64]:
        return np.where(is_call, self.call_sign, self.put_sign).astype(np.float64)


CONVENTIONS: dict[str, SignConvention] = {
    "conventional": SignConvention(
        name="conventional",
        call_sign=+1.0,
        put_sign=-1.0,
        rationale=(
            "El supuesto estandar del sector: el cliente final compra calls "
            "(especulacion alcista) y compra puts (cobertura), asi que el dealer "
            "esta corto de ambas. Corto de call = gamma negativa para el dealer... "
            "pero la convencion publica asigna +1 a calls y -1 a puts porque mide "
            "el efecto NETO de cobertura sobre el mercado. NO esta verificada con "
            "datos: es folklore razonado (A1/A2)."
        ),
    ),
    "all_positive": SignConvention(
        name="all_positive",
        call_sign=+1.0,
        put_sign=+1.0,
        rationale=(
            "No asume nada sobre quien esta enfrente: mide la MAGNITUD de gamma "
            "concentrada en cada strike, sin direccion. Util como control: si el "
            "poder predictivo del GEX neto no supera al de esta version sin signo, "
            "entonces el signo -- que es toda la teoria -- no aporta nada."
        ),
    ),
    "inverted": SignConvention(
        name="inverted",
        call_sign=-1.0,
        put_sign=+1.0,
        rationale=(
            "PLACEBO. La convencion al reves. No es una creencia alternativa: es "
            "el contraste de falsacion. Si la estrategia rinde igual de bien con "
            "el signo invertido, lo que la mueve no es el posicionamiento de "
            "dealers (PROJECT_PLAN seccion 26)."
        ),
        is_placebo=True,
    ),
}


def get_convention(name: str) -> SignConvention:
    if name not in CONVENTIONS:
        raise KeyError(
            f"convencion de signo '{name}' desconocida. Hay: {sorted(CONVENTIONS)}"
        )
    return CONVENTIONS[name]


# --------------------------------------------------------------------------- #
# Definiciones de GEX (SUPUESTO A3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GexDefinition:
    """Una forma de convertir gamma+OI en un numero, con su ficha tecnica.

    El encargo (seccion 7) pide para cada definicion: unidades, interpretacion,
    supuestos, ventajas y debilidades. Van AQUI, pegadas al objeto, y no en un
    documento aparte que se desactualiza. `describe()` las imprime.

    La formula es siempre la misma familia:

        GEX_i = gamma_i * OI_i * multiplicador * S^spot_power * escala * signo_i

    y lo que cambia entre definiciones es `spot_power` y `escala`.
    """

    name: str
    spot_power: int
    scale: float
    units: str
    interpretation: str
    assumptions: str
    advantages: str
    weaknesses: str

    def describe(self) -> str:
        return (
            f"GEX '{self.name}'  (S^{self.spot_power}, escala {self.scale:g})\n"
            f"  unidades      : {self.units}\n"
            f"  interpretacion: {self.interpretation}\n"
            f"  supuestos     : {self.assumptions}\n"
            f"  ventajas      : {self.advantages}\n"
            f"  debilidades   : {self.weaknesses}"
        )


DEFINITIONS: dict[str, GexDefinition] = {
    "naive": GexDefinition(
        name="naive",
        spot_power=0,
        scale=1.0,
        units="contratos-gamma (delta por $1 de movimiento, por contrato)",
        interpretation=(
            "Suma cruda de gamma por open interest. Es la curvatura agregada del "
            "libro, sin escalar por el nivel del subyacente."
        ),
        assumptions="Solo la convencion de signo. Ninguna sobre el tamaño del movimiento.",
        advantages=(
            "La mas transparente: no esconde ningun factor de escala. Comparable "
            "entre strikes dentro de una misma fecha."
        ),
        weaknesses=(
            "NO comparable en el tiempo ni entre activos: si el spot pasa de 400 a "
            "800, el mismo libro da el mismo numero pese a que el riesgo en dolares "
            "se ha multiplicado por cuatro. Inutil para una serie temporal larga, "
            "que es justo lo que necesita este proyecto."
        ),
    ),
    "dollar_delta": GexDefinition(
        name="dollar_delta",
        spot_power=1,
        scale=1.0,
        units="$ de delta por cada $1 de movimiento del subyacente",
        interpretation=(
            "Cuantos dolares de delta tiene que recomprar o vender el dealer si el "
            "subyacente se mueve un dolar."
        ),
        assumptions="Convencion de signo, y que la cobertura es continua y completa.",
        advantages="Unidades economicas directas. Lineal en el movimiento.",
        weaknesses=(
            "Un movimiento de $1 significa cosas distintas en SPY (0,13%) y en SPX "
            "(0,013%). Sigue sin ser comparable entre activos."
        ),
    ),
    "notional": GexDefinition(
        name="notional",
        spot_power=2,
        scale=1.0,
        units="$ de delta por cada movimiento del 100% (sin escalar)",
        interpretation=(
            "La forma S^2 sin el factor del 1%. Aparece en la literatura pero su "
            "magnitud no corresponde a ningun movimiento realista."
        ),
        assumptions="Igual que spot_scaled, pero sin normalizar la escala.",
        advantages="Proporcional a spot_scaled: identica ordenacion y correlaciones.",
        weaknesses=(
            "Los numeros no significan nada por si solos (miles de millones). "
            "Se incluye por trazabilidad con otras fuentes, no porque aporte."
        ),
    ),
    "spot_scaled": GexDefinition(
        name="spot_scaled",
        spot_power=2,
        scale=0.01,
        units="$ de delta por cada movimiento del 1% en el subyacente",
        interpretation=(
            "La definicion de referencia del sector. Responde a: si el mercado se "
            "mueve un 1%, ¿cuantos dolares de delta debe operar el conjunto de "
            "dealers para seguir cubierto? Positivo = venden en subidas y compran "
            "en bajadas (amortiguan). Negativo = compran en subidas y venden en "
            "bajadas (amplifican)."
        ),
        assumptions=(
            "Convencion de signo (A1/A2); cobertura continua y completa; que un "
            "movimiento del 1% es pequeño y por tanto vale la aproximacion de "
            "segundo orden. Para movimientos grandes la gamma misma cambia."
        ),
        advantages=(
            "Comparable en el tiempo y entre activos, porque el 1% es relativo. Es "
            "la unica de las cuatro apta para una serie temporal larga, y por eso "
            "es la que usan los graficos."
        ),
        weaknesses=(
            "El factor S^2 hace que el GEX crezca con el mercado aunque el "
            "posicionamiento no cambie: parte de cualquier tendencia en la serie es "
            "el nivel del indice, no posicionamiento. Al usarla como predictor hay "
            "que normalizar (por ejemplo dividiendo por capitalizacion o "
            "estandarizando en ventana movil), o se estara midiendo el mercado "
            "alcista."
        ),
    ),
}


def get_definition(name: str) -> GexDefinition:
    if name not in DEFINITIONS:
        raise KeyError(
            f"definicion de GEX '{name}' desconocida. Hay: {sorted(DEFINITIONS)}"
        )
    return DEFINITIONS[name]


# --------------------------------------------------------------------------- #
# Calculo a nivel de contrato
# --------------------------------------------------------------------------- #

def gex_contract(
    gamma: ArrayLike,
    open_interest: ArrayLike,
    spot: ArrayLike,
    is_call: ArrayLike,
    *,
    multiplier: int = 100,
    convention: SignConvention | str = "conventional",
    definition: GexDefinition | str = "spot_scaled",
) -> NDArray[np.float64]:
    """GEX contrato a contrato.

        GEX_i = gamma_i * OI_i * mult * S^p * escala * signo_i

    `gamma` debe venir POR ACCION (como la devuelve `greeks.bs_gamma`). El
    multiplicador se aplica AQUI y solo aqui. Aplicarlo tambien en la capa de
    datos da un GEX cien veces mayor que sigue pareciendo razonable, que es la
    peor clase de error.
    """
    conv = get_convention(convention) if isinstance(convention, str) else convention
    defn = get_definition(definition) if isinstance(definition, str) else definition

    gamma_a = np.asarray(gamma, dtype=np.float64)
    oi_a = np.asarray(open_interest, dtype=np.float64)
    spot_a = np.asarray(spot, dtype=np.float64)
    call_mask = np.asarray(is_call, dtype=bool)

    signs = conv.signs_for(call_mask)
    spot_factor = np.power(spot_a, defn.spot_power) if defn.spot_power else 1.0

    return gamma_a * oi_a * float(multiplier) * spot_factor * defn.scale * signs


# --------------------------------------------------------------------------- #
# Agregacion
# --------------------------------------------------------------------------- #

@dataclass
class GexResult:
    """GEX agregado a los cuatro niveles que pide el encargo (seccion 6).

    INVARIANTE que se comprueba en los tests: total == suma por strike ==
    suma por vencimiento. Si los tres no coinciden, hay un bug de agrupacion.
    """

    total: float
    call_total: float
    put_total: float
    by_strike: pd.DataFrame          # index=strike; columnas gex, call_gex, put_gex, oi
    by_expiration: pd.DataFrame      # index=expiration; idem
    by_contract: pd.DataFrame        # la cadena de entrada + columna gex

    spot: float
    definition: GexDefinition
    convention: SignConvention
    multiplier: int

    n_contracts: int = 0
    n_dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def net_by_strike(self) -> pd.Series:
        """Perfil strike -> GEX neto. Lo que consumen flip y muros."""
        return self.by_strike["gex"]

    def report(self) -> str:
        lines = [
            f"GEX total      : {self.total:,.0f}  [{self.definition.units}]",
            f"  calls        : {self.call_total:,.0f}",
            f"  puts         : {self.put_total:,.0f}",
            f"spot           : {self.spot:,.2f}",
            f"definicion     : {self.definition.name}",
            f"convencion     : {self.convention.name}"
            + ("   *** PLACEBO ***" if self.convention.is_placebo else ""),
            f"contratos      : {self.n_contracts:,} usados, {self.n_dropped:,} descartados",
            f"strikes        : {len(self.by_strike)}",
            f"vencimientos   : {len(self.by_expiration)}",
        ]
        lines.extend(f"AVISO: {w}" for w in self.warnings)
        return "\n".join(lines)


def compute_gex(
    chain: pd.DataFrame,
    *,
    spot: float | None = None,
    multiplier: int = 100,
    convention: SignConvention | str = "conventional",
    definition: GexDefinition | str = "spot_scaled",
    gamma_column: str = "gamma",
    oi_column: str = "open_interest",
    strike_column: str = "strike",
    expiration_column: str = "expiration",
    type_column: str = "option_type",
    spot_column: str = "underlying_price",
) -> GexResult:
    """GEX a nivel contrato, strike, vencimiento y total a partir de una cadena.

    `chain` debe seguir el esquema canonico (PROJECT_PLAN 3.1). Las filas con
    gamma u OI no finitos SE DESCARTAN Y SE CUENTAN: no se convierten en ceros,
    porque un cero se suma sin que nadie lo note y un descarte contabilizado se
    ve en el informe.

    EL OI DEBE VENIR YA RETARDADO. Este modulo no sabe que dia es y no puede
    comprobarlo (supuesto A5).
    """
    conv = get_convention(convention) if isinstance(convention, str) else convention
    defn = get_definition(definition) if isinstance(definition, str) else definition

    required = {gamma_column, oi_column, strike_column, type_column}
    missing = required - set(chain.columns)
    if missing:
        raise KeyError(f"faltan columnas en la cadena: {sorted(missing)}")

    df = chain.copy()

    if spot is None:
        if spot_column not in df.columns:
            raise KeyError(
                f"no hay spot: pasa `spot=` o incluye la columna '{spot_column}'"
            )
        spot_values = pd.to_numeric(df[spot_column], errors="coerce").dropna().unique()
        if len(spot_values) == 0:
            raise ValueError("la columna de spot no tiene ningun valor numerico")
        if len(spot_values) > 1:
            # Una cadena es una FOTO en un instante: un solo spot. Varios valores
            # significan snapshots mezclados, y agregarlos daria un GEX que no
            # corresponde a ningun momento real.
            raise ValueError(
                f"la cadena tiene {len(spot_values)} spots distintos "
                f"({spot_values[:3]}...). Parece que hay varios snapshots juntos: "
                f"agregarlos produciria un GEX de ningun instante concreto."
            )
        spot = float(spot_values[0])

    warnings: list[str] = []
    if conv.is_placebo:
        warnings.append(
            f"convencion '{conv.name}' es un PLACEBO. Cualquier resultado que "
            f"salga de aqui es un contraste de falsacion, no una estrategia."
        )

    gamma = pd.to_numeric(df[gamma_column], errors="coerce")
    oi = pd.to_numeric(df[oi_column], errors="coerce")
    is_call = _call_mask_from_column(df[type_column])

    valid = np.isfinite(gamma) & np.isfinite(oi)
    n_dropped = int((~valid).sum())
    if n_dropped:
        warnings.append(
            f"{n_dropped} contratos descartados por gamma u OI no finitos "
            f"(de {len(df)}). Descartados, NO puestos a cero."
        )
    df = df.loc[valid].copy()
    gamma, oi, is_call = gamma[valid], oi[valid], is_call[valid]

    if len(df) == 0:
        raise ValueError("no queda ningun contrato valido en la cadena")

    if (oi < 0).any():
        raise ValueError("hay open interest negativo: el dato esta corrupto")

    df["gex"] = gex_contract(
        gamma.to_numpy(), oi.to_numpy(), float(spot), is_call.to_numpy(),
        multiplier=multiplier, convention=conv, definition=defn,
    )
    df["_is_call"] = is_call.to_numpy()
    df["call_gex"] = np.where(df["_is_call"], df["gex"], 0.0)
    df["put_gex"] = np.where(df["_is_call"], 0.0, df["gex"])

    agg = {"gex": "sum", "call_gex": "sum", "put_gex": "sum", oi_column: "sum"}
    by_strike = df.groupby(strike_column, sort=True).agg(agg)
    by_strike = by_strike.rename(columns={oi_column: "open_interest"})

    if expiration_column in df.columns:
        by_expiration = df.groupby(expiration_column, sort=True).agg(agg)
        by_expiration = by_expiration.rename(columns={oi_column: "open_interest"})
    else:
        by_expiration = pd.DataFrame(columns=["gex", "call_gex", "put_gex", "open_interest"])
        warnings.append(
            f"sin columna '{expiration_column}': no hay desglose por vencimiento"
        )

    total = float(df["gex"].sum())

    return GexResult(
        total=total,
        call_total=float(df["call_gex"].sum()),
        put_total=float(df["put_gex"].sum()),
        by_strike=by_strike,
        by_expiration=by_expiration,
        by_contract=df.drop(columns=["_is_call"]),
        spot=float(spot),
        definition=defn,
        convention=conv,
        multiplier=multiplier,
        n_contracts=len(df),
        n_dropped=n_dropped,
        warnings=warnings,
    )


def _call_mask_from_column(col: pd.Series) -> pd.Series:
    """Normaliza la columna de tipo a booleano. Falla ruidosamente si no puede.

    Adivinar aqui es lo mismo que invertir el signo de medio libro sin avisar.
    """
    if col.dtype == bool:
        return col
    text = col.astype(str).str.strip().str.upper()
    is_call = text.isin(["C", "CALL"])
    is_put = text.isin(["P", "PUT"])
    unknown = ~(is_call | is_put)
    if unknown.any():
        bad = sorted(text[unknown].unique())[:5]
        raise ValueError(
            f"tipos de opcion no reconocidos: {bad}. Se espera C/P o call/put."
        )
    return is_call


# --------------------------------------------------------------------------- #
# Comparacion de definiciones y convenciones
# --------------------------------------------------------------------------- #

def compare_definitions(
    chain: pd.DataFrame,
    *,
    spot: float | None = None,
    multiplier: int = 100,
    convention: SignConvention | str = "conventional",
    **kwargs,
) -> pd.DataFrame:
    """Las cuatro definiciones sobre la misma cadena, para compararlas.

    Sirve para la Fase 8: cual de ellas predice mejor. No para elegir a ojo la
    que de un numero mas bonito.
    """
    rows = []
    for name in DEFINITIONS:
        res = compute_gex(
            chain, spot=spot, multiplier=multiplier,
            convention=convention, definition=name, **kwargs,
        )
        rows.append({
            "definicion": name,
            "gex_total": res.total,
            "call_gex": res.call_total,
            "put_gex": res.put_total,
            "unidades": res.definition.units,
        })
    return pd.DataFrame(rows).set_index("definicion")


def compare_conventions(
    chain: pd.DataFrame,
    *,
    spot: float | None = None,
    multiplier: int = 100,
    definition: GexDefinition | str = "spot_scaled",
    **kwargs,
) -> pd.DataFrame:
    """Las tres convenciones sobre la misma cadena.

    La fila 'inverted' es el placebo: su GEX total deberia ser exactamente el
    negativo del convencional si el libro fuese simetrico, y no serlo dice algo
    sobre el desequilibrio call/put.
    """
    rows = []
    for name, conv in CONVENTIONS.items():
        res = compute_gex(
            chain, spot=spot, multiplier=multiplier,
            convention=name, definition=definition, **kwargs,
        )
        rows.append({
            "convencion": name,
            "es_placebo": conv.is_placebo,
            "gex_total": res.total,
            "call_gex": res.call_total,
            "put_gex": res.put_total,
        })
    return pd.DataFrame(rows).set_index("convencion")


def describe_all() -> str:
    """Ficha tecnica de todo lo configurable. Va al informe final."""
    parts = ["=" * 70, "DEFINICIONES DE GEX", "=" * 70]
    parts.extend(d.describe() + "\n" for d in DEFINITIONS.values())
    parts += ["=" * 70, "CONVENCIONES DE SIGNO DEL DEALER", "=" * 70]
    for c in CONVENTIONS.values():
        flag = "   *** PLACEBO ***" if c.is_placebo else ""
        parts.append(
            f"'{c.name}'  calls {c.call_sign:+.0f}  puts {c.put_sign:+.0f}{flag}\n"
            f"  {c.rationale}\n"
        )
    return "\n".join(parts)
