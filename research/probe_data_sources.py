"""Verifica QUE datos nos vende de verdad cada fuente, antes de construir nada.

POR QUE ESTE SCRIPT EXISTE
--------------------------
Toda la Fase 6 (adaptadores de ingesta) depende de responder tres preguntas con
hechos y no con documentacion:

    1. ¿Cubre SPY? ¿Y SPX?
    2. ¿HASTA QUE AÑO llega el historico de verdad?
    3. ¿Trae `open_interest`, que es el campo sin el cual no hay GEX?

Construir el adaptador primero y descubrir despues que el historico empieza en
2022 seria trabajar tres dias para nada.

CUIDADO CON LA CUOTA
--------------------
El tier gratuito de Alpha Vantage son 25 PETICIONES AL DIA. Este script cuenta
cada peticion y aborta antes de pasarse. La busqueda de la fecha mas antigua es
BINARIA justamente por eso: encontrar el año de inicio entre 2008 y hoy cuesta
~5 peticiones en vez de las ~18 de probar año por año.

    python research/probe_data_sources.py            # SPY, gasta ~8 peticiones
    python research/probe_data_sources.py --spx      # añade SPX
    python research/probe_data_sources.py --budget 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gamma_quant.config import get_secret, reports_dir

AV_URL = "https://www.alphavantage.co/query"

# Campos que el esquema canonico (PROJECT_PLAN 3.1) marca como obligatorios.
REQUIRED_FIELDS = {"strike", "expiration", "type", "open_interest", "implied_volatility"}
USEFUL_FIELDS = {"bid", "ask", "volume", "delta", "gamma", "vega", "theta"}


class QuotaExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    """Contador de peticiones. Aborta antes de pasarse, no despues.

    Ademas ESPACIA las llamadas: Alpha Vantage limita la rafaga a ~1 peticion por
    segundo y, cuando se supera, responde con un aviso de ritmo EN LUGAR de los
    datos. Ese aviso se confunde facilmente con "no hay cobertura para este
    simbolo", que es un diagnostico completamente distinto y manda a rehacer
    trabajo que estaba bien.
    """

    limit: int
    min_interval: float = 1.2
    used: int = 0
    _last_call: float = 0.0

    def spend(self, what: str) -> None:
        if self.used >= self.limit:
            raise QuotaExhausted(
                f"presupuesto agotado ({self.limit} peticiones) al intentar: {what}"
            )
        elapsed = time.monotonic() - self._last_call
        if self._last_call and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()
        self.used += 1


@dataclass
class ProbeResult:
    symbol: str
    covered: bool = False
    earliest_ok: date | None = None
    latest_ok: date | None = None
    n_contracts: int = 0
    n_expirations: int = 0
    total_oi: int = 0
    fields: set[str] = field(default_factory=set)
    missing_required: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def av_historical_options(
    symbol: str, on: date, api_key: str, budget: Budget
) -> tuple[list[dict], str | None]:
    """Una cadena historica. Devuelve (contratos, mensaje_de_error)."""
    budget.spend(f"{symbol} @ {on.isoformat()}")
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "date": on.isoformat(),
        "apikey": api_key,
    }
    url = f"{AV_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "gamma_quant/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - queremos el motivo tal cual
        return [], f"{type(exc).__name__}: {exc}"

    data = payload.get("data")
    if isinstance(data, list) and data:
        # NO BASTA CON QUE VENGA UNA LISTA NO VACIA.
        # Alpha Vantage responde a los endpoints premium con una muestra de
        # ESQUEMA ARTIFICIAL cuando la clave no tiene acceso: HTTP 200, campo
        # `data` con registros bien formados... y contratos "XXYYZZ999999C00020000"
        # con vencimiento "2099-99-99". Un adaptador que solo mire `if data:` los
        # ingiere como reales y envenena el panel sin un solo error.
        # Se detecta por el mensaje y por los datos, no por uno solo.
        note = str(payload.get("message", "")) + str(payload.get("Information", ""))
        if "ARTIFICIAL" in note.upper() or "premium endpoint" in note:
            return [], f"DATOS DE MUESTRA ARTIFICIALES (endpoint premium): {note[:160]}"
        if _looks_like_placeholder(data[0]):
            return [], f"DATOS DE MUESTRA ARTIFICIALES detectados en el payload: {data[0]}"
        return data, None

    # Alpha Vantage devuelve 200 con un mensaje dentro cuando algo va mal.
    for key in ("Information", "Note", "Error Message", "message"):
        if key in payload:
            return [], str(payload[key])[:300]
    return [], "respuesta sin 'data' y sin mensaje"


def _looks_like_placeholder(record: dict) -> bool:
    """¿Es este contrato un relleno de documentacion y no un contrato real?

    Las señales son groseras a proposito: fechas imposibles, tickers de relleno,
    todos los numeros identicos. Mejor un falso positivo -- que aborta y obliga a
    mirar -- que un falso negativo, que mete basura en el panel.
    """
    text = " ".join(str(v) for v in record.values()).upper()
    if "XXYYZZ" in text or "99-99" in text:
        return True
    sym = str(record.get("symbol", "")).upper()
    if sym in {"XXYYZZ", "SYMBOL", "TICKER", ""}:
        return True
    for field_name in ("expiration", "date"):
        value = str(record.get(field_name, ""))
        try:
            date.fromisoformat(value)
        except ValueError:
            if value:
                return True
    return False


def _prev_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def probe_symbol(symbol: str, api_key: str, budget: Budget, *, deep: bool = True) -> ProbeResult:
    """Cobertura, profundidad y campos de un simbolo."""
    res = ProbeResult(symbol=symbol)

    # 1) ¿Existe? Se pregunta por una fecha reciente y cerrada (no hoy: puede que
    #    aun no este consolidada).
    recent = _prev_weekday(date.today() - timedelta(days=7))
    contracts, err = av_historical_options(symbol, recent, api_key, budget)
    if not contracts:
        res.notes.append(f"sin datos el {recent}: {err}")
        # Un fallo puede ser festivo. Se reintenta una semana antes antes de
        # declarar que no hay cobertura.
        recent2 = _prev_weekday(recent - timedelta(days=7))
        contracts, err2 = av_historical_options(symbol, recent2, api_key, budget)
        if not contracts:
            res.notes.append(f"sin datos el {recent2}: {err2}")
            return res
        recent = recent2

    res.covered = True
    res.latest_ok = recent
    res.fields = set(contracts[0].keys())
    res.missing_required = REQUIRED_FIELDS - res.fields
    res.n_contracts = len(contracts)
    res.n_expirations = len({c.get("expiration") for c in contracts})
    res.total_oi = sum(int(float(c.get("open_interest") or 0)) for c in contracts)

    if not deep:
        return res

    # 2) ¿Hasta donde llega? Busqueda BINARIA sobre el año de inicio.
    #    Barata: log2(18 años) ~ 5 peticiones en vez de 18.
    lo, hi = 2008, recent.year          # lo = quiza no cubierto, hi = cubierto
    probe_day = date(hi, 6, 15)
    while lo < hi:
        mid = (lo + hi) // 2
        probe_day = _prev_weekday(date(mid, 6, 16))
        try:
            got, msg = av_historical_options(symbol, probe_day, api_key, budget)
        except QuotaExhausted as exc:
            res.notes.append(f"busqueda de profundidad incompleta: {exc}")
            break
        if got:
            hi = mid
            res.earliest_ok = probe_day
        else:
            lo = mid + 1
            if msg:
                res.notes.append(f"{probe_day.year}: {msg[:120]}")
        time.sleep(0.3)

    return res


def render(res: ProbeResult) -> str:
    if not res.covered:
        return (
            f"\n{res.symbol}: NO CUBIERTO\n"
            + "\n".join(f"    {n}" for n in res.notes)
        )
    lines = [
        f"\n{res.symbol}: CUBIERTO",
        f"    fecha probada     : {res.latest_ok}",
        f"    contratos         : {res.n_contracts:,}",
        f"    vencimientos      : {res.n_expirations}",
        f"    OI total          : {res.total_oi:,}",
        f"    historico llega a : {res.earliest_ok or 'no determinado'}",
    ]
    if res.missing_required:
        lines.append(f"    *** FALTAN CAMPOS OBLIGATORIOS: {sorted(res.missing_required)}")
    else:
        lines.append("    campos obligatorios: TODOS presentes (open_interest incluido)")
    extra = sorted(USEFUL_FIELDS & res.fields)
    lines.append(f"    campos utiles     : {extra}")
    for n in res.notes:
        lines.append(f"    nota: {n}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spx", action="store_true", help="probar tambien SPX")
    ap.add_argument("--budget", type=int, default=18,
                    help="maximo de peticiones (gratis son 25/dia)")
    ap.add_argument("--shallow", action="store_true",
                    help="solo cobertura, sin buscar la fecha mas antigua")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("VERIFICACION DE FUENTES DE DATOS DE OPCIONES")
    print("=" * 78)

    api_key = get_secret("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print(
            "\nNo hay ALPHAVANTAGE_API_KEY.\n"
            "  1. Consigue una gratis en https://www.alphavantage.co/support/#api-key\n"
            "  2. Pegala en el fichero .env de la raiz del proyecto:\n"
            "         ALPHAVANTAGE_API_KEY=TU_CLAVE\n"
            "  3. Vuelve a lanzar este script.\n"
        )
        return 2

    print(f"clave detectada: ...{api_key[-4:]}  (presupuesto: {args.budget} peticiones)")

    budget = Budget(limit=args.budget)
    symbols = ["SPY"] + (["SPX"] if args.spx else [])
    results: list[ProbeResult] = []

    for sym in symbols:
        try:
            results.append(probe_symbol(sym, api_key, budget, deep=not args.shallow))
        except QuotaExhausted as exc:
            print(f"\n[ABORTADO] {exc}")
            break

    for r in results:
        print(render(r))

    print(f"\npeticiones gastadas: {budget.used} de {budget.limit}")

    out = reports_dir() / "data_source_probe.json"
    out.write_text(
        json.dumps(
            [
                {
                    **{k: v for k, v in r.__dict__.items() if k != "fields"},
                    "fields": sorted(r.fields),
                    "missing_required": sorted(r.missing_required),
                    "earliest_ok": r.earliest_ok.isoformat() if r.earliest_ok else None,
                    "latest_ok": r.latest_ok.isoformat() if r.latest_ok else None,
                }
                for r in results
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"informe: {out}")

    ok = bool(results) and all(r.covered and not r.missing_required for r in results)
    print("\nVEREDICTO:", "fuente utilizable" if ok else "revisar antes de construir el adaptador")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
