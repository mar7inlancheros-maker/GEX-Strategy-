"""Archivador diario de cadenas de opciones. EL RELOJ EMPIEZA CUANDO ESTO CORRE.

POR QUE ESTE SCRIPT ES URGENTE Y NO IMPORTANTE
-----------------------------------------------
Urgente e importante no son lo mismo, y esto es lo primero. El endpoint de CBOE
solo sirve EL PRESENTE: no hay archivo, no hay parametro de fecha, no hay forma
de comprar el pasado despues. Un dia que no se guarde hoy no existira nunca.

Cada ejecucion añade ~1 observacion a la muestra con la que algun dia se
contrastara la hipotesis. Sin esto, el proyecto no tiene datos y no los tendra.

USO
---
    python archive_chains.py                    # SPY y SPX, hoy
    python archive_chains.py --symbols SPY      # solo uno
    python archive_chains.py --dry-run          # descarga y valida, no guarda
    python archive_chains.py --coverage         # que hay archivado y que huecos

PROGRAMARLO EN WINDOWS (una vez, y olvidarse)
----------------------------------------------
El mercado cierra a las 16:00 de Nueva York. Con el retardo del feed, hacia las
16:30 ET el snapshot ya es el de cierre. En hora peninsular espanola son las
22:30 (o 21:30 en invierno, cuando el cambio de horario no coincide).

    schtasks /create /tn "GEX archiver" /tr ^
      "\"C:\\...\\python.exe\" \"C:\\...\\archive_chains.py\"" ^
      /sc weekly /d MON,TUE,WED,THU,FRI /st 22:30

Comprobar los HUECOS de vez en cuando con `--coverage`. Un archivador que lleva
tres semanas fallando en silencio es peor que no tenerlo, porque produce una
muestra sesgada en vez de una muestra corta.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gamma_quant.config import Config, archive_dir, get_secret, reports_dir
from gamma_quant.data.ingestion.base import ensure_canonical
from gamma_quant.data.ingestion.cboe import CboeDelayedProvider, gamma_cross_check
from gamma_quant.data.storage.panel import coverage_report, write_snapshot
from gamma_quant.data.validation.quality import validate_chain
from gamma_quant.logging_setup import get_logger

LOG_PATH = reports_dir() / "archiver.log"
log = get_logger("gamma_quant.archiver", log_file=LOG_PATH)


def archive_one(
    symbol: str,
    provider: CboeDelayedProvider,
    cfg: Config,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Descarga, valida y guarda un simbolo. Devuelve el registro del intento.

    Nunca lanza: un fallo en SPX no puede impedir que se guarde SPY. Los fallos
    se devuelven en el registro y se escriben al log, porque un hueco del que
    nadie se entera es el peor resultado posible.
    """
    record: dict = {
        "symbol": symbol,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": False,
    }
    try:
        spec = cfg.underlying(symbol)
        provider.multiplier = spec.multiplier
        provider.dividend_yield = spec.dividend_yield

        log.info("[%s] descargando cadena...", symbol)
        raw = provider.fetch_chain(symbol)
        record["n_raw"] = int(raw.attrs.get("n_raw", len(raw)))
        record["spot"] = float(raw.attrs.get("spot", float("nan")))

        cross = gamma_cross_check(raw)
        record["gamma_cross_check"] = cross
        if cross.get("share_gex_above_tol", 0.0) > 0.10:
            log.warning(
                "[%s] el %.0f%% del GEX viene de contratos que difieren >5%% de la gamma "
                "de CBOE (mediana ponderada %.2f%%). Revisar tau, q o estilo.",
                symbol, 100 * cross["share_gex_above_tol"], 100 * cross.get("weighted_median_rel_diff", 0),
            )

        clean, quarantined, report = validate_chain(raw, symbol=symbol)
        record["n_clean"] = report.n_clean
        record["n_quarantined"] = report.n_quarantined
        record["usable"] = report.is_usable
        log.info(
            "[%s] spot %.2f | %d contratos | %d limpios | %d en cuarentena",
            symbol, record["spot"], record["n_raw"], report.n_clean, report.n_quarantined,
        )

        if not report.is_usable:
            log.error("[%s] cadena NO utilizable:\n%s", symbol, report.report())
            record["error"] = "cadena no supera los controles de calidad"
            _write_quality(symbol, report)
            return record

        canonical = ensure_canonical(clean, source="cboe_delayed")
        _write_quality(symbol, report)

        if dry_run:
            log.info("[%s] --dry-run: NO se guarda", symbol)
            record["ok"] = True
            record["written"] = False
            return record

        today = date.today()
        path, written = write_snapshot(canonical, symbol, today, overwrite=overwrite)
        record["path"] = str(path)
        record["written"] = written
        record["ok"] = True
        if written:
            log.info("[%s] guardado en %s", symbol, path)
        else:
            log.info("[%s] ya existia el snapshot de hoy; no se pisa (%s)", symbol, path)
        return record

    except Exception as exc:  # noqa: BLE001 - un fallo no puede tumbar el resto
        log.exception("[%s] FALLO al archivar: %s", symbol, exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record


def _write_quality(symbol: str, report) -> None:
    out = reports_dir() / "quality"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol}_{date.today().isoformat()}.txt"
    path.write_text(report.report(), encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="pisa el snapshot de hoy (usar con cuidado: es irreemplazable)")
    ap.add_argument("--coverage", action="store_true",
                    help="solo informar de que hay archivado y que huecos hay")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    symbols = args.symbols or cfg.get("data.symbols", ["SPY", "SPX"])

    if args.coverage:
        print(f"ARCHIVO: {archive_dir()}\n")
        for sym in symbols:
            print(coverage_report(sym))
            print()
        return 0

    provider = CboeDelayedProvider(
        user_agent=get_secret("GAMMA_QUANT_USER_AGENT") or "gamma_quant/0.1",
        risk_free_rate=cfg.get("gex.risk_free_rate", 0.04),
    )

    log.info("=" * 60)
    log.info("archivador: %s", ", ".join(symbols))

    records = [
        archive_one(s, provider, cfg, dry_run=args.dry_run, overwrite=args.overwrite)
        for s in symbols
    ]

    # Bitacora append-only de intentos. Es lo que permite detectar despues que
    # faltan dias y por que faltan.
    ledger = reports_dir() / "archive_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]

    print()
    for r in records:
        mark = "OK " if r.get("ok") else "FALLO"
        extra = (
            f"spot {r.get('spot', float('nan')):,.2f} | "
            f"{r.get('n_clean', 0):,} contratos limpios"
            if r.get("ok") else r.get("error", "")
        )
        print(f"  [{mark}] {r['symbol']}: {extra}")

    print(f"\n{len(ok)}/{len(records)} simbolos archivados. Bitacora: {ledger}")
    if failed:
        print("FALLOS: los dias perdidos NO se recuperan. Revisar y relanzar hoy mismo.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
