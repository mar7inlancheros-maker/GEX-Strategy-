"""Registro de experimentos: append-only, nunca se sobreescribe.

POR QUE EXISTE
--------------
Un sistema con convencion de signo, cuatro definiciones de GEX, umbrales,
horizontes, ventanas y escalones de coste tiene facilmente cientos de
configuraciones alcanzables. El maximo Sharpe de cientos de configuraciones sobre
RUIDO PURO es un numero alto: con 200 intentos y retornos sin ninguna señal, el
mejor Sharpe anual esperado ronda 0,7.

La unica defensa es contar los intentos honestamente. El Sharpe deflactado y el
PBO necesitan `n_trials`, y `n_trials` solo es creible si nadie puede borrar los
intentos que salieron mal. De ahi el append-only: cada evaluacion añade una linea
JSON y ninguna operacion del modulo reescribe el fichero.

Contar solo los intentos "en serio" es hacer trampa. Si se evaluo, cuenta.

FORMATO
-------
JSONL, una linea por experimento, en `reports/experiment_registry.jsonl`. Se lee
con pandas (`pd.read_json(..., lines=True)`) y se inspecciona con cualquier
editor. Deliberadamente no es una base de datos: debe poder leerse dentro de
cinco años sin instalar nada.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import Config, project_root, reports_dir


def _git_commit() -> str | None:
    """Commit actual, para poder reproducir el resultado. None si no hay git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_is_dirty() -> bool | None:
    """True si hay cambios sin comitear: el commit no basta para reproducir."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def config_fingerprint(cfg: Config | dict[str, Any]) -> str:
    """Hash estable de una configuracion.

    Dos experimentos con la misma huella corrieron con los mismos parametros.
    Sirve para detectar el reintento accidental del mismo test y para agrupar
    barridos.
    """
    data = cfg.snapshot() if isinstance(cfg, Config) else cfg
    blob = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class ExperimentRecord:
    """Una evaluacion. Todos los campos que exige el encargo (seccion 32).

    `data_is_synthetic` no es opcional ni cosmetico: separa una prueba del codigo
    de una afirmacion sobre el mercado. Un registro sintetico jamas debe contarse
    como evidencia, y el informe filtra por este campo.
    """

    experiment_id: str
    strategy_version: str
    description: str

    # Datos
    data_period: str                      # "2026-08-31/2027-08-31"
    data_symbols: list[str]
    data_is_synthetic: bool
    data_source: str

    # Diseño
    features: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    cost_assumptions: dict[str, Any] = field(default_factory=dict)

    # Particiones. Vacias mientras no haya datos reales.
    train_period: str | None = None
    validation_period: str | None = None
    test_period: str | None = None

    # Resultados
    performance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    # Trazabilidad (se rellena sola)
    timestamp_utc: str = ""
    config_fingerprint: str = ""
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str = ""
    hostname: str = ""

    def finalize(self) -> "ExperimentRecord":
        self.timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.git_commit = _git_commit()
        self.git_dirty = _git_is_dirty()
        self.python_version = platform.python_version()
        self.hostname = platform.node()
        return self


class ExperimentRegistry:
    """Fichero JSONL append-only.

    No hay metodo para borrar ni para editar. Es intencionado.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else reports_dir() / "experiment_registry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: Config) -> "ExperimentRegistry":
        rel = cfg.get("research.experiment_registry", "reports/experiment_registry.jsonl")
        return cls(project_root() / rel)

    def log(self, record: ExperimentRecord, cfg: Config | None = None) -> ExperimentRecord:
        """Añade un experimento. Devuelve el registro ya sellado."""
        if cfg is not None and not record.config_fingerprint:
            record.config_fingerprint = config_fingerprint(cfg)
        record.finalize()

        line = json.dumps(asdict(record), ensure_ascii=False, default=str)
        # Apertura en modo "a" por escritura, para que dos procesos concurrentes
        # no se pisen media linea.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return iter(())
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    yield json.loads(raw)

    def records(self) -> list[dict[str, Any]]:
        return list(self)

    def n_trials(self, *, exclude_synthetic: bool = True) -> int:
        """Numero de intentos, para el Sharpe deflactado y el PBO.

        Por defecto NO cuenta los sintéticos: son pruebas del motor, no busquedas
        sobre datos de mercado, y contarlos penalizaria de mas. Pero tampoco se
        borran: quedan en el fichero y se pueden contar pasando False.
        """
        return sum(
            1 for r in self
            if not (exclude_synthetic and r.get("data_is_synthetic", False))
        )

    def summary(self) -> str:
        recs = self.records()
        if not recs:
            return f"registro vacio ({self.path})"
        real = sum(1 for r in recs if not r.get("data_is_synthetic", False))
        synth = len(recs) - real
        return (
            f"{len(recs)} experimentos en {self.path}\n"
            f"  sobre datos reales : {real}   <- este es el n_trials que cuenta\n"
            f"  sinteticos         : {synth}  (pruebas del motor, no evidencia)\n"
            f"  primero            : {recs[0].get('timestamp_utc')}\n"
            f"  ultimo             : {recs[-1].get('timestamp_utc')}"
        )
