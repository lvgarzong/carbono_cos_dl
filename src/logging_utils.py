"""
logging_utils.py
================
Configuracion de logging unificado: a consola y a archivo.

Si el codigo falla, el archivo de log (logs/run_*.log y el log por experimento)
contiene el traceback completo para diagnosticar el error con precision.
"""
from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "cos", log_dir: Path | str | None = None,
               filename: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Devuelve un logger que escribe a consola y, opcionalmente, a un archivo."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Evita handlers duplicados si se llama varias veces.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_dir / filename, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_dir / filename)

    return logger


def add_file_handler(logger: logging.Logger, path: Path | str) -> logging.FileHandler:
    """Agrega un handler de archivo extra (p.ej. uno por experimento)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(fh)
    return fh
