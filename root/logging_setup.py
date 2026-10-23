"""Structured logging configuration."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from .paths import log_dir

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(name: str = "aura_g", level: int = logging.INFO,
                  to_file: bool = True):
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger(name)

    root.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(sh)

    if to_file:
        fh = RotatingFileHandler(
            log_dir() / f"{name}.log",
            maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(fh)

    return logging.getLogger(name)

def get_logger(name: str = "aura_g"):
    return logging.getLogger(name)
