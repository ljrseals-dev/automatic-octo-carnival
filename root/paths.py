"""Path resolution utilities."""
from pathlib import Path
from .config import get_config

def ckpt_dir(kind: str = "amp") -> Path:
    cfg = get_config()
    p = cfg.paths.root / cfg.paths.checkpoints / kind
    p.mkdir(parents=True, exist_ok=True)
    return p

def data_dir(sub: str = "") -> Path:
    cfg = get_config()
    p = cfg.paths.root / cfg.paths.data / sub if sub else cfg.paths.root / cfg.paths.data
    p.mkdir(parents=True, exist_ok=True)
    return p

def log_dir() -> Path:
    cfg = get_config()
    p = cfg.paths.root / cfg.paths.logs
    p.mkdir(parents=True, exist_ok=True)
    return p

def preset_dir() -> Path:
    cfg = get_config()
    p = cfg.paths.root / cfg.paths.presets
    p.mkdir(parents=True, exist_ok=True)
    return p

def output_dir() -> Path:
    cfg = get_config()
    p = cfg.paths.root / cfg.paths.output
    p.mkdir(parents=True, exist_ok=True)
    return p
