"""Central configuration for AURA-G."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class AudioConfig:
    sample_rate: int = 48000
    hop: int = 480
    window: int = 2048
    block_size: int = 256
    channels: int = 1

@dataclass
class ModelConfig:
    latent_dim: int = 32
    amp_sr: int = 48000
    fx_hidden: int = 32
    conformer_d_model: int = 256
    conformer_layers: int = 6
    conformer_heads: int = 4
    conformer_ff: int = 1024

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 50
    segment_len: int = 32768
    val_split: float = 0.1
    seed: int = 42

@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    checkpoints: Path = field(default_factory=lambda: Path("checkpoints"))
    data: Path = field(default_factory=lambda: Path("data"))
    logs: Path = field(default_factory=lambda: Path("logs"))
    presets: Path = field(default_factory=lambda: Path("presets"))
    output: Path = field(default_factory=lambda: Path("output"))

    def resolve(self):
        for p in (self.checkpoints, self.data, self.logs,
                  self.presets, self.output):
            (self.root / p).mkdir(parents=True, exist_ok=True)

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_upload_mb: int = 100
    job_timeout_sec: int = 300

@dataclass
class AuraGConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    device: str = field(default_factory=lambda: os.environ.get(
        "AURAG_DEVICE", "cuda" if _cuda_available() else "cpu"))
    debug: bool = field(default_factory=lambda:
                        os.environ.get("AURAG_DEBUG", "0") == "1")

    def __post_init__(self):
        self.paths.resolve()

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

_GLOBAL: Optional[AuraGConfig] = None

def get_config(reload: bool = False) -> AuraGConfig:
    global _GLOBAL
    if _GLOBAL is None or reload:
        _GLOBAL = AuraGConfig()
    return _GLOBAL
