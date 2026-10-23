"""Module registry — lazy-import the heavy components."""
from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Callable[[], Any]] = {}

def register(name: str, factory: Callable[[], Any]):
    _REGISTRY[name] = factory

def get(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Component '{name}' not registered. "
                       f"Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()

def list_components() -> list:
    return sorted(_REGISTRY.keys())

# ----- Default registrations -----
def _make_pipeline():
    from aura_g.pipeline import AURAG
    return AURAG()

def _make_amp():
    from aura_g.ddsp_amp import DDSPAmp
    return DDSPAmp()

def _make_fx():
    from aura_g.rnn_fx import SampleRateIndependentFX
    return SampleRateIndependentFX()

def _make_phys():
    from aura_g.physical_model import PhysicalGuitarModel
    return PhysicalGuitarModel()

def _make_conformer():
    from transcription.conformer import GuitarConformer
    return GuitarConformer()

register("pipeline", _make_pipeline)
register("amp", _make_amp)
register("fx", _make_fx)
register("phys", _make_phys)
register("conformer", _make_conformer)
