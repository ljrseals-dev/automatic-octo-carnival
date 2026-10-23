import numpy as np
import pytest

from aura_g.pipeline import AURAG

def test_pipeline_runs():
    sr = 48000
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 196 * t)).astype(np.float32)
    ag = AURAG(sr=sr)
    out, events, bus = ag.process(x)
    assert len(out) == len(x)
    assert np.isfinite(out).all()
    assert isinstance(events, list)

def test_pipeline_module_toggle():
    sr = 48000
    x = np.random.randn(sr).astype(np.float32) * 0.1
    ag = AURAG(sr=sr)
    out, _, _ = ag.process(x, use_ddsp=False, use_fx=False, use_phys=False)
    assert np.isfinite(out).all()
