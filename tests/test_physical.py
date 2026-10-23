import numpy as np
import pytest

from aura_g.physical_model import (DigitalWaveguideString,
                                   BodyResonance,
                                   PhysicalGuitarModel)

def test_waveguide_produces_sound():
    w = DigitalWaveguideString(sr=48000, f0=196.0)
    w.pluck(fret=0, pluck_pos=0.3, dynamics=1.0)
    y = w.step(2048, fret=0)
    assert len(y) == 2048
    assert np.abs(y).max() > 0

def test_fret_damping_decreases():
    w = DigitalWaveguideString(sr=48000, f0=196.0)
    d0 = w.fret_damping(0)
    d5 = w.fret_damping(5)
    d12 = w.fret_damping(12)
    assert d0 > d5 > d12

def test_body_resonance_runs():
    b = BodyResonance(sr=48000)
    x = np.random.randn(2048).astype(np.float32)
    y = b(x)
    assert y.shape == x.shape

def test_full_model():
    m = PhysicalGuitarModel(sr=48000, f0=196.0)
    y = m.synthesize(4096, fret=3, pluck_pos=0.4, dynamics=0.9)
    assert len(y) == 4096
    assert np.isfinite(y).all()
