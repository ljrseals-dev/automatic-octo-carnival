import numpy as np
import pytest

from aura_g.front_end import yin_pitch, spectral_features, FrontEndAnalyzer

def test_yin_on_sine():
    sr = 48000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t)
    f0, conf = yin_pitch(x[:2048], sr)
    assert abs(f0 - 220) < 5
    assert conf > 0.5

def test_yin_silence():
    sr = 48000
    f0, conf = yin_pitch(np.zeros(2048), sr)
    assert f0 == 0.0

def test_spectral_features_range():
    sr = 48000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048).astype(np.float32)
    f = spectral_features(x, sr)
    for k, v in f.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

def test_analyzer_yields_buses():
    sr = 48000
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 196 * t).astype(np.float32)
    ag = FrontEndAnalyzer(sr=sr, hop=480, win=2048)
    buses = list(ag.process(x))
    assert len(buses) > 0
    assert buses[0].read().shape == (32,)
