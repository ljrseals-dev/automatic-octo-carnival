import numpy as np
import torch
import pytest

from aura_g.ddsp_amp import DDSPAmp, Biquad

def test_amp_forward_shape():
    amp = DDSPAmp(sr=48000).eval()
    x = torch.randn(1, 2048)
    with torch.no_grad():
        y = amp(x)
    assert y.shape == x.shape

def test_amp_output_bounded():
    amp = DDSPAmp(sr=48000).eval()
    x = torch.randn(1, 4096) * 5
    with torch.no_grad():
        y = amp(x)
    assert torch.isfinite(y).all()

def test_biquad_gains():
    q = Biquad(sr=48000, f0=1000, Q=1.0, gain_db=0.0)
    assert q.gain_db.item() == 0.0
    x = torch.randn(1, 1024)
    y = q(x)
    assert y.shape == x.shape
