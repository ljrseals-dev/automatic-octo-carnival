import torch
import pytest

from aura_g.rnn_fx import SampleRateIndependentFX, InterpolationFilter

def test_interp_identity():
    f = InterpolationFilter(taps=32, ratio=1.0)
    x = torch.randn(1, 1024)
    y = f(x)
    assert y.shape == x.shape

def test_fx_forward():
    fx = SampleRateIndependentFX(sr_in=48000, sr_internal=48000).eval()
    x = torch.randn(1, 2048)
    with torch.no_grad():
        y = fx(x)
    assert y.shape[0] == 1

def test_fx_with_latent():
    fx = SampleRateIndependentFX().eval()
    x = torch.randn(1, 1024)
    latent = torch.rand(1, 32)
    with torch.no_grad():
        y = fx(x, latent)
    assert y.shape == x.shape
