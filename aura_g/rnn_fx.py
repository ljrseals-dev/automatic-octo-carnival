"""Sample-rate-independent RNN FX with interpolation-filter resampling."""
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F

class InterpolationFilter(nn.Module):
    def __init__(self, taps=64, ratio=1.0, cutoff=0.45):
        super().__init__()
        self.taps = taps
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff * n) * np.kaiser(taps, 8.0)
        h = h / h.sum()
        self.register_buffer('kernel', torch.tensor(h, dtype=torch.float32))
        self.ratio = ratio

    def forward(self, x):
        k = self.kernel.flip(0).view(1, 1, -1)
        x_pad = torch.nn.functional.pad(x.unsqueeze(1), (self.taps - 1, 0))
        y = torch.nn.functional.conv1d(x_pad, k).squeeze(1)
        if self.ratio != 1.0:
            new_len = int(round(y.shape[-1] * self.ratio))
            y = F.resample(y, y.shape[-1], new_len)
        return y

class GRUFX(nn.Module):
    def __init__(self, hidden=32, sr=48000):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        self.latent_proj = nn.Linear(32, hidden)

    def forward(self, x, latent=None):
        B, T = x.shape
        h0 = torch.zeros(1, B, self.gru.hidden_size, device=x.device)
        if latent is not None:
            h0 = torch.tanh(self.latent_proj(latent)).unsqueeze(0)
        y, _ = self.gru(x.unsqueeze(-1), h0)
        return self.out(y).squeeze(-1)

class SampleRateIndependentFX(nn.Module):
    def __init__(self, sr_in=48000, sr_internal=48000, hidden=32):
        super().__init__()
        self.pre = InterpolationFilter(ratio=sr_internal / sr_in)
        self.fx = GRUFX(hidden=hidden)
        self.post = InterpolationFilter(ratio=sr_in / sr_internal)

    def forward(self, x, latent=None):
        x = self.pre(x)
        x = self.fx(x, latent)
        x = self.post(x)
        return x
