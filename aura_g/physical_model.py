"""Digital waveguide with fret-aware damping + modal body resonance."""
import numpy as np

class DigitalWaveguideString:
    def __init__(self, sr=48000, f0=196.0, fret_damping_scale=1.0):
        self.sr = sr
        self.f0 = f0
        self.fret_damping_scale = fret_damping_scale
        self.delay = int(sr / f0)
        self.buffer = np.random.randn(self.delay) * 0.01

    def fret_damping(self, fret):
        return 1.0 - 0.05 * fret * self.fret_damping_scale

    def pluck(self, fret=0, pluck_pos=0.5, dynamics=1.0):
        n = self.delay
        idx = int(pluck_pos * n)
        exc = np.zeros(n)
        exc[:idx] = np.linspace(0, 1, idx)
        exc[idx:] = np.linspace(1, 0, n - idx)
        self.buffer = exc * dynamics

    def step(self, n_samples, fret=0):
        out = np.zeros(n_samples)
        damp = self.fret_damping(fret)
        for i in range(n_samples):
            out[i] = self.buffer[0]
            avg = 0.5 * (self.buffer[0] + self.buffer[1])
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = avg * damp
        return out

class BodyResonance:
    def __init__(self, sr=48000, modes=None):
        self.sr = sr
        self.modes = modes or [
            [100, 5, 3], [200, 6, 2], [400, 8, 1],
            [800, 10, 0.5], [1600, 12, -1], [3200, 15, -3]]

    def _biquad_peak(self, x, f0, Q, gain_db):
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * f0 / self.sr
        alpha = np.sin(w0) / (2 * Q)
        b0, b1, b2 = 1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A
        y = np.zeros_like(x)
        x1 = x2 = y1 = y2 = 0.0
        for i, xi in enumerate(x):
            yi = (b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
            x2, x1 = x1, xi
            y2, y1 = y1, yi
            y[i] = yi
        return y

    def __call__(self, x):
        y = x.copy()
        for f0, Q, g in self.modes:
            y = self._biquad_peak(y, f0, Q, g)
        return y

class PhysicalGuitarModel:
    def __init__(self, sr=48000, f0=196.0):
        self.sr = sr
        self.string = DigitalWaveguideString(sr=sr, f0=f0)
        self.body = BodyResonance(sr=sr)

    def synthesize(self, n_samples, fret=0, pluck_pos=0.5, dynamics=1.0):
        self.string.pluck(fret=fret, pluck_pos=pluck_pos, dynamics=dynamics)
        raw = self.string.step(n_samples, fret=fret)
        return self.body(raw)
