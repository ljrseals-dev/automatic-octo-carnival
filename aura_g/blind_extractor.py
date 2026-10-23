"""Blind effect extraction: STFT-domain Wiener inversion from dry/wet pairs."""
import numpy as np

class BlindExtractor:
    def __init__(self, sr=48000, n_fft=4096, hop=1024, reg=1e-3):
        self.sr, self.n_fft, self.hop, self.reg = sr, n_fft, hop, reg
        self.H = None

    def _stft(self, x):
        frames = []
        for i in range(0, len(x) - self.n_fft, self.hop):
            frames.append(np.fft.rfft(x[i:i + self.n_fft] * np.hanning(self.n_fft)))
        return np.array(frames)

    def fit(self, dry, wet):
        D = self._stft(dry)
        W = self._stft(wet)
        n = min(len(D), len(W))
        D, W = D[:n], W[:n]
        num = np.sum(W * np.conj(D), axis=0)
        den = np.sum(np.abs(D) ** 2, axis=0) + self.reg
        self.H = num / den
        return self.H

    def apply(self, x):
        if self.H is None:
            raise RuntimeError("Call fit() first.")
        X = self._stft(x)
        Y = X * self.H
        out = np.zeros(len(x))
        win = np.hanning(self.n_fft)
        for i, frame in enumerate(Y):
            start = i * self.hop
            out[start:start + self.n_fft] += np.fft.irfft(frame) * win
        return out
