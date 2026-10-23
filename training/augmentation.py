"""Audio augmentation for training robustness."""
import numpy as np
import torch

class AudioAugment:
    def __init__(self, sr: int = 48000, p: float = 0.5):
        self.sr = sr
        self.p = p

    def gain(self, x: torch.Tensor, db_range=(-6, 6)) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        db = np.random.uniform(*db_range)
        return x * (10 ** (db / 20))

    def add_noise(self, x: torch.Tensor, snr_db=(20, 40)) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        snr = np.random.uniform(*snr_db)
        sig_pow = x.pow(2).mean()
        noise_pow = sig_pow / (10 ** (snr / 10))
        noise = torch.randn_like(x) * noise_pow.sqrt()
        return x + noise

    def lowpass(self, x: torch.Tensor, cutoff=(4000, 12000)) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        fc = np.random.uniform(*cutoff)
        # Simple one-pole IIR via FFT masking
        X = torch.fft.rfft(x)
        freqs = torch.fft.rfftfreq(x.shape[-1], 1 / self.sr).to(x.device)
        mask = (freqs <= fc).float()
        return torch.fft.irfft(X * mask, n=x.shape[-1])

    def pitch_shift(self, x: torch.Tensor, semitones=(-1, 1)) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        st = np.random.uniform(*semitones)
        try:
            import torchaudio.functional as F
            return F.pitch_shift(x, self.sr, st)
        except Exception:
            return x

    def time_stretch(self, x: torch.Tensor, rate=(0.95, 1.05)) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
        r = np.random.uniform(*rate)
        try:
            import torchaudio.functional as F
            return F.phase_vocoder(
                torch.stft(x, 1024, 256, return_complex=True),
                r, torch.hann_window(1024).to(x.device))
        except Exception:
            return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gain(x)
        x = self.add_noise(x)
        x = self.lowpass(x)
        return x
