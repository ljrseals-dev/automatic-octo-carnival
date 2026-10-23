"""AURA-G real-time engine using sounddevice callbacks."""
import sys
from pathlib import Path
import numpy as np
import torch
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))
from aura_g.latent_bus import LatentBus
from aura_g.ddsp_amp import DDSPAmp
from aura_g.rnn_fx import SampleRateIndependentFX
from aura_g.physical_model import PhysicalGuitarModel
from aura_g.mixer import RoutingMatrix

class RealtimeAURAG:
    def __init__(self, sr=48000, blocksize=256, channels=1,
                 amp_ckpt=None, fx_ckpt=None):
        self.sr, self.blocksize, self.channels = sr, blocksize, channels
        self.amp = DDSPAmp(sr=sr).eval()
        if amp_ckpt and Path(amp_ckpt).exists():
            ck = torch.load(amp_ckpt, map_location='cpu')
            self.amp.load_state_dict(ck['model_state'])
        self.fx = SampleRateIndependentFX(sr_in=sr, sr_internal=sr).eval()
        if fx_ckpt and Path(fx_ckpt).exists():
            self.fx.load_state_dict(torch.load(fx_ckpt, map_location='cpu'))
        self.phys = PhysicalGuitarModel(sr=sr)
        self.mixer = RoutingMatrix()
        self.latent = LatentBus(sample_rate=sr, hop=blocksize)
        self.phys.string.buffer = np.random.randn(max(1, int(sr / 196.0))) * 0.01

        dummy = torch.zeros(1, blocksize)
        with torch.no_grad():
            self.amp(dummy)
            self.fx(dummy)

        self.callback_count, self.underruns = 0, 0

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.underruns += 1
        x_np = indata[:, 0].copy()
        rms = np.sqrt(np.mean(x_np ** 2) + 1e-9)
        dynamics = float(np.clip(rms * 5, 0, 1))
        self.latent.write_pitch_gesture(0.0, 0.0, 0.0, 0.5, 0.0, 0.5, dynamics, 0.0)
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lt = torch.tensor(self.latent.read(), dtype=torch.float32).unsqueeze(0)
            amp_out = self.amp(x_t, lt).squeeze(0).numpy()
            fx_out = self.fx(x_t, lt).squeeze(0).numpy()
            self.phys.string.f0 = 196.0
            self.phys.string.delay = int(self.sr / 196.0)
            phys_out = self.phys.synthesize(frames, fret=3, pluck_pos=0.3,
                                            dynamics=dynamics)
        self.mixer.morph(self.latent.read())
        y = self.mixer.blend(amp_out, fx_out, phys_out, frames)
        outdata[:, 0] = np.clip(y, -1.0, 1.0)
        self.callback_count += 1

    def run(self, input_device=None, output_device=None):
        print(f"Sample rate: {self.sr} Hz")
        print(f"Block size:  {self.blocksize} samples "
              f"({self.blocksize/self.sr*1000:.1f} ms)")
        print("Press Ctrl+C to stop.")
        stream = sd.Stream(
            samplerate=self.sr, blocksize=self.blocksize,
            channels=self.channels, dtype='float32',
            device=(input_device, output_device),
            callback=self.audio_callback, latency='low')
        with stream:
            try:
                while stream.active:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopping...")
        print(f"Callbacks: {self.callback_count}, Underruns: {self.underruns}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sr', type=int, default=48000)
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--amp_ckpt', default=None)
    p.add_argument('--fx_ckpt', default=None)
    p.add_argument('--list_devices', action='store_true')
    a = p.parse_args()
    if a.list_devices:
        print(sd.query_devices()); sys.exit(0)
    RealtimeAURAG(sr=a.sr, blocksize=a.block,
                  amp_ckpt=a.amp_ckpt, fx_ckpt=a.fx_ckpt).run()
