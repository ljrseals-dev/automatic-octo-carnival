"""Full AURA-G pipeline: analyzer -> latent bus -> modules -> output + transcription."""
import numpy as np
import torch
from .front_end import FrontEndAnalyzer
from .ddsp_amp import DDSPAmp
from .rnn_fx import SampleRateIndependentFX
from .physical_model import PhysicalGuitarModel
from .transcription import TranscriptionEngine
from .mixer import RoutingMatrix

class AURAG:
    def __init__(self, sr=48000, hop=480):
        self.sr, self.hop = sr, hop
        self.analyzer = FrontEndAnalyzer(sr=sr, hop=hop)
        self.amp = DDSPAmp(sr=sr).eval()
        self.fx = SampleRateIndependentFX(sr_in=sr, sr_internal=sr).eval()
        self.phys = PhysicalGuitarModel(sr=sr)
        self.transcriber = TranscriptionEngine(sr=sr, hop=hop)
        self.mixer = RoutingMatrix()

    def process(self, audio, use_ddsp=True, use_fx=True, use_phys=True):
        bus_history = list(self.analyzer.process(audio))
        x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean_latent = torch.tensor(
                np.mean([b.read() for b in bus_history], axis=0),
                dtype=torch.float32)
            amp_out = self.amp(x, mean_latent).squeeze(0).numpy() if use_ddsp \
                else np.zeros_like(audio)
            fx_out = self.fx(x, mean_latent.unsqueeze(0)).squeeze(0).numpy() if use_fx \
                else np.zeros_like(audio)

        if use_phys and bus_history:
            pg = bus_history[0].get_pitch_gesture()
            f0 = 440.0 * (2 ** (pg[0] * 6.0)) if pg[0] != 0 else 196.0
            self.phys.string.f0 = f0 if f0 > 0 else 196.0
            self.phys.string.delay = int(self.sr / self.phys.string.f0)
            self.phys.string.buffer = np.random.randn(self.phys.string.delay) * 0.01
            phys_out = self.phys.synthesize(len(audio), fret=3,
                                            pluck_pos=0.3, dynamics=pg[6])
        else:
            phys_out = np.zeros_like(audio)

        if bus_history:
            self.mixer.morph(bus_history[0].read())
        out = self.mixer.blend(amp_out, fx_out, phys_out, len(audio))
        events = self.transcriber.process(bus_history)
        return out.astype(np.float32), events, bus_history
