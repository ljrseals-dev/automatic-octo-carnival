"""Streaming Conformer inference with overlapping windows and hysteresis."""
import sys
import queue
import threading
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from transcription.conformer import GuitarConformer, NoteEventExtractor

class StreamingConformer:
    """Chunked inference with overlapping analysis windows.

    At each new audio hop, we run the model on the most recent
    `window_sec` of audio and merge frame-level activations
    into a rolling state via a small hysteresis filter.
    """
    def __init__(self, ckpt_path=None, sr=22050, hop=256,
                 window_sec=2.0, overlap=0.5, device=None):
        self.sr = sr
        self.hop = hop
        self.window = int(window_sec * sr)
        self.step = int(self.window * (1.0 - overlap))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GuitarConformer(n_mels=229, n_pitches=88).to(self.device).eval()
        if ckpt_path and Path(ckpt_path).exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        # Mel frontend
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=hop, n_mels=229
        ).to(self.device)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(self.device)

        self.buffer = np.zeros(self.window, dtype=np.float32)
        self.activations = np.zeros((1, 88), dtype=np.float32)
        self.prev_activations = np.zeros_like(self.activations)
        self.onset_state = np.zeros(1, dtype=np.float32)
        self.offset_state = np.zeros(1, dtype=np.float32)
        self.extractor = NoteEventExtractor(sr=sr, hop=hop)
        self.events = []

    def push(self, samples: np.ndarray):
        """Feed new audio samples; run inference if enough buffered."""
        n = len(samples)
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = samples

        with torch.no_grad():
            t = torch.tensor(self.buffer, dtype=torch.float32, device=self.device)
            mel = self.db(self.mel(t)).transpose(0, 1).unsqueeze(0)
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            pitch, onset, offset = self.model(mel)
            # pitch: (1, T, 88), onset/offset: (1, T)
            # Take the last frame's activation with hysteresis
            p = pitch[0, -1].cpu().numpy()
            o = float(onset[0, -1].cpu())
            f = float(offset[0, -1].cpu())

        # Hysteresis on pitch
        active = (p > 0.4) & (self.prev_activations[0] > 0.3)
        new_notes = (p > 0.5) & (self.prev_activations[0] <= 0.3)

        for i in np.where(new_notes)[0]:
            self.events.append({
                'midi': 21 + int(i),
                'start': (len(self.events) * self.hop) / self.sr,
                'confidence': float(p[i]),
            })

        self.prev_activations[0] = np.where(active, p, p * 0.5)
        self.onset_state[0] = o
        self.offset_state[0] = f

    def get_events(self):
        return list(self.events)

class RealtimeConformerEngine:
    """Full engine: audio callback + streaming Conformer + MIDI out."""
    def __init__(self, ckpt_path=None, sr=22050, blocksize=512):
        import sounddevice as sd
        self.sd = sd
        self.sr = sr
        self.blocksize = blocksize
        self.conformer = StreamingConformer(ckpt_path=ckpt_path, sr=sr)
        self.audio_queue = queue.Queue(maxsize=64)
        self.running = False

    def callback(self, indata, outdata, frames, time_info, status):
        outdata[:] = indata
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def worker(self):
        print("Streaming Conformer worker started")
        while self.running:
            try:
                block = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.conformer.push(block)
            for ev in self.conformer.get_events()[-3:]:
                print(f"  [{ev['start']:.2f}s] MIDI {ev['midi']} "
                      f"(conf={ev['confidence']:.2f})")
        print("Worker stopped")

    def run(self):
        self.running = True
        w = threading.Thread(target=self.worker, daemon=True)
        w.start()
        stream = self.sd.Stream(
            samplerate=self.sr, blocksize=self.blocksize,
            channels=1, dtype='float32', callback=self.callback)
        print(f"Real-time Conformer (sr={self.sr}, block={self.blocksize})")
        with stream:
            try:
                while True:
                    self.sd.sleep(100)
            except KeyboardInterrupt:
                pass
        self.running = False
        w.join(timeout=2.0)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--block', type=int, default=512)
    a = p.parse_args()
    RealtimeConformerEngine(ckpt_path=a.ckpt, sr=a.sr,
                            blocksize=a.block).run()
