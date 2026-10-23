"""Parallel real-time transcription + audio pass-through."""
import sys, queue, threading
import numpy as np
import sounddevice as sd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from aura_g.front_end import yin_pitch
from aura_g.transcription import hz_to_midi, midi_to_name

class LiveTranscriber:
    def __init__(self, sr=48000, blocksize=1024):
        self.sr, self.blocksize = sr, blocksize
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False

    def audio_callback(self, indata, outdata, frames, time_info, status):
        outdata[:] = indata
        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def transcribe_worker(self):
        print("Transcription thread started")
        fc, prev = 0, None
        while self.running:
            try:
                block = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            rms = np.sqrt(np.mean(block ** 2))
            fc += 1
            if rms > 0.01:
                f0, conf = yin_pitch(block, self.sr)
                if conf > 0.6 and f0 > 50:
                    midi = hz_to_midi(f0)
                    name = midi_to_name(midi)
                    if name != prev:
                        print(f"[t={fc * self.blocksize / self.sr:.2f}s] "
                              f"{name} (f0={f0:.1f} Hz, conf={conf:.2f})")
                        prev = name
        print("Transcription thread stopped")

    def run(self):
        self.running = True
        w = threading.Thread(target=self.transcribe_worker, daemon=True)
        w.start()
        stream = sd.Stream(samplerate=self.sr, blocksize=self.blocksize,
                           channels=1, dtype='float32',
                           callback=self.audio_callback)
        print(f"Live transcription (sr={self.sr}, block={self.blocksize})")
        with stream:
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                pass
        self.running = False
        w.join(timeout=2.0)

if __name__ == '__main__':
    LiveTranscriber().run()
