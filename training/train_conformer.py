"""Conformer polyphonic training on a GuitarSet-like dataset.

Expected per-clip layout:
    audio.wav   (mono or stereo, any SR)
    notes.txt   lines of: onset_sec  offset_sec  midi  velocity
"""
import os, sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from transcription.conformer import GuitarConformer

class GuitarSetDataset(Dataset):
    def __init__(self, root_dir, sr=22050, hop=256, n_mels=229,
                 segment_sec=4.0, min_midi=21, max_midi=108):
        self.root = Path(root_dir)
        self.sr = sr
        self.hop = hop
        self.n_mels = n_mels
        self.segment_len = int(segment_sec * sr)
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.n_pitches = max_midi - min_midi + 1
        self.items = sorted(self.root.glob('**/audio.wav'))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=hop, n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self): return len(self.items)

    def _load_notes(self, notes_path, total_sec):
        n_frames = int(math.ceil(total_sec * self.sr / self.hop))
        pitch = np.zeros((n_frames, self.n_pitches), dtype=np.float32)
        onset = np.zeros(n_frames, dtype=np.float32)
        offset = np.zeros(n_frames, dtype=np.float32)
        for line in Path(notes_path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 3: continue
            o, e, m = float(parts[0]), float(parts[1]), int(parts[2])
            if not (self.min_midi <= m <= self.max_midi): continue
            fo = int(o * self.sr / self.hop)
            fe = int(e * self.sr / self.hop)
            pitch[max(0, fo):min(n_frames, fe),
                  m - self.min_midi] = 1.0
            if 0 <= fo < n_frames: onset[fo] = 1.0
            if 0 <= fe < n_frames: offset[fe] = 1.0
        return pitch, onset, offset

    def __getitem__(self, idx):
        path = self.items[idx]
        notes_path = path.with_name('notes.txt')
        audio, sr = sf.read(str(path), dtype='float32')
        if audio.ndim > 1: audio = audio.mean(axis=-1)
        audio = torch.tensor(audio)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        total_sec = len(audio) / self.sr

        # Random segment
        if len(audio) > self.segment_len:
            start = np.random.randint(0, len(audio) - self.segment_len)
            audio = audio[start:start + self.segment_len]
            seg_start_sec = start / self.sr
            seg_sec = self.segment_len / self.sr
        else:
            seg_start_sec = 0.0
            seg_sec = total_sec
            pad = self.segment_len - len(audio)
            if pad > 0:
                audio = torch.nn.functional.pad(audio, (0, pad))

        mel = self.db(self.mel(audio)).transpose(0, 1)  # (T, n_mels)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        pitch, onset, offset = self._load_notes(notes_path, total_sec)
        f_start = int(seg_start_sec * self.sr / self.hop)
        f_len = int(seg_sec * self.sr / self.hop)
        pitch = pitch[f_start:f_start + f_len]
        onset = onset[f_start:f_start + f_len]
        offset = offset[f_start:f_start + f_len]

        # Align to mel frames
        T = mel.shape[0]
        def fix(a):
            if len(a) >= T: return a[:T]
            return np.pad(a, ((0, T - len(a)), (0, 0)) if a.ndim == 2
                          else (0, T - len(a)))
        pitch = torch.tensor(fix(pitch))
        onset = torch.tensor(fix(onset))
        offset = torch.tensor(fix(offset))
        return mel, pitch, onset, offset

class WeightedBCE(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight
    def forward(self, pred, target):
        w = 1.0 + (self.pos_weight - 1.0) * target
        bce = -(target * torch.log(pred + 1e-7) +
                (1 - target) * torch.log(1 - pred + 1e-7))
        return (bce * w).mean()

def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for mel, tp_t, on_t, off_t in loader:
            mel = mel.to(device)
            pitch, onset, offset = model(mel)
            pred = (pitch > 0.4).cpu().numpy()
            true = tp_t.numpy().astype(bool)
            tp += int((pred & true).sum())
            fp += int((pred & ~true).sum())
            fn += int((~pred & true).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1

def train_guitarset(root_dir, epochs=100, batch_size=8, lr=1e-4,
                    output_dir='checkpoints/conformer'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = GuitarSetDataset(root_dir)
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2)

    model = GuitarConformer(n_mels=ds.n_mels, n_pitches=ds.n_pitches).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = WeightedBCE(pos_weight=5.0)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for mel, tp_t, on_t, off_t in tl:
            mel = mel.to(device)
            tp_t, on_t, off_t = tp_t.to(device), on_t.to(device), off_t.to(device)
            opt.zero_grad()
            pitch, onset, offset = model(mel)
            loss = bce(pitch, tp_t) + bce(onset, on_t) + bce(offset, off_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        prec, rec, f1 = evaluate(model, vl, device)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {total/len(tl):.4f} | "
              f"P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best.pt'))
            print(f"  -> saved (F1: {f1:.3f})")
    print(f"\nDone. Best F1: {best_f1:.3f}")
    return model

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True,
                   help='Dataset root with **/audio.wav + notes.txt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output', default='checkpoints/conformer')
    a = p.parse_args()
    train_guitarset(a.root, epochs=a.epochs, batch_size=a.batch_size,
                    output_dir=a.output)
