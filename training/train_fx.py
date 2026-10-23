"""RNN FX training: jointly learn GRU + interpolation filters."""
import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from aura_g.rnn_fx import SampleRateIndependentFX

class FXDataset(Dataset):
    def __init__(self, dry_path, wet_path, sr=48000, segment_len=16384,
                 resample_ratios=(0.5, 0.75, 1.0, 1.5, 2.0)):
        dry, self.sr = sf.read(dry_path, dtype='float32')
        wet, _ = sf.read(wet_path, dtype='float32')
        if dry.ndim > 1: dry = dry.mean(axis=-1)
        if wet.ndim > 1: wet = wet.mean(axis=-1)
        min_len = min(len(dry), len(wet))
        self.dry, self.wet = dry[:min_len], wet[:min_len]
        self.segment_len, self.ratios = segment_len, resample_ratios
        self.n_segments = max(1, min_len // segment_len)

    def __len__(self):
        return self.n_segments * len(self.ratios)

    def __getitem__(self, idx):
        seg_idx = idx // len(self.ratios)
        ratio = self.ratios[idx % len(self.ratios)]
        s = seg_idx * self.segment_len
        e = s + self.segment_len
        x = torch.tensor(self.dry[s:e], dtype=torch.float32)
        y = torch.tensor(self.wet[s:e], dtype=torch.float32)
        if ratio != 1.0:
            nl = int(self.segment_len * ratio)
            x = torch.nn.functional.interpolate(
                x.view(1, 1, -1), size=nl, mode='linear',
                align_corners=False).view(-1)
            x = torch.nn.functional.interpolate(
                x.view(1, 1, -1), size=self.segment_len, mode='linear',
                align_corners=False).view(-1)
        return x, y

class FXTrainer:
    def __init__(self, sr=48000, hidden=32, lr=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SampleRateIndependentFX(
            sr_in=sr, sr_internal=sr, hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.best_loss = float('inf')

    def train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x.unsqueeze(0)).squeeze(0)[:y.shape[-1]]
            loss = self.loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def validate(self, loader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x.unsqueeze(0)).squeeze(0)[:y.shape[-1]]
                total += self.loss_fn(pred, y).item()
        return total / len(loader)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

def train_fx(dry_path, wet_path, epochs=30, batch_size=8, segment_len=16384,
             sr=48000, output_dir='checkpoints/fx'):
    os.makedirs(output_dir, exist_ok=True)
    ds = FXDataset(dry_path, wet_path, sr=sr, segment_len=segment_len)
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False)
    trainer = FXTrainer(sr=sr)
    print(f"Train: {n_train}, Val: {n_val}")
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(tl)
        val_loss = trainer.validate(vl)
        print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.save(os.path.join(output_dir, 'best.pt'))
    print(f"\nDone. Best Val: {trainer.best_loss:.6f}")
    return trainer

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry', required=True)
    p.add_argument('--wet', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--output', default='checkpoints/fx')
    a = p.parse_args()
    train_fx(a.dry, a.wet, epochs=a.epochs, output_dir=a.output)
