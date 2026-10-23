"""DDSP amp training: fit interpretable model to paired dry/wet audio."""
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from aura_g.ddsp_amp import DDSPAmp

class PairedAudioDataset(Dataset):
    def __init__(self, dry_path, wet_path, sr=48000, segment_len=65536):
        self.dry, self.sr = sf.read(dry_path, dtype='float32')
        self.wet, _ = sf.read(wet_path, dtype='float32')
        if self.dry.ndim > 1: self.dry = self.dry.mean(axis=-1)
        if self.wet.ndim > 1: self.wet = self.wet.mean(axis=-1)
        min_len = min(len(self.dry), len(self.wet))
        self.dry = self.dry[:min_len]
        self.wet = self.wet[:min_len]
        self.segment_len = segment_len
        self.n_segments = max(1, min_len // segment_len)

    def __len__(self): return self.n_segments

    def __getitem__(self, idx):
        s = idx * self.segment_len
        e = s + self.segment_len
        return (torch.tensor(self.dry[s:e], dtype=torch.float32),
                torch.tensor(self.wet[s:e], dtype=torch.float32))

class ESRLoss(nn.Module):
    def forward(self, pred, target):
        return torch.sum((target - pred) ** 2) / (torch.sum(target ** 2) + 1e-8)

class MultiResSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes

    def forward(self, pred, target):
        loss = 0.0
        for fft, hop in zip(self.fft_sizes, self.hop_sizes):
            win = torch.hann_window(fft).to(pred.device)
            P = torch.stft(pred, fft, hop, window=win, return_complex=True)
            T = torch.stft(target, fft, hop, window=win, return_complex=True)
            Pm, Tm = P.abs() + 1e-7, T.abs() + 1e-7
            loss += torch.mean(torch.abs(Pm - Tm) / Tm)
            loss += torch.mean((torch.log(Pm) - torch.log(Tm)) ** 2)
        return loss / len(self.fft_sizes)

class AmpTrainer:
    def __init__(self, sr=48000, lr=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = DDSPAmp(sr=sr).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.esr_loss, self.stft_loss = ESRLoss(), MultiResSTFTLoss()
        self.best_esr = float('inf')

    def train_epoch(self, loader, alpha=0.1):
        self.model.train()
        tl, te = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x.unsqueeze(0)).squeeze(0)
            le, ls = self.esr_loss(pred, y), self.stft_loss(pred, y)
            (le + alpha * ls).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            tl += (le + alpha * ls).item()
            te += le.item()
        return tl / len(loader), te / len(loader)

    def validate(self, loader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x.unsqueeze(0)).squeeze(0)
                total += self.esr_loss(pred, y).item()
        return total / len(loader)

    def save(self, path):
        torch.save({'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'best_esr': self.best_esr}, path)

def train_amp(dry_path, wet_path, epochs=50, batch_size=4, segment_len=32768,
              sr=48000, output_dir='checkpoints/amp'):
    os.makedirs(output_dir, exist_ok=True)
    ds = PairedAudioDataset(dry_path, wet_path, sr=sr, segment_len=segment_len)
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    tr, va = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False)
    trainer = AmpTrainer(sr=sr)
    print(f"Train: {n_train}, Val: {n_val}, Device: {trainer.device}")
    for epoch in range(1, epochs + 1):
        train_loss, train_esr = trainer.train_epoch(tl)
        val_esr = trainer.validate(vl)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.6f} | "
              f"Train ESR: {train_esr:.4%} | Val ESR: {val_esr:.4%}")
        if val_esr < trainer.best_esr:
            trainer.best_esr = val_esr
            trainer.save(os.path.join(output_dir, 'best.pt'))
            print(f"  -> saved (ESR: {val_esr:.4%})")
    print(f"\nDone. Best Val ESR: {trainer.best_esr:.4%}")
    return trainer

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry', required=True)
    p.add_argument('--wet', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--output', default='checkpoints/amp')
    a = p.parse_args()
    train_amp(a.dry, a.wet, epochs=a.epochs, batch_size=a.batch_size,
              output_dir=a.output)
