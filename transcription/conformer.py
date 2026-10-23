"""Conformer-based polyphonic guitar transcriber."""
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=kernel_size // 2, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return residual + self.act(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        r = x
        x, _ = self.attn(x, x, x, attn_mask=mask)
        return r + x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.net(self.norm(x))

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConvModule(d_model, kernel_size)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ff1(x)
        x = self.attn(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)

class GuitarConformer(nn.Module):
    def __init__(self, n_mels=229, d_model=256, n_heads=4, n_layers=6,
                 d_ff=1024, n_pitches=88, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)])
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_pitches))
        self.onset_head = nn.Linear(d_model, 1)
        self.offset_head = nn.Linear(d_model, 1)

    def forward(self, mel):
        x = self.pos_enc(self.input_proj(mel))
        for layer in self.layers:
            x = layer(x)
        return (torch.sigmoid(self.output_head(x)),
                torch.sigmoid(self.onset_head(x)).squeeze(-1),
                torch.sigmoid(self.offset_head(x)).squeeze(-1))

class NoteEventExtractor:
    def __init__(self, sr=22050, hop=256, onset_thresh=0.5,
                 pitch_thresh=0.4, min_duration=0.05):
        self.sr, self.hop = sr, hop
        self.onset_thresh = onset_thresh
        self.pitch_thresh = pitch_thresh
        self.min_duration = min_duration

    def extract(self, pitch_act, onset, offset):
        T = pitch_act.shape[0]
        onsets = (onset > self.onset_thresh).nonzero(as_tuple=True)[0]
        events = []
        for start in onsets:
            end = start
            while end < T and offset[end] < 0.5:
                end += 1
            if (end - start) * self.hop / self.sr < self.min_duration:
                continue
            mean_act = pitch_act[start:end].mean(dim=0)
            active = (mean_act > self.pitch_thresh).nonzero(as_tuple=True)[0]
            for p in active:
                events.append({'start': start * self.hop / self.sr,
                               'end': end * self.hop / self.sr,
                               'midi': 21 + p.item(),
                               'confidence': float(mean_act[p])})
        events.sort(key=lambda e: (e['start'], -e['confidence']))
        return events

def train_conformer(model, train_loader, val_loader, epochs=50, lr=1e-4,
                    device='cuda'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCELoss()
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for mel, tp, to, tq in train_loader:
            mel, tp, to, tq = mel.to(device), tp.to(device), to.to(device), tq.to(device)
            opt.zero_grad()
            pitch, onset, offset = model(mel)
            loss = bce(pitch, tp) + bce(onset, to) + bce(offset, tq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        sched.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for mel, tp, to, tq in val_loader:
                mel = mel.to(device)
                pitch, onset, offset = model(mel)
                vl += (bce(pitch, tp.to(device)) + bce(onset, to.to(device)) +
                       bce(offset, tq.to(device))).item()
        tl /= len(train_loader)
        vl /= len(val_loader)
        print(f"Epoch {epoch:3d}/{epochs} | Train: {tl:.4f} | Val: {vl:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), 'checkpoints/conformer_best.pt')
    return model
