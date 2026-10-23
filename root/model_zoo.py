"""Central registry for model checkpoints."""
import json
from pathlib import Path
from typing import Optional

import torch

from .logging_setup import setup_logging
from .paths import ckpt_dir

log = setup_logging()

class ModelZoo:
    """Track available checkpoints, versions, and metadata."""

    def __init__(self):
        self.index_path = ckpt_dir("_index") / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def _save_index(self):
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def register(self, kind: str, ckpt_path: str, metadata: dict = None):
        meta = metadata or {}
        meta['path'] = str(ckpt_path)
        meta['exists'] = Path(ckpt_path).exists()
        if meta['exists']:
            meta['size_bytes'] = Path(ckpt_path).stat().st_size
        self.index.setdefault(kind, []).append(meta)
        self._save_index()
        log.info(f"Registered {kind}: {ckpt_path}")

    def list_kind(self, kind: str):
        return self.index.get(kind, [])

    def best(self, kind: str) -> Optional[str]:
        entries = [e for e in self.list_kind(kind) if e.get('exists')]
        if not entries:
            return None
        def score(e):
            for k in ('f1', 'esr', 'loss', 'val_loss'):
                if k in e:
                    return -float(e[k]) if k == 'f1' else float(e[k])
            return 0.0
        entries.sort(key=score)
        return entries[0]['path']

    def load(self, kind: str, device: str = 'cpu'):
        path = self.best(kind)
        if not path:
            raise FileNotFoundError(f"No checkpoint registered for '{kind}'")
        log.info(f"Loading {kind} from {path}")
        return torch.load(path, map_location=device)

    def summary(self):
        lines = []
        for kind, entries in self.index.items():
            lines.append(f"{kind}: {len(entries)} checkpoints")
            for e in entries:
                flag = 'OK ' if e.get('exists') else 'MISS'
                lines.append(f"  [{flag}] {e['path']}")
        return "\n".join(lines)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--register', nargs=2, metavar=('KIND', 'PATH'))
    p.add_argument('--list', action='store_true')
    p.add_argument('--best', metavar='KIND')
    a = p.parse_args()

    zoo = ModelZoo()
    if a.register:
        zoo.register(a.register[0], a.register[1])
    if a.list:
        print(zoo.summary())
    if a.best:
        print(zoo.best(a.best) or "(none)")
