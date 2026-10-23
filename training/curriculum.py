"""Curriculum learning: progressively harder examples."""
import numpy as np
from torch.utils.data import Dataset

class CurriculumScheduler:
    """Order training by difficulty; increase difficulty over epochs."""

    def __init__(self, n_stages: int = 4, epochs_per_stage: int = 10):
        self.n_stages = n_stages
        self.epochs_per_stage = epochs_per_stage

    def stage(self, epoch: int) -> int:
        return min(self.n_stages - 1, epoch // self.epochs_per_stage)

    def difficulty_threshold(self, epoch: int) -> float:
        s = self.stage(epoch)
        return 0.25 + 0.75 * (s / max(1, self.n_stages - 1))

class DifficultyAwareDataset(Dataset):
    """Wraps a dataset; returns only items below current difficulty."""

    def __init__(self, base_dataset, difficulty_fn, seed: int = 42):
        self.base = base_dataset
        self.difficulty_fn = difficulty_fn
        self.rng = np.random.default_rng(seed)
        self.scores = np.array(
            [difficulty_fn(i) for i in range(len(base_dataset))],
            dtype=np.float32)
        self.threshold = 1.0

    def set_threshold(self, t: float):
        self.threshold = float(t)

    def eligible(self):
        return np.where(self.scores <= self.threshold)[0]

    def __len__(self):
        return max(1, len(self.eligible()))

    def __getitem__(self, idx):
        pool = self.eligible()
        return self.base[int(pool[idx % len(pool)])]

def amp_difficulty(i: int) -> float:
    """Placeholder: real difficulty would derive from dataset stats."""
    return (i % 100) / 100.0
