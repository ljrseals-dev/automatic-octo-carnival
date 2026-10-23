"""Distributed Data Parallel training wrapper for AURA-G models."""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_distributed(rank: int, world_size: int,
                      backend: str = 'nccl'):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0

def wrap_model(model: nn.Module, rank: int, device: str):
    model = model.to(device)
    return DDP(model, device_ids=[rank], output_device=rank,
               find_unused_parameters=False)

def make_loader(dataset, batch_size, rank, world_size, shuffle=True,
                num_workers=4):
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                 rank=rank, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True)

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def save_checkpoint(model, optimizer, epoch, path, rank):
    if not is_main(rank):
        return
    state = {
        'epoch': epoch,
        'model': (model.module if isinstance(model, DDP) else model).state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def spawn_train(train_fn, world_size: int):
    """Spawn `world_size` processes running `train_fn(rank, world_size)`."""
    import torch.multiprocessing as mp
    mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)

# ------------------------------------------------------------------
# Example: DDSP amp distributed trainer
# ------------------------------------------------------------------
def _distributed_amp_worker(rank: int, world_size: int,
                            dry_path: str, wet_path: str,
                            epochs: int = 50, batch_size: int = 4):
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent))

    from aura_g.ddsp_amp import DDSPAmp
    from training.train_amp import PairedAudioDataset, ESRLoss, MultiResSTFTLoss

    setup_distributed(rank, world_size)
    device = f'cuda:{rank}'

    dataset = PairedAudioDataset(dry_path, wet_path)
    loader = make_loader(dataset, batch_size, rank, world_size)

    model = wrap_model(DDSPAmp(), rank, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    esr = ESRLoss()
    stft = MultiResSTFTLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        loader.sampler.set_epoch(epoch)
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x.unsqueeze(0)).squeeze(0)
            loss = esr(pred, y) + 0.1 * stft(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = all_reduce_mean(torch.tensor(total / len(loader)))
        if is_main(rank):
            print(f"Epoch {epoch}: loss={avg.item():.6f}")

        save_checkpoint(model, optimizer, epoch,
                        f"checkpoints/amp/ddp_epoch_{epoch}.pt", rank)

    cleanup()

def run_distributed_amp(dry_path: str, wet_path: str,
                        world_size: int = 2, epochs: int = 50):
    def worker(rank, ws):
        _distributed_amp_worker(rank, ws, dry_path, wet_path, epochs)
    spawn_train(worker, world_size)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry', required=True)
    p.add_argument('--wet', required=True)
    p.add_argument('--world_size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=50)
    a = p.parse_args()
    run_distributed_amp(a.dry, a.wet, a.world_size, a.epochs)
