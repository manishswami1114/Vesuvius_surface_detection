import os
from pathlib import Path
import torch
import csv

class CheckpointManager:
    def __init__(self, out_dir, keep_best=3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self._saved = []

    def save(self, model, optimizer, scheduler, epoch, metric, name_prefix='ckpt'):
        fname = f"{name_prefix}_epoch{epoch:03d}_metric{metric:.4f}.pth"
        path = self.out_dir / fname
        payload = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
            'metric': metric
        }
        torch.save(payload, str(path))
        self._saved.append(path)
        if len(self._saved) > self.keep_best:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
        return path

class CSVLogger:
    def __init__(self, out_path):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def log_row(self, row: dict):
        if not self._initialized:
            with open(self.out_path, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            self._initialized = True
        with open(self.out_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
