import time
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                 cfg, device='cuda', logger=None, checkpoint_manager=None, csv_logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.csv_logger = csv_logger
        self.scaler = GradScaler()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        t0 = time.time()
        for batch in tqdm(self.train_loader, desc=f"Train E{epoch}"):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * images.size(0)
        avg = running_loss / len(self.train_loader.dataset)
        if self.logger:
            self.logger.info(f"Epoch {epoch} train loss: {avg:.6f} time: {(time.time()-t0):.1f}s")
        return avg

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        metric_acc = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val E{epoch}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                with autocast():
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)
                preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
                running_loss += loss.item() * images.size(0)
                p = preds.cpu().numpy()
                t = labels.cpu().numpy()
                dice = ((p==1) & (t==1)).sum() * 2.0 / (((p==1).sum() + (t==1).sum()) + 1e-8)
                metric_acc.append(dice)
        avg_loss = running_loss / len(self.val_loader.dataset)
        avg_dice = np.mean(metric_acc) if metric_acc else 0.0
        if self.logger:
            self.logger.info(f"Epoch {epoch} val loss: {avg_loss:.6f} dice1: {avg_dice:.4f}")
        if self.scheduler is not None:
            try:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
            except Exception:
                self.scheduler.step()
        return avg_loss, avg_dice

    def fit(self, n_epochs, start_epoch=1):
        best_metric = -1.0
        for epoch in range(start_epoch, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate(epoch)
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice1': val_metric
            }
            if self.csv_logger:
                self.csv_logger.log_row(row)
            if self.checkpoint_manager:
                if val_metric > best_metric:
                    best_metric = val_metric
                    self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, epoch, best_metric, name_prefix='best')
        return best_metric
