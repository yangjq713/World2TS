from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss, model, path):
        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "checkpoint.pth")
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
