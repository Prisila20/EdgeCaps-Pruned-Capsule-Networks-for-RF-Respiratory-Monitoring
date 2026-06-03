import torch
from torch.optim import Optimizer
import torch.optim as optim

# class EarlyStopping:
#     def __init__(self, patience=10):
#         self.patience = patience
#         self.best = None
#         self.counter = 0
#         self.stop = False

#     def step(self, metric):
#         if self.best is None or metric > self.best:
#             self.best = metric
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.stop = True


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def step(self, metric, model=None):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0

            if model is not None and self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def get_lr_scheduler(
    optimizer: Optimizer,
    schedule_type: str,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int = 5
):
    """
    Returns a LR scheduler based on schedule_type.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        schedule_type (str): One of ["cosine", "linear", "warmup_constant", ...].
        total_epochs (int): Number of training epochs.
        base_lr (float): Initial learning rate (and sometimes target LR).
        warmup_epochs (int): Used for 'warmup_constant' schedule, or can be extended for other schedules.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The chosen scheduler.
    """

    if schedule_type == "cosine":
        # CosineAnnealingLR: Gradually decreases LR following a cosine curve.
        # Goes from base_lr down to eta_min over total_epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=0.0
        )

    elif schedule_type == "linear":
        # Linear decay from base_lr down to 0 over total_epochs
        def linear_decay(epoch):
            # epoch is zero-based; scale goes from 1.0 -> 0.0
            return 1.0 - float(epoch) / float(total_epochs - 1 if total_epochs > 1 else 1)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_decay)

    elif schedule_type == "warmup_constant":
        # 1) Warm-up linearly from 0 -> 1 over warmup_epochs
        # 2) Then remain constant (multiplier=1) after warmup.
        def warmup_then_constant(epoch):
            if epoch < warmup_epochs:
                # linear ramp from 0.0 -> 1.0
                return float(epoch + 1) / float(warmup_epochs)
            else:
                return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_constant)

    else:
        # Default/fallback: do no scheduling (constant LR).
        print(f"[Warning] Unknown schedule_type='{schedule_type}'. Using constant LR.")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    return scheduler
