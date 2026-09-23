import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class WarmupLinearDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return [self.min_lr + (base_lr - self.min_lr) * (1 - progress) for base_lr in self.base_lrs]


class WarmupStableDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, decay_fraction, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.stable_end = total_steps - self.decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        if step < self.stable_end:
            return list(self.base_lrs)
        progress = (step - self.stable_end) / max(1, self.decay_steps)
        progress = min(1.0, max(0.0, progress))
        return [self.min_lr + (base_lr - self.min_lr) * (1 - progress) for base_lr in self.base_lrs]


class ConstantScheduler(_LRScheduler):
    def get_lr(self):
        return list(self.base_lrs)


def build_scheduler(optimizer, lr_schedule, warmup_steps, total_steps):
    if lr_schedule == "cos":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    if lr_schedule in ["linear", "lin"]:
        return WarmupLinearDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    if lr_schedule in ["constant", "const", "flat"]:
        return ConstantScheduler(optimizer)
    if lr_schedule.startswith("wsd"):
        decay_fraction = float(lr_schedule[3:])
        return WarmupStableDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_fraction=decay_fraction,
        )
    raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def set_scheduler_to_completed_steps(scheduler, completed_steps):
    scheduler.last_epoch = completed_steps
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = completed_steps + 1
    lrs = scheduler.get_lr()
    for param_group, lr in zip(scheduler.optimizer.param_groups, lrs):
        param_group["lr"] = lr
    scheduler._last_lr = list(lrs)
