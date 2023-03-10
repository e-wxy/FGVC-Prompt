# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch.optim.lr_scheduler import _LRScheduler
import math
from .misc import make_params_dict


def build_scheduler(train_cfg, optimizer, n_iter_per_epoch):
    num_steps = int(train_cfg.MAX_EPOCHS * n_iter_per_epoch)
    warmup_steps = int(train_cfg.WARMUP_EPOCHS * n_iter_per_epoch)
    scheduler_params = make_params_dict(train_cfg.SCHEDULER.PARAMS)

    lr_scheduler = None
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py#L18
    if train_cfg.SCHEDULER.NAME == 'warm_cosine':
        if scheduler_params is None:
            lr_scheduler = WarmupCosineLR(
                optimizer, 
                max_steps=num_steps,
                warmup_steps=warmup_steps,
            )
        else:
            lr_scheduler = WarmupCosineLR(
                optimizer, 
                max_steps=num_steps,
                warmup_steps=warmup_steps,
                **scheduler_params
            )
    elif train_cfg.SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=train_cfg.MIN_LR,
            warmup_lr_init=train_cfg.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif train_cfg.SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=train_cfg.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif train_cfg.SCHEDULER.NAME == 'step':
        decay_steps = int(train_cfg.SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=train_cfg.SCHEDULER.DECAY_RATE,
            warmup_lr_init=train_cfg.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class WarmupCosineLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            max_steps=10000,
            min_factor=1e-4,
            warmup_steps=1000,
            warmup_factor=0.1,
            warmup_method='linear',
            last_epoch=-1,
    ):

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )

        self.max_steps = max_steps
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.min_factor = min_factor
        self.warmup_method = warmup_method
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        factor = 1

        if self.last_epoch < self.warmup_steps:
            if self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_steps
                factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                factor = self.warmup_factor
        else:
            progress = float(self.last_epoch - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
            factor = self.min_factor + (1 - self.min_factor) * (1. + math.cos(math.pi * progress)) / 2

        return [base_lr * factor for base_lr in self.base_lrs]