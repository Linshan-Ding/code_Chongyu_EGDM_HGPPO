"""Single output-preserving adaptive normalized value head (PopArt)."""

import torch
from torch import nn


class PopArt(nn.Module):
    def __init__(self, dimension, beta=0.01):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)
        self.beta = beta
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("second", torch.tensor(1.0))
        self.register_buffer("std", torch.tensor(1.0))
        self.register_buffer("updates", torch.tensor(0))

    def forward(self, x):
        return self.linear(x).squeeze(-1).float()

    def denormalize(self, value):
        return value * self.std + self.mean

    def normalize(self, value):
        return (value - self.mean) / self.std

    @torch.no_grad()
    def update(self, targets, optimizer=None):
        targets = targets.float()
        if not torch.isfinite(targets).all():
            raise ValueError("nonfinite PopArt targets")
        old_mean, old_std = self.mean.clone(), self.std.clone()
        beta = 1.0 if self.updates == 0 else self.beta
        self.mean.lerp_(targets.mean(), beta)
        self.second.lerp_(targets.square().mean(), beta)
        self.std.copy_((self.second - self.mean.square()).clamp_min(1e-4).sqrt())
        ratio = old_std / self.std
        self.linear.weight.mul_(ratio)
        self.linear.bias.copy_(
            (old_std * self.linear.bias + old_mean - self.mean) / self.std
        )
        self.updates.add_(1)
        # Rescale Adam's moments in the changed output coordinate system.
        if optimizer is not None:
            for p in self.linear.parameters():
                state = optimizer.state.get(p, {})
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if name in state:
                        state[name].mul_(ratio if name == "exp_avg" else ratio.square())
