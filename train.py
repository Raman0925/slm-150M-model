"""Pure helpers for Phase 5 pretraining (data loading, LR schedule, batching)."""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass

import numpy as np


def load_windows(tokens_dir: str, seq_len: int) -> np.ndarray:
    """Load every *.bin shard in tokens_dir and stack into (N, seq_len) uint16."""
    paths = sorted(glob.glob(f"{tokens_dir}/*.bin"))
    if not paths:
        raise FileNotFoundError(f"no .bin shards found in {tokens_dir}")
    parts = []
    for p in paths:
        arr = np.fromfile(p, dtype=np.uint16)
        n = arr.size // seq_len
        parts.append(arr[: n * seq_len].reshape(n, seq_len))
    return np.concatenate(parts, axis=0)


def lr_at_step(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


@dataclass(frozen=True)
class Schedule:
    micro_batch_size: int
    grad_accum_steps: int
    warmup_steps: int

    @property
    def global_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


def build_schedule(*, micro_batch_size: int, global_batch_tokens: int, seq_len: int,
                    warmup_tokens: int) -> Schedule:
    micro_tokens = micro_batch_size * seq_len
    grad_accum_steps = max(1, round(global_batch_tokens / micro_tokens))
    warmup_steps = max(1, warmup_tokens // (grad_accum_steps * micro_tokens))
    return Schedule(micro_batch_size, grad_accum_steps, warmup_steps)
