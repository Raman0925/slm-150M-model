"""Tests for train.py — pure helpers for Phase 5 (data loading, LR schedule)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import train


def test_lr_at_step_linear_warmup():
    lr = train.lr_at_step(0, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1)
    assert lr == pytest.approx(1.0 * 1 / 10)


def test_lr_at_step_reaches_peak_at_end_of_warmup():
    lr = train.lr_at_step(9, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1)
    assert lr == pytest.approx(1.0)


def test_lr_at_step_continuous_at_warmup_boundary():
    lr = train.lr_at_step(10, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1)
    assert lr == pytest.approx(1.0)


def test_lr_at_step_cosine_midpoint():
    lr = train.lr_at_step(55, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.0)
    assert lr == pytest.approx(0.5, abs=1e-6)


def test_lr_at_step_min_lr_after_max_steps():
    lr = train.lr_at_step(100, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1)
    assert lr == pytest.approx(0.1)
    lr_beyond = train.lr_at_step(150, warmup_steps=10, max_steps=100, lr=1.0, min_lr=0.1)
    assert lr_beyond == pytest.approx(0.1)


def test_build_schedule_matches_config_defaults():
    # config.TRAIN defaults: micro_batch_size=32, global_batch_tokens=524288, seq_len=1024,
    # warmup_tokens=200_000_000 -> matches the real calibration run's printed schedule.
    sched = train.build_schedule(
        micro_batch_size=32, global_batch_tokens=524_288, seq_len=1024, warmup_tokens=200_000_000
    )
    assert sched.grad_accum_steps == 16
    assert sched.warmup_steps == 381
    assert sched.global_batch_size == 512


def test_build_schedule_smaller_micro_batch_keeps_global_batch_fixed():
    # Overriding micro_batch_size (e.g. for a smaller GPU) must not change the
    # effective global batch size — only grad_accum_steps should absorb the difference.
    sched = train.build_schedule(
        micro_batch_size=16, global_batch_tokens=524_288, seq_len=1024, warmup_tokens=200_000_000
    )
    assert sched.grad_accum_steps == 32
    assert sched.global_batch_size == 512


def test_load_windows_concatenates_and_reshapes_shards(tmp_path):
    seq_len = 4
    shard_a = np.arange(8, dtype=np.uint16)  # 2 complete windows
    shard_b = np.arange(8, 14, dtype=np.uint16)  # 1 complete window + 2 leftover tokens
    (tmp_path / "a.bin").write_bytes(shard_a.tobytes())
    (tmp_path / "b.bin").write_bytes(shard_b.tobytes())

    windows = train.load_windows(str(tmp_path), seq_len)

    assert windows.shape == (3, 4)  # 2 from shard a + 1 from shard b; 2 leftover tokens dropped
    assert windows.dtype == np.uint16
    np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(windows[1], [4, 5, 6, 7])
    np.testing.assert_array_equal(windows[2], [8, 9, 10, 11])


def test_load_windows_raises_when_no_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.load_windows(str(tmp_path), 1024)
