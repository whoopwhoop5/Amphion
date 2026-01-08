# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


NormalizeAlign = Literal["start", "end"]


def normalize_length(
    wav: np.ndarray,
    target_len: int,
    *,
    align: NormalizeAlign = "end",
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == target_len:
        return wav

    if len(wav) > target_len:
        if align == "start":
            return wav[:target_len]
        if align == "end":
            return wav[-target_len:]
        raise ValueError(f"Unknown align: {align}")

    pad = target_len - len(wav)
    if align == "start":
        return np.pad(wav, (0, pad), mode="constant")
    if align == "end":
        return np.pad(wav, (pad, 0), mode="constant")
    raise ValueError(f"Unknown align: {align}")


def smooth_boundary_inplace(
    current: np.ndarray,
    prev_last: Optional[float],
    fade_len: int,
) -> np.ndarray:
    if prev_last is None or fade_len <= 0:
        return current

    current = np.asarray(current, dtype=np.float32).reshape(-1)
    fade_len = min(fade_len, len(current))
    if fade_len <= 0:
        return current

    delta = float(prev_last) - float(current[0])
    if not np.isfinite(delta):
        return current

    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    current[:fade_len] = (current[:fade_len] + delta * fade).astype(np.float32, copy=False)
    return current


def _rms_db(
    wav: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(wav * wav) + eps))
    return 20.0 * float(np.log10(rms + eps))


def is_silent_rms_db(
    wav: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float = 10.0,
    silence_db: float = -60.0,
    percentile: float = 95.0,
    eps: float = 1e-9,
) -> bool:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return True

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    frame = int(round(float(frame_ms) / 1000.0 * float(sample_rate)))
    frame = max(1, frame)

    n = len(wav) // frame
    if n <= 0:
        return _rms_db(wav, eps=eps) < float(silence_db)

    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + eps)
    db = 20.0 * np.log10(rms + eps)
    return float(np.percentile(db, percentile)) < float(silence_db)


def apply_peak_limiter(
    wav: np.ndarray,
    *,
    peak_limit: float = 0.99,
    eps: float = 1e-9,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if peak_limit <= 0:
        return wav
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if not np.isfinite(peak) or peak <= peak_limit or peak <= eps:
        return wav
    gain = float(peak_limit) / peak
    return (wav * gain).astype(np.float32, copy=False)


class AudioRingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return

        if n >= self._capacity:
            x = x[-self._capacity :]
            n = len(x)

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos : end] = x
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos :] = x[:first]
            self._buf[: end % self._capacity] = x[first:]

        self._write_pos = end % self._capacity
        self._size = min(self._capacity, self._size + n)

    def read_last(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if n > self._size:
            raise ValueError(f"Requested n={n}, but size={self._size}")

        start = (self._write_pos - n) % self._capacity
        if start < self._write_pos:
            return self._buf[start : self._write_pos].copy()
        return np.concatenate([self._buf[start:], self._buf[: self._write_pos]]).copy()

