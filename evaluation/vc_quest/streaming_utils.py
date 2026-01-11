# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


NormalizeAlign = Literal["start", "end"]
VadFrameMs = Literal[10, 20, 30]


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


def crossfade_prefix_inplace(
    current: np.ndarray,
    prev_tail: Optional[np.ndarray],
    fade_len: int,
) -> np.ndarray:
    """Blend the start of `current` with the tail of the previous chunk.

    This is a stronger boundary smoother than `smooth_boundary_inplace` because it uses
    a full overlap region instead of only a single endpoint sample.
    """

    if prev_tail is None or fade_len <= 0:
        return current

    current = np.asarray(current, dtype=np.float32).reshape(-1)
    prev_tail = np.asarray(prev_tail, dtype=np.float32).reshape(-1)
    fade_len = int(min(fade_len, len(current), len(prev_tail)))
    if fade_len <= 0:
        return current

    a = prev_tail[-fade_len:]
    b = current[:fade_len]
    w = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    current[:fade_len] = (a * (1.0 - w) + b * w).astype(np.float32, copy=False)
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


def rms_db(
    wav: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """RMS loudness estimate in dBFS."""

    return _rms_db(wav, eps=eps)


def build_rms_mask(
    wav: np.ndarray,
    *,
    in_sample_rate: int,
    out_sample_rate: int,
    out_len: int,
    frame_ms: float = 10.0,
    threshold_db: float = -50.0,
    smooth_ms: float = 0.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Build a per-sample [0..1] mask (length=out_len) from input RMS frames.

    Intended to suppress output during input silence even when the model hallucinates audio.
    """

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(wav) == 0:
        return np.zeros(out_len, dtype=np.float32)

    if in_sample_rate <= 0 or out_sample_rate <= 0:
        raise ValueError("sample rates must be > 0")

    frame_in = int(round(float(frame_ms) / 1000.0 * float(in_sample_rate)))
    frame_out = int(round(float(frame_ms) / 1000.0 * float(out_sample_rate)))
    frame_in = max(1, frame_in)
    frame_out = max(1, frame_out)

    n = int(np.ceil(len(wav) / float(frame_in)))
    wav_p = np.pad(wav, (0, n * frame_in - len(wav)), mode="constant")
    frames = wav_p.reshape(n, frame_in)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + eps)
    db = 20.0 * np.log10(rms + eps)
    mask_frames = (db >= float(threshold_db)).astype(np.float32, copy=False)

    mask = np.repeat(mask_frames, frame_out).astype(np.float32, copy=False)
    if len(mask) < out_len:
        mask = np.pad(mask, (0, out_len - len(mask)), mode="constant")
    else:
        mask = mask[:out_len]

    smooth_len = int(round(float(smooth_ms) / 1000.0 * float(out_sample_rate)))
    if smooth_len > 1:
        kernel = np.ones(smooth_len, dtype=np.float32) / float(smooth_len)
        mask = np.convolve(mask, kernel, mode="same").astype(np.float32, copy=False)
        mask = np.clip(mask, 0.0, 1.0)

    return mask


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


def is_voiced_webrtcvad(
    wav: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: VadFrameMs = 30,
    aggressiveness: int = 2,
    min_voiced_ratio: float = 0.1,
) -> bool:
    """Voice activity detection using WebRTC VAD.

    Returns True if >= `min_voiced_ratio` of frames are voiced.
    If `webrtcvad` is unavailable, returns True (i.e., do not gate).
    """

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return False

    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("webrtcvad only supports 8k/16k/32k/48k sample rates")
    if frame_ms not in (10, 20, 30):
        raise ValueError("webrtcvad only supports 10/20/30ms frame sizes")

    try:
        import webrtcvad  # type: ignore
    except Exception:
        return True

    vad = webrtcvad.Vad(int(aggressiveness))
    frame = int(round(float(frame_ms) / 1000.0 * float(sample_rate)))
    frame = max(1, frame)

    # Pad to a whole number of frames (WebRTC VAD expects fixed size).
    n = int(np.ceil(len(wav) / float(frame)))
    wav_p = np.pad(wav, (0, n * frame - len(wav)), mode="constant")

    pcm16 = np.clip(wav_p, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False)

    voiced = 0
    for i in range(n):
        s = i * frame
        chunk = pcm16[s : s + frame]
        if vad.is_speech(chunk.tobytes(), sample_rate):
            voiced += 1

    return float(voiced) / float(n) >= float(min_voiced_ratio)


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
