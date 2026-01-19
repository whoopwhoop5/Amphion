# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import threading
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


def _ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    frac = Fraction(int(dst_sr), int(src_sr)).limit_denominator(1000)
    return int(frac.numerator), int(frac.denominator)


def resample_audio(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(src_sr) == int(dst_sr):
        return x.astype(np.float32, copy=False)
    up, down = _ratio(int(src_sr), int(dst_sr))
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def normalize_len_end(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[-n:]
    return np.pad(x, (n - len(x), 0), mode="constant")


class AudioRingBuffer:
    def __init__(self, capacity: int):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._size = 0

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
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if n > self._size:
            raise ValueError(f"Requested n={n}, but size={self._size}")

        start = (self._write_pos - n) % self._capacity
        if start < self._write_pos:
            return self._buf[start : self._write_pos].copy()
        return np.concatenate([self._buf[start:], self._buf[: self._write_pos]]).copy()


class OutputRingBuffer:
    def __init__(self, capacity: int):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
        self._lock = threading.Lock()

        self.underflows = 0
        self.overflows = 0

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return

        with self._lock:
            if n > self._capacity:
                x = x[-self._capacity :]
                n = len(x)

            free = self._capacity - self._size
            if n > free:
                drop = n - free
                self._read_pos = (self._read_pos + drop) % self._capacity
                self._size -= drop
                self.overflows += 1

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = x
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = x[:first]
                self._buf[: end % self._capacity] = x[first:]

            self._write_pos = end % self._capacity
            self._size += n

    def read(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            if self._size < n:
                self.underflows += 1
                out = np.zeros(n, dtype=np.float32)
                avail = int(self._size)
                if avail > 0:
                    data = self._read_no_lock(avail)
                    out[-avail:] = data
                return out
            return self._read_no_lock(n)

    def _read_no_lock(self, n: int) -> np.ndarray:
        end = self._read_pos + n
        if end <= self._capacity:
            out = self._buf[self._read_pos : end].copy()
        else:
            first = self._capacity - self._read_pos
            out = np.concatenate([self._buf[self._read_pos :], self._buf[: end % self._capacity]]).copy()
        self._read_pos = end % self._capacity
        self._size -= n
        return out

    def size(self) -> int:
        with self._lock:
            return int(self._size)


def parse_device_arg(v: Optional[str]) -> Optional[object]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s.lower() in {"default", "auto"}:
        return None
    return s


def print_device_help(sd) -> None:  # noqa: ANN001
    try:
        devices = list(sd.query_devices())
    except Exception as e:  # pragma: no cover
        print(f"Failed to query devices: {e}")
        return

    try:
        hostapis = list(sd.query_hostapis())
    except Exception:
        hostapis = []

    hostapi_names: dict[int, str] = {}
    for i, ha in enumerate(hostapis):
        try:
            hostapi_names[i] = str(ha.get("name", "")).strip()
        except Exception:
            hostapi_names[i] = ""

    default_in: Optional[int] = None
    default_out: Optional[int] = None
    try:
        default_in, default_out = sd.default.device  # type: ignore[misc]
    except Exception:
        pass

    def fmt_line(idx: int, dev: dict, *, kind: str) -> str:
        name = str(dev.get("name", "")).strip()
        hostapi = hostapi_names.get(int(dev.get("hostapi", -1)), "").strip()
        max_in = int(dev.get("max_input_channels", 0) or 0)
        max_out = int(dev.get("max_output_channels", 0) or 0)
        sr = dev.get("default_samplerate", None)
        try:
            sr_s = f"{int(round(float(sr)))}" if sr is not None else "?"
        except Exception:
            sr_s = "?"

        mark = " "
        if kind == "in" and default_in is not None and idx == int(default_in):
            mark = ">"
        if kind == "out" and default_out is not None and idx == int(default_out):
            mark = "<"

        caps = f"in={max_in} out={max_out} sr={sr_s}"
        ha = f" [{hostapi}]" if hostapi else ""
        return f"{mark} {idx:>3} {name}{ha} ({caps})"

    inputs: list[str] = []
    outputs: list[str] = []
    duplex: list[str] = []

    for idx, dev in enumerate(devices):
        try:
            max_in = int(dev.get("max_input_channels", 0) or 0)
            max_out = int(dev.get("max_output_channels", 0) or 0)
        except Exception:
            continue

        if max_in > 0:
            inputs.append(fmt_line(idx, dev, kind="in"))
        if max_out > 0:
            outputs.append(fmt_line(idx, dev, kind="out"))
        if max_in > 0 and max_out > 0:
            duplex.append(fmt_line(idx, dev, kind="in"))

    print("Input devices (use --input_device):")
    if inputs:
        print("\n".join(inputs))
    else:
        print("  (none)")

    print("\nOutput devices (use --output_device):")
    if outputs:
        print("\n".join(outputs))
    else:
        print("  (none)")

    if duplex:
        print("\nDuplex devices (input+output):")
        print("\n".join(duplex))

    print("\nNotes:")
    print("- Pass a device index (e.g. --input_device 4) or exact device name.")
    print("- Use --input_device default / --output_device default (or omit) to use system defaults.")
