# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    build_rms_mask,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
    rms_db,
    smooth_boundary_inplace,
)


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    fade_ms: int
    normalize_align: str
    emit_align: str
    drop_warmup_hops: bool
    vad_mode: str
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_aggressiveness: int
    vad_webrtc_frame_ms: int
    vad_webrtc_min_voiced_ratio: float
    gain_mode: str
    gain_target_delta_db: float
    gain_max_boost_db: float
    gain_smoothing: float
    mask_mode: str
    mask_db: float
    mask_frame_ms: float
    mask_smooth_ms: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    mmcxli_dir: str
    model_device: str
    seed: int
    ref_max_sec: float
    stream: Optional[StreamConfig] = None


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _resample_if_needed(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(
        np.float32, copy=False
    )


def _trim_wav(wav: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    if float(max_sec) <= 0:
        return wav
    max_samples = int(round(float(max_sec) * float(sr)))
    if max_samples <= 0:
        return wav
    return np.asarray(wav[:max_samples], dtype=np.float32).reshape(-1)


def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: name={name} path={path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mmcxli_utils_shim() -> types.ModuleType:
    # mmcxli's vc_engine.py imports:
    #   from utils import pred_contentvec_len, make_cross_extra_kernel, make_beep
    #
    # Amphion also has a top-level `utils/` package, so we inject a minimal shim to
    # satisfy the import without polluting the rest of the process.
    mod = types.ModuleType("utils")

    def pred_contentvec_len(length: int) -> int:
        return (int(length) - 80) // 320

    def make_cross_extra_kernel(
        blocksize: tuple = (1, 2048),
        extra: int = 128,
        divide: bool = False,
    ):
        assert extra > 0 and extra < blocksize[-1], "'extra' should be shorter than the blocksize"
        curve_length = int(extra) * 2
        time_array = np.linspace(0, np.pi, curve_length)

        if len(blocksize) == 2:
            sin_array_list = [time_array for _ in range(blocksize[0])]
        elif len(blocksize) == 3:
            sin_array_list = [[time_array for _ in range(blocksize[1])] for _ in range(blocksize[0])]
        elif len(blocksize) == 4:
            sin_array_list = [
                [[time_array for _ in range(blocksize[2])] for _ in range(blocksize[1])]
                for _ in range(blocksize[0])
            ]
        else:
            raise NotImplementedError(f"Unsupported blocksize rank: {len(blocksize)}")

        curve_array = np.sin(np.array(sin_array_list)) ** 2
        plateau_array = np.zeros(blocksize)[..., : (blocksize[-1] - extra)] + 1

        if not divide:
            return np.concatenate(
                [curve_array[..., :extra], plateau_array, curve_array[..., -extra:]],
                axis=-1,
            )
        current_half = np.concatenate([curve_array[..., :extra], plateau_array], axis=-1)
        previous_half = curve_array[..., -extra:]
        return [current_half, previous_half]

    def make_beep(
        sampling_freq: int = 44100,
        frequency: float = 440,
        duration: float = 1.0,
        beep_rate: float = 0.1,
        level: float = 0.2,
        n_channel: int = 1,
        channel_last: bool = True,
        dtype: np.dtype = np.float64,
    ):
        num_samples = int(round(float(sampling_freq) * float(duration)))
        if num_samples <= 0:
            out = np.zeros((0, n_channel), dtype=dtype)
            return out if channel_last else out.T
        t = np.linspace(0, float(duration), num_samples, endpoint=False)
        signal = float(level) * np.sin(2.0 * np.pi * float(frequency) * t)
        assert beep_rate > 0
        if float(beep_rate) < 1.0:
            signal[int(num_samples * float(beep_rate)) :] *= 0.0
        signal = np.stack([signal] * int(n_channel), axis=-1)
        if not channel_last:
            signal = signal.T
        return signal.astype(dtype)

    mod.pred_contentvec_len = pred_contentvec_len  # type: ignore[attr-defined]
    mod.make_cross_extra_kernel = make_cross_extra_kernel  # type: ignore[attr-defined]
    mod.make_beep = make_beep  # type: ignore[attr-defined]
    return mod


def _import_mmcxli_audio_efx(mmcxli_dir: str):
    root = Path(mmcxli_dir).resolve()
    vc_engine_py = root / "vc_engine.py"
    config_manager_py = root / "config_manager.py"
    if not vc_engine_py.is_file():
        raise FileNotFoundError(f"Missing mmcxli vc_engine.py: {vc_engine_py}")
    if not config_manager_py.is_file():
        raise FileNotFoundError(f"Missing mmcxli config_manager.py: {config_manager_py}")

    # Load config_manager under a private module name.
    cfg_mod = _load_module_from_path("_mmcxli_config_manager", config_manager_py)

    prev_utils = sys.modules.get("utils")
    sys.modules["utils"] = _mmcxli_utils_shim()
    try:
        vc_mod = _load_module_from_path("_mmcxli_vc_engine", vc_engine_py)
    finally:
        if prev_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prev_utils

    return vc_mod.AudioEfx, cfg_mod.load_make_vc_config


class _DummySoundControl:
    def __init__(
        self,
        *,
        sr_out: int,
        sr_proc: int,
        block_roll_size: int,
        content_expand_rate: float,
    ) -> None:
        self.sr_out = int(sr_out)
        self.sr_proc = int(sr_proc)
        self.block_roll_size = int(block_roll_size)
        self.blocksize = int(round(float(self.sr_out) * float(self.block_roll_size) * 0.02))
        self.n_ch_in_use = [1, 1, 1]
        self.content_expand_rate = float(content_expand_rate)
        self.current_target_style = np.zeros((1, 128), dtype=np.float32)


def _resolve_weight(path: Path, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ClassicVC/MMCXLI weight file: {path}\n"
            f"Expected it at: {name} (see scripts/vc_quest/classicvc_setup_gpu.sh)"
        )
    return str(path.resolve())


def _compute_style_embedding(audio_efx, ref_wav_16k: np.ndarray) -> np.ndarray:
    ref_wav_16k = np.asarray(ref_wav_16k, dtype=np.float32).reshape(1, -1)
    _, _, _, spec = audio_efx.sess_HarmoF0.run(
        ["freq_t", "act_t", "energy_t", "spec"],
        {"input": ref_wav_16k},
    )
    # Expect spec: (batch, 352, frames). Slice to match mmcxli Style Encoder usage.
    spec = np.asarray(spec, dtype=np.float32)
    spec = spec[:, 48:, :]
    if int(getattr(audio_efx, "len_style_encoder", 0)) > 0:
        n = int(min(int(audio_efx.len_style_encoder), spec.shape[-1]))
        spec = spec[:, :, -n:]

    spec_t4 = (spec.shape[-1] // 4) * 4
    if spec_t4 <= 0:
        return np.zeros((1, 128), dtype=np.float32)
    spec = spec[:, :, :spec_t4]
    style = audio_efx.sess_SE.run(
        ["output"],
        {"input": spec[:, np.newaxis, :, :]},
    )[0]
    return np.asarray(style, dtype=np.float32).reshape(1, -1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ClassicVC/MMCXLI one-shot VC runner (offline or streaming simulation)."
    )
    parser.add_argument("--mmcxli_dir", type=str, required=True, help="Path to MMCXLI repo checkout.")
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX Runtime device selection via MMCXLI config (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (reserved).")
    parser.add_argument(
        "--ref_max_sec",
        type=float,
        default=10.0,
        help="Trim reference audio to this many seconds (0 disables).",
    )
    parser.add_argument(
        "--absolute_pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use target-conditioned absolute pitch prediction (MMCXLI f0n predictor).",
    )
    parser.add_argument(
        "--estimate_energy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use target-conditioned energy prediction (MMCXLI f0n predictor).",
    )
    parser.add_argument(
        "--pitch_shift",
        type=float,
        default=0.0,
        help="Pitch shift in semitones (applied after f0 selection).",
    )
    parser.add_argument(
        "--content_expand_rate",
        type=float,
        default=0.1,
        help="Optional ContentVec tail expansion rate (0 disables; MMCXLI default is 0.1).",
    )

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument(
        "--window_ms",
        type=int,
        default=800,
        help="Streaming window size in ms. Use 0 to run one full-utterance window (equivalence test).",
    )
    parser.add_argument(
        "--hop_ms",
        type=int,
        default=400,
        help="Streaming hop size in ms. Use 0 to run one full-utterance hop (equivalence test).",
    )
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="center", choices=["start", "center", "end"])
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first full window is available (recommended for eval).",
    )
    parser.add_argument("--vad_mode", type=str, default="rms", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)

    parser.add_argument(
        "--gain_mode",
        type=str,
        default="off",
        choices=["off", "match_src_rms"],
        help="Optional loudness compensation for streaming stability.",
    )
    parser.add_argument(
        "--gain_target_delta_db",
        type=float,
        default=10.0,
        help="When gain_mode=match_src_rms, aim for output RMS ≈ (input RMS - gain_target_delta_db).",
    )
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)

    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    mmcxli_dir = os.path.abspath(str(args.mmcxli_dir))
    AudioEfx, load_make_vc_config = _import_mmcxli_audio_efx(mmcxli_dir)

    # Build an MMCXLI config dict without writing files.
    vc_cfg = load_make_vc_config("/tmp/mmcxli_vc_config.json", save=False)
    vc_cfg["model"]["model_device"] = str(args.model_device)
    vc_cfg["auto_encode"] = False
    vc_cfg["spec_rt_o"] = 2  # avoid GUI-only plotting paths
    vc_cfg["absolute_pitch"] = bool(args.absolute_pitch)
    vc_cfg["estimate_energy"] = bool(args.estimate_energy)
    vc_cfg["pitch_shift"] = float(args.pitch_shift)
    vc_cfg["content_expand_rate"] = float(args.content_expand_rate)

    # Resolve weights (MMCXLI expects relative paths by default).
    wdir = Path(mmcxli_dir) / "weights"
    vc_cfg["model"]["harmof0_ckpt"] = _resolve_weight(wdir / "harmof0.onnx", "harmof0.onnx")
    vc_cfg["model"]["CE_ckpt"] = _resolve_weight(wdir / "hubert500.onnx", "hubert500.onnx")
    vc_cfg["model"]["SE_ckpt"] = _resolve_weight(wdir / "style_encoder_304.onnx", "style_encoder_304.onnx")
    vc_cfg["model"]["f0n_ckpt"] = _resolve_weight(
        wdir / "f0n_predictor_hubert500.onnx", "f0n_predictor_hubert500.onnx"
    )
    vc_cfg["model"]["decoder_ckpt"] = _resolve_weight(wdir / "decoder_24k.onnx", "decoder_24k.onnx")
    vc_cfg["model"]["style_compressor_ckpt"] = _resolve_weight(
        wdir / "pumap_encoder_2dim.onnx", "pumap_encoder_2dim.onnx"
    )
    vc_cfg["model"]["style_decoder_ckpt"] = _resolve_weight(
        wdir / "pumap_decoder_2dim.onnx", "pumap_decoder_2dim.onnx"
    )

    in_sr = 16000
    out_sr = int(vc_cfg["backend"]["sr_decode"])

    sc = _DummySoundControl(
        sr_out=out_sr,
        sr_proc=in_sr,
        block_roll_size=int(vc_cfg["backend"]["block_roll_size"]),
        content_expand_rate=float(vc_cfg["content_expand_rate"]),
    )
    audio_efx = AudioEfx(sc=sc, vc_config=vc_cfg, hop_size=160, dim_spec=352, ch_map=[0], bypass=False)

    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)
    ref_wav = _trim_wav(ref_wav, ref_sr, float(args.ref_max_sec))

    ref_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)
    style = _compute_style_embedding(audio_efx, ref_16k)
    sc.current_target_style = np.asarray(style, dtype=np.float32).reshape(1, -1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0
    warmup_hops = 0

    if not bool(args.stream):
        t0 = time.time()
        out = audio_efx.convert_offline(src_16k[None, :])
        timings.append(time.time() - t0)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        sf.write(args.out, out, out_sr)
    else:
        # window_ms/hop_ms allow 0 as a sentinel meaning "use the full utterance length"
        # (useful for streaming-vs-offline equivalence tests without padding).
        if float(args.window_ms) > 0:
            window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
        else:
            window_in = int(len(src_16k))
        if float(args.hop_ms) > 0:
            hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
        else:
            hop_in = int(window_in)
        if window_in <= 0 or hop_in <= 0:
            raise ValueError("window_ms and hop_ms must be > 0 (or 0 to use full utterance)")
        if hop_in > window_in:
            raise ValueError("hop_ms must be <= window_ms")

        if float(args.window_ms) > 0:
            window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        else:
            window_out = int(round(float(window_in) * float(out_sr) / float(in_sr)))
        if float(args.hop_ms) > 0:
            hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        else:
            hop_out = int(round(float(hop_in) * float(out_sr) / float(in_sr)))
        fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

        if args.emit_align == "start":
            emit_start_out = 0
            emit_start_in = 0
        elif args.emit_align == "center":
            emit_start_out = max(0, (window_out - hop_out) // 2)
            emit_start_in = max(0, (window_in - hop_in) // 2)
        elif args.emit_align == "end":
            emit_start_out = max(0, window_out - hop_out)
            emit_start_in = max(0, window_in - hop_in)
        else:
            raise ValueError(f"Unknown emit_align: {args.emit_align}")

        ring = AudioRingBuffer(window_in)
        prev_last: Optional[float] = None
        outs: list[np.ndarray] = []
        drop_warmup_hops = bool(args.drop_warmup_hops)
        window_count = 0

        hop_ms_eff = float(args.hop_ms) if float(args.hop_ms) > 0 else (1000.0 * float(hop_in) / float(in_sr))
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(hop_ms_eff, 1e-6)))
        hangover_left = 0
        gain_db_state = 0.0

        for start in range(0, len(src_16k), hop_in):
            hop = src_16k[start : start + hop_in]
            if len(hop) < hop_in:
                hop = np.pad(hop, (0, hop_in - len(hop)), mode="constant")
            ring.write(hop)

            if ring.size < window_in:
                warmup_hops += 1
                prev_last = 0.0
                if not drop_warmup_hops:
                    outs.append(np.zeros(hop_out, dtype=np.float32))
                continue

            window = ring.read_last(window_in)
            vad_segment = window[emit_start_in : emit_start_in + hop_in]

            vad_mode = str(args.vad_mode)
            silent_rms = bool(
                float(args.vad_db) > -200.0
                and is_silent_rms_db(
                    vad_segment,
                    sample_rate=in_sr,
                    frame_ms=float(args.vad_frame_ms),
                    silence_db=float(args.vad_db),
                )
            )

            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not silent_rms
            elif vad_mode == "webrtc":
                webrtc_voiced = is_voiced_webrtcvad(
                    vad_segment,
                    sample_rate=in_sr,
                    frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                    aggressiveness=int(args.vad_webrtc_aggressiveness),
                    min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                )
                voiced = bool(webrtc_voiced) and (not silent_rms)
            else:
                raise ValueError(f"Unknown vad_mode: {vad_mode}")

            if not voiced and hangover_left > 0 and (not silent_rms):
                voiced = True
                hangover_left -= 1
            elif voiced:
                hangover_left = hangover_hops

            if not voiced:
                out_hop = np.zeros(hop_out, dtype=np.float32)
            else:
                t0 = time.time()
                out_window = audio_efx.convert_offline(window[None, :])
                timings.append(time.time() - t0)
                out_window = np.asarray(out_window, dtype=np.float32).reshape(-1)
                out_window = normalize_length(out_window, window_out, align=str(args.normalize_align))
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

            gain_mode = str(args.gain_mode)
            if voiced and gain_mode == "match_src_rms":
                alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                src_db = rms_db(vad_segment, eps=1e-9)
                out_db = rms_db(out_hop, eps=1e-9)
                desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                desired_boost_db = float(np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db)))
                gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                gain = float(10.0 ** (gain_db_state / 20.0))
                out_hop = (out_hop * gain).astype(np.float32, copy=False)
            elif not voiced:
                gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

            if str(args.mask_mode) == "rms":
                mask = build_rms_mask(
                    vad_segment,
                    in_sample_rate=in_sr,
                    out_sample_rate=out_sr,
                    out_len=hop_out,
                    frame_ms=float(args.mask_frame_ms),
                    threshold_db=float(args.mask_db),
                    smooth_ms=float(args.mask_smooth_ms),
                )
                out_hop = (out_hop * mask).astype(np.float32, copy=False)

            out_hop = smooth_boundary_inplace(out_hop, prev_last, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            prev_last = float(out_hop[-1]) if len(out_hop) else prev_last

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(args.out, out, out_sr)

        if bool(args.drop_warmup_hops):
            delay_samples = int(
                int(warmup_hops) * int(hop_out) + (int(hop_out) - int(window_out)) + int(emit_start_out)
            )
        else:
            delay_samples = int(emit_start_out)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                fade_ms=int(args.fade_ms),
                normalize_align=str(args.normalize_align),
                emit_align=str(args.emit_align),
                drop_warmup_hops=bool(args.drop_warmup_hops),
                vad_mode=str(args.vad_mode),
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                vad_hangover_ms=float(args.vad_hangover_ms),
                vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
                vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),
                vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                gain_mode=str(args.gain_mode),
                gain_target_delta_db=float(args.gain_target_delta_db),
                gain_max_boost_db=float(args.gain_max_boost_db),
                gain_smoothing=float(args.gain_smoothing),
                mask_mode=str(args.mask_mode),
                mask_db=float(args.mask_db),
                mask_frame_ms=float(args.mask_frame_ms),
                mask_smooth_ms=float(args.mask_smooth_ms),
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )

        cfg = RunConfig(
            mmcxli_dir=str(mmcxli_dir),
            model_device=str(args.model_device),
            seed=int(args.seed),
            ref_max_sec=float(args.ref_max_sec),
            stream=stream_cfg,
        )

        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": int(warmup_hops) if bool(args.stream) else 0,
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64))) if timings else 0.0,
            "p95_window_sec": float(np.percentile(np.asarray(timings, dtype=np.float64), 95))
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
        }
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
