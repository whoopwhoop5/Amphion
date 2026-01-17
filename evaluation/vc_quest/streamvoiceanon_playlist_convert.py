# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from evaluation.vc_quest.playlist import load_vc_playlist_manifest
from evaluation.vc_quest.streaming_utils import (
    apply_peak_limiter,
    build_rms_mask,
    rms_db,
    smooth_boundary_inplace,
)


def _set_determinism(seed: int) -> None:
    import torch

    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _torch_sync(device: "torch.device") -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _purge_import_cache(prefix: str) -> None:
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _remove_sys_path_entry(path: str) -> None:
    while path in sys.path:
        sys.path.remove(path)


@contextlib.contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_inference_wrapper(streamvoiceanon_dir: str):
    # StreamVoiceAnon assumes `torch._inductor.config` is available as an attribute.
    # In Torch 2.4, it exists as a submodule but isn't re-exported from `torch._inductor`.
    try:
        import torch  # type: ignore

        try:
            import torch._inductor  # type: ignore

            if not hasattr(torch._inductor, "config"):
                import torch._inductor.config as _inductor_config  # type: ignore

                torch._inductor.config = _inductor_config  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass

    root = Path(streamvoiceanon_dir).resolve()
    infer_py = root / "evaluations" / "infer_arvc.py"
    if not infer_py.is_file():
        raise FileNotFoundError(f"Missing StreamVoiceAnon infer script: {infer_py}")

    # StreamVoiceAnon defines a top-level `modules/` package that can conflict with Amphion's
    # `modules/`. Ensure StreamVoiceAnon is first on sys.path and purge `modules` imports.
    amphion_root = str(Path(__file__).resolve().parents[2])
    _remove_sys_path_entry("")
    _remove_sys_path_entry(amphion_root)
    _add_sys_path_first(str(root))
    _purge_import_cache("modules")

    name = "_streamvoiceanon_infer_arvc"
    if name in sys.modules:
        return sys.modules[name].InferenceWrapper  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(name, str(infer_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {infer_py}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod

    with _pushd(str(root)):
        spec.loader.exec_module(mod)

    if not hasattr(mod, "InferenceWrapper"):
        raise ImportError(f"StreamVoiceAnon infer module missing InferenceWrapper: {infer_py}")
    return mod.InferenceWrapper


def _load_audio_mono(path: str, *, sr: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=int(sr), mono=True)
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _case_id(source_id: str, target_id: str) -> str:
    return f"{source_id}__to__{target_id}"


@dataclass(frozen=True)
class _PromptCache:
    ref_audio_codes: "torch.Tensor"
    ref_content_codes: "torch.Tensor"
    style_vectors: "torch.Tensor"
    timbre_latents: "torch.Tensor"
    ref_wav_tensor: "torch.Tensor"


def _compute_prompt_cache(
    *,
    wrapper: Any,
    ref_wav_path: str,
    max_prompt_frames: int,
) -> _PromptCache:
    import torch

    ref_wav = _load_audio_mono(ref_wav_path, sr=int(wrapper.sr))
    ref_wav_tensor = torch.from_numpy(ref_wav).unsqueeze(0).to(wrapper.device)
    with torch.no_grad():
        ref_audio_codes, ref_content_codes, style_vectors, timbre_latents = wrapper.calculate_prompt(ref_wav_tensor)

    # Pre-trim the codes to reduce repeated slicing / memory.
    ref_audio_codes = ref_audio_codes[:, :, : int(max_prompt_frames)].contiguous()
    ref_content_codes = ref_content_codes[:, : int(max_prompt_frames)].contiguous()

    return _PromptCache(
        ref_audio_codes=ref_audio_codes,
        ref_content_codes=ref_content_codes,
        style_vectors=style_vectors,
        timbre_latents=timbre_latents,
        ref_wav_tensor=ref_wav_tensor[:, : int(max_prompt_frames) * 2048].contiguous(),
    )


def _prefill_prompt_from_cache(
    *,
    wrapper: Any,
    prompt: _PromptCache,
    delay_frames: int,
    autocast_ctx: contextlib.AbstractContextManager[None],
) -> None:
    import torch

    wrapper.ref_audio_codes = prompt.ref_audio_codes
    wrapper.ref_content_codes = prompt.ref_content_codes
    wrapper.style_vectors = prompt.style_vectors
    wrapper.timbre_latents = prompt.timbre_latents
    wrapper.ref_wav_tensor = prompt.ref_wav_tensor
    wrapper.ref_wav_tensor_len = int(prompt.ref_wav_tensor.size(-1))

    wrapper.delay = int(delay_frames)
    wrapper.model.set_delay(delay=int(delay_frames))

    with torch.no_grad(), autocast_ctx:
        wrapper.model.prefill_prompt(
            wrapper.ref_content_codes,
            wrapper.ref_audio_codes,
            wrapper.style_vectors,
            wrapper.timbre_latents,
        )


def _process_one_chunk(
    *,
    wrapper: Any,
    src_wav_chunk: "torch.Tensor",
    autocast_ctx: contextlib.AbstractContextManager[None],
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    chunk_len = int(src_wav_chunk.size(-1))
    if chunk_len <= 0:
        return src_wav_chunk

    # Slide the encoder window buffer.
    wrapper.src_wav_tensor[:, :-chunk_len] = wrapper.src_wav_tensor[:, chunk_len:].clone()
    wrapper.src_wav_tensor[:, -chunk_len:] = src_wav_chunk.clone()

    chunk_lens = torch.LongTensor([int(wrapper.src_wav_tensor.size(1))]).to(wrapper.device)
    with autocast_ctx:
        src_chunk_content_codes = wrapper.compiled_speech_tokenizer_encode(
            wrapper.src_wav_tensor.reshape(1, int(wrapper.encode_window_wave_lens)),
            chunk_lens,
        )[0].squeeze(0).clone()

    wrapper.src_content_codes = torch.cat(
        [wrapper.src_content_codes, src_chunk_content_codes[..., -int(wrapper.decode_chunk_frames) :]],
        dim=-1,
    )

    if int(wrapper.delay) > 0 and int(wrapper.src_content_codes.size(-1)) < int(wrapper.delay):
        return torch.zeros_like(src_wav_chunk)
    if int(wrapper.delay) > 0 and (not bool(wrapper.src_condition4delay_prefilled)):
        wrapper.model.prefill_src_condition4delay(wrapper.src_content_codes[:, -int(wrapper.delay) :])
        wrapper.src_condition4delay_prefilled = True
        return torch.zeros_like(src_wav_chunk)

    # Decode one or more frames.
    current_pos = None
    with autocast_ctx:
        for i in range(int(wrapper.decode_chunk_frames)):
            idx = -(int(wrapper.decode_chunk_frames) - int(i))
            vc_code, current_pos = wrapper.model.decode_one(
                src_chunk_content_codes[..., idx].unsqueeze(0),
            )
            wrapper.pred_codes = torch.cat([wrapper.pred_codes, vc_code.clone()[None]], dim=-1)

        if current_pos is not None and int(current_pos) // 2 >= int(wrapper.max_seq_frames):
            extended_ref_audio_codes = torch.cat(
                [wrapper.ref_audio_codes, wrapper.pred_codes[..., -int(wrapper.buffer_frames) :]],
                dim=-1,
            )
            extended_ref_content_codes = torch.cat(
                [
                    wrapper.ref_content_codes,
                    wrapper.src_content_codes[..., -int(wrapper.buffer_frames) - int(wrapper.delay) : -int(wrapper.delay)],
                ],
                dim=-1,
            )
            wrapper.model.prefill_prompt(
                extended_ref_content_codes,
                extended_ref_audio_codes,
                wrapper.style_vectors,
                wrapper.timbre_latents,
            )
            if int(wrapper.delay) > 0:
                wrapper.model.prefill_src_condition4delay(
                    wrapper.src_content_codes[..., -int(wrapper.delay) :],
                )

        # Decode with vocoder.
        vc_codes_chunk = wrapper.pred_codes[..., -int(wrapper.decode_window_frames) :]
        pad_len = int(wrapper.decode_window_frames) - int(vc_codes_chunk.size(-1))
        if pad_len > 0:
            vc_codes_chunk = F.pad(vc_codes_chunk, (pad_len, 0), value=0)

        pred_wave = wrapper.code2wav_fn(
            vc_codes_chunk.reshape(1, 8, int(wrapper.decode_window_frames)),
        ).clone()

    # Restrict kept history (matches upstream).
    wrapper.pred_codes = wrapper.pred_codes[..., -2048:]
    wrapper.src_content_codes = wrapper.src_content_codes[..., -2048:]

    return pred_wave[..., -2048 * int(wrapper.decode_chunk_frames) :].squeeze(1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run StreamVoiceAnon over a playlist manifest (streaming sim)."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--streamvoiceanon_dir",
        type=str,
        required=True,
        help="Path to StreamVoiceAnon repo checkout.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config_firefly_arvcasr_8192_delay0_8.yaml",
        help="Path inside StreamVoiceAnon repo (or absolute).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained_checkpoints/dual_ar_delay_0_8.pth",
        help="Path inside StreamVoiceAnon repo (or absolute).",
    )

    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    # StreamVoiceAnon streaming knobs (in codec frames, 1 frame = 2048 samples @ sr).
    parser.add_argument("--encode_window_frames", type=int, default=128)
    parser.add_argument("--decode_window_frames", type=int, default=64)
    parser.add_argument("--max_prompt_frames", type=int, default=256)
    parser.add_argument("--max_seq_frames", type=int, default=768)
    parser.add_argument("--buffer_frames", type=int, default=32)
    parser.add_argument("--decode_chunk_frames", type=int, default=1)
    parser.add_argument("--delay_frames", type=int, default=2)

    # Optional compile flags (torch.compile / compiled decode).
    parser.add_argument("--compile_ar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile_encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile_decoder", action=argparse.BooleanOptionalAction, default=False)

    # Wrapper-side postprocessing (optional).
    parser.add_argument("--fade_ms", type=float, default=10.0)
    parser.add_argument(
        "--gain_mode",
        type=str,
        default="off",
        choices=["off", "match_src_rms"],
    )
    parser.add_argument("--gain_target_delta_db", type=float, default=10.0)
    parser.add_argument("--gain_max_boost_db", type=float, default=18.0)
    parser.add_argument("--gain_smoothing", type=float, default=0.0)
    parser.add_argument("--mask_mode", type=str, default="off", choices=["off", "rms"])
    parser.add_argument("--mask_db", type=float, default=-50.0)
    parser.add_argument("--mask_frame_ms", type=float, default=10.0)
    parser.add_argument("--mask_smooth_ms", type=float, default=10.0)
    parser.add_argument("--peak_limit", type=float, default=0.99)

    args = parser.parse_args(argv)

    import torch

    streamvoiceanon_dir = str(Path(args.streamvoiceanon_dir).resolve())
    InferenceWrapper = _load_inference_wrapper(streamvoiceanon_dir)

    device_str = str(args.device).strip() if args.device else ""
    if not device_str:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    root = Path(streamvoiceanon_dir).resolve()
    config_path = Path(args.config_path)
    checkpoint_path = Path(args.checkpoint_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config_path: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint_path: {checkpoint_path}")

    # Instantiate once; reuse across all pairs.
    with _pushd(str(root)):
        wrapper = InferenceWrapper(
            str(config_path),
            str(checkpoint_path),
            compile_ar=bool(args.compile_ar),
            compile_encoder=bool(args.compile_encoder),
            compile_decoder=bool(args.compile_decoder),
        )
    wrapper.device = device
    wrapper.model.to(device)
    wrapper.speech_tokenizer.to(device)
    wrapper.firefly.to(device)
    wrapper.style_encoder.to(device)
    wrapper.timbre_encoder.to(device)

    sr = int(wrapper.sr)
    hop_samples = 2048 * int(args.decode_chunk_frames)
    hop_ms = float(hop_samples) / float(sr) * 1000.0
    delay_samples = int(int(args.delay_frames) * int(hop_samples))

    # For latency scoring, encode the model delay into the "window_ms" (see score_playlist).
    stream_window_ms = float(int(args.delay_frames) + 1) * float(hop_ms)

    fade_len = int(round(float(args.fade_ms) / 1000.0 * float(sr)))
    autocast_ctx: contextlib.AbstractContextManager[None]
    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = contextlib.nullcontext()

    manifest = load_vc_playlist_manifest(args.manifest).resolve_paths(args.manifest)
    sources = manifest.sources_by_id()
    targets = manifest.targets_by_id()
    pairs = list(manifest.pairs)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    out_root = Path(args.out_dir)
    wav_dir = out_root / "wavs"
    meta_dir = out_root / "meta"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Cache prompts per target (10 total in the default FLEURS playlist).
    prompt_cache: dict[str, _PromptCache] = {}

    for pair_idx, pair in enumerate(pairs):
        s = sources[pair.source_id]
        t = targets[pair.target_id]
        cid = _case_id(pair.source_id, pair.target_id)

        out_wav = wav_dir / f"{cid}.wav"
        out_meta = meta_dir / f"{cid}.json"
        if bool(args.resume) and out_wav.exists() and out_meta.exists():
            continue

        if pair.target_id not in prompt_cache:
            _set_determinism(int(args.seed))
            prompt_cache[pair.target_id] = _compute_prompt_cache(
                wrapper=wrapper,
                ref_wav_path=t.wav_path,
                max_prompt_frames=int(args.max_prompt_frames),
            )

        _set_determinism(int(args.seed) + int(pair_idx))
        _prefill_prompt_from_cache(
            wrapper=wrapper,
            prompt=prompt_cache[pair.target_id],
            delay_frames=int(args.delay_frames),
            autocast_ctx=autocast_ctx,
        )

        wrapper.setup_stream_caches(
            encode_window_frames=int(args.encode_window_frames),
            decode_window_frames=int(args.decode_window_frames),
            max_seq_frames=int(args.max_seq_frames),
            buffer_frames=int(args.buffer_frames),
            decode_chunk_frames=int(args.decode_chunk_frames),
            delay=int(args.delay_frames),
        )

        # Load + right-pad source to full chunks.
        src_wav = _load_audio_mono(s.wav_path, sr=sr)
        pad_end = (-len(src_wav)) % hop_samples
        if pad_end:
            src_wav = np.pad(src_wav, (0, pad_end), mode="constant")
        src_wav_tensor = torch.from_numpy(src_wav).unsqueeze(0).to(device)

        num_chunks = int(len(src_wav) // hop_samples) if hop_samples > 0 else 0
        timings: list[float] = []
        out_chunks: list[np.ndarray] = []

        for chunk_idx in range(num_chunks):
            start = int(chunk_idx * hop_samples)
            chunk = src_wav_tensor[:, start : start + hop_samples]
            if chunk.size(-1) < hop_samples:
                chunk = torch.nn.functional.pad(chunk, (0, hop_samples - int(chunk.size(-1))))

            _torch_sync(device)
            t0 = time.time()
            out_chunk_t = _process_one_chunk(
                wrapper=wrapper,
                src_wav_chunk=chunk,
                autocast_ctx=autocast_ctx,
            )
            _torch_sync(device)
            timings.append(time.time() - t0)

            out_chunks.append(out_chunk_t.squeeze(0).detach().cpu().float().numpy())

        # Drop warmup chunks to align output timeline to source for scoring.
        warmup_hops = int(max(0, int(args.delay_frames)))
        out_chunks = out_chunks[warmup_hops:] if warmup_hops else out_chunks

        # Postprocess (optional): boundary smoothing + gain/mask + limiter.
        gain_db_state = 0.0
        processed: list[np.ndarray] = []
        prev_last: Optional[float] = 0.0
        for i, out_chunk in enumerate(out_chunks):
            out_chunk = out_chunk.astype(np.float32, copy=False)

            # Align to the *emitted* timeline: output chunk i corresponds to source at (i+warmup)*hop.
            src_start = int((i + warmup_hops) * hop_samples)
            src_seg = src_wav[src_start : src_start + hop_samples]
            if len(src_seg) < hop_samples:
                src_seg = np.pad(src_seg, (0, hop_samples - len(src_seg)), mode="constant")

            if fade_len > 0:
                out_chunk = smooth_boundary_inplace(out_chunk, prev_last, fade_len)

            if str(args.gain_mode) == "match_src_rms":
                alpha = float(np.clip(float(args.gain_smoothing), 0.0, 0.999))
                src_db = rms_db(src_seg, eps=1e-9)
                out_db = rms_db(out_chunk, eps=1e-9)
                desired_boost_db = (src_db - out_db) - float(args.gain_target_delta_db)
                desired_boost_db = float(
                    np.clip(desired_boost_db, 0.0, float(args.gain_max_boost_db))
                )
                gain_db_state = alpha * gain_db_state + (1.0 - alpha) * desired_boost_db
                gain = float(10.0 ** (gain_db_state / 20.0))
                out_chunk = (out_chunk * gain).astype(np.float32, copy=False)
            else:
                gain_db_state *= float(np.clip(float(args.gain_smoothing), 0.0, 0.999))

            if str(args.mask_mode) == "rms":
                mask = build_rms_mask(
                    src_seg,
                    in_sample_rate=sr,
                    out_sample_rate=sr,
                    out_len=hop_samples,
                    frame_ms=float(args.mask_frame_ms),
                    threshold_db=float(args.mask_db),
                    smooth_ms=float(args.mask_smooth_ms),
                )
                out_chunk = (out_chunk * mask).astype(np.float32, copy=False)

            out_chunk = apply_peak_limiter(out_chunk, peak_limit=float(args.peak_limit))
            prev_last = float(out_chunk[-1]) if len(out_chunk) else prev_last
            processed.append(out_chunk)

        out = np.concatenate(processed) if processed else np.zeros(0, dtype=np.float32)
        sf.write(str(out_wav), out, sr)

        cfg: dict[str, Any] = {
            "streamvoiceanon_dir": str(root),
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "seed": int(args.seed),
            "compile_ar": bool(args.compile_ar),
            "compile_encoder": bool(args.compile_encoder),
            "compile_decoder": bool(args.compile_decoder),
            "stream": {
                "encode_window_frames": int(args.encode_window_frames),
                "decode_window_frames": int(args.decode_window_frames),
                "max_prompt_frames": int(args.max_prompt_frames),
                "max_seq_frames": int(args.max_seq_frames),
                "buffer_frames": int(args.buffer_frames),
                "decode_chunk_frames": int(args.decode_chunk_frames),
                "delay_frames": int(args.delay_frames),
                # scoring fields:
                "window_ms": float(stream_window_ms),
                "hop_ms": float(hop_ms),
                "emit_align": "start",
                "drop_warmup_hops": True,
                # wrapper fields:
                "fade_ms": float(args.fade_ms),
                "gain_mode": str(args.gain_mode),
                "gain_target_delta_db": float(args.gain_target_delta_db),
                "gain_max_boost_db": float(args.gain_max_boost_db),
                "gain_smoothing": float(args.gain_smoothing),
                "mask_mode": str(args.mask_mode),
                "mask_db": float(args.mask_db),
                "mask_frame_ms": float(args.mask_frame_ms),
                "mask_smooth_ms": float(args.mask_smooth_ms),
                "peak_limit": float(args.peak_limit),
            },
        }

        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": int(warmup_hops),
            "mean_window_sec": float(np.mean(np.asarray(timings, dtype=np.float64)))
            if timings
            else 0.0,
            "p95_window_sec": float(
                np.percentile(np.asarray(timings, dtype=np.float64), 95)
            )
            if len(timings) >= 2
            else (float(timings[0]) if timings else 0.0),
            "windows": int(len(timings)),
            "out_chunks": int(len(out_chunks)),
        }

        out_meta.write_text(json.dumps({"config": cfg, "stats": stats}, indent=2))

    meta = {
        "manifest": str(Path(args.manifest).resolve()),
        "model": "streamvoiceanon",
        "stream": True,
        "out_sample_rate": int(sr),
        "out_dir": str(out_root.resolve()),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[streamvoiceanon_playlist] Wrote: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
