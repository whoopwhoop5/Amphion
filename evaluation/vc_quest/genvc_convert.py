# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    apply_peak_limiter,
    is_silent_rms_db,
    is_voiced_webrtcvad,
)


@dataclass(frozen=True)
class StreamConfig:
    chunk_ms: int
    seg_len_sec: float
    stream_chunk_size: int
    overlap_len: int
    drop_warmup_chunks: bool
    vad_mode: str
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_aggressiveness: int
    vad_webrtc_frame_ms: int
    vad_webrtc_min_voiced_ratio: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    genvc_dir: str
    model_path: str
    device: str
    seed: int
    top_k: int
    stream: Optional[StreamConfig] = None


def _set_determinism(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _patch_sys_path(repo_dir: str) -> None:
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # GenVC has a top-level utils.py; Amphion has a top-level utils/ package.
    # Ensure we import GenVC's utils.py (and friends) by evicting Amphion's cached module if present.
    utils_mod = sys.modules.get("utils")
    if utils_mod is not None:
        mod_path = getattr(utils_mod, "__file__", "") or ""
        if mod_path and not os.path.abspath(mod_path).startswith(repo_dir):
            sys.modules.pop("utils", None)


def _import_genvc(repo_dir: str):
    _patch_sys_path(repo_dir)
    from inference.inference_utils import handle_chunks, synthesize_utt  # type: ignore[import-not-found]
    from inference.model_init import model_init  # type: ignore[import-not-found]
    from utils import load_audio  # type: ignore[import-not-found]

    return model_init, load_audio, synthesize_utt, handle_chunks


def _segment_is_voiced(
    segment: np.ndarray,
    *,
    sample_rate: int,
    vad_mode: str,
    vad_db: float,
    vad_frame_ms: float,
    vad_webrtc_aggressiveness: int,
    vad_webrtc_frame_ms: int,
    vad_webrtc_min_voiced_ratio: float,
) -> bool:
    if vad_mode == "off":
        return True
    if vad_mode == "rms":
        return not is_silent_rms_db(
            segment,
            sample_rate=sample_rate,
            frame_ms=float(vad_frame_ms),
            silence_db=float(vad_db),
        )
    if vad_mode == "webrtc":
        return is_voiced_webrtcvad(
            segment,
            sample_rate=sample_rate,
            frame_ms=int(vad_webrtc_frame_ms),  # type: ignore[arg-type]
            aggressiveness=int(vad_webrtc_aggressiveness),
            min_voiced_ratio=float(vad_webrtc_min_voiced_ratio),
        )
    raise ValueError(f"Unknown vad_mode: {vad_mode}")


@torch.inference_mode()
def _synthesize_streaming(
    *,
    model,
    handle_chunks,
    src_wav: torch.Tensor,
    ref_audio: torch.Tensor,
    seg_len_sec: float,
    stream_chunk_size: int,
    overlap_len: int,
    top_k: int,
    seed: int,
    vad_mode: str,
    vad_db: float,
    vad_frame_ms: float,
    vad_hangover_ms: float,
    vad_webrtc_aggressiveness: int,
    vad_webrtc_frame_ms: int,
    vad_webrtc_min_voiced_ratio: float,
    drop_warmup_chunks: bool,
    peak_limit: float,
) -> tuple[np.ndarray, int, dict]:
    _set_determinism(seed)

    device = model.device
    content_sr = int(model.content_sample_rate)
    out_sr = int(model.config.audio.sample_rate)

    model.config.top_k = int(top_k)

    seg_len = int(round(float(seg_len_sec) * float(content_sr)))
    seg_len = max(1, seg_len)
    min_chunk_duration = int(0.32 * float(content_sr))

    # Conditioning latent from reference (speaker/style).
    ref_audio = ref_audio.to(device)
    cond_latent = model.get_gpt_cond_latents(ref_audio, out_sr)

    src_wav = src_wav.to(device)
    total_wavlen = int(src_wav.shape[-1])

    timings_step: list[float] = []
    timings_chunk: list[float] = []
    emitted_samples: list[int] = []

    begin_time = time.time()
    first_emit_wall_sec: Optional[float] = None

    wav_gen_prev = None
    wav_overlap = None

    pred_chunks: list[np.ndarray] = []

    hangover_chunks = int(np.ceil(float(vad_hangover_ms) / max(seg_len_sec * 1000.0, 1e-6)))
    hangover_left = 0

    is_begin = True
    emitted_any = False

    for i in range(0, total_wavlen, seg_len):
        step_t0 = time.time()
        seg_end = min(i + seg_len, total_wavlen)
        src_wav_seg = src_wav[:, i:seg_end]
        if src_wav_seg.shape[-1] < min_chunk_duration:
            src_wav_seg = torch.nn.functional.pad(
                src_wav_seg,
                (0, min_chunk_duration - int(src_wav_seg.shape[-1])),
                mode="constant",
                value=0.0,
            )

        seg_np = src_wav_seg.detach().cpu().float().numpy().reshape(-1)
        voiced = _segment_is_voiced(
            seg_np,
            sample_rate=content_sr,
            vad_mode=vad_mode,
            vad_db=vad_db,
            vad_frame_ms=vad_frame_ms,
            vad_webrtc_aggressiveness=vad_webrtc_aggressiveness,
            vad_webrtc_frame_ms=vad_webrtc_frame_ms,
            vad_webrtc_min_voiced_ratio=vad_webrtc_min_voiced_ratio,
        )
        if voiced:
            hangover_left = hangover_chunks
        else:
            hangover_left = max(0, hangover_left - 1)
            voiced = hangover_left > 0

        if not voiced:
            # Reset overlap state across silence to avoid “ghost” bleed-through.
            wav_gen_prev = None
            wav_overlap = None

            out_len = int(round(float(seg_end - i) / float(content_sr) * float(out_sr)))
            if out_len <= 0:
                timings_step.append(time.time() - step_t0)
                continue

            if drop_warmup_chunks and not emitted_any:
                # Still in warmup region; skip emitting entirely.
                timings_step.append(time.time() - step_t0)
                continue

            silent = np.zeros(out_len, dtype=np.float32)
            pred_chunks.append(silent)
            emitted_samples.append(int(out_len))
            emitted_any = True
            timings_step.append(time.time() - step_t0)
            continue

        # Content feature extraction on this segment, then incremental GPT emission.
        # Stream in chunks of `stream_chunk_size` tokens.
        with torch.no_grad():
            content_feat = model.content_extractor.extract_content_features(src_wav_seg)
            content_codes = model.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
            gpt_inputs = model.gpt.compute_embeddings(cond_latent, content_codes)

            # Reset RNG per segment for deterministic output across chunking.
            _set_determinism(seed + (i // seg_len))

            gpt_generator = model.gpt.get_generator(
                fake_inputs=gpt_inputs,
                top_p=model.config.top_p,
                top_k=model.config.top_k,
                temperature=model.config.temperature,
                length_penalty=model.config.length_penalty,
                repetition_penalty=model.config.repetition_penalty,
                do_sample=True,
                num_beams=1,
                num_return_sequences=1,
                output_attentions=False,
                output_hidden_states=True,
            )

        last_tokens = []
        all_latents = []
        is_end = False

        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True

            if not is_end and stream_chunk_size > 0 and len(last_tokens) < int(stream_chunk_size):
                continue

            # Emit this chunk (or the final remainder).
            chunk_t0 = time.time()
            acoustic_latents = torch.cat(all_latents, dim=0)[None, :]
            mel_input = torch.nn.functional.interpolate(
                acoustic_latents.transpose(1, 2),
                scale_factor=[model.hifigan_scale_factor],
                mode="linear",
            ).squeeze(1)
            audio_pred = model.hifigan.forward(mel_input)
            wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                audio_pred.squeeze(), wav_gen_prev, wav_overlap, int(overlap_len)
            )
            timings_chunk.append(time.time() - chunk_t0)

            wav_np = wav_chunk.detach().cpu().float().numpy().reshape(-1)
            wav_np = apply_peak_limiter(wav_np, peak_limit=float(peak_limit))
            if drop_warmup_chunks and not emitted_any:
                # Do not emit audio until the first chunk is available (eval-friendly).
                emitted_any = True
                if first_emit_wall_sec is None:
                    first_emit_wall_sec = time.time() - begin_time
            else:
                pred_chunks.append(wav_np)
                emitted_samples.append(int(len(wav_np)))
                emitted_any = True
                if first_emit_wall_sec is None:
                    first_emit_wall_sec = time.time() - begin_time

            last_tokens = []
            all_latents = []

            if is_begin:
                is_begin = False

        timings_step.append(time.time() - step_t0)

    out = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.zeros(0, dtype=np.float32)
    total_time = time.time() - begin_time
    src_sec = float(total_wavlen) / float(content_sr)
    rtf = float(total_time) / max(src_sec, 1e-9)

    stats = {
        "content_sample_rate": int(content_sr),
        "deg_sample_rate": int(out_sr),
        "seg_len_sec": float(seg_len_sec),
        "stream_chunk_size": int(stream_chunk_size),
        "overlap_len": int(overlap_len),
        "first_emit_wall_sec": float(first_emit_wall_sec) if first_emit_wall_sec is not None else float("nan"),
        "rtf_total": float(rtf),
        "total_sec": float(total_time),
        "step_sec_mean": float(np.mean(timings_step)) if timings_step else float("nan"),
        "step_sec_p95": float(np.percentile(timings_step, 95)) if timings_step else float("nan"),
        "emit_samples_total": int(np.sum(emitted_samples)) if emitted_samples else 0,
        "emit_sec_total": float(np.sum(emitted_samples)) / float(out_sr) if emitted_samples else 0.0,
        "chunk_sec_mean": float(np.mean(timings_chunk)) if timings_chunk else float("nan"),
        "chunk_sec_p95": float(np.percentile(timings_chunk, 95)) if timings_chunk else float("nan"),
        "delay_samples": 0,
    }
    return out, out_sr, stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GenVC one-shot VC runner (offline or streaming simulation).")
    parser.add_argument("--genvc_dir", type=str, required=True, help="Path to GenVC repo checkout.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to GenVC checkpoint (.pth).")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base.")
    parser.add_argument("--top_k", type=int, default=1, help="GPT top-k (top_k=1 is deterministic).")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run GenVC streaming simulation.")
    parser.add_argument("--chunk_ms", type=int, default=1000, help="Streaming chunk size in ms (seg_len).")
    parser.add_argument("--stream_chunk_size", type=int, default=8, help="GPT tokens per emission in streaming mode.")
    parser.add_argument("--overlap_len", type=int, default=1024, help="Overlap samples for GenVC handle_chunks().")
    parser.add_argument(
        "--drop_warmup_chunks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop initial emitted chunk audio (eval-friendly).",
    )
    parser.add_argument("--vad_mode", type=str, default="webrtc", choices=["rms", "webrtc", "off"])
    parser.add_argument("--vad_db", type=float, default=-55.0)
    parser.add_argument("--vad_frame_ms", type=float, default=10.0)
    parser.add_argument("--vad_hangover_ms", type=float, default=200.0)
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    genvc_dir = os.path.abspath(str(args.genvc_dir))
    if not os.path.isdir(genvc_dir):
        raise FileNotFoundError(f"genvc_dir not found: {genvc_dir}")
    model_path = os.path.abspath(str(args.model_path))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")

    model_init, load_audio, synthesize_utt, _handle_chunks = _import_genvc(genvc_dir)

    # GenVC's model_init expects device string ("cuda"/"cpu"), not torch.device.
    device_str = str(device)

    # GenVC's configs often reference checkpoints via relative paths (e.g. `pre_trained/contentVec.pt`).
    # Ensure these resolve inside the GenVC repo checkout.
    prev_cwd = os.getcwd()
    try:
        os.chdir(genvc_dir)
        model, _ = model_init(model_path, device_str)
    finally:
        os.chdir(prev_cwd)
    model.config.top_k = int(args.top_k)

    src_wav = load_audio(args.src, int(model.content_sample_rate))
    ref_audio = load_audio(args.ref, int(model.config.audio.sample_rate))
    if src_wav is None:
        raise RuntimeError(f"GenVC failed to load src wav: {args.src}")
    if ref_audio is None:
        raise RuntimeError(f"GenVC failed to load ref wav: {args.ref}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    stats: dict = {"delay_samples": 0}
    out_sr = int(model.config.audio.sample_rate)

    if not args.stream:
        _set_determinism(int(args.seed))
        t0 = time.time()
        wav = synthesize_utt(model, src_wav, ref_audio, seg_len=6.0)
        timings.append(time.time() - t0)
        out = wav.detach().cpu().float().numpy().reshape(-1)
        out = apply_peak_limiter(out, peak_limit=float(args.peak_limit))
        sf.write(args.out, out, out_sr)
        stats = {
            "content_sample_rate": int(model.content_sample_rate),
            "deg_sample_rate": int(out_sr),
            "total_sec": float(timings[-1]) if timings else float("nan"),
            "rtf_total": float(timings[-1]) / max(float(len(src_wav.reshape(-1))) / float(model.content_sample_rate), 1e-9)
            if timings
            else float("nan"),
            "delay_samples": 0,
        }
    else:
        seg_len_sec = float(args.chunk_ms) / 1000.0
        out_np, out_sr, stream_stats = _synthesize_streaming(
            model=model,
            handle_chunks=_handle_chunks,
            src_wav=src_wav,
            ref_audio=ref_audio,
            seg_len_sec=seg_len_sec,
            stream_chunk_size=int(args.stream_chunk_size),
            overlap_len=int(args.overlap_len),
            top_k=int(args.top_k),
            seed=int(args.seed),
            vad_mode=str(args.vad_mode),
            vad_db=float(args.vad_db),
            vad_frame_ms=float(args.vad_frame_ms),
            vad_hangover_ms=float(args.vad_hangover_ms),
            vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
            vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),
            vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
            drop_warmup_chunks=bool(args.drop_warmup_chunks),
            peak_limit=float(args.peak_limit),
        )
        sf.write(args.out, out_np, out_sr)
        stats = stream_stats

    run_cfg = RunConfig(
        genvc_dir=genvc_dir,
        model_path=model_path,
        device=str(device),
        seed=int(args.seed),
        top_k=int(args.top_k),
        stream=StreamConfig(
            chunk_ms=int(args.chunk_ms),
            seg_len_sec=float(args.chunk_ms) / 1000.0,
            stream_chunk_size=int(args.stream_chunk_size),
            overlap_len=int(args.overlap_len),
            drop_warmup_chunks=bool(args.drop_warmup_chunks),
            vad_mode=str(args.vad_mode),
            vad_db=float(args.vad_db),
            vad_frame_ms=float(args.vad_frame_ms),
            vad_hangover_ms=float(args.vad_hangover_ms),
            vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
            vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),
            vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
            peak_limit=float(args.peak_limit),
        )
        if args.stream
        else None,
    )

    if args.meta_json:
        meta = {
            "config": asdict(run_cfg),
            "stats": stats,
        }
        Path(args.meta_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_json).write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
