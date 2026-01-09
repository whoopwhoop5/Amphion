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
from typing import Any, Literal, Optional

import numpy as np
import soundfile as sf

from evaluation.vc_quest.streaming_utils import (
    VadFrameMs,
    apply_peak_limiter,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
)
from evaluation.vevo_live.common import write_wav


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _add_sys_paths_first(paths: list[str]) -> None:
    # Insert in reverse so the first entry ends up at sys.path[0].
    for p in reversed(paths):
        _add_sys_path_first(p)


def _set_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_audio_mono(path: str, *, sr: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _extract_fbanks(samples: np.ndarray, *, sample_rate: int, frame_shift_ms: float = 10.0) -> "torch.Tensor":
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav = np.asarray(samples, dtype=np.float32).reshape(-1)
    wav = wav * float(1 << 15)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav_t,
        frame_length=25,
        frame_shift=float(frame_shift_ms),
        snip_edges=True,
        num_mel_bins=80,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=float(sample_rate),
    )
    return fbanks.unsqueeze(0)


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    fade_ms: int
    normalize_align: Literal["start", "end"]
    emit_align: Literal["start", "center", "end"]
    drop_warmup_hops: bool
    vad_mode: Literal["rms", "webrtc", "off"]
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_frame_ms: VadFrameMs
    vad_webrtc_aggressiveness: int
    vad_webrtc_min_voiced_ratio: float
    peak_limit: float


class MeanVCStreamer:
    def __init__(
        self,
        *,
        meanvc_dir: str,
        device: "torch.device",
        steps: int,
        seed: int,
        reference_wav_16k: np.ndarray,
    ) -> None:
        import torch

        self.device = device
        self.sample_rate = 16000
        self.steps = int(steps)
        self.seed = int(seed)

        if self.steps == 1:
            self.timesteps = torch.tensor([1.0, 0.0], device=self.device)
        elif self.steps == 2:
            self.timesteps = torch.tensor([1.0, 0.8, 0.0], device=self.device)
        else:
            self.timesteps = torch.linspace(1.0, 0.0, self.steps + 1, device=self.device)

        root = Path(meanvc_dir).resolve()
        self.model_config_path = root / "src" / "config" / "config_200ms.json"
        self.model_ckpt_path = root / "src" / "ckpt" / "model_200ms.safetensors"
        self.asr_ckpt_path = root / "src" / "ckpt" / "fastu2++.pt"
        self.vocoder_ckpt_path = root / "src" / "ckpt" / "vocos.pt"
        self.sv_ckpt_path = root / "src" / "runtime" / "speaker_verification" / "ckpt" / "wavlm_large_finetune.pth"

        for p in [
            self.model_config_path,
            self.model_ckpt_path,
            self.asr_ckpt_path,
            self.vocoder_ckpt_path,
            self.sv_ckpt_path,
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Missing MeanVC file: {p}")

        with self.model_config_path.open("r") as f:
            model_config = json.load(f)

        from src.infer.dit_kvcache import DiT  # type: ignore[import-not-found]
        from src.model.utils import load_checkpoint  # type: ignore[import-not-found]
        from src.runtime.speaker_verification.verification import (  # type: ignore[import-not-found]
            init_model as init_sv_model,
        )
        from src.infer.infer_ref import MelSpectrogramFeatures  # type: ignore[import-not-found]

        self.vc = DiT(**model_config["model"])
        # Keep float32 for stability/determinism; some TorchScript components may not support fp16 well.
        self.vc = load_checkpoint(self.vc, str(self.model_ckpt_path), device=str(self.device), use_ema=False).float()
        self.vc.eval()

        self.asr = torch.jit.load(str(self.asr_ckpt_path)).to(self.device)
        self.vocoder = torch.jit.load(str(self.vocoder_ckpt_path)).to(self.device)

        self.sv = init_sv_model("wavlm_large", str(self.sv_ckpt_path)).to(self.device)
        self.sv.eval()

        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=self.sample_rate,
            n_fft=1024,
            win_size=640,
            hop_length=160,
            n_mels=80,
            fmin=0,
            fmax=8000,
            center=True,
        ).to(self.device)

        ref_t = torch.from_numpy(np.asarray(reference_wav_16k, dtype=np.float32).reshape(-1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.spk_emb = self.sv(ref_t)
            prompt_mel = self.mel_extractor(ref_t).transpose(1, 2)
        self.prompt_mel = prompt_mel

        # Streaming chunking setup (MeanVC "200ms").
        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        self.fbank_window = int(decoding_window)
        self.chunk_samples = int(160 * stride)  # 200ms at 16kHz
        self.vc_chunk = int(decoding_chunk_size * 4)  # 20 frames ~= 200ms

        # Vocoder overlap-add parameters (copied from MeanVC runtime).
        self.vocoder_overlap = 3
        upsample_factor = 160
        self.vocoder_wav_overlap = int((self.vocoder_overlap - 1) * upsample_factor)
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap, device="cpu").numpy()
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap, device="cpu").numpy()

        self.reset_state()

    def reset_state(self) -> None:
        import torch

        self.samples_cache_len = 720  # MeanVC default (400 + 2 * 160)
        self.samples_cache = np.zeros(self.samples_cache_len, dtype=np.float32)

        self.att_cache = torch.zeros((0, 0, 0, 0), device=self.device)
        self.cnn_cache = torch.zeros((0, 0, 0, 0), device=self.device)
        self.asr_offset = 0
        self.encoder_output_cache: Optional["torch.Tensor"] = None

        self.vc_offset = 0
        self.vc_cache: Optional["torch.Tensor"] = None
        self.vc_kv_cache = None

        self.vocoder_cache: Optional["torch.Tensor"] = None
        self.last_wav: Optional[np.ndarray] = None

    @staticmethod
    def _trim_kv_cache(kv_cache: Any, *, max_len: int = 100) -> Any:
        try:
            if (
                kv_cache is not None
                and len(kv_cache) > 0
                and kv_cache[0] is not None
                and kv_cache[0][0] is not None
                and kv_cache[0][0].shape[2] > max_len
            ):
                for i in range(len(kv_cache)):
                    k, v = kv_cache[i]
                    kv_cache[i] = (k[:, :, -max_len:, :], v[:, :, -max_len:, :])
        except Exception:
            return kv_cache
        return kv_cache

    def infer_chunk(self, chunk_16k: np.ndarray) -> tuple[np.ndarray, float]:
        import torch

        chunk_16k = np.asarray(chunk_16k, dtype=np.float32).reshape(-1)
        if len(chunk_16k) != self.chunk_samples:
            raise ValueError(f"Expected chunk_samples={self.chunk_samples}, got {len(chunk_16k)}")

        t0 = time.time()

        samples = np.concatenate([self.samples_cache, chunk_16k], axis=0)
        self.samples_cache = samples[-self.samples_cache_len:]

        fbanks = _extract_fbanks(samples, sample_rate=self.sample_rate, frame_shift_ms=10.0).float().to(self.device)
        encoder_output, self.att_cache, self.cnn_cache = self.asr.forward_encoder_chunk(
            fbanks,
            self.asr_offset,
            self.required_cache_size,
            self.att_cache,
            self.cnn_cache,
        )
        self.asr_offset += int(encoder_output.size(1))

        if self.encoder_output_cache is None:
            encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
        else:
            encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
        self.encoder_output_cache = encoder_output[:, -1:, :]

        cond = encoder_output.transpose(1, 2)
        cond = torch.nn.functional.interpolate(cond, size=self.vc_chunk + 1, mode="linear", align_corners=True)
        cond = cond.transpose(1, 2)[:, 1:, :]

        x = torch.randn(1, cond.shape[1], 80, device=self.device, dtype=cond.dtype)
        tmp_kv_cache = None
        for i in range(self.steps):
            t = self.timesteps[i]
            r = self.timesteps[i + 1]
            t_tensor = torch.full((1,), float(t), device=self.device)
            r_tensor = torch.full((1,), float(r), device=self.device)
            u, tmp_kv_cache = self.vc(
                x,
                t_tensor,
                r_tensor,
                cache=self.vc_cache,
                cond=cond,
                spks=self.spk_emb,
                prompts=self.prompt_mel,
                offset=self.vc_offset,
                is_inference=True,
                kv_cache=self.vc_kv_cache,
            )
            x = x - (float(t) - float(r)) * u

        self.vc_kv_cache = self._trim_kv_cache(tmp_kv_cache)
        self.vc_offset += int(x.shape[1])
        self.vc_cache = x

        mel = x.transpose(1, 2)
        if self.vocoder_cache is not None:
            mel = torch.cat([self.vocoder_cache, mel], dim=-1)
        self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
        mel = (mel + 1.0) / 2.0

        wav = self.vocoder.decode(mel).squeeze().detach().cpu().float().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        if self.vocoder_wav_overlap > 0:
            if self.last_wav is not None and len(wav) >= self.vocoder_wav_overlap * 2:
                front = wav[: self.vocoder_wav_overlap]
                smooth_front = self.last_wav * self.down_linspace + front * self.up_linspace
                mid = wav[self.vocoder_wav_overlap : -self.vocoder_wav_overlap]
                out = np.concatenate([smooth_front, mid], axis=0)
            else:
                out = wav[:-self.vocoder_wav_overlap] if len(wav) > self.vocoder_wav_overlap else wav
            self.last_wav = wav[-self.vocoder_wav_overlap :] if len(wav) >= self.vocoder_wav_overlap else wav
        else:
            out = wav

        dt = time.time() - t0
        return out, float(dt)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MeanVC zero-shot VC runner (offline or streaming simulation).")
    parser.add_argument("--meanvc_dir", type=str, required=True, help="Path to MeanVC repo checkout.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source audio to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")
    parser.add_argument("--steps", type=int, default=2)

    parser.add_argument("--stream", action="store_true", help="Run streaming simulation (200ms chunks).")
    parser.add_argument("--window_ms", type=int, default=200)
    parser.add_argument("--hop_ms", type=int, default=200)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="end", choices=["start", "center", "end"])
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first chunk is processed (recommended for eval).",
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

    import torch

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meanvc_dir = os.path.abspath(str(args.meanvc_dir))
    if not os.path.isdir(meanvc_dir):
        raise FileNotFoundError(f"meanvc_dir not found: {meanvc_dir}")

    # Ensure MeanVC's top-level modules shadow Amphion packages with same names (e.g., `modules`).
    infer_dir = os.path.join(meanvc_dir, "src", "infer")
    _add_sys_paths_first([infer_dir, meanvc_dir])
    # If Amphion's `modules` package was imported earlier, evict it so MeanVC can import its own `modules.py`.
    sys.modules.pop("modules", None)

    sr = 16000
    ref_16k = _load_audio_mono(args.ref, sr=sr)
    src_16k = _load_audio_mono(args.src, sr=sr)

    _set_determinism(int(args.seed))
    streamer = MeanVCStreamer(
        meanvc_dir=meanvc_dir,
        device=device,
        steps=int(args.steps),
        seed=int(args.seed),
        reference_wav_16k=ref_16k,
    )

    out_sr = sr
    meta: dict[str, Any] = {
        "model": "meanvc",
        "meanvc_dir": meanvc_dir,
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "paths": {"ref": str(args.ref), "src": str(args.src), "out": str(args.out)},
    }

    if not bool(args.stream):
        # Offline: use MeanVC's reference inference path (full mel decode; best quality baseline).
        from src.infer.infer_ref import extract_features_from_audio, inference  # type: ignore[import-not-found]

        _set_determinism(int(args.seed))
        t0 = time.time()
        bn, spk_emb, prompt_mel = extract_features_from_audio(
            args.src, args.ref, streamer.asr, streamer.sv, streamer.mel_extractor, str(device)
        )
        _, wav_t, infer_sec = inference(
            streamer.vc,
            streamer.vocoder,
            bn,
            spk_emb,
            prompt_mel,
            chunk_size=int(streamer.vc_chunk),
            steps=int(args.steps),
            device=str(device),
        )
        total_sec = float(time.time() - t0)
        wav = wav_t.detach().cpu().float().numpy().reshape(-1)
        write_wav(args.out, wav, out_sr)

        dur = float(len(src_16k)) / float(sr)
        meta["stats"] = {
            "duration_sec": float(dur),
            # `infer_sec` is model+vocoder time; `total_sec` includes feature extraction.
            "infer_sec": float(infer_sec),
            "total_sec": float(total_sec),
            "rtf_infer": float(float(infer_sec) / max(dur, 1e-6)),
            "rtf_total": float(total_sec / max(dur, 1e-6)),
        }
    else:
        window_in = int(round(float(args.window_ms) / 1000.0 * float(sr)))
        hop_in = int(round(float(args.hop_ms) / 1000.0 * float(sr)))
        hop_out = hop_in
        fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

        if window_in != streamer.chunk_samples or hop_in != streamer.chunk_samples:
            raise ValueError(
                f"MeanVC streamer currently supports window_ms=hop_ms=200 (got window_ms={args.window_ms}, hop_ms={args.hop_ms})."
            )

        if args.emit_align == "start":
            emit_start_out = 0
        elif args.emit_align == "center":
            emit_start_out = max(0, (hop_out - hop_out) // 2)
        elif args.emit_align == "end":
            emit_start_out = 0
        else:
            raise ValueError(f"Unknown emit_align: {args.emit_align}")

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
        hangover_left = 0

        prev_tail: Optional[np.ndarray] = None
        outs = []
        timings = []
        warmup_hops = 0
        chunk_index = 0

        drop_warmup_hops = bool(args.drop_warmup_hops)

        for start in range(0, len(src_16k), hop_in):
            hop = src_16k[start : start + hop_in]
            if len(hop) < hop_in:
                hop = np.pad(hop, (0, hop_in - len(hop)), mode="constant")

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop,
                        sample_rate=sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                voiced = is_voiced_webrtcvad(
                    hop,
                    sample_rate=sr,
                    frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                    aggressiveness=int(args.vad_webrtc_aggressiveness),
                    min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                )
            else:
                raise ValueError(f"Unknown vad_mode: {vad_mode}")

            if not voiced and hangover_left > 0:
                voiced = True
                hangover_left -= 1
            elif voiced:
                hangover_left = hangover_hops

            # Always run the first chunk (if voiced) to populate caches, but optionally drop its output for eval.
            if chunk_index == 0 and drop_warmup_hops:
                warmup_hops += 1
                if voiced:
                    _, dt = streamer.infer_chunk(hop)
                    timings.append(float(dt))
                if fade_out > 0:
                    prev_tail = np.zeros(fade_out, dtype=np.float32)
                chunk_index += 1
                continue

            if not voiced:
                streamer.reset_state()
                out_hop = np.zeros(hop_out, dtype=np.float32)
            else:
                y, dt = streamer.infer_chunk(hop)
                timings.append(float(dt))
                out_hop = normalize_length(y, hop_out, align=str(args.normalize_align))  # type: ignore[arg-type]
                out_hop = out_hop[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

            out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            if fade_out > 0:
                prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

            outs.append(out_hop)
            chunk_index += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        write_wav(args.out, out, out_sr)

        hop_sec = float(args.hop_ms) / 1000.0
        p95 = float(np.percentile(timings, 95)) if timings else 0.0

        meta["streaming"] = asdict(
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                fade_ms=int(args.fade_ms),
                normalize_align=str(args.normalize_align),  # type: ignore[arg-type]
                emit_align=str(args.emit_align),  # type: ignore[arg-type]
                drop_warmup_hops=bool(args.drop_warmup_hops),
                vad_mode=str(args.vad_mode),  # type: ignore[arg-type]
                vad_db=float(args.vad_db),
                vad_frame_ms=float(args.vad_frame_ms),
                vad_hangover_ms=float(args.vad_hangover_ms),
                vad_webrtc_frame_ms=int(args.vad_webrtc_frame_ms),  # type: ignore[arg-type]
                vad_webrtc_aggressiveness=int(args.vad_webrtc_aggressiveness),
                vad_webrtc_min_voiced_ratio=float(args.vad_webrtc_min_voiced_ratio),
                peak_limit=float(args.peak_limit),
            )
        )
        meta["stats"] = {
            "warmup_hops": int(warmup_hops),
            "delay_samples": int(warmup_hops * hop_out),
            "mean_window_sec": float(np.mean(timings)) if timings else 0.0,
            "p95_window_sec": float(p95),
            "rtf_p95": float(p95 / max(hop_sec, 1e-6)),
        }

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta, indent=2))
        print(f"Wrote meta: {meta_p}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
