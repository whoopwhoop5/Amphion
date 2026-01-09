# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    AudioRingBuffer,
    apply_peak_limiter,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
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
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    model_repo: str
    nfe_step: int
    cfg_strength: float
    sway_sampling_coef: float
    device: str
    seed: int
    ref_max_sec: float
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


def _resample_if_needed(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32, copy=False)


def _trim_or_pad_ref(wav: np.ndarray, *, sample_rate: int, max_sec: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if max_sec <= 0:
        return wav
    max_len = int(round(float(max_sec) * float(sample_rate)))
    if len(wav) <= max_len:
        return wav
    return wav[:max_len]


def _dedup_units(unit_string: str) -> str:
    if not unit_string:
        return ""
    out = [unit_string[0]]
    for c in unit_string[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


class _TorchKMeans:
    def __init__(self, cluster_centers: np.ndarray, device: torch.device):
        # Expect (n_clusters, dim); store as (dim, n_clusters)
        c_np = np.asarray(cluster_centers, dtype=np.float32).T
        cnorm_np = (c_np**2).sum(0, keepdims=True)
        self._c = torch.from_numpy(c_np).to(device)
        self._cnorm = torch.from_numpy(cnorm_np).to(device)
        self._device = device

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self._device)
        dist = x.pow(2).sum(1, keepdim=True) - 2.0 * (x @ self._c) + self._cnorm
        return dist.argmin(dim=1).detach().cpu().numpy()


@torch.no_grad()
def _xeus_encode_units(
    *,
    xeus_model,
    km: _TorchKMeans,
    unit_map: dict[str, str],
    wav_16k: np.ndarray,
    device: torch.device,
    layer_index: int = 14,
) -> tuple[np.ndarray, float]:
    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    if len(wav_16k) == 0:
        return np.zeros(0, dtype=np.int64), 320.0

    wav_t = torch.from_numpy(wav_16k).to(device).unsqueeze(0)
    wav_len = torch.LongTensor([wav_t.shape[-1]]).to(device)

    outputs = xeus_model.encode(wav_t, wav_len, use_mask=False, use_final_output=False)
    feats = outputs[0][int(layer_index)]
    feats = feats.squeeze(0)
    if feats.ndim != 2:
        raise ValueError(f"Unexpected XEUS feature shape: {tuple(feats.shape)}")

    units = km.assign(feats).astype(np.int64, copy=False)
    stride = float(len(wav_16k)) / float(max(len(units), 1))
    return units, stride


@torch.no_grad()
def _units_to_text(unit_ids: np.ndarray, unit_map: dict[str, str]) -> str:
    unit_ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
    if len(unit_ids) == 0:
        return ""
    chars = [unit_map.get(str(int(i)), "") for i in unit_ids.tolist()]
    return _dedup_units("".join(chars))


@torch.no_grad()
def _infer_ezvc(
    *,
    model,
    vocoder,
    convert_char_to_pinyin,
    ref_audio: torch.Tensor,
    ref_units: str,
    gen_units: str,
    ref_audio_len_frames: int,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    window_frames: int,
    seed: int,
) -> np.ndarray:
    _set_determinism(seed)

    if not gen_units:
        return np.zeros(0, dtype=np.float32)

    if ref_audio.ndim != 2:
        raise ValueError(f"Expected ref_audio waveform tensor shaped (B, N), got {tuple(ref_audio.shape)}")

    full_text = [ref_units + gen_units]
    final_text_list = convert_char_to_pinyin(full_text)

    duration = int(ref_audio_len_frames + window_frames)
    if duration <= ref_audio_len_frames:
        duration = int(ref_audio_len_frames + 1)

    generated, _ = model.sample(
        cond=ref_audio,
        text=final_text_list,
        duration=duration,
        steps=int(nfe_step),
        cfg_strength=float(cfg_strength),
        sway_sampling_coef=float(sway_sampling_coef),
    )

    generated = generated.to(torch.float32)
    generated = generated[:, ref_audio_len_frames:, :]
    generated = generated.permute(0, 2, 1)

    if hasattr(vocoder, "decode"):
        wave = vocoder.decode(generated)
    else:
        wave = vocoder(generated)
    wave = wave.squeeze().detach().cpu().float().numpy().reshape(-1)
    return np.asarray(wave, dtype=np.float32).reshape(-1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EZ-VC (F5-TTS) zero-shot VC runner (offline or streaming sim).")
    parser.add_argument("--model_repo", type=str, default="SPRINGLab/EZ-VC")
    parser.add_argument("--vocoder_name", type=str, default="bigvgan", choices=["bigvgan", "vocos"])
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--ref_max_sec", type=float, default=6.0, help="Trim reference audio to this many seconds.")

    parser.add_argument("--nfe_step", type=int, default=12, help="Diffusion sampling steps (lower is faster).")
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0)

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=300)
    parser.add_argument("--fade_ms", type=int, default=10)
    parser.add_argument("--normalize_align", type=str, default="end", choices=["start", "end"])
    parser.add_argument("--emit_align", type=str, default="center", choices=["start", "center", "end"])
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop output until the first full window is available (recommended for eval).",
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
    _set_determinism(int(args.seed))

    # Imports from EZ-VC package.
    from huggingface_hub import hf_hub_download
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import load_model, load_vocoder as _f5_load_vocoder
    from f5_tts.model.utils import convert_char_to_pinyin

    # XEUS units.
    from espnet2.tasks.ssl import SSLTask  # type: ignore[import-not-found]
    import joblib  # type: ignore[import-not-found]
    from importlib.resources import files

    model_repo = str(args.model_repo)
    ckpt_file = hf_hub_download(model_repo, filename="model_2700000.safetensors")
    vocab_file = hf_hub_download(model_repo, filename="vocab.txt")
    km_path = hf_hub_download(model_repo, filename="kmeans_xeus_500_multilingual.pkl")

    cfg_path = str(files("f5_tts").joinpath("configs/F5TTS_Base_EZ-VC.yaml"))
    model_cfg = OmegaConf.load(cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    def _load_vocoder(vocoder_name: str):
        vocoder_name = str(vocoder_name)
        if vocoder_name == "bigvgan":
            try:
                import bigvgan  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - only hit if upstream layout changes
                raise RuntimeError(
                    "Failed to import BigVGAN. Ensure EZ-VC submodules are initialized "
                    "(git submodule update --init --recursive)."
                ) from exc

            vocoder = bigvgan.BigVGAN.from_pretrained("SPRINGLab/bigvgan_16khz", use_cuda_kernel=False)
            vocoder.remove_weight_norm()
            return vocoder.eval().to(device)

        return _f5_load_vocoder(vocoder_name=vocoder_name, device=str(device))

    vocoder = _load_vocoder(str(args.vocoder_name))
    model = load_model(
        model_cls,
        model_arc,
        ckpt_file,
        mel_spec_type=str(args.vocoder_name),
        vocab_file=vocab_file,
        device=str(device),
    )
    model.eval()

    xeus_ckpt = hf_hub_download("espnet/xeus", filename="model/xeus_checkpoint_old.pth")
    xeus_cfg = str(files("f5_tts").joinpath("infer/xeus/config.yaml"))
    xeus_model, _ = SSLTask.build_model_from_file(xeus_cfg, xeus_ckpt, str(device))
    xeus_model.eval()

    km_model = joblib.load(km_path)
    km = _TorchKMeans(km_model.cluster_centers_, device=device)
    unit_map = json.loads(files("f5_tts").joinpath("infer/xeus/char_map.json").read_text(encoding="utf-8"))

    # Audio I/O.
    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)

    in_sr = 16000
    ref_16k = _resample_if_needed(ref_wav, ref_sr, in_sr)
    src_16k = _resample_if_needed(src_wav, src_sr, in_sr)
    ref_16k = _trim_or_pad_ref(ref_16k, sample_rate=in_sr, max_sec=float(args.ref_max_sec))

    ref_units_ids, _ = _xeus_encode_units(
        xeus_model=xeus_model,
        km=km,
        unit_map=unit_map,
        wav_16k=ref_16k,
        device=device,
    )
    ref_units = _units_to_text(ref_units_ids, unit_map)

    # Reference conditioning audio (tensor).
    ref_audio = torch.from_numpy(ref_16k).to(device).unsqueeze(0)
    rms = torch.sqrt(torch.mean(ref_audio**2) + 1e-9)
    target_rms = 0.1
    scaled = False
    if float(rms) < float(target_rms):
        ref_audio = ref_audio * float(target_rms) / rms
        scaled = True

    hop_length = int(getattr(getattr(model, "mel_spec", None), "hop_length", 160))
    # Use the model's mel extractor to estimate the conditioning frame count. Different mel backends
    # (e.g., vocos vs bigvgan) may pad/center differently, so waveform_len//hop can be off.
    with torch.no_grad():
        ref_audio_len_frames = int(model.mel_spec(ref_audio).shape[-1])

    out_sr = 16000
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        window_frames = int(round(float(len(src_16k)) / float(hop_length)))
        t0 = time.time()
        out = _infer_ezvc(
            model=model,
            vocoder=vocoder,
            convert_char_to_pinyin=convert_char_to_pinyin,
            ref_audio=ref_audio,
            ref_units=ref_units,
            gen_units=_units_to_text(
                _xeus_encode_units(
                    xeus_model=xeus_model,
                    km=km,
                    unit_map=unit_map,
                    wav_16k=src_16k,
                    device=device,
                )[0],
                unit_map,
            ),
            ref_audio_len_frames=ref_audio_len_frames,
            nfe_step=int(args.nfe_step),
            cfg_strength=float(args.cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            window_frames=window_frames,
            seed=int(args.seed),
        )
        timings.append(time.time() - t0)
        if scaled:
            out = (out * float(rms) / float(target_rms)).astype(np.float32, copy=False)
        sf.write(args.out, out, out_sr)
    else:
        window_in = int(round(float(args.window_ms) / 1000.0 * float(in_sr)))
        hop_in = int(round(float(args.hop_ms) / 1000.0 * float(in_sr)))
        if window_in <= 0 or hop_in <= 0:
            raise ValueError("window_ms and hop_ms must be > 0")
        if hop_in > window_in:
            raise ValueError("hop_ms must be <= window_ms")

        window_out = int(round(float(args.window_ms) / 1000.0 * float(out_sr)))
        hop_out = int(round(float(args.hop_ms) / 1000.0 * float(out_sr)))
        fade_out = int(round(float(args.fade_ms) / 1000.0 * float(out_sr)))

        ring = AudioRingBuffer(window_in)
        prev_tail: Optional[np.ndarray] = None

        drop_warmup_hops = bool(args.drop_warmup_hops)
        outs: list[np.ndarray] = []
        warmup_hops = 0
        window_count = 0

        if args.emit_align == "start":
            emit_start_out = 0
        elif args.emit_align == "center":
            emit_start_out = max(0, (window_out - hop_out) // 2)
        elif args.emit_align == "end":
            emit_start_out = max(0, window_out - hop_out)
        else:
            raise ValueError(f"Unknown emit_align: {args.emit_align}")

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
        hangover_left = 0

        src_units_ids, stride = _xeus_encode_units(
            xeus_model=xeus_model,
            km=km,
            unit_map=unit_map,
            wav_16k=src_16k,
            device=device,
        )

        window_frames = int(round(float(args.window_ms) / 1000.0 * float(out_sr) / float(hop_length)))

        for start in range(0, len(src_16k), hop_in):
            hop = src_16k[start : start + hop_in]
            if len(hop) < hop_in:
                hop = np.pad(hop, (0, hop_in - len(hop)), mode="constant")
            ring.write(hop)

            if ring.size < window_in:
                warmup_hops += 1
                if fade_out > 0:
                    prev_tail = np.zeros(fade_out, dtype=np.float32)
                if not drop_warmup_hops:
                    outs.append(np.zeros(hop_out, dtype=np.float32))
                continue

            window = ring.read_last(window_in)

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop,
                        sample_rate=in_sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                voiced = is_voiced_webrtcvad(
                    hop,
                    sample_rate=in_sr,
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

            if not voiced:
                out_hop = np.zeros(hop_out, dtype=np.float32)
            else:
                # Use unit IDs from the global XEUS pass; slice by approximate stride.
                s = int(round(float(start) / float(stride)))
                e = int(round(float(start + window_in) / float(stride)))
                s = max(0, min(s, len(src_units_ids)))
                e = max(s, min(e, len(src_units_ids)))
                gen_units = _units_to_text(src_units_ids[s:e], unit_map)

                t0 = time.time()
                out_window = _infer_ezvc(
                    model=model,
                    vocoder=vocoder,
                    convert_char_to_pinyin=convert_char_to_pinyin,
                    ref_audio=ref_audio,
                    ref_units=ref_units,
                    gen_units=gen_units,
                    ref_audio_len_frames=ref_audio_len_frames,
                    nfe_step=int(args.nfe_step),
                    cfg_strength=float(args.cfg_strength),
                    sway_sampling_coef=float(args.sway_sampling_coef),
                    window_frames=window_frames,
                    seed=int(args.seed) + int(window_count),
                )
                timings.append(time.time() - t0)
                if scaled:
                    out_window = (out_window * float(rms) / float(target_rms)).astype(np.float32, copy=False)

                out_window = normalize_length(
                    out_window, window_out, align=str(args.normalize_align)  # type: ignore[arg-type]
                )
                out_hop = out_window[emit_start_out : emit_start_out + hop_out].astype(np.float32, copy=False)

            out_hop = crossfade_prefix_inplace(out_hop, prev_tail, fade_out)
            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))
            if fade_out > 0:
                prev_tail = out_hop[-fade_out:].astype(np.float32, copy=True)

            outs.append(out_hop)
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        sf.write(args.out, out, out_sr)
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
                peak_limit=float(args.peak_limit),
            )
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            model_repo=str(model_repo),
            nfe_step=int(args.nfe_step),
            cfg_strength=float(args.cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            device=str(device),
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
