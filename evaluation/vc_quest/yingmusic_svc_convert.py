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
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch

from evaluation.vc_quest.streaming_utils import (
    VadFrameMs,
    apply_peak_limiter,
    crossfade_prefix_inplace,
    is_silent_rms_db,
    is_voiced_webrtcvad,
    normalize_length,
)


VadMode = Literal["rms", "webrtc", "off"]


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _purge_import_cache(prefix: str) -> None:
    # YingMusic-SVC uses a top-level `modules/` package name that conflicts with Amphion's own
    # `modules/` package. Purge the import cache to force imports from YingMusic-SVC after we
    # prepend ymsvc_dir to sys.path.
    for name in list(sys.modules.keys()):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def _remove_sys_path_entry(path: str) -> None:
    while path in sys.path:
        sys.path.remove(path)


def _set_determinism(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass


def _torch_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


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


def _calculate_semitone_shift(source_f0: np.ndarray, reference_f0: np.ndarray) -> float:
    src_valid = np.asarray(source_f0, dtype=np.float32).reshape(-1)
    ref_valid = np.asarray(reference_f0, dtype=np.float32).reshape(-1)
    src_valid = src_valid[src_valid > 1.0]
    ref_valid = ref_valid[ref_valid > 1.0]
    if len(src_valid) == 0 or len(ref_valid) == 0:
        return 0.0

    mean_log_diff = float(np.mean(np.log(ref_valid)) - np.mean(np.log(src_valid)))
    return float(12.0 * mean_log_diff / float(np.log(2.0)))


def _adaptive_pitch_shift_factor(
    voiced_f0_ori: np.ndarray,
    *,
    low_threshold: float = 120.0,
    high_threshold: float = 205.0,
    min_factor: float = 0.3,
    max_factor: float = 1.0,
) -> float:
    voiced_f0_ori = np.asarray(voiced_f0_ori, dtype=np.float32).reshape(-1)
    valid = voiced_f0_ori[voiced_f0_ori > 1.0]
    if len(valid) == 0:
        return float(max_factor)
    mean_f0 = float(np.mean(valid))

    if mean_f0 > high_threshold:
        return float(max(min_factor, max_factor - (mean_f0 - high_threshold) / (560.0 - high_threshold)))
    if mean_f0 < low_threshold:
        return float(max_factor - (low_threshold - mean_f0) / low_threshold * 1.2)
    return float(max_factor)


def _adjust_f0_semitones(f0_sequence: torch.Tensor, n_semitones: float) -> torch.Tensor:
    if abs(float(n_semitones)) < 0.1:
        return f0_sequence
    factor = float(2.0 ** (float(n_semitones) / 12.0))
    out = f0_sequence.clone()
    mask = out > 1.0
    out[mask] = out[mask] * factor
    return out


def _semitone_map(x: float, *, threshold: float = 7.0) -> int:
    if x >= threshold:
        return 12
    if x <= -threshold:
        return -12
    return 0


def _preprocess_voice_conversion(
    *,
    voiced_f0_ori: np.ndarray,
    voiced_f0_alt: np.ndarray,
    shifted_f0_alt: torch.Tensor,
    enable_adaptive: bool = True,
    max_shift_semitones: float = 24.0,
    forced_pitch_shift: Optional[int],
) -> tuple[torch.Tensor, int]:
    if forced_pitch_shift is None:
        base_pitch_shift = _calculate_semitone_shift(voiced_f0_alt, voiced_f0_ori)
        if enable_adaptive:
            adaptive_factor = _adaptive_pitch_shift_factor(voiced_f0_ori)
            pitch_shift = float(base_pitch_shift) * float(adaptive_factor) + 3.5
        else:
            pitch_shift = float(base_pitch_shift)
        pitch_shift = float(np.clip(pitch_shift, -float(max_shift_semitones), float(max_shift_semitones)))
        pitch_shift = _semitone_map(pitch_shift)
    else:
        pitch_shift = int(forced_pitch_shift)

    final_f0_alt = _adjust_f0_semitones(shifted_f0_alt, float(pitch_shift))
    return final_f0_alt, int(pitch_shift)


def _semantic_encode(semantic_fn, wav_16k: torch.Tensor) -> torch.Tensor:
    # Mirror upstream behavior: if <=30s, run once; else run with overlap.
    # wav_16k: (1, N) on device
    if wav_16k.size(-1) <= 16000 * 30:
        return semantic_fn(wav_16k)

    overlapping_time = 5  # seconds
    s_list = []
    buffer: Optional[torch.Tensor] = None
    traversed = 0
    while traversed < wav_16k.size(-1):
        if buffer is None:
            chunk = wav_16k[:, traversed : traversed + 16000 * 30]
        else:
            chunk = torch.cat(
                [
                    buffer,
                    wav_16k[:, traversed : traversed + 16000 * (30 - overlapping_time)],
                ],
                dim=-1,
            )
        s_chunk = semantic_fn(chunk)
        if traversed == 0:
            s_list.append(s_chunk)
        else:
            s_list.append(s_chunk[:, 50 * overlapping_time :])
        buffer = chunk[:, -16000 * overlapping_time :]
        traversed += 30 * 16000 if traversed == 0 else int(chunk.size(-1) - 16000 * overlapping_time)
    return torch.cat(s_list, dim=1)


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    diffusion_steps: int
    inference_cfg_rate: float
    drop_warmup_hops: bool
    vad_mode: VadMode
    vad_db: float
    vad_frame_ms: float
    vad_hangover_ms: float
    vad_webrtc_frame_ms: VadFrameMs
    vad_webrtc_aggressiveness: int
    vad_webrtc_min_voiced_ratio: float
    peak_limit: float


@dataclass(frozen=True)
class RunConfig:
    ymsvc_dir: str
    checkpoint_repo: str
    checkpoint_filename: str
    config_path: str
    device: str
    seed: int
    fp16: bool
    ref_max_sec: float
    length_adjust: float
    inference_cfg_rate: float
    diffusion_steps: int
    forced_pitch_shift: Optional[int]
    stream: Optional[StreamConfig] = None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="YingMusic-SVC runner (offline or streaming sim).",
    )
    parser.add_argument("--ymsvc_dir", type=str, required=True, help="Path to YingMusic-SVC repo checkout.")
    parser.add_argument("--checkpoint_repo", type=str, default="GiantAILab/YingMusic-SVC")
    parser.add_argument("--checkpoint_filename", type=str, default="YingMusic-SVC-full.pt")
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="YAML config path. Defaults to <ymsvc_dir>/configs/YingMusic-SVC.yml",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
    parser.add_argument("--length_adjust", type=float, default=1.0)
    parser.add_argument(
        "--forced_pitch_shift",
        type=int,
        default=None,
        help="Force pitch shift in semitones (e.g. -12/0/+12). Default: auto.",
    )

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")
    parser.add_argument("--ref_max_sec", type=float, default=10.0, help="Trim reference audio to this many seconds.")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=300)
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
    parser.add_argument("--vad_webrtc_frame_ms", type=int, default=30, choices=[10, 20, 30])
    parser.add_argument("--vad_webrtc_aggressiveness", type=int, default=2)
    parser.add_argument("--vad_webrtc_min_voiced_ratio", type=float, default=0.1)
    parser.add_argument("--peak_limit", type=float, default=0.99)
    args = parser.parse_args(argv)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    _set_determinism(int(args.seed))

    ymsvc_dir = str(Path(args.ymsvc_dir).resolve())
    _add_sys_path_first(ymsvc_dir)
    _purge_import_cache("modules")

    # Import YingMusic-SVC modules after sys.path swap.
    # `modules` is an implicit namespace package in YingMusic-SVC (no __init__.py), but Amphion's own
    # `modules/` is a regular package. Python will prefer the regular package unless we temporarily
    # remove Amphion from sys.path for the import.
    saved_sys_path = list(sys.path)
    try:
        amphion_root = str(Path(__file__).resolve().parents[2])
        _remove_sys_path_entry("")
        _remove_sys_path_entry(amphion_root)

        import yaml
        from huggingface_hub import hf_hub_download

        from modules.commons import build_model, load_checkpoint, recursive_munch  # type: ignore[import-not-found]
        from modules.rmvpe import RMVPE  # type: ignore[import-not-found]

        from modules.campplus.DTDNN import CAMPPlus  # type: ignore[import-not-found]
        from modules.audio import mel_spectrogram  # type: ignore[import-not-found]

        from transformers import AutoFeatureExtractor, WhisperModel  # type: ignore[import-not-found]
    finally:
        sys.path = saved_sys_path

    # Resolve config/checkpoint.
    config_path = (
        str(Path(args.config_path).resolve())
        if str(args.config_path).strip()
        else str(Path(ymsvc_dir).joinpath("configs/YingMusic-SVC.yml").resolve())
    )
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Missing config_path: {config_path}")

    ckpt_path = hf_hub_download(
        repo_id=str(args.checkpoint_repo),
        filename=str(args.checkpoint_filename),
    )

    config = yaml.safe_load(Path(config_path).read_text())
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"

    preprocess = config["preprocess_params"]
    sr = int(preprocess["sr"])
    hop_length = int(preprocess["spect_params"]["hop_length"])

    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(
        model,
        None,
        ckpt_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # F0 extractor (RMVPE weights from HF).
    rmvpe_path = hf_hub_download("lj1995/VoiceConversionWebUI", filename="rmvpe.pt")
    f0_extractor = RMVPE(rmvpe_path, is_half=False, device=device)
    f0_fn = f0_extractor.infer_from_audio

    # Style encoder (CAMP++) weights from HF.
    campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)

    # Vocoder (BigVGAN) weights from HF.
    if str(model_params.vocoder.type) != "bigvgan":
        raise ValueError(f"Unsupported vocoder type: {model_params.vocoder.type}")
    from modules.bigvgan import bigvgan  # type: ignore[import-not-found]

    vocoder_fn = bigvgan.BigVGAN.from_pretrained(str(model_params.vocoder.name), use_cuda_kernel=False)
    vocoder_fn.remove_weight_norm()
    vocoder_fn = vocoder_fn.eval().to(device)
    for p in vocoder_fn.parameters():
        p.requires_grad_(False)

    # Semantic encoder (Whisper).
    if str(model_params.speech_tokenizer.type) != "whisper":
        raise ValueError(f"Unsupported speech_tokenizer type: {model_params.speech_tokenizer.type}")
    whisper_name = str(model_params.speech_tokenizer.name)
    whisper_dtype = torch.float16 if device.type == "cuda" else torch.float32
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=whisper_dtype).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
        waves_16k = waves_16k.reshape(1, -1)
        ori_inputs = whisper_feature_extractor(
            [waves_16k.squeeze(0).detach().cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
        )
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features,
            attention_mask=ori_inputs.attention_mask,
        ).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        s = ori_outputs.last_hidden_state.to(torch.float32)
        s = s[:, : waves_16k.size(-1) // 320 + 1]
        return s

    # Mel spectrogram function.
    mel_fn_args = {
        "n_fft": int(preprocess["spect_params"]["n_fft"]),
        "win_size": int(preprocess["spect_params"]["win_length"]),
        "hop_size": hop_length,
        "num_mels": int(preprocess["spect_params"]["n_mels"]),
        "sampling_rate": sr,
        "fmin": float(preprocess["spect_params"].get("fmin", 0)),
        "fmax": None
        if str(preprocess["spect_params"].get("fmax", "None")) == "None"
        else float(preprocess["spect_params"].get("fmax", 8000)),
        "center": False,
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)  # noqa: E731

    # Audio I/O.
    ref_wav, ref_sr = _load_mono(args.ref)
    src_wav, src_sr = _load_mono(args.src)

    ref_sr_wav = _resample_if_needed(ref_wav, ref_sr, sr)
    src_sr_wav = _resample_if_needed(src_wav, src_sr, sr)
    ref_sr_wav = _trim_or_pad_ref(ref_sr_wav, sample_rate=sr, max_sec=float(args.ref_max_sec))

    ref_16k = _resample_if_needed(ref_sr_wav, sr, 16000)
    src_16k = _resample_if_needed(src_sr_wav, sr, 16000)

    ref_sr_t = torch.from_numpy(ref_sr_wav).to(device).unsqueeze(0)
    src_sr_t = torch.from_numpy(src_sr_wav).to(device).unsqueeze(0)
    ref_16k_t = torch.from_numpy(ref_16k).to(device).unsqueeze(0)
    src_16k_t = torch.from_numpy(src_16k).to(device).unsqueeze(0)

    # Semantic features.
    s_alt = _semantic_encode(semantic_fn, src_16k_t)
    s_ori = _semantic_encode(semantic_fn, ref_16k_t)

    # Prompt mel + source mel (target lengths are mel frames).
    mel_src = to_mel(src_sr_t.float())
    mel_ref = to_mel(ref_sr_t.float())
    target_lengths = torch.LongTensor([int(mel_src.size(2) * float(args.length_adjust))]).to(device)
    target2_lengths = torch.LongTensor([int(mel_ref.size(2))]).to(device)

    # Style embedding.
    import torchaudio.compliance.kaldi as kaldi  # type: ignore[import-not-found]

    feat2 = kaldi.fbank(
        ref_16k_t.detach().cpu(),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0).to(device))

    # F0 conditioning (optional, default on in config).
    f0_condition = bool(model_params.length_regulator.get("f0_condition", True))
    shifted_f0_alt = None
    if f0_condition:
        f0_ori = f0_fn(ref_16k_t[0], thred=0.03)
        f0_alt = f0_fn(src_16k_t[0], thred=0.03)
        f0_ori_t = torch.from_numpy(np.asarray(f0_ori, dtype=np.float32)).to(device)[None]
        f0_alt_t = torch.from_numpy(np.asarray(f0_alt, dtype=np.float32)).to(device)[None]
        voiced_f0_ori = f0_ori_t[f0_ori_t > 1.0].detach().cpu().numpy()
        voiced_f0_alt = f0_alt_t[f0_alt_t > 1.0].detach().cpu().numpy()
        log_f0_alt = torch.log(f0_alt_t + 1e-5)
        shifted_f0_alt = torch.exp(log_f0_alt)
        shifted_f0_alt, _pitch_shift = _preprocess_voice_conversion(
            voiced_f0_ori=voiced_f0_ori,
            voiced_f0_alt=voiced_f0_alt,
            shifted_f0_alt=shifted_f0_alt,
            enable_adaptive=True,
            max_shift_semitones=24.0,
            forced_pitch_shift=args.forced_pitch_shift,
        )
        f0_ori_t_for_lr = f0_ori_t
        shifted_f0_alt_for_lr = shifted_f0_alt
    else:
        f0_ori_t_for_lr = None
        shifted_f0_alt_for_lr = None

    # Length regulation (content -> mel time).
    cond, _, _, _, _, style_cond = model.length_regulator(
        s_alt,
        ylens=target_lengths,
        n_quantizers=3,
        f0=shifted_f0_alt_for_lr,
        style=style2,
        return_style_residual=True,
    )
    prompt_condition, _, _, _, _, style_prompt = model.length_regulator(
        s_ori,
        ylens=target2_lengths,
        n_quantizers=3,
        f0=f0_ori_t_for_lr,
        style=style2,
        return_style_residual=True,
    )

    use_style_residual = bool(model_params.length_regulator.get("use_style_residual", False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        _torch_sync(device)
        t0 = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            cat_style_cond = None
            if use_style_residual:
                cat_style_cond = torch.cat([style_prompt, style_cond], dim=1)
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(device),
                mel_ref,
                style2,
                None,
                int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
                style_r=cat_style_cond,
            )
            vc_target = vc_target[:, :, mel_ref.size(-1) :]
        with torch.no_grad():
            wave = vocoder_fn(vc_target.float()).squeeze().detach().cpu().float().numpy().reshape(-1)
        timings.append(time.time() - t0)
        sf.write(args.out, wave, sr)
    else:
        window_frames = int(round(float(args.window_ms) / 1000.0 * float(sr) / float(hop_length)))
        hop_frames = int(round(float(args.hop_ms) / 1000.0 * float(sr) / float(hop_length)))
        window_frames = max(1, window_frames)
        hop_frames = max(1, hop_frames)
        if hop_frames > window_frames:
            raise ValueError("hop_ms must be <= window_ms")

        overlap_frames = int(max(0, window_frames - hop_frames))
        window_wave_len = int(window_frames * hop_length)
        hop_wave_len = int(hop_frames * hop_length)
        overlap_wave_len = int(overlap_frames * hop_length)

        outs: list[np.ndarray] = []
        prev_tail: Optional[np.ndarray] = None

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
        hangover_left = 0

        warmup_hops = int(np.ceil(float(window_frames) / float(hop_frames))) - 1 if hop_frames > 0 else 0
        warmup_hops = int(max(0, warmup_hops))

        processed = 0
        window_count = 0

        while processed < int(cond.size(1)):
            remaining = int(cond.size(1) - processed)
            chunk_len = int(min(window_frames, remaining))
            chunk_cond = cond[:, processed : processed + chunk_len]
            is_last = processed + window_frames >= int(cond.size(1))

            # Hop audio for VAD (use source waveform timeline).
            hop_start = int(processed * hop_length)
            hop_end = hop_start + int(min(hop_wave_len, max(0, len(src_sr_wav) - hop_start)))
            hop_wav = src_sr_wav[hop_start:hop_end]
            if len(hop_wav) < hop_wave_len:
                hop_wav = np.pad(hop_wav, (0, hop_wave_len - len(hop_wav)), mode="constant")

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop_wav,
                        sample_rate=sr,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                hop_16k = _resample_if_needed(hop_wav, sr, 16000)
                voiced = is_voiced_webrtcvad(
                    hop_16k,
                    sample_rate=16000,
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
                chunk_wave = np.zeros(int(chunk_len * hop_length), dtype=np.float32)
            else:
                _set_determinism(int(args.seed) + int(window_count))
                _torch_sync(device)
                t0 = time.time()
                with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    cat_style_cond = None
                    if use_style_residual:
                        chunk_style = style_cond[:, processed : processed + chunk_len]
                        cat_style_cond = torch.cat([style_prompt, chunk_style], dim=1)
                    vc_target = model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(device),
                        mel_ref,
                        style2,
                        None,
                        int(args.diffusion_steps),
                        inference_cfg_rate=float(args.inference_cfg_rate),
                        style_r=cat_style_cond,
                    )
                    vc_target = vc_target[:, :, mel_ref.size(-1) :]
                with torch.no_grad():
                    wave_t = vocoder_fn(vc_target.float()).squeeze()
                _torch_sync(device)
                timings.append(time.time() - t0)
                chunk_wave = wave_t.detach().cpu().float().numpy().reshape(-1).astype(np.float32, copy=False)

            expected = int(chunk_len * hop_length)
            chunk_wave = normalize_length(chunk_wave, expected, align="start")

            if overlap_wave_len > 0 and prev_tail is not None:
                chunk_wave = crossfade_prefix_inplace(chunk_wave, prev_tail, int(overlap_wave_len))

            if not is_last and overlap_wave_len > 0:
                out_hop = chunk_wave[:hop_wave_len].astype(np.float32, copy=False)
                prev_tail = chunk_wave[-overlap_wave_len:].astype(np.float32, copy=True)
            else:
                out_hop = chunk_wave.astype(np.float32, copy=False)
                prev_tail = None

            out_hop = apply_peak_limiter(out_hop, peak_limit=float(args.peak_limit))

            outs.append(out_hop)

            if is_last:
                break
            processed += hop_frames
            window_count += 1

        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        out = normalize_length(out, int(cond.size(1) * hop_length), align="start")
        sf.write(args.out, out, sr)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                diffusion_steps=int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
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
            if bool(args.stream)
            else None
        )
        cfg = RunConfig(
            ymsvc_dir=str(ymsvc_dir),
            checkpoint_repo=str(args.checkpoint_repo),
            checkpoint_filename=str(args.checkpoint_filename),
            config_path=str(config_path),
            device=str(device),
            seed=int(args.seed),
            fp16=bool(args.fp16),
            ref_max_sec=float(args.ref_max_sec),
            length_adjust=float(args.length_adjust),
            inference_cfg_rate=float(args.inference_cfg_rate),
            diffusion_steps=int(args.diffusion_steps),
            forced_pitch_shift=args.forced_pitch_shift,
            stream=stream_cfg,
        )

        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": 0 if not bool(args.stream) else int(warmup_hops),
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
