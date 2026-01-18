# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
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
    is_silent_rms_db,
    is_voiced_webrtcvad,
)


VadMode = Literal["rms", "webrtc", "off"]


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _purge_import_cache(prefix: str) -> None:
    # Seed-VC uses a top-level `modules/` package name that conflicts with Amphion's own
    # `modules/` package. If Amphion's `modules` is already imported, Python will keep using
    # it from `sys.modules` even after we prepend Seed-VC to sys.path. Purge the import
    # cache to force a clean import from Seed-VC.
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


def _set_determinism(seed: int) -> None:
    import torch

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


def _load_audio_mono(path: str, *, sr: Optional[int] = None) -> tuple[np.ndarray, int]:
    import librosa

    wav, got_sr = librosa.load(path, sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32).reshape(-1), int(got_sr)


def _resample_np(wav: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32, copy=False)


def _torch_sync(device: "torch.device") -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    crossfade_ms: int
    extra_time_ce_ms: int
    extra_time_ms: int
    extra_time_right_ms: int
    diffusion_steps: int
    inference_cfg_rate: float
    max_prompt_length_sec: float
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
    seedvc_dir: str
    device: str
    seed: int
    checkpoint_path: str
    config_path: str
    hf_repo: str
    hf_checkpoint_name: str
    hf_config_name: str
    fp16: bool
    length_adjust: float
    diffusion_steps: int
    inference_cfg_rate: float
    max_prompt_length_sec: float
    stream: Optional[StreamConfig] = None


class SeedVCModelSet:
    def __init__(
        self,
        *,
        model: Any,
        semantic_fn: Any,
        vocoder_fn: Any,
        campplus_model: Any,
        to_mel: Any,
        mel_fn_args: dict[str, Any],
        sr: int,
        hop_length: int,
        fp16: bool,
    ) -> None:
        self.model = model
        self.semantic_fn = semantic_fn
        self.vocoder_fn = vocoder_fn
        self.campplus_model = campplus_model
        self.to_mel = to_mel
        self.mel_fn_args = mel_fn_args
        self.sr = int(sr)
        self.hop_length = int(hop_length)
        self.fp16 = bool(fp16)


def _load_seedvc_models(
    *,
    seedvc_dir: str,
    device: "torch.device",
    checkpoint_path: str,
    config_path: str,
    fp16: bool,
    hf_repo: str,
    hf_checkpoint_name: str,
    hf_config_name: str,
) -> SeedVCModelSet:
    import torch
    import yaml

    # IMPORTANT: Seed-VC's `modules/` has no `__init__.py` (namespace package), while Amphion's
    # `modules/` is a regular package. If Amphion's repo root is on sys.path (often via ""),
    # Python can resolve `modules.*` to Amphion even if we prepend Seed-VC. Remove Amphion's
    # repo root from sys.path to make Seed-VC imports unambiguous.
    amphion_root = str(Path(__file__).resolve().parents[2])
    _remove_sys_path_entry("")
    _remove_sys_path_entry(amphion_root)

    _add_sys_path_first(seedvc_dir)
    _purge_import_cache("modules")

    with _pushd(seedvc_dir):
        from hf_utils import load_custom_model_from_hf  # type: ignore[import-not-found]
        from modules.commons import (  # type: ignore[import-not-found]
            build_model,
            load_checkpoint,
            recursive_munch,
        )

        if not checkpoint_path:
            checkpoint_path, config_path = load_custom_model_from_hf(
                str(hf_repo),
                str(hf_checkpoint_name),
                str(hf_config_name),
            )
        elif not config_path:
            raise ValueError("config_path must be provided when checkpoint_path is set.")

        cfg = yaml.safe_load(Path(config_path).read_text())
        model_params = recursive_munch(cfg["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")

        sr = int(cfg["preprocess_params"]["sr"])
        hop_length = int(cfg["preprocess_params"]["spect_params"]["hop_length"])

        model, _, _, _ = load_checkpoint(
            model,
            None,
            str(checkpoint_path),
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in model:
            model[key].eval()
            model[key].to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Style encoder (CAMPPlus).
        from modules.campplus.DTDNN import CAMPPlus  # type: ignore[import-not-found]

        campplus_ckpt = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None)
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(str(campplus_ckpt), map_location="cpu"))
        campplus_model.eval().to(device)

        # Vocoder.
        vocoder_type = str(model_params.vocoder.type)
        if vocoder_type == "hifigan":
            from modules.hifigan.generator import HiFTGenerator  # type: ignore[import-not-found]
            from modules.hifigan.f0_predictor import (  # type: ignore[import-not-found]
                ConvRNNF0Predictor,
            )

            hift_cfg = yaml.safe_load(Path("configs/hifigan.yml").read_text())
            vocoder_fn = HiFTGenerator(
                **hift_cfg["hift"],
                f0_predictor=ConvRNNF0Predictor(**hift_cfg["f0_predictor"]),
            )
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
            vocoder_fn.load_state_dict(torch.load(str(hift_path), map_location="cpu"))
            vocoder_fn.eval().to(device)
        elif vocoder_type == "bigvgan":
            from modules.bigvgan import bigvgan  # type: ignore[import-not-found]

            bigvgan_name = str(model_params.vocoder.name)
            vocoder_fn = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            vocoder_fn.remove_weight_norm()
            vocoder_fn.eval().to(device)
        else:
            raise ValueError(f"Unsupported vocoder type for vc_quest integration: {vocoder_type}")

        # Content encoder / semantic tokenizer.
        tok_type = str(model_params.speech_tokenizer.type)
        if tok_type == "xlsr":
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model  # type: ignore[import-not-found]

            name = str(cfg["model_params"]["speech_tokenizer"]["name"])
            output_layer = int(cfg["model_params"]["speech_tokenizer"]["output_layer"])
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
            wav2vec = Wav2Vec2Model.from_pretrained(name)
            wav2vec.encoder.layers = wav2vec.encoder.layers[:output_layer]
            wav2vec.eval().to(device)
            if fp16 and device.type != "cpu":
                wav2vec.half()

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                waves_16k = waves_16k.float()
                inp_list = [waves_16k[b].detach().cpu().numpy() for b in range(len(waves_16k))]
                inputs = extractor(
                    inp_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000,
                ).to(device)
                with torch.no_grad():
                    x = inputs.input_values
                    if fp16 and device.type != "cpu":
                        x = x.half()
                    out = wav2vec(x)
                return out.last_hidden_state.float()

        elif tok_type == "whisper":
            from transformers import WhisperFeatureExtractor, WhisperModel  # type: ignore[import-not-found]

            name = str(cfg["model_params"]["speech_tokenizer"]["name"])
            extractor = WhisperFeatureExtractor.from_pretrained(name)
            whisper = WhisperModel.from_pretrained(name)
            whisper.eval().to(device)
            if fp16 and device.type != "cpu":
                whisper.half()

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                import torch.nn.functional as F

                waves_16k = waves_16k.float()
                inp_list = [waves_16k[b].detach().cpu().numpy() for b in range(len(waves_16k))]
                inputs = extractor(
                    inp_list,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=16000,
                ).to(device)
                x = inputs.input_features
                # Whisper expects a fixed 30s (3000 frames) mel length.
                # Pad/truncate to match.
                if int(x.size(-1)) < 3000:
                    x = F.pad(x, (0, int(3000 - x.size(-1))))
                elif int(x.size(-1)) > 3000:
                    x = x[..., :3000]
                if fp16 and device.type != "cpu":
                    x = x.half()
                with torch.no_grad():
                    out = whisper.encoder(x)
                return out.last_hidden_state.float()

        else:
            raise ValueError(f"Unsupported tokenizer type for vc_quest integration: {tok_type}")

        mel_fn_args = {
            "n_fft": int(cfg["preprocess_params"]["spect_params"]["n_fft"]),
            "win_size": int(cfg["preprocess_params"]["spect_params"]["win_length"]),
            "hop_size": int(cfg["preprocess_params"]["spect_params"]["hop_length"]),
            "num_mels": int(cfg["preprocess_params"]["spect_params"]["n_mels"]),
            "sampling_rate": int(sr),
            "fmin": float(cfg["preprocess_params"]["spect_params"].get("fmin", 0.0) or 0.0),
            "fmax": float(sr) / 2.0
            if (cfg["preprocess_params"]["spect_params"].get("fmax") in (None, "None", "null", ""))
            else float(cfg["preprocess_params"]["spect_params"].get("fmax", float(sr) / 2.0)),
            "center": False,
        }

        from modules.audio import mel_spectrogram  # type: ignore[import-not-found]

        def to_mel(x: torch.Tensor) -> torch.Tensor:
            return mel_spectrogram(x, **mel_fn_args)

    return SeedVCModelSet(
        model=model,
        semantic_fn=semantic_fn,
        vocoder_fn=vocoder_fn,
        campplus_model=campplus_model,
        to_mel=to_mel,
        mel_fn_args=mel_fn_args,
        sr=sr,
        hop_length=hop_length,
        fp16=fp16,
    )


def _crossfade(prev: np.ndarray, cur: np.ndarray, overlap: int) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float32).reshape(-1)
    cur = np.asarray(cur, dtype=np.float32).reshape(-1)
    overlap = int(min(overlap, len(prev), len(cur)))
    if overlap <= 0:
        return cur
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap, dtype=np.float32)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap, dtype=np.float32)) ** 2
    cur[:overlap] = cur[:overlap] * fade_in + prev[-overlap:] * fade_out
    return cur


@contextlib.contextmanager
def _autocast_if_needed(device: "torch.device", fp16: bool):
    import torch

    if not fp16:
        yield
        return
    if device.type == "cpu":
        yield
        return
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        yield


@contextlib.contextmanager
def _torch_inference_mode():
    import torch

    # Seed-VC's HiFT vocoder (weight_norm + f0_predictor) is not compatible with
    # `torch.inference_mode()` on some PyTorch versions ("Inference tensors cannot be saved for backward").
    # `torch.no_grad()` is still deterministic and safe for inference here.
    with torch.no_grad():
        yield


def _offline_convert(
    *,
    model_set: SeedVCModelSet,
    device: "torch.device",
    src_wav: np.ndarray,
    ref_wav: np.ndarray,
    seed: int,
    diffusion_steps: int,
    inference_cfg_rate: float,
    length_adjust: float,
    max_prompt_length_sec: float,
) -> np.ndarray:
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi

    sr = int(model_set.sr)
    hop = int(model_set.hop_length)
    src_wav = np.asarray(src_wav, dtype=np.float32).reshape(-1)
    ref_wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
    if max_prompt_length_sec > 0:
        ref_wav = ref_wav[: int(round(float(max_prompt_length_sec) * float(sr)))]

    src_t = torch.from_numpy(src_wav).unsqueeze(0).to(device)
    ref_t = torch.from_numpy(ref_wav).unsqueeze(0).to(device)

    converted_16k = torchaudio.functional.resample(src_t, sr, 16000)
    S_alt = model_set.semantic_fn(converted_16k)

    ori_16k = torchaudio.functional.resample(ref_t, sr, 16000)
    S_ori = model_set.semantic_fn(ori_16k)

    mel = model_set.to_mel(src_t.float())
    mel2 = model_set.to_mel(ref_t.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * float(length_adjust))]).to(device)
    target2_lengths = torch.LongTensor([int(mel2.size(2))]).to(device)

    feat2 = kaldi.fbank(
        ori_16k,
        num_mel_bins=80,
        dither=0.0,
        sample_frequency=16000.0,
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = model_set.campplus_model(feat2.unsqueeze(0))

    prompt_condition = model_set.model.length_regulator(
        S_ori,
        ylens=target2_lengths,
        n_quantizers=3,
        f0=None,
    )[0]
    cond = model_set.model.length_regulator(
        S_alt,
        ylens=target_lengths,
        n_quantizers=3,
        f0=None,
    )[0]

    max_context_window = int(sr // hop) * 30
    overlap_frame_len = 16
    overlap_wave_len = int(overlap_frame_len * hop)

    max_source_window = max(1, int(max_context_window - int(mel2.size(2))))
    processed = 0
    previous_chunk: Optional[np.ndarray] = None
    chunks: list[np.ndarray] = []
    chunk_idx = 0

    while processed < int(cond.size(1)):
        chunk_cond = cond[:, processed : processed + max_source_window]
        is_last = processed + max_source_window >= int(cond.size(1))
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        _set_determinism(int(seed) + int(chunk_idx))
        with _torch_inference_mode():
            with _autocast_if_needed(device, bool(model_set.fp16)):
                vc_target = model_set.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    mel2,
                    style2,
                    None,
                    n_timesteps=int(diffusion_steps),
                    inference_cfg_rate=float(inference_cfg_rate),
                )
        vc_target = vc_target[:, :, mel2.size(-1) :]

        # Seed-VC's CFM inference can return "inference tensors" on some PyTorch versions
        # (internally using inference_mode). HiFT's weight_norm stack isn't compatible with
        # inference tensors (it tries to save tensors for backward), so force a regular tensor.
        vc_target = vc_target.float().clone()
        vc_wave = model_set.vocoder_fn(vc_target).squeeze()
        vc_wave_np = vc_wave.detach().cpu().float().numpy().reshape(-1).astype(np.float32, copy=False)

        if processed == 0:
            if is_last:
                chunks.append(vc_wave_np)
                break
            chunks.append(vc_wave_np[:-overlap_wave_len])
            previous_chunk = vc_wave_np[-overlap_wave_len:].copy()
            processed += int(vc_target.size(2)) - overlap_frame_len
        elif is_last:
            assert previous_chunk is not None
            chunks.append(_crossfade(previous_chunk, vc_wave_np, overlap_wave_len))
            processed += int(vc_target.size(2)) - overlap_frame_len
            break
        else:
            assert previous_chunk is not None
            chunks.append(_crossfade(previous_chunk, vc_wave_np[:-overlap_wave_len], overlap_wave_len))
            previous_chunk = vc_wave_np[-overlap_wave_len:].copy()
            processed += int(vc_target.size(2)) - overlap_frame_len

        chunk_idx += 1

    out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return np.asarray(out, dtype=np.float32).reshape(-1)


class SeedVCStreamingEngine:
    def __init__(
        self,
        *,
        model_set: SeedVCModelSet,
        device: "torch.device",
        reference_wav: np.ndarray,
        max_prompt_length_sec: float,
        fp16: bool,
        block_ms: int,
        crossfade_ms: int,
        extra_time_ce_ms: int,
        extra_time_ms: int,
        extra_time_right_ms: int,
        diffusion_steps: int,
        inference_cfg_rate: float,
        seed: int,
    ) -> None:
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi

        self.model_set = model_set
        self.device = device
        self.fp16 = bool(fp16)
        self.seed = int(seed)
        self.diffusion_steps = int(diffusion_steps)
        self.inference_cfg_rate = float(inference_cfg_rate)

        self.sr = int(model_set.sr)
        self.hop_length = int(model_set.hop_length)
        self.zc = int(self.sr // 50)
        if self.zc <= 0:
            raise ValueError(f"Invalid zc (sr//50): sr={self.sr}")

        block_time = float(block_ms) / 1000.0
        crossfade_time = float(crossfade_ms) / 1000.0
        extra_time_ce = float(extra_time_ce_ms) / 1000.0
        extra_time = float(extra_time_ms) / 1000.0
        extra_time_right = float(extra_time_right_ms) / 1000.0

        if extra_time_ce - extra_time < 0:
            raise ValueError("extra_time_ce must be >= extra_time (content encoder context >= DiT context)")

        self.block_frame = int(np.round(block_time * self.sr / self.zc)) * self.zc
        self.crossfade_frame = int(np.round(crossfade_time * self.sr / self.zc)) * self.zc
        self.sola_buffer_frame = int(min(self.crossfade_frame, 4 * self.zc))
        self.sola_search_frame = int(self.zc)

        self.extra_frame = int(np.round(extra_time_ce * self.sr / self.zc)) * self.zc
        self.extra_frame_right = int(np.round(extra_time_right * self.sr / self.zc)) * self.zc

        self.skip_head = int(self.extra_frame // self.zc)
        self.skip_tail = int(self.extra_frame_right // self.zc)
        self.return_length = int((self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc)
        self.ce_dit_frame_difference = int(round((extra_time_ce - extra_time) * 50.0))

        self.buffer_len = int(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right
        )
        self.input_wav = np.zeros(self.buffer_len, dtype=np.float32)

        # 50fps mapping: zc samples @ sr == 320 samples @16k.
        self.block_frame_16k = int(320 * self.block_frame // self.zc)
        self.buffer_len_16k = int(320 * self.buffer_len // self.zc)
        self.input_wav_res = np.zeros(self.buffer_len_16k, dtype=np.float32)

        self.fade_in_window = (np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, self.sola_buffer_frame)) ** 2).astype(
            np.float32
        )
        self.fade_out_window = (1.0 - self.fade_in_window).astype(np.float32)
        self.sola_buffer = np.zeros(self.sola_buffer_frame, dtype=np.float32)

        # Cache prompt conditioning.
        ref = np.asarray(reference_wav, dtype=np.float32).reshape(-1)
        if max_prompt_length_sec > 0:
            ref = ref[: int(round(float(max_prompt_length_sec) * float(self.sr)))]

        ref_t = torch.from_numpy(ref).to(device)
        ori_16k = torchaudio.functional.resample(ref_t, self.sr, 16000)
        S_ori = model_set.semantic_fn(ori_16k.unsqueeze(0))

        feat2 = kaldi.fbank(
            ori_16k.unsqueeze(0),
            num_mel_bins=80,
            dither=0.0,
            sample_frequency=16000.0,
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        self.style2 = model_set.campplus_model(feat2.unsqueeze(0))

        self.mel2 = model_set.to_mel(ref_t.unsqueeze(0))
        target2_lengths = torch.LongTensor([int(self.mel2.size(2))]).to(device)
        self.prompt_condition = model_set.model.length_regulator(
            S_ori,
            ylens=target2_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

    def _infer_window(self, *, window_16k: np.ndarray, window_idx: int) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        waves_16k = torch.from_numpy(np.asarray(window_16k, dtype=np.float32).reshape(-1)).to("cpu")
        _set_determinism(int(self.seed) + int(window_idx))

        with _torch_inference_mode():
            # `semantic_fn` converts the input to numpy internally, so keep the input on CPU
            # to avoid an unnecessary CPU->GPU->CPU round-trip.
            S_alt = self.model_set.semantic_fn(waves_16k.unsqueeze(0))

            if self.ce_dit_frame_difference > 0:
                S_alt = S_alt[:, self.ce_dit_frame_difference :]

            # Length in mel frames (integer math; matches upstream real-time logic).
            total_frames_50fps = int(self.skip_head + self.return_length + self.skip_tail)
            total_frames_50fps = int(total_frames_50fps - self.ce_dit_frame_difference)
            total_frames_50fps = max(1, total_frames_50fps)
            target_mel_frames = int(total_frames_50fps * self.sr // (50 * self.hop_length))
            target_lengths = torch.LongTensor([target_mel_frames]).to(self.device)

            cond = self.model_set.model.length_regulator(
                S_alt,
                ylens=target_lengths,
                n_quantizers=3,
                f0=None,
            )[0]
            cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

            with _autocast_if_needed(self.device, self.fp16):
                vc_target = self.model_set.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(self.device),
                    self.mel2,
                    self.style2,
                    None,
                    n_timesteps=int(self.diffusion_steps),
                    inference_cfg_rate=float(self.inference_cfg_rate),
                )
            vc_target = vc_target[:, :, self.mel2.size(-1) :]
            vc_target = vc_target.float().clone()
            vc_wave = self.model_set.vocoder_fn(vc_target).squeeze()

        out_len = int(self.return_length * self.sr // 50)
        tail_len = int(self.skip_tail * self.sr // 50)
        vc_wave = vc_wave.detach().cpu().float().numpy().reshape(-1).astype(np.float32, copy=False)

        if tail_len > 0:
            seg = vc_wave[-out_len - tail_len : -tail_len]
        else:
            seg = vc_wave[-out_len:]
        seg = np.asarray(seg, dtype=np.float32).reshape(-1)

        # SOLA alignment (same as upstream real-time GUI).
        if self.sola_buffer_frame > 0:
            head = seg[: self.sola_buffer_frame + self.sola_search_frame].astype(np.float32, copy=False)
            conv_input = torch.from_numpy(head).to(self.device)[None, None, :]
            sola_buf = torch.from_numpy(self.sola_buffer).to(self.device)[None, None, :]

            cor_nom = F.conv1d(conv_input, sola_buf)[0, 0]
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
                )[0, 0]
                + 1e-8
            )
            tensor = cor_nom / cor_den
            sola_offset = int(torch.argmax(tensor, dim=0).item()) if tensor.numel() > 1 else int(tensor.item())

            seg = seg[sola_offset:]
            fade_len = min(self.sola_buffer_frame, len(seg))
            seg[:fade_len] = seg[:fade_len] * self.fade_in_window[:fade_len] + self.sola_buffer[:fade_len] * self.fade_out_window[
                :fade_len
            ]

            # Update SOLA buffer with the next overlap segment.
            tail_start = int(self.block_frame)
            tail_end = int(self.block_frame + self.sola_buffer_frame)
            if tail_end <= len(seg):
                self.sola_buffer[:] = seg[tail_start:tail_end]
            else:
                buf = np.zeros(self.sola_buffer_frame, dtype=np.float32)
                avail = max(0, len(seg) - tail_start)
                if avail > 0:
                    buf[:avail] = seg[tail_start:]
                self.sola_buffer[:] = buf

        block = seg[: self.block_frame]
        if len(block) < self.block_frame:
            block = np.pad(block, (0, self.block_frame - len(block)), mode="constant")
        return np.asarray(block, dtype=np.float32).reshape(-1)

    def step(self, *, hop: np.ndarray, window_idx: int) -> np.ndarray:
        hop = np.asarray(hop, dtype=np.float32).reshape(-1)
        if len(hop) != self.block_frame:
            raise ValueError(f"Expected hop of {self.block_frame} samples, got {len(hop)}")

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :]
        self.input_wav[-self.block_frame :] = hop

        # Incremental resample update (matches upstream real-time GUI logic).
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :]
        seg = self.input_wav[-self.block_frame - 2 * self.zc :].astype(np.float32, copy=False)
        seg_16k = _resample_np(seg, orig_sr=self.sr, target_sr=16000)
        seg_16k = seg_16k[320:]  # drop 1 frame to keep alignment

        expected = int(320 * (self.block_frame // self.zc + 1))
        seg_16k = seg_16k[:expected]
        if len(seg_16k) < expected:
            seg_16k = np.pad(seg_16k, (0, expected - len(seg_16k)), mode="constant")
        self.input_wav_res[-expected:] = seg_16k

        return self._infer_window(window_16k=self.input_wav_res, window_idx=window_idx)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-VC zero-shot VC runner (offline or streaming simulation).")
    parser.add_argument("--seedvc_dir", type=str, required=True, help="Path to seed-vc repo checkout.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--checkpoint_path", type=str, default="", help="Optional checkpoint (else downloads xlsr-tiny).")
    parser.add_argument("--config_path", type=str, default="", help="Optional config (required if checkpoint_path set).")
    parser.add_argument("--hf_repo", type=str, default="Plachta/Seed-VC", help="HF repo for default checkpoint/config.")
    parser.add_argument(
        "--hf_checkpoint_name",
        type=str,
        default="DiT_uvit_tat_xlsr_ema.pth",
        help="HF checkpoint filename to download when checkpoint_path is empty.",
    )
    parser.add_argument(
        "--hf_config_name",
        type=str,
        default="config_dit_mel_seed_uvit_xlsr_tiny.yml",
        help="HF config filename to download when checkpoint_path is empty.",
    )

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice audio")
    parser.add_argument("--src", type=str, required=True, help="Source audio to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7)
    parser.add_argument("--length_adjust", type=float, default=1.0)
    parser.add_argument("--max_prompt_length_sec", type=float, default=3.0)

    parser.add_argument("--stream", action="store_true", help="Run streaming simulation (block-based + SOLA).")
    parser.add_argument("--window_ms", type=int, default=300, help="Chunk size (maps to Seed-VC block_time).")
    parser.add_argument("--hop_ms", type=int, default=300, help="Chunk hop (must equal window_ms for Seed-VC).")
    parser.add_argument("--crossfade_ms", type=int, default=40)
    parser.add_argument("--extra_time_ce_ms", type=int, default=2500)
    parser.add_argument("--extra_time_ms", type=int, default=500)
    parser.add_argument("--extra_time_right_ms", type=int, default=20)
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop algorithmic delay so output aligns to source start (recommended for eval).",
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

    import torch

    seedvc_dir = os.path.abspath(str(args.seedvc_dir))
    if not os.path.isdir(seedvc_dir):
        raise FileNotFoundError(f"seedvc_dir not found: {seedvc_dir}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    )

    model_set = _load_seedvc_models(
        seedvc_dir=seedvc_dir,
        device=device,
        checkpoint_path=str(args.checkpoint_path),
        config_path=str(args.config_path),
        fp16=bool(args.fp16),
        hf_repo=str(args.hf_repo),
        hf_checkpoint_name=str(args.hf_checkpoint_name),
        hf_config_name=str(args.hf_config_name),
    )

    sr = int(model_set.sr)
    ref_wav, _ = _load_audio_mono(args.ref, sr=sr)
    src_wav, _ = _load_audio_mono(args.src, sr=sr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0
    warmup_hops = 0

    if not args.stream:
        t0 = time.perf_counter()
        out = _offline_convert(
            model_set=model_set,
            device=device,
            src_wav=src_wav,
            ref_wav=ref_wav,
            seed=int(args.seed),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.inference_cfg_rate),
            length_adjust=float(args.length_adjust),
            max_prompt_length_sec=float(args.max_prompt_length_sec),
        )
        _torch_sync(device)
        timings.append(time.perf_counter() - t0)
        sf.write(args.out, out, sr)
    else:
        if int(args.hop_ms) != int(args.window_ms):
            raise ValueError("Seed-VC streaming uses hop_ms == window_ms (block_time).")
        block_ms = int(args.window_ms)
        hop_ms = int(args.hop_ms)

        engine = SeedVCStreamingEngine(
            model_set=model_set,
            device=device,
            reference_wav=ref_wav,
            max_prompt_length_sec=float(args.max_prompt_length_sec),
            fp16=bool(args.fp16),
            block_ms=block_ms,
            crossfade_ms=int(args.crossfade_ms),
            extra_time_ce_ms=int(args.extra_time_ce_ms),
            extra_time_ms=int(args.extra_time_ms),
            extra_time_right_ms=int(args.extra_time_right_ms),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.inference_cfg_rate),
            seed=int(args.seed),
        )

        block = int(engine.block_frame)
        total_blocks = int(np.ceil(len(src_wav) / float(block)))

        # Approx algorithm delay (as per Seed-VC README).
        algo_delay_sec = 2.0 * (float(block_ms) / 1000.0) + float(args.extra_time_right_ms) / 1000.0
        algo_delay_samples = int(round(algo_delay_sec * float(sr)))

        # Run for extra time to flush tail.
        pad_len = int(algo_delay_samples + block)
        src_pad = np.pad(src_wav, (0, pad_len), mode="constant")

        outs: list[np.ndarray] = []
        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(hop_ms), 1e-6)))
        hangover_left = 0

        for i, start in enumerate(range(0, len(src_pad), block)):
            hop = src_pad[start : start + block]
            if len(hop) < block:
                hop = np.pad(hop, (0, block - len(hop)), mode="constant")

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
                hop_16k = _resample_np(hop, orig_sr=sr, target_sr=16000)
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
                out_block = np.zeros(block, dtype=np.float32)
                engine.sola_buffer[:] = 0.0
            else:
                t0 = time.perf_counter()
                out_block = engine.step(hop=hop, window_idx=i)
                _torch_sync(device)
                timings.append(time.perf_counter() - t0)

            out_block = apply_peak_limiter(out_block, peak_limit=float(args.peak_limit))
            outs.append(out_block)

        out_full = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

        if bool(args.drop_warmup_hops):
            delay_samples = 0
            # Remove algorithmic delay, then trim back to the original source length.
            start = int(max(0, algo_delay_samples))
            out_full = out_full[start : start + len(src_wav)]
        else:
            delay_samples = int(algo_delay_samples)
            out_full = out_full[: len(src_wav)]

        warmup_hops = int(np.ceil(float(algo_delay_samples) / float(block))) if algo_delay_samples > 0 else 0
        out_full = np.asarray(out_full, dtype=np.float32).reshape(-1)
        sf.write(args.out, out_full, sr)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                crossfade_ms=int(args.crossfade_ms),
                extra_time_ce_ms=int(args.extra_time_ce_ms),
                extra_time_ms=int(args.extra_time_ms),
                extra_time_right_ms=int(args.extra_time_right_ms),
                diffusion_steps=int(args.diffusion_steps),
                inference_cfg_rate=float(args.inference_cfg_rate),
                max_prompt_length_sec=float(args.max_prompt_length_sec),
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
            seedvc_dir=str(seedvc_dir),
            device=str(device),
            seed=int(args.seed),
            checkpoint_path=str(args.checkpoint_path),
            config_path=str(args.config_path),
            hf_repo=str(args.hf_repo),
            hf_checkpoint_name=str(args.hf_checkpoint_name),
            hf_config_name=str(args.hf_config_name),
            fp16=bool(args.fp16),
            length_adjust=float(args.length_adjust),
            diffusion_steps=int(args.diffusion_steps),
            inference_cfg_rate=float(args.inference_cfg_rate),
            max_prompt_length_sec=float(args.max_prompt_length_sec),
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
            "output_sample_rate": int(sr),
        }
        meta_p.write_text(json.dumps({"config": asdict(cfg), "stats": stats}, indent=2))

    print(f"Wrote: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
