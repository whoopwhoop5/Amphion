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
from typing import Literal, Optional

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


VadMode = Literal["rms", "webrtc", "off"]


def _add_sys_path_first(path: str) -> None:
    path = os.path.abspath(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


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


def _torch_sync(device: "torch.device") -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _load_audio_mono(path: str, *, sr: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def _trim_or_pad_ref(wav: np.ndarray, *, sample_rate: int, max_sec: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if max_sec <= 0:
        return wav
    max_len = int(round(float(max_sec) * float(sample_rate)))
    if len(wav) <= max_len:
        return wav
    return wav[:max_len]


@dataclass(frozen=True)
class StreamConfig:
    window_ms: int
    hop_ms: int
    crossfade_ms: int
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
    samoye_dir: str
    device: str
    seed: int
    fp16: bool
    config_path: str
    model_path: str
    whisper_path: str
    hubert_path: str
    speaker_encoder_path: str
    reference_max_sec: float
    stream: Optional[StreamConfig] = None


class SaMoyeModelSet:
    def __init__(
        self,
        *,
        model,
        hp,
        whisper_model,
        hubert_model,
        sr: int,
        hop_length: int,
        fp16: bool,
    ) -> None:
        self.model = model
        self.hp = hp
        self.whisper_model = whisper_model
        self.hubert_model = hubert_model
        self.sr = int(sr)
        self.hop_length = int(hop_length)
        self.fp16 = bool(fp16)


def _load_samoye_models(
    *,
    samoye_dir: str,
    device: "torch.device",
    fp16: bool,
    config_path: str,
    model_path: str,
    whisper_path: str,
    hubert_path: str,
) -> SaMoyeModelSet:
    import torch

    saved_sys_path = list(sys.path)
    try:
        amphion_root = str(Path(__file__).resolve().parents[2])
        _remove_sys_path_entry("")
        _remove_sys_path_entry(amphion_root)
        _add_sys_path_first(samoye_dir)

        with _pushd(samoye_dir):
            from omegaconf import OmegaConf  # type: ignore[import-not-found]

            from vits.models import SynthesizerInfer  # type: ignore[import-not-found]

            from whisper_svc.model import (  # type: ignore[import-not-found]
                ModelDimensions,
                Whisper,
            )
            from hubert import hubert_model  # type: ignore[import-not-found]
    finally:
        sys.path = saved_sys_path

    with _pushd(samoye_dir):
        hp = OmegaConf.load(config_path)
        spec_channels = int(hp.data.filter_length // 2 + 1)
        segment_size = int(hp.data.segment_size // hp.data.hop_length)
        model = SynthesizerInfer(spec_channels, segment_size, hp)
        ckpt = torch.load(model_path, map_location="cpu")
        if "model_g" in ckpt:
            model.load_state_dict(ckpt["model_g"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        model.eval()
        model.to(device)
        if fp16 and device.type == "cuda":
            model.half()
            # SourceModuleHnNSF mixes float buffers with float activations. SineGen currently emits
            # float32 even when the module is cast to fp16, so keep the merge weights in fp32 and
            # cast the resulting harmonic source to fp16 in the wrapper.
            try:
                model.dec.m_source.merge_w = model.dec.m_source.merge_w.float()
                model.dec.m_source.merge_b = model.dec.m_source.merge_b.float()
            except Exception:
                pass

        # Whisper PPG model (same as whisper_svc/inference.py, but loaded once).
        whisper_ckpt = torch.load(whisper_path, map_location="cpu")
        dims = ModelDimensions(**whisper_ckpt["dims"])
        whisper_model = Whisper(dims)
        # We only need the encoder.
        del whisper_model.decoder
        cut = len(whisper_model.encoder.blocks) // 4
        del whisper_model.encoder.blocks[-cut:]
        whisper_model.load_state_dict(whisper_ckpt["model_state_dict"], strict=False)
        whisper_model.eval()
        whisper_model.to(device)
        if fp16 and device.type == "cuda":
            whisper_model.half()

        hubert = hubert_model.hubert_soft(hubert_path)
        hubert.eval()
        hubert.to(device)
        if fp16 and device.type == "cuda":
            hubert.half()

    sr = int(hp.data.sampling_rate)
    hop_length = int(hp.data.hop_length)
    return SaMoyeModelSet(
        model=model,
        hp=hp,
        whisper_model=whisper_model,
        hubert_model=hubert,
        sr=sr,
        hop_length=hop_length,
        fp16=fp16,
    )


def _compute_ppg(
    whisper_model,
    *,
    wav_16k: np.ndarray,
    device: "torch.device",
    fp16: bool,
) -> np.ndarray:
    import torch

    from whisper_svc.audio import log_mel_spectrogram  # type: ignore[import-not-found]

    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    audln = int(wav_16k.shape[0])

    parts: list[np.ndarray] = []
    idx_s = 0
    while idx_s + 15 * 16000 < audln:
        short = wav_16k[idx_s : idx_s + 15 * 16000]
        idx_s += 15 * 16000
        ppgln = int(15 * 16000 // 320)

        mel = log_mel_spectrogram(short).to(device)
        if fp16 and device.type == "cuda":
            mel = mel.half()

        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0)
        ppg_np = ppg.detach().float().cpu().numpy()
        parts.append(ppg_np[:ppgln])

    if idx_s < audln:
        short = wav_16k[idx_s:audln]
        ppgln = int((audln - idx_s) // 320)

        mel = log_mel_spectrogram(short).to(device)
        if fp16 and device.type == "cuda":
            mel = mel.half()
        with torch.no_grad():
            mel = mel + torch.randn_like(mel) * 0.1
            ppg = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0)
        ppg_np = ppg.detach().float().cpu().numpy()
        parts.append(ppg_np[:ppgln])

    ppg_all = np.concatenate(parts, axis=0) if parts else np.zeros((0, 1280), dtype=np.float32)
    # 320-hop -> 160-hop (x2)
    return np.repeat(ppg_all, 2, axis=0).astype(np.float32, copy=False)


def _compute_vec(
    hubert,
    *,
    wav_16k: np.ndarray,
    device: "torch.device",
    fp16: bool,
) -> np.ndarray:
    import torch

    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    audln = int(wav_16k.shape[0])

    parts: list[np.ndarray] = []
    idx_s = 0
    while idx_s + 20 * 16000 < audln:
        seg = wav_16k[idx_s : idx_s + 20 * 16000]
        idx_s += 20 * 16000
        feats = torch.from_numpy(seg).to(device)[None, None, :]
        if fp16 and device.type == "cuda":
            feats = feats.half()
        with torch.no_grad():
            vec = hubert.units(feats).squeeze(0)
        parts.append(vec.detach().float().cpu().numpy())

    if idx_s < audln:
        seg = wav_16k[idx_s:audln]
        feats = torch.from_numpy(seg).to(device)[None, None, :]
        if fp16 and device.type == "cuda":
            feats = feats.half()
        with torch.no_grad():
            vec = hubert.units(feats).squeeze(0)
        parts.append(vec.detach().float().cpu().numpy())

    vec_all = np.concatenate(parts, axis=0) if parts else np.zeros((0, 256), dtype=np.float32)
    return np.repeat(vec_all, 2, axis=0).astype(np.float32, copy=False)


def _compute_f0(
    *,
    wav_16k: np.ndarray,
    device: "torch.device",
) -> np.ndarray:
    wav_16k = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
    _ = device

    import pyworld  # type: ignore[import-not-found]

    x = wav_16k.astype(np.double, copy=False)
    f0, t = pyworld.dio(
        x,
        fs=16000,
        f0_ceil=900,
        frame_period=10.0,  # ms (matches 10ms SVC frames)
    )
    f0 = pyworld.stonemask(x, f0, t, fs=16000)
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    return np.round(f0, 1)


def _infer_full(
    *,
    model_set: SaMoyeModelSet,
    device: "torch.device",
    spk_wav_path: list[str],
    ppg: np.ndarray,
    vec: np.ndarray,
    pit: np.ndarray,
) -> np.ndarray:
    import torch

    ppg = np.asarray(ppg, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    pit = np.asarray(pit, dtype=np.float32).reshape(-1)

    all_frame = int(min(len(pit), len(ppg), len(vec)))
    if all_frame <= 0:
        return np.zeros(0, dtype=np.float32)
    pit = pit[:all_frame]
    ppg = ppg[:all_frame]
    vec = vec[:all_frame]

    hp = model_set.hp
    hop_size = int(model_set.hop_length)

    pit_t = torch.from_numpy(pit).to(device)
    ppg_t = torch.from_numpy(ppg).to(device)
    vec_t = torch.from_numpy(vec).to(device)
    if model_set.fp16 and device.type == "cuda":
        ppg_t = ppg_t.half()
        vec_t = vec_t.half()

    with torch.no_grad():
        source = model_set.model.pitch2source(pit_t.unsqueeze(0))
        source = source.to(dtype=ppg_t.dtype)

    hop_frame = 10
    out_chunk = 2500  # frames (~25s at 10ms frame)
    out_index = 0
    out_audio: list[np.ndarray] = []

    with torch.no_grad():
        while out_index < all_frame:
            if out_index == 0:
                cut_s = 0
                cut_s_out = 0
            else:
                cut_s = int(out_index - hop_frame)
                cut_s_out = int(hop_frame * hop_size)

            if out_index + out_chunk + hop_frame > all_frame:
                cut_e = int(all_frame)
                cut_e_out = -1
            else:
                cut_e = int(out_index + out_chunk + hop_frame)
                cut_e_out = int(-hop_frame * hop_size)

            sub_ppg = ppg_t[cut_s:cut_e].unsqueeze(0)
            sub_vec = vec_t[cut_s:cut_e].unsqueeze(0)
            sub_pit = pit_t[cut_s:cut_e].unsqueeze(0)
            sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
            sub_har = source[:, :, cut_s * hop_size : cut_e * hop_size]

            sub_out = model_set.model.inference(sub_ppg, sub_vec, sub_pit, spk_wav_path, sub_len, sub_har)
            sub_out_np = sub_out[0, 0].detach().float().cpu().numpy().reshape(-1)
            if cut_e_out == -1:
                out_audio.append(sub_out_np[cut_s_out:])
            else:
                out_audio.append(sub_out_np[cut_s_out:cut_e_out])

            out_index += out_chunk

    out = np.concatenate(out_audio) if out_audio else np.zeros(0, dtype=np.float32)
    return out.astype(np.float32, copy=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SaMoye-SVC runner (offline or streaming sim).")
    parser.add_argument("--samoye_dir", type=str, required=True, help="Path to SaMoye-Model directory.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="YAML config path. Defaults to <samoye_dir>/configs/sovits_spk_1700h.yaml",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Model checkpoint path. Defaults to <samoye_dir>/sovits_spk_1700h_0020.pt",
    )
    parser.add_argument(
        "--whisper_path",
        type=str,
        default="",
        help="Whisper checkpoint path. Defaults to <samoye_dir>/whisper_pretrain/large-v2.pt",
    )
    parser.add_argument(
        "--hubert_path",
        type=str,
        default="",
        help="HuBERT-soft checkpoint path. Defaults to <samoye_dir>/hubert_pretrain/hubert-soft-0d54a1f4.pt",
    )
    parser.add_argument(
        "--speaker_encoder_path",
        type=str,
        default="",
        help="Speaker encoder checkpoint path. Defaults to <samoye_dir>/speaker_pretrain/best_model.pth.tar",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda:0, cpu")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed base (per-window adds index).")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference_max_sec", type=float, default=10.0, help="Trim reference audio to this many seconds.")

    parser.add_argument("--ref", type=str, required=True, help="Target/reference voice wav")
    parser.add_argument("--src", type=str, required=True, help="Source wav to convert")
    parser.add_argument("--out", type=str, required=True, help="Output wav path")
    parser.add_argument("--meta_json", type=str, default="", help="Optional JSON path to write run metadata")

    parser.add_argument("--stream", action="store_true", help="Run window/hop streaming simulation.")
    parser.add_argument("--window_ms", type=int, default=600)
    parser.add_argument("--hop_ms", type=int, default=300)
    parser.add_argument("--crossfade_ms", type=int, default=10)
    parser.add_argument(
        "--drop_warmup_hops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute warmup hops count for reporting (no output dropping currently).",
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

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    _set_determinism(int(args.seed))

    samoye_dir = str(Path(args.samoye_dir).resolve())
    config_path = (
        str(Path(args.config_path).resolve())
        if str(args.config_path).strip()
        else str(Path(samoye_dir).joinpath("configs/sovits_spk_1700h.yaml").resolve())
    )
    model_path = (
        str(Path(args.model_path).resolve())
        if str(args.model_path).strip()
        else str(Path(samoye_dir).joinpath("sovits_spk_1700h_0020.pt").resolve())
    )
    whisper_path = (
        str(Path(args.whisper_path).resolve())
        if str(args.whisper_path).strip()
        else str(Path(samoye_dir).joinpath("whisper_pretrain/large-v2.pt").resolve())
    )
    hubert_path = (
        str(Path(args.hubert_path).resolve())
        if str(args.hubert_path).strip()
        else str(Path(samoye_dir).joinpath("hubert_pretrain/hubert-soft-0d54a1f4.pt").resolve())
    )
    speaker_encoder_path = (
        str(Path(args.speaker_encoder_path).resolve())
        if str(args.speaker_encoder_path).strip()
        else str(Path(samoye_dir).joinpath("speaker_pretrain/best_model.pth.tar").resolve())
    )

    for req in [config_path, model_path, whisper_path, hubert_path, speaker_encoder_path]:
        if not Path(req).exists():
            raise FileNotFoundError(f"Missing required file: {req}")

    _add_sys_path_first(samoye_dir)

    models = _load_samoye_models(
        samoye_dir=samoye_dir,
        device=device,
        fp16=bool(args.fp16),
        config_path=config_path,
        model_path=model_path,
        whisper_path=whisper_path,
        hubert_path=hubert_path,
    )

    # Speaker embedding caching: compute once, then override SpkEncoderHelper.forward.
    saved_sys_path = list(sys.path)
    try:
        amphion_root = str(Path(__file__).resolve().parents[2])
        _remove_sys_path_entry("")
        _remove_sys_path_entry(amphion_root)
        _add_sys_path_first(samoye_dir)
        from prepare.preprocess_speaker import SpkEncoderHelper  # type: ignore[import-not-found]
    finally:
        sys.path = saved_sys_path

    spk_helper = SpkEncoderHelper(root_path=samoye_dir)
    spk_helper.to(device)
    with torch.no_grad():
        ref_16k = _load_audio_mono(args.ref, sr=16000)
        ref_16k = _trim_or_pad_ref(ref_16k, sample_rate=16000, max_sec=float(args.reference_max_sec))
        tmp_ref_path = str(Path(args.out).with_suffix(".ref16k_tmp.wav"))
        Path(tmp_ref_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(tmp_ref_path, ref_16k, 16000)
        spk_embed = spk_helper.forward([tmp_ref_path], infer=True).detach()
        if models.fp16 and device.type == "cuda":
            spk_embed = spk_embed.half()
        try:
            Path(tmp_ref_path).unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass

    def _cached_forward(_wav_files, infer: bool = False):
        _ = infer
        return spk_embed

    models.model.spk_encoder_helper = spk_helper
    models.model.spk_encoder_helper.forward = _cached_forward  # type: ignore[method-assign]

    # Features on 16k source waveform.
    wav_16k = _load_audio_mono(args.src, sr=16000)
    ppg = _compute_ppg(models.whisper_model, wav_16k=wav_16k, device=device, fp16=models.fp16)
    vec = _compute_vec(models.hubert_model, wav_16k=wav_16k, device=device, fp16=models.fp16)
    pit = _compute_f0(wav_16k=wav_16k, device=device)

    all_frame = int(min(len(pit), len(ppg), len(vec)))
    ppg = ppg[:all_frame]
    vec = vec[:all_frame]
    pit = pit[:all_frame]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    delay_samples = 0

    if not bool(args.stream):
        _torch_sync(device)
        t0 = time.time()
        out = _infer_full(
            model_set=models,
            device=device,
            spk_wav_path=[args.ref],
            ppg=ppg,
            vec=vec,
            pit=pit,
        )
        _torch_sync(device)
        timings.append(time.time() - t0)
        out = apply_peak_limiter(out, peak_limit=float(args.peak_limit))
        sf.write(str(out_path), out.astype(np.float32, copy=False), models.sr)
        warmup_hops = 0
    else:
        window_frames = int(round(float(args.window_ms) / 1000.0 / 0.01))
        hop_frames = int(round(float(args.hop_ms) / 1000.0 / 0.01))
        window_frames = max(1, window_frames)
        hop_frames = max(1, hop_frames)
        if hop_frames > window_frames:
            raise ValueError("hop_ms must be <= window_ms")

        overlap_frames = int(max(0, window_frames - hop_frames))
        window_wave_len = int(window_frames * models.hop_length)
        hop_wave_len = int(hop_frames * models.hop_length)
        overlap_wave_len = int(overlap_frames * models.hop_length)
        crossfade_wave_len = int(round(float(args.crossfade_ms) / 1000.0 * float(models.sr)))
        crossfade_wave_len = int(min(max(0, crossfade_wave_len), overlap_wave_len))

        warmup_hops = int(np.ceil(float(window_frames) / float(hop_frames))) - 1 if hop_frames > 0 else 0
        warmup_hops = int(max(0, warmup_hops))

        hangover_hops = int(np.ceil(float(args.vad_hangover_ms) / max(float(args.hop_ms), 1e-6)))
        hangover_left = 0

        # Precompute harmonic source once and slice.
        import torch

        pit_t = torch.from_numpy(pit).to(device)
        with torch.no_grad():
            har_source_full = models.model.pitch2source(pit_t.unsqueeze(0))
            if models.fp16 and device.type == "cuda":
                har_source_full = har_source_full.half()

        processed = 0
        window_count = 0
        prev_tail: Optional[np.ndarray] = None
        outs: list[np.ndarray] = []

        while processed < all_frame:
            remaining = int(all_frame - processed)
            chunk_len = int(min(window_frames, remaining))
            is_last = processed + window_frames >= all_frame

            # Hop waveform for VAD (use source timeline).
            # Approximate 32k timeline by scaling 16k samples x2.
            # For VAD we always resample to 16k anyway, so use the source 16k hop slice directly.
            hop_16k_start = int(processed * 160)
            hop_16k_end = hop_16k_start + int(min(int(args.hop_ms / 1000.0 * 16000), max(0, len(wav_16k) - hop_16k_start)))
            hop_wav_16k = wav_16k[hop_16k_start:hop_16k_end]
            if hop_wav_16k.size == 0:
                hop_wav_16k = np.zeros(int(args.hop_ms / 1000.0 * 16000), dtype=np.float32)

            vad_mode = str(args.vad_mode)
            if vad_mode == "off":
                voiced = True
            elif vad_mode == "rms":
                voiced = not (
                    float(args.vad_db) > -200.0
                    and is_silent_rms_db(
                        hop_wav_16k,
                        sample_rate=16000,
                        frame_ms=float(args.vad_frame_ms),
                        silence_db=float(args.vad_db),
                    )
                )
            elif vad_mode == "webrtc":
                voiced = is_voiced_webrtcvad(
                    hop_wav_16k,
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
                chunk_wave = np.zeros(int(chunk_len * models.hop_length), dtype=np.float32)
            else:
                _set_determinism(int(args.seed) + int(window_count))
                _torch_sync(device)
                t0 = time.time()

                # Build window tensors (pad to full window for stable shapes).
                ppg_chunk = ppg[processed : processed + chunk_len]
                vec_chunk = vec[processed : processed + chunk_len]
                pit_chunk = pit[processed : processed + chunk_len]
                if chunk_len < window_frames:
                    pad = window_frames - chunk_len
                    ppg_chunk = np.pad(ppg_chunk, ((0, pad), (0, 0)), mode="constant")
                    vec_chunk = np.pad(vec_chunk, ((0, pad), (0, 0)), mode="constant")
                    pit_chunk = np.pad(pit_chunk, (0, pad), mode="constant")

                import torch

                ppg_t = torch.from_numpy(ppg_chunk).to(device).unsqueeze(0)
                vec_t = torch.from_numpy(vec_chunk).to(device).unsqueeze(0)
                pit_t = torch.from_numpy(pit_chunk).to(device).unsqueeze(0)
                if models.fp16 and device.type == "cuda":
                    ppg_t = ppg_t.half()
                    vec_t = vec_t.half()
                    pit_t = pit_t.half()

                sub_len = torch.LongTensor([chunk_len]).to(device)
                sub_har = har_source_full[:, :, processed * models.hop_length : processed * models.hop_length + window_wave_len]
                if sub_har.size(-1) < window_wave_len:
                    sub_har = torch.nn.functional.pad(sub_har, (0, window_wave_len - sub_har.size(-1)))

                with torch.no_grad():
                    sub_out = models.model.inference(ppg_t, vec_t, pit_t, [args.ref], sub_len, sub_har)
                _torch_sync(device)
                timings.append(time.time() - t0)

                chunk_wave = sub_out[0, 0].detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

            expected = int(chunk_len * models.hop_length)
            chunk_wave = normalize_length(chunk_wave, expected, align="start")

            # Optional overlap smoothing.
            if overlap_wave_len > 0 and prev_tail is not None and crossfade_wave_len > 0:
                chunk_wave = crossfade_prefix_inplace(chunk_wave, prev_tail, int(crossfade_wave_len))

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
        out = normalize_length(out, int(all_frame * models.hop_length), align="start")
        sf.write(str(out_path), out.astype(np.float32, copy=False), models.sr)

    if args.meta_json:
        meta_p = Path(args.meta_json)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        stream_cfg = (
            StreamConfig(
                window_ms=int(args.window_ms),
                hop_ms=int(args.hop_ms),
                crossfade_ms=int(args.crossfade_ms),
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
            samoye_dir=samoye_dir,
            device=str(device),
            seed=int(args.seed),
            fp16=bool(args.fp16),
            config_path=config_path,
            model_path=model_path,
            whisper_path=whisper_path,
            hubert_path=hubert_path,
            speaker_encoder_path=speaker_encoder_path,
            reference_max_sec=float(args.reference_max_sec),
            stream=stream_cfg,
        )
        stats = {
            "delay_samples": int(delay_samples),
            "warmup_hops": int(warmup_hops),
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
