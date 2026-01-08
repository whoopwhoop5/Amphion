# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from models.vc.vevo.runner import VevoConverter, VevoKind


NormalizeAlign = Literal["start", "end"]


def normalize_length(
    wav: np.ndarray,
    target_len: int,
    *,
    align: NormalizeAlign = "end",
) -> np.ndarray:
    if wav.ndim != 1:
        raise ValueError(f"Expected mono wav [T], got shape={wav.shape}")

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


def crossfade_inplace(
    current: np.ndarray,
    prev_tail: Optional[np.ndarray],
    fade_len: int,
) -> np.ndarray:
    if prev_tail is None or fade_len <= 0:
        return current

    fade_len = min(fade_len, len(current), len(prev_tail))
    if fade_len <= 0:
        return current

    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = 1.0 - fade_in
    current[:fade_len] = current[:fade_len] * fade_in + prev_tail[-fade_len:] * fade_out
    return current


def smooth_boundary_inplace(
    current: np.ndarray,
    prev_last: Optional[float],
    fade_len: int,
) -> np.ndarray:
    """Reduce boundary clicks by matching the first sample to the previous chunk end.

    Unlike a true crossfade (which assumes time-overlap), this applies a short, tapering
    offset to the start of the current chunk so the first sample equals `prev_last`.
    """

    if prev_last is None or fade_len <= 0:
        return current

    fade_len = min(fade_len, len(current))
    if fade_len <= 0:
        return current

    delta = float(prev_last) - float(current[0])
    if not np.isfinite(delta):
        return current

    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    current[:fade_len] = (current[:fade_len] + delta * fade).astype(np.float32, copy=False)
    return current


def normalize_rms_db(
    wav: np.ndarray,
    *,
    target_db: float = -25.0,
    eps: float = 1e-9,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(wav * wav) + eps))
    cur_db = 20.0 * float(np.log10(rms + eps))
    gain_db = float(target_db) - cur_db
    gain = float(10.0 ** (gain_db / 20.0))
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


@dataclass
class VevoReferenceState:
    ref24k: torch.Tensor  # [1, T]
    ref16k: torch.Tensor  # [1, T]

    # FM prompt
    timbre_ref_codecs: torch.Tensor  # [1, T]
    timbre_prompt_mels: torch.Tensor  # [1, T, n_mels]

    # AR prompt (vevovoice only)
    style_prompt_output_ids: Optional[torch.Tensor] = None  # [1, T]
    style_prompt_mels: Optional[torch.Tensor] = None  # [1, T, n_mels]


class VevoStreamingEngine:
    model_sr = 24000

    def __init__(self, converter: VevoConverter):
        self.converter = converter
        self.pipeline = converter.pipeline
        self.device = converter.device
        self.kind: VevoKind = converter.kind
        self._ref: Optional[VevoReferenceState] = None

    def prepare_reference_bytes(self, reference_wav_bytes: bytes) -> None:
        wav, sr = sf.read(io.BytesIO(reference_wav_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]

        ref = torch.from_numpy(wav).unsqueeze(0)
        ref = ref.to(self.device)

        if sr != self.model_sr:
            ref24k = torchaudio.functional.resample(ref, sr, self.model_sr)
        else:
            ref24k = ref
        ref16k = torchaudio.functional.resample(ref24k, self.model_sr, 16000)

        with torch.no_grad():
            timbre_ref_codecs, _ = self.pipeline.extract_hubert_codec(
                self.pipeline.content_style_tokenizer,
                ref16k,
                duration_reduction=False,
            )
            timbre_prompt_mels = self.pipeline.extract_mel_feature(ref24k)

            style_prompt_output_ids = None
            style_prompt_mels = None
            if self.kind == "vevovoice":
                style_prompt_output_ids, _ = self.pipeline.extract_hubert_codec(
                    self.pipeline.content_style_tokenizer,
                    ref16k,
                    duration_reduction=False,
                )
                style_prompt_mels = self.pipeline.extract_prompt_mel_feature(ref16k)

        self._ref = VevoReferenceState(
            ref24k=ref24k,
            ref16k=ref16k,
            timbre_ref_codecs=timbre_ref_codecs,
            timbre_prompt_mels=timbre_prompt_mels,
            style_prompt_output_ids=style_prompt_output_ids,
            style_prompt_mels=style_prompt_mels,
        )

    @torch.no_grad()
    def convert_window(
        self,
        window_24k: np.ndarray,
        *,
        flow_matching_steps: int = 16,
        diffusion_cfg: float = 1.0,
        diffusion_rescale_cfg: float = 0.75,
        seed: Optional[int] = None,
        target_db: float = -25.0,
        clip: bool = True,
        # vevovoice knobs
        ar_max_length: int = 2000,
        ar_temperature: float = 0.8,
        ar_top_k: int = 50,
        ar_top_p: float = 0.9,
        ar_repeat_penalty: float = 1.0,
        ar_min_new_tokens: int = 50,
        prepend_style_ref_to_input: bool = True,
    ) -> np.ndarray:
        if self._ref is None:
            raise RuntimeError("Reference not prepared. Call prepare_reference_bytes().")
        ref = self._ref

        x24k = torch.from_numpy(np.asarray(window_24k, dtype=np.float32)).unsqueeze(0)
        x24k = x24k.to(self.device)
        x16k = torchaudio.functional.resample(x24k, self.model_sr, 16000)

        fm_generator = None
        ar_generator = None
        if seed is not None:
            seed = int(seed)
            fm_generator = torch.Generator(device=self.device).manual_seed(seed)
            ar_generator = torch.Generator(device=self.device).manual_seed(seed + 1)

        if self.kind == "vevotimbre":
            src_hubert_codecs, _ = self.pipeline.extract_hubert_codec(
                self.pipeline.content_style_tokenizer,
                x16k,
                duration_reduction=False,
            )
            diffusion_input_codecs = torch.cat([ref.timbre_ref_codecs, src_hubert_codecs], dim=1)

            predict_mel_feat = self.pipeline.fmt_model.reverse_diffusion(
                cond=self.pipeline.fmt_model.cond_emb(diffusion_input_codecs),
                prompt=ref.timbre_prompt_mels,
                n_timesteps=flow_matching_steps,
                cfg=diffusion_cfg,
                rescale_cfg=diffusion_rescale_cfg,
                generator=fm_generator,
            )
        elif self.kind == "vevovoice":
            if ref.style_prompt_output_ids is None or ref.style_prompt_mels is None:
                raise RuntimeError("AR prompt state missing for vevovoice.")

            if prepend_style_ref_to_input:
                ar_input_wav16k = torch.cat([ref.ref16k, x16k], dim=1)
            else:
                ar_input_wav16k = x16k

            ar_input_ids, _ = self.pipeline.extract_hubert_codec(
                self.pipeline.content_tokenizer,
                ar_input_wav16k,
                token_type=self.pipeline.ar_cfg.model.vc_input_token_type,
                duration_reduction=True,
                duration_reduction_n_gram=getattr(
                    self.pipeline.ar_cfg.model, "vc_input_reduced_n_gram", 1
                ),
            )

            predicted_hubert_codecs = self.pipeline.ar_model.generate(
                input_ids=ar_input_ids,
                prompt_mels=ref.style_prompt_mels,
                prompt_output_ids=ref.style_prompt_output_ids,
                max_length=ar_max_length,
                temperature=ar_temperature,
                top_k=ar_top_k,
                top_p=ar_top_p,
                repeat_penalty=ar_repeat_penalty,
                min_new_tokens=ar_min_new_tokens,
                generator=ar_generator,
            )

            diffusion_input_codecs = torch.cat([ref.timbre_ref_codecs, predicted_hubert_codecs], dim=1)
            predict_mel_feat = self.pipeline.fmt_model.reverse_diffusion(
                cond=self.pipeline.fmt_model.cond_emb(diffusion_input_codecs),
                prompt=ref.timbre_prompt_mels,
                n_timesteps=flow_matching_steps,
                cfg=diffusion_cfg,
                rescale_cfg=diffusion_rescale_cfg,
                generator=fm_generator,
            )
        else:
            raise ValueError(f"Unsupported kind: {self.kind}")

        synthesized_audio = self.pipeline.vocoder_model(predict_mel_feat.transpose(1, 2)).detach()
        audio = synthesized_audio[0, 0].float().cpu().numpy()
        audio = normalize_rms_db(audio, target_db=target_db)
        if clip:
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        return audio
