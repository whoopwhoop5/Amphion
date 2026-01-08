# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from huggingface_hub import snapshot_download

from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio


VevoKind = Literal["vevotimbre", "vevovoice"]


@dataclass(frozen=True)
class VevoModelPaths:
    content_tokenizer_ckpt_path: Optional[str]
    content_style_tokenizer_ckpt_path: str
    ar_cfg_path: Optional[str]
    ar_ckpt_path: Optional[str]
    fmt_cfg_path: str
    fmt_ckpt_path: str
    vocoder_cfg_path: str
    vocoder_ckpt_path: str


def _download_vevo(repo_cache_dir: str, patterns: list[str]) -> str:
    return snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=repo_cache_dir,
        allow_patterns=patterns,
    )


def resolve_vevo_model_paths(
    kind: VevoKind,
    repo_cache_dir: str = "./ckpts/Vevo",
) -> VevoModelPaths:
    patterns = [
        "tokenizer/vq8192/*",
        "acoustic_modeling/Vq8192ToMels/*",
        "acoustic_modeling/Vocoder/*",
    ]
    if kind == "vevovoice":
        patterns.extend(
            [
                "tokenizer/vq32/*",
                "contentstyle_modeling/Vq32ToVq8192/*",
            ]
        )

    local_dir = _download_vevo(repo_cache_dir=repo_cache_dir, patterns=patterns)

    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"

    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    if kind == "vevovoice":
        content_tokenizer_ckpt_path = os.path.join(
            local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
        )
        ar_cfg_path = "./models/vc/vevo/config/Vq32ToVq8192.json"
        ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")
    else:
        content_tokenizer_ckpt_path = None
        ar_cfg_path = None
        ar_ckpt_path = None

    return VevoModelPaths(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
    )


class VevoConverter:
    def __init__(
        self,
        kind: VevoKind,
        device: torch.device,
        paths: VevoModelPaths,
    ) -> None:
        self.kind = kind
        self.device = device
        self.paths = paths

        self.pipeline = VevoInferencePipeline(
            content_tokenizer_ckpt_path=paths.content_tokenizer_ckpt_path,
            content_style_tokenizer_ckpt_path=paths.content_style_tokenizer_ckpt_path,
            ar_cfg_path=paths.ar_cfg_path,
            ar_ckpt_path=paths.ar_ckpt_path,
            fmt_cfg_path=paths.fmt_cfg_path,
            fmt_ckpt_path=paths.fmt_ckpt_path,
            vocoder_cfg_path=paths.vocoder_cfg_path,
            vocoder_ckpt_path=paths.vocoder_ckpt_path,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        kind: VevoKind,
        device: Optional[str] = None,
        repo_cache_dir: str = "./ckpts/Vevo",
    ) -> "VevoConverter":
        if device is None:
            if torch.cuda.is_available():
                torch_device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch_device = torch.device("mps")
            else:
                torch_device = torch.device("cpu")
        else:
            torch_device = torch.device(device)

        paths = resolve_vevo_model_paths(kind=kind, repo_cache_dir=repo_cache_dir)
        return cls(kind=kind, device=torch_device, paths=paths)

    @torch.no_grad()
    def convert_file(
        self,
        src_wav_path: str,
        reference_wav_path: str,
        output_path: str,
        *,
        flow_matching_steps: int = 32,
        seed: Optional[int] = None,
        ar_max_length: int = 2000,
        ar_temperature: float = 0.8,
        ar_top_k: int = 50,
        ar_top_p: float = 0.9,
        ar_repeat_penalty: float = 1.0,
        ar_min_new_tokens: int = 50,
        diffusion_cfg: float = 1.0,
        diffusion_rescale_cfg: float = 0.75,
        target_db: float = -25.0,
    ) -> str:
        if self.kind == "vevotimbre":
            gen_audio = self.pipeline.inference_fm(
                src_wav_path=src_wav_path,
                timbre_ref_wav_path=reference_wav_path,
                flow_matching_steps=flow_matching_steps,
                diffusion_cfg=diffusion_cfg,
                diffusion_rescale_cfg=diffusion_rescale_cfg,
                seed=seed,
            )
        elif self.kind == "vevovoice":
            gen_audio = self.pipeline.inference_ar_and_fm(
                src_wav_path=src_wav_path,
                src_text=None,
                style_ref_wav_path=reference_wav_path,
                timbre_ref_wav_path=reference_wav_path,
                vc_input_mask_ratio=-1,
                use_global_guided_inference=False,
                flow_matching_steps=flow_matching_steps,
                ar_max_length=ar_max_length,
                ar_temperature=ar_temperature,
                ar_top_k=ar_top_k,
                ar_top_p=ar_top_p,
                ar_repeat_penalty=ar_repeat_penalty,
                ar_min_new_tokens=ar_min_new_tokens,
                diffusion_cfg=diffusion_cfg,
                diffusion_rescale_cfg=diffusion_rescale_cfg,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported kind: {self.kind}")

        return save_audio(gen_audio, output_path=output_path, target_db=target_db)
