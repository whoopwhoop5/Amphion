# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from models.vc.vevo.runner import VevoConverter


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Vevo voice conversion (WAV->WAV).")
    parser.add_argument("--kind", type=str, default="vevotimbre", choices=["vevotimbre", "vevovoice"])
    parser.add_argument("--src", type=str, required=True, help="Source audio path (wav/flac/mp3).")
    parser.add_argument("--ref", type=str, required=True, help="Reference voice clip (wav).")
    parser.add_argument("--out", type=str, default=None, help="Output WAV path.")
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--repo_cache_dir", type=str, default="./ckpts/Vevo")

    parser.add_argument("--flow_matching_steps", type=int, default=32)
    parser.add_argument("--diffusion_cfg", type=float, default=1.0)
    parser.add_argument("--diffusion_rescale_cfg", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target_db", type=float, default=-25.0)

    # AR decoding knobs (used only for vevovoice)
    parser.add_argument("--ar_max_length", type=int, default=2000)
    parser.add_argument("--ar_temperature", type=float, default=0.8)
    parser.add_argument("--ar_top_k", type=int, default=50)
    parser.add_argument("--ar_top_p", type=float, default=0.9)
    parser.add_argument("--ar_repeat_penalty", type=float, default=1.0)
    parser.add_argument("--ar_min_new_tokens", type=int, default=50)

    parser.add_argument("--overwrite", action="store_true")
    return parser


def _default_out_path(src: str, kind: str) -> str:
    src_path = Path(src)
    return str(src_path.with_suffix(f".{kind}.wav"))


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    out_path = args.out or _default_out_path(args.src, args.kind)
    out_path_p = Path(out_path)
    if out_path_p.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path_p} (use --overwrite)")
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    converter = VevoConverter.from_pretrained(
        kind=args.kind,  # type: ignore[arg-type]
        device=args.device,
        repo_cache_dir=args.repo_cache_dir,
    )

    converter.convert_file(
        src_wav_path=os.fspath(Path(args.src)),
        reference_wav_path=os.fspath(Path(args.ref)),
        output_path=os.fspath(out_path_p),
        flow_matching_steps=args.flow_matching_steps,
        diffusion_cfg=args.diffusion_cfg,
        diffusion_rescale_cfg=args.diffusion_rescale_cfg,
        seed=args.seed,
        ar_max_length=args.ar_max_length,
        ar_temperature=args.ar_temperature,
        ar_top_k=args.ar_top_k,
        ar_top_p=args.ar_top_p,
        ar_repeat_penalty=args.ar_repeat_penalty,
        ar_min_new_tokens=args.ar_min_new_tokens,
        target_db=args.target_db,
    )

    print(f"Wrote: {out_path_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
