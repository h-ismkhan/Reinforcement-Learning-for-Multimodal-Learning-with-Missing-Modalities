#!/usr/bin/env python
"""
run_mosi.py
===========
Thin CLI around ``rl.main()`` for the real CMU-MOSI experiments, so paths and the
most common experiment knobs can be set from the command line / environment
(e.g. inside Docker) instead of editing the PARAMETERS block in rl.py.

    python run_mosi.py --mosi-root /data --missing-config complex_20_20_20_10_10_10_10 \
                       --variant with --strategy knn --seed 42 --out-dir artifacts/mosi

Only knobs that rl.py reads at call time are exposed. Hyper-parameters that rl.py
binds as *default arguments* at import time (e.g. FUSION_HIDDEN, MAX_BFS_DEPTH)
still have to be changed in rl.py itself.

Expected layout under --mosi-root (same sub-paths rl.py uses by default):
    Audio/WAV_16000/Segmented/   Video/Segmented/   Transcript/Segmented/
    mosi_splits-70train.json
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

import rl


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mosi-root", default=os.environ.get("MOSI_ROOT"),
                   help="folder containing the raw CMU-MOSI data (or set $MOSI_ROOT)")
    p.add_argument("--missing-config", default=rl.MISSING_CONFIG)
    p.add_argument("--variant", choices=["with", "just_scaling", "without"],
                   default="with", help="'with' = modality-fair distance (method default)")
    p.add_argument("--strategy", choices=["knn", "unn"], default=rl.STRATEGY)
    p.add_argument("--seed", type=int, default=rl.SEED)
    p.add_argument("--ppo-epochs", type=int, default=rl.PPO_EPOCHS)
    p.add_argument("--fusion-epochs", type=int, default=rl.FUSION_EPOCHS)
    p.add_argument("--out-dir", default="artifacts/mosi", help="where checkpoints are written")
    args = p.parse_args(argv)
    if not args.mosi_root:
        p.error("--mosi-root (or $MOSI_ROOT) is required")
    return args


def apply_overrides(args) -> None:
    root = Path(args.mosi_root)
    rl.AUDIO_DIR = str(root / "Audio/WAV_16000/Segmented")
    rl.VIDEO_DIR = str(root / "Video/Segmented")
    rl.TEXT_DIR = str(root / "Transcript/Segmented")
    rl.SPLIT_FILE = str(root / "mosi_splits-70train.json")
    rl.MISSING_CONFIG, rl.SEED = args.missing_config, args.seed
    rl.VARIANT, rl.STRATEGY = args.variant, args.strategy
    rl.PPO_EPOCHS, rl.FUSION_EPOCHS = args.ppo_epochs, args.fusion_epochs
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_{args.strategy}_{args.missing_config}_seed{args.seed}"
    rl.RL_CKPT = str(out / f"policy_{tag}.pt")
    rl.FUSION_CKPT = str(out / f"fusion_{tag}.pt")


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    apply_overrides(args)
    rl.main()


if __name__ == "__main__":
    main()
