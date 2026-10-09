#!/usr/bin/env python
"""
demo_synthetic.py
=================
End-to-end, CPU-only demo of the RL4M4 pipeline on *synthetic* 3-modal data.

Why this exists
---------------
The real experiments need CMU-MOSI (raw audio/video/transcripts), three large
pretrained encoders and a GPU. This script lets anyone reproduce the *whole
pipeline* in a couple of minutes with no download:

    synthetic embeddings -> missing-modality simulation -> z-score + modality-fair
    scaling -> k-NN graph -> BFS candidate sets -> PPO neighbour policy
    -> cross-attention fusion head -> test metrics vs. simple baselines

Nothing here re-implements the method: it imports and calls the functions in
``rl.py`` (and the missingness logic in ``mosi_reg.py``) directly, only
replacing "load media + run frozen encoders" with a synthetic embedding
generator. Numbers are therefore NOT comparable to the paper's CMU-MOSI results;
the point is a reproducible, testable pipeline.

Outputs (in --out-dir)
----------------------
    metrics.json            per-config metrics for every method
    bundle/                 everything serve.py needs (policy, fusion head,
                            training index, normalisation stats, config)
    bundle/example_request.json   a ready-to-POST payload for the API

Usage
-----
    python demo_synthetic.py                       # ~ a few minutes on CPU
    python demo_synthetic.py --quick               # ~ 1 minute, used in CI
    python demo_synthetic.py --configs 100_text_100_audio_100_video
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

import rl
from mosi_reg import MOSIDatasetRegression

MODS = ("text", "audio", "video")
DEFAULT_DIMS = {"text": 48, "audio": 64, "video": 48}
DEFAULT_CONFIGS = ["complex_20_20_20_10_10_10_10", "100_text_100_audio_100_video"]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic multimodal "embeddings"
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticWorld:
    """
    Fixed generative process shared by train/val/test.

    A latent sentiment y in [-3, 3] drives all three modalities through
    modality-specific (partly non-linear) maps. Modalities differ in
      * informativeness  (text > audio > video),
      * scale and offset: the *least* informative modality (video) is given the
        *largest* scale, so un-normalised concatenation is dominated by it.
    This is a deliberate design choice that illustrates why per-modality
    normalisation matters; real encoders need not behave this way.
    """

    SIGNAL = {"text": 1.0, "audio": 0.6, "video": 0.3}
    SCALE = {"text": 1.0, "audio": 0.5, "video": 8.0}
    OFFSET = {"text": 0.5, "audio": -3.0, "video": 10.0}

    def __init__(self, dims: dict[str, int], seed: int):
        self.dims = dims
        rng = np.random.RandomState(seed)
        self.w1 = {m: rng.randn(d) for m, d in dims.items()}
        self.w2 = {m: rng.randn(d) for m, d in dims.items()}

    def sample(self, n: int, seed: int):
        rng = np.random.RandomState(seed)
        y = np.clip(rng.normal(0.2, 1.4, size=n), -3.0, 3.0).astype(np.float32)
        raw = []
        for m in MODS:
            sig = self.SIGNAL[m]
            x = (
                np.outer(y, self.w1[m]) * sig
                + np.outer(np.tanh(y), self.w2[m]) * 0.5 * sig
                + rng.randn(n, self.dims[m])
            )
            raw.append((x * self.SCALE[m] + self.OFFSET[m]).astype(np.float32))
        return raw, y


def sample_availability(n: int, config: str, seed: int) -> np.ndarray:
    """
    Reuse the project's own missing-modality logic (mosi_reg.py) without needing
    any media files: build a throw-away split file and read the precomputed
    ``modality_availability`` table from the dataset object.
    """
    with tempfile.TemporaryDirectory() as d:
        split = Path(d) / "split.json"
        split.write_text(
            json.dumps({"train": [{"name": f"s_{i}", "label": 0.0} for i in range(n)]})
        )
        ds = MOSIDatasetRegression("", "", "", str(split), split="train",
                                   missing_config=config, seed=seed)
    return np.array([ds.modality_availability[i] for i in range(n)], dtype=bool)


def build_split_embeddings(raw, avail, stats, variant):
    """Same step as ``rl.get_all_embeddings`` after the encoders have run."""
    n = raw[0].shape[0]
    out = np.zeros((n, sum(a.shape[1] for a in raw)), dtype=np.float32)
    for i in range(n):
        out[i] = rl.build_partial_embedding([raw[m][i] for m in range(len(raw))],
                                            avail[i], stats, variant)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def regression_and_cls_metrics(preds: np.ndarray, y: np.ndarray) -> dict:
    preds = np.asarray(preds, dtype=np.float32)
    pb, yb = rl._to_binary(preds), rl._to_binary(y)
    p7, y7 = rl._to_7class(preds), rl._to_7class(y)
    return {
        "mse": float(np.mean((preds - y) ** 2)),
        "mae": float(np.mean(np.abs(preds - y))),
        "f1_micro_binary": float(f1_score(yb, pb, average="micro")),
        "f1_macro_binary": float(f1_score(yb, pb, average="macro")),
        "f1_micro_7class": float(f1_score(y7, p7, average="micro", zero_division=0)),
        "f1_macro_7class": float(f1_score(y7, p7, average="macro", zero_division=0)),
    }


def knn_regress(tr_emb, tr_y, q_emb, k):
    nn_ = NearestNeighbors(n_neighbors=k).fit(tr_emb)
    _, idx = nn_.kneighbors(q_emb)
    return tr_y[idx].mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact export (no pickle: npz + json + state_dicts)
# ─────────────────────────────────────────────────────────────────────────────
def adj_to_csr(adj):
    indptr, idx, w = [0], [], []
    for row in adj:
        for j, d in row:
            idx.append(j)
            w.append(d)
        indptr.append(len(idx))
    return (np.array(indptr, dtype=np.int64), np.array(idx, dtype=np.int64),
            np.array(w, dtype=np.float32))


def save_bundle(bundle_dir: Path, tr_emb, tr_y, adj, sizes, stats, meta, example):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    indptr, idx, w = adj_to_csr(adj)
    arrays = dict(tr_emb=tr_emb, tr_y=tr_y, set_sizes=sizes,
                  adj_indptr=indptr, adj_idx=idx, adj_w=w)
    for m in range(len(MODS)):
        mu, sigma = stats[m]
        arrays[f"mu_{m}"] = mu.cpu().numpy()
        arrays[f"sigma_{m}"] = sigma.cpu().numpy()
    np.savez_compressed(bundle_dir / "index.npz", **arrays)
    (bundle_dir / "config.json").write_text(json.dumps(meta, indent=2))
    (bundle_dir / "example_request.json").write_text(json.dumps(example, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# One full run for one missing-modality configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(config: str, args, world: SyntheticWorld, ckpt_dir: Path,
               bundle_dir: Path | None) -> dict:
    t0 = time.time()
    set_seed(args.seed)
    dims = [world.dims[m] for m in MODS]
    rl.TEXT_DIM, rl.AUDIO_DIM, rl.VIDEO_DIM = dims
    rl.VARIANT, rl.STRATEGY = "with", "knn"
    rl.MAX_NODE2_RL = args.max_nodes
    rl.PPO_EPOCHS = args.ppo_epochs
    rl.FUSION_EPOCHS = args.fusion_epochs
    rl.RL_CKPT = str(ckpt_dir / "policy.pt")
    rl.FUSION_CKPT = str(ckpt_dir / "fusion.pt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw, y, avail, split_embs = {}, {}, {}, {}
    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for off, (name, n) in enumerate(sizes.items()):
        raw[name], y[name] = world.sample(n, seed=args.seed + 10 * (off + 1))
        avail[name] = sample_availability(n, config, seed=args.seed + off)

    stats = rl.compute_norm_stats(*[torch.tensor(a) for a in raw["train"]])
    for name in sizes:
        split_embs[name] = build_split_embeddings(raw[name], avail[name], stats, "with")
    tr_emb, va_emb, te_emb = (split_embs[s] for s in ("train", "val", "test"))
    tr_y, va_y, te_y = y["train"], y["val"], y["test"]

    # naive (no modality-fair scaling) embedding, only for a baseline row
    naive = {s: np.concatenate(raw[s], axis=1) * np.concatenate(
        [avail[s][:, [m]].repeat(dims[m], axis=1) for m in range(3)], axis=1)
        for s in sizes}

    print(f"\n{'=' * 70}\nConfig: {config}   "
          f"(train/val/test = {args.n_train}/{args.n_val}/{args.n_test})\n{'=' * 70}")
    adj = rl._build_knn_graph(tr_emb, k=rl.K_GRAPH)
    cand_sets, oracles, set_sizes = rl.build_all_sets(tr_emb, tr_y, adj)

    policy = rl.train_policy(tr_emb, tr_y, cand_sets, oracles, va_emb, va_y, adj, set_sizes)
    preds_rl = rl.predict_test(policy, te_emb, te_y, tr_emb, tr_y, adj, set_sizes)
    head = rl.train_fusion(policy, tr_emb, tr_y, cand_sets, va_emb, va_y, adj, set_sizes)
    preds_fus = rl.predict_test_fusion(head, policy, te_emb, te_y, tr_emb, tr_y, adj, set_sizes)

    results = {
        "Mean predictor": regression_and_cls_metrics(np.full_like(te_y, tr_y.mean()), te_y),
        "kNN-5, raw concat (no modality-fair scaling)": regression_and_cls_metrics(
            knn_regress(naive["train"], tr_y, naive["test"], 5), te_y),
        "kNN-1 (modality-fair)": regression_and_cls_metrics(
            knn_regress(tr_emb, tr_y, te_emb, 1), te_y),
        "kNN-5 (modality-fair)": regression_and_cls_metrics(
            knn_regress(tr_emb, tr_y, te_emb, 5), te_y),
        "RL4M4: PPO hard selection": regression_and_cls_metrics(preds_rl, te_y),
        "RL4M4: PPO + fusion head": regression_and_cls_metrics(preds_fus, te_y),
    }

    if bundle_dir is not None:
        i = 0  # first test sample -> example API request, in *raw* embedding space
        present = avail["test"][i]
        example = {m: (raw["test"][k][i].tolist() if present[k] else None)
                   for k, m in enumerate(MODS)}
        if not any(present):  # make sure the example has at least one modality
            example["text"] = raw["test"][0][i].tolist()
        meta = {
            "missing_config": config, "seed": args.seed, "dims": dict(zip(MODS, dims)),
            "variant": "with", "strategy": "knn",
            "k_graph": rl.K_GRAPH, "k_approx": rl.K_APPROX,
            "synthetic": True, "example_true_label": float(te_y[i]),
        }
        save_bundle(bundle_dir, tr_emb, tr_y, adj, set_sizes, stats, meta, example)

    return {"results": results, "seconds": round(time.time() - t0, 1),
            "mean_candidate_set_size": float(set_sizes[set_sizes > 0].mean())}


def print_table(all_results: dict) -> None:
    for config, block in all_results.items():
        print(f"\n┌─ {config}   ({block['seconds']}s, "
              f"mean candidate-set size {block['mean_candidate_set_size']:.1f})")
        print(f"│ {'method':<46s} {'MSE':>7s} {'MAE':>7s} {'F1bin':>7s} {'F1-7c':>7s}")
        for name, m in block["results"].items():
            print(f"│ {name:<46s} {m['mse']:>7.3f} {m['mae']:>7.3f} "
                  f"{m['f1_micro_binary']:>7.3f} {m['f1_micro_7class']:>7.3f}")
        print("└" + "─" * 78)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="missing-modality configs (same syntax as mosi_reg.py); "
                        "the FIRST one is exported as the serving bundle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-val", type=int, default=120)
    p.add_argument("--n-test", type=int, default=150)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--fusion-epochs", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=60,
                   help="cap on BFS candidate-set size (rl.MAX_NODE2_RL)")
    p.add_argument("--quick", action="store_true", help="tiny settings for CI / smoke tests")
    args = p.parse_args(argv)
    if args.quick:
        args.n_train, args.n_val, args.n_test = 150, 40, 40
        args.ppo_epochs, args.fusion_epochs, args.max_nodes = 2, 2, 25
        args.configs = args.configs[:1]
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = SyntheticWorld(DEFAULT_DIMS, seed=args.seed)

    all_results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, cfg in enumerate(args.configs):
            all_results[cfg] = run_config(
                cfg, args, world,
                ckpt_dir=(out / "bundle") if i == 0 else Path(tmp) / f"cfg{i}",
                bundle_dir=(out / "bundle") if i == 0 else None,
            )
    print_table(all_results)
    (out / "metrics.json").write_text(json.dumps(
        {"args": vars(args), "results": all_results}, indent=2))
    print(f"\nWrote {out / 'metrics.json'} and serving bundle {out / 'bundle'}/")
    return all_results


if __name__ == "__main__":
    main()
