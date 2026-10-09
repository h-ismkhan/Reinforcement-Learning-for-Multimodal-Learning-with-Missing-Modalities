"""
serve.py
========
Minimal inference API around a trained RL4M4 bundle (see demo_synthetic.py).

Design notes
------------
* The service receives *pre-computed* per-modality embeddings (the frozen
  BERT / WavLM / CLIP encoders are a separate concern). Any modality may be
  omitted / null: it is zero-imputed exactly as in training
  (``rl.build_partial_embedding``), so no special-casing is needed.
* The artifact bundle contains only ``.npz`` / ``.json`` / ``state_dict`` files.
  State dicts are loaded with ``weights_only=True``; nothing is unpickled.
* One bundle per process (``BUNDLE_DIR`` env var, default ``artifacts/bundle``).

Run
---
    uvicorn serve:app --host 0.0.0.0 --port 8000
    curl -s localhost:8000/predict -H 'content-type: application/json' \
         -d @artifacts/bundle/example_request.json
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

import rl

MODS = ("text", "audio", "video")


# ─────────────────────────────────────────────────────────────────────────────
# Model bundle
# ─────────────────────────────────────────────────────────────────────────────
class Bundle:
    def __init__(self, bundle_dir: str | Path):
        d = Path(bundle_dir)
        for f in ("config.json", "index.npz", "policy.pt", "fusion.pt"):
            if not (d / f).exists():
                raise FileNotFoundError(
                    f"{d / f} not found. Create a bundle first: python demo_synthetic.py")
        self.cfg = json.loads((d / "config.json").read_text())
        self.dims = [self.cfg["dims"][m] for m in MODS]

        # rl.py keeps its configuration in module globals; align them with the bundle.
        rl.TEXT_DIM, rl.AUDIO_DIM, rl.VIDEO_DIM = self.dims
        rl.VARIANT, rl.STRATEGY = self.cfg["variant"], self.cfg["strategy"]
        rl.DEVICE = torch.device("cpu")

        z = np.load(d / "index.npz")
        self.tr_emb, self.tr_y, self.set_sizes = z["tr_emb"], z["tr_y"], z["set_sizes"]
        self.stats = {m: (z[f"mu_{m}"], z[f"sigma_{m}"]) for m in range(len(MODS))}
        indptr, idx, w = z["adj_indptr"], z["adj_idx"], z["adj_w"]
        self.adj = [[(int(idx[j]), float(w[j])) for j in range(indptr[i], indptr[i + 1])]
                    for i in range(len(indptr) - 1)]
        self.index = NearestNeighbors(n_neighbors=min(rl.K_APPROX + 1, len(self.tr_emb)),
                                      metric="euclidean").fit(self.tr_emb)

        D = self.tr_emb.shape[1]
        self.policy = rl.ScoringMLP(D + 2, D + 4)
        self.policy.load_state_dict(torch.load(d / "policy.pt", map_location="cpu",
                                               weights_only=True))
        self.head = rl.FusionHead(emb_dim=D)
        self.head.load_state_dict(torch.load(d / "fusion.pt", map_location="cpu",
                                             weights_only=True))
        self.policy.eval()
        self.head.eval()

    def predict(self, **modalities) -> dict:
        t0 = time.perf_counter()
        present = np.array([modalities.get(m) is not None for m in MODS])
        if not present.any():
            raise ValueError("at least one modality must be provided")
        raw = []
        for m, dim in zip(MODS, self.dims):
            v = modalities.get(m)
            if v is None:
                raw.append(np.zeros(dim, dtype=np.float32))
                continue
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape != (dim,):
                raise ValueError(f"'{m}' must have length {dim}, got shape {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"'{m}' contains NaN/inf")
            raw.append(arr)

        q = rl.build_partial_embedding(raw, present, self.stats, rl.VARIANT)
        dists, idxs = self.index.kneighbors(q[None, :])
        args = (q, idxs[0], dists[0], self.tr_emb, self.tr_y, self.adj, self.set_sizes)
        with torch.no_grad():
            p_rl = rl._infer_one(*args, self.policy)
            p_fus = rl._infer_one_fusion(*args, self.policy, self.head)
        if not (math.isfinite(p_rl) and math.isfinite(p_fus)):
            raise RuntimeError("non-finite prediction")
        return {
            "prediction": float(p_fus),
            "prediction_rl_only": float(p_rl),
            "sentiment": "positive" if p_fus >= rl.BINARISE_THRESHOLD else "negative",
            "modalities_used": [m for m, p in zip(MODS, present) if p],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: list[float] | None = Field(None, description="frozen text embedding or null")
    audio: list[float] | None = Field(None, description="frozen audio embedding or null")
    video: list[float] | None = Field(None, description="frozen video embedding or null")


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="RL selection + fusion head, sentiment in [-3, 3]")
    prediction_rl_only: float = Field(..., description="label of the single policy-selected neighbour")
    sentiment: str
    modalities_used: list[str]
    latency_ms: float


def create_app(bundle_dir: str | None = None) -> FastAPI:
    bundle_dir = bundle_dir or os.environ.get("BUNDLE_DIR", "artifacts/bundle")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = Bundle(bundle_dir)
        yield

    app = FastAPI(title="RL4M4 missing-modality sentiment API", version="1.0.0",
                  lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "train_points": int(len(request.app.state.bundle.tr_emb))}

    @app.get("/info")
    def info(request: Request):
        return request.app.state.bundle.cfg

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        try:
            return request.app.state.bundle.predict(
                text=req.text, audio=req.audio, video=req.video)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


app = create_app()
