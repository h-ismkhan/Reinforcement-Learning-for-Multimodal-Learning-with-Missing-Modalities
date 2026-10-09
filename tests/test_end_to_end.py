"""Smoke test of the whole synthetic pipeline + the inference API."""
import json
import math

import pytest

import demo_synthetic


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    results = demo_synthetic.main(["--quick", "--out-dir", str(out)])
    return out, results


def test_demo_writes_metrics_and_bundle(demo_run):
    out, results = demo_run
    metrics = json.loads((out / "metrics.json").read_text())
    block = next(iter(metrics["results"].values()))["results"]
    assert {"Mean predictor", "RL4M4: PPO hard selection", "RL4M4: PPO + fusion head"} <= set(block)
    for m in block.values():
        assert all(math.isfinite(v) for v in m.values())
        assert 0.0 <= m["f1_micro_binary"] <= 1.0
    for f in ("config.json", "index.npz", "policy.pt", "fusion.pt", "example_request.json"):
        assert (out / "bundle" / f).exists(), f


def test_learned_models_beat_the_mean_predictor(demo_run):
    _, results = demo_run
    block = next(iter(results.values()))["results"]
    assert block["kNN-5 (modality-fair)"]["mse"] < block["Mean predictor"]["mse"]


@pytest.fixture(scope="module")
def client(demo_run):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import serve
    out, _ = demo_run
    with TestClient(serve.create_app(str(out / "bundle"))) as c:
        yield c, json.loads((out / "bundle" / "example_request.json").read_text())


def test_api_health_and_predict(client):
    c, example = client
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=example)
    assert r.status_code == 200, r.text
    body = r.json()
    assert -10 < body["prediction"] < 10 and body["sentiment"] in ("positive", "negative")
    assert body["modalities_used"]


def test_api_handles_each_single_modality(client):
    c, example = client
    full = c.get("/info").json()["dims"]
    for keep in ("text", "audio", "video"):
        payload = {k: [0.1] * full[k] if k == keep else None for k in full}
        assert c.post("/predict", json=payload).status_code == 200


def test_api_rejects_bad_input(client):
    c, example = client
    assert c.post("/predict", json={}).status_code == 422                       # no modality
    assert c.post("/predict", json={"text": [1.0, 2.0]}).status_code == 422     # wrong length
