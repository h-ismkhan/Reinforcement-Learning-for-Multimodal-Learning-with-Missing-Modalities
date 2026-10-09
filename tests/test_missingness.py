"""Missing-modality simulation in mosi_reg.py (no media files needed)."""
import json

import numpy as np
import pytest

from mosi_reg import MOSIDatasetRegression


def make(tmp_path, config, n=4000, seed=0):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [{"name": f"s_{i}", "label": 0.0} for i in range(n)]}))
    return MOSIDatasetRegression("", "", "", str(split), split="train",
                                 missing_config=config, seed=seed)


def avail(ds):
    return np.array([ds.modality_availability[i] for i in range(len(ds))])


def test_simple_config_rates(tmp_path):
    a = avail(make(tmp_path, "20_text_100_audio_50_video"))
    assert a[:, 0].mean() == pytest.approx(0.2, abs=0.03)
    assert a[:, 1].all()
    assert a[:, 2].mean() == pytest.approx(0.5, abs=0.03)


def test_complex_config_joint_distribution(tmp_path):
    a = avail(make(tmp_path, "complex_20_20_20_10_10_10_10", n=8000))
    patterns = {(1, 1, 1): .2, (1, 0, 0): .2, (0, 1, 0): .2, (0, 0, 1): .1,
                (1, 1, 0): .1, (1, 0, 1): .1, (0, 1, 1): .1}
    for pat, frac in patterns.items():
        assert (a == np.array(pat, dtype=bool)).all(axis=1).mean() == pytest.approx(frac, abs=0.02)
    assert a.any(axis=1).all(), "complex config never produces a sample with no modality"


def test_complex_config_must_sum_to_100(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, "complex_20_20_20_10_10_10_5")


def test_availability_is_deterministic(tmp_path):
    a1 = avail(make(tmp_path, "50_text_50_audio_50_video", seed=3))
    a2 = avail(make(tmp_path, "50_text_50_audio_50_video", seed=3))
    a3 = avail(make(tmp_path, "50_text_50_audio_50_video", seed=4))
    assert (a1 == a2).all()
    assert not (a1 == a3).all()
