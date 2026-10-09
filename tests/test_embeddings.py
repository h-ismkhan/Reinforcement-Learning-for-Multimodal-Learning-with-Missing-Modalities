"""Normalisation / modality-fair scaling and zero-imputation (rl.py)."""
import numpy as np
import pytest
import torch


@pytest.fixture
def small_dims(rl_module):
    rl_module.TEXT_DIM, rl_module.AUDIO_DIM, rl_module.VIDEO_DIM = 40, 80, 20
    return [40, 80, 20]


@pytest.fixture
def raw_mods(small_dims):
    """Three modalities with wildly different scales/offsets."""
    rng = np.random.RandomState(0)
    scales, offs = [1.0, 50.0, 0.01], [0.0, 7.0, -3.0]
    return [(rng.randn(2000, d) * s + o).astype(np.float32)
            for d, s, o in zip(small_dims, scales, offs)]


def test_norm_stats_use_unit_floor_for_constant_dims(rl_module):
    z = torch.zeros(10, 4)
    z[:, 0] = torch.arange(10.0)
    stats = rl_module.compute_norm_stats(z, z, z)
    assert stats[0][1][1].item() == 1.0       # constant column -> sigma clamped to 1
    assert stats[0][1][0].item() > 1.0


def test_modality_fair_scaling_equalises_blocks(rl_module, raw_mods, small_dims):
    """After 'with' scaling every block has the same expected squared norm 1/M."""
    stats = rl_module.compute_norm_stats(*[torch.tensor(a) for a in raw_mods])
    full = np.ones(3, dtype=bool)
    emb = np.stack([rl_module.build_partial_embedding([a[i] for a in raw_mods], full, stats, "with")
                    for i in range(500)])
    edges = np.cumsum([0] + small_dims)
    block_sq = [(emb[:, edges[m]:edges[m + 1]] ** 2).sum(1).mean() for m in range(3)]
    assert block_sq == pytest.approx([1 / 3] * 3, rel=0.1)

    # ... whereas raw concatenation is dominated by the large-scale modality
    emb_raw = np.stack([rl_module.build_partial_embedding([a[i] for a in raw_mods], full, stats, "without")
                        for i in range(500)])
    raw_sq = [(emb_raw[:, edges[m]:edges[m + 1]] ** 2).sum(1).mean() for m in range(3)]
    assert raw_sq[1] > 100 * raw_sq[0]


def test_missing_modality_blocks_are_zero_and_rescaled(rl_module, raw_mods, small_dims):
    stats = rl_module.compute_norm_stats(*[torch.tensor(a) for a in raw_mods])
    edges = np.cumsum([0] + small_dims)
    present = np.array([True, False, True])
    vecs = np.stack([rl_module.build_partial_embedding([a[i] for a in raw_mods], present, stats, "with")
                     for i in range(500)])
    assert (vecs[:, edges[1]:edges[2]] == 0).all()
    # with M'=2 present modalities, each present block has expected squared norm 1/2
    for m in (0, 2):
        assert (vecs[:, edges[m]:edges[m + 1]] ** 2).sum(1).mean() == pytest.approx(0.5, rel=0.1)


def test_all_missing_does_not_crash(rl_module, raw_mods, stats=None):
    stats = rl_module.compute_norm_stats(*[torch.tensor(a) for a in raw_mods])
    v = rl_module.build_partial_embedding([a[0] for a in raw_mods], np.zeros(3, bool), stats, "with")
    assert (v == 0).all()


def test_label_conversions(rl_module):
    s = np.array([-3.0, -2.5, -0.01, 0.0, 0.4, 2.99, 3.0])
    assert rl_module._to_binary(s).tolist() == [0, 0, 0, 1, 1, 1, 1]
    c7 = rl_module._to_7class(s)
    assert c7.min() == 0 and c7.max() == 6
    assert c7.tolist() == sorted(c7.tolist())     # monotone in the score
