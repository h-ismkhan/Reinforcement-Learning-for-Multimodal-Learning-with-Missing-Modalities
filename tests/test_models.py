"""Policy network, PPO update and fusion head (rl.py)."""
import torch


def test_scoring_mlp_shapes(rl_module):
    net = rl_module.ScoringMLP(state_dim=12, cand_dim=14)
    assert net(torch.randn(12), torch.randn(7, 14)).shape == (7,)
    assert net(torch.randn(3, 12), torch.randn(3, 7, 14)).shape == (3, 7)


def test_ppo_update_raises_probability_of_rewarded_action(rl_module):
    torch.manual_seed(0)
    net = rl_module.ScoringMLP(8, 10)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    state, cands = torch.randn(8), torch.randn(6, 10)
    with torch.no_grad():
        old_lp = torch.log_softmax(net(state, cands), -1)[2]
    p_before = old_lp.exp().item()
    for _ in range(30):
        loss = rl_module.ppo_update(net, opt, ([state], [cands], [2], [old_lp],
                                               [torch.tensor(1.0)]))
        assert loss == loss                       # not NaN
    p_after = torch.softmax(net(state, cands), -1)[2].item()
    assert p_after > p_before


def test_fusion_head_scalar_output_and_gradients(rl_module):
    head = rl_module.FusionHead(emb_dim=16, hidden=32, n_heads=4)
    out = head(torch.randn(16), torch.randn(9, 16), torch.randn(9))
    assert out.shape == ()
    out.backward()
    assert head.q_proj.weight.grad is not None and head.k_proj.weight.grad is not None


def test_fusion_head_is_permutation_invariant_over_candidates(rl_module):
    torch.manual_seed(1)
    head = rl_module.FusionHead(emb_dim=16, hidden=32, n_heads=4).eval()
    q, ce, cl = torch.randn(16), torch.randn(9, 16), torch.randn(9)
    perm = torch.randperm(9)
    assert torch.allclose(head(q, ce, cl), head(q, ce[perm], cl[perm]), atol=1e-5)
