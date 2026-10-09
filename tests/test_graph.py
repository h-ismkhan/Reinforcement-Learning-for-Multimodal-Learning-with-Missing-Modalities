"""k-NN graph, seed sets and BFS candidate-set construction (rl.py)."""
import numpy as np


def test_knn_graph_has_no_self_loops_and_bounded_degree(rl_module, toy_graph_data):
    emb, _ = toy_graph_data
    adj = rl_module._build_knn_graph(emb, k=4)
    assert len(adj) == len(emb)
    for i, row in enumerate(adj):
        assert len(row) <= 4
        assert all(j != i and d > 0 for j, d in row)


def test_unn_seeds_are_a_pruned_subset_of_knn(rl_module, toy_graph_data):
    emb, _ = toy_graph_data
    d = np.linalg.norm(emb[1:21] - emb[0], axis=1)
    order = np.argsort(d)
    idxs, dists = (np.arange(1, 21)[order], d[order])
    seeds = rl_module.get_seeds_unn(emb[0], emb, idxs, dists)
    assert 1 <= len(seeds) <= len(idxs)
    assert seeds[0][0] == idxs[0]                 # nearest neighbour is never pruned
    assert {s for s, _ in seeds} <= set(idxs.tolist())


def test_candidate_set_oracle_is_closest_label(rl_module, toy_graph_data):
    emb, labels = toy_graph_data
    rl_module.MAX_NODE2_RL = 30
    adj = rl_module._build_knn_graph(emb, k=4)
    seeds = [(1, 0.5), (2, 0.7)]
    target = 1.37
    seq, oracle = rl_module.build_candidate_set(seeds, target, labels, adj,
                                                rng=np.random.RandomState(0))
    nodes = [n for n, _, _ in seq]
    assert len(nodes) == len(set(nodes)), "candidate nodes must be unique"
    assert {1, 2} <= set(nodes), "seeds are always part of the candidate set"
    best = min(abs(labels[n] - target) for n in nodes)
    assert abs(labels[nodes[oracle]] - target) == best
    assert all(depth <= rl_module.MAX_BFS_DEPTH for _, _, depth in seq)


def test_candidate_sets_are_reproducible(rl_module, toy_graph_data):
    emb, labels = toy_graph_data
    adj = rl_module._build_knn_graph(emb, k=4)
    a = rl_module.build_candidate_set([(3, 1.0)], 0.0, labels, adj, rng=np.random.RandomState(1))
    b = rl_module.build_candidate_set([(3, 1.0)], 0.0, labels, adj, rng=np.random.RandomState(1))
    assert a == b
