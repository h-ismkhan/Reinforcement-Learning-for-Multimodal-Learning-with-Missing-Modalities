import sys
from pathlib import Path

import numpy as np
import pytest

# make `import rl`, `import serve` work regardless of where pytest is launched
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def rl_module():
    import rl
    return rl


@pytest.fixture
def toy_graph_data():
    """60 points in 5-d, labels = noisy function of the first coordinate."""
    rng = np.random.RandomState(0)
    emb = rng.randn(60, 5).astype(np.float32)
    labels = (np.clip(emb[:, 0], -3, 3) + 0.1 * rng.randn(60)).astype(np.float32)
    return emb, labels
