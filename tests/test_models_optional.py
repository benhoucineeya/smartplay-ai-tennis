import numpy as np, pytest
from core import cf

def _ratings():
    rng = np.random.default_rng(1); R = np.full((30, 8), np.nan)
    for u in range(30):
        liked = range(0, 4) if u < 15 else range(4, 8)
        for i in range(8):
            if rng.random() < .7: R[u, i] = rng.uniform(4, 5) if i in liked else rng.uniform(1, 2)
    return R

def test_cf_hybrid_beats_random_on_synthetic_groups():
    hr = cf.hit_rate_at_k(_ratings(), k=2); assert hr is not None and hr > 2 / 4      # random HR@2 over ~4 unrated items is about 0.5
    R = _ratings(); R[0, :] = np.nan; R[0, 0] = 5
    assert len(cf.recommend(R, 0, k=2)) == 2                                           # cold start -> popular items

def test_negative_sampling_never_hangs():
    from core import neumf                      # pure numpy part: no torch needed
    rng = np.random.default_rng(0); dense = {(u, i) for u in range(5) for i in range(4)}
    assert neumf.sample_negatives(range(5), dense, 4, 4, rng) == ([], [])          # every item already positive: returns, empty
    u, i = neumf.sample_negatives(range(5), {(0, 0)}, 4, 3, rng); assert len(u) == 15 and (0, 0) not in set(zip(u, i))

def test_neumf_guard_and_training():
    torch = pytest.importorskip("torch"); from core import neumf
    with pytest.raises(ValueError): neumf.train_neumf(np.zeros((10, 2), int), 5, 5)
    rng = np.random.default_rng(0); pairs = np.unique(np.c_[rng.integers(0, 30, 300), rng.integers(0, 40, 300)], axis=0)
    m, h = neumf.train_neumf(pairs, 30, 40, epochs=5, min_interactions=50); assert len(h) >= 1

def test_tracknet_shapes_and_loss_decreases():
    torch = pytest.importorskip("torch"); from core import tracknet
    X, Y = tracknet.synth_batch(2); assert tracknet.build_tracknet()(X).shape == Y.shape
    _, L = tracknet.train_synthetic(steps=30); assert np.mean(L[-5:]) < np.mean(L[:5])
