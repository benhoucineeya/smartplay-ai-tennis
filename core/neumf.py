"""NeuMF (GMF + MLP) in PyTorch. Guarded: refuses to train on too little data. NOT executed in the build environment (no torch)."""
import numpy as np

def build_neumf(n_users, n_items, d=8, hidden=(16, 8)):
    import torch, torch.nn as nn
    class NeuMF(nn.Module):
        def __init__(s):
            super().__init__(); s.pg, s.qg, s.pm, s.qm = (nn.Embedding(n, d) for n in (n_users, n_items, n_users, n_items))
            L, inp = [], 2 * d
            for h in hidden: L += [nn.Linear(inp, h), nn.ReLU()]; inp = h
            s.mlp, s.out = nn.Sequential(*L), nn.Linear(d + inp, 1)
        def forward(s, u, i):
            z = torch.cat([s.pg(u) * s.qg(i), s.mlp(torch.cat([s.pm(u), s.qm(i)], -1))], -1)
            return torch.sigmoid(s.out(z)).squeeze(-1)
    return NeuMF()

def sample_negatives(users, pos, n_items, neg, rng, tries=50):
    """Random (user, item) pairs not in `pos`. Bounded retries: a user who interacted with every item gets no negatives (no infinite loop)."""
    u, i = [], []
    for a in users:
        for _ in range(neg):
            for _t in range(tries):
                j = int(rng.integers(n_items))
                if (int(a), j) not in pos: u.append(int(a)); i.append(j); break
    return u, i

def train_neumf(pairs, n_users, n_items, epochs=50, neg=4, patience=5, seed=0, min_interactions=200, lr=0.01):
    """pairs: array (n,2) of (user,item) positive interactions. 80/20 split, BCE, Adam, early stopping."""
    import torch
    if len(pairs) < min_interactions: raise ValueError(f"Only {len(pairs)} interactions (< {min_interactions}): use content/CF baselines instead.")
    torch.manual_seed(seed); rng = np.random.default_rng(seed); pairs = np.asarray(pairs)
    idx = rng.permutation(len(pairs)); cut = int(.8 * len(pairs)); tr, va = pairs[idx[:cut]], pairs[idx[cut:]]
    pos = set(map(tuple, pairs.tolist()))
    def batch(P):
        nu, ni = sample_negatives(P[:, 0], pos, n_items, neg, rng)
        u = torch.tensor(list(map(int, P[:, 0])) + nu); i = torch.tensor(list(map(int, P[:, 1])) + ni)
        return u, i, torch.tensor([1.0] * len(P) + [0.0] * len(nu))
    m, best, bad, hist = build_neumf(n_users, n_items), 1e9, 0, []
    opt, bce = torch.optim.Adam(m.parameters(), lr=lr), torch.nn.BCELoss()
    vu, vi, vy = batch(va)
    for _ in range(epochs):
        u, i, y = batch(tr); opt.zero_grad(); l = bce(m(u, i), y); l.backward(); opt.step()
        with torch.no_grad(): vl = float(bce(m(vu, vi), vy))
        hist.append((float(l), vl))
        if vl < best - 1e-4: best, bad = vl, 0
        else:
            bad += 1
            if bad >= patience: break          # early stopping
    return m, hist
