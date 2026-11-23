"""TrackNet-style heatmap ball detector (small). Mechanism + synthetic smoke training only: NOT trained on padel, NOT executed in the build environment (no torch)."""
import numpy as np

def build_tracknet(in_frames=3):
    import torch, torch.nn as nn
    blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(o))
    class TrackNetLite(nn.Module):       # H and W must be divisible by 4
        def __init__(s):
            super().__init__(); c = 3 * in_frames
            s.e1, s.e2, s.e3, s.d2, s.d1 = blk(c, 16), blk(16, 32), blk(32, 64), blk(96, 32), blk(48, 16)
            s.pool, s.up, s.out = nn.MaxPool2d(2), nn.Upsample(scale_factor=2), nn.Conv2d(16, 1, 1)
        def forward(s, x):               # x: (B, 3*in_frames, H, W) -> heatmap logits (B,1,H,W)
            a = s.e1(x); b = s.e2(s.pool(a)); c = s.e3(s.pool(b))
            d = s.d2(torch.cat([s.up(c), b], 1)); d = s.d1(torch.cat([s.up(d), a], 1)); return s.out(d)
    return TrackNetLite()

def decode(logits, thr=0.5):
    """Heatmap logits (1,1,H,W) -> (x, y, confidence) in input pixels, or None if below threshold."""
    import torch
    h = torch.sigmoid(logits)[0, 0]; conf = float(h.max()); y, x = divmod(int(h.argmax()), h.shape[1])
    return (x, y, conf) if conf >= thr else None

def synth_batch(n, size=(64, 96), frames=3, seed=0):
    """Moving bright blob on noise; target = Gaussian at the blob position in the LAST frame. Synthetic only."""
    import torch
    rng = np.random.default_rng(seed); H, W = size; X = np.zeros((n, 3 * frames, H, W), np.float32); Y = np.zeros((n, 1, H, W), np.float32)
    yy, xx = np.mgrid[:H, :W]
    for b in range(n):
        p = rng.uniform([10, 10], [W - 10, H - 10]); v = rng.uniform(-4, 4, 2)
        for f in range(frames):
            q = p + v * f; X[b, 3 * f:3 * f + 3] = rng.normal(0.2, 0.05, (3, H, W)) + 0.8 * np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / 6)
        Y[b, 0] = np.exp(-((xx - q[0]) ** 2 + (yy - q[1]) ** 2) / 8)
    return torch.tensor(X), torch.tensor(Y)

def train_synthetic(steps=60, bs=8, lr=1e-3, seed=0):
    import torch
    torch.manual_seed(seed); m = build_tracknet(); opt = torch.optim.Adam(m.parameters(), lr=lr); lossf = torch.nn.BCEWithLogitsLoss(); L = []
    for s in range(steps):
        X, Y = synth_batch(bs, seed=seed + s); opt.zero_grad(); l = lossf(m(X), Y); l.backward(); opt.step(); L.append(float(l))
    return m, L
