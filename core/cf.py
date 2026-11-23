"""Collaborative filtering on a users x items rating matrix (NaN = unrated): user-based + SVD hybrid (thesis eq. 20-21)."""
import numpy as np

def _cos(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(a[m] @ b[m] / (np.linalg.norm(a[m]) * np.linalg.norm(b[m]) + 1e-9)) if m.any() else 0.0

def user_based(R, u, i, k=5):
    nb = sorted(((_cos(R[u], R[v]), v) for v in range(len(R)) if v != u and not np.isnan(R[v, i])), reverse=True)[:k]
    den = sum(abs(s) for s, _ in nb)
    return sum(s * R[v, i] for s, v in nb) / den if den > 1e-9 else np.nan

def svd_matrix(R, rank=2):
    mu = np.nanmean(R, axis=0); mu = np.where(np.isnan(mu), np.nanmean(R), mu)
    U, s, Vt = np.linalg.svd(np.where(np.isnan(R), mu, R) - mu, full_matrices=False)
    return (U[:, :rank] * s[:rank]) @ Vt[:rank] + mu

def hybrid(R, u, i, alpha=0.5, rank=2, k=5, _svd=None):
    ub = user_based(R, u, i, k); sv = (_svd if _svd is not None else svd_matrix(R, rank))[u, i]
    return float(sv) if np.isnan(ub) else float(alpha * ub + (1 - alpha) * sv)

def popular(R, k=3):
    """Cold start: most-rated, then best-rated items."""
    return list(np.lexsort((-np.nan_to_num(np.nanmean(R, axis=0)), -(~np.isnan(R)).sum(0)))[:k])

def recommend(R, u, k=3, min_ratings=3, **kw):
    if (~np.isnan(R[u])).sum() < min_ratings: return popular(R, k)
    sv = svd_matrix(R, kw.get("rank", 2)); todo = [i for i in range(R.shape[1]) if np.isnan(R[u, i])]
    return sorted(todo, key=lambda i: -hybrid(R, u, i, _svd=sv, **kw))[:k]

def hit_rate_at_k(R, k=5, seed=0, **kw):
    """Leave-one-out HR@k: hide one rating per user, rank the user's unrated items, hit if the hidden one is in the top k."""
    rng = np.random.default_rng(seed); hits = n = 0
    for u in range(len(R)):
        rated = np.flatnonzero(~np.isnan(R[u]))
        if len(rated) < 3: continue
        h = int(rng.choice(rated)); T = R.copy(); T[u, h] = np.nan; sv = svd_matrix(T, kw.get("rank", 2))
        todo = [i for i in range(R.shape[1]) if np.isnan(T[u, i])]
        top = sorted(todo, key=lambda i: -hybrid(T, u, i, _svd=sv, **kw))[:k]; hits += h in top; n += 1
    return hits / n if n else None
