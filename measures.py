"""Permutation entropy (H), statistical complexity (C), bidimensional fuzzy
entropy (FuzzyEn) and Lempel-Ziv complexity (LZC) of a 2-D map."""
import numpy as np
from scipy.spatial.distance import pdist

N_PAT = 24                                   # 4! ordinal patterns of a 2x2 window
FACT = np.array([6, 2, 1, 1])


def ordinal_codes(M, valid=None):
    """Pattern index (0-23) of every 2x2 window, raster order.
    Window read row-major (TL, TR, BL, BR); ties keep order of appearance."""
    M = np.asarray(M, np.float64)
    w = np.stack([M[:-1, :-1], M[:-1, 1:], M[1:, :-1], M[1:, 1:]], -1).reshape(-1, 4)
    if valid is not None:
        v = valid
        w = w[(v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]).ravel()]
    perm = np.argsort(w, axis=1, kind="stable")
    codes = np.zeros(len(w), np.int64)
    for i in range(4):                       # Lehmer code of the permutation
        codes += (perm[:, i + 1:] < perm[:, [i]]).sum(1) * FACT[i]
    return codes


def _S(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def hc(counts, n=N_PAT):
    """Normalised permutation entropy H and MPR statistical complexity C."""
    p = counts / counts.sum()
    u = np.full(n, 1 / n)
    H = _S(p) / np.log(n)
    D = _S((p + u) / 2) - _S(p) / 2 - _S(u) / 2                 # Jensen-Shannon
    Dmax = -0.5 * ((n + 1) / n * np.log(n + 1) + np.log(n) - 2 * np.log(2 * n))
    return H, D / Dmax * H


def hc_fixed_n(M, valid, n_target, n_draws=100, seed=20250902):
    """H and C from n_target windows (sampled without replacement), averaged
    over draws, so every painting uses the same number of windows."""
    codes = ordinal_codes(M, valid)
    if len(codes) <= n_target:
        return hc(np.bincount(codes, minlength=N_PAT).astype(float))
    rng = np.random.default_rng(seed)
    r = [hc(np.bincount(codes[rng.choice(len(codes), n_target, replace=False)],
                        minlength=N_PAT).astype(float)) for _ in range(n_draws)]
    return tuple(np.mean(r, axis=0))


def fuzzy_entropy_2d(U, m=2, r=0.2, n=2.0):
    """ln(phi_m) - ln(phi_m+1); z-scored map, mean-removed m x m and
    (m+1) x (m+1) templates at the same positions, Chebyshev distance,
    similarity exp(-d^n / r)."""
    U = (U - U.mean()) / U.std()
    nr, nc = U.shape[0] - m, U.shape[1] - m
    phi = []
    for k in (m, m + 1):
        T = np.stack([U[i:i + nr, j:j + nc] for i in range(k) for j in range(k)], -1)
        T = T.reshape(-1, k * k)
        T = T - T.mean(1, keepdims=True)
        phi.append(np.exp(-pdist(T, "chebyshev") ** n / r).mean())
    return np.log(phi[0]) - np.log(phi[1])


def lz76(s):
    """Lempel-Ziv (1976) complexity, Kaspar-Schuster algorithm."""
    s, n = s.tolist(), len(s)
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                return c + 1
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    return c
                i, k, kmax = 0, 1, 1
            else:
                k = 1


def lzc_2d(U, n_sur=10, seed=20250902):
    """Median-binarised map, row- and column-major readouts, each divided by
    the mean LZ76 of 10 shuffled copies; the two readouts are averaged."""
    b = (U > np.median(U)).astype(np.int8)
    rng = np.random.default_rng(seed)
    out = []
    for seq in (b.ravel("C"), b.ravel("F")):
        sur = np.mean([lz76(rng.permutation(seq)) for _ in range(n_sur)])
        out.append(lz76(seq) / sur)
    return float(np.mean(out))


def valid_rect(M, valid):
    rows, cols = np.where(valid.any(1))[0], np.where(valid.any(0))[0]
    return np.asarray(M[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], np.float64)
