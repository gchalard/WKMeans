"""Quick old-vs-new benchmark for WKMeans.fit() on synthetic 1-D distributions.

Run with:  uv run python scripts/bench_fit.py
"""
from __future__ import annotations

import time

import numpy as np
import ot

from wkmeans import WKMeans


# ----- old implementation, copied verbatim from git for comparison ---------
class WKMeansOld:
    def __init__(self, k: int, p: float = 1.0, max_iter: int = 100, tolerance: float = 1e-4, seed: int = 42):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.seed = seed
        self.centroids: list[np.ndarray] = []

    def wasserstein_distance(self, mu1, mu2):
        mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
        mu2 = np.atleast_1d(np.asarray(mu2, dtype=float))
        n1, n2 = len(mu1), len(mu2)
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        a = a / a.sum()
        b = b / b.sum()
        M = ot.dist(mu1.reshape(-1, 1), mu2.reshape(-1, 1), metric="minkowski", p=self.p)
        return ot.emd2(a, b, M)

    def wasserstein_barycenter(self, cluster_samples):
        return np.median(np.sort(cluster_samples, axis=0), axis=0)

    def fit(self, samples):
        np.random.seed(self.seed)
        self.centroids = [samples[i] for i in np.random.choice(len(samples), self.k, replace=False)]
        for _ in range(self.max_iter):
            clusters = {i: [] for i in range(self.k)}
            for sample in samples:
                d = [self.wasserstein_distance(sample, c) for c in self.centroids]
                clusters[int(np.argmin(d))].append(sample)
            new_c = []
            for i in range(self.k):
                if clusters[i]:
                    new_c.append(self.wasserstein_barycenter(np.array(clusters[i])))
                else:
                    new_c.append(self.centroids[i])
            loss = sum(self.wasserstein_distance(self.centroids[i], new_c[i]) for i in range(self.k))
            if loss < self.tolerance:
                break
            self.centroids = new_c


def make_data(n_samples=1000, n_atoms=57, n_components=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(n_components, n_atoms))
    labels = rng.integers(0, n_components, size=n_samples)
    return centers[labels] + rng.normal(0, 1.0, size=(n_samples, n_atoms))


def time_it(fn, *a, **kw):
    t0 = time.perf_counter()
    fn(*a, **kw)
    return time.perf_counter() - t0


def main():
    print(f"{'config':<28} {'old (s)':>10} {'new (s)':>10} {'speedup':>10}")
    print("-" * 62)
    for n_samples in (200, 1000, 5000):
        X = make_data(n_samples=n_samples, n_atoms=57)
        # Old needs a list of 1-D arrays, new accepts the (N, n) array directly.
        X_list = [row for row in X]

        old = WKMeansOld(k=8, p=1, max_iter=20)
        new = WKMeans(k=8, p=1, max_iter=20)

        t_new = time_it(new.fit, X)        # warm BLAS first to be fair
        t_new = time_it(new.fit, X)
        t_old = time_it(old.fit, X_list)

        print(f"N={n_samples:<5} k=8 n=57       {t_old:>10.3f} {t_new:>10.3f} {t_old / t_new:>9.1f}x")


if __name__ == "__main__":
    main()
