from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import pickle
import numpy as np
import ot
import cloudpickle
from pathlib import Path
import yaml


# A scaler is either a callable (e.g. lambda / function) or any object exposing
# a sklearn-style ``transform(X)`` method (e.g. ``StandardScaler``).
Scaler = Union[Callable[[np.ndarray], np.ndarray], Any]


def _apply_scaler(scaler: Scaler, samples: np.ndarray) -> np.ndarray:
    """Apply a scaler regardless of whether it is callable or sklearn-style."""
    if hasattr(scaler, "transform"):
        return scaler.transform(samples)
    return scaler(samples)


@dataclass
class WKMeans:
    k: int # number of clusters
    p: float = 1 # order of the Wasserstein distance
    tolerance: float = 1e-4 # convergence threshold
    max_iter: int = 100 # maximum number of iterations
    seed: int = 42 # random seed
    centroids: list[np.ndarray] = field(default_factory=list) # list of centroids
    features: list[str] = field(default_factory=list) # list of features
    scaler: Optional[Scaler] = None


    def wasserstein_distance(self, mu1, mu2):
        """p-Wasserstein distance between two 1-D empirical distributions (uniform weights).

        Uses the closed-form order-statistics expression:
            W_p(mu, nu)^p = (1/n) * sum_i |sorted(mu)_i - sorted(nu)_i|^p
        for samples of equal size, and POT's quantile-based 1-D solver otherwise.
        Equivalent to the previous LP-based implementation but ~100-1000x faster.
        """
        mu1 = np.sort(np.atleast_1d(np.asarray(mu1, dtype=np.float64)))
        mu2 = np.sort(np.atleast_1d(np.asarray(mu2, dtype=np.float64)))
        if mu1.shape == mu2.shape:
            cost_p = np.mean(np.abs(mu1 - mu2) ** self.p)
        else:
            # Unequal sample sizes: fall back to POT's vectorised quantile solver.
            cost_p = ot.wasserstein_1d(mu1, mu2, p=self.p) ** self.p
        return float(cost_p ** (1.0 / self.p))

    def wasserstein_barycenter(self, cluster_samples):
        """p-Wasserstein barycenter of 1-D empirical distributions with uniform weights.

        Each row of ``cluster_samples`` is one empirical distribution; all rows must
        have the same number of atoms. The barycenter is the pointwise median (p=1)
        or pointwise mean (p>1) of the per-sample sorted quantile vectors.
        """
        cluster_samples = np.atleast_2d(np.asarray(cluster_samples, dtype=np.float64))
        sorted_samples = np.sort(cluster_samples, axis=1)
        if self.p == 1:
            return np.median(sorted_samples, axis=0)
        return np.mean(sorted_samples, axis=0)

    @staticmethod
    def _pairwise_cost_p(X_sorted: np.ndarray, C_sorted: np.ndarray, p: float) -> np.ndarray:
        """Pairwise W_p^p between every row of ``X_sorted`` and every row of ``C_sorted``.

        Both arrays must already be sorted along axis=1 and share the same number of
        atoms. Returns an array of shape ``(N, k)``. Argmin over k is identical to
        the argmin of the true W_p distance (monotone transform), so this is what
        the assignment step needs.
        """
        diff = X_sorted[:, None, :] - C_sorted[None, :, :]
        if p == 2:
            return np.einsum("nkd,nkd->nk", diff, diff)
        if p == 1:
            return np.sum(np.abs(diff), axis=-1)
        return np.sum(np.abs(diff) ** p, axis=-1)

    def fit(self, samples):
        """Fit the WK-means clustering algorithm.

        Parameters
        ----------
        samples : array-like of shape (N, n) or list of N arrays of length n
            Each row is one empirical 1-D distribution with ``n`` atoms (uniform weights).

        Notes
        -----
        Vectorised, closed-form 1-D Wasserstein implementation. Memory of the
        assignment step is ``O(N * k * n)`` floats; if that is too large for your
        machine, batch ``samples`` into chunks and feed them sequentially.
        """
        X = np.asarray(samples, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                "WKMeans.fit expects samples shaped (N, n) where each row is a "
                f"1-D empirical distribution with the same number of atoms; got "
                f"shape {X.shape}."
            )
        if self.k > X.shape[0]:
            raise ValueError(
                f"k={self.k} is larger than the number of samples ({X.shape[0]})."
            )

        rng = np.random.default_rng(self.seed)
        # Sort each sample once: this is the only per-row sort we ever do.
        X_sorted = np.sort(X, axis=1)
        init_idx = rng.choice(X.shape[0], self.k, replace=False)
        # Centroids are stored in already-sorted form throughout the loop.
        C = X_sorted[init_idx].copy()
        n_atoms = X_sorted.shape[1]

        for _ in range(self.max_iter):
            cost_p = self._pairwise_cost_p(X_sorted, C, self.p)  # (N, k)
            labels = np.argmin(cost_p, axis=1)

            new_C = C.copy()
            for i in range(self.k):
                mask = labels == i
                if not mask.any():
                    continue
                cluster_sorted = X_sorted[mask]
                if self.p == 1:
                    new_C[i] = np.median(cluster_sorted, axis=0)
                else:
                    new_C[i] = np.mean(cluster_sorted, axis=0)

            # Convergence: sum of W_p(c_i_old, c_i_new) across centroids, like before.
            move_p = np.sum(np.abs(new_C - C) ** self.p, axis=1) / n_atoms
            loss = float(np.sum(move_p ** (1.0 / self.p)))
            C = new_C
            if loss < self.tolerance:
                break

        self.centroids = [C[i].copy() for i in range(self.k)]

    def predict(self, samples):
        """Assign each sample to its closest centroid (in W_p).

        Parameters
        ----------
        samples : array-like of shape (N, n) or list of N arrays of length n.

        Returns
        -------
        list[int] of length N.
        """
        X = np.asarray(samples, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if self.scaler is not None:
            X = np.asarray(_apply_scaler(self.scaler, X), dtype=np.float64)

        X_sorted = np.sort(X, axis=1)
        # Centroids written by the new fit() are already sorted, but old saved
        # models or hand-built centroids may not be — sort defensively.
        C_sorted = np.sort(np.stack(self.centroids, axis=0), axis=1)
        cost_p = self._pairwise_cost_p(X_sorted, C_sorted, self.p)
        return np.argmin(cost_p, axis=1).tolist()

    def _export_centroids(self, path_prefix: Path) -> tuple[Path, str]:
        M = np.stack(self.centroids, axis=0).astype(np.float64, copy=False)  # shape (k, d)
        M = np.ascontiguousarray(M)  # ensure C-order for stable hashing

        blob = M.tobytes()

        hash_ = sha256(blob).hexdigest()
        path = path_prefix / f"centroids:{hash_}.bytes"

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_bytes(blob)

        return path, hash_


    def _export_scaler(self, hash_: str, path_prefix: Path) -> Optional[Path]:
        """Serialize the scaler with cloudpickle so that lambdas, closures and
        sklearn-style scalers are all supported. Returns the file path, or
        ``None`` if no scaler is configured."""
        if self.scaler is None:
            return None

        path = path_prefix / f"scaler:{hash_}.pkl"

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            cloudpickle.dump(self.scaler, f)

        return path

    def _export_metadata(self, centroids_path: Path, hash_: str, path_prefix: Path, scaler_path: Optional[Path] = None) -> Path:
        createdAt = datetime.now().isoformat()

        metadata = {
            "createdAt": createdAt,
            **{
                attr: getattr(self, attr)
                for attr in dir(self)
                if (
                    not attr.startswith("_") and 
                    not callable(getattr(self, attr)) and 
                    not attr == "centroids" and 
                    not attr == "features" and
                    not attr == "scaler"
                )
            }
        }

        features_prefix = list(set([
            feature.replace(feature.split("_")[-1], "") for feature in self.features
        ]))

        features_prefix = list(set([
            "_".join(feature.split("_")[:-1]) for feature in self.features
        ]))

        metadata["features"] = features_prefix
        metadata["centroids"] = centroids_path.name
        metadata["scaler"] = scaler_path.name if scaler_path is not None else None

        metadata_path = path_prefix / f"metadata:{hash_}.yaml"

        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

        return metadata_path

    def export(self, path_prefix: Path) -> tuple[Path, Path, Optional[Path]]:
        centroids_path, hash_ = self._export_centroids(path_prefix=path_prefix)
        scaler_path = self._export_scaler(hash_=hash_, path_prefix=path_prefix)
        metadata_path = self._export_metadata(
            centroids_path=centroids_path,
            hash_=hash_,
            path_prefix=path_prefix,
            scaler_path=scaler_path,
        )

        return metadata_path, centroids_path, scaler_path

    @classmethod
    def from_file(cls, path_prefix: Optional[Path] = None, hash_: Optional[str] = None, metadata_path: Optional[Path] = None, centroids_path: Optional[Path] = None) -> "WKMeans":

        metadata = None
        centroids = None


        if path_prefix is None and metadata_path is None and centroids_path is None and hash_ is None:
            raise ValueError("Either path_prefix, metadata_path, centroids_path, or hash_ must be provided")

        if path_prefix is not None and hash_ is not None:
            metadata_path = path_prefix / f"metadata:{hash_}.yaml"
            centroids_path = path_prefix / f"centroids:{hash_}.bytes"
        elif centroids_path is None and metadata_path is not None:
            with open(metadata_path, "r") as f:
                metadata = yaml.load(f, Loader=yaml.FullLoader)
            centroids_path = metadata_path.parent / metadata["centroids"]
            
        if centroids_path is None or metadata_path is None:
            raise ValueError("Either path_prefix, metadata_path, centroids_path, or hash_ must be provided")

        if metadata is None:
            with open(metadata_path, "r") as f:
                metadata = yaml.load(f, Loader=yaml.FullLoader)
        
        with open(centroids_path, "rb") as f:
            flat = np.frombuffer(f.read(), dtype=np.float64)

        k = metadata["k"]
        
        if flat.size % k != 0:
            raise ValueError(
                f"Centroid blob length {flat.size} is not divisible by k={k}"
            )
        M = flat.reshape(k, -1)
        # predict iterates centroids; must be k separate 1D arrays, not a flat buffer
        centroids = [M[i].copy() for i in range(k)]

        scaler = None
        scaler_filename = metadata.get("scaler")
        if scaler_filename:
            scaler_path = metadata_path.parent / scaler_filename
            with open(scaler_path, "rb") as f:
                # cloudpickle output is loadable by stdlib pickle
                scaler = pickle.load(f)

        return cls(
            k=k,
            p=metadata["p"],
            tolerance=metadata["tolerance"],
            max_iter=metadata["max_iter"],
            seed=metadata["seed"],
            centroids=centroids,
            features=metadata["features"],
            scaler=scaler,
        )