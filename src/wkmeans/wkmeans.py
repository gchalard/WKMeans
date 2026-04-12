from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import numpy as np
import ot
from pathlib import Path
import yaml


@dataclass
class WKMeans:
    k: int # number of clusters
    p: float = 1 # order of the Wasserstein distance
    tolerance: float = 1e-4 # convergence threshold
    max_iter: int = 100 # maximum number of iterations
    seed: int = 42 # random seed
    centroids: list[np.ndarray] = field(default_factory=list) # list of centroids
    features: list[str] = field(default_factory=list) # list of features


    def wasserstein_distance(self, mu1, mu2):
        """Compute the p-Wasserstein distance between two empirical distributions (uniform mass on each point)."""
        mu1 = np.atleast_1d(np.asarray(mu1, dtype=float))
        mu2 = np.atleast_1d(np.asarray(mu2, dtype=float))
        n1, n2 = len(mu1), len(mu2)
        # Marginals must be in the simplex (non-negative, sum=1) and match cost matrix dimensions
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        a = a / a.sum()
        b = b / b.sum()
        M = ot.dist(mu1.reshape(-1, 1), mu2.reshape(-1, 1), metric="minkowski", p=self.p)
        return ot.emd2(a, b, M)

    def wasserstein_barycenter(self, cluster_samples):
        """Compute the Wasserstein barycenter (median of sorted distributions)."""
        sorted_samples = np.sort(cluster_samples, axis=0)
        return np.median(sorted_samples, axis=0)

    def fit(self, samples):
        """
        Fit the WK-means clustering algorithm.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)
        """
        # Initialize centroids by randomly selecting k samples
        np.random.seed(self.seed)
        self.centroids = [samples[i] for i in np.random.choice(len(samples), self.k, replace=False)]

        for _ in range(self.max_iter):
            clusters = {i: [] for i in range(self.k)}

            # Assign each sample to the closest centroid
            for sample in samples:
                distances = [self.wasserstein_distance(sample, centroid) for centroid in self.centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(sample)

            # Update centroids as Wasserstein barycenters
            new_centroids = []
            for i in range(self.k):
                if clusters[i]:
                    new_centroids.append(self.wasserstein_barycenter(np.array(clusters[i])))
                else:
                    new_centroids.append(self.centroids[i])  # Keep previous centroid if no samples assigned

            # Compute loss function (sum of Wasserstein distances)
            loss = sum(self.wasserstein_distance(self.centroids[i], new_centroids[i]) for i in range(self.k))

            # Check convergence
            if loss < self.tolerance:
                break

            self.centroids = new_centroids

    def predict(self, samples):
        """
        Predict the cluster for each sample.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)

        Returns:
        - List of cluster indices
        """
        return [np.argmin([self.wasserstein_distance(sample, centroid) for centroid in self.centroids]) for sample in samples]

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


    def _export_metadata(self, centroids_path: Path, hash_: str, path_prefix: Path) -> None:
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
                    not attr == "features"
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

        with open(path_prefix / f"metadata:{hash_}.yaml", "w") as f:
            yaml.dump(metadata, f)
        


    def export(self, path_prefix: Path) -> None:
        centroids_path, hash_ = self._export_centroids(path_prefix=path_prefix)
        self._export_metadata(centroids_path=centroids_path, hash_=hash_, path_prefix=path_prefix)

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

        return cls(
            k=k,
            p=metadata["p"],
            tolerance=metadata["tolerance"],
            max_iter=metadata["max_iter"],
            seed=metadata["seed"],
            centroids=centroids,
            features=metadata["features"]
        )