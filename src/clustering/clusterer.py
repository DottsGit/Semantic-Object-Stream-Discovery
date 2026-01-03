"""Unsupervised clustering module for object type discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler


class ClusterAlgorithm(Enum):
    """Supported clustering algorithms."""

    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    DBSCAN = "dbscan"


@dataclass
class ClusterInfo:
    """Information about a discovered cluster."""

    cluster_id: int
    size: int
    centroid: np.ndarray
    label: str = ""  # User-assigned or auto-generated label
    color: tuple[int, int, int] = (0, 255, 0)  # Display color (BGR)

    # Statistics
    mean_confidence: float = 0.0
    std_embedding: float = 0.0


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    labels: np.ndarray  # Cluster label for each sample
    n_clusters: int
    clusters: dict[int, ClusterInfo] = field(default_factory=dict)
    noise_count: int = 0  # Samples labeled as noise (-1)

    # Dimensionality reduction results (for visualization)
    embeddings_2d: np.ndarray | None = None


class ObjectClusterer:
    """Unsupervised clusterer for DINOv2 embeddings."""

    # Distinct colors for clusters (BGR)
    CLUSTER_COLORS = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Orange
        (255, 128, 0),  # Light blue
        (128, 255, 0),  # Light green
        (0, 128, 255),  # Light orange
    ]

    def __init__(
        self,
        algorithm: str = "hdbscan",
        n_clusters: int | None = None,
        min_cluster_size: int = 10,
        use_pca: bool = True,
        pca_n_components: int = 32,
        min_samples: int = 50,
        cluster_scale: float = 0.025,
    ):
        """Initialize the clusterer.

        Args:
            algorithm: Clustering algorithm (kmeans, hdbscan, dbscan)
            n_clusters: Number of clusters (required for kmeans, ignored for hdbscan)
            min_cluster_size: Minimum cluster size for HDBSCAN
            use_pca: Whether to use PCA for dimensionality reduction before clustering
            pca_n_components: Target dimensions for PCA
            min_samples: Minimum samples required before clustering
        """
        self.algorithm = ClusterAlgorithm(algorithm)
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.min_samples = min_samples
        self.cluster_scale = cluster_scale

        self._scaler = StandardScaler()
        self._pca_model: Any = None
        self._cluster_model: Any = None
        self._fitted = False
        self._last_result: ClusterResult | None = None

    def _init_pca(self, n_samples: int) -> Any:
        """Initialize PCA model."""
        from sklearn.decomposition import PCA
        
        # Determine components (cannot exceed min(n_samples, features))
        # feature dim is usually 768.
        n_components = min(self.pca_n_components, n_samples)
        
        return PCA(n_components=n_components, random_state=42)

    def _init_cluster_model(self, n_samples: int) -> Any:
        """Initialize the clustering model."""
        if self.algorithm == ClusterAlgorithm.KMEANS:
            from sklearn.cluster import KMeans

            n_clusters = self.n_clusters or 3
            return KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            )

        elif self.algorithm == ClusterAlgorithm.HDBSCAN:
            try:
                import hdbscan

                # Adjust min_cluster_size based on sample size to prevent fragmentation
                # For small datasets, use provided min (capped at likely max).
                # For large datasets, scale up (e.g. 1% of data).
                # This prevents getting 50 tiny clusters when we really want 3-5 big ones.
                
                # 1. Base floor from config
                base_min = self.min_cluster_size
                
                # 2. Scale factor (Dynamic)
                scale_min = int(n_samples * self.cluster_scale)
                
                # 3. Take the LARGER of the two (enforcing bigger clusters for bigger data)
                # But don't exceed a reasonable cap (e.g. 50% of data)
                final_min_cluster = max(base_min, scale_min)
                final_min_cluster = min(final_min_cluster, n_samples // 2)

                logger.info(f"Dynamic min_cluster_size: {final_min_cluster} (samples: {n_samples})")

                return hdbscan.HDBSCAN(
                    min_cluster_size=final_min_cluster,
                    min_samples=final_min_cluster,  # Conservative: Require high density to prevent bridging
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
            except ImportError:
                logger.error("hdbscan not installed. Run: pip install hdbscan")
                raise

        elif self.algorithm == ClusterAlgorithm.DBSCAN:
            from sklearn.cluster import DBSCAN

            return DBSCAN(
                eps=0.5,
                min_samples=max(5, self.min_cluster_size // 2),
                metric="euclidean",
            )

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            ClusterResult with labels and cluster information
        """
        n_samples = len(embeddings)

        if n_samples < self.min_samples:
            logger.warning(
                f"Not enough samples for clustering: {n_samples} < {self.min_samples}"
            )
            return ClusterResult(
                labels=np.full(n_samples, -1),
                n_clusters=0,
            )

        logger.info(f"Clustering {n_samples} embeddings with {self.algorithm.value}")

        # Normalize embeddings
        embeddings_scaled = self._scaler.fit_transform(embeddings)

        # Optional PCA dimensionality reduction
        if self.use_pca and embeddings.shape[1] > self.pca_n_components:
            self._pca_model = self._init_pca(n_samples)
            if self._pca_model:
                logger.info(f"Reducing dimensions with PCA: {embeddings.shape[1]} -> {self._pca_model.n_components}")
                embeddings_reduced = self._pca_model.fit_transform(embeddings_scaled)
            else:
                embeddings_reduced = embeddings_scaled
        else:
            embeddings_reduced = embeddings_scaled

        # Fit clustering model
        self._cluster_model = self._init_cluster_model(n_samples)
        labels = self._cluster_model.fit_predict(embeddings_reduced)

        # Create 2D embeddings for visualization using PCA (faster than UMAP)
        embeddings_2d = None
        try:
            from sklearn.decomposition import PCA
            pca_2d = PCA(n_components=2, random_state=42)
            embeddings_2d = pca_2d.fit_transform(embeddings_scaled)
        except ImportError:
            pass

        # Build cluster info
        unique_labels = set(labels)
        noise_count = np.sum(labels == -1)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        clusters = {}
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            mask = labels == label
            cluster_embeddings = embeddings[mask]

            clusters[label] = ClusterInfo(
                cluster_id=label,
                size=int(np.sum(mask)),
                centroid=np.mean(cluster_embeddings, axis=0),
                label=f"Cluster {label}",
                color=self.CLUSTER_COLORS[label % len(self.CLUSTER_COLORS)],
                std_embedding=float(np.mean(np.std(cluster_embeddings, axis=0))),
            )

        self._fitted = True
        self._last_result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            clusters=clusters,
            noise_count=int(noise_count),
            embeddings_2d=embeddings_2d,
        )

        logger.info(
            f"Found {n_clusters} clusters "
            f"(sizes: {[c.size for c in clusters.values()]}, noise: {noise_count})"
        )

        return self._last_result

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Array of cluster labels
        """
        if not self._fitted:
            logger.warning("Clusterer not fitted, returning -1 labels")
            return np.full(len(embeddings), -1)

        # Normalize
        embeddings_scaled = self._scaler.transform(embeddings)

        # PCA transform if used during fit
        if self._pca_model is not None:
            embeddings_reduced = self._pca_model.transform(embeddings_scaled)
        else:
            embeddings_reduced = embeddings_scaled

        # Predict
        if self.algorithm == ClusterAlgorithm.KMEANS:
            return self._cluster_model.predict(embeddings_reduced)
        else:
            # HDBSCAN/DBSCAN: use approximate prediction
            if hasattr(self._cluster_model, "approximate_predict"):
                labels, _ = self._cluster_model.approximate_predict(
                    self._cluster_model, embeddings_reduced
                )
                return labels
            else:
                # Fallback: assign to nearest centroid
                return self._assign_to_nearest_centroid(embeddings)

    def _assign_to_nearest_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign embeddings to nearest cluster centroid."""
        if self._last_result is None or not self._last_result.clusters:
            return np.full(len(embeddings), -1)

        centroids = np.vstack(
            [c.centroid for c in self._last_result.clusters.values()]
        )
        cluster_ids = list(self._last_result.clusters.keys())

        # Compute distances to all centroids
        from scipy.spatial.distance import cdist

        distances = cdist(embeddings, centroids, metric="cosine")
        nearest_idx = np.argmin(distances, axis=1)

        return np.array([cluster_ids[i] for i in nearest_idx])

    def get_cluster_info(self, cluster_id: int) -> ClusterInfo | None:
        """Get information about a specific cluster."""
        if self._last_result and cluster_id in self._last_result.clusters:
            return self._last_result.clusters[cluster_id]
        return None

    def set_cluster_label(self, cluster_id: int, label: str) -> None:
        """Set a human-readable label for a cluster."""
        if self._last_result and cluster_id in self._last_result.clusters:
            self._last_result.clusters[cluster_id].label = label
            logger.info(f"Set cluster {cluster_id} label to: {label}")

    @property
    def is_fitted(self) -> bool:
        """Check if clusterer has been fitted."""
        return self._fitted

    @property
    def result(self) -> ClusterResult | None:
        """Get the last clustering result."""
        return self._last_result
