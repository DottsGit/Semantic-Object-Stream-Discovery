"""Auto-tuning module for clustering parameters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from src.clustering.clusterer import ObjectClusterer


@dataclass
class TuningResult:
    """Result of an auto-tuning run."""

    best_config: dict[str, Any]
    score: float
    pca_dims: int
    cluster_scale: float
    n_clusters: int
    noise_ratio: float
    silhouette: float = 0.0

    # The actual result and models from the winning config
    # This allows direct use without re-clustering
    cluster_result: Any = None  # ClusterResult
    scaler: Any = None
    pca_model: Any = None


class AutoTuner:
    """Automated parameter tuner for ObjectClusterer."""

    # Search space - expanded for better coverage
    PCA_GRID = [16, 24, 32, 48]
    # Smaller scales = smaller min_cluster_size = more clusters
    SCALE_GRID = [0.005, 0.010, 0.015, 0.025, 0.04]

    # Target cluster range for traffic scenes (pedestrians, bikes, cars, trucks, buses, etc.)
    MIN_CLUSTERS = 3
    MAX_CLUSTERS = 15
    IDEAL_CLUSTERS = 6  # Sweet spot for traffic

    def __init__(self):
        """Initialize auto-tuner."""
        pass

    def _compute_silhouette(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for cluster quality.

        Silhouette ranges from -1 to 1:
        - 1: Perfect clusters (tight, well-separated)
        - 0: Overlapping clusters
        - -1: Wrong assignments
        """
        try:
            from sklearn.metrics import silhouette_score

            # Need at least 2 clusters and some non-noise points
            valid_mask = labels >= 0
            if np.sum(valid_mask) < 10:
                return 0.0

            unique_labels = np.unique(labels[valid_mask])
            if len(unique_labels) < 2:
                return 0.0

            # Compute on non-noise samples only
            return silhouette_score(embeddings[valid_mask], labels[valid_mask])
        except Exception:
            return 0.0

    def tune(self, embeddings: np.ndarray) -> TuningResult:
        """Run grid search to find best parameters.

        Args:
            embeddings: Feature embeddings (N, 768)

        Returns:
            Best configuration and stats
        """
        best_score = -float("inf")
        best_result = None

        total_configs = len(self.PCA_GRID) * len(self.SCALE_GRID)
        logger.info(f"Starting auto-tune grid search ({total_configs} configs)...")

        start_time = time.time()

        for pca_dim in self.PCA_GRID:
            for scale in self.SCALE_GRID:
                # 1. Config
                config = {
                    "algorithm": "hdbscan",
                    "use_pca": True,
                    "pca_n_components": pca_dim,
                    "min_cluster_size": 5,  # Will be overridden by scale
                    "min_samples": 5,       # Will be overridden by scale
                    "cluster_scale": scale,
                }

                # 2. Run Clusterer
                try:
                    clusterer = ObjectClusterer(**config)
                    result = clusterer.fit(embeddings)

                    # 3. Evaluate
                    n_clusters = result.n_clusters
                    n_noise = result.noise_count
                    n_total = len(embeddings)
                    noise_ratio = n_noise / n_total if n_total > 0 else 1.0

                    # Compute silhouette score for cluster quality
                    silhouette = self._compute_silhouette(
                        clusterer._pca_model.transform(clusterer._scaler.transform(embeddings))
                        if clusterer._pca_model else clusterer._scaler.transform(embeddings),
                        result.labels
                    )

                    # Score Calculation (Improved)
                    # ============================
                    # Goal: 3-15 clusters, low noise, good separation

                    score = 0.0

                    # 1. Cluster count scoring (prefer 4-8 clusters for traffic)
                    if n_clusters < 2:
                        score -= 100  # Useless - everything in one bucket
                    elif n_clusters == 2:
                        score -= 30   # Too few - likely missing distinctions
                    elif n_clusters < self.MIN_CLUSTERS:
                        score += 10
                    elif n_clusters <= self.MAX_CLUSTERS:
                        # Reward being in ideal range, peak at IDEAL_CLUSTERS
                        distance_from_ideal = abs(n_clusters - self.IDEAL_CLUSTERS)
                        score += 40 - (distance_from_ideal * 3)
                    else:
                        # Too fragmented
                        score -= (n_clusters - self.MAX_CLUSTERS) * 5

                    # 2. Noise penalty (30% is acceptable for HDBSCAN)
                    if noise_ratio > 0.5:
                        score -= 40  # Too much noise
                    elif noise_ratio > 0.3:
                        score -= (noise_ratio - 0.3) * 100
                    else:
                        score += (0.3 - noise_ratio) * 30  # Bonus for low noise

                    # 3. Silhouette score (cluster quality) - major factor
                    # Silhouette ranges -1 to 1, we want higher
                    score += silhouette * 50

                    # 4. Minor tie-breakers
                    if pca_dim == 32:
                        score += 2
                    if 0.010 <= scale <= 0.020:
                        score += 2

                    logger.info(
                        f"[Tune] PCA={pca_dim} Scale={scale:.3f} -> "
                        f"Clusters={n_clusters} Noise={noise_ratio:.2f} "
                        f"Silhouette={silhouette:.3f} Score={score:.2f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_result = TuningResult(
                            best_config=config,
                            score=score,
                            pca_dims=pca_dim,
                            cluster_scale=scale,
                            n_clusters=n_clusters,
                            noise_ratio=noise_ratio,
                            silhouette=silhouette,
                            # Store the winning clusterer's state for direct reuse
                            cluster_result=result,
                            scaler=clusterer._scaler,
                            pca_model=clusterer._pca_model,
                        )

                except Exception as e:
                    logger.warning(f"Tuning failed for config {config}: {e}")
                    continue

        elapsed = time.time() - start_time
        if best_result:
            logger.info(
                f"Auto-tune finished in {elapsed:.2f}s. "
                f"Winner: PCA={best_result.pca_dims}, Scale={best_result.cluster_scale}, "
                f"Clusters={best_result.n_clusters}, Silhouette={best_result.silhouette:.3f}"
            )
        else:
            logger.error("Auto-tune failed to find any valid configuration")

        return best_result
