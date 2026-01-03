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


class AutoTuner:
    """Automated parameter tuner for ObjectClusterer."""

    # Search space
    PCA_GRID = [16, 24, 32, 48]
    SCALE_GRID = [0.015, 0.025, 0.04]

    def __init__(self):
        """Initialize auto-tuner."""
        pass

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
                    "min_cluster_size": 5, # Will be overridden by scale
                    "min_samples": 5,      # Will be overridden by scale
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
                    
                    # Score Calculation
                    # -----------------
                    # Goal: 3-15 clusters, Low Noise.
                    
                    score = 0.0
                    
                    # Penalty for too few/many clusters
                    if n_clusters < 2:
                        score -= 50  # Basically useless
                    elif n_clusters > 20:
                        score -= 20  # Too fragmented
                    else:
                        score += 20  # Good range
                        
                    # Penalty for noise
                    # We accept up to 30% noise. Above that, heavy penalty.
                    score -= (noise_ratio * 50) 
                    
                    # Tie-breaker: Prefer lower PCA (faster/simpler) and mid-range scale
                    if pca_dim == 32: score += 5
                    if scale == 0.025: score += 5

                    logger.info(
                        f"[Tune] PCA={pca_dim} Scale={scale:.3f} -> "
                        f"Clusters={n_clusters} Noise={noise_ratio:.2f} Score={score:.2f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_result = TuningResult(
                            best_config=config,
                            score=score,
                            pca_dims=pca_dim,
                            cluster_scale=scale,
                            n_clusters=n_clusters,
                            noise_ratio=noise_ratio
                        )

                except Exception as e:
                    logger.warning(f"Tuning failed for config {config}: {e}")
                    continue

        elapsed = time.time() - start_time
        logger.info(f"Auto-tune finished in {elapsed:.2f}s. Winner: PCA={best_result.pca_dims}, Scale={best_result.cluster_scale}")
        
        return best_result
