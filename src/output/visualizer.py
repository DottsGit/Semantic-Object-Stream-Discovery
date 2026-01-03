"""Visualization module for clustering results."""

from __future__ import annotations

import cv2
import numpy as np

from src.clustering.clusterer import ClusterResult


class ClusterVisualizer:
    """Real-time 2D scatter plot visualizer using OpenCV."""

    def __init__(self, width: int = 400, height: int = 400, bg_color: tuple = (30, 30, 30)):
        """Initialize visualizer.

        Args:
            width: Window width
            height: Window height
            bg_color: Background color (BGR)
        """
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.padding = 40

    def draw(self, result: ClusterResult | None) -> np.ndarray:
        """Draw scatter plot of clusters.

        Args:
            result: Clustering result with 2D embeddings

        Returns:
            OpenCV image (BGR)
        """
        # Create blank canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color

        if result is None or result.embeddings_2d is None:
            cv2.putText(
                canvas,
                "Waiting for clusters...",
                (20, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )
            return canvas

        points = result.embeddings_2d
        labels = result.labels

        # Normalize points to fit canvas
        if len(points) > 0:
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            range_vals = max_vals - min_vals
            
            # Avoid division by zero
            range_vals[range_vals == 0] = 1.0

            # Scale to usable area (minus padding)
            usable_w = self.width - 2 * self.padding
            usable_h = self.height - 2 * self.padding
            
            normalized = (points - min_vals) / range_vals
            
            # Map to pixels
            pixel_coords = normalized * [usable_w, usable_h] + self.padding
            pixel_coords = pixel_coords.astype(np.int32)

            # Draw points
            for i, (x, y) in enumerate(pixel_coords):
                label = labels[i]
                
                if label == -1:
                    color = (100, 100, 100)  # Noise = Grey
                    radius = 1
                else:
                    # Get color from cluster info if available
                    if label in result.clusters:
                        color = result.clusters[label].color
                    else:
                        # Fallback random-ish color based on label
                        np.random.seed(label)
                        color = tuple(int(x) for x in np.random.randint(50, 255, 3))
                    radius = 2

                cv2.circle(canvas, (x, y), radius, color, -1)

            # Draw Centroids and Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            for cid, info in result.clusters.items():
                # We need to find the centroid in 2D space. 
                # The info.centroid is likely high-dim.
                # So we calculate the mean of the 2D points for this cluster.
                mask = labels == cid
                if np.any(mask):
                    cluster_points = pixel_coords[mask]
                    cx, cy = np.mean(cluster_points, axis=0).astype(int)
                    
                    # Draw label background
                    text = f"{info.label or cid}"
                    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
                    
                    cv2.rectangle(
                        canvas, 
                        (cx - 2, cy - th - 2), 
                        (cx + tw + 2, cy + 2), 
                        (0, 0, 0), 
                        -1
                    )
                    
                    # Draw label
                    cv2.putText(canvas, text, (cx, cy), font, 0.5, info.color, 1)

        # Status text
        status = f"Clusters: {result.n_clusters} | Noise: {result.noise_count}"
        if len(points) < 10000: # Don't show total if massive
             status += f" | Points: {len(points)}"
             
        cv2.putText(
            canvas,
            status,
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return canvas
