"""Object tracking module using SORT-like algorithm with cluster integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from filterpy.kalman import KalmanFilter
from loguru import logger
from scipy.optimize import linear_sum_assignment
from collections import deque, Counter

if TYPE_CHECKING:
    from src.detection.detector import Detection
    from src.features.extractor import ObjectFeature


def iou(bb1: tuple, bb2: tuple) -> float:
    """Calculate IoU between two bounding boxes.

    Args:
        bb1, bb2: Bounding boxes as (x1, y1, x2, y2)

    Returns:
        Intersection over Union value
    """
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    union_area = bb1_area + bb2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def bbox_to_z(bbox: tuple) -> np.ndarray:
    """Convert bounding box to Kalman state [x, y, s, r].

    Args:
        bbox: (x1, y1, x2, y2)

    Returns:
        [x_center, y_center, scale (area), aspect_ratio]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2
    y = bbox[1] + h / 2
    s = w * h  # scale (area)
    r = w / float(h) if h > 0 else 1.0  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def z_to_bbox(z: np.ndarray) -> tuple:
    """Convert Kalman state to bounding box.

    Args:
        z: [x_center, y_center, scale, aspect_ratio]

    Returns:
        (x1, y1, x2, y2)
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w if w > 0 else 0
    x1 = z[0] - w / 2
    y1 = z[1] - h / 2
    x2 = z[0] + w / 2
    y2 = z[1] + h / 2
    return (int(x1), int(y1), int(x2), int(y2))


@dataclass
class Track:
    """A tracked object with state and history."""

    track_id: int
    bbox: tuple[int, int, int, int]
    cluster_id: int = -1  # Assigned cluster (-1 = unknown)
    cluster_label: str = "Unknown"

    # Kalman filter state
    kf: KalmanFilter = field(default=None, repr=False)

    # Tracking state
    hits: int = 0
    hit_streak: int = 0
    age: int = 0
    time_since_update: int = 0

    # History
    positions: list[tuple[int, int]] = field(default_factory=list)
    velocities: list[tuple[float, float]] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    cluster_history: Counter = field(default_factory=lambda: Counter())
    locked_cluster_id: int | None = None  # To prevent flickering

    # Flow statistics
    total_distance: float = 0.0
    entry_time: float = 0.0
    entry_position: tuple[int, int] | None = None

    def __post_init__(self):
        if self.kf is None:
            self._init_kalman()
        self.entry_position = self.center
        self.positions.append(self.center)

    def _init_kalman(self):
        """Initialize Kalman filter for tracking."""
        # State: [x, y, s, r, vx, vy, vs]
        # Measurement: [x, y, s, r]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        z = bbox_to_z(self.bbox)
        self.kf.x[:4] = z

    @property
    def center(self) -> tuple[int, int]:
        """Get current center position."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def velocity(self) -> tuple[float, float]:
        """Get current velocity."""
        if self.kf is not None:
            return (float(self.kf.x[4]), float(self.kf.x[5]))
        return (0.0, 0.0)

    @property
    def speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2)



    def predict(self, reset_streak: bool = True) -> tuple:
        """Predict next state and return predicted bbox."""
        # Handle negative area
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        if reset_streak and self.time_since_update > 2:
            self.hit_streak = 0

        predicted_bbox = z_to_bbox(self.kf.x.flatten())
        return predicted_bbox

    @property
    def stable_cluster_id(self) -> int:
        """Get the most frequent cluster ID from history."""
        # Return locked ID if set
        if self.locked_cluster_id is not None:
            return self.locked_cluster_id

        if not self.cluster_history:
            return self.cluster_id
        
        # Check if we should lock
        most_common = self.cluster_history.most_common(1)[0]
        cid, count = most_common
        total = self.cluster_history.total()
        
        # Lock if > 60% agreement and > 10 samples
        if total > 10 and (count / total) > 0.6 and cid >= 0:
            self.locked_cluster_id = cid
            
        return cid

    def update(self, detection: "Detection", embedding: np.ndarray | None = None):
        """Update track with new detection."""
        old_center = self.center

        # Update Kalman filter
        z = bbox_to_z(detection.bbox)
        self.kf.update(z)

        # Update bbox
        self.bbox = z_to_bbox(self.kf.x.flatten())

        # Update tracking state
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Update history
        new_center = self.center
        self.positions.append(new_center)
        self.velocities.append(self.velocity)

        # Update distance
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        self.total_distance += np.sqrt(dx**2 + dy**2)

        # Store embedding
        if embedding is not None:
            self.embeddings.append(embedding)
            # Keep only recent embeddings
            if len(self.embeddings) > 10:
                self.embeddings = self.embeddings[-10:]

    @property
    def mean_embedding(self) -> np.ndarray | None:
        """Get mean embedding for cluster assignment."""
        if self.embeddings:
            return np.mean(self.embeddings, axis=0)
        return None


class ObjectTracker:
    """Multi-object tracker with cluster integration."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """Initialize tracker.

        Args:
            max_age: Max frames to keep lost track
            min_hits: Min hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_count = 0

    def update(
        self,
        detections: list["Detection"],
        features: list["ObjectFeature"] | None = None,
        cluster_labels: np.ndarray | None = None,
        cluster_names: dict[int, str] | None = None,
        delay_frames: int = 0,
    ) -> list[Track]:
        """Update tracks with new detections, handling potential latency.

        Args:
            detections: List of detections in current frame
            features: Optional features for each detection
            cluster_labels: Optional cluster labels for each detection
            cluster_names: Optional mapping of cluster IDs to names
            delay_frames: Estimated lag (in frames) of the detections


        Returns:
            List of currently active tracks
        """
        self._frame_count += 1

        # NOTE: Predict is handled by predict_only() in the main loop for smooth visualization.
        # We assume tracks are already predicted to the current time t.
        
        # Match detections to tracks
        # If there is a delay, match against back-projected locations
        matched, unmatched_dets, unmatched_tracks = self._associate(detections, delay_frames, features)

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self._tracks[track_idx]
            det = detections[det_idx]

            # Get embedding if available
            embedding = None
            if features and det_idx < len(features):
                embedding = features[det_idx].embedding

            # If delayed, project detection forward to current time using track velocity
            if delay_frames > 0:
                vx, vy = track.velocity
                dx = int(vx * delay_frames)
                dy = int(vy * delay_frames)
                
                # Clone detection to avoid modifying original
                from src.detection.detector import Detection
                det = Detection(
                    bbox=(det.bbox[0] + dx, det.bbox[1] + dy, det.bbox[2] + dx, det.bbox[3] + dy),
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    frame_number=det.frame_number + delay_frames,
                    timestamp=det.timestamp, # Approx
                )

            track.update(det, embedding)

            # Update cluster assignment
            if cluster_labels is not None and det_idx < len(cluster_labels):
                track.cluster_id = int(cluster_labels[det_idx])
                if cluster_names and track.cluster_id in cluster_names:
                    track.cluster_label = cluster_names[track.cluster_id]
                
                # Update history
                track.cluster_history[track.cluster_id] += 1
                # Decay old history occasionally or strictly keep length?
                # Counter doesn't support maxlen.
                # Let's just keep it simple for now or change to deque.
                # If using deque of IDs:
                pass

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]

            embedding = None
            if features and det_idx < len(features):
                embedding = features[det_idx].embedding

            cluster_id = -1
            cluster_label = "Unknown"
            if cluster_labels is not None and det_idx < len(cluster_labels):
                cluster_id = int(cluster_labels[det_idx])
                if cluster_names and cluster_id in cluster_names:
                    cluster_label = cluster_names[cluster_id]

            track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                entry_time=det.timestamp,
            )

            # Fast-forward new track if delayed
            if delay_frames > 0:
                for _ in range(delay_frames):
                    track.predict(reset_streak=False)

            if embedding is not None:
                track.embeddings.append(embedding)

            self._tracks.append(track)
            self._next_id += 1

        # Remove dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        return [t for t in self._tracks if t.hit_streak >= self.min_hits or self._frame_count <= self.min_hits]

    def _associate(
        self, detections: list["Detection"], delay_frames: int = 0, features: list["ObjectFeature"] | None = None
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections with existing tracks using Hungarian algorithm.

        Returns:
            (matched pairs, unmatched detection indices, unmatched track indices)
        """
        if not self._tracks:
            return [], list(range(len(detections))), []

        if not detections:
            return [], [], list(range(len(self._tracks)))

        # Build IoU cost matrix
        n_tracks = len(self._tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets))

        for t, track in enumerate(self._tracks):
            # Calculate comparison bbox (back-projected if delayed)
            compare_bbox = track.bbox
            if delay_frames > 0:
                 vx, vy = track.velocity
                 dx = int(vx * delay_frames)
                 dy = int(vy * delay_frames)
                 compare_bbox = (
                     track.bbox[0] - dx,
                     track.bbox[1] - dy,
                     track.bbox[2] - dx,
                     track.bbox[3] - dy,
                 )

            for d, det in enumerate(detections):
                iou_score = iou(compare_bbox, det.bbox)
                
                # Appearance cost (cosine distance)
                appearance_cost = 0.0
                if features and d < len(features) and track.mean_embedding is not None:
                    # Cosine distance = 1 - (A . B) / (|A| |B|)
                    # Embeddings are usually normalized by DINOv2Extractor, but let's be safe
                    track_emb = track.mean_embedding
                    det_emb = features[d].embedding
                    
                    # Normalize if needed (assuming already normalized for speed, but...)
                    sim = np.dot(track_emb, det_emb) / (np.linalg.norm(track_emb) * np.linalg.norm(det_emb) + 1e-6)
                    appearance_cost = 1.0 - max(0.0, min(1.0, sim))

                # Hybrid cost: 80% IoU, 20% Appearance
                # prevents lane jumping (IoU=0) while allowing fast approach (IoU=0.2)
                if appearance_cost > 0:
                     cost_matrix[t, d] = 0.8 * (1 - iou_score) + 0.2 * appearance_cost
                else:
                     cost_matrix[t, d] = 1 - iou_score

        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by IoU threshold (allow looser IoU if appearance is strong)
        matched = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))

        for t, d in zip(row_indices, col_indices):
            # Dynamic threshold?
            # If cost is low, match.
            # But we must verify IoU is not ZERO (impossible match)
            is_match = False
            if cost_matrix[t, d] <= (1 - self.iou_threshold):
                is_match = True
            
            # Rescue match if IoU failed (fast movement) but appearance is VERY high
            # (e.g. oncoming car jumped far but looks identical)
            # Cost logic above already blends them. If cost is low, it means good match.
            # Strict IoU check solely to prevent teleportation across screen
            current_iou = 1 - cost_matrix[t, d] # Approx, not real IoU if hybrid
            
            if is_match:
                matched.append((t, d))
                if d in unmatched_dets:
                    unmatched_dets.remove(d)
                if t in unmatched_tracks:
                    unmatched_tracks.remove(t)

        return matched, unmatched_dets, unmatched_tracks

    @property
    def tracks(self) -> list[Track]:
        """Get all current tracks."""
        return self._tracks

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Get confirmed tracks only."""
        return [t for t in self._tracks if t.hit_streak >= self.min_hits]

    def get_tracks_by_cluster(self, cluster_id: int) -> list[Track]:
        """Get all tracks assigned to a cluster."""
        return [t for t in self._tracks if t.cluster_id == cluster_id]

    def reset(self):
        """Reset the tracker."""
        self._tracks = []
        self._next_id = 1
        self._frame_count = 0

    def predict_only(self) -> list[Track]:
        """Advance state without update (for frames with no detection)."""
        self._frame_count += 1
        for track in self._tracks:
            track.predict(reset_streak=False)
        
        # Return currently valid tracks
        return [t for t in self._tracks if t.hit_streak >= self.min_hits or self._frame_count <= self.min_hits]
