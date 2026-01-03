"""Main SOSD pipeline orchestrating all components."""

from __future__ import annotations

import os

# CRITICAL: Limit threads to prevent CPU starvation by background processes
# This must be set BEFORE numpy/cv2/sklearn are imported
# We enforce this globally to ensure the video player (UI thread) is never starved
# by heavy math operations in background threads/processes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Any

import cv2
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

# Redirect C-level stderr to null to silence FFMPEG/TLS noise
try:
    # 1. Save original stderr fd
    original_stderr_fd = os.dup(sys.stderr.fileno())
    
    # 2. Open devnull
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    # 3. Replace stderr (fd 2) with devnull
    os.dup2(devnull, sys.stderr.fileno())
    
    # 4. Create a new sys.stderr pointing to the saved original stderr
    # This allows Python logs to still show up
    sys.stderr = os.fdopen(original_stderr_fd, 'w')
    
    # 5. Close the devnull handle (dup2 verified it)
    os.close(devnull)
    
    # 6. Configure loguru to use the new sys.stderr
    logger.remove()
    logger.add(sys.stderr, level="INFO")

except Exception as e:
    # If redirection fails (e.g. no console), just continue
    pass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.analytics.flow_analyzer import FlowAnalyzer
from src.clustering.clusterer import ObjectClusterer
from src.detection.detector import ObjectDetector
from src.features.extractor import DINOv2Extractor, FeatureBuffer
from src.ingestion.video_source import BufferedVideoSource, Frame, create_video_source
from src.output.gcp_outputs import BigQueryWriter, PubSubPublisher
from src.output.visualizer import ClusterVisualizer
from src.tracking.tracker import ObjectTracker

console = Console()


class PipelineState(Enum):
    """Pipeline state machine."""

    INITIALIZING = "initializing"
    WARMUP = "warmup"
    CLUSTERING = "clustering"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""

    # Video
    video_source: str = ""
    video_type: str = "auto"
    target_resolution: tuple[int, int] = (1280, 720)  # Higher res for better detection
    processing_fps: int = -1
    display_fps: int = -1
    pca_n_components: int = 32
    min_cluster_scale: float = 0.025
    auto_tune: bool = False  # -1 = auto (source FPS)

    # Detection
    detector_model: str = "yolov8m.pt"  # Medium model for best accuracy
    detection_confidence: float = 0.4  # Increased to filter phantom objects
    detection_classes: list[int] | None = None
    min_box_area: int = 500  # Increased to ignore noise

    # Features
    feature_model: str = "facebook/dinov2-base"
    device: str = "cuda"

    # Clustering
    warmup_duration: int = 60  # seconds
    min_samples: int = 200  # Aggressive merging
    cluster_algorithm: str = "hdbscan"
    n_clusters: int | None = None
    min_cluster_size: int = 400  # Only large, distinct flows
    recluster_interval: int = 300  # seconds
    max_buffer_size: int = 10000  # Max samples to store for clustering

    # Tracking
    max_track_age: int = 10  # Very low to kill ghosts instantly (1s)
    min_hits: int = 3  # Reduced for faster detection (0.3s)
    iou_threshold: float = 0.2  # Threshold tuned for 80/20 hybrid cost

    # Output
    enable_display: bool = True
    enable_pubsub: bool = False
    pubsub_project: str = ""
    pubsub_topic: str = "sosd-events"
    enable_bigquery: bool = False
    bigquery_dataset: str = "sosd"
    bigquery_table: str = "tracking_events"
    log_interval: int = 5

    # Cluster labels (user-provided names)
    cluster_labels: dict[int, str] = field(default_factory=dict)


def run_clustering_job(config: dict, embeddings: np.ndarray) -> Any:
    """Run clustering in a separate process (Stateless).
    
    Args:
        config: Dictionary of clustering parameters
        embeddings: Numpy array of embeddings
        
    Returns:
        ClusterResult object (lightweight dataclass)
    """
    import os
    import sys
    import psutil
    from loguru import logger
    from src.clustering.clusterer import ObjectClusterer
    
    # Configure logger for worker process (force stderr output)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(sys.stdout, level="INFO")
    
    # CRITICAL: Lower process priority to ensure Video/UI thread is never starved
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:
        logger.warning(f"Failed to lower process priority: {e}")

    # Enforce limits again just to be safe in this process
    os.environ["OMP_NUM_THREADS"] = "1"
    
    logger.info(f"Starting clustering process for {len(embeddings)} samples...")
    
    # Reconstruct clusterer from config
    clusterer = ObjectClusterer(**config)
    result = clusterer.fit(embeddings)
    
    # Return everything needed to sync state
    return {
        "result": result,
        "scaler": clusterer._scaler,
        "pca_model": clusterer._pca_model,
        "cluster_model": clusterer._cluster_model if clusterer.algorithm != clusterer.algorithm.HDBSCAN else None 
        # Note: HDBSCAN model might be heavy or complex to pickle, but often fine. 
        # If it fails, we fall back to centroid matching which only needs 'result'.
        # For now, let's NOT return HDBSCAN object to keep it light/safe, 
        # relying on centroid prediction (which is robust).
    }


def run_autotune_job(embeddings: np.ndarray) -> dict:
    """Run auto-tuning in background process.
    
    Args:
        embeddings: Feature embeddings
        
    Returns:
        Dictionary with 'best_result' (TuningResult)
    """
    import os
    import sys
    import psutil
    from loguru import logger
    from src.clustering.autotune import AutoTuner
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(sys.stdout, level="INFO")
    
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:
        logger.warning(f"Failed to lower process priority: {e}")
        
    os.environ["OMP_NUM_THREADS"] = "1"
    
    tuner = AutoTuner()
    result = tuner.tune(embeddings)
    
    return {"best_result": result}
    

def warmup_worker() -> bool:
    """Pre-load heavy libraries only (imports only)."""
    import os
    # Limit threads just in case
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        import numpy as np
        import sklearn.cluster
        import sklearn.preprocessing
        from sklearn.decomposition import PCA
        try:
            import hdbscan
        except ImportError:
            pass
            
        return True
    except ImportError:
        return False


class Pipeline:
    """Main SOSD pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.state = PipelineState.INITIALIZING

        # Components
        self._video_source: BufferedVideoSource | None = None
        self._detector: ObjectDetector | None = None
        self._feature_extractor: DINOv2Extractor | None = None
        self._feature_buffer: FeatureBuffer | None = None
        self._clusterer: ObjectClusterer | None = None
        self._tracker: ObjectTracker | None = None
        self._flow_analyzer: FlowAnalyzer | None = None

        # GCP outputs
        self._pubsub: PubSubPublisher | None = None
        self._bigquery: BigQueryWriter | None = None
        
        # Visualization
        self._cluster_visualizer: ClusterVisualizer | None = None
        
        # Async processing
        self._executor: ThreadPoolExecutor | None = None
        self._process_executor: ProcessPoolExecutor | None = None
        self._future: Future | None = None
        self._clustering_future: Future | None = None

        # State
        self._stop_event = Event()
        self._start_time: float = 0
        self._warmup_start: float = 0
        self._last_cluster_time: float = 0
        self._tuned: bool = False
        self._frame_count: int = 0
        self._frame_count: int = 0
        self._last_log_time: float = 0
        self._last_process_time: float = 0
        self._last_process_time: float = 0
        self._process_interval: float = 0
        self._last_detections: list = []

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        # Video source
        source = create_video_source(
            self.config.video_source,
            self.config.video_type,
            self.config.target_resolution,
            self.config.display_fps,
        )
        self._video_source = BufferedVideoSource(source, buffer_size=30)

        # Detector
        self._detector = ObjectDetector(
            model_name=self.config.detector_model,
            confidence_threshold=self.config.detection_confidence,
            classes=self.config.detection_classes,
            min_box_area=self.config.min_box_area,
            device=self.config.device,
        )
        self._detector.load()

        # Feature extractor
        self._feature_extractor = DINOv2Extractor(
            model_name=self.config.feature_model,
            device=self.config.device,
        )
        self._feature_extractor.load()

        # Feature buffer for warmup
        self._feature_buffer = FeatureBuffer(max_size=self.config.max_buffer_size)

        # Clusterer
        # Clusterer
        self._clusterer = ObjectClusterer(
            algorithm=self.config.cluster_algorithm,
            n_clusters=self.config.n_clusters,
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            use_pca=True,
            pca_n_components=self.config.pca_n_components,
            cluster_scale=self.config.min_cluster_scale
        )

        # Tracker
        self._tracker = ObjectTracker(
            max_age=self.config.max_track_age,
            min_hits=self.config.min_hits,
            iou_threshold=self.config.iou_threshold,
        )

        # Flow analyzer
        self._flow_analyzer = FlowAnalyzer()

        # GCP outputs
        if self.config.enable_pubsub:
            self._pubsub = PubSubPublisher(
                self.config.pubsub_project,
                self.config.pubsub_topic,
            )
            self._pubsub.connect()

            self._bigquery = BigQueryWriter(
                self.config.pubsub_project,
                self.config.bigquery_dataset,
                self.config.bigquery_table,
            )
            self._bigquery.connect()
            
        # Visualizer
        self._cluster_visualizer = ClusterVisualizer(width=600, height=600)

        logger.info("All components initialized")
        self._process_interval = 1.0 / self.config.processing_fps

        logger.info("All components initialized")
        self._process_interval = 1.0 / self.config.processing_fps
        self._process_interval = 1.0 / self.config.processing_fps
        # Use 1 worker for detection thread
        self._executor = ThreadPoolExecutor(max_workers=1)
        # Use 1 process for heavy clustering
        self._process_executor = ProcessPoolExecutor(max_workers=1)
        
        # Pre-warm the process pool to absorb import overhead
        logger.info("Pre-warming background process pool...")
        self._process_executor.submit(warmup_worker)

    def _run_detection(self, frame: Frame) -> tuple[Frame, list, list, np.ndarray | None, dict]:
        """Run heavy detection task in background."""
        detections = self._detector.detect(frame)
        
        features = []
        if detections:
            features = self._feature_extractor.extract(frame.image, detections)
            
        cluster_labels = None
        cluster_names = self.config.cluster_labels.copy() # Safe copy
        
        if self._clusterer.is_fitted and features:
            embeddings = np.vstack([f.embedding for f in features])
            cluster_labels = self._clusterer.predict(embeddings)
            
            # We can't access result.clusters safely if it changes, but reading is mostly fine
            # For strict safety we might need a lock, but let's assume atomic replace
            if self._clusterer.result:
                for cid, info in self._clusterer.result.clusters.items():
                   if cid not in cluster_names:
                        cluster_names[cid] = info.label

        return frame, detections, features, cluster_labels, cluster_names

    def _update_tracker(self, frame: Frame, detections: list, features: list, cluster_labels: Any, cluster_names: dict, delay: int = 0) -> dict:
        """Update tracker with results from background task."""
        self._frame_count += 1 # Note: This might jump if we rely on this for "processed" count
        
        # Update tracker
        tracks = self._tracker.update(  
            detections,
            features,
            cluster_labels,
            cluster_names,
            delay_frames=delay,
        )

        # Update flow analytics
        if self._flow_analyzer:
            self._flow_analyzer.update(tracks, cluster_names)
            
        # Write to GCP if enabled
        if self._pubsub:
            for track in tracks:
                self._pubsub.publish_track_event("track_updated", track)

        if self._bigquery:
            for track in tracks:
                self._bigquery.write_track(track)
                
        return {
            "frame": frame,
            "detections": detections,
            "features": features,
            "tracks": tracks,
            "cluster_labels": cluster_labels,
        }

    def _warmup_phase(self) -> bool:
        """Run the warmup phase to collect data for clustering.

        Returns:
            True if warmup succeeded, False otherwise
        """
        logger.info(f"Starting warmup phase ({self.config.warmup_duration}s)...")
        self.state = PipelineState.WARMUP
        self._warmup_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Warmup", total=self.config.warmup_duration)

            while not self._stop_event.is_set():
                elapsed = time.time() - self._warmup_start
                progress.update(task, completed=min(elapsed, self.config.warmup_duration))

                if elapsed >= self.config.warmup_duration:
                    break

                # Get frame
                frame = self._video_source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Detect and extract features
                detections = self._detector.detect(frame)
                if detections:
                    features = self._feature_extractor.extract(frame.image, detections)
                    self._feature_buffer.add(features)

                # Show progress
                if int(elapsed) % 5 == 0:
                    progress.update(
                        task,
                        description=f"Warmup ({len(self._feature_buffer)} samples)",
                    )

                # Visualize
                if self.config.enable_display:
                    display_result = {
                        "frame": frame,
                        "detections": detections,
                        "tracks": [],
                        "is_warmup": True
                    }
                    self._display_frame(display_result)

        # Check if we have enough samples
        if len(self._feature_buffer) < self.config.min_samples:
            logger.warning(
                f"Insufficient samples for clustering: {len(self._feature_buffer)} < {self.config.min_samples}"
            )
            return False

        logger.info(f"Warmup complete with {len(self._feature_buffer)} samples")
        return True

    def _perform_clustering_task(self) -> None:
        """Submit clustering task to background process."""
        logger.info("Submitting background unsupervised clustering...")
        
        embeddings = self._feature_buffer.embeddings
        logger.info(f"Buffer size: {len(embeddings)}")
        
        # Downsample if too large to ensure constant-time clustering (~1-2s max)
        # This prevents the "60s warmup" pause issue.
        MAX_SAMPLES = 5000
        if len(embeddings) > MAX_SAMPLES:
            logger.info(f"Downsampling clustering input: {len(embeddings)} -> {MAX_SAMPLES}")
            # Use random sampling
            indices = np.random.choice(len(embeddings), MAX_SAMPLES, replace=False)
            embeddings = embeddings[indices]
            logger.info("Downsampling complete.")
        if self.config.auto_tune and not self._tuned:
            logger.info("Auto-tuning enabled: Starting grid search...")
            logger.info(f"Submitting {len(embeddings)} samples to autotune process...")
            self._clustering_future = self._process_executor.submit(
                run_autotune_job, 
                embeddings
            )
        else:
            # Standard Clustering
            
            # This avoids pickling the heavy ObjectClusterer instance
            config = {
                "algorithm": self._clusterer.algorithm.value,
                "n_clusters": self._clusterer.n_clusters,
                "min_cluster_size": self._clusterer.min_cluster_size,
                "use_pca": self._clusterer.use_pca,
                "pca_n_components": self.config.pca_n_components,
                "min_samples": self._clusterer.min_samples,
                "cluster_scale": self._clusterer.cluster_scale,
            }

            # We submit config AND embeddings to the process
            logger.info(f"Submitting {len(embeddings)} samples to background process...")
            self._clustering_future = self._process_executor.submit(
                run_clustering_job, 
                config, 
                embeddings
            )
        
        self._last_cluster_time = time.time()


    def _display_frame(self, result: dict) -> None:
        """Display annotated frame with tracking visualization."""
        frame = result["frame"]
        tracks = result.get("tracks", [])
        detections = result.get("detections", [])

        # Get cluster colors
        cluster_colors = {}
        if self._clusterer.result:
            for cid, info in self._clusterer.result.clusters.items():
                cluster_colors[cid] = info.color

        # Draw on frame
        annotated = frame.image.copy()

        # Helper function to draw text with black outline
        def draw_text_with_outline(img, text, pos, scale, color, thickness=2):
            x, y = pos
            # Draw black outline
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
            # Draw colored text on top
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

        # Draw raw detections first (yellow boxes)
        # Only draw if we don't have active tracks to avoid double-box clutter
        # or if we are in warmup where tracks don't exist yet.
        if not tracks:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # Draw Warmup Overlay if needed
        if result.get("is_warmup", False):
            cv2.putText(annotated, "WARMUP PHASE - COLLECTING SAMPLES", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(annotated, f"Samples: {len(self._feature_buffer)}/{self.config.min_samples}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # Draw tracked objects on top (colored by cluster)
        if tracks:
            logger.debug(f"Drawing {len(tracks)} tracks")

        for track in tracks:
            # Skip unconfirmed tracks (filtering phantom objects)
            if track.hit_streak < self.config.min_hits:
                continue

            x1, y1, x2, y2 = track.bbox
            cid = track.stable_cluster_id
            color = cluster_colors.get(cid, (0, 255, 0))  # Default to green

            # Draw box with thicker line
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label with outline
            c_label = track.cluster_label
            if cid in self.config.cluster_labels:
                c_label = self.config.cluster_labels[cid]
            elif self._clusterer.result and cid in self._clusterer.result.clusters:
                c_label = self._clusterer.result.clusters[cid].label
            
            label = f"{c_label} #{track.track_id}"
            draw_text_with_outline(annotated, label, (x1, y1 - 10), 0.6, color)

            # Draw trajectory
            if len(track.positions) > 1:
                pts = np.array(track.positions[-20:], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, color, 2)

        # Add stats overlay with black outline
        if self._flow_analyzer:
            stats = self._flow_analyzer.get_flow_summary()
            y_offset = 30
            draw_text_with_outline(
                annotated,
                f"Active: {stats['active_tracks']} | Total: {stats['total_tracked']} | Detections: {len(detections)}",
                (10, y_offset),
                0.7,
                (255, 255, 255),
            )
            y_offset += 30

            for cluster_name, cluster_stats in stats.get("clusters", {}).items():
                text = f"{cluster_name}: {cluster_stats['active']} active, {cluster_stats['total']} total ({cluster_stats['flow_rate']})"
                draw_text_with_outline(annotated, text, (10, y_offset), 0.5, (255, 255, 255))
                y_offset += 22

        cv2.imshow("SOSD - Object Flow Tracker", annotated)
        
        # Draw and show cluster visualizer
        if self._cluster_visualizer:
            viz_img = self._cluster_visualizer.draw(self._clusterer.result)
            cv2.imshow("SOSD - Clusters", viz_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._stop_event.set()

    def _reassign_track_clusters(self) -> None:
        """Update cluster assignments for all active tracks using the new model.

        This is critical when re-clustering happens: old cluster IDs become invalid,
        so we must re-predict based on the track's embedded features.
        """
        if not self._clusterer._fitted:
            return

        # Use .tracks or .confirmed_tracks (Tracker doesn't have active_tracks property)
        # Using .tracks to cover everything including young candidates
        active_tracks = self._tracker.tracks
        if not active_tracks:
            return

        logger.info(f"Re-assigning clusters for {len(active_tracks)} tracks...")

        # Collect embeddings from tracks
        embeddings = []
        valid_tracks = []
        for track in active_tracks:
            # Track stores embeddings as list[np.ndarray], use mean_embedding property
            if track.embeddings:
                # Use the mean embedding for stable cluster assignment
                embeddings.append(track.mean_embedding)
                valid_tracks.append(track)

        if not embeddings:
            logger.info("No tracks with embeddings to reassign")
            return

        # Bulk predict
        embeddings_arr = np.vstack(embeddings)
        labels = self._clusterer.predict(embeddings_arr)

        # Update tracks
        for track, label in zip(valid_tracks, labels):
            track.cluster_id = int(label)
            # Update cluster history (Counter uses += or direct key increment)
            track.cluster_history[int(label)] += 1

        logger.info(f"Track re-assignment complete: {len(valid_tracks)} tracks updated")

    def _log_stats(self) -> None:
        """Log current statistics."""
        if self._flow_analyzer:
            stats = self._flow_analyzer.get_flow_summary()
            logger.info(
                f"[{stats['elapsed_time']:.0f}s] "
                f"Active: {stats['active_tracks']} | "
                f"Total: {stats['total_tracked']} | "
                f"Clusters: {stats['clusters']}"
            )

            # Publish stats to Pub/Sub
            if self._pubsub and self._flow_analyzer:
                self._pubsub.publish_stats(self._flow_analyzer.get_all_stats())

    def run(self) -> None:
        """Run the main pipeline loop."""
        try:
            # Initialize
            self._init_components()

            # Start video source
            if not self._video_source.start():
                logger.error("Failed to start video source")
                return

            self._start_time = time.time()

            # Warmup phase
            if not self._warmup_phase():
                logger.error("Warmup phase failed")
                return

            # Clustering phase
            # Clustering phase (Background)
            if len(self._feature_buffer) >= self.config.min_samples:
                logger.info("Starting processing... Clustering will happen in background process")
                self._perform_clustering_task()
            else:
                 logger.warning("Insufficient samples for clustering - generic tracking only")

            # Main processing loop
            self.state = PipelineState.RUNNING
            logger.info("Pipeline running - press 'q' to stop")

            while not self._stop_event.is_set():
                frame = self._video_source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Main Loop Logic
                current_time = time.time()
                tracks = []
                
                # 1. Advance Tracker (Prediction) - ALWAYS run this for smooth visualization
                # Note: predict_only() returns the tracks in their predicted state
                tracks = self._tracker.predict_only()
                
                # 2. Check if async processing finished
                if self._future and self._future.done():
                    try:
                        # Get results
                        p_frame, p_dets, p_feats, p_labels, p_names = self._future.result()
                        
                        # Update tracker with these (lagged) results
                        # This will "correct" the tracks
                        delay = frame.frame_number - p_frame.frame_number
                        self._update_tracker(p_frame, p_dets, p_feats, p_labels, p_names, delay)
                        
                        # Store for persistent display
                        self._last_detections = p_dets

                        # Get updated tracks after correction
                        tracks = self._tracker.tracks
                        self._future = None
                    except Exception as e:
                        logger.error(f"Detection task failed: {e}")
                        self._future = None

                # Check clustering completion
                if self._clustering_future and self._clustering_future.done():
                    try:
                        # Value returned is now a dict with state
                        ret = self._clustering_future.result()
                        
                        if "best_result" in ret:
                            # This was an Auto-Tune job
                            tuning_res = ret["best_result"]
                            logger.info(f"Auto-Tune Winner: PCA={tuning_res.pca_dims}, Scale={tuning_res.cluster_scale:.3f}")
                            logger.info(f"Score={tuning_res.score:.1f}, Clusters={tuning_res.n_clusters}, Noise={tuning_res.noise_ratio:.2f}")

                            # Update config for future re-clustering
                            self.config.pca_n_components = tuning_res.pca_dims
                            self.config.min_cluster_scale = tuning_res.cluster_scale

                            # Update Clusterer locally
                            self._clusterer.pca_n_components = tuning_res.pca_dims
                            self._clusterer.cluster_scale = tuning_res.cluster_scale

                            self._tuned = True

                            # USE THE WINNING RESULT DIRECTLY instead of re-clustering
                            # This avoids the issue where a new random sample gives different results
                            if tuning_res.cluster_result and tuning_res.n_clusters > 0:
                                logger.info(f"Using winning result directly: {tuning_res.n_clusters} clusters")
                                self._clusterer._last_result = tuning_res.cluster_result
                                self._clusterer._scaler = tuning_res.scaler
                                self._clusterer._pca_model = tuning_res.pca_model
                                self._clusterer._fitted = True

                                # Re-assign existing tracks to new clusters
                                self._reassign_track_clusters()
                            else:
                                logger.warning("Auto-tune found no valid clusters, triggering re-clustering...")
                                self._perform_clustering_task()
                            
                        else:
                            # Standard Clustering Result
                            cluster_result = ret["result"]
                            
                            logger.info("Background clustering process complete!")
                            
                            # Apply result AND MODELS to our local clusterer
                            # This enables predict() to work correctly
                            self._clusterer._last_result = cluster_result
                            self._clusterer._scaler = ret["scaler"]
                            self._clusterer._pca_model = ret["pca_model"]
                            # We don't sync hdbscan object to avoid issues, we use centroid prediction
                            
                            self._clusterer._fitted = True
                            
                            # Print generic info
                            if cluster_result:
                                 logger.info(f"Clustering finished: {cluster_result.n_clusters} clusters found.")
                                 
                            # CRITICAL: Re-assign existing tracks to new clusters
                            self._reassign_track_clusters()
                        
                    except Exception as e:
                        logger.error(f"Clustering/Tuning process failed: {e}")
                    self._clustering_future = None

                # 3. Submit new task if idle and interval elapsed
                if self._future is None and (current_time - self._last_process_time) >= self._process_interval:
                     # Submit copy of frame? Or frame itself (read-only safe?)
                     # Frame object holds numpy array. YOLO/CV2 might modify? 
                     # Usually safe if we don't modify in main thread.
                     self._future = self._executor.submit(self._run_detection, frame)
                     self._last_process_time = current_time

                # 4. Display
                if self.config.enable_display:
                    # We display the CURRENT frame with the TRACKS (which might be predicted or updated)
                    # Use last known detections for display to avoid empty boxes/flickering
                    display_result = {
                        "frame": frame,
                        "tracks": tracks,
                        "detections": self._last_detections
                    }
                    self._display_frame(display_result)


                # Periodic logging
                current_time = time.time()
                if current_time - self._last_log_time >= self.config.log_interval:
                    self._log_stats()
                    self._last_log_time = current_time

                # Periodic re-clustering
                if (
                    self.config.recluster_interval > 0
                    and current_time - self._last_cluster_time >= self.config.recluster_interval
                ):
                    # Only submit if not already running
                    if self._clustering_future is None:
                        logger.info("Triggering periodic re-clustering (background)...")
                        self._perform_clustering_task()
                        self._last_cluster_time = current_time

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the pipeline."""
        logger.info("Stopping pipeline...")
        self._stop_event.set()
        self.state = PipelineState.STOPPED

        if self._video_source:
            self._video_source.stop()

        if self._pubsub:
            self._pubsub.close()

        if self._bigquery:
            self._bigquery.close()

        if self._executor:
            self._executor.shutdown(wait=False)
            
        if self._process_executor:
            self._process_executor.shutdown(wait=False)

        if self.config.enable_display:
            cv2.destroyAllWindows()

        # Final stats
        if self._flow_analyzer:
            final_stats = self._flow_analyzer.get_flow_summary()
            console.print(Panel.fit(
                f"[bold]Final Statistics[/bold]\n\n"
                f"Total Runtime: {final_stats['elapsed_time']:.1f}s\n"
                f"Total Objects Tracked: {final_stats['total_tracked']}\n"
                f"Frames Processed: {self._frame_count}",
                title="SOSD Pipeline Complete",
            ))

        logger.info("Pipeline stopped")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SOSD - Semantic Object Stream Discovery")
    parser.add_argument("source", help="Video source (URL, file path, or webcam index)")
    parser.add_argument("--type", default="auto", choices=["auto", "rtsp", "hls", "youtube", "file", "webcam"])
    parser.add_argument("--warmup", type=int, default=60, help="Warmup duration in seconds")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters (None=auto)")
    parser.add_argument("--min-cluster-size", type=int, default=100, help="Min samples per cluster (higher=fewer clusters)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--detector", default="yolov8m.pt", help="YOLO model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--vehicles-only", action="store_true", help="Only detect vehicles")
    parser.add_argument("--processing-fps", type=int, default=-1, help="FPS limit for detection/processing")
    parser.add_argument("--display-fps", type=int, default=-1, help="Target FPS for display (default=-1 for auto/source FPS)")
    parser.add_argument("--pca-dims", type=int, default=32, help="Target dimensions for PCA reduction (default=32)")
    parser.add_argument("--min-cluster-scale", type=float, default=0.025, help="Scale factor for min cluster size (fraction of samples, default=0.025)")
    parser.add_argument("--auto-tune", action="store_true", help="Automatically tune clustering parameters during warmup")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples to collect during warmup (default=10000)")

    args = parser.parse_args()

    # Build config
    config = PipelineConfig(
        video_source=args.source,
        video_type=args.type,
        warmup_duration=args.warmup,
        n_clusters=args.clusters,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        enable_display=not args.no_display,
        detector_model=args.detector,
        detection_classes=[2, 3, 5, 7] if args.vehicles_only else None,  # COCO vehicle classes
        processing_fps=args.processing_fps,
        display_fps=args.display_fps,
        pca_n_components=args.pca_dims,
        min_cluster_scale=args.min_cluster_scale,
        auto_tune=args.auto_tune,
        max_buffer_size=args.max_samples,
    )

    # Setup signal handlers
    pipeline = Pipeline(config)

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    pipeline.run()


if __name__ == "__main__":
    main()
