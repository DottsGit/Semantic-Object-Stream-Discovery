"""Video stream ingestion module supporting RTSP, HLS, YouTube, and local files."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import cv2
import numpy as np
from loguru import logger


class SourceType(Enum):
    """Video source types."""

    FILE = "file"
    RTSP = "rtsp"
    HLS = "hls"
    YOUTUBE = "youtube"
    WEBCAM = "webcam"


@dataclass
class Frame:
    """Container for a video frame with metadata."""

    image: np.ndarray
    timestamp: float
    frame_number: int
    source_fps: float
    resolution: tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.resolution = (self.image.shape[1], self.image.shape[0])


class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def open(self) -> bool:
        """Open the video source."""
        pass

    @abstractmethod
    def read(self) -> Frame | None:
        """Read a single frame."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video source."""
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Check if source is open."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Get source FPS."""
        pass


class OpenCVSource(VideoSource):
    """OpenCV-based video source for files, RTSP, and webcams."""

    def __init__(
        self,
        source: str | int,
        target_resolution: tuple[int, int] | None = None,
        fps_limit: int | None = None,
    ):
        self.source = source
        self.target_resolution = target_resolution
        self.fps_limit = fps_limit
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0
        self._start_time: float | None = None
        self._last_frame_time = 0.0
        self._min_frame_interval = 0.0

    def open(self) -> bool:
        """Open the video capture."""
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source)
        else:
            # Use appropriate backend for RTSP
            if str(self.source).startswith("rtsp://"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        # Calculate frame interval
        if self.fps_limit and self.fps_limit > 0:
            self._min_frame_interval = 1.0 / self.fps_limit
        else:
            # Use native FPS
            native_fps = self.fps
            if native_fps > 0:
                self._min_frame_interval = 1.0 / native_fps

        self._start_time = time.perf_counter()
        logger.info(f"Opened video source: {self.source} @ {self.fps:.1f} FPS (Interval: {self._min_frame_interval:.3f}s)")
        return True

    def read(self) -> Frame | None:
        """Read and optionally resize a frame."""
        if self._cap is None or not self._cap.isOpened():
            return None

        # Rate limiting for livestreams
        # Rate limiting for livestreams
        if self._min_frame_interval > 0:
            target_time = self._last_frame_time + self._min_frame_interval
            current_time = time.perf_counter()
            remaining = target_time - current_time
            
            if remaining > 0:
                # Sleep for most of the time
                if remaining > 0.002:
                    time.sleep(remaining - 0.002)
                
                # Busy wait for the last bit for precision
                while time.perf_counter() < target_time:
                    pass

        ret, frame = self._cap.read()
        if not ret:
            return None

        self._last_frame_time = time.perf_counter()
        self._frame_count += 1

        # Resize if needed
        if self.target_resolution:
            frame = cv2.resize(frame, self.target_resolution)

        return Frame(
            image=frame,
            timestamp=time.time() - (self._start_time or time.time()),
            frame_number=self._frame_count,
            source_fps=self.fps,
        )

    def close(self) -> None:
        """Release the video capture."""
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"Closed video source after {self._frame_count} frames")

    def is_open(self) -> bool:
        """Check if capture is open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get source FPS."""
        if self._cap:
            # Set buffer size (frames)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)

            video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            return video_fps
        return 30.0


class YouTubeSource(VideoSource):
    """YouTube stream source using yt-dlp (with streamlink fallback)."""

    def __init__(
        self,
        url: str,
        quality: str = "best",
        target_resolution: tuple[int, int] | None = None,
        fps_limit: int | None = None,
    ):
        self.url = url
        self.quality = quality
        self.target_resolution = target_resolution
        self.fps_limit = fps_limit
        self._opencv_source: OpenCVSource | None = None

    def _try_ytdlp(self) -> str | None:
        """Try to get stream URL using yt-dlp."""
        try:
            import yt_dlp

            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                if info and "url" in info:
                    return info["url"]
                # For playlists or live streams, get the first entry
                if info and "entries" in info and info["entries"]:
                    return info["entries"][0].get("url")
                # Try formats directly
                if info and "formats" in info:
                    for fmt in reversed(info["formats"]):
                        if fmt.get("url"):
                            return fmt["url"]
            return None
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"yt-dlp failed: {e}")
            return None

    def _try_streamlink(self) -> str | None:
        """Try to get stream URL using streamlink (fallback)."""
        try:
            import streamlink

            streams = streamlink.streams(self.url)
            if not streams:
                return None

            stream = streams.get(self.quality) or streams.get("best")
            if stream:
                return stream.url
            return None
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"streamlink failed: {e}")
            return None

    def open(self) -> bool:
        """Open YouTube stream via yt-dlp or streamlink."""
        # Try yt-dlp first (handles more videos including protected ones)
        stream_url = self._try_ytdlp()

        # Fallback to streamlink
        if not stream_url:
            logger.info("yt-dlp failed, trying streamlink...")
            stream_url = self._try_streamlink()

        if not stream_url:
            logger.error(
                f"Failed to open YouTube stream: {self.url}\n"
                "Install yt-dlp: pip install yt-dlp"
            )
            return False

        logger.info(f"Resolved YouTube stream URL: {stream_url[:80]}...")

        self._opencv_source = OpenCVSource(
            stream_url, self.target_resolution, self.fps_limit
        )
        return self._opencv_source.open()

    def read(self) -> Frame | None:
        """Read a frame from the stream."""
        if self._opencv_source:
            return self._opencv_source.read()
        return None

    def close(self) -> None:
        """Close the stream."""
        if self._opencv_source:
            self._opencv_source.close()

    def is_open(self) -> bool:
        """Check if stream is open."""
        return self._opencv_source is not None and self._opencv_source.is_open()

    @property
    def fps(self) -> float:
        """Get stream FPS."""
        if self._opencv_source:
            return self._opencv_source.fps
        return 30.0


class BufferedVideoSource:
    """Threaded video source with frame buffer for smooth streaming."""

    def __init__(self, source: VideoSource, buffer_size: int = 30):
        self.source = source
        self.buffer_size = buffer_size
        self._buffer: deque[Frame] = deque(maxlen=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the buffered reader."""
        if not self.source.open():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info("Started buffered video reader")
        return True

    def _reader_loop(self):
        """Background thread for continuous reading."""
        while self._running:
            if len(self._buffer) < self.buffer_size:
                frame = self.source.read()
                if frame is None:
                    # Stream ended or error
                    if not self.source.is_open():
                        logger.warning("Video source closed, stopping reader")
                        break
                    time.sleep(0.001)
                    continue

                with self._lock:
                    self._buffer.append(frame)
            else:
                time.sleep(0.001)  # Buffer full, wait

    def read(self) -> Frame | None:
        """Get the next frame from the buffer."""
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def stop(self):
        """Stop the reader."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.source.close()
        logger.info("Stopped buffered video reader")

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames."""
        while self._running or self._buffer:
            frame = self.read()
            if frame:
                yield frame
            else:
                time.sleep(0.01)


def detect_source_type(source: str) -> SourceType:
    """Auto-detect video source type from URL or path."""
    source_lower = source.lower()

    if source.isdigit():
        return SourceType.WEBCAM
    elif source_lower.startswith("rtsp://"):
        return SourceType.RTSP
    elif "youtube.com" in source_lower or "youtu.be" in source_lower:
        return SourceType.YOUTUBE
    elif source_lower.endswith((".m3u8", ".m3u")):
        return SourceType.HLS
    elif any(source_lower.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return SourceType.FILE
    else:
        # Default to file
        return SourceType.FILE


def create_video_source(
    source: str,
    source_type: str = "auto",
    target_resolution: tuple[int, int] | None = None,
    fps_limit: int | None = None,
) -> VideoSource:
    """Factory function to create the appropriate video source.

    Args:
        source: Video source URL, path, or webcam index
        source_type: Explicit type or "auto" for detection
        target_resolution: Optional (width, height) to resize frames
        fps_limit: Optional max FPS for processing

    Returns:
        Configured VideoSource instance
    """
    if source_type == "auto":
        detected_type = detect_source_type(source)
    else:
        detected_type = SourceType(source_type)

    logger.info(f"Creating {detected_type.value} source for: {source}")

    if detected_type == SourceType.YOUTUBE:
        return YouTubeSource(source, "best", target_resolution, fps_limit)
    elif detected_type == SourceType.WEBCAM:
        return OpenCVSource(int(source), target_resolution, fps_limit)
    else:
        # RTSP, HLS, and files all work with OpenCV
        return OpenCVSource(source, target_resolution, fps_limit)
