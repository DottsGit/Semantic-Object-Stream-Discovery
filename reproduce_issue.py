
from src.tracking.tracker import ObjectTracker
from src.detection.detector import Detection
import numpy as np

def test_tracker_min_hits():
    # Initialize tracker with min_hits=1
    tracker = ObjectTracker(min_hits=1)
    
    # Create a dummy detection
    det = Detection(
        bbox=(0, 0, 100, 100),
        confidence=0.9,
        class_id=1,
        class_name="car",
        frame_number=0,
        timestamp=0.0
    )
    
    print(f"Tracker initialized with min_hits={tracker.min_hits}")
    
    # Frame 1
    tracks = tracker.update([det])
    print(f"Frame 1: {len(tracks)} tracks (Expected: 1)")
    if tracks:
        print(f"  Track streak: {tracks[0].hit_streak}")

    # Frame 2
    tracks = tracker.update([det])
    print(f"Frame 2: {len(tracks)} tracks (Expected: 1)")
    if tracks:
        print(f"  Track streak: {tracks[0].hit_streak}")
    else:
        print("  NO TRACKS RETURNED - Bug confirmed")

    # Frame 3
    tracks = tracker.update([det])
    print(f"Frame 3: {len(tracks)} tracks (Expected: 1)")
    if tracks:
        print(f"  Track streak: {tracks[0].hit_streak}")

if __name__ == "__main__":
    test_tracker_min_hits()
