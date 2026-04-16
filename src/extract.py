import cv2
import os
import math
from typing import List, Dict

def extract_frames(video_path: str, output_dir: str, interval_seconds: int = 5) -> List[Dict]:
    """Extracts frames from a video at a specified interval."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = math.floor(fps * interval_seconds)
    
    count = 0
    extracted_count = 0
    frame_metadata = []

    print(f"Extracting frames from {video_path} every {interval_seconds} seconds...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            timestamp = count / fps
            mins, secs = int(timestamp // 60), int(timestamp % 60)
            
            frame_filename = f"frame_{mins:02d}m_{secs:02d}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            
            frame_metadata.append({
                "frame_path": frame_path,
                "timestamp_seconds": timestamp,
                "timestamp_formatted": f"{mins:02d}:{secs:02d}"
            })
            extracted_count += 1

        count += 1

    cap.release()
    print(f"Done! Extracted {extracted_count} frames.")
    return frame_metadata