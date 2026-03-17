import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from core.model import PersonData

def write_image(
    frame: np.ndarray,
    output_path: str,
) -> bool:
    try:
        if frame is None or frame.size == 0:
            return False
            
        return cv2.imwrite(output_path, frame)
    except Exception as e:
        print(f"Error writing image to {output_path}: {e}")
        return False

def draw_bbox_with_label(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

def draw_person_data_in_bbox(
    frame: np.ndarray,
    track_id: int,
    person: PersonData,
    color: Tuple[int, int, int] = (0, 0, 255),
    font_scale: float = 0.5,
    line_thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = person.head_bbox
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    y_offset = 20
    line_height = 20
    
    timestamp_str = person.timestamp.strftime("%Y-%m-%d %H:%M:%S") if person.timestamp else "None"
    
    last_looking_str = person.last_write_times_looking.strftime("%H:%M:%S") if person.last_write_times_looking else "None"
    last_not_looking_str = person.last_write_times_not_looking.strftime("%H:%M:%S") if person.last_write_times_not_looking else "None"
    
    texts = [
        f"ID: {track_id}",
        f"Timestamp: {timestamp_str}",
        f"Dwell Looking: {person.dwelltime_looking}s",
        f"Dwell Not Looking: {person.dwelltime_not_looking}s",
        f"Last Looking: {last_looking_str}",
        f"Last Not Looking: {last_not_looking_str}",
    ]
    
    for i, text in enumerate(texts):
        y_pos = y1 + y_offset + (i * line_height)
        cv2.putText(
            frame,
            text,
            (x1 + 5, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_thickness,
            cv2.LINE_AA,
        )

def draw_person_text_overlay(
    frame: np.ndarray,
    track_id: int,
    person: PersonData,
    color: Tuple[int, int, int] = (0, 0, 255),
    font_scale: float = 0.5,
    line_thickness: int = 2,
) -> None:
    """Draw text overlay on cropped head frame (coordinates relative to cropped frame)"""
    y_offset = 20
    line_height = 20
    x_start = 5
    
    timestamp_str = person.timestamp.strftime("%Y-%m-%d %H:%M:%S") if person.timestamp else "None"
    
    last_looking_str = person.last_write_times_looking.strftime("%H:%M:%S") if person.last_write_times_looking else "None"
    last_not_looking_str = person.last_write_times_not_looking.strftime("%H:%M:%S") if person.last_write_times_not_looking else "None"
    
    texts = [
        f"ID: {track_id}",
        f"Timestamp: {timestamp_str}",
        f"Dwell Looking: {person.dwelltime_looking}s",
        f"Dwell Not Looking: {person.dwelltime_not_looking}s",
        f"Last Looking: {last_looking_str}",
        f"Last Not Looking: {last_not_looking_str}",
    ]
    
    for i, text in enumerate(texts):
        y_pos = y_offset + (i * line_height)
        cv2.putText(
            frame,
            text,
            (x_start, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_thickness,
            cv2.LINE_AA,
        )

def draw_person_overlay(frame: np.ndarray, person_cropped_frame: np.ndarray, track_id: int, person: PersonData) -> None:
    if person.person_bbox is not None:
        px1, py1, px2, py2 = person.person_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 3)
    
    if person.face_bbox is not None:
        fx1, fy1, fx2, fy2 = person.face_bbox
        cv2.rectangle(person_cropped_frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)    
        
    draw_person_text_overlay(person_cropped_frame, track_id, person)
