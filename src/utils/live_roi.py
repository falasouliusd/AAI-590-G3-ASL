"""
Utilities for extracting signer ROIs from live webcam frames.

This module adapts the MediaPipe-based ROI pipeline from
03_roi_mediapipe_resume.ipynb so it can be reused in real-time
applications (e.g., Streamlit live demo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional dependency
    mp = None  # type: ignore


def _bbox_from_landmarks(landmarks: List[Tuple[float, float]], width: int, height: int) -> Optional[List[int]]:
    if not landmarks:
        return None
    xs = np.clip([x for x, _ in landmarks], 0, width - 1)
    ys = np.clip([y for _, y in landmarks], 0, height - 1)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expand_square(box: List[int], width: int, height: int, margin: float) -> List[int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = int(max(x2 - x1, y2 - y1) * (1.0 + margin))
    half = side // 2
    nx1, ny1 = max(0, int(cx - half)), max(0, int(cy - half))
    nx2, ny2 = min(width, nx1 + side), min(height, ny1 + side)
    side = min(nx2 - nx1, ny2 - ny1)
    return [nx1, ny1, nx1 + side, ny1 + side]


def _center_square(width: int, height: int) -> List[int]:
    side = min(width, height)
    cx = (width - side) // 2
    cy = (height - side) // 2
    return [cx, cy, cx + side, cy + side]


@dataclass
class LiveROICropper:
    """Keeps MediaPipe detectors alive across frames for smoother live ROI crops."""

    margin: float = 0.15
    smooth: float = 0.8

    def __post_init__(self):
        self._enabled = mp is not None
        self._current_box: Optional[List[float]] = None
        if not self._enabled:
            self._hands = None
            self._pose = None
            return

        mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
        mp_pose = mp.solutions.pose    # type: ignore[attr-defined]
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        for attr in ("_hands", "_pose"):
            obj = getattr(self, attr, None)
            try:
                if obj:
                    obj.close()
            except Exception:
                pass
        self._hands = None
        self._pose = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        self.close()

    def enabled(self) -> bool:
        return bool(self._enabled)

    def _detect_landmarks(self, frame_bgr: np.ndarray) -> List[Tuple[float, float]]:
        if not self._enabled or self._hands is None or self._pose is None:
            return []
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks: List[Tuple[float, float]] = []

        hands_res = self._hands.process(rgb)
        if hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:
                for pt in hand.landmark:
                    landmarks.append((pt.x * width, pt.y * height))

        pose_res = self._pose.process(rgb)
        if pose_res.pose_landmarks:
            for pt in pose_res.pose_landmarks.landmark:
                y = pt.y * height
                if y < 0.75 * height:  # focus on upper body to avoid lower-body drift
                    landmarks.append((pt.x * width, y))
        return landmarks

    def _update_box(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]
        landmarks = self._detect_landmarks(frame_bgr)
        new_box = _bbox_from_landmarks(landmarks, width, height)
        if new_box is None:
            # fallback to previous box or centered square
            if self._current_box is None:
                self._current_box = list(map(float, _center_square(width, height)))
            return

        expanded = _expand_square(new_box, width, height, margin=self.margin)
        if self._current_box is None:
            self._current_box = list(map(float, expanded))
        else:
            sm = self.smooth
            self._current_box = [
                sm * prev + (1.0 - sm) * curr
                for prev, curr in zip(self._current_box, expanded)
            ]

    def crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a signer-focused crop. If detectors are unavailable, original frame is returned."""
        if not self._enabled:
            return frame_bgr
        self._update_box(frame_bgr)
        if not self._current_box:
            return frame_bgr
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(val) for val in self._current_box]
        x1 = max(0, min(x1, width - 1))
        x2 = max(x1 + 1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(y1 + 1, min(y2, height))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return frame_bgr
        return crop
