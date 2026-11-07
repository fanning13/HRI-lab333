"""One-shot ArUco detection and the scan overlay.

Detection runs when the student presses Scan. Preview code must not
call these functions on every camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reachy_lego.block_library import ARUCO_DICTIONARY, BY_ID


@dataclass(frozen=True)
class MarkerHit:
    """One detected marker. Duplicate IDs are separate hits."""

    marker_id: int
    corners: np.ndarray  # (4, 2) float64, image pixels

    @property
    def center_x(self) -> float:
        return float(np.mean(self.corners[:, 0]))

    @property
    def center_y(self) -> float:
        return float(np.mean(self.corners[:, 1]))


def opencv_status() -> str:
    """Describe the single OpenCV build this process imported."""
    import cv2

    has_api = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "generateImageMarker")
    flag = "yes" if has_api else "NO"
    return f"OpenCV {cv2.__version__}; generateImageMarker={flag}"


def aruco_dictionary():
    """The dictionary named by the action registry."""
    import cv2

    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "generateImageMarker"):
        raise RuntimeError(
            "This OpenCV build cannot generate ArUco markers. Use one wheel: "
            "opencv-python 5 (ArUco included) or, on OpenCV 4.7–4.x, "
            "opencv-contrib-python instead of opencv-python. Do not install both."
        )
    if not hasattr(cv2.aruco, ARUCO_DICTIONARY):
        raise RuntimeError(f"cv2.aruco has no {ARUCO_DICTIONARY}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICTIONARY))


def order_hits(hits: list[MarkerHit]) -> list[MarkerHit]:
    """Order marker instances from top to bottom. Keep duplicate IDs."""
    return sorted(hits, key=lambda hit: (hit.center_y, hit.center_x, hit.marker_id))


def detect_markers(frame) -> list[MarkerHit]:
    """Detect ArUco markers in one BGR or gray frame, top to bottom."""
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []
    hits: list[MarkerHit] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2).copy()
        hits.append(MarkerHit(int(marker_id), points))
    return order_hits(hits)


def highlight_markers(frame, hits: list[MarkerHit]):
    """Blur outside each detected marker polygon and outline that polygon.

    ArUco finds the marker square, not the LEGO block around it. The
    overlay must not be described as a full-block segmentation.
    """
    import cv2

    if frame.ndim == 2:
        color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        color = frame.copy()
    if not hits:
        return color

    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    polygons = []
    for hit in hits:
        polygon = np.round(hit.corners).astype(np.int32).reshape(-1, 1, 2)
        polygons.append(polygon)
    cv2.fillPoly(mask, polygons, 255)

    short = min(color.shape[:2])
    kernel = min(51, short if short % 2 else short - 1)
    kernel = max(3, kernel)
    blurred = cv2.GaussianBlur(color, (kernel, kernel), 0)
    color[mask == 0] = blurred[mask == 0]
    cv2.polylines(color, polygons, True, (46, 180, 70), 3, cv2.LINE_AA)

    for hit in hits:
        action = BY_ID.get(hit.marker_id)
        label = action.label if action is not None else f"unknown {hit.marker_id}"
        x = int(np.min(hit.corners[:, 0]))
        y = max(18, int(np.min(hit.corners[:, 1])) - 8)
        cv2.putText(
            color, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 90, 40), 2, cv2.LINE_AA,
        )
    return color
