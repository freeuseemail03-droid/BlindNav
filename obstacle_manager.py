"""
obstacle_manager.py
===========================================================================
Obstacle priority assessment, temporal stability and lightweight tracking.
===========================================================================
"""

from collections import defaultdict

import config
from detector import Detection


def _compute_priority_score(det, distance: float) -> float:
    """Return a numeric score where higher = more urgent."""
    conf = det.confidence
    label = det.label

    dist_score = max(0.05, 1.0 / max(distance, 0.1))
    type_weight = config.OBJECT_TYPE_WEIGHTS.get(label, config.DEFAULT_OBJECT_WEIGHT)
    conf_weight = max(conf, 0.1)

    return dist_score * type_weight * conf_weight


def _estimate_distance(det: Detection, focal_length: float = None) -> float:
    x1, y1, x2, y2 = det.box
    bbox_width_px = max(x2 - x1, 1)
    real_width = config.REFERENCE_OBJECT_WIDTHS.get(det.label, config.AVERAGE_REFERENCE_WIDTH)
    f = focal_length if focal_length and focal_length > 0 else config.FOCAL_LENGTH
    distance = (real_width * f) / bbox_width_px
    return max(round(distance, 1), 0.3)


def _priority_label(dist: float) -> str:
    if dist < config.CRITICAL_DISTANCE:
        return "CRITICAL"
    elif dist < config.HIGH_DISTANCE:
        return "HIGH"
    elif dist < config.MEDIUM_DISTANCE:
        return "MEDIUM"
    return "LOW"


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObstacleManager:
    """Tracks detections across frames, smooths positions, ranks by priority."""

    def __init__(self) -> None:
        self._position_history: dict[int, list[str]] = defaultdict(list)
        self._prev_tracks: list[tuple[Detection, int]] = []
        self._next_id = 0
        self._frame_width = 640
        self._focal_length = config.FOCAL_LENGTH

    def set_frame_width(self, width: int) -> None:
        self._frame_width = width

    def set_focal_length(self, focal: float) -> None:
        """Use a calibrated focal length (falls back silently when invalid)."""
        if focal and float(focal) > 0:
            self._focal_length = float(focal)

    def process_frame(self, detections: list[Detection]) -> list[dict]:
        """Return enriched, ranked obstacle dicts for this frame."""
        matched = self._match_detections(detections)
        enriched = []

        for det, track_id in matched:
            cx = det.center[0]
            ratio = cx / max(self._frame_width, 1)
            if ratio < config.LEFT_THRESHOLD:
                raw_position = "LEFT"
            elif ratio > config.RIGHT_THRESHOLD:
                raw_position = "RIGHT"
            else:
                raw_position = "CENTER"

            smoothed_position = self._smooth_position(track_id, raw_position)
            dist = _estimate_distance(det, self._focal_length)
            enriched.append({
                "label": det.label,
                "confidence": det.confidence,
                "bbox": det.box,
                "center": det.center,
                "distance": dist,
                "position": smoothed_position,
                "priority": _priority_label(dist),
                "priority_score": _compute_priority_score(det, dist),
                "track_id": track_id,
            })

        enriched.sort(key=lambda x: x["priority_score"], reverse=True)

        self._prev_tracks = matched
        return enriched

    def _match_detections(self, current: list[Detection]) -> list[tuple[Detection, int]]:
        if not self._prev_tracks:
            result = []
            for d in current:
                self._next_id += 1
                result.append((d, self._next_id))
            return result

        used_prev = set()
        matched = []

        for det in current:
            best_iou = 0.0
            best_idx = -1
            for j, (prev_det, prev_id) in enumerate(self._prev_tracks):
                if j in used_prev:
                    continue
                if prev_det.label != det.label:
                    continue
                iou_val = _iou(det.box, prev_det.box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j

            if best_iou > 0.3 and best_idx >= 0:
                used_prev.add(best_idx)
                track_id = self._prev_tracks[best_idx][1]
                matched.append((det, track_id))
            else:
                self._next_id += 1
                matched.append((det, self._next_id))

        return matched

    def _smooth_position(self, track_id: int, raw: str) -> str:
        history = self._position_history[track_id]
        history.append(raw)
        if len(history) > config.POSITION_SMOOTH_FRAMES * 2:
            history.pop(0)
        if len(history) < config.POSITION_SMOOTH_FRAMES:
            return raw
        recent = history[-config.POSITION_SMOOTH_FRAMES:]
        if len(set(recent)) == 1:
            return raw
        counts = defaultdict(int)
        for p in recent:
            counts[p] += 1
        return max(counts, key=counts.get)
