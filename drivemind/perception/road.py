"""Road perception: detection -> multi-object tracking -> time-to-collision.

ML concepts (see docs/CONCEPTS.md for the long version):

* **Single-stage object detection (YOLO11).** One forward pass predicts boxes + classes
  for the whole image, which is why it runs in real time.
* **Multi-object tracking (ByteTrack).** Detections are per-frame and have no identity.
  ByteTrack links them across frames with a Kalman-filter motion model + IoU matching,
  and (its trick) also matches *low-confidence* boxes so objects aren't lost when they
  are briefly occluded or blurred. Stable IDs are what make per-object reasoning possible.
* **Monocular time-to-collision (TTC) from "looming".** With one camera we don't know
  distance, but we don't need it. If an object's image size is s and it is approaching at
  constant speed, TTC = s / (ds/dt) = 1 / (d ln s / dt). We fit a line to ln(s) over a
  short window (least squares is robust to per-frame jitter). Biological vision uses the
  same cue (the "tau" variable).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from drivemind.config import settings

# COCO class ids we care about on the road.
CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}
COLLIDABLE = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def classify_traffic_light(crop_bgr: np.ndarray) -> str | None:
    """Classic CV: count saturated, bright pixels in red / yellow / green hue bands.
    Not everything needs a neural net. A cheap, explainable heuristic on a crop that a
    neural net already localized is a common production pattern."""
    if crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] > 90) & (hsv[..., 2] > 140)
    h = hsv[..., 0]
    counts = {
        "red": int(np.count_nonzero(bright & ((h < 10) | (h > 165)))),
        "yellow": int(np.count_nonzero(bright & (h >= 15) & (h <= 35))),
        "green": int(np.count_nonzero(bright & (h >= 45) & (h <= 95))),
    }
    color, n = max(counts.items(), key=lambda kv: kv[1])
    return color if n >= max(6, 0.02 * crop_bgr.shape[0] * crop_bgr.shape[1]) else None


@dataclass
class _TrackHistory:
    samples: deque  # (t, ln(scale))
    last_seen: float


class RoadPerception:
    def __init__(self) -> None:
        from ultralytics import YOLO

        self.device = _pick_device()
        self.model = YOLO(settings.yolo_model)
        self.histories: dict[int, _TrackHistory] = {}
        self.ttc_window_s = 0.8

    def reset(self) -> None:
        """New video source -> tracker state is meaningless; start over."""
        from ultralytics import YOLO

        self.model = YOLO(settings.yolo_model)
        self.histories.clear()

    def _ttc(self, tid: int, t: float, scale: float) -> float | None:
        hist = self.histories.setdefault(tid, _TrackHistory(deque(maxlen=40), t))
        hist.samples.append((t, math.log(max(scale, 1e-6))))
        hist.last_seen = t
        pts = [(ti, si) for ti, si in hist.samples if t - ti <= self.ttc_window_s]
        if len(pts) < 5 or pts[-1][0] - pts[0][0] < 0.3:
            return None
        ts = np.array([p[0] for p in pts])
        ss = np.array([p[1] for p in pts])
        slope = np.polyfit(ts - ts[0], ss, 1)[0]  # d ln(s) / dt
        if slope <= 0.02:  # not approaching (or noise)
            return None
        # A least-squares slope over a window estimates TTC at the window's *centroid*,
        # which is in the past. Under constant closing speed TTC falls 1 s per second,
        # so shift the estimate forward to "now". (Without this it is ~0.4 s optimistic.)
        ttc = float(1.0 / slope) - float(t - ts.mean())
        if ttc <= 0:
            return 0.05
        return ttc if ttc < 12.0 else None  # beyond ~12 s the estimate is noise, not a threat

    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> dict:
        t = time.monotonic() if t is None else t
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz,
            classes=list(CLASSES),
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        tracks: list[dict] = []
        min_ttc: float | None = None
        lead_id: int | None = None

        if boxes is not None and len(boxes):
            # .tolist() -> plain Python floats/ints (numpy scalars aren't JSON-serializable)
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            cls = boxes.cls.cpu().numpy().astype(int).tolist()
            conf = boxes.conf.cpu().numpy().tolist()
            ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(cls)
            for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
                label = CLASSES.get(int(c), str(c))
                track = {
                    "id": None if tid is None else int(tid),
                    "label": label,
                    "conf": round(float(p), 2),
                    "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                }
                if label == "traffic light":
                    crop = frame_bgr[int(y1) : int(y2), int(x1) : int(x2)]
                    track["state"] = classify_traffic_light(crop)
                if tid is not None and label in COLLIDABLE:
                    scale = math.sqrt(max((x2 - x1) * (y2 - y1), 1.0))
                    ttc = self._ttc(int(tid), t, scale)
                    cx = (x1 + x2) / 2 / w
                    # "Ego corridor": only objects roughly ahead of us can be collision threats.
                    in_path = bool(0.3 < cx < 0.7 and y2 / h > 0.45)
                    track["in_path"] = in_path
                    if ttc is not None:
                        track["ttc"] = round(ttc, 2)
                        if in_path and (min_ttc is None or ttc < min_ttc):
                            min_ttc, lead_id = ttc, int(tid)
                tracks.append(track)

        # Forget tracks we haven't seen for a while (bounded memory).
        for tid in [k for k, v in self.histories.items() if t - v.last_seen > 2.0]:
            del self.histories[tid]

        level = "none"
        if min_ttc is not None:
            if min_ttc < settings.ttc_critical_s:
                level = "critical"
            elif min_ttc < settings.ttc_warn_s:
                level = "warning"

        return {
            "tracks": tracks,
            "fcw": level,
            "min_ttc": None if min_ttc is None else round(min_ttc, 2),
            "lead_id": lead_id,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def summarize_road(result: dict | None) -> str:
    """Compact, LLM-friendly text summary of the current road state."""
    if not result or not result.get("tracks"):
        return "No objects detected on the road."
    counts: dict[str, int] = {}
    extras: list[str] = []
    for tr in result["tracks"]:
        counts[tr["label"]] = counts.get(tr["label"], 0) + 1
        if tr["label"] == "traffic light" and tr.get("state"):
            extras.append(f"a {tr['state']} traffic light")
    parts = [f"{n} {lbl}{'s' if n > 1 else ''}" for lbl, n in counts.items()]
    s = "Detected: " + ", ".join(parts) + "."
    if extras:
        s += " Including " + ", ".join(extras) + "."
    if result.get("min_ttc") is not None:
        s += f" Closest object ahead: time-to-collision ~{result['min_ttc']}s ({result['fcw']})."
    return s
