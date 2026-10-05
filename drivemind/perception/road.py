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
* **Which objects are in our path? (also monocular).** A fixed "corridor" box in the image
  fails on real streets: perspective puts parked cars at the curb inside it, and they loom
  as you drive past, so the first version fired "Brake!" for up to 31 of 40 s of ordinary
  driving (BDD100K clips). Instead use an invariant: an object at lateral offset X metres
  and width W appears at image offset f*X/Z with width f*W/Z, so offset/width = X/W does
  not depend on distance. Cars in our lane stay below ~0.7 widths off our heading; parked
  cars sit several widths out for their whole approach. Tiny boxes are ignored (a pixel of
  jitter is a big fraction of their size), and the warning must hold for 3 frames in a row.
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
MAX_OFFSET_WIDTHS = 0.7  # |object centre - image centre| / object width, see is_in_path()
MIN_WIDTH_FRAC = 0.04  # boxes narrower than 4% of the frame are too small for looming TTC
PERSIST_FRAMES = 3  # consecutive below-threshold estimates before warning
MAX_FRAME_JUMP = 0.25  # max |change in ln(size)| between consecutive frames before resetting a track


def is_in_path(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> bool:
    """Is this object roughly in our lane? Uses offset/width = X/W, which stays constant as we
    approach a stationary object, unlike a fixed region of the image."""
    bw = x2 - x1
    if bw < MIN_WIDTH_FRAC * w or y2 / h < 0.45:  # too small to judge, or above the horizon band
        return False
    return abs((x1 + x2) / 2 - w / 2) / bw < MAX_OFFSET_WIDTHS


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
        self.below: dict[int, int] = {}  # consecutive below-warning-threshold TTCs per track
        self.aspect: dict[int, float] = {}  # height/width while fully visible, per track
        self.ttc_window_s = 0.8

    def reset(self) -> None:
        """New video source -> tracker state is meaningless; start over."""
        from ultralytics import YOLO

        self.model = YOLO(settings.yolo_model)
        self.histories.clear()
        self.below.clear()
        self.aspect.clear()

    def _ttc(self, tid: int, t: float, scale: float) -> float | None:
        hist = self.histories.setdefault(tid, _TrackHistory(deque(maxlen=40), t))
        ln_s = math.log(max(scale, 1e-6))
        # Outlier guard: a real approaching object can't change size by >28% between frames
        # ~67-200 ms apart (that would be TTC < ~0.25 s). Jumps like that come from the
        # tracker swapping identities (scene cut, occlusion) or a box suddenly
        # expanding when a partly hidden car emerges, so start the history over.
        if hist.samples and t - hist.samples[-1][0] < 0.2 and abs(ln_s - hist.samples[-1][1]) > MAX_FRAME_JUMP:
            hist.samples.clear()
        hist.samples.append((t, ln_s))
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

    def _scale(self, tid: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> float | None:
        """Object size for looming: sqrt(width * height), the least noisy measure. A box cut
        off by the frame edge stops growing in that direction (tall vehicles close ahead hit
        the top edge), which made TTC drift *up* as they got closer. So remember each track's
        height/width ratio while it is fully visible, and fill in the clipped side from it:
        the ratio doesn't change as the object approaches, so the measure stays continuous."""
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        clip_tb, clip_lr = y1 <= 1 or y2 >= h - 1, x1 <= 1 or x2 >= w - 1
        if not clip_tb and not clip_lr:
            prev = self.aspect.get(tid)
            self.aspect[tid] = bh / bw if prev is None else 0.8 * prev + 0.2 * bh / bw
            return math.sqrt(bw * bh)
        ratio = self.aspect.get(tid)
        if ratio is None or (clip_tb and clip_lr):
            return None  # no fully-visible reference yet, or clipped on both axes
        return bw * math.sqrt(ratio) if clip_tb else bh / math.sqrt(ratio)

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
                    scale = self._scale(int(tid), x1, y1, x2, y2, w, h)
                    ttc = None if scale is None else self._ttc(int(tid), t, scale)
                    in_path = is_in_path(x1, y1, x2, y2, w, h)
                    track["in_path"] = in_path
                    threat = in_path and ttc is not None and ttc < settings.ttc_warn_s
                    self.below[int(tid)] = self.below.get(int(tid), 0) + 1 if threat else 0
                    if ttc is not None:
                        track["ttc"] = round(ttc, 2)
                        if self.below[int(tid)] >= PERSIST_FRAMES and (min_ttc is None or ttc < min_ttc):
                            min_ttc, lead_id = ttc, int(tid)
                tracks.append(track)

        # Forget tracks we haven't seen for a while (bounded memory).
        for tid in [k for k, v in self.histories.items() if t - v.last_seen > 2.0]:
            del self.histories[tid]
            self.below.pop(tid, None)
            self.aspect.pop(tid, None)

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
