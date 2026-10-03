"""Learned eye-closure detector (streaming), trained on Eyeblink8.

Why this exists: evaluated on Eyeblink8 (eval/results/eyeblink8.md), the original
calibrated-EAR threshold detected blinks reasonably (F1 0.91) but mislabelled many
half-closed / looking-down frames as "closed". That made PERCLOS 4.5 points too high and
caused ~33 false microsleep alerts per hour on people who were wide awake. A gradient-boosted
classifier over a 13-frame window of eye features cut PERCLOS error to ~0.6 points and false
microsleep alerts to zero (leave-one-person-out).

Concepts:
* **Temporal context window.** A blink is a ~100-400 ms *shape* in the EAR signal (dip and
  recovery), so classifying frame t from frames t-6..t+6 is far more reliable than a single
  frame. The cost is latency: we must wait 6 frames (~200 ms) for the "future" half.
  That's fine for PERCLOS and for 1 s microsleep alerts.
* **Feature fusion.** Geometric EAR (hand-designed) + MediaPipe's eyeBlink blendshape
  (a learned score) + head pitch (looking down shrinks EAR without closing the eye).
* **Person-independent normalization.** EAR is divided by the driver's own open-eye
  baseline (start-of-drive calibration) and by a rolling 10 s baseline (adapts to posture
  and lighting changes).

The feature math lives in `window_features()`, which the offline evaluation also uses, so
training and serving cannot silently diverge (training/serving skew).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

HALF = 6  # 13-frame window
ADAPT_N = 300  # rolling baseline length (~10 s at 30 fps)
RAW = ["ear", "ear_l", "ear_r", "blink", "squint", "look_down", "pitch", "yaw"]


def window_features(rows: np.ndarray, adapt: np.ndarray, base: float) -> np.ndarray:
    """Feature vector for the centre row of a (13, len(RAW)) window.

    rows  : raw values per frame (columns = RAW), oldest first
    adapt : rolling 85th-percentile EAR baseline for each of the 13 frames
    base  : the driver's calibrated open-eye EAR
    """
    c = HALF
    ear = rows[:, 0]
    ear_n = ear / base
    ear_a = ear / adapt
    blink = rows[:, 3]
    return np.concatenate(
        [
            ear_n, ear_a, blink,
            [rows[c, 1] / base, rows[c, 2] / base, rows[c, 4], rows[c, 5], rows[c, 6], abs(rows[c, 7])],
            [ear_n[c] - ear_n[c - 3], ear_n[c + 3] - ear_n[c]],
        ]
    )


class EyeStateModel:
    """Streaming wrapper: push one frame's raw values, get the decision for the frame
    HALF frames ago (or None while the window fills)."""

    def __init__(self, path: Path) -> None:
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        assert bundle["half_window"] == HALF
        self.reset()

    def reset(self) -> None:
        self.rows: deque[np.ndarray] = deque(maxlen=2 * HALF + 1)
        self.ear_hist: deque[float] = deque(maxlen=ADAPT_N + 2 * HALF)
        self.adapt: deque[float] = deque(maxlen=2 * HALF + 1)
        self.faces: deque[bool] = deque(maxlen=2 * HALF + 1)
        self.last: np.ndarray | None = None

    def push(self, raw: dict | None, base: float | None) -> tuple[bool, float] | None:
        """raw: dict of RAW values for this frame, or None if no face (values are
        forward-filled, matching the offline pipeline). Returns (closed, probability) for
        the frame HALF frames ago."""
        if raw is None:
            if self.last is None:
                return None
            row, has_face = self.last, False
        else:
            row = np.array([raw[k] for k in RAW], dtype=np.float64)
            # Forward-fill any missing value (e.g. head pose solve failed).
            if self.last is not None:
                row = np.where(np.isnan(row), self.last, row)
            has_face = True
        if not self.ear_hist:  # pad history with the first value, like the offline pipeline
            self.ear_hist.extend([row[0]] * (ADAPT_N - 1))
        self.last = row
        self.ear_hist.append(row[0])
        self.adapt.append(float(np.percentile(list(self.ear_hist)[-ADAPT_N:], 85)))
        self.rows.append(row)
        self.faces.append(has_face)
        if len(self.rows) < 2 * HALF + 1 or base is None:  # window filling / still calibrating
            return None
        if not self.faces[HALF]:
            return False, 0.0
        x = window_features(np.stack(self.rows), np.array(self.adapt), base)
        p = float(self.model.predict_proba(x[None, :])[0, 1])
        return p >= self.threshold, p
