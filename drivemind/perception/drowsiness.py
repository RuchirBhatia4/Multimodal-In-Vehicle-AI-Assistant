"""Drowsiness alert rule, pre-registered and validated on held-out people.

Rule (frozen in eval/results/preregistration.json before the test ran):
  * a closure = MediaPipe eyeBlink score > 0.5 (mean of both eyes); re-opening for less
    than 0.1 s doesn't end it (matches the offline run-merging of <= 2 frames at 30 fps)
  * **drowsy** while >= 4 closures lasting >= 0.5 s ended within the last 60 s
  * **microsleep** while the current closure has lasted >= 1.0 s

Why closure *duration* rather than PERCLOS: drowsy eyes close slowly and stay closed
longer, while awake people at a desk (Eyeblink8) half-close their eyes when looking down,
which inflates PERCLOS. Results (eval/results/rldd_heldout_test.md):
  - development (6 people) + Eyeblink8 awake check: 0 false alerts/h
  - held-out test (6 new people): 0 false alerts/h on alert videos, 5/6 drowsy videos
    alerted. That *ties* the previous PERCLOS rule on detection, with fewer false alarms overall.

The rule is defined in seconds, so it works at any camera frame rate (the dashboard sends
~15 fps; the evaluation also held up with every video degraded to 12 fps).
"""

from __future__ import annotations

from collections import deque

SCORE_THRESHOLD = 0.5
MIN_CLOSURE_S = 0.5
K_PER_WINDOW = 4
WINDOW_S = 60.0
MICROSLEEP_S = 1.0
GAP_S = 0.1


class DrowsinessRule:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.closed_start: float | None = None  # first frame of the current closure
        self.last_closed: float | None = None  # last frame seen closed
        self.dt = 1 / 30
        self.prev_t: float | None = None
        self.long_closures: deque[float] = deque()  # end times of closures >= MIN_CLOSURE_S

    def _end_closure(self) -> None:
        dur = self.last_closed - self.closed_start + self.dt  # inclusive of the last frame
        if dur >= MIN_CLOSURE_S:
            self.long_closures.append(self.last_closed + self.dt)
        self.closed_start = self.last_closed = None

    def update(self, t: float, blink_score: float | None) -> dict:
        """Feed one frame (time in s, MediaPipe eyeBlink score or None if no face)."""
        if self.prev_t is not None and t > self.prev_t:
            self.dt = 0.8 * self.dt + 0.2 * min(t - self.prev_t, 0.2)  # smoothed frame interval
        self.prev_t = t
        closed = blink_score is not None and blink_score > SCORE_THRESHOLD
        if closed:
            if self.closed_start is None:
                self.closed_start = t
            self.last_closed = t
        elif self.closed_start is not None and t - self.last_closed >= GAP_S:
            self._end_closure()
        while self.long_closures and t - self.long_closures[0] > WINDOW_S:
            self.long_closures.popleft()
        closed_for = (self.last_closed - self.closed_start + self.dt) if self.closed_start is not None else 0.0
        return {
            "closed": closed,
            "closed_for": closed_for,
            "long_closures_60s": len(self.long_closures),
            "microsleep": closed_for >= MICROSLEEP_S,
            "drowsy": len(self.long_closures) >= K_PER_WINDOW,
        }
