"""Driver Monitoring System (DMS): facial landmarks -> drowsiness & distraction.

ML / CV concepts:

* **Dense facial landmarks (MediaPipe Face Landmarker).** A lightweight CNN regresses
  478 3-D points on the face at >30 FPS on a CPU. Everything below is geometry on top.
* **Eye Aspect Ratio (EAR)** (Soukupová & Čech, 2016): ratio of the eye's vertical
  openings to its width. It is roughly constant while open and drops to ~0 on a blink.
  Scale-invariant, so it doesn't matter how far the driver sits from the camera.
* **Per-driver calibration.** Eye shapes differ, so a fixed EAR threshold is wrong for
  many people. We learn each driver's *open-eye baseline* in the first seconds and set the
  threshold relative to it (a simple form of personalization / domain adaptation).
* **PERCLOS** (PERcentage of eyelid CLOSure over time): the fatigue metric used in
  NHTSA research and Euro NCAP-style DMS. Blinks are normal; a high fraction of
  closed-eye time over a sliding window is fatigue.
* **Mouth Aspect Ratio (MAR)** for yawns, with a minimum duration so talking isn't a yawn.
* **Head pose via Perspective-n-Point (PnP).** Given 2-D landmark pixels and a generic
  3-D face model, `cv2.solvePnP` recovers the rotation (yaw/pitch/roll) of the head.
  This is the same math used for AR and camera calibration.
* **Hysteresis state machine.** Enter "drowsy" at a high threshold, leave at a lower one,
  so the alert doesn't flicker on and off at the boundary. Alert fatigue is a real reason
  drivers disable safety systems.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from drivemind.config import ROOT, settings
from drivemind.perception.eye_state import EyeStateModel

# MediaPipe face-mesh landmark indices.
LEFT_EYE = [362, 385, 387, 263, 373, 380]  # p1..p6 in the EAR paper's ordering
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = {"left": 61, "right": 291, "top": [81, 13, 311], "bottom": [178, 14, 402]}
POSE_IDS = [1, 152, 263, 33, 291, 61]  # nose tip, chin, eye corners, mouth corners
# Generic 3-D face model (mm) matching POSE_IDS.
POSE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (43.3, 32.7, -26.0),
        (-43.3, 32.7, -26.0),
        (28.9, -28.9, -24.1),
        (-28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


def _ear(pts: np.ndarray, idx: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return float(
        (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)
    )


def _mar(pts: np.ndarray) -> float:
    vertical = sum(np.linalg.norm(pts[t] - pts[b]) for t, b in zip(MOUTH["top"], MOUTH["bottom"]))
    width = np.linalg.norm(pts[MOUTH["left"]] - pts[MOUTH["right"]])
    return float(vertical / (3.0 * width + 1e-6))


def _head_pose(pts: np.ndarray, w: int, h: int) -> tuple[float, float, float] | None:
    image_pts = np.array([pts[i] for i in POSE_IDS], dtype=np.float64)
    focal = float(w)  # pinhole approximation: focal length ~ image width
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_MODEL, image_pts, cam, np.zeros(4), flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix into Euler angles (degrees).
    sy = np.hypot(rot[0, 0], rot[1, 0])
    pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
    roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    # Fold pitch into [-90, 90] (solvePnP's frame is flipped relative to the camera's).
    pitch = pitch - 180 if pitch > 90 else pitch + 180 if pitch < -90 else pitch
    return float(yaw), float(pitch), float(roll)


class DriverMonitor:
    def __init__(self) -> None:
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.face_model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts_ms = 0
        # Eye-closure decision, best available first (see eval/results/eyeblink8.md):
        #   1. gradient-boosted model trained on Eyeblink8 (blink F1 0.95, 0 false microsleeps/h)
        #   2. MediaPipe's own eyeBlink blendshape > 0.5 (no training needed; PERCLOS-accurate)
        # The original calibrated-EAR threshold is no longer used for alerts: it produced
        # ~33 false microsleep alerts per hour on wide-awake people.
        model_path = ROOT / "models" / "eye_state_gbm.joblib"
        self.eye_model = EyeStateModel(model_path) if model_path.exists() else None
        self.eye_source = "trained_gbm" if self.eye_model else "mediapipe_blendshape"
        self.reset()

    def reset(self) -> None:
        self.calib: list[float] = []
        self.ear_open: float | None = None
        self.closed_hist: deque[tuple[float, bool]] = deque()
        self.eyes_closed_since: float | None = None
        self.mouth_open_since: float | None = None
        self.yawns: deque[float] = deque()
        self.away_since: float | None = None
        self.pitch_zero: float | None = None
        self.yaw_zero: float | None = None
        self.state = "calibrating"
        self.ear_smooth: float | None = None
        self.eye_prob: float | None = None
        if getattr(self, "eye_model", None):
            self.eye_model.reset()

    def close(self) -> None:
        """Release the MediaPipe graph. Call this explicitly, from the thread that used the
        monitor. Never leave it to the garbage collector: MediaPipe's __del__ -> close() waits
        on an internal worker thread, and when the collector happens to run it inside another
        thread (we caught it inside PyTorch's torch.load) that wait never returns: a deadlock."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    @property
    def ear_threshold(self) -> float:
        return 0.72 * self.ear_open if self.ear_open else 0.2

    def process(self, frame_bgr: np.ndarray, t: float | None = None) -> dict:
        t = time.monotonic() if t is None else t
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts_ms = ts_ms
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

        if not res.face_landmarks:
            if self.eye_model:
                self.eye_model.push(None, self.ear_open)  # keeps the temporal window aligned
            self.eyes_closed_since = None
            self.away_since = self.away_since or t
            state = "no_face" if self.ear_open else "calibrating"
            if self.ear_open and t - self.away_since > settings.distraction_s:
                state = "distracted"
            self.state = state
            return {"face": False, "state": state, "ms": round((time.perf_counter() - t0) * 1000, 1)}

        lm = res.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
        ear_l, ear_r = _ear(pts, LEFT_EYE), _ear(pts, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        blink_bs = (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)) / 2
        # Exponential moving average: a 1-pole low-pass filter to suppress landmark jitter.
        self.ear_smooth = ear if self.ear_smooth is None else 0.6 * ear + 0.4 * self.ear_smooth
        mar = _mar(pts)
        pose = _head_pose(pts, w, h)
        eye_decision = None
        if self.eye_model:
            eye_decision = self.eye_model.push(
                {
                    "ear": ear, "ear_l": ear_l, "ear_r": ear_r, "blink": blink_bs,
                    "squint": (bs.get("eyeSquintLeft", np.nan) + bs.get("eyeSquintRight", np.nan)) / 2,
                    "look_down": (bs.get("eyeLookDownLeft", np.nan) + bs.get("eyeLookDownRight", np.nan)) / 2,
                    "pitch": pose[1] if pose else np.nan, "yaw": pose[0] if pose else np.nan,
                },
                self.ear_open,
            )

        # ---- Calibration: learn this driver's open-eye EAR and neutral head pose ----
        if self.ear_open is None:
            self.calib.append(ear)
            if pose:
                self.yaw_zero = pose[0] if self.yaw_zero is None else 0.9 * self.yaw_zero + 0.1 * pose[0]
                self.pitch_zero = pose[1] if self.pitch_zero is None else 0.9 * self.pitch_zero + 0.1 * pose[1]
            if len(self.calib) >= 45:
                # 80th percentile ignores the frames where the driver happened to blink.
                self.ear_open = float(np.percentile(self.calib, 80))
            self.state = "calibrating"
            return {
                "face": True, "state": "calibrating", "ear": round(ear, 3), "mar": round(mar, 3),
                "calib_progress": round(len(self.calib) / 45, 2),
                "eyes": self._eye_points(lm), "ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        # ---- Eyes: PERCLOS + microsleep ----
        if self.eye_model:
            # The model decides about the frame 6 frames (~200 ms) ago; a constant delay
            # doesn't change PERCLOS or closure durations.
            closed, self.eye_prob = eye_decision if eye_decision else (False, None)
        else:
            closed, self.eye_prob = blink_bs > 0.5, blink_bs
        self.closed_hist.append((t, closed))
        while self.closed_hist and t - self.closed_hist[0][0] > settings.perclos_window_s:
            self.closed_hist.popleft()
        perclos = sum(c for _, c in self.closed_hist) / max(len(self.closed_hist), 1)
        if closed:
            self.eyes_closed_since = self.eyes_closed_since or t
        else:
            self.eyes_closed_since = None
        closed_for = t - self.eyes_closed_since if self.eyes_closed_since else 0.0

        # ---- Mouth: yawns (open wide for >1.2s) ----
        if mar > 0.55:
            self.mouth_open_since = self.mouth_open_since or t
        else:
            if self.mouth_open_since and t - self.mouth_open_since > 1.2:
                self.yawns.append(t)
            self.mouth_open_since = None
        while self.yawns and t - self.yawns[0] > 120:
            self.yawns.popleft()

        # ---- Head pose: eyes-off-road ----
        yaw = pitch = roll = None
        looking_away = False
        if pose:
            yaw = pose[0] - (self.yaw_zero or 0)
            pitch = pose[1] - (self.pitch_zero or 0)
            roll = pose[2]
            looking_away = abs(yaw) > 30 or pitch < -22  # side glance or looking down at a phone
        if looking_away:
            self.away_since = self.away_since or t
        else:
            self.away_since = None
        away_for = t - self.away_since if self.away_since else 0.0

        # ---- State machine with hysteresis ----
        prev = self.state
        if closed_for >= settings.microsleep_s:
            state = "microsleep"
        elif away_for >= settings.distraction_s:
            state = "distracted"
        elif prev == "drowsy":
            state = "drowsy" if (perclos > 0.08 or len(self.yawns) >= 2) else "alert"
        elif perclos > 0.15 or len(self.yawns) >= 3:
            state = "drowsy"
        else:
            state = "alert"
        self.state = state

        return {
            "face": True,
            "state": state,
            "ear": round(self.ear_smooth, 3),
            "ear_threshold": round(self.ear_threshold, 3),
            "eye_prob": None if self.eye_prob is None else round(self.eye_prob, 3),
            "eye_threshold": self.eye_model.threshold if self.eye_model else 0.5,
            "eye_source": self.eye_source,
            "mar": round(mar, 3),
            "perclos": round(perclos, 3),
            "closed_for": round(closed_for, 2),
            "yawns_2min": len(self.yawns),
            "yaw": None if yaw is None else round(yaw, 1),
            "pitch": None if pitch is None else round(pitch, 1),
            "roll": None if roll is None else round(roll, 1),
            "eyes": self._eye_points(lm),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @staticmethod
    def _eye_points(lm) -> list[list[float]]:
        """Normalized eye + mouth contour points for the UI overlay."""
        ids = LEFT_EYE + RIGHT_EYE + [MOUTH["left"], MOUTH["right"], *MOUTH["top"], *MOUTH["bottom"]]
        return [[round(lm[i].x, 4), round(lm[i].y, 4)] for i in ids]


def summarize_driver(result: dict | None) -> str:
    if not result:
        return "Driver camera not active."
    s = result.get("state", "unknown")
    if s == "calibrating":
        return "Driver monitor is calibrating."
    if s == "no_face":
        return "Driver's face not visible."
    extra = []
    if result.get("perclos") is not None:
        extra.append(f"PERCLOS {result['perclos']:.0%}")
    if result.get("yawns_2min"):
        extra.append(f"{result['yawns_2min']} yawns in last 2 min")
    return f"Driver state: {s}" + (f" ({', '.join(extra)})." if extra else ".")
