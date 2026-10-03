"""Extract per-frame eye features + ground-truth labels from the Eyeblink8 dataset.

Dataset: Fogelton & Benesova, "Eye blink detection based on motion vectors analysis",
CVIU 2016. 8 videos (640x480, 30 fps) of 4 people, 408 annotated blinks, GPL-3.
Download: https://www.blinkingmatters.com/research  ->  data/eyeblink8/<id>/*.avi|*.tag

Features use the *same* landmark model and geometry as the shipping DriverMonitor
(drivemind/perception/driver.py), so evaluation results describe the real system.

    .venv/bin/python -m eval.eyeblink8_extract
"""

from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from drivemind.config import ROOT, settings
from drivemind.perception.driver import LEFT_EYE, RIGHT_EYE, _ear, _head_pose, _mar

DATA = ROOT / "data" / "eyeblink8"
OUT = ROOT / "data" / "features"
# Identity of the person in each video (from inspecting frames; the .tag "#author" field is
# the annotator, not the subject). Used for leave-one-person-out evaluation.
PERSON = {"1": "A", "2": "A", "3": "B", "4": "B", "8": "C", "9": "C", "10": "D", "11": "D"}
BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeLookDownLeft", "eyeLookDownRight"]


def parse_tag(path: str, n_frames: int) -> dict[str, np.ndarray]:
    """Annotation rows: frame:blinkID:NF:LE_FC:LE_NV:RE_FC:RE_NV:face bbox:eye corners.
    blinkID = -1 outside blinks; FC = 'C' when that eye is fully closed; NV = 'N' when
    not visible; NF = 'N' when the face is non-frontal."""
    blink = np.full(n_frames, -1, np.int32)
    closed = np.zeros(n_frames, bool)
    not_visible = np.zeros(n_frames, bool)
    started = False
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line == "#start":
            started = True
            continue
        if not started or not line or line.startswith("#"):
            continue
        f = line.split(":")
        fid = int(f[0])
        if fid >= n_frames:
            continue
        blink[fid] = int(f[1])
        closed[fid] = f[3] == "C" or f[5] == "C"
        not_visible[fid] = f[4] == "N" or f[6] == "N" or f[2] == "N"
    return {"blink_id": blink, "closed": closed, "not_visible": not_visible}


def extract(video_dir: str) -> dict[str, np.ndarray]:
    avi = glob.glob(os.path.join(video_dir, "*.avi"))[0]
    tag = glob.glob(os.path.join(video_dir, "*.tag"))[0]
    cap = cv2.VideoCapture(avi)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.face_model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
    )
    cols = ["face", "ear_l", "ear_r", "mar", "yaw", "pitch", *BLENDSHAPES]
    feats = {c: np.full(n, np.nan, np.float32) for c in cols}
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
        if not res.face_landmarks:
            feats["face"][i] = 0
            continue
        lm = res.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
        feats["face"][i] = 1
        feats["ear_l"][i] = _ear(pts, LEFT_EYE)
        feats["ear_r"][i] = _ear(pts, RIGHT_EYE)
        feats["mar"][i] = _mar(pts)
        pose = _head_pose(pts, w, h)
        if pose:
            feats["yaw"][i], feats["pitch"][i] = pose[0], pose[1]
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        for b in BLENDSHAPES:
            feats[b][i] = bs.get(b, np.nan)
    landmarker.close()
    labels = parse_tag(tag, n)
    return {**feats, **labels, "fps": np.float32(fps)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for vid in sorted(PERSON, key=int):
        out = OUT / f"eyeblink8_{vid}.npz"
        if out.exists():
            print(f"video {vid}: cached")
            continue
        t0 = time.perf_counter()
        d = extract(str(DATA / vid))
        np.savez_compressed(out, **d, person=PERSON[vid], video=vid)
        n = len(d["face"])
        print(
            f"video {vid} (person {PERSON[vid]}): {n} frames, face found {np.nanmean(d['face']):.1%}, "
            f"closed frames {d['closed'].sum()}, blinks {len(set(d['blink_id'][d['blink_id'] >= 0]))}, "
            f"{time.perf_counter() - t0:.0f}s"
        )


if __name__ == "__main__":
    main()
