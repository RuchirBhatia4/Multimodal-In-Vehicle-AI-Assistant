"""Extract per-frame eye/face features from UTA-RLDD videos, one video at a time.

Dataset: Ghoddoosian, Kornmayer & Athitsos, "A Realistic Dataset and Baseline Temporal
Model for Early Drowsiness Detection", CVPRW 2019. 60 people filmed themselves (phone/web
camera, < 30 fps) for ~10 min in three self-reported states (KSS-based): 0 = alert,
5 = low vigilant, 10 = drowsy. https://sites.google.com/view/utarldd/home (cite if used).

Disk is tight, so videos are extracted from the fold zip one at a time, processed, and
deleted. Frames are downscaled to <= 640 px wide (EAR is scale-invariant, and Eyeblink8,
which the eye model was trained on, is 640x480). Raw per-frame features + timestamps are saved;
resampling to 30 fps happens in eval/rldd_eval.py.

    .venv/bin/python -m eval.rldd_extract data/rldd/Fold3_part2.zip [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from drivemind.config import ROOT

OUT = ROOT / "data" / "features"
TMP = ROOT / "data" / "rldd" / "tmp"
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp")
# Third-party Kaggle mirror of the same files (byte sizes verified against the official
# Google Drive zips); has per-video files, so it avoids Drive's per-zip download quota.
KAGGLE_DATASET = "rishab260/uta-reallife-drowsiness-dataset"
BLENDSHAPES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeLookDownLeft", "eyeLookDownRight"]


def parse_member(name: str) -> tuple[str, int, str] | None:
    """'Fold3_part2/37/10.mov' -> ('37', 10, ''); split recordings like '32/10_2.mp4' ->
    ('32', 10, '_2'). Parts are concatenated in eval/rldd_eval.py."""
    m = re.search(r"(\d+)[/\\](0|5|10)(_\d+)?\.[A-Za-z0-9]+$", name)
    return (m.group(1), int(m.group(2)), m.group(3) or "") if m and name.lower().endswith(VIDEO_EXT) else None


def process_video(zip_path: str, member: str, pid: str, label: int, part: str = "") -> str:
    """Zip source: extract one member, featurize it, delete it."""
    out = OUT / f"rldd_{pid}_{label}{part}.npz"
    if out.exists():
        return f"{pid}/{label}{part}: cached"
    job_dir = TMP / f"{pid}_{label}{part}"  # one folder per job: parallel extracts must not race on mkdir
    job_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        path = z.extract(member, job_dir)
    try:
        return featurize(path, out, pid, label, part, member.split("/")[0])
    finally:
        os.remove(path)


def process_kaggle_video(name: str, pid: str, label: int, part: str = "") -> str:
    """Kaggle source (per-video files): download one video, featurize it, delete it."""
    import glob
    import subprocess

    out = OUT / f"rldd_{pid}_{label}{part}.npz"
    if out.exists():
        return f"{pid}/{label}{part}: cached"
    job_dir = TMP / f"k_{pid}_{label}{part}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), "datasets", "download", KAGGLE_DATASET,
                            "-f", name, "-p", str(job_dir), "--unzip", "-q"], capture_output=True, text=True)
        # Kaggle serves single files wrapped in a zip (e.g. "10.mp4.zip"), and --unzip does not
        # unwrap single-file downloads in CLI 2.x, so unwrap it here.
        for zpath in glob.glob(str(job_dir / "**" / "*.zip"), recursive=True):
            with zipfile.ZipFile(zpath) as z:
                z.extractall(job_dir)
            os.remove(zpath)
        vids = [f for f in glob.glob(str(job_dir / "**" / "*"), recursive=True) if f.lower().endswith(VIDEO_EXT)]
        if r.returncode != 0 or not vids:
            raise RuntimeError(f"kaggle download failed for {name}: {(r.stderr or r.stdout)[-300:]}")
        return featurize(vids[0], out, pid, label, part, name.split("/")[0])
    finally:
        import shutil

        shutil.rmtree(job_dir, ignore_errors=True)


def featurize(path: str, out, pid: str, label: int, part: str, fold: str) -> str:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    from drivemind.config import settings
    from drivemind.perception.driver import LEFT_EYE, RIGHT_EYE, _ear, _head_pose, _mar

    t0 = time.perf_counter()
    if True:  # (kept for a minimal diff of the body below)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        lm_opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.face_model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        )
        landmarker = vision.FaceLandmarker.create_from_options(lm_opts)
        cols = ["t", "face", "ear_l", "ear_r", "mar", "yaw", "pitch", *BLENDSHAPES]
        rows: list[list[float]] = []
        last_ms, i, size = -1, 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not t_ms or t_ms <= last_ms:  # some phone videos report bad timestamps
                t_ms = last_ms + 1000.0 / fps if last_ms >= 0 else 0.0
            last_ms = t_ms
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            size = (w, h)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t_ms))
            row = [t_ms / 1000.0, 0.0] + [np.nan] * (len(cols) - 2)
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
                pose = _head_pose(pts, w, h)
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                row = [
                    t_ms / 1000.0, 1.0, _ear(pts, LEFT_EYE), _ear(pts, RIGHT_EYE), _mar(pts),
                    pose[0] if pose else np.nan, pose[1] if pose else np.nan,
                    *[bs.get(b, np.nan) for b in BLENDSHAPES],
                ]
            rows.append(row)
            i += 1
        landmarker.close()
        cap.release()
        arr = np.array(rows, dtype=np.float32)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{c: arr[:, k] for k, c in enumerate(cols)}, fps=np.float32(fps), pid=pid, label=label,
                            fold=fold,
                            width=size[0] if size else 0, height=size[1] if size else 0)
        dur = arr[-1, 0] if len(arr) else 0
        return (f"{pid}/{label}{part}: {i} frames, {fps:.1f} fps, {dur / 60:.1f} min, face {arr[:, 1].mean():.1%}, "
                f"{time.perf_counter() - t0:.0f}s")


def kaggle_files(fold_part: str) -> list[str]:
    import json
    import urllib.parse
    import urllib.request

    base = f"https://www.kaggle.com/api/v1/datasets/list/{KAGGLE_DATASET}"
    names, token = [], None
    while True:
        url = base + (f"?pageToken={urllib.parse.quote(token)}" if token else "")
        d = json.load(urllib.request.urlopen(url, timeout=30))
        names += [f["name"] for f in d.get("datasetFiles", [])]
        token = d.get("nextPageTokenNullable")
        if not token:
            return [n for n in names if n.startswith(fold_part + "/")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", nargs="?", help="official fold zip (Google Drive)")
    ap.add_argument("--kaggle-part", help="e.g. Fold2_part1: fetch videos one by one from the Kaggle mirror instead")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.kaggle_part:
        jobs = [(n, *parse_member(n)) for n in kaggle_files(args.kaggle_part) if parse_member(n)]
        submit = lambda ex, m, p, lab, part: ex.submit(process_kaggle_video, m, p, lab, part)  # noqa: E731
    else:
        with zipfile.ZipFile(args.zip) as z:
            jobs = [(m, *parse_member(m)) for m in z.namelist() if parse_member(m)]
        submit = lambda ex, m, p, lab, part: ex.submit(process_video, args.zip, m, p, lab, part)  # noqa: E731
    print(f"{len(jobs)} videos: {sorted({j[1] for j in jobs})}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [submit(ex, m, p, lab, part) for m, p, lab, part in jobs]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001 - report and keep going; rerun picks up the rest
                print(f"FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
