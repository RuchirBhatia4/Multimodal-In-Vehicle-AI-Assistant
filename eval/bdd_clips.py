"""Fetch individual BDD100K driving clips from a Kaggle mirror and re-encode them upright.

Source: "Driving Video with Object Tracking" (robikscube/driving-video-with-object-tracking),
1,000 40-second clips from the BDD100K tracking set. Kaggle lists the mirror as CC0, but the
data is BDD100K's, whose license governs it: educational, research and not-for-profit use,
keeping the copyright notice (https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst).
Downloading needs your own Kaggle API token in ~/.kaggle/.

The clips are phone recordings stored sideways with a "rotate 90" flag that OpenCV ignores,
so every clip is re-encoded with ffmpeg (which applies the rotation) to an upright
1280x720, 30 fps H.264 MP4 that plays the same in Chrome, QuickTime and OpenCV.
"""

from __future__ import annotations

import glob
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

from drivemind.config import ROOT

DATASET = "robikscube/driving-video-with-object-tracking"
PREFIX = "bdd100k_videos_train_00/bdd100k/videos/train/"
RAW, NORM = ROOT / "data" / "bdd100k" / "raw", ROOT / "data" / "bdd100k" / "norm"

BDD100K_NOTICE = """Road footage: BDD100K (Berkeley DeepDrive), https://bdd-data.berkeley.edu/
Copyright (c) 2018. The Regents of the University of California (Regents). All Rights Reserved.
Used for educational, research and not-for-profit purposes under the BDD100K license:
https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst
Clips obtained via the Kaggle mirror "Driving Video with Object Tracking" (robikscube)."""


def _fetch_one(clip: str) -> str:
    out = RAW / f"{clip}.mov"
    if out.exists():
        return out.name
    tmp = RAW / f"_dl_{clip}"
    tmp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "kaggle"), "datasets", "download", DATASET,
                        "-f", f"{PREFIX}{clip}.mov", "-p", str(tmp), "-q"], capture_output=True, text=True)
    for z in glob.glob(str(tmp / "*.zip")):  # Kaggle wraps single files in a zip
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        os.remove(z)
    got = glob.glob(str(tmp / "**" / "*.mov"), recursive=True)
    if not got:
        raise RuntimeError(f"download failed for {clip}: {(r.stderr or r.stdout)[-300:]}")
    os.replace(got[0], out)
    subprocess.run(["rm", "-rf", str(tmp)])
    return out.name


def normalize(clip: str) -> str:
    out = NORM / f"{clip}.mp4"
    if not out.exists():
        NORM.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW / f"{clip}.mov"),
                        "-vf", "scale=1280:720,fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(out)], check=True)
    return str(out)


def get_clips(clips: list[str], workers: int = 6) -> list[str]:
    """Download (if missing) and normalize the given clip ids; returns normalized MP4 paths."""
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_fetch_one, clips))
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(normalize, clips))
