"""Check that eval.eyeblink8_eval.deployed_rule() reproduces the ORIGINAL DriverMonitor rule (calibrated EAR, still exposed as ear/ear_threshold)
frame-for-frame on one Eyeblink8 video, so 'deployed rule' results describe shipping code.

    .venv/bin/python -m eval.verify_legacy_rule [video_id]
"""

import glob
import sys

import cv2
import numpy as np

from drivemind.config import ROOT
from drivemind.perception.driver import DriverMonitor
from eval.eyeblink8_eval import deployed_rule, features, load

vid = sys.argv[1] if len(sys.argv) > 1 else "4"
d = next(v for v in load() if v["video"] == vid)
expected = deployed_rule(d, features(d))

cap = cv2.VideoCapture(glob.glob(str(ROOT / "data" / "eyeblink8" / vid / "*.avi"))[0])
dm = DriverMonitor()
actual = np.zeros(len(expected), bool)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = dm.process(frame, t=i / 30.0)
    if r.get("face") and r["state"] != "calibrating":
        actual[i] = r["ear"] < r["ear_threshold"]
    i += 1
agree = (actual == expected).mean()
print(f"video {vid}: {i} frames, agreement {agree:.4%}, closed frames real={actual.sum()} replica={expected.sum()}")
sys.exit(0 if agree > 0.999 else 1)
