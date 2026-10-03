"""Training/serving parity: does the live DriverMonitor make the same eye-closed decisions
as the offline pipeline that produced the reported numbers?

(a) Replays every Eyeblink8 video's recorded features through the streaming EyeStateModel.
(b) Runs the real DriverMonitor (camera frames in) on one video end to end.

This checks *agreement*, not accuracy: the deployed model was trained on all of Eyeblink8,
so accuracy numbers come only from the leave-one-person-out evaluation.

    .venv/bin/python -m eval.verify_online
"""

import glob
import sys

import cv2
import joblib
import numpy as np

from drivemind.config import ROOT
from drivemind.perception.driver import DriverMonitor
from drivemind.perception.eye_state import HALF, RAW, EyeStateModel
from eval.eyeblink8_eval import calibration_baseline, features, load

MODEL = ROOT / "models" / "eye_state_gbm.joblib"
bundle = joblib.load(MODEL)
vids = load()
ok = True

print("(a) streaming EyeStateModel vs offline features")
for d in vids:
    f = features(d)
    offline = (bundle["model"].predict_proba(f["gbm"])[:, 1] * (f["face"] == 1)) >= bundle["threshold"]
    m = EyeStateModel(MODEL)
    base = calibration_baseline(f["ear"], f["face"])
    online = np.zeros_like(offline)
    for i in range(len(offline)):
        raw = None
        if f["face"][i] == 1:
            raw = {
                "ear": f["ear"][i], "ear_l": d["ear_l"][i], "ear_r": d["ear_r"][i], "blink": (d["eyeBlinkLeft"][i] + d["eyeBlinkRight"][i]) / 2,
                "squint": (d["eyeSquintLeft"][i] + d["eyeSquintRight"][i]) / 2, "look_down": (d["eyeLookDownLeft"][i] + d["eyeLookDownRight"][i]) / 2,
                "pitch": d["pitch"][i], "yaw": d["yaw"][i],
            }
        r = m.push(raw, base)
        if r is not None:
            online[i - HALF] = r[0]
    core = slice(HALF, len(offline) - HALF)  # window edges are padded differently by design
    agree = (online[core] == offline[core]).mean()
    ok &= agree > 0.999
    print(f"  video {d['video']}: agreement {agree:.4%}  (closed frames offline={offline[core].sum()}, online={online[core].sum()})")

vid = sys.argv[1] if len(sys.argv) > 1 else "4"
print(f"(b) real DriverMonitor on video {vid}")
d = next(v for v in vids if v["video"] == vid)
f = features(d)
offline = (bundle["model"].predict_proba(f["gbm"])[:, 1] * (f["face"] == 1)) >= bundle["threshold"]
cap = cv2.VideoCapture(glob.glob(str(ROOT / "data" / "eyeblink8" / vid / "*.avi"))[0])
dm = DriverMonitor()
assert dm.eye_source == "trained_gbm"
live = np.zeros_like(offline)
i = 0
while True:
    good, frame = cap.read()
    if not good:
        break
    r = dm.process(frame, t=i / 30.0)
    if r.get("eye_prob") is not None and i >= HALF:
        live[i - HALF] = r["eye_prob"] >= r["eye_threshold"]
    i += 1
core = slice(60, i - HALF)  # after calibration (45 frames) + window fill
agree = (live[core] == offline[core]).mean()
ok &= agree > 0.995
print(f"  agreement {agree:.4%}  (closed frames offline={offline[core].sum()}, live={live[core].sum()})")
sys.exit(0 if ok else 1)
