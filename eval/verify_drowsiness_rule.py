"""Parity: does the live DrowsinessRule (drivemind/perception/drowsiness.py) raise alerts on the
same videos as the offline rule that was evaluated (eval/drowsy_rule.long_closure_alerts)?
Also replays every video at 15 fps, the rate the dashboard sends cabin frames.

    .venv/bin/python -m eval.verify_drowsiness_rule
"""

import json
import sys

import numpy as np

from drivemind.config import ROOT
from drivemind.perception.drowsiness import DrowsinessRule
from eval.drowsy_rule import COOLDOWN_S, long_closure_alerts
from eval.eyeblink8_eval import FPS
from eval.eyeblink8_eval import features as eb8_features
from eval.eyeblink8_eval import load as load_eyeblink8
from eval.rldd_eval import load_videos

rule = json.loads((ROOT / "eval" / "results" / "preregistration.json").read_text())["frozen_rule"]


def online_alerts(score: np.ndarray, face: np.ndarray, step: int = 1) -> int:
    """Like the app's AttentionManager: alert whenever the state is drowsy/microsleep and the
    cooldown has passed (it re-alerts while the driver *stays* drowsy)."""
    r, n, last = DrowsinessRule(), 0, -1e9
    for i in range(0, len(score), step):
        t = i / FPS
        out = r.update(t, float(score[i]) if face[i] else None)
        if (out["drowsy"] or out["microsleep"]) and t - last >= COOLDOWN_S:
            n, last = n + 1, t
    return n


rows = []
for v in load_videos(folds={"Fold3_part2", "Fold5_part1", "Fold2_part1"}):
    face = v["face"] > 0.5
    score = np.nan_to_num((v["eyeBlinkLeft"] + v["eyeBlinkRight"]) / 2)
    rows.append((f"rldd {v['fold']} {v['pid']}/{v['label']}", score, face))
for d in load_eyeblink8():
    f = eb8_features(d)
    rows.append((f"eyeblink8 {d['video']}", f["blink"], f["face"] == 1))

agree30 = agree15 = 0
for name, score, face in rows:
    off = len(long_closure_alerts(face & (score > 0.5), rule["min_closure_s"], rule["k_per_60s"]))
    on30, on15 = online_alerts(score, face), online_alerts(score, face, step=2)
    a30, a15 = (off > 0) == (on30 > 0), (off > 0) == (on15 > 0)
    agree30 += a30
    agree15 += a15
    flag = "" if a30 and a15 else "   <-- differs"
    print(f"{name:28s} offline {off:3d}  live@30fps {on30:3d}  live@15fps {on15:3d}{flag}")
print(f"\nvideos with/without an alert, live vs offline: 30 fps {agree30}/{len(rows)}, 15 fps {agree15}/{len(rows)}")
sys.exit(0 if agree30 == len(rows) else 1)
