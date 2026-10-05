"""How often does the forward-collision warning (FCW) fire on ordinary real driving?

24 random 40-second BDD100K clips (16 min of US city driving, day and night; picked with
random.seed(7) from the 1,000-clip Kaggle mirror; see eval/bdd_clips.py). None of them
contains a crash, so nearly every critical "Brake!" second is a false alarm. There is no
ground truth here, so this measures the false-alarm *rate*, not precision/recall; the
detection side is covered by the TTC unit tests and scripts/e2e_check.py.

Frames go through the exact dashboard path: 640x360, ~15 fps, one RoadPerception per clip.

    .venv/bin/python -m eval.fcw_footage_eval           # downloads ~0.5 GB on first run
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from drivemind.config import ROOT
from drivemind.perception.road import RoadPerception
from eval.bdd_clips import get_clips

CLIPS = [
    "00207869-046fa443", "00268999-a4b8e39d", "002cd38e-ebe888e1", "00313a01-97de1f42", "0032419c-7d20b300",
    "003baca5-aab2e274", "003e23ee-07d32feb", "004071a4-a45d905f", "006c0799-964a2695", "008edf63-51af8ab6",
    "009bd04d-42445981", "00de601c-858a8a8d", "01041028-187a2d1f", "01118704-2d838d7f", "0128cdde-289cd9e9",
    "012c118a-b3092f3b", "015f23ad-5049df48", "01762203-e12f3f9f", "01853f47-6975b587", "019efe88-bbc2a4ac",
    "01ce75e2-fc93bd7e", "024b275b-33674b72", "028b5d16-af6a6275", "02a26ce5-42f8fd75",
]
RANK = {"none": 0, "warning": 1, "critical": 2}


def scan(path: str) -> dict:
    rp, cap = RoadPerception(), cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    secs: dict[int, str] = {}
    bright: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 2 == 0:  # ~15 fps, like the dashboard
            small = cv2.resize(frame, (640, 360))
            r = rp.process(small, i / fps)
            k = int(i / fps)
            if RANK[r["fcw"]] >= RANK[secs.get(k, "none")]:
                secs[k] = r["fcw"]
            if i % 30 == 0:
                bright.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean()))
        i += 1
    return {"seconds": len(secs), "warning_s": sum(v == "warning" for v in secs.values()),
            "critical_s": sum(v == "critical" for v in secs.values()), "night": float(np.mean(bright)) < 60,
            "timeline": "".join({"none": ".", "warning": "w", "critical": "C"}[secs[k]] for k in sorted(secs))}


def main() -> None:
    paths = get_clips(CLIPS)
    rows = {os.path.basename(p)[:-4]: scan(p) for p in paths}
    total = sum(r["seconds"] for r in rows.values())
    warn = sum(r["warning_s"] for r in rows.values())
    crit = sum(r["critical_s"] for r in rows.values())
    md = [
        "# Forward-collision warning on real driving (BDD100K, no crashes)", "",
        f"{len(rows)} clips, {total} s ({total / 60:.1f} min). Critical ('Brake!') active **{crit} s "
        f"({crit / total:.1%})**, warning {warn} s ({warn / total:.1%}). With no crashes in the footage, "
        "nearly all of these are false alarms.", "",
        "| Clip | Light | Warning s | Critical s | Timeline (. none, w warning, C critical) |", "|---|---|---|---|---|",
    ] + [f"| {k} | {'night' if r['night'] else 'day'} | {r['warning_s']} | {r['critical_s']} | `{r['timeline']}` |" for k, r in rows.items()]
    out = ROOT / "eval" / "results"
    (out / "fcw_bdd100k.md").write_text("\n".join(md) + "\n")
    (out / "fcw_bdd100k.json").write_text(json.dumps({"total_s": total, "warning_s": warn, "critical_s": crit, "clips": rows}, indent=2))
    print("\n".join(md[:3]))


if __name__ == "__main__":
    main()
