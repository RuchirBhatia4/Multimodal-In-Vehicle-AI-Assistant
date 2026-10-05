"""Build a 78-second demo reel of real driving for the dashboard's "Load dashcam video".

Segments were picked by running the app's own road perception over 24 BDD100K clips
(eval/fcw_footage_eval.py) and choosing stretches with traffic lights, pedestrians and
readable signs where the collision warning stays quiet, plus one moment that justifies it
(a minivan cutting across close in front). Writes data/demo/drivemind_demo_reel.mp4 and a
CREDITS.txt with the BDD100K notice to include wherever you publish a video made from it.

    .venv/bin/python scripts/make_demo_reel.py      # downloads 6 clips (~120 MB) on first run
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivemind.config import ROOT  # noqa: E402
from eval.bdd_clips import BDD100K_NOTICE, get_clips  # noqa: E402

SEGMENTS = [  # (BDD100K clip, start s, length s, what to look for)
    ("024b275b-33674b72", 0, 16, "Manhattan crosswalk: pedestrians, many cars, traffic lights"),
    ("01762203-e12f3f9f", 24, 16, "wide avenue: pedestrians, a mail truck, traffic lights"),
    ("00207869-046fa443", 21, 6, "a minivan cuts across close in front: collision warning at ~0:36"),
    ("02a26ce5-42f8fd75", 22, 14, "intersection with several traffic lights"),
    ("028b5d16-af6a6275", 13, 14, "'KEEP INTERSECTION CLEAR' and 'DEAD END' signs at ~1:02"),
    ("00268999-a4b8e39d", 22, 12, "night driving, green light"),
]
OUT = ROOT / "data" / "demo"


def main() -> None:
    clips = list(dict.fromkeys(c for c, *_ in SEGMENTS))  # unique, in segment order
    paths = dict(zip(clips, get_clips(clips)))  # get_clips returns paths in the same order
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (clip, start, length, _) in enumerate(SEGMENTS):
            part = Path(tmp) / f"{i:02d}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", paths[clip], "-t", str(length),
                            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-an",
                            str(part)], check=True)
            parts.append(f"file '{part.name}'")  # concat paths are relative to the list file
        (Path(tmp) / "list.txt").write_text("\n".join(parts) + "\n")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(Path(tmp) / "list.txt"),
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
                        "-movflags", "+faststart", "-an", str(OUT / "drivemind_demo_reel.mp4")], check=True)
    timeline, t = [], 0
    for clip, start, length, what in SEGMENTS:
        timeline.append(f"{t // 60}:{t % 60:02d}-{(t + length) // 60}:{(t + length) % 60:02d}  {what}")
        t += length
    (OUT / "CREDITS.txt").write_text(BDD100K_NOTICE + "\n\nClips used: " + ", ".join(c for c, *_ in SEGMENTS) + "\n")
    print(f"Wrote {OUT / 'drivemind_demo_reel.mp4'} ({t} s)\n" + "\n".join(timeline))


if __name__ == "__main__":
    main()
