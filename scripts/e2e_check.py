"""End-to-end smoke test against a running server (python -m drivemind.server.app).

Streams a synthetic road scene with a vehicle approaching at constant speed (so TTC
should fall and a forward-collision alert should fire), cabin frames, a spoken question
(macOS `say` -> 16 kHz PCM), and a typed command. Prints what comes back.

    .venv/bin/python scripts/e2e_check.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
import ultralytics
import websockets

ASSETS = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
URL = os.environ.get("DM_URL", "ws://127.0.0.1:8000/ws")


BUS_BOX = (4, 229, 796, 728)  # the bus in ultralytics' bus.jpg (x1, y1, x2, y2)


def approach_frame(sprite: np.ndarray, width_frac: float, w: int = 1280, h: int = 720) -> np.ndarray:
    """A vehicle `width_frac` of the frame wide, centred ahead on a plain sky/asphalt scene.
    Growing width_frac as T0 / (T0 - t) is exactly how a constant-speed approach looms. (An
    earlier version zoomed into the whole photo, where the bus overflowed the frame by
    TTC ~1.6 s, which no longer resembled a real approach.)"""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2], img[h // 2 :] = (200, 180, 150), (90, 90, 90)
    sw = max(int(w * width_frac), 8)
    sh = int(sw * sprite.shape[0] / sprite.shape[1])
    s = cv2.resize(sprite, (sw, sh))
    x0, yb = (w - sw) // 2, min(h, h // 2 + int(0.35 * sh))
    y0 = max(0, yb - sh)
    img[y0:yb, x0 : x0 + sw] = s[sh - (yb - y0) :, : w - x0]
    return img


def speech_pcm(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        aiff, pcm = os.path.join(d, "q.aiff"), os.path.join(d, "q.pcm")
        subprocess.run(["say", "-o", aiff, text], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", "-f", "s16le", pcm], check=True)
        return open(pcm, "rb").read()


async def main() -> None:
    bus = cv2.imread(os.path.join(ASSETS, "bus.jpg"))
    # Landscape "dashcam" frame with the bus centered in the ego corridor.
    road = cv2.copyMakeBorder(bus, 0, 0, 400, 400, cv2.BORDER_REPLICATE)
    face = cv2.imread(os.path.join(ASSETS, "zidane.jpg"))
    x1, y1, x2, y2 = BUS_BOX
    sprite = bus[y1:y2, x1:x2]
    T0, W0 = 3.0, 0.12  # true TTC when the approach starts; vehicle width then (fraction of frame)
    jpg = lambda im: cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()  # noqa: E731
    seen: dict[str, int] = {}
    truth_log: list[tuple[float, float]] = []
    first_road: list[float] = []
    replies: list[dict] = []

    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                seen[m["type"]] = seen.get(m["type"], 0) + 1
                if m["type"] == "road" and not first_road:
                    first_road.append(time.monotonic())
                if m["type"] == "road" and m.get("min_ttc") is not None and truth_log:
                    # True TTC of the most recent frame sent (results lag by one frame at most).
                    true_ttc = truth_log[-1][1] - (time.monotonic() - truth_log[-1][0])
                    print(f"  road: TTC est {m['min_ttc']:.2f}s vs true ~{true_ttc:.2f}s ({m['fcw']}), {m['ms']} ms, {m['fps']} fps")
                elif m["type"] in ("alert", "transcript", "reply", "deferred", "status"):
                    print(" ", m["type"], {k: v for k, v in m.items() if k not in ("type", "detail")})
                    if m["type"] == "reply":
                        replies.append(m)

        task = asyncio.create_task(reader())
        # Wait for the brain to be ready.
        for _ in range(240):
            st = json.loads(open("/dev/null").read() or "{}") if False else None  # noqa: F841
            import urllib.request
            h = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health"))
            if h["models"].get("router") == "ready":
                break
            await asyncio.sleep(1)
        print("models:", h)

        print("\n== 0. Warm-up: 1 s of static frames (measures how fast road perception starts)")
        t_first = time.monotonic()
        for _ in range(15):
            await ws.send(bytes([1]) + jpg(approach_frame(sprite, W0)))  # same scene the approach starts from
            await ws.send(bytes([2]) + jpg(face))
            await asyncio.sleep(1 / 15)
        print(f"  first road result {first_road[0] - t_first:.2f}s after the first frame" if first_road else "  no road result within 1 s")

        print("\n== 1. Vehicle ahead approaching at constant speed, true TTC 3.0 s -> 0.6 s (expect warning, then 'Brake!')")
        # Physically correct looming: image width w(t) = w0 * T0 / (T0 - t), so true TTC = T0 - t.
        start = time.monotonic()
        truth_log.clear()
        while (t := time.monotonic() - start) < T0 - 0.6:
            truth_log.append((time.monotonic(), T0 - t))
            await ws.send(bytes([1]) + jpg(approach_frame(sprite, W0 * T0 / (T0 - t))))
            await ws.send(bytes([2]) + jpg(face))
            await asyncio.sleep(1 / 15)

        print("\n== 2. Static scene, then a spoken question (push-to-talk)")
        truth_log.clear()
        for _ in range(20):
            await ws.send(bytes([1]) + jpg(road))
            await asyncio.sleep(1 / 15)
        await asyncio.sleep(1.0)
        pcm = speech_pcm("How many people can you see in front of the car?")
        await ws.send(json.dumps({"type": "ptt", "down": True}))
        silence = np.zeros(4000, np.int16).tobytes()
        for chunk in [silence, *[pcm[i : i + 2048] for i in range(0, len(pcm), 2048)], silence]:
            await ws.send(bytes([3]) + chunk)
            await asyncio.sleep(len(chunk) / 2 / 16000)
        await ws.send(json.dumps({"type": "ptt", "down": False}))
        t0 = time.time()
        while len(replies) < 1 and time.time() - t0 < 30:
            await asyncio.sleep(0.2)

        print("\n== 3. Typed multi-tool command")
        await ws.send(json.dumps({"type": "query", "text": "I'm cold, set it to 74 and play some lo-fi beats"}))
        while len(replies) < 2 and time.time() - t0 < 60:
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.5)
        task.cancel()
    print("\nmessage counts:", seen)


if __name__ == "__main__":
    asyncio.run(main())
