"""End-to-end smoke test against a running server (python -m drivemind.server.app).

Streams synthetic road frames that *zoom in* on a vehicle (simulated approach, so TTC
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


def zoom(img: np.ndarray, f: float) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = int(w / f), int(h / f)
    x0, y0 = (w - cw) // 2, int((h - ch) * 0.65)
    return cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw], (w, h))


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
    jpg = lambda im: cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()  # noqa: E731
    seen: dict[str, int] = {}
    truth_log: list[tuple[float, float]] = []
    replies: list[dict] = []

    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                seen[m["type"]] = seen.get(m["type"], 0) + 1
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

        print("\n== 1. Approaching vehicle at constant speed, true TTC 2.5 s -> 0.7 s (expect an FCW alert)")
        # Physically correct looming: image scale s(t) = s0 * T0 / (T0 - t), so true TTC = T0 - t.
        T0, start = 2.5, time.monotonic()
        truth_log.clear()
        while (t := time.monotonic() - start) < 1.8:
            truth_log.append((time.monotonic(), T0 - t))
            await ws.send(bytes([1]) + jpg(zoom(road, T0 / (T0 - t))))
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
