# DriveMind: Multimodal In-Vehicle AI Assistant

**An on-device co-pilot that sees the road, watches the driver, listens to the cabin, and reasons across all three in real time.**

DriveMind fuses three systems that today's cars keep separate:

- **ADAS-style road perception:** object detection, multi-object tracking, and monocular time-to-collision for forward-collision warnings.
- **Driver monitoring:** eye aspect ratio, PERCLOS, yawns, and head pose, used to detect drowsiness, microsleeps, and distraction.
- **A voice assistant with eyes:** streaming VAD → Whisper → a 4-bit vision-language model with tool calling, plus a CLIP-indexed visual memory so you can ask *"what did that sign say?"* after you've passed it.

A safety layer decides *when* the assistant may speak. It defers answers during hazards, shortens them under high workload, and starts a conversation when the driver gets drowsy.

Everything runs **on-device** on Apple Silicon (MLX + MPS). An optional **Claude** cloud brain handles harder questions, with automatic fallback to local when the network drops.

## Architecture

```
 Browser dashboard                         Python server (FastAPI + WebSocket)
 ─────────────────                         ────────────────────────────────────
 road cam / dashcam video ─JPEG─▶ YOLO11n → ByteTrack → looming TTC ───────┐
 cabin webcam ─────────────JPEG─▶ MediaPipe landmarks → EAR · PERCLOS · PnP ┤──▶ attention manager ──▶ alerts
 microphone (16 kHz PCM) ────────▶ Silero VAD → Whisper large-v3-turbo      │     (workload gating)
                                  CLIP keyframe memory (3 min) ◀────────────┘
                                  router ─▶ Qwen2.5-VL-3B 4-bit (MLX, on-device)
                                        └▶ Claude (vision + tool use, optional)
                                  vehicle tools behind a deterministic safety envelope
```

Each stage runs on its own worker thread with **latest-frame-wins backpressure**, so a slow LLM answer can never delay a collision warning.

## Measured performance (MacBook, Apple Silicon, 24 GB)

| Stage | Latency |
|---|---|
| Road: YOLO11n + ByteTrack + TTC | 9–18 ms / frame |
| Driver: landmarks + EAR/PERCLOS/head pose | 3–6 ms / frame |
| Visual memory: CLIP keyframe embedding | ~18 ms |
| Speech recognition: Whisper turbo, 3 s utterance | ~0.5–0.6 s |
| On-device VLM answer (Qwen2.5-VL-3B, 4-bit) | ~1.6–2.2 s, decode ~110 tok/s |
| End of speech → spoken answer (fully local) | ~2.5 s |
| TTC accuracy vs. analytic ground truth | within 0.15 s |

## Quick start

```bash
scripts/setup.sh      # venv, dependencies, ~5 GB of model weights (one time)
scripts/run.sh        # http://127.0.0.1:8000
```

Then open **http://127.0.0.1:8000** in Chrome:

1. **Driver monitor:** your webcam is picked automatically. Look ahead for ~3 s while it calibrates to your eyes.
2. **Road camera:** click **Load dashcam video** and choose any driving clip (search "dashcam footage" on YouTube, or record your own), or pick a second camera.
3. **Talk:** hold **Space** (or the mic button) and ask:
   - "How many cars are ahead of us?" / "Is the light green?"
   - "I'm freezing, warm it up and play some jazz" (makes two tool calls)
   - "What did that sign say?" (searches visual memory)
   - "Open all the windows" with the speed slider above 55 mph (the safety envelope vents them instead)
4. Close your eyes for 2 s, or look away from the screen, to trigger the driver alerts.

**Optional:** put `ANTHROPIC_API_KEY=...` in `.env` to enable the Claude brain. *Auto* mode then routes complex questions to the cloud and falls back to on-device if it's unreachable.

## Project layout

```
drivemind/
  perception/road.py      detection, tracking, time-to-collision, traffic-light color
  perception/driver.py    EAR, PERCLOS, yawns, head pose (PnP), hysteresis state machine
  audio/vad.py            streaming Silero VAD with hysteresis, hangover, pre-roll
  audio/asr.py            Whisper turbo on MLX with domain biasing + hallucination guards
  memory/scene_memory.py  CLIP keyframe selection + text→frame retrieval
  brain/tools.py          vehicle tools + safety envelope
  brain/safety.py         workload estimation, alert arbitration, reply gating
  brain/local_brain.py    Qwen2.5-VL 4-bit on MLX, prompted tool calling
  brain/claude_brain.py   Claude vision + native tool use
  brain/router.py         edge/cloud routing with circuit breaker
  server/session.py       per-client real-time orchestration and backpressure
web/                      dashboard (vanilla JS, AudioWorklet mic capture)
docs/CONCEPTS.md          study guide: every ML concept, its math, and interview questions
docs/VISION.md            use cases, market, and roadmap toward production ADAS
tests/                    unit tests (TTC vs ground truth, EAR geometry, safety envelope…)
scripts/e2e_check.py      end-to-end test over the WebSocket with synthetic video + speech
```

## Tests

```bash
.venv/bin/python -m pytest -q          # unit tests
.venv/bin/python scripts/e2e_check.py  # full pipeline (server must be running)
```

## Safety note

DriveMind is a research prototype, not a certified safety system. Its warnings are advisory. The LLM never actuates anything directly: every vehicle action passes through deterministic validation.

## Roadmap

See [docs/VISION.md](docs/VISION.md): dataset benchmarks (DMD, NTHU-DDD, DriveLM), lane detection, latency work (prefix caching, streaming TTS), a learned router, and porting perception to Jetson Orin / TensorRT.
