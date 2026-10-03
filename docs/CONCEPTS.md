# DriveMind: The ML Concepts, Explained

This is your study guide for the project. Each section covers **what** the technique is, **why** it's here, the **key math**, where to find it in the code, an **interview question** you should be able to answer, and an **exercise** to deepen it.

Read it alongside the code: every module's docstring is a short version of its section here.

```
 Browser (web/)                        Server (drivemind/)
 ─────────────                         ───────────────────
 road camera / dashcam ──JPEG 15fps──▶ perception/road.py     YOLO11 → ByteTrack → TTC ─┐
 cabin webcam ───────────JPEG 15fps──▶ perception/driver.py   landmarks → EAR/PERCLOS/pose ┤
 mic (AudioWorklet 16k) ─PCM─────────▶ audio/vad.py → asr.py  Silero VAD → Whisper turbo   │
                                       memory/scene_memory.py CLIP keyframes + retrieval   │
                                       brain/safety.py        workload + alert arbitration ◀┘
                                       brain/router.py ──▶ local_brain.py (Qwen2.5-VL 4-bit, MLX)
                                                      └──▶ claude_brain.py (Claude + tools)
 dashboard ◀──────── JSON results, alerts, replies ── server/session.py (threads + backpressure)
```

---

## 1. Real-time object detection (YOLO11)
**What.** A single-stage detector: one CNN forward pass predicts, for a grid over the image, box coordinates, an "objectness" score and class probabilities. Non-Maximum Suppression (NMS) then removes duplicate boxes (keep the highest-scoring box; drop others with IoU > threshold).

**Why here.** It runs in ~10–15 ms per frame on Apple Silicon (MPS). Two-stage detectors (Faster R-CNN) are more accurate on small objects but far too slow for a 15 fps loop.

**Key ideas.** IoU = area(A∩B)/area(A∪B). Detection quality is measured with mAP (mean average precision over IoU thresholds 0.5:0.95). YOLO11n ("nano") has ~2.6M parameters; the s/m/l/x variants trade speed for accuracy.

**Code.** `perception/road.py → RoadPerception.process`

**Interview Q.** *Why does a detector need NMS, and what goes wrong in crowded scenes?* (Overlapping pedestrians get suppressed as "duplicates". Soft-NMS and NMS-free designs such as YOLOv10/DETR address this.)

**Exercise.** Swap `yolo11n.pt` for `yolo11s.pt` and plot latency vs. how many distant cars you detect on a highway clip.

## 2. Multi-object tracking (ByteTrack)
**What.** Assigns persistent IDs across frames. Each track has a **Kalman filter** (a constant-velocity motion model) that predicts where the box will be next frame. New detections are matched to predictions using IoU with the **Hungarian algorithm** (optimal bipartite matching; the `lap` package).

**ByteTrack's trick.** It does a second matching round using *low-confidence* detections, which rescues objects that are briefly blurred or occluded instead of dropping them and spawning a new ID.

**Why here.** Time-to-collision needs the *same* car's size over time. Without IDs you can't compute a rate of change.

**Interview Q.** *What is an ID switch, and how would you reduce them?* (Add appearance embeddings, as in DeepSORT/BoT-SORT, or tune the match thresholds.)

## 3. Monocular time-to-collision ("looming")
**What.** Estimate seconds-until-impact from **one camera**, with no depth.

**Math.** With a pinhole camera, an object of width W at distance Z appears with size s = fW/Z. If it approaches at constant speed v: TTC = Z/v = s/(ds/dt) = 1 / (d ln s / dt). Distance and speed cancel out. We fit a least-squares line to ln(s) over a 0.8 s window (robust to per-frame box jitter) and take 1/slope.

**Subtle bug we found and fixed.** A regression over a window estimates the slope at the window's *centroid*, which is 0.4 s in the past. Under constant closing speed TTC drops 1 s per second, so we subtract (t_now − t_centroid). Without the fix, warnings were ~0.4 s late. `tests/test_core.py` checks this against analytic ground truth.

**Ego corridor.** Only objects roughly ahead of us (center 40% of the frame, lower part) count as threats.

**Interview Q.** *When does looming TTC fail?* (Turning objects, partial occlusion that changes box size, ego-motion such as pitch over bumps, and objects that are moving away but still growing in the box because the crop is changing.)

**Exercise.** Read speed via OBD-II and fuse it in. With known ego-speed you can also estimate *distance*.

## 4. Driver monitoring: landmarks, EAR, PERCLOS, head pose
- **Face Landmarker** (MediaPipe): a BlazeFace detector, then a regression network for 478 3-D landmarks. Runs on CPU in ~4 ms.
- **EAR** (Eye Aspect Ratio) = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖). It is scale-invariant and drops toward 0 when the eye closes.
- **Calibration.** We take the 80th percentile of EAR over the first ~3 s as this driver's "open" baseline, and use 72% of it as the threshold. Fixed thresholds fail across face shapes and camera angles.
- **Smoothing.** An EMA (exponential moving average, a 1-pole IIR low-pass filter) removes landmark jitter.
- **PERCLOS.** The fraction of the last N seconds with eyes closed. This is the fatigue measure from NHTSA research; the industry uses it at roughly 15% and above.
- **Head pose via PnP.** Given 6 2-D landmarks and a generic 3-D face model, `solvePnP` finds the rotation and translation that best project the 3-D points onto the 2-D ones. Rodrigues converts that to a rotation matrix, then to yaw, pitch and roll.
- **Hysteresis state machine.** The driver enters "drowsy" at PERCLOS > 15% and leaves at < 8%. Microsleep means eyes closed ≥ 1 s. Distracted means the head is turned or pitched down for ≥ 2 s.

**Code.** `perception/driver.py`

**Interview Q.** *Why do production DMS use infrared cameras?* (They work at night, see through sunglasses, and give consistent illumination. The 940 nm IR is invisible to the driver.)

**Exercise.** Validate on the **NTHU-DDD** or **DMD** dataset: compute precision and recall of the "drowsy" state against the labels.

## 5. Streaming voice activity detection (Silero VAD)
A ~2 MB recurrent network scores each 32 ms chunk for speech probability, carrying state between chunks (so each chunk is O(1)). We add **hysteresis** (start above 0.5, end below 0.35), a **hangover** (600 ms of silence ends the utterance) and a **pre-roll** (300 ms before onset, so the first syllable isn't clipped).

**Why it matters.** Whisper hallucinates on silence ("Thanks for watching!"). The VAD also sets *when we can start answering* (endpointing latency), which users feel directly.

## 6. On-device speech recognition (Whisper large-v3-turbo on MLX)
An encoder–decoder transformer over log-Mel spectrograms. *Turbo* prunes the decoder from 32 layers to 4: the encoder runs once (parallel), while the decoder runs per token (sequential). Cutting decoder depth gives ~8× faster decoding for a small loss in word error rate (WER).

We bias the vocabulary with `initial_prompt` (domain terms) and drop segments with high `no_speech_prob`.

**Measured:** ~0.5–0.6 s for a 3–4 s utterance on this Mac.

## 7. Vision-language models + 4-bit quantization (Qwen2.5-VL-3B on MLX)
- **Architecture.** A ViT encodes the image into patches (14×14 px, merged 2×2). A projector maps those patches into the LLM's embedding space, and the LLM attends over the image and text tokens together.
- **Image tokens scale with area.** At 448 px wide a frame is ~250 visual tokens; at 896 px it is ~1000. We downscale for normal queries and *upscale* retrieved frames when the user wants text read ("what did that sign say?").
- **Quantization.** Group-wise affine 4-bit: each block of 64 weights stores a scale and bias, with w ≈ scale·q + bias, q ∈ {0..15}. This takes 3B params from ~6 GB to ~2 GB.
- **Prefill vs. decode.** Prefill is compute-bound (we measure ~370–690 tok/s on the prompt). Decode is memory-bandwidth-bound (~110 tok/s). Quantization helps decode the most, because each generated token has to stream all the weights.
- **Prompted tool calling + few-shot.** The 3B model initially made only one of two requested tool calls. Adding worked examples to the prompt fixed it (*in-context learning*).

**Code.** `brain/local_brain.py`

**Interview Q.** *Your on-device LLM is too slow. List five levers.* (Smaller or quantized model; fewer image tokens; prefix/KV caching of the static system prompt; speculative decoding; shorter outputs; streaming TTS so speech starts at the first sentence.)

## 8. Tool calling & the safety envelope
The LLM proposes actions as structured JSON; deterministic code validates, clamps or blocks them (no 200°F, no fully-open windows above 55 mph). When the envelope overrides the model, **the envelope's message is what gets spoken**. We found the local model confidently saying "Opening all the windows" while the car actually vented them, and fixed it.

This is the core idea behind using LLMs in safety-relevant products: **the LLM is advisory, the guardrails are code.** (ISO 26262 functional safety, ISO 21448 SOTIF.)

## 9. Semantic visual memory (CLIP + retrieval)
**CLIP** is trained contrastively (InfoNCE loss) so that matching image–caption pairs have high cosine similarity and mismatched pairs low. Result: text and images share one embedding space.

- **Keyframe selection.** We store a frame when 1 − cos(e_t, e_last) > 0.10 (the *meaning* changed) or 5 s have passed. Pixel differencing would fire on every lighting change.
- **Retrieval.** score = cos(text_query, frame) + 0.05·recency. The top frame goes back to the VLM as context. That's RAG, with frames instead of documents.
- **Bounded memory.** A 3-minute ring buffer.

**Interview Q.** *CLIP retrieval returns the wrong frame for "the exit sign". How would you improve it?* (Region-level embeddings for crops of detected signs, OCR text indexed alongside, a larger CLIP model, or re-ranking the top-k with the VLM.)

## 10. Hybrid edge/cloud routing
A transparent heuristic: complex or long queries go to Claude, simple ones stay local. A **circuit breaker** marks the cloud as down for 30 s after a failure, and the system **degrades gracefully** to on-device.

**Next step (a great exercise).** Make it a *learned router*. Log queries plus whether the local answer was right (judged by Claude or by you), then train a small classifier on sentence embeddings to predict "local will fail".

## 11. Attention management (when to speak)
Decision-level sensor fusion: workload = f(TTC, driver state, vulnerable road users in path).

- At **critical** workload, replies are *deferred* until the hazard clears.
- At **high** workload, replies are shortened to one sentence.
- Alerts are arbitrated by priority and rate-limited with cooldowns, because alert fatigue makes drivers switch systems off.
- Drowsiness triggers **conversation** rather than a chime.

## 12. Real-time systems engineering
- **Pipeline parallelism.** One worker thread per stage, so a 2 s VLM answer never delays a collision warning.
- **Backpressure (latest-frame-wins).** If a stage is busy, the new frame is dropped rather than queued. A queue in a real-time system just turns into latency. Watch the "dropped" counter: when the VLM is answering, YOLO and the VLM contend for the same GPU and road FPS dips. On a car you'd pin perception to a dedicated accelerator (DLA/NPU).
- **Client backpressure.** The browser skips frames when `ws.bufferedAmount` grows.
- **Binary protocol.** Raw JPEG/PCM with a 1-byte type header (base64 would add 33% overhead).

---

## Measured numbers (MacBook, Apple Silicon, 24 GB)
| Stage | Latency |
|---|---|
| YOLO11n + ByteTrack + TTC | 9–18 ms / frame |
| Face landmarks + EAR/PERCLOS/pose | 3–6 ms / frame |
| CLIP keyframe embedding | ~18 ms |
| Whisper turbo (3 s utterance) | ~0.5–0.6 s |
| Qwen2.5-VL-3B 4-bit answer | ~1.6–2.2 s (prefill ~1000 tokens, decode ~110 tok/s) |
| Speech end → answer (local) | ~2.5 s |

## Where to go next (in order of interview value)
1. **Benchmark on real datasets** (DMD / NTHU-DDD for drowsiness; DriveLM or BDD100K for road VQA) and publish a table.
2. **Cut voice latency.** Prefix-cache the system prompt, stream TTS by sentence, and try speculative decoding.
3. **Learned router** (section 10).
4. **Lane detection** (e.g. UFLD) → lane-departure warning + lane-relative ego corridor.
5. **Port perception to Jetson Orin / TensorRT** and report FPS and power.
6. **ROS 2 node wrapper** so it plugs into robotics and AV stacks.
