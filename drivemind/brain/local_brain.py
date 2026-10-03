"""On-device brain: Qwen2.5-VL-3B, 4-bit quantized, running on Apple MLX.

Concepts:

* **Vision-Language Model (VLM) architecture.** A ViT image encoder turns the frame into
  patch embeddings, a projector maps them into the LLM's token space, and the LLM attends
  over image tokens and text tokens together. "Seeing" is just more tokens in the context.
* **4-bit weight quantization.** 3B params x 16 bits = 6 GB. At 4 bits (group-wise affine
  quantization: each group of 64 weights shares a scale + bias) that's ~2 GB, with a small
  quality loss. Decoding is memory-bandwidth-bound, so fewer bytes per weight is *directly*
  more tokens/sec. This is the single most important trick for edge LLMs.
* **Prefill vs. decode.** Prefill processes the whole prompt (incl. image tokens) in
  parallel and is compute-bound; decode generates one token at a time and is
  bandwidth-bound. We report both speeds so you can see which one dominates latency.
* **Prompted tool calling + constrained output.** Small local models don't have a native
  tool API, so we describe tools in the prompt and ask for JSON, then parse defensively.
  (Production systems use grammar-constrained decoding to *guarantee* valid JSON.)
"""

from __future__ import annotations

import json
import re
import threading
import time

from drivemind.brain.common import SYSTEM_PROMPT, BrainContext, BrainResult, downscale_jpeg
from drivemind.brain.tools import TOOL_NAMES, execute_vehicle_tool, tools_prompt_block
from drivemind.config import settings

JSON_INSTRUCTIONS = """Available tools:
{tools}

Respond with ONLY one JSON object, no other text:
{{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}], "say": "<what to say aloud>"}}
Use an empty tool_calls list when no tool is needed. If you call recall_scene, leave "say" empty.
A request can need several tools: include one entry per action.

Examples:
Driver says: "I'm freezing, warm it up and put on some Taylor Swift"
{{"tool_calls": [{{"name": "set_temperature", "args": {{"temperature_f": 74}}}}, {{"name": "play_media", "args": {{"query": "Taylor Swift"}}}}], "say": "Warming it up to 74 and playing Taylor Swift."}}
Driver says: "What did that sign say?"
{{"tool_calls": [{{"name": "recall_scene", "args": {{"query": "road sign with text"}}}}], "say": ""}}
Driver says: "Is the light green?"
{{"tool_calls": [], "say": "Yes, the light ahead is green."}}"""


def _extract_json(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LocalBrain:
    name = "local"

    def __init__(self, mlx_lock: threading.Lock) -> None:
        from mlx_vlm import load

        self.lock = mlx_lock  # MLX work (Whisper + VLM) is serialized on the GPU
        self.model, self.processor = load(settings.local_vlm)
        self.config = self.model.config

    def _generate(self, system: str, user: str, image, max_tokens: int) -> tuple[str, dict]:
        from mlx_vlm import apply_chat_template, generate

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = apply_chat_template(self.processor, self.config, messages, num_images=1 if image else 0)
        with self.lock:
            out = generate(
                self.model, self.processor, prompt, image=[image] if image else None,
                max_tokens=max_tokens, temperature=0.0, verbose=False,
            )
        stats = {
            "prompt_tokens": out.prompt_tokens,
            "generation_tokens": out.generation_tokens,
            "prefill_tps": round(out.prompt_tps, 1),
            "decode_tps": round(out.generation_tps, 1),
            "peak_mem_gb": round(out.peak_memory, 2),
        }
        return out.text, stats

    def answer(self, ctx: BrainContext, memory=None) -> BrainResult:
        t0 = time.perf_counter()
        image = downscale_jpeg(ctx.road_jpeg, settings.vlm_image_width) if ctx.road_jpeg else None
        system = SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTIONS.format(tools=tools_prompt_block())
        user = f"{ctx.context_block()}\n\nDriver says: \"{ctx.query}\""
        raw, stats = self._generate(system, user, image, max_tokens=160)
        parsed = _extract_json(raw)

        calls: list[dict] = []
        retrieved: list[dict] = []
        if parsed is None:  # model ignored the format: treat output as plain speech
            say = raw.strip().strip('"')
            parsed = {"tool_calls": [], "say": say}
        say = str(parsed.get("say") or "").strip()

        for call in parsed.get("tool_calls") or []:
            name, args = call.get("name"), call.get("args") or {}
            if name not in TOOL_NAMES:
                continue
            if name == "recall_scene":
                if memory is None:
                    continue
                hits = memory.search(str(args.get("query") or ctx.query), k=1)
                calls.append({"name": name, "args": args, "result": f"{len(hits)} frame(s)"})
                if hits:
                    hit = hits[0]
                    retrieved.append({k: hit[k] for k in ("seconds_ago", "similarity")})
                    # Second pass: answer from the retrieved past frame (RAG over video).
                    past = downscale_jpeg(hit["jpeg"], settings.vlm_image_width * 2)  # more pixels to read text
                    q = (
                        f"This image is the road camera view from {hit['seconds_ago']:.0f} seconds ago. "
                        f"Answer the driver's question in one short spoken sentence: \"{ctx.query}\""
                    )
                    say, stats2 = self._generate(SYSTEM_PROMPT, q, past, max_tokens=80)
                    stats["second_pass"] = stats2
                else:
                    say = "I don't have anything in my recent visual memory that matches."
                continue
            result = execute_vehicle_tool(ctx.car, name, args)
            calls.append({"name": name, "args": args, "result": result})
            if not say:
                say = result

        # The safety envelope has the final word: if a tool clamped/blocked/failed, say what
        # actually happened instead of what the model *predicted* would happen.
        notes = [c["result"] for c in calls if any(w in c["result"] for w in ("blocked", "outside", "Error"))]
        if notes:
            say = " ".join(notes)
        if not say:
            say = "Sorry, I didn't catch that."
        return BrainResult(
            text=say.strip(), brain=self.name, tool_calls=calls, retrieved=retrieved,
            ms=round((time.perf_counter() - t0) * 1000, 1), detail={"raw": raw[:400], **stats},
        )
