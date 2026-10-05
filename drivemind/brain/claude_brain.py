"""Cloud brain: Claude with vision + native tool use.

Concepts:

* **Agentic tool-use loop.** Claude returns `tool_use` blocks; we execute them and send
  back `tool_result` blocks until it answers in plain text (stop_reason == "end_turn").
  A tool result can itself contain an *image*, which is how recall_scene hands a past
  frame back to the model.
* **Effort as a latency knob.** Voice UX lives or dies on latency, so we run at low effort.
* **Stateless requests.** Each turn is a fresh request with recent history summarized in
  the context block. Within a turn the message list is append-only.
"""

from __future__ import annotations

import base64
import time

import anthropic

from drivemind.brain.common import SYSTEM_PROMPT, BrainContext, BrainResult, downscale_jpeg, memory_intent, to_jpeg
from drivemind.brain.tools import TOOLS, execute_vehicle_tool
from drivemind.config import settings

MAX_STEPS = 4


def _image_block(jpeg: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpeg).decode()},
    }


class ClaudeUnavailable(RuntimeError):
    pass


class ClaudeBrain:
    name = "claude"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(timeout=20.0, max_retries=1)

    def answer(self, ctx: BrainContext, memory=None) -> BrainResult:
        t0 = time.perf_counter()
        content: list[dict] = []
        if ctx.road_jpeg:
            frame = to_jpeg(downscale_jpeg(ctx.road_jpeg, 768))
            content.append(_image_block(frame))
        content.append({"type": "text", "text": f"{ctx.context_block()}\n\nDriver says: \"{ctx.query}\""})
        messages: list[dict] = [{"role": "user", "content": content}]

        calls: list[dict] = []
        retrieved: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        text = ""
        for _ in range(MAX_STEPS):
            try:
                resp = self.client.beta.messages.create(
                    model=settings.claude_model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                    output_config={"effort": settings.claude_effort},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise ClaudeUnavailable(f"Claude credentials missing or invalid: {e}") from e
            except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError) as e:
                raise ClaudeUnavailable(f"Claude unreachable: {e}") from e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    raise ClaudeUnavailable(f"Claude server error {e.status_code}") from e
                raise

            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens

            if resp.stop_reason == "refusal":
                text = "Sorry, I can't help with that one."
                break
            text = " ".join(b.text for b in resp.content if b.type == "text").strip()
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                break

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                if tu.name == "recall_scene" and not memory_intent(ctx.query):
                    # Same rule as the on-device brain: memory only for questions about the past.
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content":
                                    "Not searched: visual memory is only for things already passed, and this "
                                    "question is about what is visible now. Answer from the current camera image."})
                elif tu.name == "recall_scene":
                    hits = memory.search(tu.input.get("query", ctx.query), k=2) if memory else []
                    block: list[dict] = []
                    for h in hits:
                        retrieved.append({k: h[k] for k in ("seconds_ago", "similarity")})
                        block.append({"type": "text", "text": f"Frame from {h['seconds_ago']:.0f}s ago (similarity {h['similarity']:.2f}). Detections then: {h['summary']}"})
                        block.append(_image_block(to_jpeg(downscale_jpeg(h["jpeg"], 1024))))
                    if not block:
                        block = [{"type": "text", "text": "Visual memory is empty."}]
                    calls.append({"name": tu.name, "args": tu.input, "result": f"{len(hits)} frame(s)"})
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": block})
                else:
                    out = execute_vehicle_tool(ctx.car, tu.name, tu.input)
                    calls.append({"name": tu.name, "args": tu.input, "result": out})
                    results.append(
                        {"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": out.startswith("Error")}
                    )
            messages.append({"role": "user", "content": results})

        return BrainResult(
            text=text or "Done.", brain=self.name, tool_calls=calls, retrieved=retrieved,
            ms=round((time.perf_counter() - t0) * 1000, 1), detail={"model": settings.claude_model, **usage},
        )
