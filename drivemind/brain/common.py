from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image

from drivemind.brain.tools import CarState

SYSTEM_PROMPT = """You are DriveMind, the voice assistant built into this car. You can see the road through the front camera (the attached image is the current view) and you control vehicle functions through tools.

How to respond:
- Your reply is spoken aloud to someone who is driving. Use one or two short, natural sentences. No markdown, lists, or emoji.
- Ground visual answers in the image and the perception summary. If something isn't clearly visible, say so; never invent details like sign text you can't read.
- Visual memory is only for things that are no longer visible ("what did that sign say?"). Anything about what is visible now is answered from the current image.
- When the user asks for a vehicle function, call the tool and then confirm briefly.
- You are not a safety system. If asked whether it is safe to turn, merge, or proceed, describe what you see but tell the driver to check for themselves.
- If the driver workload is high, answer in under ten words."""


# Deciding when to search visual memory is a rule, not the model's call. The on-device 3B
# model searched memory for "Which car is on my left?" in a live demo (answering from a frame
# 17 s old) and skipped it for "Which car was on my left a moment ago?" in
# eval/brain_qa_eval.py. Memory is for questions that point to the past AND to something visible.
_PAST = re.compile(
    r"\b(did|was|were|passed|went by|drove|drive past|driving past|earlier|ago|back there|behind us|"
    r"previous|just saw|missed|said|saw|showed|remind me|last (?:sign|light|exit|car|truck|one|turn|street|building|store))\b",
    re.I,
)
_VISUAL = re.compile(
    r"\b(signs?|lights?|cars?|trucks?|bus|vans?|bikes?|bicycles?|motorcycles?|person|people|pedestrians?|"
    r"buildings?|stores?|shops?|restaurants?|exits?|streets?|roads?|intersections?|billboards?|plates?|colou?r|"
    r"written|see|saw|pass|passed|past|went by|drove by)\b",
    re.I,
)


def memory_intent(query: str) -> bool:
    """Should this question be answered from visual memory (a frame from the last few minutes)?"""
    return bool(_PAST.search(query) and _VISUAL.search(query))


# Car functions go through the tool prompt; everything else gets a plain "look and answer"
# prompt. Mixing them failed both ways on the 3B model: with example answers in the prompt it
# copied them ("Is the light green?" -> "Yes, the light ahead is green." at a RED light), and
# without them it called tools for plain questions ("Navigating to the red light").
_COMMAND = re.compile(
    r"\b(temperature|degrees|heat|heater|heating|a/?c|air ?con\w*|fan|defrost\w*|navigate|navigation|"
    r"directions|take me|drive me|route to|play|music|songs?|podcast|radio|volume|louder|quieter|mute|"
    r"windows?|seat heaters?|heated seats?|seat warmers?|wipers?|cold|freezing|chilly|warm (?:it|me)|"
    r"cool (?:it|me)|too hot|hot in here|stuffy)\b",
    re.I,
)
# Go/no-go judgments are the driver's call: the assistant describes, never decides.
_SAFETY = re.compile(
    r"\b(?:safe|ok|okay|clear) to\b|\bis it (?:safe|clear)\b|\bgood to go\b|"
    r"\b(?:can|should|could) i (?:go|turn|merge|pass|cross|change lanes|pull out|proceed)\b",
    re.I,
)


def command_intent(query: str) -> bool:
    return bool(_COMMAND.search(query))


def safety_intent(query: str) -> bool:
    return bool(_SAFETY.search(query))


@dataclass
class BrainContext:
    query: str
    road_jpeg: bytes | None
    road_summary: str
    driver_summary: str
    car: CarState
    workload: str
    history: list[tuple[str, str]] = field(default_factory=list)

    def context_block(self) -> str:
        hist = "\n".join(f"Driver: {u}\nYou: {a}" for u, a in self.history[-4:]) or "(none)"
        c = self.car
        return (
            f"[Live context]\nRoad perception: {self.road_summary}\n{self.driver_summary}\n"
            f"Driver workload: {self.workload}. Speed: {c.speed_mph:.0f} mph.\n"
            f"Car: {c.temperature_f:g}°F, fan {c.fan_level}, defrost {'on' if c.defrost else 'off'}, "
            f"media: {c.media or 'off'}, navigation: {c.destination or 'none'}.\n"
            f"[Recent conversation]\n{hist}"
        )


@dataclass
class BrainResult:
    text: str
    brain: str
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    ms: float = 0.0
    detail: dict = field(default_factory=dict)


def downscale_jpeg(jpeg: bytes, width: int) -> Image.Image:
    """Fewer pixels -> fewer vision tokens -> faster prefill. For a ViT encoder with 14 px
    patches (and 2x2 merging in Qwen2.5-VL), tokens scale with area: 448 px wide is ~4x
    fewer tokens than 896."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    return img


def to_jpeg(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
