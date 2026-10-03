from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image

from drivemind.brain.tools import CarState

SYSTEM_PROMPT = """You are DriveMind, the voice assistant built into this car. You can see the road through the front camera (the attached image is the current view) and you control vehicle functions through tools.

How to respond:
- Your reply is spoken aloud to someone who is driving. Use one or two short, natural sentences. No markdown, lists, or emoji.
- Ground visual answers in the image and the perception summary. If something isn't clearly visible, say so; never invent details like sign text you can't read.
- For questions about something already passed ("what did that sign say?"), search visual memory with recall_scene.
- When the user asks for a vehicle function, call the tool and then confirm briefly.
- You are not a safety system. If asked whether it is safe to turn, merge, or proceed, describe what you see but tell the driver to check for themselves.
- If the driver workload is high, answer in under ten words."""


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
