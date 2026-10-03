"""Vehicle functions exposed to the language model as *tools* (function calling).

Concepts:

* **Tool calling.** The LLM never touches the car directly. It emits a structured request
  ("call set_temperature with temperature_f=72"), and *our* deterministic code decides
  whether and how to execute it. The JSON Schema below is the contract.
* **Safety envelope.** Every tool validates and clamps its arguments (no 200°F, no
  opening windows at highway speed). In automotive terms the LLM is a *QM / advisory*
  component; safety is enforced by deterministic code it cannot bypass. That's the
  ISO 26262 mindset: never trust the non-deterministic part with a safety invariant.
* **Strict schemas.** `additionalProperties: false` + `required` let the API guarantee the
  arguments match the schema exactly (Claude's `strict: true`).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field


@dataclass
class CarState:
    temperature_f: float = 70.0
    fan_level: int = 2
    defrost: bool = False
    destination: str | None = None
    eta_min: int | None = None
    media: str | None = None
    volume: int = 4
    windows: dict = field(default_factory=lambda: {"driver": "closed", "passenger": "closed", "rear": "closed"})
    seat_heater: dict = field(default_factory=lambda: {"driver": 0, "passenger": 0})
    wipers: str = "off"
    speed_mph: float = 0.0  # set by the UI's simulated speed slider

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))


def _schema(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


TOOLS: list[dict] = [
    {
        "name": "set_temperature",
        "description": "Set the cabin climate target temperature in Fahrenheit (60-85). Use when the user says they are hot/cold or names a temperature.",
        "input_schema": _schema({"temperature_f": {"type": "number"}}),
    },
    {
        "name": "set_fan",
        "description": "Set the climate fan level, 0 (off) to 5 (max).",
        "input_schema": _schema({"level": {"type": "integer"}}),
    },
    {
        "name": "set_defrost",
        "description": "Turn the windshield defroster on or off. Use for foggy/icy windshields.",
        "input_schema": _schema({"on": {"type": "boolean"}}),
    },
    {
        "name": "navigate",
        "description": "Start navigation to a destination (address, place name, 'home', 'work', or 'nearest rest stop').",
        "input_schema": _schema({"destination": {"type": "string"}}),
    },
    {
        "name": "play_media",
        "description": "Play music or a podcast. query is an artist, song, genre, station or podcast name; use 'stop' to stop playback.",
        "input_schema": _schema({"query": {"type": "string"}}),
    },
    {
        "name": "set_volume",
        "description": "Set media volume 0-10.",
        "input_schema": _schema({"level": {"type": "integer"}}),
    },
    {
        "name": "set_window",
        "description": "Open, close or vent a window.",
        "input_schema": _schema(
            {
                "window": {"type": "string", "enum": ["driver", "passenger", "rear", "all"]},
                "position": {"type": "string", "enum": ["open", "closed", "vent"]},
            }
        ),
    },
    {
        "name": "set_seat_heater",
        "description": "Set a seat heater level 0 (off) to 3.",
        "input_schema": _schema(
            {"seat": {"type": "string", "enum": ["driver", "passenger"]}, "level": {"type": "integer"}}
        ),
    },
    {
        "name": "set_wipers",
        "description": "Set windshield wipers.",
        "input_schema": _schema({"mode": {"type": "string", "enum": ["off", "low", "high", "auto"]}}),
    },
    {
        "name": "recall_scene",
        "description": (
            "Search the road camera's visual memory of the last few minutes for a past moment "
            "and return the best-matching frames. Use when the user asks about something already "
            "passed, e.g. 'what did that sign say', 'what was the exit number', 'what color was that car'. "
            "query should describe what the frame looks like, e.g. 'green highway exit sign'."
        ),
        "input_schema": _schema({"query": {"type": "string"}}),
    },
]
for _t in TOOLS:
    _t["strict"] = True

TOOL_NAMES = {t["name"] for t in TOOLS}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def execute_vehicle_tool(state: CarState, name: str, args: dict) -> str:
    """Run a vehicle tool against the (simulated) car. Returns a short result string the
    model can read. Unknown/invalid calls return an error string rather than raising."""
    try:
        if name == "set_temperature":
            req = float(args["temperature_f"])
            state.temperature_f = _clamp(round(req * 2) / 2, 60, 85)
            note = "" if state.temperature_f == req else f" (requested {req:g}°F was outside 60-85°F)"
            return f"Temperature set to {state.temperature_f:g}°F{note}."
        if name == "set_fan":
            state.fan_level = int(_clamp(int(args["level"]), 0, 5))
            return f"Fan set to {state.fan_level}."
        if name == "set_defrost":
            state.defrost = bool(args["on"])
            if state.defrost:
                state.fan_level = max(state.fan_level, 4)
            return f"Defrost {'on' if state.defrost else 'off'}."
        if name == "navigate":
            dest = str(args["destination"]).strip()[:120]
            state.destination = dest
            state.eta_min = 6 + (sum(map(ord, dest)) % 30)  # simulated routing
            return f"Navigating to {dest}. ETA {state.eta_min} minutes."
        if name == "play_media":
            q = str(args["query"]).strip()[:120]
            state.media = None if q.lower() in {"stop", "pause", "off"} else q
            return "Playback stopped." if state.media is None else f"Now playing: {q}."
        if name == "set_volume":
            state.volume = int(_clamp(int(args["level"]), 0, 10))
            return f"Volume {state.volume}."
        if name == "set_window":
            pos = args["position"]
            # Safety envelope: don't fully open windows at highway speed.
            if pos == "open" and state.speed_mph > 55:
                pos = "vent"
                note = " Fully opening is blocked above 55 mph, so I vented it instead."
            else:
                note = ""
            targets = ["driver", "passenger", "rear"] if args["window"] == "all" else [args["window"]]
            for w in targets:
                state.windows[w] = pos
            return f"{', '.join(targets).capitalize()} window(s) {pos}.{note}"
        if name == "set_seat_heater":
            state.seat_heater[args["seat"]] = int(_clamp(int(args["level"]), 0, 3))
            return f"{args['seat'].capitalize()} seat heater {state.seat_heater[args['seat']]}."
        if name == "set_wipers":
            state.wipers = args["mode"]
            return f"Wipers {state.wipers}."
    except (KeyError, TypeError, ValueError) as e:
        return f"Error: invalid arguments for {name}: {e}"
    return f"Error: unknown tool {name}"


def tools_prompt_block() -> str:
    """Compact tool list for the local model, which has no native tool-calling API."""
    lines = []
    for t in TOOLS:
        props = t["input_schema"]["properties"]
        sig = ", ".join(
            f"{k}: {'|'.join(v['enum']) if 'enum' in v else v['type']}" for k, v in props.items()
        )
        lines.append(f"- {t['name']}({sig}): {t['description']}")
    return "\n".join(lines)


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))
