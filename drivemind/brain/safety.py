"""Attention management: deciding *when* to talk, not just *what* to say.

Concepts:

* **Driver workload estimation.** Fuse road perception (time-to-collision) with driver
  state into a single workload level. This is *sensor fusion at the decision level*.
* **Workload-aware gating.** When the road is demanding, a chatty assistant is itself a
  hazard. Answers are deferred (held until the situation clears) or shortened. NHTSA's
  driver-distraction guidelines and the "15-second rule" for in-vehicle UIs are
  the motivation here.
* **Alert arbitration + rate limiting.** Safety alerts preempt everything; each alert type
  has a cooldown so the driver isn't nagged into turning the system off.
* **Proactive engagement.** For drowsiness, *conversation* re-engages the brain better
  than a beep, so the assistant asks a question rather than just chiming.
"""

from __future__ import annotations

import random
import time

ENGAGE_PROMPTS = [
    "You seem tired. Want me to find the nearest rest stop?",
    "Hey, your eyes have been closing a lot. How about a coffee break? I can navigate to one.",
    "I'm noticing signs of fatigue. Want me to turn the temperature down and play something upbeat?",
]


class AttentionManager:
    def __init__(self) -> None:
        self.cooldowns = {"fcw": 2.5, "microsleep": 3.0, "distracted": 6.0, "drowsy": 60.0}
        self.last_fired: dict[str, float] = {}
        self.deferred: list[dict] = []

    def workload(self, road: dict | None, driver: dict | None) -> str:
        fcw = (road or {}).get("fcw", "none")
        dstate = (driver or {}).get("state")
        if fcw == "critical" or dstate == "microsleep":
            return "critical"
        if fcw == "warning" or dstate == "distracted":
            return "high"
        n_vulnerable = sum(
            1 for t in (road or {}).get("tracks", []) if t["label"] in {"person", "bicycle"} and t.get("in_path")
        )
        return "high" if n_vulnerable else "low"

    def _ready(self, kind: str, now: float) -> bool:
        if now - self.last_fired.get(kind, -1e9) < self.cooldowns[kind]:
            return False
        self.last_fired[kind] = now
        return True

    def proactive_alerts(self, road: dict | None, driver: dict | None) -> list[dict]:
        now = time.monotonic()
        alerts: list[dict] = []
        if (road or {}).get("fcw") == "critical" and self._ready("fcw", now):
            alerts.append({"level": "critical", "kind": "fcw", "text": "Brake! Vehicle ahead."})
        dstate = (driver or {}).get("state")
        if dstate == "microsleep" and self._ready("microsleep", now):
            alerts.append({"level": "critical", "kind": "microsleep", "text": "Wake up! Eyes on the road."})
        elif dstate == "distracted" and (road or {}).get("tracks") and self._ready("distracted", now):
            alerts.append({"level": "warning", "kind": "distracted", "text": "Eyes on the road, please."})
        elif dstate == "drowsy" and self._ready("drowsy", now):
            alerts.append({"level": "engage", "kind": "drowsy", "text": random.choice(ENGAGE_PROMPTS)})
        return alerts

    def gate_reply(self, reply: dict, workload: str) -> dict | None:
        """Returns the reply to deliver now (possibly shortened), or None if deferred."""
        if workload == "critical":
            self.deferred.append(reply)
            return None
        if workload == "high" and reply.get("text"):
            first = reply["text"].split(". ")[0].rstrip(".") + "."
            if first != reply["text"]:
                reply = {**reply, "text": first, "shortened": True}
        return reply

    def release_deferred(self, workload: str) -> list[dict]:
        if workload == "critical" or not self.deferred:
            return []
        out, self.deferred = self.deferred, []
        return [{**r, "deferred": True} for r in out]
