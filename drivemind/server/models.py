"""Process-wide model registry. Heavy models load once, in the background, so the UI is
usable immediately and lights up each capability as it becomes ready."""

from __future__ import annotations

import logging
import threading
import time
import traceback

log = logging.getLogger("drivemind.models")


class Models:
    def __init__(self) -> None:
        self.mlx_lock = threading.Lock()
        self.asr = None
        self.memory = None
        self.local_brain = None
        self.claude_brain = None
        self.router = None
        self.status: dict[str, str] = {
            "road": "pending", "asr": "pending", "memory": "pending", "local_brain": "pending", "claude": "pending",
        }
        self.load_times: dict[str, float] = {}
        self.listeners: list = []

    def _set(self, key: str, value: str) -> None:
        self.status[key] = value
        for cb in list(self.listeners):
            try:
                cb(dict(self.status))
            except Exception:  # noqa: BLE001 - a dead listener must not break loading
                pass

    def _load(self, key: str, fn) -> None:
        self._set(key, "loading")
        t0 = time.perf_counter()
        try:
            fn()
            self.load_times[key] = round(time.perf_counter() - t0, 1)
            self._set(key, "ready")
            log.info("%s ready in %.1fs", key, self.load_times[key])
        except Exception as e:  # noqa: BLE001
            log.error("%s failed: %s\n%s", key, e, traceback.format_exc())
            self._set(key, f"error: {str(e)[:120]}")

    def load_all(self) -> None:
        from drivemind.brain.router import BrainRouter

        def asr():
            from drivemind.audio.asr import SpeechRecognizer
            with self.mlx_lock:
                self.asr = SpeechRecognizer()

        def memory():
            from drivemind.memory.scene_memory import SceneMemory
            self.memory = SceneMemory()

        def local():
            from drivemind.brain.local_brain import LocalBrain
            self.local_brain = LocalBrain(self.mlx_lock)

        def claude():
            import anthropic

            from drivemind.brain.claude_brain import ClaudeBrain
            brain = ClaudeBrain()
            try:
                brain.client.models.retrieve(brain_model())  # cheap credential + model check
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, TypeError) as e:
                raise RuntimeError("no Anthropic credentials (set ANTHROPIC_API_KEY in .env)") from e
            self.claude_brain = brain

        def road():
            # Warm-up: import torch/ultralytics and compile the GPU kernels once at startup, so
            # the first session doesn't drive "blind" for its first ~1-2 s (measured: the first
            # frame of the first session took ~640 ms, longer still on a fresh install).
            import numpy as np

            from drivemind.perception.road import RoadPerception
            # GPU kernels are compiled per input shape: warm the two shapes the dashboard sends
            # (16:9 dashcam video and 4:3 webcams, both scaled to 640 px wide).
            rp = RoadPerception()
            for h in (360, 480):
                rp.process(np.zeros((h, 640, 3), np.uint8))

        def brain_model():
            from drivemind.config import settings
            return settings.claude_model

        # Import the heavy libraries *serially* before any loader threads start. transformers
        # resolves its submodules lazily, and two threads importing it at once can see a
        # half-initialized module ("cannot import name 'AutoProcessor'"), an intermittent race.
        import mlx_vlm  # noqa: F401
        import mlx_whisper  # noqa: F401
        import sentence_transformers  # noqa: F401
        from transformers import AutoProcessor  # noqa: F401
        import ultralytics  # noqa: F401

        # Claude check, CLIP and the road warm-up are independent of the MLX models: run them in parallel.
        side = [threading.Thread(target=self._load, args=(k, f), daemon=True)
                for k, f in (("claude", claude), ("memory", memory), ("road", road))]
        for th in side:
            th.start()
        self._load("asr", asr)
        self._load("local_brain", local)
        for th in side:
            th.join()
        self.router = BrainRouter(self.local_brain, self.claude_brain)
        self._set("router", "ready")


models = Models()
