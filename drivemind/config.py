"""Central configuration. Every knob can be overridden with an environment variable
(or a .env file) so experiments don't require code changes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Road perception -------------------------------------------------
    yolo_model: str = _env("DM_YOLO_MODEL", str(ROOT / "models" / "yolo11n.pt"))
    yolo_conf: float = float(_env("DM_YOLO_CONF", "0.35"))
    yolo_imgsz: int = int(_env("DM_YOLO_IMGSZ", "640"))
    ttc_warn_s: float = float(_env("DM_TTC_WARN", "3.0"))
    ttc_critical_s: float = float(_env("DM_TTC_CRITICAL", "1.6"))

    # --- Driver monitoring -------------------------------------------------
    face_model_path: Path = ROOT / "models" / "face_landmarker.task"
    perclos_window_s: float = float(_env("DM_PERCLOS_WINDOW", "20"))
    distraction_s: float = float(_env("DM_DISTRACTION_S", "2.0"))

    # --- Audio ---------------------------------------------------------------
    sample_rate: int = 16_000
    whisper_model: str = _env("DM_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

    # --- Brains --------------------------------------------------------------
    local_vlm: str = _env("DM_LOCAL_VLM", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    claude_model: str = _env("DM_CLAUDE_MODEL", "claude-opus-5-5")
    claude_effort: str = _env("DM_CLAUDE_EFFORT", "low")  # voice UX: latency matters most
    default_brain: str = _env("DM_BRAIN", "auto")  # local | claude | auto
    vlm_image_width: int = int(_env("DM_VLM_IMAGE_WIDTH", "448"))

    # --- Scene memory --------------------------------------------------------
    clip_model: str = _env("DM_CLIP_MODEL", "clip-ViT-B-32")
    memory_horizon_s: float = float(_env("DM_MEMORY_HORIZON", "180"))
    keyframe_min_gap_s: float = float(_env("DM_KEYFRAME_GAP", "0.75"))
    keyframe_change_threshold: float = float(_env("DM_KEYFRAME_CHANGE", "0.10"))

    host: str = _env("DM_HOST", "127.0.0.1")
    port: int = int(_env("DM_PORT", "8000"))
    web_dir: Path = field(default=ROOT / "web")


settings = Settings()
