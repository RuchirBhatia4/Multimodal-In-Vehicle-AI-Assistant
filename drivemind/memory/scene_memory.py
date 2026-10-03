"""Short-term visual memory: "What did that sign say?", asked after you've passed it.

Concepts:

* **Contrastive vision-language embeddings (CLIP).** CLIP was trained on 400M image-caption
  pairs to put an image and its description at nearby points in the *same* vector space.
  So we can embed frames once and later search them with a *text* query, with no captioning
  and no labels. This is zero-shot retrieval.
* **Semantic keyframe selection.** Storing every frame is wasteful (consecutive frames
  are near-identical). We store a new keyframe only when the scene *meaning* changes:
  cosine distance between consecutive CLIP embeddings exceeds a threshold. That's
  smarter than pixel differences, which fire on lighting changes and camera shake.
* **Retrieval-Augmented Generation (RAG) over video.** The retrieved frame(s) get passed
  to the VLM as extra context, exactly like RAG passes retrieved documents to an LLM.
* **Recency-weighted scoring.** score = cosine_similarity + λ·recency. When you ask
  "what was that sign", you usually mean the *most recent* matching one.
* **Bounded memory.** A time-based ring buffer: anything older than the horizon is evicted,
  so memory use is constant no matter how long you drive.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from drivemind.config import settings


@dataclass
class Keyframe:
    t: float
    wall_time: float
    jpeg: bytes
    embedding: np.ndarray
    summary: str


class SceneMemory:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clip = SentenceTransformer(settings.clip_model, device=device)
        self.frames: deque[Keyframe] = deque()
        self._last_emb: np.ndarray | None = None
        self._last_t = 0.0
        self._lock = threading.Lock()

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        return self.clip.encode([img], normalize_embeddings=True, show_progress_bar=False)[0]

    def _embed_text(self, text: str) -> np.ndarray:
        return self.clip.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

    def observe(self, jpeg: bytes, summary: str, t: float | None = None) -> dict:
        """Consider a road frame for storage. Returns {'stored': bool, 'change': float}."""
        t = time.monotonic() if t is None else t
        if t - self._last_t < settings.keyframe_min_gap_s:
            return {"stored": False}
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        emb = self._embed_image(img)
        change = 1.0 if self._last_emb is None else float(1.0 - emb @ self._last_emb)
        stale = t - self._last_t > 5.0  # always keep at least one frame every 5 s
        stored = change > settings.keyframe_change_threshold or stale
        if stored:
            with self._lock:
                self.frames.append(Keyframe(t, time.time(), jpeg, emb, summary))
                while self.frames and t - self.frames[0].t > settings.memory_horizon_s:
                    self.frames.popleft()
            self._last_emb, self._last_t = emb, t
        return {
            "stored": stored,
            "change": round(change, 3),
            "size": len(self.frames),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def search(self, query: str, k: int = 2, recency_weight: float = 0.05) -> list[dict]:
        with self._lock:
            frames = list(self.frames)
        if not frames:
            return []
        q = self._embed_text(query)
        now = time.monotonic()
        embs = np.stack([f.embedding for f in frames])
        sims = embs @ q
        ages = np.array([now - f.t for f in frames])
        recency = 1.0 - np.clip(ages / settings.memory_horizon_s, 0, 1)
        scores = sims + recency_weight * recency
        order = np.argsort(-scores)[:k]
        return [
            {
                "jpeg": frames[i].jpeg,
                "seconds_ago": round(float(ages[i]), 1),
                "similarity": round(float(sims[i]), 3),
                "summary": frames[i].summary,
            }
            for i in order
        ]

    def clear(self) -> None:
        with self._lock:
            self.frames.clear()
        self._last_emb = None
        self._last_t = 0.0
