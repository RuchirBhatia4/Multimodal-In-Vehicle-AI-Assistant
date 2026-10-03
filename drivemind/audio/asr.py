"""On-device speech recognition: Whisper large-v3-turbo on Apple MLX.

Concepts:

* **Encoder-decoder transformer ASR.** Whisper turns 30 s of audio into a log-Mel
  spectrogram, encodes it, and autoregressively decodes text tokens.
* **"Turbo" = decoder pruning.** large-v3-turbo keeps the 32-layer encoder but cuts the
  decoder from 32 layers to 4. Decoding is the sequential (slow) part, so this is ~8x
  faster with a small accuracy cost. It's a nice example of putting compute where it counts.
* **MLX + unified memory.** Apple Silicon's CPU and GPU share RAM, so MLX never copies
  tensors between devices. That is a big deal for latency on edge hardware.
* **Contextual biasing via `initial_prompt`.** Feeding domain words ("defrost", "navigate",
  street names) as a fake previous transcript nudges the decoder toward that vocabulary.
* **Hallucination guards.** We drop segments Whisper itself flags as likely non-speech
  (`no_speech_prob`) and a blocklist of well-known silence hallucinations.
"""

from __future__ import annotations

import time

import numpy as np

from drivemind.config import settings

DOMAIN_PROMPT = (
    "In-car voice assistant. Commands like: set the temperature to 70, turn on the defroster, "
    "navigate home, play music, what does that sign say, how many cars are ahead, is the light green."
)
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", ".",
    "thanks for watching.", "subtitles by the amara.org community",
}


class SpeechRecognizer:
    def __init__(self) -> None:
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        # Warm-up: the first call compiles kernels and loads weights (seconds). Do it at
        # startup so the first real utterance is fast.
        self.transcribe(np.zeros(16_000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> dict:
        t0 = time.perf_counter()
        # Peak-normalize quiet mics; Whisper is fairly robust to gain, but very quiet input hurts.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if 0 < peak < 0.3:
            audio = audio * (0.3 / peak)
        out = self._mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=settings.whisper_model,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=DOMAIN_PROMPT,
            no_speech_threshold=0.6,
            verbose=None,
        )
        segs = [s for s in out.get("segments", []) if s.get("no_speech_prob", 0) < 0.6]
        text = " ".join(s["text"].strip() for s in segs).strip()
        if text.lower() in HALLUCINATIONS:
            text = ""
        return {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1)}
