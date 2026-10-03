"""Streaming Voice Activity Detection (VAD) with Silero.

Concepts:

* **Why VAD at all?** Whisper is expensive and *hallucinates on silence* (it was trained
  on subtitled video, so it "hears" "Thanks for watching!" in noise). A tiny (~2 MB)
  neural VAD decides *when* someone is speaking, so we only run ASR on real utterances
  and know immediately when the user has finished talking (endpointing).
* **Streaming inference with recurrent state.** Silero processes 32 ms chunks (512 samples
  at 16 kHz) and carries an internal state between calls. That's why it is O(1) per chunk
  and why we must reset it between utterances.
* **Hysteresis + hangover.** Start speech at p > 0.5, but only end it after p < 0.35 for
  ~600 ms, so natural pauses between words don't cut the sentence in half.
* **Pre-roll buffer.** We keep ~300 ms of audio from *before* speech was detected, since
  the VAD fires slightly late and otherwise the first syllable gets clipped.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

CHUNK = 512  # samples @ 16 kHz = 32 ms


class StreamingVAD:
    def __init__(
        self,
        sample_rate: int = 16_000,
        start_threshold: float = 0.5,
        end_threshold: float = 0.35,
        min_silence_ms: int = 600,
        preroll_ms: int = 300,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15.0,
    ) -> None:
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self.model = load_silero_vad()
        self.sr = sample_rate
        self.start_th = start_threshold
        self.end_th = end_threshold
        self.silence_chunks = int(min_silence_ms / 32)
        self.min_speech_chunks = int(min_speech_ms / 32)
        self.max_chunks = int(max_utterance_s * 1000 / 32)
        self.preroll: deque[np.ndarray] = deque(maxlen=int(preroll_ms / 32))
        self.pending = np.zeros(0, dtype=np.float32)
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self.speaking = False
        self.buf: list[np.ndarray] = []
        self.silent_run = 0
        self.speech_chunks = 0

    def reset(self) -> None:
        self.model.reset_states()
        self.preroll.clear()
        self.pending = np.zeros(0, dtype=np.float32)
        self._reset_utterance()

    def feed(self, pcm16: bytes) -> list[dict]:
        """Feed raw little-endian int16 mono PCM. Returns a list of events:
        {"type": "speech_start"} / {"type": "utterance", "audio": float32 array}."""
        events: list[dict] = []
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, audio])
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            p = float(self.model(torch.from_numpy(chunk), self.sr).item())
            if not self.speaking:
                self.preroll.append(chunk)
                if p > self.start_th:
                    self.speaking = True
                    self.buf = list(self.preroll)
                    self.preroll.clear()
                    events.append({"type": "speech_start"})
                continue
            self.buf.append(chunk)
            self.speech_chunks += p > self.end_th
            self.silent_run = self.silent_run + 1 if p < self.end_th else 0
            if self.silent_run >= self.silence_chunks or len(self.buf) >= self.max_chunks:
                if self.speech_chunks >= self.min_speech_chunks:
                    events.append({"type": "utterance", "audio": np.concatenate(self.buf)})
                else:
                    events.append({"type": "speech_cancel"})
                self.model.reset_states()
                self._reset_utterance()
        return events

    def flush(self) -> np.ndarray | None:
        """Force-end the current utterance (used by push-to-talk release)."""
        audio = np.concatenate(self.buf) if self.buf else None
        if self.pending.size:
            audio = self.pending if audio is None else np.concatenate([audio, self.pending])
        self.reset()
        return audio
