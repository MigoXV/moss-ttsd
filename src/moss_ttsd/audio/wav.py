from __future__ import annotations

import io
import math
import wave
from typing import Optional

import numpy as np


def _to_time_channels(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[:, None]

    if audio.ndim != 2:
        raise ValueError(f"Expected 1D/2D audio, got shape={audio.shape}")

    # Heuristic: common ML format is (channels, time); wav expects (time, channels).
    channels_first = audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]
    return audio.T if channels_first else audio


def pcm16_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode audio into 16-bit PCM WAV bytes.

    Accepts audio shaped as:
    - (time,)
    - (channels, time)
    - (time, channels)
    """
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate: {sample_rate}")

    audio = np.asarray(audio)
    if audio.size == 0:
        audio = np.zeros((0,), dtype=np.float32)

    time_channels = _to_time_channels(audio).astype(np.float32, copy=False)
    time_channels = np.clip(time_channels, -1.0, 1.0)
    pcm16 = (time_channels * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(pcm16.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())

    return buffer.getvalue()


def sine_wav_bytes(
    *,
    sample_rate: int = 24000,
    duration_s: float = 0.5,
    freq_hz: float = 440.0,
    amplitude: float = 0.08,
    fade_ms: float = 10.0,
) -> bytes:
    if duration_s <= 0:
        duration_s = 0.1
    if sample_rate <= 0:
        sample_rate = 24000
    if freq_hz <= 0:
        freq_hz = 440.0

    num_samples = int(sample_rate * duration_s)
    if num_samples <= 0:
        num_samples = max(1, int(sample_rate * 0.1))

    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    audio = amplitude * np.sin(2.0 * math.pi * float(freq_hz) * t)

    fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if fade_samples > 0 and fade_samples * 2 < num_samples:
        fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[:fade_samples] *= fade
        audio[-fade_samples:] *= fade[::-1]

    return pcm16_wav_bytes(audio, sample_rate)
