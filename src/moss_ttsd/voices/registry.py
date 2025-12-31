from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceSpec:
    name: str
    audio_path: Path
    transcription_path: Path
    transcription_text: str


class VoiceRegistry:
    """
    Voice directory convention:
      voices_dir/
        <voice_name>/
          audio.wav
          transcription.txt   (text)  [extension is not strictly enforced]
    """

    def __init__(self, voices_dir: str | Path | None) -> None:
        self.voices_dir = Path(voices_dir) if voices_dir else None

    def resolve(self, voice_name: str) -> Optional[VoiceSpec]:
        if self.voices_dir is None:
            return None

        base = self.voices_dir / voice_name
        if not base.exists() or not base.is_dir():
            return None

        audio_path = base / "audio.wav"
        if not audio_path.exists():
            logger.warning("Voice '%s' missing file: %s", voice_name, audio_path)
            return None

        transcription_path = self._find_transcription_file(base)
        if transcription_path is None:
            logger.warning("Voice '%s' missing transcription file in: %s", voice_name, base)
            return None

        transcription_text = transcription_path.read_text(encoding="utf-8", errors="ignore").strip()
        return VoiceSpec(
            name=voice_name,
            audio_path=audio_path,
            transcription_path=transcription_path,
            transcription_text=transcription_text,
        )

    @staticmethod
    def _find_transcription_file(voice_dir: Path) -> Optional[Path]:
        # The user asked for "transcription.wav" (likely a typo); accept common variants.
        for filename in ("transcription.txt", "transcription.wav", "transcription", "prompt.txt", "text.txt"):
            path = voice_dir / filename
            if path.exists() and path.is_file():
                return path
        return None

