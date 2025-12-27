from __future__ import annotations

import logging
from typing import Optional

from moss_ttsd.inferencers import MossTTSDInferencer

logger = logging.getLogger(__name__)

_inferencer_instance: Optional[MossTTSDInferencer] = None


def get_inferencer() -> Optional[MossTTSDInferencer]:
    return _inferencer_instance


def initialize_inferencer(
    model_dir: str = "./model-bin/MOSS-TTSD-v0.7",
    *,
    codec_path: Optional[str] = None,
    voices_dir: Optional[str] = None,
    device: str = "cpu",
    dtype=None,
    max_new_tokens: int = 4096,
    fallback_audio: str = "error",
) -> Optional[MossTTSDInferencer]:
    global _inferencer_instance
    logger.info("Loading MOSS-TTSD inferencer")
    logger.info("  - model_dir      : %s", model_dir)
    logger.info("  - codec_path     : %s", codec_path)
    logger.info("  - voices_dir     : %s", voices_dir)
    logger.info("  - device         : %s", device)
    logger.info("  - dtype          : %s", dtype)
    logger.info("  - max_new_tokens : %s", max_new_tokens)
    logger.info("  - fallback_audio : %s", fallback_audio)

    _inferencer_instance = MossTTSDInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=voices_dir,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        fallback_audio=fallback_audio,
    )
    logger.info("✓ Inferencer initialized")
    return _inferencer_instance
