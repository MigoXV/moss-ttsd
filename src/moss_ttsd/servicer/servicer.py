from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from moss_ttsd.audio import pcm16_wav_bytes
from moss_ttsd.inferencers import MossTTSDInferencer
from moss_ttsd.inferencers.inferencer import PretrainedVoice

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model identifier")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field("default", description="Voice name")
    response_format: str = Field("wav", description="Only 'wav' is supported currently")
    stream: bool = Field(False, description="Enable streaming response")


def _openai_error(status_code: int, message: str, *, param: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


def _load_pretrained_voice_from_npz(npz_path: Path) -> PretrainedVoice:
    """从 .npz 文件加载预训练音色"""
    data = np.load(npz_path, allow_pickle=True)
    audio_codes = data["audio_codes"]
    metadata = data["metadata"].item()

    return PretrainedVoice(
        name=metadata["name"],
        prompt_text=metadata["prompt_text"],
        audio_codes=audio_codes,
        sample_rate=metadata["sample_rate"],
    )


def create_app(
    get_inferencer_func: Callable[[], Optional[MossTTSDInferencer]],
    voices_cache_dir: str | Path | None = None,
) -> FastAPI:
    """
    创建 FastAPI 应用。
    
    Args:
        get_inferencer_func: 获取 inferencer 实例的函数
        voices_cache_dir: 预训练音色缓存目录，包含 .npz 文件
    
    注意：所有端点使用同步函数，FastAPI 会自动在线程池中运行它们，
    这对于 CPU 密集型的深度学习推理任务更合适。
    """
    app = FastAPI(title="MOSS-TTSD OpenAI TTS API")

    # 预加载音色缓存
    _voice_cache: Dict[str, PretrainedVoice] = {}
    
    if voices_cache_dir is not None:
        cache_dir = Path(voices_cache_dir)
        if cache_dir.exists():
            for npz_file in cache_dir.glob("*.npz"):
                try:
                    voice = _load_pretrained_voice_from_npz(npz_file)
                    _voice_cache[voice.name] = voice
                    logger.info("Loaded pretrained voice: %s from %s", voice.name, npz_file)
                except Exception as exc:
                    logger.warning("Failed to load voice from %s: %s", npz_file, exc)
        else:
            logger.warning("Voices cache directory not found: %s", cache_dir)

    def _get_pretrained_voice(voice: str) -> PretrainedVoice:
        """获取预训练音色"""
        if voice not in _voice_cache:
            available = list(_voice_cache.keys())
            raise ValueError(f"Voice '{voice}' not found. Available voices: {available}")
        return _voice_cache[voice]

    def _audio_speech_impl(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer],
    ):
        if inferencer is None:
            raise HTTPException(status_code=500, detail="Inferencer not initialized")

        response_format = (request.response_format or "wav").lower()
        if response_format != "wav":
            return _openai_error(400, f"Unsupported response_format: {response_format}", param="response_format")

        # 获取预训练音色
        try:
            pretrained_voice = _get_pretrained_voice(request.voice)
        except ValueError as exc:
            return _openai_error(400, str(exc), param="voice")

        # 流式响应
        if request.stream:
            def stream_generator() -> Iterator[bytes]:
                try:
                    header_sent = False
                    for audio_chunk, sample_rate in inferencer.tts_pretrained_stream(
                        request.input,
                        pretrained_voice,
                    ):
                        # 第一个 chunk 带 WAV header
                        wav_bytes = pcm16_wav_bytes(
                            audio_chunk,
                            sample_rate,
                            include_header=not header_sent,
                        )
                        header_sent = True
                        yield wav_bytes
                except ValueError as exc:
                    logger.error("TTS stream error: %s", exc)
                    raise
                except Exception as exc:
                    logger.exception("TTS stream failed")
                    raise

            return StreamingResponse(
                content=stream_generator(),
                media_type="audio/wav",
            )

        # 非流式响应
        try:
            audio, sample_rate, voice_name = inferencer.tts_pretrained(
                request.input,
                pretrained_voice,
            )
            wav_bytes = pcm16_wav_bytes(audio, sample_rate)
        except ValueError as exc:
            return _openai_error(400, str(exc))
        except Exception as exc:  # pragma: no cover
            logger.exception("TTS failed")
            return _openai_error(500, f"TTS failed: {exc}")

        return StreamingResponse(
            content=iter([wav_bytes]),
            media_type="audio/wav",
        )

    @app.post("/v1/audio/speech")
    def audio_speech_v1(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer] = Depends(get_inferencer_func),
    ):
        return _audio_speech_impl(request, inferencer)

    @app.post("/audio/speech")
    def audio_speech_no_v1(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer] = Depends(get_inferencer_func),
    ):
        return _audio_speech_impl(request, inferencer)

    @app.get("/health")
    def health_check():
        inferencer = get_inferencer_func()
        if inferencer is None:
            return {"status": "uninitialized"}
        return {"status": "healthy"}

    return app
