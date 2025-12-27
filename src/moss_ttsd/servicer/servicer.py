from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from moss_ttsd.inferencers import MossTTSDInferencer

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model identifier")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field("default", description="Voice name (ignored by default)")
    response_format: str = Field("wav", description="Only 'wav' is supported currently")


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


def create_app(get_inferencer_func: Callable[[], Optional[MossTTSDInferencer]]) -> FastAPI:
    app = FastAPI(title="MOSS-TTSD OpenAI TTS API")

    async def _audio_speech_impl(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer],
    ):
        if inferencer is None:
            raise HTTPException(status_code=500, detail="Inferencer not initialized")

        response_format = (request.response_format or "wav").lower()
        if response_format != "wav":
            return _openai_error(400, f"Unsupported response_format: {response_format}", param="response_format")

        try:
            result = await asyncio.to_thread(
                inferencer.tts,
                request.input,
                model=request.model,
                voice=request.voice,
                response_format=response_format,
            )
        except ValueError as exc:
            return _openai_error(400, str(exc))
        except Exception as exc:  # pragma: no cover
            logger.exception("TTS failed")
            return _openai_error(500, f"TTS failed: {exc}")

        return StreamingResponse(
            content=iter([result.wav_bytes]),
            media_type="audio/wav",
        )

    @app.post("/v1/audio/speech")
    async def audio_speech_v1(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer] = Depends(get_inferencer_func),
    ):
        return await _audio_speech_impl(request, inferencer)

    @app.post("/audio/speech")
    async def audio_speech_no_v1(
        request: SpeechRequest,
        inferencer: Optional[MossTTSDInferencer] = Depends(get_inferencer_func),
    ):
        return await _audio_speech_impl(request, inferencer)

    @app.get("/health")
    async def health_check():
        inferencer = get_inferencer_func()
        if inferencer is None:
            return {"status": "uninitialized"}
        return {
            "status": "healthy" if inferencer.is_ready else "degraded",
            "ready": inferencer.is_ready,
            "init_error": inferencer.init_error,
        }

    return app
