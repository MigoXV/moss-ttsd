from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

from moss_ttsd.audio import pcm16_wav_bytes, sine_wav_bytes, trim_silence
from moss_ttsd.voices import VoiceRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSResult:
    wav_bytes: bytes
    sample_rate: int
    model: str
    voice: str
    used_fallback: bool


class AudioTokenStreamer:
    """用于流式音频生成的 Token Streamer，收集 tokens 并定期解码为音频块"""
    
    def __init__(
        self,
        processor: Any,
        torch_module: Any,
        chunk_size: int = 50,  # 每多少个 token 解码一次
        sample_rate: int = 24000,
    ):
        self.processor = processor
        self.torch = torch_module
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        
        self.token_buffer: list = []
        self.audio_queue: queue.Queue = queue.Queue()
        self.finished = False
        self._lock = threading.Lock()
        
    def put(self, tokens):
        """接收生成的 tokens"""
        with self._lock:
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            if isinstance(tokens, list):
                self.token_buffer.extend(tokens)
            else:
                self.token_buffer.append(tokens)
    
    def end(self):
        """标记生成结束"""
        with self._lock:
            self.finished = True
            # 处理剩余的 tokens
            self.audio_queue.put(None)  # 结束信号
    
    def get_audio_chunks(self) -> Generator[bytes, None, None]:
        """获取音频块的生成器"""
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                with self._lock:
                    if self.finished:
                        break
                continue


class MossTTSDInferencer:
    """Text-to-speech wrapper around MOSS-TTSD using the same flow as tests/test01.py."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        codec_path: str | Path | None = None,
        voices_dir: str | Path | None = None,
        device: str = "cpu",
        dtype: Any | None = None,
        max_new_tokens: int = 4096,
        fallback_audio: str = "error",
        attn_implementation: str | None = None,
        trim_silence: bool = True,
        trim_silence_top_db: float = 30.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.fallback_audio = fallback_audio
        self.attn_implementation = attn_implementation
        self.trim_silence_enabled = trim_silence
        self.trim_silence_top_db = trim_silence_top_db

        self.codec_path = self._resolve_codec_path(codec_path)
        self.voice_registry = VoiceRegistry(voices_dir or os.getenv("VOICES_DIR"))
        self._processor = None
        self._model = None
        self._torch = None
        self._init_error: Optional[str] = None

        self._load()

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    @property
    def is_ready(self) -> bool:
        return self._processor is not None and self._model is not None and self._torch is not None

    def _set_init_error(self, message: str) -> None:
        if self._init_error is None:
            self._init_error = message

    def _resolve_codec_path(self, codec_path: str | Path | None) -> Optional[Path]:
        if codec_path:
            return Path(codec_path)

        env_path = os.getenv("MOSS_TTSD_CODEC_PATH") or os.getenv("CODEC_PATH")
        if env_path:
            return Path(env_path)

        local_default = self.model_dir / "XY_Tokenizer"
        if local_default.exists():
            return local_default

        # Common layout when users download codec separately next to model.
        for name in ("XY_Tokenizer_TTSD_V0_hf", "XY_Tokenizer", "codec", "codec-bin"):
            candidate = self.model_dir.parent / name
            if candidate.exists():
                return candidate

        return None

    def _load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {self.model_dir}")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("PyTorch not installed; TTS will fall back. (%s)", exc)
            self._set_init_error("PyTorch not installed (pip/poetry install torch).")
            self._torch = None
            return

        self._torch = torch

        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("transformers not installed; TTS will fall back. (%s)", exc)
            self._set_init_error("transformers not installed (pip/poetry install transformers).")
            return

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.codec_path is not None:
            processor_kwargs["codec_path"] = str(self.codec_path)

        if self.codec_path is None:
            logger.warning(
                "codec_path not set; MOSS-TTSD processor cannot be loaded (set MOSS_TTSD_CODEC_PATH)."
            )
            self._set_init_error(
                "codec_path not set; pass --codec-path or set MOSS_TTSD_CODEC_PATH to your XY_Tokenizer model directory."
            )
            return

        try:
            self._processor = AutoProcessor.from_pretrained(str(self.model_dir), **processor_kwargs)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load processor; TTS will fall back. (%s)", exc)
            self._processor = None
            self._set_init_error(f"Failed to load processor: {exc}")
            return

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
            try:
                self._model = AutoModel.from_pretrained(str(self.model_dir), **model_kwargs).eval()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load model; TTS will fall back. (%s)", exc)
                self._model = None
                self._set_init_error(f"Failed to load model: {exc}")
            return

        try:
            self._model = AutoModel.from_pretrained(str(self.model_dir), **model_kwargs).eval()
            self._model.to(torch.device(self.device))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load model; TTS will fall back. (%s)", exc)
            self._model = None
            self._set_init_error(f"Failed to load model: {exc}")
            return

        self._init_error = None

    def _ensure_speaker_tag(self, text: str) -> str:
        if "[S1]" in text or "[S2]" in text:
            return text
        return f"[S1]{text}"

    @staticmethod
    def _parse_voice_name(voice: str) -> str:
        """
        OpenAI clients often pass `voice` like: "repo_id:voice_name".
        We map the actual directory name to the suffix after the last ':'.
        """
        voice = (voice or "").strip()
        if not voice:
            return "default"
        return voice.split(":")[-1]

    def _decode_tokens(self, processor: Any, token_ids: Any) -> tuple[Optional[np.ndarray], int]:
        """
        自定义 decode 逻辑，绕过 model-bin 中 processor.decode 的 bug。
        
        原始 processor.decode 假设 audio_tokenizer.decode() 返回的 audio_values 
        是一个 tensor，但实际上它返回的是一个 list，导致 .detach() 失败。
        
        Returns:
            tuple: (audio_numpy_array 或 None, sample_rate)
        """
        torch = self._torch
        assert token_ids.ndim == 2 and token_ids.shape[1] == processor.max_channels
        
        normal = processor.shifting_outputs(
            token_ids.unsqueeze(0), 
            processor.speech_token_range, 
            processor.max_channels
        )
        audio_frags = processor._find_max_valid_positions(normal, processor.audio_pad_token_id)[0]
        
        sample_rate = int(getattr(processor, "output_sample_rate", 24000))
        
        if not len(audio_frags):
            return None, sample_rate
        
        frag = torch.cat([f.permute(1, 0).unsqueeze(1) for f in audio_frags], dim=1)
        decode_result = processor.audio_tokenizer.decode(frag, overlap_seconds=10)
        audio_values = decode_result["audio_values"]
        
        # 修复 bug：audio_values 可能是 list 而非 tensor
        if audio_values is None:
            return None, sample_rate
        
        if isinstance(audio_values, list):
            # audio_values 是一个 list of tensors，取第一个并转换
            if len(audio_values) == 0:
                return None, sample_rate
            audio_tensor = audio_values[0]
            if hasattr(audio_tensor, "detach"):
                audio = audio_tensor.detach().cpu().numpy()
            else:
                audio = np.asarray(audio_tensor)
        else:
            # 正常情况：audio_values 是 tensor
            audio = audio_values.detach().cpu().numpy()
        
        return audio, sample_rate

    def clear_cache(self) -> None:
        torch = self._torch
        if torch is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if hasattr(torch, "npu") and torch.npu.is_available():  # pragma: no cover
            torch.npu.synchronize()
            torch.npu.empty_cache()

    def tts(
        self,
        text: str,
        *,
        model: str = "moss-ttsd",
        voice: str = "default",
        response_format: str = "wav",
    ) -> TTSResult:
        response_format = (response_format or "wav").lower()
        if response_format != "wav":
            raise ValueError(f"Unsupported response_format: {response_format}")

        if not text:
            raise ValueError("input text cannot be empty")

        if not self.is_ready:
            if self.fallback_audio == "dummy":
                wav_bytes = sine_wav_bytes()
                return TTSResult(
                    wav_bytes=wav_bytes,
                    sample_rate=24000,
                    model=model,
                    voice=voice,
                    used_fallback=True,
                )
            detail = self._init_error or "Inferencer is not initialized."
            raise RuntimeError(detail)

        torch = self._torch
        processor = self._processor
        model_obj = self._model

        text = self._ensure_speaker_tag(text)
        try:
            voice_name = self._parse_voice_name(voice)
            voice_spec = self.voice_registry.resolve(voice_name)

            item: dict[str, Any] = {"text": text}
            if voice_spec is not None:
                item["prompt_audio"] = str(voice_spec.audio_path)
                item["prompt_text"] = voice_spec.transcription_text

            inputs = processor(item)

            # 将输入张量移动到模型所在设备（修复 CPU/GPU 设备不匹配问题）
            model_device = next(model_obj.parameters()).device
            if hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            else:
                inputs = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            with torch.inference_mode():
                token_ids = model_obj.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                )
            # 自定义 decode 逻辑，绕过 model-bin 中 processor.decode 的 bug
            # （audio_tokenizer.decode 返回 list 而非 tensor，导致 .detach() 失败）
            audio, sample_rate = self._decode_tokens(processor, token_ids[0])

            if audio is None:
                wav_bytes = sine_wav_bytes(sample_rate=sample_rate)
            else:
                # 使用能量 VAD 去除开头的静音部分
                audio = trim_silence(np.asarray(audio), sample_rate, top_db=30.0, trim_start=True, trim_end=False)
                wav_bytes = pcm16_wav_bytes(audio, sample_rate)

            return TTSResult(
                wav_bytes=wav_bytes,
                sample_rate=sample_rate,
                model=model,
                voice=voice,
                used_fallback=False,
            )
        finally:
            self.clear_cache()

    def tts_stream(
        self,
        text: str,
        *,
        model: str = "moss-ttsd",
        voice: str = "default",
        response_format: str = "wav",
        chunk_size: int = 100,  # 每多少个 token 解码一次
    ) -> Generator[bytes, None, None]:
        """
        流式 TTS 生成，逐块返回音频数据。
        
        由于 MOSS-TTSD 模型的特殊多通道架构，流式生成需要累积足够的 tokens 才能解码。
        这里采用的策略是：在生成过程中累积 tokens，最后一次性解码并返回完整音频。
        
        对于真正的实时流式音频，需要模型原生支持增量解码，这里提供的是"分块流式"，
        即服务端生成完成后分块发送给客户端。
        
        Args:
            text: 要合成的文本
            model: 模型标识符
            voice: 音色名称
            response_format: 响应格式，目前仅支持 'wav'
            chunk_size: 每个音频块的字节数（用于流式传输）
            
        Yields:
            bytes: WAV 格式的音频数据块
        """
        response_format = (response_format or "wav").lower()
        if response_format != "wav":
            raise ValueError(f"Unsupported response_format: {response_format}")

        if not text:
            raise ValueError("input text cannot be empty")

        if not self.is_ready:
            if self.fallback_audio == "dummy":
                wav_bytes = sine_wav_bytes()
                yield wav_bytes
                return
            detail = self._init_error or "Inferencer is not initialized."
            raise RuntimeError(detail)

        torch = self._torch
        processor = self._processor
        model_obj = self._model

        text = self._ensure_speaker_tag(text)
        try:
            voice_name = self._parse_voice_name(voice)
            voice_spec = self.voice_registry.resolve(voice_name)

            item: dict[str, Any] = {"text": text}
            if voice_spec is not None:
                item["prompt_audio"] = str(voice_spec.audio_path)
                item["prompt_text"] = voice_spec.transcription_text

            inputs = processor(item)

            # 将输入张量移动到模型所在设备
            model_device = next(model_obj.parameters()).device
            if hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            else:
                inputs = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            # 生成 tokens
            with torch.inference_mode():
                token_ids = model_obj.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                )
            
            # 解码音频
            audio, sample_rate = self._decode_tokens(processor, token_ids[0])

            if audio is None:
                wav_bytes = sine_wav_bytes(sample_rate=sample_rate)
            else:
                audio = np.asarray(audio)
                # 使用能量 VAD 去除开头的静音部分
                if self.trim_silence_enabled:
                    audio = trim_silence(
                        audio, sample_rate,
                        top_db=self.trim_silence_top_db,
                        trim_start=True,
                        trim_end=False,
                    )
                wav_bytes = pcm16_wav_bytes(audio, sample_rate)

            # 分块发送
            stream_chunk_size = 4096  # 每块 4KB
            for i in range(0, len(wav_bytes), stream_chunk_size):
                yield wav_bytes[i:i + stream_chunk_size]
                
        finally:
            self.clear_cache()
