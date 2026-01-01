from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

from moss_ttsd.audio import pcm16_wav_bytes, trim_silence
from moss_ttsd.voices import VoiceRegistry

from .utils import keep_zh_en_space

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSResult:
    wav_bytes: bytes
    sample_rate: int
    voice: str


@dataclass(frozen=True)
class PretrainedVoice:
    """预训练音色的缓存表示，包含编码后的 audio_codes 以跳过重复的音频编码步骤"""

    name: str
    prompt_text: str
    audio_codes: np.ndarray  # shape: (time, num_codebooks)
    sample_rate: int


class MossTTSDInferencer:
    """Text-to-speech wrapper around MOSS-TTSD."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        codec_path: str | Path,
        voices_dir: str | Path | None = None,
        device: str = "cpu",
        dtype: Any | None = None,
        max_new_tokens: int = 4096,
        attn_implementation: str | None = None,
        trim_silence: bool = True,
        trim_silence_top_db: float = 30.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.codec_path = Path(codec_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.trim_silence_enabled = trim_silence
        self.trim_silence_top_db = trim_silence_top_db

        self.voice_registry = VoiceRegistry(voices_dir or os.getenv("VOICES_DIR"))

        self._load()

    def _load(self) -> None:
        # 加载 processor
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "codec_path": str(self.codec_path),
        }
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_dir), **processor_kwargs
        )

        # 加载 model
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), **model_kwargs
            ).eval()
        else:
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), **model_kwargs
            ).eval()
            self._model.to(torch.device(self.device))

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

    def _decode_tokens(self, token_ids: Any) -> tuple[Optional[np.ndarray], int]:
        """
        自定义 decode 逻辑，绕过 model-bin 中 processor.decode 的 bug。
        """
        assert (
            token_ids.ndim == 2 and token_ids.shape[1] == self._processor.max_channels
        )

        normal = self._processor.shifting_outputs(
            token_ids.unsqueeze(0),
            self._processor.speech_token_range,
            self._processor.max_channels,
        )
        audio_frags = self._processor._find_max_valid_positions(
            normal, self._processor.audio_pad_token_id
        )[0]

        sample_rate = int(getattr(self._processor, "output_sample_rate", 24000))

        if not len(audio_frags):
            return None, sample_rate

        frag = torch.cat([f.permute(1, 0).unsqueeze(1) for f in audio_frags], dim=1)
        decode_result = self._processor.audio_tokenizer.decode(frag, overlap_seconds=10)
        audio_values = decode_result["audio_values"]

        if audio_values is None:
            return None, sample_rate

        if isinstance(audio_values, list):
            if len(audio_values) == 0:
                return None, sample_rate
            audio_tensor = audio_values[0]
            if hasattr(audio_tensor, "detach"):
                audio = audio_tensor.detach().cpu().numpy()
            else:
                audio = np.asarray(audio_tensor)
        else:
            audio = audio_values.detach().cpu().numpy()

        return audio, sample_rate

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
            torch.npu.empty_cache()

    def get_pretrained(self, voice: str = "default") -> PretrainedVoice:
        """
        提取并缓存音色的 audio_codes，用于后续快速推理。
        """
        voice_name = self._parse_voice_name(voice)
        voice_spec = self.voice_registry.resolve(voice_name)

        if voice_spec is None:
            raise ValueError(f"Voice '{voice_name}' not found in voices_dir")

        # 获取 sample_processor 内部组件
        sample_processor = self._processor.sample_processor
        feature_extractor = sample_processor.feature_extractor
        audio_tokenizer = sample_processor.audio_tokenizer
        input_sample_rate = sample_processor.input_sample_rate

        # 加载音频并重采样
        audio, sr = torchaudio.load(str(voice_spec.audio_path))
        if sr != input_sample_rate:
            audio = torchaudio.functional.resample(audio, sr, input_sample_rate)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 提取特征并编码
        feat = feature_extractor(
            audio,
            sampling_rate=input_sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            enc = audio_tokenizer.encode(feat)
            audio_codes = enc["audio_codes"][:, 0].permute(1, 0).cpu().numpy()

        sample_rate = int(getattr(self._processor, "output_sample_rate", 24000))

        return PretrainedVoice(
            name=voice_name,
            prompt_text=voice_spec.transcription_text,
            audio_codes=audio_codes,
            sample_rate=sample_rate,
        )

    def _build_inputs_with_pretrained(
        self,
        text: str,
        pretrained_voice: PretrainedVoice,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用缓存的 PretrainedVoice 构建模型输入，跳过音频加载和编码步骤。
        """
        sample_processor = self._processor.sample_processor
        # 构建完整文本：prompt_text + text
        full_text = (pretrained_voice.prompt_text or "") + text
        final_text = full_text.replace("[S1]", "<speaker1>").replace(
            "[S2]", "<speaker2>"
        )
        # 应用 chat template
        prompt = self._processor.apply_chat_template(conversation=None, text=final_text)
        # 文本编码
        text_ids = np.array(
            self._processor.tokenizer.encode(prompt, add_special_tokens=False)
        )
        grid = np.full(
            (text_ids.shape[0], sample_processor.max_channels),
            sample_processor.audio_pad_token_id,
            dtype=np.int64,
        )
        grid[:, 0] = text_ids
        # 直接使用缓存的 audio_codes
        audio_codes = pretrained_voice.audio_codes.copy()
        audio_codes[:, 0] = audio_codes[:, 0] + sample_processor.speech_token_range[0]
        grid = np.concatenate([grid, audio_codes], axis=0)
        # 应用时间偏移 (shifting)
        T, C = grid.shape
        max_channels = sample_processor.max_channels
        new_len = T + max_channels - 1
        shifted = np.full((new_len, max_channels), fill_value=1024, dtype=np.int64)
        shifted[:, 0] = np.full(
            new_len, self._processor.tokenizer.pad_token_id, dtype=np.int64
        )
        for j in range(max_channels):
            shifted[j : (T + j), j] = grid[:, j]
        # 构建 attention_mask
        input_ids = torch.from_numpy(shifted).long().unsqueeze(0).to(self.device)
        attention_mask = torch.ones((1, new_len), dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    def tts_pretrained(
        self, raw_text: str, pretrained_voice: PretrainedVoice
    ) -> Tuple[np.ndarray, int, str]:
        """
        使用预训练音色进行 TTS 推理，跳过音频编码步骤以加速推理。
        """
        text = keep_zh_en_space(raw_text)
        logger.info(f"before:{raw_text} | after:{text}")
        text = self._ensure_speaker_tag(text)
        try:
            input_ids, attention_mask = self._build_inputs_with_pretrained(
                text, pretrained_voice
            )
            # 生成 token_ids
            with torch.inference_mode():
                token_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                )
            # 解码生成的 token_ids 为音频
            audio, sample_rate = self._decode_tokens(token_ids[0])
            if audio is None:
                raise RuntimeError("Failed to decode audio from generated tokens")

            audio = np.asarray(audio)
            if self.trim_silence_enabled:
                audio = trim_silence(
                    audio,
                    sample_rate,
                    top_db=self.trim_silence_top_db,
                    trim_start=True,
                    trim_end=False,
                )

            return audio, sample_rate, pretrained_voice.name
        except Exception as e:
            logger.error(f"TTS inference failed: {e}")
            raise
        finally:
            self.clear_cache()

    def tts_pretrained_stream(
        self, raw_text: str, pretrained_voice: PretrainedVoice, chunk_size: int = 3200
    ) -> Generator[np.ndarray, None, None]:
        """使用预训练音色进行流式 TTS 推理，逐块生成音频以减少延迟。"""
        # 使用换行符和句号分割文本
        texts = re.split(r"[\n。]+", raw_text)
        texts = [t for t in texts if t.strip() and len(t.strip()) > 1]
        for text in texts:
            audio, sample_rate, voice_name = self.tts_pretrained(text, pretrained_voice)
            total_samples = audio.shape[0]
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                audio_chunk = audio[start:end]
                yield audio_chunk, sample_rate
