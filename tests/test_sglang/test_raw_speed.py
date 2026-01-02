"""
测速脚本: 使用 tts_pretrained_test_speed 方法测试 TTS 推理速度

流程:
1. 加载保存的 .npz 音色文件
2. 使用 tts_pretrained_test_speed() 进行推理并显示性能统计
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from moss_ttsd.audio import pcm16_wav_bytes, trim_silence
from moss_ttsd.inferencers.inferencer import MossTTSDInferencer, PretrainedVoice
from moss_ttsd.inferencers.utils import keep_zh_en_space

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("test_raw_speed")

TEXT = """
一二三四五六七八九十十一。是。
今天的天气真好啊。
好了，这下知道谁最厉害了。
"""

CONFIG = {
    "text": TEXT,
    "voice_cache": Path("model-bin") / "voices-cache" / "man01.npz",
    "model_dir": Path("model-bin") / "MOSS-TTSD-v0.7",
    "codec_path": Path("model-bin") / "XY_Tokenizer_TTSD_V0_hf",
    "device": "cuda:1",
    "max_new_tokens": 2048,
}
# ===============================


class TestSpeedInferencer(MossTTSDInferencer):

    @torch.inference_mode()
    def tts_pretrained_test_speed(
        self, raw_text: str, pretrained_voice: PretrainedVoice
    ) -> Tuple[np.ndarray, int, str]:
        """
        使用预训练音色进行 TTS 推理，跳过音频编码步骤以加速推理。
        打印 generate 和 decode 环节的绝对时长和占推理时长的百分比。
        """
        import time

        text = keep_zh_en_space(raw_text)
        logger.info(f"before:{raw_text} | after:{text}")
        text = self._ensure_speaker_tag(text)
        try:
            input_ids, attention_mask = self._build_inputs_with_pretrained(
                text, pretrained_voice
            )

            # 记录总推理开始时间
            total_start = time.time()

            # 生成 token_ids (计时)
            generate_start = time.time()
            with torch.inference_mode():
                token_ids = self._model.generate(
                    input_ids=input_ids,
                    # attention_mask=233,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                )
            generate_end = time.time()
            generate_time = generate_end - generate_start

            # 解码生成的 token_ids 为音频 (计时)
            decode_start = time.time()
            audio, sample_rate = self._decode_tokens(token_ids[0])
            decode_end = time.time()
            decode_time = decode_end - decode_start

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

            # 计算总时长和百分比
            total_time = time.time() - total_start
            generate_percent = (
                (generate_time / total_time) * 100 if total_time > 0 else 0
            )
            decode_percent = (decode_time / total_time) * 100 if total_time > 0 else 0

            # 打印时间统计
            print("=" * 60)
            print(f"推理时间统计:")
            print(f"  Generate 时长: {generate_time:.4f}s ({generate_percent:.2f}%)")
            print(f"  Decode 时长:   {decode_time:.4f}s ({decode_percent:.2f}%)")
            print(f"  总推理时长:    {total_time:.4f}s")
            print(f"  音频时长:      {audio.shape[-1] / sample_rate:.4f}s")
            print(
                f"  RTF (实时率):  { ((audio.shape[-1] / sample_rate) / total_time):.4f}"
            )
            print("=" * 60)

            return audio, sample_rate, pretrained_voice.name
        except Exception as e:
            logger.error(f"TTS inference failed: {e}")
            raise
        finally:
            # self.clear_cache()
            pass


def load_pretrained_voice(npz_path: Path) -> PretrainedVoice:
    """从 .npz 文件加载音色"""
    data = np.load(npz_path, allow_pickle=True)
    audio_codes = data["audio_codes"]
    metadata = data["metadata"].item()

    return PretrainedVoice(
        name=metadata["name"],
        prompt_text=metadata["prompt_text"],
        audio_codes=audio_codes,
        sample_rate=metadata["sample_rate"],
    )


def main():
    text = CONFIG["text"]
    voice_cache = Path(CONFIG["voice_cache"])
    model_dir = CONFIG["model_dir"]
    codec_path = CONFIG["codec_path"]
    device = CONFIG["device"]
    max_new_tokens = CONFIG["max_new_tokens"]

    # 输出目录
    out_dir = (
        Path("data-bin") / "tts-outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载预保存的音色
    print(f"加载音色: {voice_cache}")
    pretrained_voice = load_pretrained_voice(voice_cache)
    print(f"  名称: {pretrained_voice.name}")
    print(f"  audio_codes shape: {pretrained_voice.audio_codes.shape}")
    print(f"  prompt_text: {pretrained_voice.prompt_text[:50]}...")

    # 2. 初始化 Inferencer
    print("\n初始化模型...")
    inferencer = TestSpeedInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=None,
        device=device,
        max_new_tokens=max_new_tokens,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    # 2.1 预分配6GB显存以提高首token速度
    print("\n预分配显存...")
    try:
        # 计算需要分配的元素数量 (6GB / 4 bytes per float32)
        num_elements = (6 * 1024 * 1024 * 1024) // 4
        dummy_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        print(f"  已分配 {dummy_tensor.element_size() * dummy_tensor.nelement() / 1024**3:.2f} GB 显存")
        del dummy_tensor
        torch.cuda.empty_cache()
        print("  显存预分配完成")
    except Exception as e:
        print(f"  显存预分配失败 (继续执行): {e}")

    # 3. 使用预保存音色进行推理并测速
    print(f"\n推理中... 文本: {text.strip()}")
    print("-" * 60)

    audio, sample_rate, voice_name = inferencer.tts_pretrained_test_speed(
        text, pretrained_voice
    )
    audio, sample_rate, voice_name = inferencer.tts_pretrained_test_speed(
        text, pretrained_voice
    )
    audio, sample_rate, voice_name = inferencer.tts_pretrained_test_speed(
        text, pretrained_voice
    )
    audio, sample_rate, voice_name = inferencer.tts_pretrained_test_speed(
        text, pretrained_voice
    )
    print("-" * 60)
    print(f"✓ 推理完成")
    print(f"  音频长度: {audio.shape[-1]} samples")
    print(f"  音频时长: {audio.shape[-1] / sample_rate:.4f}s")
    print(f"  采样率: {sample_rate} Hz")

    # 4. 保存结果
    wav_bytes = pcm16_wav_bytes(audio, sample_rate)
    wav_path = out_dir / "speech.wav"
    wav_path.write_bytes(wav_bytes)

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "text": text,
                "voice": voice_name,
                "sample_rate": sample_rate,
                "voice_cache": str(voice_cache),
                "audio_duration": audio.shape[-1] / sample_rate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✓ 完成！音频已保存: {wav_path}")


if __name__ == "__main__":
    main()
