"""
使用预保存音色进行推理的最简 Demo

流程：
1. 加载保存的 .npz 音色文件
2. 使用 tts_pretrained() 进行快速推理
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from moss_ttsd.audio import pcm16_wav_bytes
from moss_ttsd.inferencers.inferencer import MossTTSDInferencer, PretrainedVoice

TEXT = """
今天的天气真好啊。
好了，这下知道
谁最厉害了。一二三四五六。
七八九十十一。是。
"""

CONFIG = {
    "text": TEXT,
    "voice_cache": Path("model-bin") / "voices-cache-05" / "paimon.npz",
    "model_dir": Path("model-bin") / "MOSS-TTSD-v0.5",
    "codec_path": Path("model-bin") / "XY_Tokenizer_TTSD_V0_hf",
    "device": "cuda:1",
}
# ===============================


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

    # 2. 初始化 Inferencer
    print("\n初始化模型...")
    inferencer = MossTTSDInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=None,
        device=device,
        max_new_tokens=2048,
    )

    # 3. 使用预保存音色进行流式推理
    print(f"\n推理中... 文本: {text}")
    audio_chunks = []
    sample_rate = None
    for audio_chunk, sr in inferencer.tts_pretrained_stream(text, pretrained_voice):
        audio_chunks.append(audio_chunk)
        sample_rate = sr
        print(f"  收到音频块: {audio_chunk.shape[-1]} samples", end="\r", flush=True)

    # 合并所有音频块
    audio = np.concatenate(audio_chunks, axis=-1)
    print(
        f"  总音频长度: {audio.shape[-1]} samples ({audio.shape[-1] / sample_rate:.2f}s)"
    )

    # 4. 保存结果
    wav_bytes = pcm16_wav_bytes(audio, sample_rate)
    wav_path = out_dir / "speech.wav"
    wav_path.write_bytes(wav_bytes)

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "text": text,
                "voice": pretrained_voice.name,
                "sample_rate": sample_rate,
                "voice_cache": str(voice_cache),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✓ 完成！音频已保存: {wav_path}")


if __name__ == "__main__":
    main()
