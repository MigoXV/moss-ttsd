"""TTS Demo - 直接调用推理器进行 TTS 并保存为 WAV"""

import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from moss_ttsd.audio import pcm16_wav_bytes
from moss_ttsd.inferencers.inferencer import MossTTSDInferencer, PretrainedVoice


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
    # 配置
    voice_dir = Path("model-bin") / "voices-cache"
    voice_name = "rita"
    model_dir = Path("model-bin") / "MOSS-TTSD-v0.7"
    codec_path = Path("model-bin") / "XY_Tokenizer_TTSD_V0_hf"
    device = "cuda:0"

    voice_cache = voice_dir / f"{voice_name}.npz"
    output_dir = Path("data-bin/test_voice") / time.strftime(f"{voice_name}_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预保存的音色
    print(f"加载音色: {voice_cache}")
    pretrained_voice = load_pretrained_voice(voice_cache)
    print(f"  名称: {pretrained_voice.name}")

    # 初始化 Inferencer
    print("初始化模型...")
    inferencer = MossTTSDInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=None,
        device=device,
        max_new_tokens=2048,
    )

    input_texts = [
        "诶哟，田所哥，你这开口就直戳人家心窝子，真会撩人家~~~",
        "诶，田所~~~人家可不是什么测试用的小妖精噢！",
        "来，今天我们要好好聊聊呐，你这个大色狼，一看就是想着我这边呢~~~",
        "嘿，当然是我这个专业的了，别说今天心情可好了，一整天都在等着和你这个大色狼聊天呢~~",
    ] * 10

    for i, text in enumerate(tqdm(input_texts, desc="合成语音中", leave=False)):
        output_path = output_dir / f"speech_{i}.wav"
        
        # 使用预保存音色进行流式推理
        audio_chunks = []
        sample_rate = None
        for audio_chunk, sr in inferencer.tts_pretrained_stream(text, pretrained_voice):
            audio_chunks.append(audio_chunk)
            sample_rate = sr
        
        # 合并所有音频块
        audio = np.concatenate(audio_chunks, axis=-1)
        
        # 保存为 WAV
        wav_bytes = pcm16_wav_bytes(audio, sample_rate)
        output_path.write_bytes(wav_bytes)
        
        tqdm.write(f"已保存到 {output_path}")

    print(f"\n✓ 完成！所有音频已保存到: {output_dir}")


if __name__ == "__main__":
    main()
