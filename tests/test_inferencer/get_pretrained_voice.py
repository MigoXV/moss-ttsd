"""
音色保存工具

功能：输入音色目录，保存音色到磁盘，方便下次推理时快速加载

音色目录结构要求：
    voice_dir/
        audio.wav          # 音频文件（必须命名为 audio.wav）
        transcription.txt  # 转写文本（也可以是 prompt.txt 或 text.txt）

使用方法：
    1. 修改 CONFIG 中的配置
    2. 运行: python tests/test_inferencer/get_pretrained_voice.py
"""
from pathlib import Path
import json
import time
import numpy as np

from moss_ttsd.inferencers.inferencer import MossTTSDInferencer, PretrainedVoice


# ============= 配置 =============
repo_root = Path(__file__).parent.parent.parent

CONFIG = {
    # 输入：音色目录（需包含 audio.wav 和 transcription.txt）
    "voice_dir": repo_root / "model-bin" / "voices" / "paimon",
    
    # 输出目录
    "output_dir": repo_root / "model-bin" / "voices-cache",
    
    # 模型配置
    "model_dir": repo_root / "model-bin" / "MOSS-TTSD-v0.7",
    "codec_path": repo_root / "model-bin" / "XY_Tokenizer_TTSD_V0_hf",
    "device": "cuda:1",
}
# ===============================


def save_pretrained_voice(pretrained_voice: PretrainedVoice, save_path: Path) -> None:
    """保存音色到磁盘"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 .npz 文件
    npz_path = save_path.with_suffix(".npz")
    np.savez(
        npz_path,
        audio_codes=pretrained_voice.audio_codes,
        metadata=np.array([{
            "name": pretrained_voice.name,
            "prompt_text": pretrained_voice.prompt_text,
            "sample_rate": pretrained_voice.sample_rate,
            "audio_codes_shape": list(pretrained_voice.audio_codes.shape),
        }], dtype=object)
    )
    
    # 保存元数据 JSON
    json_path = save_path.with_suffix(".json")
    metadata = {
        "name": pretrained_voice.name,
        "prompt_text": pretrained_voice.prompt_text,
        "sample_rate": pretrained_voice.sample_rate,
        "audio_codes_shape": list(pretrained_voice.audio_codes.shape),
        "audio_codes_dtype": str(pretrained_voice.audio_codes.dtype),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"✓ 音色已保存: {npz_path}")
    print(f"  元数据: {json_path}")


def main():
    # 读取配置
    voice_dir = Path(CONFIG["voice_dir"])
    output_dir = Path(CONFIG["output_dir"])
    model_dir = Path(CONFIG["model_dir"])
    codec_path = Path(CONFIG["codec_path"])
    device = CONFIG["device"]
    
    # 检查音色目录
    if not voice_dir.exists():
        print(f"✗ 音色目录不存在: {voice_dir}")
        return
    
    if not voice_dir.is_dir():
        print(f"✗ 不是有效的目录: {voice_dir}")
        return
    
    # 检查必要文件
    audio_path = voice_dir / "audio.wav"
    if not audio_path.exists():
        print(f"✗ 音频文件不存在: {audio_path}")
        print("  提示: 音频文件必须命名为 audio.wav")
        return
    
    # 查找转写文本文件
    txt_path = None
    for name in ("transcription.txt", "prompt.txt", "text.txt"):
        candidate = voice_dir / name
        if candidate.exists():
            txt_path = candidate
            break
    
    if txt_path is None:
        print(f"✗ 转写文本文件不存在: {voice_dir}")
        print("  提示: 需要 transcription.txt, prompt.txt 或 text.txt")
        return
    
    # 读取转写文本
    transcription = txt_path.read_text(encoding="utf-8").strip()
    if not transcription:
        print(f"✗ 转写文本文件为空: {txt_path}")
        return
    
    # 音色名称使用目录名
    voice_name = voice_dir.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("音色保存工具")
    print("=" * 60)
    print(f"音色目录: {voice_dir}")
    print(f"音频文件: {audio_path.name}")
    print(f"转写文件: {txt_path.name}")
    print(f"转写内容: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
    print(f"音色名称: {voice_name}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    
    # 初始化 Inferencer（使用音色目录的父目录作为 voices_dir）
    print("\n[1/3] 初始化模型...")
    start_time = time.perf_counter()
    inferencer = MossTTSDInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=voice_dir.parent,  # 音色目录的父目录
        device=device,
        max_new_tokens=2048,
    )
    init_time = time.perf_counter() - start_time
    print(f"✓ 模型初始化完成 ({init_time:.2f}s)")
    
    # 提取音色
    print(f"\n[2/3] 提取音色...")
    start_time = time.perf_counter()
    pretrained_voice = inferencer.get_pretrained(voice_name)
    create_time = time.perf_counter() - start_time
    print(f"✓ 音色提取完成 ({create_time:.2f}s)")
    print(f"  audio_codes shape: {pretrained_voice.audio_codes.shape}")
    print(f"  sample_rate: {pretrained_voice.sample_rate}")
    
    # 保存音色
    print(f"\n[3/3] 保存音色...")
    save_path = output_dir / voice_name
    save_pretrained_voice(pretrained_voice, save_path)
    
    print(f"\n{'=' * 60}")
    print(f"✓ 完成！")
    print(f"  总耗时: {init_time + create_time:.2f}s")
    print(f"  下次使用时直接加载: {save_path.with_suffix('.npz')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
