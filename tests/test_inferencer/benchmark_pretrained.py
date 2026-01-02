"""
预训练音色推理性能测试

测试内容：
1. 提取 PretrainedVoice 的耗时
2. 使用 tts_pretrained (预训练音色) 的推理耗时
3. 多次推理的平均耗时和稳定性

使用方法：
    python tests/test_inferencer/benchmark_pretrained.py
"""
from pathlib import Path
from datetime import datetime
import json
import sys
import time

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from moss_ttsd.inferencers.inferencer import MossTTSDInferencer


def benchmark():
    # 配置参数
    text = "诶，田所，你这反应可真够淡定的"
    voice = "man01"
    num_runs = 10  # 每种方法运行次数
    
    model_dir = repo_root / "model-bin" / "MOSS-TTSD-v0.7"
    codec_path = repo_root / "model-bin" / "XY_Tokenizer_TTSD_V0_hf"
    voices_dir = repo_root / "model-bin" / "voices"

    out_dir = repo_root / "data-bin" / "tts-outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOSS-TTSD 预训练音色性能测试")
    print("=" * 60)
    print(f"测试文本: {text[:30]}...")
    print(f"音色: {voice}")
    print(f"推理运行次数: {num_runs}")
    print()

    # 初始化 inferencer
    print("[1/3] 初始化 Inferencer...")
    init_start = time.perf_counter()
    inferencer = MossTTSDInferencer(
        model_dir=model_dir,
        codec_path=codec_path,
        voices_dir=voices_dir,
        device="cuda:1",
        max_new_tokens=2048,
        dtype="bfloat16",
    )
    init_time = time.perf_counter() - init_start
    print(f"    初始化耗时: {init_time:.2f}s")
    print()

    # 提取预训练音色
    print("[2/3] 提取预训练音色 (get_pretrained)...")
    extract_start = time.perf_counter()
    pretrained_voice = inferencer.get_pretrained(voice)
    extract_time = time.perf_counter() - extract_start
    print(f"    提取耗时: {extract_time:.3f}s")
    print(f"    audio_codes shape: {pretrained_voice.audio_codes.shape}")
    print(f"    prompt_text: {pretrained_voice.prompt_text[:50]}...")
    print()

    # 使用预训练音色推理 (tts_pretrained)
    print(f"[3/3] 使用预训练音色推理 (tts_pretrained) x{num_runs}...")
    pretrained_times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result_pretrained = inferencer.tts_pretrained(text, pretrained_voice)
        elapsed = time.perf_counter() - start
        pretrained_times.append(elapsed)
        print(f"    第 {i+1} 次: {elapsed:.3f}s")
    
    pretrained_avg = sum(pretrained_times) / len(pretrained_times)
    pretrained_min = min(pretrained_times)
    pretrained_max = max(pretrained_times)
    
    print(f"    平均耗时: {pretrained_avg:.3f}s")
    print(f"    最快: {pretrained_min:.3f}s")
    print(f"    最慢: {pretrained_max:.3f}s")
    
    # 保存推理结果
    wav_path = out_dir / "speech_pretrained.wav"
    wav_path.write_bytes(result_pretrained.wav_bytes)
    print()

    # 汇总结果
    print("=" * 60)
    print("性能测试汇总")
    print("=" * 60)
    print(f"初始化耗时:                    {init_time:.3f}s (仅需执行一次)")
    print(f"提取音色 (get_pretrained):     {extract_time:.3f}s (每个音色仅需执行一次)")
    print(f"推理耗时 (tts_pretrained):     {pretrained_avg:.3f}s (平均)")
    print(f"    最快: {pretrained_min:.3f}s")
    print(f"    最慢: {pretrained_max:.3f}s")
    print()
    
    # 计算总开销
    total_time_first = extract_time + pretrained_times[0]
    total_time_subsequent = pretrained_avg
    print(f"首次推理总耗时 (包含提取):     {total_time_first:.3f}s")
    print(f"后续推理耗时 (使用缓存):       {total_time_subsequent:.3f}s")
    print()
    
    print(f"输出文件:")
    print(f"    音频: {wav_path}")

    # 保存详细结果
    results = {
        "text": text,
        "voice": voice,
        "num_runs": num_runs,
        "init_time_s": init_time,
        "extract_pretrained_time_s": extract_time,
        "audio_codes_shape": list(pretrained_voice.audio_codes.shape),
        "tts_pretrained_times_s": pretrained_times,
        "tts_pretrained_avg_s": pretrained_avg,
        "tts_pretrained_min_s": pretrained_min,
        "tts_pretrained_max_s": pretrained_max,
        "first_inference_total_s": total_time_first,
        "subsequent_inference_s": total_time_subsequent,
        "model_dir": str(model_dir),
        "codec_path": str(codec_path),
        "voices_dir": str(voices_dir),
    }
    
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    详细结果: {out_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    benchmark()
