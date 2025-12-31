"""TTS Demo - 长文本分段并行请求

这个示例展示如何：
1. 将长文本按标点符号切分成多个段落
2. 使用多线程并行请求 TTS API
3. 流式接收每个段落的音频数据
4. 按顺序拼接成完整的 numpy 数组并保存
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import httpx
import numpy as np
import soundfile as sf

# 配置
BASE_URL = "http://localhost:10003"
OUTPUT_PATH = Path("data-bin/speech_segment_merged.wav")
MAX_WORKERS = 4  # 最大并行线程数
TIMEOUT = 120.0

# 测试长文本
LONG_TEXT = """
人工智能正在改变我们的生活方式。从智能手机到自动驾驶汽车，AI技术无处不在。
机器学习让计算机能够从数据中学习，而深度学习则模仿人脑的神经网络结构。
自然语言处理使机器能够理解和生成人类语言。计算机视觉让机器能够"看见"世界。
未来，人工智能将在医疗、教育、金融等领域发挥更大的作用。
我们应该拥抱这项技术，同时也要注意其带来的伦理和社会问题。
"""


def split_text_by_punctuation(text: str) -> List[str]:
    """按标点符号切分文本
    
    支持的标点：句号、问号、感叹号、分号、冒号（中英文）
    """
    # 清理文本：去除首尾空白，合并多个空白字符
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # 按标点切分，保留标点符号
    # 匹配中英文标点：。！？；：.!?;:
    pattern = r'([。！？；.!?;])'
    parts = re.split(pattern, text)
    
    # 将标点符号与前面的文本合并
    segments = []
    i = 0
    while i < len(parts):
        segment = parts[i].strip()
        # 如果下一个是标点，合并
        if i + 1 < len(parts) and re.match(pattern, parts[i + 1]):
            segment += parts[i + 1]
            i += 2
        else:
            i += 1
        
        # 过滤空段落
        if segment and len(segment) > 0:
            segments.append(segment)
    
    return segments


def fetch_audio_stream(segment_index: int, text: str) -> Tuple[int, bytes]:
    """请求单个文本段落的音频（流式接收）
    
    Args:
        segment_index: 段落索引（用于保持顺序）
        text: 文本内容
        
    Returns:
        (段落索引, 音频字节数据)
    """
    print(f"[段落 {segment_index}] 开始请求: {text[:30]}...")
    
    audio_data = io.BytesIO()
    
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={
            "model": "OpenMOSS-Team/MOSS-TTSD-v0.7",
            "voice": "man01",
            "input": text,
            "response_format": "wav",
            "stream": True,
        },
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        
        total_bytes = 0
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_data.write(chunk)
            total_bytes += len(chunk)
        
        print(f"[段落 {segment_index}] 完成，接收 {total_bytes} 字节")
    
    return segment_index, audio_data.getvalue()


def wav_bytes_to_numpy(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """将 WAV 字节数据转换为 numpy 数组
    
    Returns:
        (音频数据数组, 采样率)
    """
    audio_io = io.BytesIO(wav_bytes)
    data, samplerate = sf.read(audio_io)
    return data, samplerate


def merge_audio_segments(segments: List[Tuple[int, bytes]]) -> Tuple[np.ndarray, int]:
    """合并多个音频段落
    
    Args:
        segments: [(索引, wav字节数据), ...] 列表
        
    Returns:
        (合并后的音频数组, 采样率)
    """
    # 按索引排序
    segments_sorted = sorted(segments, key=lambda x: x[0])
    
    audio_arrays = []
    samplerate = None
    
    for idx, wav_bytes in segments_sorted:
        data, sr = wav_bytes_to_numpy(wav_bytes)
        
        if samplerate is None:
            samplerate = sr
        elif samplerate != sr:
            print(f"警告: 段落 {idx} 采样率 {sr} 与预期 {samplerate} 不符")
        
        audio_arrays.append(data)
        print(f"[段落 {idx}] 音频长度: {len(data)} 采样点")
    
    # 拼接所有音频
    merged_audio = np.concatenate(audio_arrays)
    return merged_audio, samplerate


def main():
    """主函数"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 切分文本
    print("=" * 50)
    print("步骤 1: 切分文本")
    print("=" * 50)
    segments = split_text_by_punctuation(LONG_TEXT)
    print(f"共切分为 {len(segments)} 个段落:")
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg}")
    
    # 2. 多线程并行请求
    print("\n" + "=" * 50)
    print("步骤 2: 多线程并行请求 TTS API")
    print("=" * 50)
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(fetch_audio_stream, i, seg): i 
            for i, seg in enumerate(segments)
        }
        
        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                print(f"[段落 {idx}] 请求失败: {e}")
    
    if len(results) != len(segments):
        print(f"警告: 只有 {len(results)}/{len(segments)} 个段落成功")
    
    # 3. 合并音频
    print("\n" + "=" * 50)
    print("步骤 3: 合并音频段落")
    print("=" * 50)
    
    merged_audio, samplerate = merge_audio_segments(results)
    print(f"合并后音频: {len(merged_audio)} 采样点, 采样率 {samplerate} Hz")
    print(f"音频时长: {len(merged_audio) / samplerate:.2f} 秒")
    
    # 4. 保存音频
    print("\n" + "=" * 50)
    print("步骤 4: 保存音频文件")
    print("=" * 50)
    
    sf.write(str(OUTPUT_PATH), merged_audio, samplerate)
    print(f"已保存到: {OUTPUT_PATH}")
    print(f"文件大小: {OUTPUT_PATH.stat().st_size} 字节")


if __name__ == "__main__":
    main()
