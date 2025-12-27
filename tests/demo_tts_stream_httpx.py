"""TTS Demo - 使用 httpx 直接调用流式 TTS API

这个示例展示如何使用 httpx 库直接调用流式 TTS API，
通过设置 stream=true 参数来启用服务端的流式生成模式。
"""

from pathlib import Path

import httpx

base_url = "http://localhost:10003"
output_path = Path("data-bin/speech_httpx_stream.wav")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("开始流式请求...")

# 使用 httpx 发送流式请求
with httpx.stream(
    "POST",
    f"{base_url}/v1/audio/speech",
    json={
        "model": "OpenMOSS-Team/MOSS-TTSD-v0.7",
        "voice": "man01",
        "input": "你好，这是一段使用 httpx 进行流式传输的测试语音。流式传输可以减少首字节延迟。",
        "response_format": "wav",
        "stream": True,  # 启用流式模式
    },
    timeout=120.0,
) as response:
    response.raise_for_status()
    
    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes(chunk_size=4096):
            f.write(chunk)
            total_bytes += len(chunk)
            print(f"已接收 {total_bytes} 字节...")

print(f"已保存到 {output_path}，共 {total_bytes} 字节")
