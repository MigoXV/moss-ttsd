"""TTS Demo - 使用 OpenAI SDK 调用 TTS 并保存为 WAV（流式响应）

OpenAI SDK 的 with_streaming_response 会以流式方式接收服务端响应，
这样可以在音频还没完全生成时就开始接收数据，适合大文件或需要低延迟的场景。
"""

from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10003/v1",
    api_key="no-key",
)

output_path = Path("data-bin/speech_stream.wav")
output_path.parent.mkdir(parents=True, exist_ok=True)

print("开始流式请求...")

# 使用流式响应接收音频数据
with client.audio.speech.with_streaming_response.create(
    model="OpenMOSS-Team/MOSS-TTSD-v0.7",
    voice="man01",
    input="你好，这是一段流式传输的测试语音。通过流式传输，客户端可以在音频生成的同时开始接收数据。",
    response_format="wav",
) as response:
    # 分块写入文件
    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes(chunk_size=4096):
            f.write(chunk)
            total_bytes += len(chunk)
            print(f"已接收 {total_bytes} 字节...")

print(f"已保存到 {output_path}，共 {total_bytes} 字节")
