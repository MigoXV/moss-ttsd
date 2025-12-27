"""TTS Demo - 使用 OpenAI SDK 调用 TTS 并保存为 WAV"""

from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10003/v1",
    api_key="no-key",
)

output_path = Path("data-bin/speech.wav")
output_path.parent.mkdir(parents=True, exist_ok=True)

with client.audio.speech.with_streaming_response.create(
    model="OpenMOSS-Team/MOSS-TTSD-v0.7",
    voice="man01",
    input="你好，这是一段测试语音。",
    response_format="wav",
) as response:
    response.stream_to_file(output_path)

print(f"已保存到 {output_path}")
