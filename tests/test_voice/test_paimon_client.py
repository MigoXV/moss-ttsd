"""TTS Demo - 使用 OpenAI SDK 调用 TTS 并保存为 WAV"""

import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def main():
    client = OpenAI(
        base_url="http://localhost:10003/v1",
        api_key="no-key",
    )

    output_dir = Path("data-bin/test_voice") / time.strftime("paimon_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_texts = [
        "诶哟，田所哥，你这开口就直戳人家心窝子，真会撩人家~~~",
        "诶，田所~~~人家可不是什么测试用的小妖精噢！",
        "来，今天我们要好好聊聊呐，你这个大色狼，一看就是想着我这边呢~~~",
        "嘿，当然是我这个专业的了，别说今天心情可好了，一整天都在等着和你这个大色狼聊天呢~~",
    ] * 10

    for i, text in enumerate(tqdm(input_texts, desc="合成语音中", leave=False)):
        output_path = output_dir / f"speech_{i}.wav"
        with client.audio.speech.with_streaming_response.create(
            model="OpenMOSS-Team/MOSS-TTSD-v0.7",
            voice="paimon",
            input=text,
            response_format="wav",
        ) as response:
            response.stream_to_file(output_path)
        tqdm.write(f"已保存到 {output_path}")


if __name__ == "__main__":
    main()
