# moss-ttsd

本项目提供一个 **OpenAI SDK 兼容** 的 TTS 服务端，实现 `POST /v1/audio/speech`，用于调用本地 `model-bin/MOSS-TTSD-v0.7` 推理并返回 `wav`。

## 运行服务

1) 安装依赖（需自行安装 `torch`；以及用于解码的 codec 模型）  
2) 启动服务：

```bash
python -m moss_ttsd.commands.app serve \
  --model-dir ./model-bin/MOSS-TTSD-v0.7 \
  --codec-path /path/to/XY_Tokenizer_TTSD_V0_hf \
  --voices-dir ./model-bin/voices \
  --host 0.0.0.0 --port 10003
```

如果未提供 `codec-path`（或 `MOSS_TTSD_CODEC_PATH`），默认会直接报错；如需“占位 WAV”以便先打通 OpenAI SDK 流程，可加 `--fallback-audio dummy`。

## 音色目录（voices_dir）

`voices_dir` 下每个子目录就是一个音色名（对应 OpenAI SDK 的 `voice` 参数，支持形如 `repo:voice_name`，会取最后一段 `voice_name`）。

目录结构：

```
model-bin/voices/
  man01/
    audio.wav
    transcription.txt
```

- `audio.wav`：参考音频
- `transcription.txt`：参考音频对应文本（可包含 `[S1]...[S2]...`）

## 运行客户端 Demo

先启动服务，然后：

```bash
python tests/demo_tts.py
```
