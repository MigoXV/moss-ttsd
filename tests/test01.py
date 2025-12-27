import os
import torchaudio
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained(
    "./model-bin/MOSS-TTSD-v0.7", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "./model-bin/MOSS-TTSD-v0.7", trust_remote_code=True, device_map="auto"
).eval()

data = [
    {
        "base_path": "/path/to/audio/files",
        "text": "[S1]Speaker 1 dialogue content[S2]Speaker 2 dialogue content[S1]...",
        "prompt_audio": "path/to/shared_reference_audio.wav",
        "prompt_text": "[S1]Reference text for speaker 1[S2]Reference text for speaker 2",
    }
]

inputs = processor(data)
token_ids = model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)
text, audios = processor.batch_decode(token_ids)

if not os.path.exists("outputs/"):
    os.mkdir("outputs/")
for i, data in enumerate(audios):
    for j, fragment in enumerate(data):
        torchaudio.save(f"outputs/audio_{i}_{j}.wav", fragment.cpu(), 24000)
