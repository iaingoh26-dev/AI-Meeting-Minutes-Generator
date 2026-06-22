# 📝 AI Meeting Minutes Generator

A Google Colab notebook that transcribes an audio recording of a meeting and automatically generates structured meeting minutes — including summary, discussion points, takeaways, and action items — using open-source and OpenAI models.

> ✅ Runs on a free/low-cost **T4 GPU** runtime in Google Colab.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Step 1 — Transcribe Audio](#step-1--transcribe-audio)
  - [Option A: Open Source (Whisper via HuggingFace)](#option-a-open-source-whisper-via-huggingface)
  - [Option B: OpenAI API](#option-b-openai-api)
- [Step 2 — Generate Meeting Minutes (LLaMA)](#step-2--generate-meeting-minutes-llama)
- [Sample Data](#sample-data)
- [Known Issues & Notes](#known-issues--notes)
- [Credits](#credits)

---

## Overview

This notebook implements a two-step pipeline:

```
Audio File (.mp3)
      │
      ▼
 [STEP 1] Transcription
  ├── Option A: Whisper (open-source, runs on GPU)
  └── Option B: OpenAI gpt-4o-mini-transcribe (API)
      │
      ▼
 [STEP 2] Meeting Minutes Generation
  └── LLaMA 3.2-3B-Instruct (4-bit quantized, local)
      │
      ▼
 Markdown Meeting Minutes
  ├── Summary (attendees, location, date)
  ├── Discussion points
  ├── Takeaways
  └── Action items with owners
```

---

## Setup

### Install Dependencies

```bash
pip install -q --upgrade bitsandbytes accelerate transformers==4.57.6
```

### Imports

```python
import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
```

### Model

```python
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
```

### Mount Google Drive & Set Audio Path

Place your audio file in Google Drive under `MyDrive/llms/` named `denver_extract.mp3`:

```python
from google.colab import drive
drive.mount("/content/drive")
audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"
```

### Authenticate

```python
from huggingface_hub import login
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

audio_file = open(audio_filename, "rb")
```

> 💡 Store `HF_TOKEN` and `OPENAI_API_KEY` as Colab secrets under **Runtime → Manage secrets**.

---

## Step 1 — Transcribe Audio

Two options are provided. You can compare their outputs side by side:

```python
display(Markdown(open_source_transcription))
print("\n\n")
display(Markdown(transcription))  # OpenAI version
```

### Option A: Open Source (Whisper via HuggingFace)

Runs locally on the T4 GPU — no API key required.

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device='cuda',
    return_timestamps=True
)

result = pipe(audio_filename)
transcription = result["text"]
open_source_transcription = transcription
print(transcription)
```

### Option B: OpenAI API

Uses `gpt-4o-mini-transcribe` — fast and high quality, requires an OpenAI API key.

```python
AUDIO_MODEL = "gpt-4o-mini-transcribe"

openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)

transcription = openai.audio.transcriptions.create(
    model=AUDIO_MODEL,
    file=audio_file,
    response_format="text"
)
print(transcription)
```

---

## Step 2 — Generate Meeting Minutes (LLaMA)

Uses LLaMA 3.2-3B-Instruct quantized to 4-bit to generate structured minutes from the transcription.

### Prompt

```python
system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
```

### Quantization Config

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

### Load Model & Generate

```python
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config
)

outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
response = tokenizer.decode(outputs[0])
display(Markdown(response))
```

---

## Sample Data

The notebook uses an extract from **Denver City Council meeting audio**.

| Resource | Link |
|----------|------|
| Audio extract (MP3) | [Google Drive](https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing) |
| Full HuggingFace dataset | [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank) |
| Full audio dataset | [MeetingBank Audio](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main) |

You can also **record your own audio** and use that instead.

---

## Known Issues & Notes

| # | Issue | Impact |
|---|-------|--------|
| 1 | `tokenizer.decode(outputs[0])` decodes the **entire sequence** including the input prompt | The displayed Markdown will include the prompt tokens before the actual response; slice with `outputs[0][inputs.shape[-1]:]` to get only the generated text |
| 2 | `transformers==4.57.6` is pinned | Required for compatibility with `bitsandbytes` on Colab — do not upgrade without testing |
| 3 | `.to("cuda")` is hardcoded | Will crash without a GPU; safe on Colab T4 but not portable |
| 4 | `audio_file` file handle is opened once and never closed | If OpenAI transcription is retried, the handle may be exhausted; re-open before each call |
| 5 | `TextStreamer` streams to stdout only | Output is printed live during generation but not captured for downstream use; use `TextIteratorStreamer` if you need to process or display the output in a UI (see Credits) |

### Fix for Issue #1 — Decode only the generated response

```python
# Instead of:
response = tokenizer.decode(outputs[0])

# Use:
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
```

### Fix for Issue #4 — Re-open audio file before OpenAI call

```python
audio_file = open(audio_filename, "rb")
transcription = openai.audio.transcriptions.create(
    model=AUDIO_MODEL, file=audio_file, response_format="text"
)
audio_file.close()
```

### ⚠️ Common Colab Runtime Error

If you encounter:
```
Runtime error: CUDA is required but not available for bitsandbytes.
```

This is **not** a package version problem. Google silently swapped your runtime. Fix:

1. **Kernel → Disconnect and delete runtime**
2. Reload the notebook and **Edit → Clear All Outputs**
3. Reconnect to a fresh **T4 GPU** runtime
4. Confirm GPU via **View resources** (top-right menu)
5. Re-run all cells from the top

---

## Credits

- **Sample data:** Denver City Council via [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank)
- **Streaming Gradio UI variation** by student **Emad S.** — uses `TextIteratorStreamer` and background threads for a real-time streaming chat interface:  
  [View on Colab](https://colab.research.google.com/drive/1Ja5zyniyJo5y8s1LKeCTSkB2xyDPOt6D)
