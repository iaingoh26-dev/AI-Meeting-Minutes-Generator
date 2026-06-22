# 📝 AI Meeting Minutes Generator

Ever wish you could skip re-listening to a long meeting recording just to write up the notes? This project does it for you. Give it an audio file, and it will transcribe what was said and automatically produce structured meeting minutes — summary, key points, decisions, and action items.

> ✅ Runs for free on Google Colab using a T4 GPU (no local setup needed).

---

## 📋 Table of Contents

- [What Does It Do?](#what-does-it-do)
- [How It Works — In Plain English](#how-it-works--in-plain-english)
- [Before You Start](#before-you-start)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Install the tools](#1-install-the-tools)
  - [2. Connect your Google Drive](#2-connect-your-google-drive)
  - [3. Get your audio file](#3-get-your-audio-file)
  - [4. Transcribe the audio](#4-transcribe-the-audio)
  - [5. Generate the meeting minutes](#5-generate-the-meeting-minutes)
- [Choosing a Transcription Method](#choosing-a-transcription-method)
- [Sample Data](#sample-data)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## What Does It Do?

You give it a meeting recording (.mp3), and it produces a neatly formatted document like this:

```
## Meeting Summary
Date: March 5, 2024 | Location: Denver City Hall

## Discussion Points
- Budget allocation for Q2 infrastructure projects
- ...

## Takeaways
- Council agreed to defer the zoning vote
- ...

## Action Items
- [ ] John to submit revised proposal by March 12
- [ ] Finance team to review budget amendments
```

---

## How It Works — In Plain English

The notebook runs two steps automatically:

**Step 1 — Listen & Transcribe**
It listens to your audio file and converts the spoken words into text. Think of it like a very accurate auto-caption tool.

**Step 2 — Understand & Summarise**
It feeds that text to an AI language model (similar to ChatGPT, but running locally on the Colab machine) and asks it to write up proper meeting minutes.

No audio ever leaves your Colab session unless you choose to use OpenAI's transcription service (see below).

---

## Before You Start

You'll need:

- A **free Google account** (to use Google Colab and Google Drive)
- A **Hugging Face account** — free at [huggingface.co](https://huggingface.co) — to download the AI models
- *(Optional)* An **OpenAI API key** if you want to use OpenAI's transcription instead of the free open-source option

**Set up your secret keys in Colab:**

Go to the 🔑 **Secrets** panel (padlock icon in the left sidebar) and add:
- `HF_TOKEN` — your Hugging Face token (found at huggingface.co → Settings → Access Tokens)
- `OPENAI_API_KEY` — only needed if using OpenAI transcription

---

## Step-by-Step Guide

### 1. Install the tools

The first cell in the notebook installs everything automatically. Just run it and wait about a minute.

```
Installs: bitsandbytes, accelerate, transformers
```

> ⚠️ The version of `transformers` is fixed to `4.57.6` on purpose — newer versions can cause compatibility errors on Colab. Don't change it.

---

### 2. Connect your Google Drive

The notebook needs to read your audio file from Google Drive. When you run this cell, a popup will ask you to sign in and grant permission — just follow the prompts.

---

### 3. Get your audio file

You have two options:

**Option A — Use the provided sample** (recommended for first-timers)
Download this Denver City Council meeting extract and upload it to your Google Drive:
👉 [Download denver_extract.mp3](https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing)

Place it here on your Drive: `My Drive → llms → denver_extract.mp3`

**Option B — Use your own recording**
Put your own `.mp3` file in the same folder and update the filename in the notebook.

---

### 4. Transcribe the audio

This step converts the audio into text. You have two methods to choose from — pick one (see [Choosing a Transcription Method](#choosing-a-transcription-method) below).

The transcription will be printed to the screen so you can review it before moving on.

---

### 5. Generate the meeting minutes

This is where the magic happens. The notebook sends the transcribed text to a local AI model (LLaMA 3.2) with instructions to produce structured minutes. 

The model runs entirely on the Colab machine — no data is sent anywhere externally.

You'll see the minutes being written out in real time as the model generates them, then a nicely formatted version is displayed at the end.

---

## Choosing a Transcription Method

| | Option A: Free (Open Source) | Option B: OpenAI API |
|---|---|---|
| **Cost** | Free | Costs a small amount per minute of audio |
| **Needs API key?** | No | Yes (`OPENAI_API_KEY`) |
| **Speed** | Moderate | Fast |
| **Quality** | Very good | Excellent |
| **Privacy** | Audio stays on Colab | Audio sent to OpenAI servers |

**Recommendation:** Start with Option A (free). Switch to Option B if you need faster or higher-quality transcription.

---

## Sample Data

The included example uses a real excerpt from a **Denver City Council meeting**.

| Resource | Link |
|----------|------|
|  Audio extract (the file you need) | [Download from Google Drive](https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing) |
|  Full meeting dataset | [MeetingBank on HuggingFace](https://huggingface.co/datasets/huuuyeah/meetingbank) |
|  Full audio dataset | [MeetingBank Audio](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main) |
