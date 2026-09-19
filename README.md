# 🎙️ VoxTask: Voice-First Multilingual Meeting Task Extractor

VoxTask is a demo-ready, voice-first meeting and task extraction platform built for hackathons. It transcribes English, Telugu, and Tenglish (code-switched) speech using fine-tuned Telugu Wav2Vec2 (`nandinipapisetti/swecha-gonthuka-asr`) alongside `openai/whisper-small`, and extracts structured tasks using local Ollama (Mistral) with BYOK cloud fallbacks (OpenAI / Anthropic).

---

## 🏗️ Architecture

- **`app.py`**: Streamlit dashboard entrypoint (audio uploader, mic recorder, task checklist, PDF export).
- **`asr.py`**: Hybrid speech-to-text pipeline (Whisper-small + Swecha Wav2Vec2 CTC Telugu decoder + routing).
- **`extractor.py`**: Task extraction engine (Ollama local LLM + BYOK Cloud + Regex fallback).
- **`export.py`**: PDF report generator (ReportLab) & CSV exporter (pandas).
- **`config.py`**: Model settings, Ollama host configuration, API keys.
- **`create_sample_audio.py`**: Presets & synthetic WAV audio generator for instant stage rehearsal.

---

## ⚡ Quick Start & Setup

### 1. Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Optional: Run Local Ollama (Mistral)
If you have Ollama installed locally:
```bash
ollama run mistral
```
*Note: If Ollama is not running, VoxTask automatically degrades gracefully to BYOK (OpenAI/Anthropic) or the built-in offline rule-based task engine.*

### 3. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 🎯 Hackathon Stage Rehearsal Walkthrough (2-Minute Demo Script)

1. **Launch App**: Run `streamlit run app.py` and open `http://localhost:8501`.
2. **Preset Demo (Instant 1-Click)**:
   - In the sidebar under **"Quick Demo Presets"**, select **`Sprint Planning (Tenglish Code-Switched)`**.
   - Click **`⚡ Load Preset Data`**.
   - **Showcase to Judges**: Point out how Tenglish speech (`"repu morning 10 AM ki sprint report finalized gaa ready cheyyali"`) is parsed correctly into:
     - **Action Item**: Submit sprint report
     - **Owner**: Anil
     - **Deadline**: Tomorrow 10 AM
     - **Priority**: High
3. **Live Mic / WAV Upload**:
   - Click **`🎵 Generate Demo Audio WAV`** or use **`Live Microphone Recording`** to record 5 seconds of voice.
   - Click **`🚀 Transcribe & Extract Tasks`** to demonstrate live end-to-end processing.
4. **PDF / CSV Export**:
   - Navigate to the **"📊 Analytics & Export"** tab.
   - Click **`📄 Download PDF Report`** to generate a clean PDF meeting summary.

---

## 🛡️ Hackathon Safety & Fallback Notes (Read Before Going On Stage)

> [!IMPORTANT]
> 1. **ASR Model Downloads**: On the very first run, HuggingFace downloads model weights (~1GB total). If wifi at the venue is slow, use the **Quick Demo Presets** or pre-run `python asr.py` to cache weights.
> 2. **Ollama / Cloud Fallback**: If local Ollama hangs or takes >15s on CPU, switch the **Task Extractor Provider** in the sidebar to **Rule-Based Parser (Local Offline)** or **OpenAI BYOK**.
> 3. **FFmpeg on Windows**: Audio decoders use `soundfile` which reads `.wav` files natively without requiring external `ffmpeg.exe` binary installation.
