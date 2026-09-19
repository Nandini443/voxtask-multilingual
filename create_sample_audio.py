import os
import numpy as np
import soundfile as sf

SAMPLE_PRESETS = {
    "Sprint Planning (Tenglish Code-Switched)": {
        "description": "Code-switched Telugu + English sprint planning discussion with deadlines.",
        "segments": [
            {
                "start": 0.0,
                "end": 4.5,
                "speaker": "Speaker 1 (Anil)",
                "language": "Tenglish (Code-Switched)",
                "text": "Hi team, repu morning 10 AM ki sprint report finalized gaa ready cheyyali. Priya will handle the API docs.",
                "engine": "Whisper-small + Swecha ASR"
            },
            {
                "start": 4.5,
                "end": 9.2,
                "speaker": "Speaker 2 (Priya)",
                "language": "Tenglish (Code-Switched)",
                "text": "Sure Anil! Repu afternoon 2 PM kalla API documentation complete chestanu. Rahul please verify the database schema by Friday.",
                "engine": "Whisper-small + Swecha ASR"
            },
            {
                "start": 9.2,
                "end": 14.0,
                "speaker": "Speaker 3 (Rahul)",
                "language": "English",
                "text": "Got it. I will review the database migration scripts and post updates on Slack by end of day tomorrow.",
                "engine": "Whisper-small"
            }
        ]
    },
    "Client Deliverable (Telugu Dominant)": {
        "description": "Telugu-dominant conversation regarding design reviews and client presentation.",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Speaker 1 (Sneha)",
                "language": "Telugu",
                "text": "Ee client presentation design fast gaa modify cheyyali, color theme customer request chesina format lo undali.",
                "engine": "Swecha-Wav2Vec2"
            },
            {
                "start": 5.0,
                "end": 9.8,
                "speaker": "Speaker 2 (Kiran)",
                "language": "Tenglish (Code-Switched)",
                "text": "Nenu modern dark theme apply chestanu. Today evening 6 PM kalla slide deck share chestanu.",
                "engine": "Swecha-Wav2Vec2 + Whisper"
            },
            {
                "start": 9.8,
                "end": 14.5,
                "speaker": "Speaker 1 (Sneha)",
                "language": "English",
                "text": "Great! Also ensure all chart figures match the quarterly revenue targets.",
                "engine": "Whisper-small"
            }
        ]
    },
    "Emergency Bug Fix (Tech Sync)": {
        "description": "Technical sync call about urgent production hotfix.",
        "segments": [
            {
                "start": 0.0,
                "end": 4.2,
                "speaker": "Speaker 1 (Suresh)",
                "language": "English",
                "text": "We detected a memory leak in the background worker process. This is URGENT.",
                "engine": "Whisper-small"
            },
            {
                "start": 4.2,
                "end": 8.8,
                "speaker": "Speaker 2 (Ravi)",
                "language": "Tenglish (Code-Switched)",
                "text": "Nenu immediately memory dump inspect chesi hotfix release chestanu. By 4 PM today issue resolved aipotundi.",
                "engine": "Whisper-small + Swecha ASR"
            }
        ]
    }
}

def generate_sample_wav_file(filename: str = "sample_tenglish.wav", duration_sec: float = 5.0, sr: int = 16000) -> str:
    """Generates a clean synthetic audio WAV file for testing audio upload components."""
    os.makedirs("samples", exist_ok=True)
    filepath = os.path.join("samples", filename)
    
    t = np.linspace(0, duration_sec, int(sr * duration_sec), False)
    # Generate multi-tone synthetic audio signal representing voice frequencies
    audio_signal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.sin(2 * np.pi * 220 * t)
    audio_signal = audio_signal * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) # amplitude modulation
    
    sf.write(filepath, audio_signal.astype(np.float32), sr)
    return filepath

if __name__ == "__main__":
    path = generate_sample_wav_file()
    print(f"Sample WAV generated at: {os.path.abspath(path)}")
