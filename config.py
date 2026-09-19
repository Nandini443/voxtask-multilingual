import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # ASR Models
    WHISPER_MODEL_NAME: str = "openai/whisper-small"
    TELUGU_ASR_MODEL_NAME: str = "nandinipapisetti/swecha-gonthuka-asr"
    AUDIO_SAMPLE_RATE: int = 16000
    
    # LLM Settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    
    # Default extraction backend: "ollama", "openai", "anthropic", "regex"
    LLM_BACKEND: str = os.getenv("VOXTASK_LLM_BACKEND", "ollama")
    
    # Cloud BYOK Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Mock / Demo Mode
    USE_DEMO_MOCK_IF_UNAVAILABLE: bool = True

config = AppConfig()
