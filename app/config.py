"""
config.py
Application settings. Override via environment variables.
"""

import os
import torch


class Settings:
    """App configuration loaded from environment variables."""
    
    # Path to cloned AASIST repository
    AASIST_DIR: str = os.getenv("AASIST_DIR", "./aasist")
    
    # Model variant: "AASIST" (full) or "AASIST-L" (lightweight)
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "AASIST-L")
    
    # Device: "cuda" or "cpu" (auto-detected if not set)
    DEVICE: str = os.getenv(
        "DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
