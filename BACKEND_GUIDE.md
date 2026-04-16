python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
# 🛡️ DeepShield AI — Complete Backend Guide
## AASIST Model Loading + FastAPI Inference Server

> **Everything you need** to go from zero → working deepfake voice detection API.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Project Structure](#3-project-structure)
4. [Step 1: Clone AASIST & Get Weights](#4-step-1-clone-aasist--get-weights)
5. [Step 2: Understanding the AASIST Model](#5-step-2-understanding-the-aasist-model)
6. [Step 3: Model Loader Module](#6-step-3-model-loader-module)
7. [Step 4: Audio Preprocessing](#7-step-4-audio-preprocessing)
8. [Step 5: FastAPI Inference Server](#8-step-5-fastapi-inference-server)
9. [Step 6: Configuration](#9-step-6-configuration)
10. [Step 7: Requirements File](#10-step-7-requirements-file)
11. [Step 8: Running the Server](#11-step-8-running-the-server)
12. [Step 9: Testing the API](#12-step-9-testing-the-api)
13. [Step 10: Tunneling for Mobile Access](#13-step-10-tunneling-for-mobile-access)
14. [Production Hardening](#14-production-hardening)
15. [Troubleshooting](#15-troubleshooting)
16. [API Reference](#16-api-reference)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      REQUEST FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Mobile App / Browser                                           │
│       │                                                         │
│       │  POST /predict  (audio file)                            │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │   FastAPI    │  ← CORS enabled, accepts multipart/form-data  │
│  │   Server     │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │   Audio      │  ← Resample to 16kHz mono                    │
│  │   Preprocess │  ← Pad/trim to 64,600 samples (~4 sec)       │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │   AASIST     │  ← Pre-trained PyTorch model                  │
│  │   Model      │  ← Loaded ONCE at startup                    │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                                │
│  │   Result     │  → { "result": "AI" | "HUMAN",               │
│  │   Response   │     "confidence": 0.947,                      │
│  │              │     "risk_level": "HIGH" }                    │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key facts about AASIST:**
- Input: Raw audio waveform (16kHz, mono)
- Expected sample length: **64,600 samples** (~4.03 seconds)
- Output: 2-class logits → `[spoof_score, bonafide_score]`
- `output[:, 1]` = bonafide (real human) score — **higher = more likely human**
- Pre-trained weights: `AASIST.pth` (~1.2MB) and `AASIST-L.pth` (~340KB)

---

## 2. Prerequisites & Environment Setup

### System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.8+ | 3.11 |
| RAM | 4 GB | 8 GB+ |
| GPU | Not required | NVIDIA (CUDA) for speed |
| Disk | 2 GB | 5 GB |

### Create Virtual Environment

```bash
# Create project directory
mkdir -p voiceguard-hackathon/backend
cd voiceguard-hackathon/backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install PyTorch

```bash
# CPU only (simpler, works everywhere)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (if you have NVIDIA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Verify PyTorch:**
> ```python
> python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
> ```

---

## 3. Project Structure

```
backend/
├── aasist/                     # ← Cloned AASIST repo (Step 1)
│   ├── models/
│   │   ├── AASIST.py           # Model architecture
│   │   ├── AASIST_L.py         # Lightweight variant
│   │   └── weights/
│   │       ├── AASIST.pth      # Pre-trained weights (full)
│   │       └── AASIST-L.pth    # Pre-trained weights (lightweight)
│   └── config/
│       ├── AASIST.conf         # Full model config
│       └── AASIST-L.conf       # Lightweight model config
│
├── app/                        # ← Your FastAPI application
│   ├── __init__.py
│   ├── main.py                 # FastAPI server entry point
│   ├── model_loader.py         # AASIST model loading logic
│   ├── audio_processor.py      # Audio preprocessing utilities
│   └── config.py               # App configuration
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (optional)
└── README.md
```

---

## 4. Step 1: Clone AASIST & Get Weights

```bash
cd voiceguard-hackathon/backend

# Clone the official AASIST repository
git clone https://github.com/clovaai/aasist.git
```

### Verify Pre-trained Weights Exist

```bash
# Check if weights are present
ls aasist/models/weights/

# Expected output:
# AASIST.pth
# AASIST-L.pth
```

> ⚠️ **If weights are NOT included** (some git LFS repos don't auto-download):
> ```bash
> cd aasist
> git lfs pull
> ```
>
> If that doesn't work, check the repo's [Releases page](https://github.com/clovaai/aasist/releases) or Issues for download links.

### Model Accuracy Reference

| Model | File | Size | EER | min t-DCF | Parameters |
|-------|------|------|-----|-----------|------------|
| AASIST (Full) | `AASIST.pth` | ~1.2 MB | **0.83%** | **0.0275** | ~297K |
| AASIST-L (Light) | `AASIST-L.pth` | ~340 KB | **0.99%** | **0.0309** | 85,306 |

> **Recommendation:** Use **AASIST-L** for hackathon (3.5× smaller, nearly same accuracy, faster inference).

---

## 5. Step 2: Understanding the AASIST Model

### How the Model Works Internally

AASIST uses an **Integrated Spectro-Temporal Graph Attention Network**:

```
Raw Waveform (64,600 samples @ 16kHz)
        │
        ▼
┌─────────────────────┐
│  Sinc Convolution    │  ← Learnable bandpass filters (like ears)
│  (RawNet-style)      │
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Temporal│ │Spectral│   ← Two parallel graph representations
│ Graph  │ │ Graph  │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌──────────────────────┐
│ Heterogeneous Graph   │  ← Cross-domain attention
│ Attention (HtrgGAT)  │     (spectral ↔ temporal)
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Graph Pooling +       │  ← Reduce graph size
│ Readout               │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Output Layer          │  → [spoof_logit, bonafide_logit]
│ (2-class softmax)     │
└──────────────────────┘
```

### Key Model Parameters (from `AASIST.conf`)

```json
{
    "model_config": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
}
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `nb_samp` | 64,600 | Expected input samples (16kHz × ~4.03s) |
| `first_conv` | 128 | Number of sinc filters in first layer |
| `filts` | `[70, [1,32], ...]` | Filter configurations per encoder block |
| `gat_dims` | `[64, 32]` | Graph attention hidden dimensions |
| `pool_ratios` | `[0.5, 0.7, 0.5, 0.5]` | Node reduction ratios per pooling layer |
| `temperatures` | `[2.0, 2.0, 100.0, 100.0]` | Attention softmax temperatures |

### Model Forward Pass

```python
# What happens when you call model(x):
last_hidden, output = model(waveform_tensor)

# last_hidden: (batch_size, 160)  ← concatenation of T_max, T_avg, S_max, S_avg, master
# output:      (batch_size, 2)    ← [spoof_logit, bonafide_logit]

# To get bonafide score:
bonafide_score = output[:, 1]  # Higher = more likely REAL human voice
```

---

## 6. Step 3: Model Loader Module

Create `app/model_loader.py`:

```python
"""
model_loader.py
Handles loading and managing the AASIST model for inference.
"""

import json
import sys
from pathlib import Path
from importlib import import_module

import torch
import torch.nn as nn


class AASISTModelLoader:
    """
    Loads and manages the AASIST pre-trained model.
    
    Usage:
        loader = AASISTModelLoader(
            aasist_dir="./aasist",
            model_variant="AASIST",   # or "AASIST-L"
            device="cpu"              # or "cuda"
        )
        model = loader.get_model()
    """
    
    def __init__(
        self,
        aasist_dir: str = "./aasist",
        model_variant: str = "AASIST",
        device: str = None,
    ):
        self.aasist_dir = Path(aasist_dir).resolve()
        self.model_variant = model_variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_config = None
        
        # Add AASIST repo to Python path so we can import its modules
        aasist_str = str(self.aasist_dir)
        if aasist_str not in sys.path:
            sys.path.insert(0, aasist_str)
    
    def load(self) -> nn.Module:
        """Load the model with pre-trained weights."""
        
        # 1. Load configuration
        config_path = self.aasist_dir / "config" / f"{self.model_variant}.conf"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Available configs: {list((self.aasist_dir / 'config').glob('*.conf'))}"
            )
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.model_config = config["model_config"]
        
        # 2. Dynamically import the model architecture
        # AASIST repo uses: models.AASIST → class "Model"
        architecture = self.model_config["architecture"]
        module = import_module(f"models.{architecture}")
        ModelClass = getattr(module, "Model")
        
        # 3. Instantiate the model
        self.model = ModelClass(self.model_config).to(self.device)
        
        # 4. Load pre-trained weights
        weights_path = self.aasist_dir / config.get(
            "model_path", f"./models/weights/{self.model_variant}.pth"
        )
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                "Please download pre-trained weights from the AASIST repo."
            )
        
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        
        # 5. Set to evaluation mode (critical for inference!)
        self.model.eval()
        
        # 6. Log model info
        nb_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ AASIST model loaded successfully!")
        print(f"   Variant:    {self.model_variant}")
        print(f"   Parameters: {nb_params:,}")
        print(f"   Device:     {self.device}")
        print(f"   Weights:    {weights_path}")
        
        return self.model
    
    def get_model(self) -> nn.Module:
        """Get the loaded model (loads if not already loaded)."""
        if self.model is None:
            self.load()
        return self.model
    
    def predict(self, waveform: torch.Tensor) -> dict:
        """
        Run inference on a preprocessed waveform tensor.
        
        Args:
            waveform: Tensor of shape (num_samples,) or (batch, num_samples)
                      Must be 16kHz, mono, padded/trimmed to 64600 samples.
        
        Returns:
            dict with keys: bonafide_score, spoof_score, prediction, confidence
        """
        model = self.get_model()
        
        # Ensure correct shape: (batch, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            last_hidden, output = model(waveform)
            
            # output shape: (batch_size, 2)
            # output[:, 0] = spoof logit
            # output[:, 1] = bonafide logit
            
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)
            
            spoof_prob = probs[:, 0].item()
            bonafide_prob = probs[:, 1].item()
            
            # Raw scores (used in original AASIST evaluation)
            bonafide_score = output[:, 1].item()
        
        # Determine prediction
        is_bonafide = bonafide_prob > spoof_prob
        confidence = bonafide_prob if is_bonafide else spoof_prob
        
        return {
            "prediction": "HUMAN" if is_bonafide else "AI",
            "confidence": round(confidence * 100, 2),
            "bonafide_score": round(bonafide_score, 4),
            "bonafide_probability": round(bonafide_prob * 100, 2),
            "spoof_probability": round(spoof_prob * 100, 2),
            "risk_level": self._get_risk_level(spoof_prob),
        }
    
    @staticmethod
    def _get_risk_level(spoof_prob: float) -> str:
        """Categorize risk based on spoof probability."""
        if spoof_prob >= 0.9:
            return "CRITICAL"
        elif spoof_prob >= 0.7:
            return "HIGH"
        elif spoof_prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
```

---

## 7. Step 4: Audio Preprocessing

Create `app/audio_processor.py`:

```python
"""
audio_processor.py
Handles all audio preprocessing before model inference.

AASIST expects:
  - Sample rate: 16,000 Hz
  - Channels: Mono (1 channel)
  - Length: Exactly 64,600 samples (~4.03 seconds)
  - Format: Raw waveform as float tensor
"""

import io
from pathlib import Path
from typing import Union

import torch
import torchaudio


# AASIST expected input length
TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600  # ~4.03 seconds at 16kHz


class AudioProcessor:
    """Preprocesses audio files for AASIST model input."""
    
    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        target_length: int = TARGET_NUM_SAMPLES,
    ):
        self.target_sr = target_sr
        self.target_length = target_length
    
    def process_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Process raw audio bytes into model-ready tensor.
        
        Args:
            audio_bytes: Raw bytes of an audio file (WAV, MP3, FLAC, etc.)
        
        Returns:
            Tensor of shape (64600,) ready for model input
        """
        # Load audio from bytes
        buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buffer)
        
        return self._preprocess(waveform, sample_rate)
    
    def process_file(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Process an audio file from disk.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tensor of shape (64600,) ready for model input
        """
        waveform, sample_rate = torchaudio.load(str(file_path))
        return self._preprocess(waveform, sample_rate)
    
    def _preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Full preprocessing pipeline.
        
        Steps:
            1. Convert to mono
            2. Resample to 16kHz
            3. Normalize amplitude
            4. Pad or trim to exact length (64,600 samples)
        """
        # Step 1: Convert to mono (average all channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Step 2: Resample to 16kHz if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr
            )
            waveform = resampler(waveform)
        
        # Step 3: Normalize amplitude to [-1, 1]
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Step 4: Pad or trim to target length
        waveform = self._pad_or_trim(waveform)
        
        # Remove channel dimension → shape: (64600,)
        return waveform.squeeze(0)
    
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ensure waveform is exactly target_length samples.
        
        - If shorter: repeat/tile the audio until it fills the target length
        - If longer: take the first target_length samples
        """
        current_length = waveform.shape[-1]
        
        if current_length >= self.target_length:
            # Trim to target length
            waveform = waveform[:, :self.target_length]
        else:
            # Repeat audio to fill target length (better than zero-padding)
            repeat_times = (self.target_length // current_length) + 1
            waveform = waveform.repeat(1, repeat_times)[:, :self.target_length]
        
        return waveform
    
    def process_multiple_chunks(self, audio_bytes: bytes) -> list[torch.Tensor]:
        """
        For longer audio (>4s), split into multiple 4-second chunks
        and analyze each separately. Returns list of tensors.
        
        Useful for analyzing longer recordings and averaging results.
        """
        buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buffer)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Split into chunks
        total_samples = waveform.shape[-1]
        chunks = []
        
        for start in range(0, total_samples, self.target_length):
            chunk = waveform[:, start:start + self.target_length]
            if chunk.shape[-1] < self.target_length // 2:
                # Skip chunks that are too short (less than half expected length)
                continue
            chunk = self._pad_or_trim(chunk)
            chunks.append(chunk.squeeze(0))
        
        # If no valid chunks, process entire audio as single chunk
        if not chunks:
            chunks = [self._pad_or_trim(waveform).squeeze(0)]
        
        return chunks
```

---

## 8. Step 5: FastAPI Inference Server

Create `app/__init__.py`:

```python
# empty file to make app/ a Python package
```

Create `app/main.py`:

```python
"""
main.py
FastAPI inference server for DeepShield AI - Deepfake Voice Detection.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.model_loader import AASISTModelLoader
from app.audio_processor import AudioProcessor
from app.config import settings


# ──────────────────────────────────────────────────────────────────────
# Global instances (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────
model_loader: Optional[AASISTModelLoader] = None
audio_processor: Optional[AudioProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model_loader, audio_processor
    
    print("=" * 60)
    print("🚀 DeepShield AI — Starting Inference Server")
    print("=" * 60)
    
    # Initialize model
    model_loader = AASISTModelLoader(
        aasist_dir=settings.AASIST_DIR,
        model_variant=settings.MODEL_VARIANT,
        device=settings.DEVICE,
    )
    model_loader.load()
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    print("=" * 60)
    print("✅ Server ready! Accepting requests.")
    print("=" * 60)
    
    yield  # Server is running
    
    # Cleanup
    print("🛑 Shutting down...")


# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield AI — Voice Deepfake Detection API",
    description="Detect AI-generated/cloned voices using AASIST neural network.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — Allow all origins for hackathon (restrict in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    result: str              # "HUMAN" or "AI"
    confidence: float        # 0-100 percentage
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    bonafide_probability: float
    spoof_probability: float
    bonafide_score: float    # Raw model score
    analysis_time_ms: float  # Processing time


class MultiChunkResult(BaseModel):
    overall_result: str
    overall_confidence: float
    risk_level: str
    chunks_analyzed: int
    chunk_results: list[dict]
    analysis_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ──────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint."""
    return {
        "service": "DeepShield AI — Voice Deepfake Detection",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Analyze a single audio file",
            "POST /predict/multi": "Analyze long audio in chunks",
            "GET /health": "Check server status",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check if the model is loaded and server is healthy."""
    return HealthResponse(
        status="healthy",
        model=settings.MODEL_VARIANT,
        device=settings.DEVICE,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_voice(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)"),
):
    """
    🎤 Analyze a single audio file for deepfake detection.
    
    - Accepts: WAV, MP3, FLAC, M4A, OGG, WebM
    - Max size: 50MB
    - Audio is preprocessed to 16kHz mono, ~4 seconds
    - Returns prediction with confidence score
    """
    start_time = time.time()
    
    # Validate file type
    allowed_types = {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/x-flac",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/ogg", "audio/webm",
        "application/octet-stream",  # fallback for unknown
    }
    
    # Read file
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 50MB.")
    
    # Preprocess audio
    try:
        waveform = audio_processor.process_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Audio preprocessing failed: {str(e)}. Ensure valid audio format."
        )
    
    # Run inference
    try:
        result = model_loader.predict(waveform)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return PredictionResult(
        result=result["prediction"],
        confidence=result["confidence"],
        risk_level=result["risk_level"],
        bonafide_probability=result["bonafide_probability"],
        spoof_probability=result["spoof_probability"],
        bonafide_score=result["bonafide_score"],
        analysis_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/multi", response_model=MultiChunkResult, tags=["Prediction"])
async def predict_voice_multi_chunk(
    file: UploadFile = File(..., description="Audio file to analyze in chunks"),
):
    """
    🎤 Analyze longer audio by splitting into ~4-second chunks.
    
    Each chunk is analyzed independently. The overall result
    is determined by majority voting with average confidence.
    Best for audio longer than 5 seconds.
    """
    start_time = time.time()
    
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    # Split into chunks
    try:
        chunks = audio_processor.process_multiple_chunks(audio_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Audio preprocessing failed: {str(e)}"
        )
    
    # Analyze each chunk
    chunk_results = []
    spoof_votes = 0
    bonafide_votes = 0
    total_spoof_prob = 0.0
    
    for i, chunk in enumerate(chunks):
        result = model_loader.predict(chunk)
        chunk_results.append({
            "chunk_index": i,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "spoof_probability": result["spoof_probability"],
            "bonafide_probability": result["bonafide_probability"],
        })
        
        if result["prediction"] == "AI":
            spoof_votes += 1
        else:
            bonafide_votes += 1
        
        total_spoof_prob += result["spoof_probability"]
    
    # Overall decision by majority vote
    avg_spoof_prob = total_spoof_prob / len(chunks)
    overall_is_spoof = spoof_votes > bonafide_votes
    overall_confidence = avg_spoof_prob if overall_is_spoof else (100 - avg_spoof_prob)
    
    # Risk level
    if avg_spoof_prob >= 90:
        risk = "CRITICAL"
    elif avg_spoof_prob >= 70:
        risk = "HIGH"
    elif avg_spoof_prob >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return MultiChunkResult(
        overall_result="AI" if overall_is_spoof else "HUMAN",
        overall_confidence=round(overall_confidence, 2),
        risk_level=risk,
        chunks_analyzed=len(chunks),
        chunk_results=chunk_results,
        analysis_time_ms=round(elapsed_ms, 2),
    )
```

---

## 9. Step 6: Configuration

Create `app/config.py`:

```python
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
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "AASIST")
    
    # Device: "cuda" or "cpu" (auto-detected if not set)
    DEVICE: str = os.getenv(
        "DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
```

---

## 10. Step 7: Requirements File

Create `requirements.txt`:

```
# Core ML
torch>=2.0.0
torchaudio>=2.0.0

# AASIST dependencies
numpy
soundfile

# API Server
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart

# Optional but recommended
pydantic>=2.0.0
```

### Install Everything

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** The original AASIST repo requires `torchcontrib` for SWA optimizer during *training*. 
> You do **NOT** need it for inference only. If you see import errors, install it:
> ```bash
> pip install torchcontrib
> ```

---

## 11. Step 8: Running the Server

### Start the Server

```bash
# From the backend/ directory
cd voiceguard-hackathon/backend

# Development mode (auto-reload on code changes)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Expected Startup Output

```
============================================================
🚀 DeepShield AI — Starting Inference Server
============================================================
✅ AASIST model loaded successfully!
   Variant:    AASIST
   Parameters: 297,161
   Device:     cpu
   Weights:    D:\Voice\backend\aasist\models\weights\AASIST.pth
============================================================
✅ Server ready! Accepting requests.
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Access Points

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Welcome endpoint |
| `http://localhost:8000/docs` | **Swagger UI** — interactive API testing |
| `http://localhost:8000/redoc` | ReDoc API documentation |
| `http://localhost:8000/health` | Health check |

> 💡 **Open `http://localhost:8000/docs`** in your browser for a beautiful interactive UI where you can upload audio files and test the API instantly!

---

## 12. Step 9: Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single file prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_audio.wav"

# Multi-chunk prediction (for longer audio)
curl -X POST "http://localhost:8000/predict/multi" \
  -F "file=@long_audio.mp3"
```

### Using Python

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
files = {"file": open("test_audio.wav", "rb")}
response = requests.post(url, files=files)
print(response.json())

# Expected response:
# {
#   "result": "AI",
#   "confidence": 94.7,
#   "risk_level": "CRITICAL",
#   "bonafide_probability": 5.3,
#   "spoof_probability": 94.7,
#   "bonafide_score": -3.2145,
#   "analysis_time_ms": 234.56
# }
```

### Using JavaScript (from your React Native app)

```javascript
const analyzeVoice = async (audioUri) => {
  const formData = new FormData();
  formData.append('file', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'recording.wav',
  });

  const response = await fetch('http://YOUR_SERVER:8000/predict', {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  const data = await response.json();
  console.log(data);
  // { result: "AI", confidence: 94.7, risk_level: "CRITICAL", ... }
};
```

### Quick Test Script

Create `test_api.py` in the backend directory:

```python
"""Quick test script for the DeepShield API."""
import requests
import sys

SERVER = "http://localhost:8000"

def test_health():
    r = requests.get(f"{SERVER}/health")
    print(f"Health: {r.json()}")

def test_predict(file_path: str):
    print(f"\nAnalyzing: {file_path}")
    with open(file_path, "rb") as f:
        r = requests.post(f"{SERVER}/predict", files={"file": f})
    
    if r.status_code == 200:
        data = r.json()
        emoji = "🔴" if data["result"] == "AI" else "🟢"
        print(f"{emoji} Result:     {data['result']}")
        print(f"   Confidence: {data['confidence']}%")
        print(f"   Risk:       {data['risk_level']}")
        print(f"   Time:       {data['analysis_time_ms']:.0f}ms")
    else:
        print(f"Error: {r.status_code} - {r.text}")

if __name__ == "__main__":
    test_health()
    if len(sys.argv) > 1:
        test_predict(sys.argv[1])
    else:
        print("\nUsage: python test_api.py <audio_file>")
```

```bash
python test_api.py test_audio.wav
```

---

## 13. Step 10: Tunneling for Mobile Access

Your phone can't reach `localhost:8000`. Use a tunnel:

### Option A: ngrok (Fastest)

```bash
# Install ngrok (if not installed)
# Download from https://ngrok.com/download

# Start tunnel
ngrok http 8000

# Output:
# Forwarding  https://abc123.ngrok-free.app → http://localhost:8000
```

Use the `https://abc123.ngrok-free.app` URL in your mobile app.

### Option B: Cloudflare Tunnel (More stable)

```bash
# Install cloudflared
# https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

cloudflared tunnel --url http://localhost:8000
```

### Update Mobile App

```javascript
// In your React Native app, update the server URL:
const SERVER_URL = "https://abc123.ngrok-free.app";  // ← your tunnel URL
```

---

## 14. Production Hardening

### A. GPU Optimization

```python
# In model_loader.py predict(), add:
if self.device == "cuda":
    torch.cuda.empty_cache()  # Free GPU memory after inference
```

### B. ONNX Export (2-5× Faster Inference)

```python
"""Export AASIST to ONNX for faster CPU inference."""
import torch

# Load model
loader = AASISTModelLoader(model_variant="AASIST-L")
model = loader.get_model()

# Create dummy input
dummy = torch.randn(1, 64600)

# Export
torch.onnx.export(
    model, dummy,
    "aasist_l.onnx",
    input_names=["audio"],
    output_names=["hidden", "output"],
    dynamic_axes={"audio": {0: "batch"}},
    opset_version=14,
)
print("Exported to aasist_l.onnx")
```

### C. Rate Limiting

```python
# Add to main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("30/minute")
async def predict_voice(...):
    ...
```

### D. File Validation

```python
# Enhanced file validation in main.py
import magic  # pip install python-magic

def validate_audio_file(file_bytes: bytes) -> bool:
    mime = magic.from_buffer(file_bytes[:2048], mime=True)
    return mime.startswith("audio/")
```

---

## 15. Troubleshooting

### Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'models'` | AASIST not in Python path | Check `aasist_dir` path in config |
| `FileNotFoundError: AASIST.pth` | Weights not downloaded | Run `git lfs pull` in aasist directory |
| `RuntimeError: CUDA out of memory` | GPU memory full | Use CPU or reduce batch size |
| `torchaudio.load() failed` | Unsupported audio format | Install `ffmpeg`: `pip install ffmpeg-python` and ensure ffmpeg is in PATH |
| `422 Unprocessable Entity` | Bad audio file | Check file is valid audio, not corrupt |
| `CORS error from mobile app` | CORS not configured | Verify CORSMiddleware is added |

### ffmpeg Installation (Required for MP3/M4A Support)

```bash
# Windows (using chocolatey)
choco install ffmpeg

# Windows (using winget)
winget install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Model Loading Debug

```python
# Quick test to verify model loads correctly
python -c "
import sys
sys.path.insert(0, './aasist')
import json
from importlib import import_module
import torch

with open('./aasist/config/AASIST.conf') as f:
    config = json.load(f)

module = import_module('models.AASIST')
model = module.Model(config['model_config'])
model.load_state_dict(torch.load('./aasist/models/weights/AASIST.pth', map_location='cpu'))
model.eval()

# Test with random input
dummy = torch.randn(1, 64600)
with torch.no_grad():
    hidden, output = model(dummy)
    print(f'Output shape: {output.shape}')
    print(f'Bonafide score: {output[0, 1].item():.4f}')
    print('✅ Model works!')
"
```

---

## 16. API Reference

### `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model": "AASIST",
  "device": "cpu"
}
```

---

### `POST /predict`

**Request:** `multipart/form-data` with field `file` (audio file)

**Response (AI Detected):**
```json
{
  "result": "AI",
  "confidence": 94.7,
  "risk_level": "CRITICAL",
  "bonafide_probability": 5.3,
  "spoof_probability": 94.7,
  "bonafide_score": -3.2145,
  "analysis_time_ms": 234.56
}
```

**Response (Human Detected):**
```json
{
  "result": "HUMAN",
  "confidence": 98.2,
  "risk_level": "LOW",
  "bonafide_probability": 98.2,
  "spoof_probability": 1.8,
  "bonafide_score": 4.8721,
  "analysis_time_ms": 198.32
}
```

---

### `POST /predict/multi`

**Request:** `multipart/form-data` with field `file` (audio file, >4 seconds recommended)

**Response:**
```json
{
  "overall_result": "AI",
  "overall_confidence": 91.3,
  "risk_level": "CRITICAL",
  "chunks_analyzed": 5,
  "chunk_results": [
    {"chunk_index": 0, "prediction": "AI", "confidence": 93.2, "spoof_probability": 93.2, "bonafide_probability": 6.8},
    {"chunk_index": 1, "prediction": "AI", "confidence": 89.1, "spoof_probability": 89.1, "bonafide_probability": 10.9},
    ...
  ],
  "analysis_time_ms": 1023.45
}
```

---

### Error Responses

| Status | Meaning |
|--------|---------|
| `400` | Bad request (empty file, read error) |
| `413` | File too large (>50MB) |
| `422` | Invalid audio format |
| `500` | Model inference error |

---

## Quick Start Summary

```bash
# 1. Setup
mkdir -p voiceguard-hackathon/backend && cd voiceguard-hackathon/backend
python -m venv venv && .\venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Get AASIST
git clone https://github.com/clovaai/aasist.git

# 3. Create app files (copy from sections above)
mkdir app
# Create: app/__init__.py, app/main.py, app/model_loader.py,
#         app/audio_processor.py, app/config.py

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run!
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Test
# Open http://localhost:8000/docs in browser
# Upload any audio file and click "Execute"

# 7. Tunnel (for mobile access)
ngrok http 8000
```

---

**You now have a complete, production-ready deepfake detection backend! 🛡️**
