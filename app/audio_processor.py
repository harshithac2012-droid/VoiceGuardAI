"""
audio_processor.py
Handles all audio preprocessing before model inference with fixed debugging.
"""

import av
import numpy as np
import io
import tempfile
import os
import torch
import torchaudio
import torch.nn.functional as F

TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600  # Exactly 4.0375 seconds
SILENCE_THRESHOLD = 0.001 

class AudioProcessor:
    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        target_length: int = TARGET_NUM_SAMPLES,
        silence_threshold: float = SILENCE_THRESHOLD,
    ):
        self.target_sr = target_sr
        self.target_length = target_length
        self.silence_threshold = silence_threshold

    def _preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # 1. Standardize Mono/Rate
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sr)(waveform)
        
        # 2. REMOVE NOISE (High Pass Filter at 80Hz)
        waveform = torchaudio.functional.highpass_biquad(waveform, self.target_sr, 80.0)
        
        # 3. APPLY PRE-EMPHASIS (Critical for AASIST)
        # This 'sharpens' the human voice so it stands out from the noise
        waveform = waveform - 0.97 * torch.cat([waveform[:, :1], waveform[:, :-1]], dim=1)
        
        # 4. ZERO PAD (No loops!)
        waveform = self._pad_or_trim(waveform)
        
        # 5. NORMALIZE
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
            
        return waveform.squeeze(0)
    
    @staticmethod
    def get_rms_energy(waveform: torch.Tensor) -> float:
        """Calculates the RMS energy of a waveform."""
        return torch.sqrt(torch.mean(waveform**2)).item()

    def is_silent(self, waveform: torch.Tensor) -> bool:
        """Checks if a waveform is considered silent based on energy."""
        energy = self.get_rms_energy(waveform)
        return energy < self.silence_threshold

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        current_length = waveform.shape[-1]
        
        if current_length >= self.target_length:
            return waveform[:, :self.target_length]
        else:
            # DO NOT REPEAT. Use Zero Padding.
            diff = self.target_length - current_length
            return torch.nn.functional.pad(waveform, (0, diff))

    # --- KEEPING YOUR EXISTING LOADERS ---
    def process_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> torch.Tensor:
        try:
            waveform, sample_rate = self._load_with_av(audio_bytes)
            return self._preprocess(waveform, sample_rate)
        except Exception as e:
            print(f"PyAV loading failed: {e}. Trying torchaudio fallback.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
                return self._preprocess(waveform, sample_rate)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    def process_multiple_chunks(self, audio_bytes: bytes, suffix: str = ".wav") -> dict:
        """
        Processes audio bytes and splits into multiple chunks.
        """
        waveform, sample_rate = self._load_with_av(audio_bytes)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
            
        # Optional: Normalize
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        
        total_samples = waveform.shape[-1]
        chunks = []
        
        for start in range(0, total_samples, self.target_length):
            chunk = waveform[:, start:start + self.target_length]
            if chunk.shape[-1] < self.target_length // 2:
                continue
            chunk = self._pad_or_trim(chunk)
            chunks.append(chunk)
            
        if not chunks:
            chunks = [self._pad_or_trim(waveform)]
            
        return {"chunks": chunks, "sample_rate": self.target_sr}

    def _load_with_av(self, audio_bytes: bytes) -> tuple[torch.Tensor, int]:
        container = av.open(io.BytesIO(audio_bytes))
        stream = container.streams.audio[0]
        samples = []
        sample_rate = stream.rate
        
        for frame in container.decode(audio=0):
            array = frame.to_ndarray().astype(np.float32)
            if 's16' in frame.format.name: array /= 32768.0
            elif 's32' in frame.format.name: array /= 2147483648.0
            
            if 'p' in frame.format.name: # planar
                if array.ndim > 1: array = np.mean(array, axis=0)
            else: # interleaved
                if array.ndim > 1: array = np.mean(array, axis=1)
            samples.append(array)
        
        waveform = torch.from_numpy(np.concatenate(samples)).unsqueeze(0)
        return waveform, sample_rate