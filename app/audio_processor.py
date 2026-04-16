"""
audio_processor.py
Handles all audio preprocessing before model inference.
"""

import av
import numpy as np
import io
import tempfile
import os
from pathlib import Path
from typing import Union

import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600  # ~4.03 seconds at 16kHz
SILENCE_THRESHOLD = 0.001    # Root Mean Square (RMS) energy threshold


class AudioProcessor:
    """Preprocesses audio files for AASIST model input."""
    
    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        target_length: int = TARGET_NUM_SAMPLES,
        silence_threshold: float = SILENCE_THRESHOLD,
    ):
        self.target_sr = target_sr
        self.target_length = target_length
        self.silence_threshold = silence_threshold

    @staticmethod
    def get_rms_energy(waveform: torch.Tensor) -> float:
        """Calculates the RMS energy of a waveform."""
        return torch.sqrt(torch.mean(waveform**2)).item()

    def is_silent(self, waveform: torch.Tensor) -> bool:
        """Checks if a waveform is considered silent based on energy."""
        energy = self.get_rms_energy(waveform)
        return energy < self.silence_threshold
    
    def process_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> torch.Tensor:
        """
        Processes audio bytes using PyAV for robust decoding.
        """
        try:
            waveform, sample_rate = self._load_with_av(audio_bytes)
            return self._preprocess(waveform, sample_rate)
        except Exception as e:
            # Fallback for simple formats
            print(f"PyAV loading failed: {e}. Trying torchaudio fallback.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
                return self._preprocess(waveform, sample_rate)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    def _load_with_av(self, audio_bytes: bytes) -> tuple[torch.Tensor, int]:
        """Loads audio from bytes using PyAV and returns (waveform, sample_rate)."""
        container = av.open(io.BytesIO(audio_bytes))
        stream = container.streams.audio[0]
        
        # Resample/Convert to float32 mono directly using PyAV if possible
        # but easier to just extract raw and then let torch handle it
        samples = []
        sample_rate = stream.rate
        
        for frame in container.decode(audio=0):
            # Convert to float32 and take mean if multi-channel
            array = frame.to_ndarray().astype(np.float32)
            if frame.format.name == 's16' or frame.format.name == 's16p':
                array /= 32768.0
            elif frame.format.name == 's32' or frame.format.name == 's32p':
                array /= 2147483648.0
            
            # handle planar vs interleaved
            if 'p' in frame.format.name: # planar
                if array.ndim > 1:
                    array = np.mean(array, axis=0)
            else: # interleaved
                if array.ndim > 1:
                    array = np.mean(array, axis=1)
                    
            samples.append(array)
        
        waveform = torch.from_numpy(np.concatenate(samples)).unsqueeze(0)
        return waveform, sample_rate
    
    @staticmethod
    def apply_pre_emphasis(waveform: torch.Tensor, coefficient: float = 0.97) -> torch.Tensor:
        """Applies pre-emphasis filter to boost speech clarity and suppress hum."""
        return torch.cat((waveform[:, :1], waveform[:, 1:] - coefficient * waveform[:, :-1]), dim=1)

    @staticmethod
    def apply_high_pass(waveform: torch.Tensor, sample_rate: int, cutoff: float = 80.0) -> torch.Tensor:
        """Removes low-frequency ambient noise (AC, fans, rumbling)."""
        return torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff)

    def _preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr
            )
            waveform = resampler(waveform)
            
        # 1. Forensic Cleaning
        waveform = self.apply_high_pass(waveform, self.target_sr, cutoff=80.0)
        waveform = self.apply_pre_emphasis(waveform)
        
        # 2. Safe Normalization: -3dB Headroom to prevent clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-8:
            waveform = (waveform / max_val) * 0.707
            
        waveform = self._pad_or_trim(waveform)
        return waveform.squeeze(0)
    
    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        current_length = waveform.shape[-1]
        
        if current_length >= self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            # Reverting to repeat-padding which worked earlier
            repeat_times = (self.target_length // current_length) + 1
            waveform = waveform.repeat(1, repeat_times)[:, :self.target_length]
        
        return waveform

    def process_multiple_chunks(self, audio_bytes: bytes, suffix: str = ".wav") -> list[torch.Tensor]:
        """
        Processes audio bytes and splits into multiple chunks using PyAV.
        """
        try:
            waveform, sample_rate = self._load_with_av(audio_bytes)
        except Exception as e:
            print(f"PyAV loading failed for multi-chunk: {e}. Trying torchaudio fallback.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        total_samples = waveform.shape[-1]
        chunks = []
        
        for start in range(0, total_samples, self.target_length):
            chunk = waveform[:, start:start + self.target_length]
            if chunk.shape[-1] < self.target_length // 2:
                continue
            chunk = self._pad_or_trim(chunk)
            chunks.append(chunk.squeeze(0))
        
        if not chunks:
            chunks = [self._pad_or_trim(waveform).squeeze(0)]
        
        return chunks
