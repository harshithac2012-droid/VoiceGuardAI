import torch
import numpy as np

class LongAudioDetector:
    """
    Optimized two-pass detector for processing long audio files (e.g., 22 minutes).
    Uses Incident Grouping to merge overlapping AI segments for UI display.
    """
    def __init__(self, model_instance, target_sr=16000):
        self.model = model_instance
        self.sr = target_sr
        self.window_size = 64600  # ~4.03s

    def analyze(self, waveform: torch.Tensor):
        """
        Analyzes long audio by sliding a 4-second window with forensic precision.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        total_samples = waveform.shape[-1]
        
        # --- PASS 1: COARSE SCAN (Fast 4s jumps) ---
        # We step by the full window size to cover the 22 mins quickly
        pass1_step = self.window_size 
        suspicious_windows = []
        
        for start in range(0, total_samples - self.window_size + 1, pass1_step):
            chunk = waveform[:, start : start + self.window_size]
            result = self.model.predict(chunk)
            
            # If AI probability > 20%, mark for closer inspection
            if result["spoof_probability"] > 20.0:
                suspicious_windows.append(start)

        # --- PASS 2: FINE SCAN (Precision 0.5s jumps) ---
        # We only look at the areas that Pass 1 flagged
        fine_step = 8000 # 0.5 seconds
        ai_segments = []
        
        for zone_start in suspicious_windows:
            # Create a 6-second search buffer around the suspicious point
            search_start = max(0, zone_start - 16000) 
            search_end = min(total_samples, zone_start + self.window_size + 16000)
            
            for start in range(search_start, search_end - self.window_size + 1, fine_step):
                chunk = waveform[:, start : start + self.window_size]
                res = self.model.predict(chunk)
                
                # High confidence threshold for Pass 2 to avoid false positives
                if res["prediction"] == "AI" and res["confidence"] > 85:
                    ai_segments.append({
                        "start_sec": start / self.sr,
                        "end_sec": (start + self.window_size) / self.sr,
                        "confidence": res["confidence"],
                        "timestamp": self._format_timestamp(start / self.sr)
                    })

        # Group overlapping segments into single "Incidents" for the UI
        incidents = self._merge_segments(ai_segments)
        
        return {
            "verdict": "AI_DETECTED" if incidents else "HUMAN",
            "total_incidents": len(incidents),
            "segments": incidents,
            "file_duration_min": round(total_samples / (self.sr * 60), 2)
        }

    def _format_timestamp(self, seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _merge_segments(self, segments):
        if not segments: return []
        # Sort by start time just in case
        segments.sort(key=lambda x: x["start_sec"])
        
        # Merges overlapping detections into one continuous range for the UI
        merged = []
        curr = segments[0]
        
        for next_seg in segments[1:]:
            # If the next segment starts within 1s of the current one ending, merge them
            if next_seg["start_sec"] <= curr["end_sec"] + 1.0: # 1s grace period
                curr["end_sec"] = max(curr["end_sec"], next_seg["end_sec"])
                curr["confidence"] = max(curr["confidence"], next_seg["confidence"])
            else:
                merged.append(curr)
                curr = next_seg
        merged.append(curr)
        
        # Add final formatted timestamps and duration to merged results
        for m in merged:
            m["display_time"] = f"{self._format_timestamp(m['start_sec'])} - {self._format_timestamp(m['end_sec'])}"
            m["duration_sec"] = round(m["end_sec"] - m["start_sec"], 2)
            
        return merged
