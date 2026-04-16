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
                f"Weights not found: {weights_path}"
            )
        
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        
        # 5. Set to evaluation mode
        self.model.eval()
        
        print(f"✅ AASIST model loaded successfully! ({self.model_variant} on {self.device})")
        return self.model
    
    def get_model(self) -> nn.Module:
        """Get the loaded model (loads if not already loaded)."""
        if self.model is None:
            self.load()
        return self.model
    
    def predict(self, waveform: torch.Tensor) -> dict:
        """Run inference on a preprocessed waveform tensor."""
        model = self.get_model()
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            _, output = model(waveform)
            
            # 1. Standard Softmax (No scaling needed as logits are already large)
            probs = torch.softmax(output, dim=1)
            
            # 2. SWAP INDICES: Index 0 is AI, Index 1 is HUMAN
            spoof_prob = probs[0, 0].item()    # Index 0
            bonafide_prob = probs[0, 1].item() # Index 1
            
            logit_ai = output[0, 0].item()
            logit_human = output[0, 1].item()
        
        # Determination Logic
        is_ai = spoof_prob > bonafide_prob
        confidence = spoof_prob if is_ai else bonafide_prob
        
        return {
            "prediction": "AI" if is_ai else "HUMAN",
            "confidence": round(confidence * 100, 2),
            "bonafide_score": round(logit_human, 4),
            "bonafide_probability": round(bonafide_prob * 100, 2),
            "spoof_probability": round(spoof_prob * 100, 2),
            "risk_level": "CRITICAL" if is_ai and confidence > 0.95 else ("HIGH" if is_ai else "LOW"),
            "debug": {
                "raw_logit_0 (AI)": round(logit_ai, 4),
                "raw_logit_1 (Human)": round(logit_human, 4),
                "label_order": "0:AI, 1:HUMAN"
            }
        }
    
    def predict_long_audio(self, long_waveform: torch.Tensor, window_size=64600, hop_size=32300) -> dict:
        """
        Analyzes long audio by sliding a 4-second window with 50% overlap.
        window_size: 64600 (approx 4s at 16kHz)
        hop_size: 32300 (step size, creates 50% overlap)
        """
        # 1. Ensure waveform is [1, T]
        if long_waveform.dim() == 1:
            long_waveform = long_waveform.unsqueeze(0)
        
        total_samples = long_waveform.shape[-1]
        chunk_results = []
        
        # 2. Slide the window across the audio
        # Only process if we have at least one full window
        if total_samples < window_size:
            # Fall back to standard predict (which handles padding)
            result = self.predict(long_waveform)
            return {
                "overall_prediction": "AI_DETECTED" if result["prediction"] == "AI" else "HUMAN",
                "risk_level": result["risk_level"],
                "ai_segments_found": 1 if result["prediction"] == "AI" else 0,
                "max_spoof_probability": result["spoof_probability"],
                "segment_details": [] if result["prediction"] != "AI" else [{"start_sec": 0, "prediction": "AI", "spoof_prob": result["spoof_probability"]}]
            }

        for start in range(0, total_samples - window_size + 1, hop_size):
            chunk = long_waveform[:, start : start + window_size]
            
            # Run standard prediction on this 4-second chunk
            result = self.predict(chunk) 
            chunk_results.append({
                "start_sec": round(start / 16000, 2),
                "prediction": result["prediction"],
                "spoof_prob": result["spoof_probability"]
            })

        # 3. Aggregate Results (The "Alarm" Logic)
        # If even ONE chunk is AI, the whole file is high risk.
        ai_chunks = [c for c in chunk_results if c["prediction"] == "AI"]
        max_spoof = max([c["spoof_prob"] for c in chunk_results]) if chunk_results else 0
        
        is_spoof_detected = len(ai_chunks) > 0

        return {
            "overall_prediction": "AI_DETECTED" if is_spoof_detected else "HUMAN",
            "risk_level": "CRITICAL" if max_spoof > 95 else ("HIGH" if is_spoof_detected else "LOW"),
            "ai_segments_found": len(ai_chunks),
            "max_spoof_probability": max_spoof,
            "segment_details": ai_chunks # Shows exactly where the AI was found
        }

    @staticmethod
    def _get_risk_level(spoof_prob: float) -> str:
        if spoof_prob >= 0.9: return "CRITICAL"
        elif spoof_prob >= 0.7: return "HIGH"
        elif spoof_prob >= 0.4: return "MEDIUM"
        else: return "LOW"
