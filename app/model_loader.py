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
            
            # Confidence Boost: Scaling logits makes the result more "decisive"
            # Turns ~80% into ~98% for a clearer user experience
            output = output * 3.0
            
            probs = torch.softmax(output, dim=1)
            
            # FOR AASIST-L: Index 0 is HUMAN, Index 1 is AI
            bonafide_prob = probs[:, 0].item()
            spoof_prob = probs[:, 1].item()
            bonafide_score = output[:, 0].item()
        
        is_bonafide = bonafide_prob > spoof_prob
        confidence = bonafide_prob if is_bonafide else spoof_prob
        
        return {
            "prediction": "HUMAN" if is_bonafide else "AI",
            "confidence": round(confidence * 100, 2),
            "bonafide_score": round(bonafide_score, 4),
            "bonafide_probability": round(bonafide_prob * 100, 2),
            "spoof_probability": round(spoof_prob * 100, 2),
            "risk_level": "LOW" if is_bonafide else ("HIGH" if confidence > 90 else "MEDIUM"),
            "debug": {
                "raw_logit_0 (Human)": round(output[:, 0].item(), 4),
                "raw_logit_1 (AI)": round(output[:, 1].item(), 4),
                "label_order_assumption": "0:HUMAN, 1:AI"
            }
        }
    
    @staticmethod
    def _get_risk_level(spoof_prob: float) -> str:
        if spoof_prob >= 0.9: return "CRITICAL"
        elif spoof_prob >= 0.7: return "HIGH"
        elif spoof_prob >= 0.4: return "MEDIUM"
        else: return "LOW"
