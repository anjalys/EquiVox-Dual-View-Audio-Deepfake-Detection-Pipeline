import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel

class WhisperSemanticExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Freeze backbone to act as an uncompromised feature extractor
        self.backbone = WhisperModel.from_pretrained("openai/whisper-small")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Convert raw audio tensor to numpy arrays for the HuggingFace feature extractor
        # Assumes x is a batch of waveforms or a single audio track at 16000Hz
        audio_inputs = [wave.cpu().numpy() for wave in x] 
        
        # Process audio to output precisely 3000 mel length frames with truncation/padding
        features = self.feature_extractor(
            audio_inputs, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        # Move processed log-mel spectrogram back to the correct device (CPU/MPS/CUDA)
        features = features.to(x.device)
        
        # Pass the correctly shaped mel features to the encoder
        outputs = self.backbone.encoder(features)
        return outputs.last_hidden_state.mean(dim=1) # Mean pool over time dimension to get fixed-size semantic vector
        # # Whisper expects log-mel spectrogram input or raw waveforms via processor.
        # # For simplicity, we assume pre-processed features or use its encoder rawly
        # outputs = self.backbone.encoder(x)
        # # Mean pool over sequence length to get a fixed-size representation
        # semantic_vector = torch.mean(outputs.last_hidden_state, dim=1)
        # return semantic_vector