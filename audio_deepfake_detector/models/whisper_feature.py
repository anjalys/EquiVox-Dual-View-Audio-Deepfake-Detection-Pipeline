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

        # Constants for frame calculations
        # Whisper encoder downsamples time by this factor (one stride-2 conv)
        self.encoder_downsample_factor = 2
        # Frames per second in the input mel spectrogram (10ms hop -> 100/sec)
        self.mel_frames_per_second = 100
            
    def forward(self, x):
        # Convert raw audio tensor to numpy arrays for the HuggingFace feature extractor
        # Assumes x is a batch of waveforms or a single audio track at 16000Hz
        audio_inputs = [wave.cpu().numpy() for wave in x] 

        # Track real (non-padded) duration of each clip in seconds, BEFORE Whisper's own padding
        # If you already standardize all clips to exactly 4s in dataset.py, this is constant,
        # but computing it from x keeps things correct if that ever changes.
        real_seconds = x.shape[-1] / 16000.0  # sampling_rate = 16000

        # Process audio to output precisely 3000 mel length frames with truncation/padding
        features = self.feature_extractor(
            audio_inputs, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features # shape [B, n_mels, 3000]
        
        # Move processed log-mel spectrogram back to the correct device (CPU/MPS/CUDA)
        features = features.to(x.device)
        
        # Pass the correctly shaped mel features to the encoder
        outputs = self.backbone.encoder(features)
        hidden = outputs.last_hidden_state  # shape [B, 1500, 768]

        # How many encoder frames correspond to REAL audio (not Whisper's internal padding)
        real_mel_frames = int(real_seconds * self.mel_frames_per_second)          # e.g. 400
        real_encoder_frames = real_mel_frames // self.encoder_downsample_factor    # e.g. 200
        real_encoder_frames = min(real_encoder_frames, hidden.shape[1])            # safety clamp

        # Mean-pool ONLY over the real-audio frames, ignore the silence-padded tail
        semantic_vector = hidden[:, :real_encoder_frames, :].mean(dim=1)
        return semantic_vector