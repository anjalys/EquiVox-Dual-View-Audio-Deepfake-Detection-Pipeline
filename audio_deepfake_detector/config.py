import os

import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # Breaks instead of lagging silently
# Configuration parameters, thresholds, and paths for the audio deepfake detection system
class Config:
    SAMPLE_RATE = 16000
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    ALIGNMENT_LAMBDA = 0.15  # Hyperparameter to control contrastive alignment penalty

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature Dimensions
    WHISPER_DIM = 768  # whisper-small hidden state dimension
    XLSR_DIM = 1024    # wav2vec2-xls-r-300m hidden state dimension
    
    # Selected layers for XLS-R Sensitive Layer Selection (SLS)
    # Autoregressive artifacts generally concentrate in lower/mid layers
    SLS_LAYERS = [4, 12, 18]