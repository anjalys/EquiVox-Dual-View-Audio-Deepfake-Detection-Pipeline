import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from models.whisper_feature import WhisperSemanticExtractor
from models.xlsr_sls import XlsrSlsExtractor
from models.ensemble import MultiViewEnsemble

def train_system(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize separate views and meta-classifier
    whisper_view = WhisperSemanticExtractor().to(Config.DEVICE)
    xlsr_view = XlsrSlsExtractor(Config.SLS_LAYERS).to(Config.DEVICE)
    ensemble_detector = MultiViewEnsemble(Config.WHISPER_DIM, Config.XLSR_DIM).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ensemble_detector.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.EPOCHS):
        ensemble_detector.train()
        total_loss = 0.0
        
        for batch in train_loader:
            waveforms = batch['waveform'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            # Step 1: Forward pass frozen extraction backbones
            with torch.no_grad():
                # Whisper expects log-mel inputs usually; this abstracts tracking hidden states
                w_feats = whisper_view(waveforms)
                x_feats = xlsr_view(waveforms)
                
            # Step 2: Feed fused structural and semantic abstractions into ensemble head
            outputs = ensemble_detector(w_feats, x_feats)
            loss = criterion(outputs, labels)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")
        
    return whisper_view, xlsr_view, ensemble_detector