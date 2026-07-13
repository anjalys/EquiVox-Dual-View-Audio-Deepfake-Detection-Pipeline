import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import Config
from dataset import DeepfakeAudioDataset
from models.whisper_feature import WhisperSemanticExtractor
from models.xlsr_sls import XlsrSlsExtractor
from models.ensemble import MultiViewEnsemble

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# CONTRASTIVE ALIGNMENT PENALTY FUNCTION
# ==========================================
def contrastive_alignment_loss(w_feats, x_feats, labels, margin=0.5):
    """
    Computes a Cross-Perspective Similarity Penalty.
    Forks feature representations based on authenticity:
      - Real (0): Maximizes Cosine Similarity between views.
      - Fake (1): Minimizes Cosine Similarity up to a safety margin.
    """
    # Normalize embeddings to unit sphere for robust cosine evaluation
    w_norm = F.normalize(w_feats, p=2, dim=-1)
    x_norm = F.normalize(x_feats, p=2, dim=-1)
    
    # Calculate batch-wise cosine similarity
    cosine_sim = torch.sum(w_norm * x_norm, dim=-1)
    
    # Loss for Real Tracks: Pull views together (Minimize 1 - Sim)
    loss_real = 1.0 - cosine_sim
    
    # Loss for Fake Tracks: Push views apart up to a specified distance margin
    loss_fake = torch.clamp(cosine_sim - margin, min=0.0)
    
    # Blend loss arrays using target binary mask labels
    total_alignment_loss = torch.mean((1 - labels) * loss_real + labels * loss_fake)
    return total_alignment_loss

# ==========================================
# GLOBAL MAIN SYSTEM TRAINING PIPELINE
# ==========================================
def train_system(train_dataset, epochs=Config.EPOCHS, alignment_lambda=Config.ALIGNMENT_LAMBDA):
    """
    Trains the dual-view ensemble system with a multi-objective loss:
        - Classification Loss: Binary Cross Entropy for Real vs. Fake
        - Alignment Loss: Contrastive Penalty to align semantic and structural views
    The system is trained end-to-end, but the feature extractors are frozen to
    focus optimization on the ensemble head and projection spaces.
    The ensemble head learns to combine the two perspectives 
    """
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize separate views and meta-classifier
    whisper_view = WhisperSemanticExtractor().to(Config.DEVICE)
    xlsr_view = XlsrSlsExtractor(Config.SLS_LAYERS).to(Config.DEVICE)

    # Projection heads for alignment loss computation
    whisper_proj = ProjectionHead(768, 256).to(Config.DEVICE)
    xlsr_proj = ProjectionHead(1024, 256).to(Config.DEVICE)

    # Freeze feature extractors to focus training on ensemble head and projection spaces
    ensemble_detector = MultiViewEnsemble(Config.WHISPER_DIM, Config.XLSR_DIM).to(Config.DEVICE)
    
    # Only the ensemble head and projection heads are trainable; feature extractors remain frozen
    criterion = nn.CrossEntropyLoss()

    # Optimizer encompasses ensemble head and projection heads for joint optimization
    optimizer = torch.optim.AdamW(
        list(ensemble_detector.parameters()) +
        list(whisper_proj.parameters()) +
        list(xlsr_proj.parameters()),
        lr=Config.LEARNING_RATE)

    for epoch in range(epochs):
        ensemble_detector.train()
        total_epoch_loss = 0.0
        total_cls_loss = 0.0
        total_align_loss = 0.0
        
        for batch in train_loader:
            waveforms = batch['waveform'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            # Step 1: Forward pass frozen extraction backbones
            with torch.no_grad():
                # Whisper expects log-mel inputs usually; this abstracts tracking hidden states
                w_feats = whisper_view(waveforms) # B[, 768]
                x_feats = xlsr_view(waveforms) # B[, 1024]

            # Step 2: Feed fused structural and semantic abstractions into ensemble head
            outputs = ensemble_detector(w_feats, x_feats)

            # Alignment branch
            w_proj = whisper_proj(w_feats)          # [B,256]
            x_proj = xlsr_proj(x_feats)             # [B,256]

            # Compute Multi-Objective Loss Boundaries
            cls_loss = criterion(outputs, labels)
            align_loss = contrastive_alignment_loss(w_proj, x_proj, labels)
            
            # Total Loss = Binary Cross Entropy + Alpha * Contrastive Alignment Penalty
            loss = cls_loss + (alignment_lambda * align_loss)

            # Backpropagation & Weight Adjustments
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_align_loss += align_loss.item()
            
        # Log training mechanics per epoch boundary
        avg_loss = total_epoch_loss / len(train_loader)
        avg_cls = total_cls_loss / len(train_loader)
        avg_align = total_align_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Loss: {avg_loss:.4f} "
              f"| Cls Loss: {avg_cls:.4f} "
              f"| Align Penalty: {avg_align:.4f}")
        
        # Save optimized parameters to file
        torch.save(ensemble_detector.state_dict(), "best_ensemble_model.pt")
        print("✓ Model parameters successfully cached to 'best_ensemble_model.pt'")
            
    return whisper_view, xlsr_view, ensemble_detector