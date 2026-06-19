import torch
import torch.nn as nn

class MultiViewEnsemble(nn.Module):
    def __init__(self, whisper_dim, xlsr_dim):
        super().__init__()
        self.whisper_proj = nn.Sequential(
            nn.Linear(whisper_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.xlsr_proj = nn.Sequential(
            nn.Linear(xlsr_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Stacking Engine / Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Outputs: [Logit_Real, Logit_Fake]
        )
        
    def forward(self, whisper_feats, xlsr_feats):
        w_p = self.whisper_proj(whisper_feats)
        x_p = self.xlsr_proj(xlsr_feats)
        
        fused = torch.cat((w_p, x_p), dim=-1)
        return self.classifier(fused)