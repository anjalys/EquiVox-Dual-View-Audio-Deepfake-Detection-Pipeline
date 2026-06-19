import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class XlsrSlsExtractor(nn.Module):
    def __init__(self, selected_layers=[4, 12, 18]):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.selected_layers = selected_layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Extract and pool specified layers
        layer_pools = []
        for idx in self.selected_layers:
            layer_state = hidden_states[idx]
            pooled = torch.mean(layer_state, dim=1) # Pool time dimension
            layer_pools.append(pooled)
            
        # Combine structural insights from all target layers
        fused_structural = torch.mean(torch.stack(layer_pools, dim=0), dim=0)
        return fused_structural