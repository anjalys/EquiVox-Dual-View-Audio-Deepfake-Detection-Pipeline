import torch
from torch.utils.data import Dataset
import torchaudio

class DeepfakeAudioDataset(Dataset):
    def __init__(self, metadata, target_sr=16000):
        """
        metadata: List of dicts containing keys: 
                  'file_path', 'label' (0 for real, 1 for fake),
                  'attack_type' (e.g., 'Maya1', 'MeloTTS'), and 'accent'
        """
        self.metadata = metadata
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        meta = self.metadata[idx]
        waveform, sr = torchaudio.load(meta['file_path'])
        
        # Resample if sample rate differs
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        # Standardize clip length (e.g., truncate/pad to 4 seconds)
        max_len = self.target_sr * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            pad_len = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
        return {
            "waveform": waveform.squeeze(0),
            "label": torch.tensor(meta['label'], dtype=torch.long),
            "attack_type": meta.get('attack_type', 'real'),
            "accent": meta.get('accent', 'unknown')
        }