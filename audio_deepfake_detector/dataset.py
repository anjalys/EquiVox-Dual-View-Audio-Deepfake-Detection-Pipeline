import librosa
import pandas as pd
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class DeepfakeAudioDataset(Dataset):
    def __init__(self, metadata_list, audio_dir, target_sr=16000):
        """
        metadata_list: List of dict records passed from pre-split DataFrames.
        """
        self.metadata = metadata_list
        self.audio_dir = audio_dir
        self.target_sr = target_sr

        # Map textual string labels uniformly into binary integers
        # spoof -> 1 (Fake), bona-fide -> 0 (Real)
        # Kept both spellings since In-the-Wild uses "bona-fide" (hyphenated)
        # and ASVspoof 2019 uses "bonafide" (no hyphen) - this lets the same
        # dataset class work against either source without editing this file again.
        self.label_mapping = {"spoof": 1, "bonafide": 0, "bona-fide": 0}
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata[idx]
        file_path = os.path.join(self.audio_dir, row['file'])

        # Load Raw Audio Signal
        waveform, sr = torchaudio.load(file_path)

        # Stereo to Mono downmixing
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Standardize Sample Rate Boundaries -> Resample if sample rate differs
        if sr != self.target_sr:
            # resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            # waveform = resampler(waveform)
            # librosa expects numpy arrays, so convert, resample, convert back
            audio_np = waveform.squeeze(0).numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.target_sr)
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        
        # Standardize clip length - Enforce Fixed Length Constraints (e.g., standardizing to 4-second chunks)
        max_len = self.target_sr * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            pad_len = max_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))
            
        # Construct metadata dictionary for downstream processing
        # Clean labels to standard integers
        raw_label = row["label"].strip().lower() if "label" in row else row["true_label"].strip().lower()

        if raw_label not in self.label_mapping:
            raise ValueError(f"Unrecognized label: {raw_label!r} in {row['file']}")
        label_int = self.label_mapping[raw_label]

        return {
            "waveform": waveform.squeeze(0),
            "label": torch.tensor(label_int, dtype=torch.long),
            "speaker": row["speaker"],
            "file_path": row["file"],
        }