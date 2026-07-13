import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch import nn
from dataset import DeepfakeAudioDataset
from config import Config

from train import train_system
from evaluate import run_forensic_and_fairness_audit

# =========================
# CONFIG
# =========================
TARGET_SR = 16000
N_MELS = 64


# =====================================================================
# PROTOCOL PARSER (ASVspoof 2019 LA format)
# =====================================================================
def parse_asvspoof_protocol(protocol_path):
    """
    Parses ASVspoof 2019 LA protocol .txt files into a DataFrame matching
    the columns DeepfakeAudioDataset expects: file, speaker, label, attack_type.

    Each protocol line looks like:
        LA_0079 LA_T_1138215 - - bonafide
        LA_0079 LA_T_1271820 A01 - spoof
    Columns: speaker_id, filename, attack_type, unused, label
    """
    rows = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines defensively
            speaker, filename, attack_type, _, label = parts
            rows.append({
                "file": f"{filename}.flac",
                "speaker": speaker,
                "label": label,          # "bonafide" or "spoof"
                "attack_type": attack_type
            })
    return pd.DataFrame(rows)

# =====================================================================
# INFERENCE ENGINE
# =====================================================================
class InferenceEngine:
    """
    Inference model executing forward-pass biometric validation.
    Bypasses vanilla log-mel spectrogram flattening by feeding raw tensors
    simultaneously through frozen semantic and structural transformer branches.
    """
    def __init__(self, whisper_view, xlsr_view, ensemble_detector, target_sr=16000):
        self.whisper_view = whisper_view
        self.xlsr_view = xlsr_view
        self.ensemble_detector = ensemble_detector
        self.target_sr = target_sr
        
        # Ensure all sub-modules are situated in evaluation topology
        self.whisper_view.eval()
        self.xlsr_view.eval()
        self.ensemble_detector.eval()

    def load_and_standardize(self, file_path):
        """Loads raw audio tracks, performs mono downmixing, and applies 16kHz sample rate targets."""
        waveform, sr = torchaudio.load(file_path)

        # Convert stereo matrix configurations to mono arrays
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Enforce exact sampling bounds matching your speech transformers
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Standardize structural clip dimensions to exactly 4 seconds
        max_len = self.target_sr * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            pad_len = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Resolve device locally to avoid dependency on external Config
        return waveform.to(Config.DEVICE)

    def predict_spoof_vector(self, waveform):
        """Extracts dual-perspective hidden states and runs predictions through the trained ensemble."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        with torch.no_grad():
            # Step 1: Forward pass frozen extraction backbones
            w_feats = self.whisper_view(waveform)
            x_feats = self.xlsr_view(waveform)
            
            # Step 2: Extract prediction probabilities via the trained Stacking Ensemble Head
            logits = self.ensemble_detector(w_feats, x_feats)
            probabilities = torch.softmax(logits, dim=-1)
            
            prediction_class = torch.argmax(probabilities, dim=-1).item()
            probabilities_numpy = probabilities.cpu().numpy()
            
        return prediction_class, probabilities_numpy
    
# =========================
# EXECUTION FUNCTION
# =========================
def run_pipeline_inference(df, audio_dir, inference_engine):
    """
    Iterates over the entire manifest dataframe to compile real-world evaluation arrays.
    Saves outputs alongside soft probability prediction confidence levels.
    """
    results = []
    print("\n[Phase 3] Deploying Advanced Multi-View Transformer Inference over Dataset...")
    for i, row in df.iterrows():
        file_path = os.path.join(audio_dir, row["file"])
        # print(f" -> Running Biometric Profiling: {row['file']} [{row['speaker']}]")

        try:
            # Run processing and inference
            waveform = inference_engine.load_and_standardize(file_path)
            pred, probs = inference_engine.predict_spoof_vector(waveform)

            results.append({
                "file": row["file"],
                "speaker": row["speaker"],
                "true_label": row["label"],
                "pred": "spoof" if pred == 1 else "bona-fide",
                "confidence_spoof": float(probs[0][1])
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(results)


# =====================================================================
# SYSTEM PIPELINE ORCHESTRATION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    # ---- ASVspoof 2019 LA paths ----
    LA_ROOT = "/data/data"
    TRAIN_PROTOCOL = f"{LA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    DEV_PROTOCOL   = f"{LA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    EVAL_PROTOCOL  = f"{LA_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

    TRAIN_AUDIO_DIR = f"{LA_ROOT}/ASVspoof2019_LA_train/flac"
    DEV_AUDIO_DIR   = f"{LA_ROOT}/ASVspoof2019_LA_dev/flac"
    EVAL_AUDIO_DIR  = f"{LA_ROOT}/ASVspoof2019_LA_eval/flac"

    # 1: Parse the ASVspoof protocol files into structured DataFrames
    train_df = parse_asvspoof_protocol(TRAIN_PROTOCOL).sample(n=5000, random_state=42)  # Subsample for quick testing
    dev_df   = parse_asvspoof_protocol(DEV_PROTOCOL).sample(n=1000, random_state=42)      # Subsample for quick testing
    print(f"Dataset located. Train: {len(train_df)} files | Dev: {len(dev_df)} files")

    # 2. ASVspoof already provides a proper non-overlapping train/dev split by
    #    design (different attack types in each), so no manual .sample() split
    #    is needed here -- unlike the In-the-Wild version above.
    train_meta = train_df.to_dict(orient="records")
    val_meta = dev_df.to_dict(orient="records")

    # 3. Wrap into PyTorch dataset layers
    train_dataset = DeepfakeAudioDataset(train_meta, audio_dir=TRAIN_AUDIO_DIR, target_sr=TARGET_SR)
    val_dataset = DeepfakeAudioDataset(val_meta, audio_dir=DEV_AUDIO_DIR, target_sr=TARGET_SR)

    # 4. Train
    print("\n[Phase 1] Launching Multi-View Backbone Optimizer...")
    whisper_view, xlsr_view, ensemble_detector = train_system(
        train_dataset,
        epochs=Config.EPOCHS,
        alignment_lambda=Config.ALIGNMENT_LAMBDA
    )

    # 5. Fairness/forensic audit on dev set
    print("\n[Phase 2] Executing Forensic Generalization & FairVoice Equity Audit...")
    run_forensic_and_fairness_audit(
        val_dataset=val_dataset,
        whisper_view=whisper_view,
        xlsr_view=xlsr_view,
        ensemble_detector=ensemble_detector
    )

    # 6. Full inference pass + CSV export, using the eval protocol (unseen attacks)
    inference_engine = InferenceEngine(whisper_view, xlsr_view, ensemble_detector)
    eval_df = parse_asvspoof_protocol(EVAL_PROTOCOL).sample(n=200, random_state=42)  # Subsample for quick testing
    results_df = run_pipeline_inference(eval_df, EVAL_AUDIO_DIR, inference_engine)

    print("\nSample Results:")
    print(results_df.head())

    results_df.to_csv("pipeline_results.csv", index=False)
    print("\nSaved to pipeline_results.csv")