from config import Config
import numpy as np
import torch
import torchaudio
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

device = Config.DEVICE

def calculate_eer(labels, scores):
    """Calculates Equal Error Rate (EER) given true labels and prediction scores.
    The EER represents the threshold point where False Accept Rate equals False Reject Rate.
    """
    labels = np.array(labels)
    scores = np.array(scores)

    # Handle edge cases where a sub-cohort contains only one class type
    # if len(np.unique(labels)) < 2:
    #     return 0.0000
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[idx]
    return eer

def run_forensic_and_fairness_audit(val_dataset, whisper_view, xlsr_view, ensemble_detector):
    """
    Performs a comprehensive forensic and fairness audit on the evaluation results.
    This function computes the global Equal Error Rate (EER), evaluates cross-model generalization
    performance against various attack types, and conducts a demographic fairness audit based on accent groups.
    Performs live multi-view inference over the validation partition.
    Conducts forensic vulnerability mapping and identity-fairness audits.
    """
    # Force batch size to 1 to cleanly isolate specific individual speaker samples
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Switch entire model topology to evaluation mode
    ensemble_detector.eval()
    whisper_view.eval()
    xlsr_view.eval()

    eval_results = []
    print("\n[Evaluation] Generating structural multi-view score space matrices...")

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass through each view and projection
            waveforms = batch['waveform'].to(device)
            label = batch['label'].item()
            speaker = batch['speaker'][0]

            # Robust extraction handling for text manifest variables without using .item()
            attack_type = batch['attack_type'][0] if 'attack_type' in batch else 'synthetic_clone'
            accent = batch['accent'][0] if 'accent' in batch else 'unknown'

            # Forward pass raw audio through our multi-view pipelines
            w_feats = whisper_view(waveforms)
            x_feats = xlsr_view(waveforms)

            # Compute Stacking Ensemble Classifier Output probabilities
            logits = ensemble_detector(w_feats, x_feats)
            probabilities = torch.softmax(logits, dim=-1)

            # Capture the exact prediction confidence level for Class 1 ("spoof")
            spoof_score = probabilities[0][1].item()

            # Store results
            eval_results.append({
                'label': label,
                'score': spoof_score,
                'attack_type': attack_type,
                'speaker': speaker,
                'accent': accent,
            })

    # Process extracted validation space arrays
    global_labels = [r['label'] for r in eval_results]
    global_scores = [r['score'] for r in eval_results]

    # =====================================================================
    # 1. GLOBAL PERFOMANCE INDEX METRICS
    # =====================================================================
    global_eer = calculate_eer(global_labels, global_scores)
    print(f"=== GLOBAL SYSTEM EER: {global_eer:.4f} ===\n")
    
    # 2. Cross-Model Generalization Evaluation (Robustness Profiling)
    print("--- Cross-Model Vulnerability Profile ---")
    attacks = set([r['attack_type'] for r in eval_results if r['label'] == 1])
    
    for attack in attacks:
        # Filter cohorts strictly comparing this specific generator threat vs genuine baseline
        attack_labels = [r['label'] for r in eval_results if r['attack_type'] == attack or r['label'] == 0]
        attack_scores = [r['score'] for r in eval_results if r['attack_type'] == attack or r['label'] == 0]
        
        eer_vs_attack = calculate_eer(attack_labels, attack_scores)
        print(f"System EER vs Generative Architecture [{attack}]: {eer_vs_attack:.4f}")
        
    # 3. Demographic Fairness Audit (The FairVoice Disparity Metric)
    print("\n--- FairVoice Demographic Equity Audit ---")
    speakers = set([r['speaker'] for r in eval_results])
    speaker_eers = {}
    
    for speaker in speakers:
        speaker_labels = [r['label'] for r in eval_results if r['speaker'] == speaker]
        speaker_scores = [r['score'] for r in eval_results if r['speaker'] == speaker]
        
        # Extract EER profiles for vocal groups containing both real and synthetic tracks
        if len(np.unique(speaker_labels)) > 1:
            eer_val = calculate_eer(speaker_labels, speaker_scores)
            speaker_eers[speaker] = eer_val
            print(f"Equal Error Rate (EER) within Speaker Cohort [{speaker}]: {eer_val:.4f}")
        else:
            print(f"Skipping Speaker Identity [{speaker}]: Cohort lacks balanced bona-fide/spoof distributions.")
            
    # Compute Bias Disparity via Mean Absolute Deviation (MAD)
    if speaker_eers:
        eer_values = list(speaker_eers.values())
        mean_eer = np.mean(eer_values)
        mad_score = np.mean(np.absolute(np.array(eer_values) - mean_eer))

        print(f"\nDemographic Bias Disparity Score (MAD): {mad_score:.4f}")
        print("Note: An optimized, equitable detector targets a MAD score approaching 0.00.")