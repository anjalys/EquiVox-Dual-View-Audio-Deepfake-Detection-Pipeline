import numpy as np
from sklearn.metrics import roc_curve

def calculate_eer(labels, scores):
    """Calculates Equal Error Rate (EER) given true labels and prediction scores."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = fpr[idx]
    return eer

def run_forensic_and_fairness_audit(eval_results):
    """
    eval_results: List of dicts with keys: 'label', 'score', 'attack_type', 'accent'
    """
    labels = np.array([r['label'] for r in eval_results])
    scores = np.array([r['score'] for r in eval_results])
    
    # 1. Total Global Performance Evaluation
    global_eer = calculate_eer(labels, scores)
    print(f"=== GLOBAL SYSTEM EER: {global_eer:.4f} ===\n")
    
    # 2. Cross-Model Generalization Evaluation (Robustness Profiling)
    print("--- Cross-Model Vulnerability Profile ---")
    attacks = set([r['attack_type'] for r in eval_results if r['attack_type'] != 'real'])
    for attack in attacks:
        attack_labels = [r['label'] for r in eval_results if r['attack_type'] in [attack, 'real']]
        attack_scores = [r['score'] for r in eval_results if r['attack_type'] in [attack, 'real']]
        eer_vs_attack = calculate_eer(attack_labels, attack_scores)
        print(f"System EER vs Architecture [{attack}]: {eer_vs_attack:.4f}")
        
    # 3. Demographic Fairness Audit (The FairVoice Disparity Metric)
    print("\n--- FairVoice Demographic Equity Audit ---")
    accents = set([r['accent'] for r in eval_results])
    accent_eers = {}
    
    for accent in accents:
        accent_labels = [r['label'] for r in eval_results if r['accent'] == accent]
        accent_scores = [r['score'] for r in eval_results if r['accent'] == accent]
        
        # Ensure sufficient cohort samples to avoid statistical noise
        if len(set(accent_labels)) > 1:
            accent_eers[accent] = calculate_eer(accent_labels, accent_scores)
            print(f"EER within Demographic Group [{accent}]: {accent_eers[accent]:.4f}")
            
    # Compute Bias Disparity via Mean Absolute Deviation (MAD)
    if accent_eers:
        eer_values = list(accent_eers.values())
        mean_eer = np.mean(eer_values)
        mad_score = np.mean(np.absolute(np.array(eer_values) - mean_eer))
        print(f"\nDemographic Bias Disparity Score (MAD): {mad_score:.4f}")
        print("Note: An optimized, equitable detector targets a MAD score approaching 0.00.")