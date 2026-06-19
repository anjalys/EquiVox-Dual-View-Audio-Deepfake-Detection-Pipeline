# Deepfake Audio Detection

This repository implements an end-to-end multi-view audio deepfake detection pipeline designed for advanced TTS and voice cloning attacks. It combines semantic and structural feature extractors with an ensemble classifier and includes a demographic bias audit.

## Overview

Modern audio deepfake attacks often use discrete neural codecs and non-autoregressive flow matching, which weakens the effectiveness of legacy spoofing artifacts. Conventional detectors can overfit low-level acoustic cues and exhibit high variance across speakers and demographic groups.

This project addresses those challenges with:

- Dual-view feature extraction: semantic and structural perspectives
- Frozen foundation backbones for stronger generalization
- A stacking ensemble classifier for binary real vs fake predictions
- A demographic audit using Equal Error Rate (EER) and Mean Absolute Deviation (MAD)

## System Architecture

The detector extracts features from two frozen foundation models and fuses them through a learnable classification head.

- **Whisper View (Semantic Front-End)**
  - Uses `openai/whisper-base` encoder hidden states
  - Captures prosody, semantics, and high-level spoken structure

- **XLS-R SLS View (Structural Front-End)**
  - Uses `facebook/wav2vec2-xls-r-300m`
  - Applies Sensitive Layer Selection to mid-tier hidden states
  - Targets codec quantization boundaries and structural signal anomalies

- **Stacking Ensemble Engine**
  - Projects each view into a shared feature space
  - Concatenates multi-view representations
  - Feeds fused features into a dense binary classifier

## Evaluation

The system was evaluated on a balanced set of 60 speaker tracks, evenly split between genuine and synthetic audio.

- Global System EER: `0.4667` (46.67%)
- Vulnerability profile EER: `0.4667`

### Demographic Bias Audit

Equal Error Rates were computed for individual target identities to assess fairness.

| Speaker Profile | Equal Error Rate (EER) |
|-----------------|------------------------|
| Bernie_Sanders  | 0.30                   |
| Barack_Obama    | 0.30                   |
| Donald_Trump    | 0.50                   |

- Demographic Bias Disparity Score (MAD): `0.0889`

> The system performs better on some speaker profiles and degrades to near-random performance on others, creating a measurable demographic disparity.

## Repository Structure

```text
audio_deepfake_detector/
├── config.py
├── dataset.py
├── models/
│   ├── __init__.py
│   ├── whisper_feature.py
│   ├── xlsr_sls.py
│   └── ensemble.py
├── train.py
├── evaluate.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

- `config.py` — hyperparameters, feature dimensions, and layer-selection maps
- `dataset.py` — dataset wrapper with 16 kHz resampling and clip standardization
- `models/` — semantic, structural, and ensemble model components
- `train.py` — multi-view training with frozen feature extractors
- `evaluate.py` — audit engine computing global/group EER and demographic MAD
- `run_pipeline.py` — end-to-end orchestration script

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 run_pipeline.py
```

## Future Improvements

Potential improvements to reduce overall error and demographic disparity:

- **InfoNCE Contrastive Realignment** — separate speaker identity from synthetic artifacts using contrastive training
- **Learnable Layer Selection Weighting** — replace fixed layer selection with trainable attention over XLS-R hidden states
- **Spectro-Temporal Graph Integration** — add a signal-level graph model to complement foundation representations
