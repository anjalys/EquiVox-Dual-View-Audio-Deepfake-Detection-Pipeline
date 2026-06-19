### Deepfake Audio Detection

Modern audio deepfake vulnerabilities have evolved past the artifacts left by traditional neural vocoders. Present-day threats leverage discrete neural codecs and non-autoregressive flow-matching algorithms, blurring the perceptual boundaries between authentic and synthetic human voices. Concurrently, biometric defense networks often exhibit high error variance across different demographic groups and individual speaker identities, frequently misinterpreting regional or stylistic vocal variations as structural cloning anomalies. 

An end-to-end multi-view (ensemble) audio deepfake detection system optimized to identify advanced text-to-speech (TTS) architectures and voice cloning attacks. This project implements a dual-view feature extractor (Linguistic/Semantic hidden states mixed with Structural/Quantization deep states) and conducts a data-driven demographic safety audit inspired by UC Berkeley’s FairVoice framework.

Traditional spoofing classifiers fail against modern speech foundation models because they over-index on low-level acoustic artifacts from legacy vocoders. Advanced text-to-speech architectures leverage discrete neural codecs and non-autoregressive flow-matching algorithms, allowing cloned voices to bridge the perceptual boundary of real speech.

Furthermore, modern detectors exhibit high algorithmic variance across individual vocal identities—creating asymmetric false-alarm patterns on public figures. This repository implements an abstraction-fused ensembling approach to remedy the architectural generalization gap and quantifies systemic demographic disparities using the Mean Absolute Deviation (MAD) of the Equal Error Rate (EER).

#### System Architecture
The detector processes input waveforms through parallel, frozen foundation networks to construct a multi-perspective feature space before classification.

Semantic Front-End (Whisper View): I used openai/whisper-base encoder hidden states to capture contextual prosody, macro-linguistic formatting and high-level conversational semantic alignments.

Structural Layer View (XLS-R SLS View): Chose facebook/wav2vec2-xls-r-300m with Sensitive Layer Selection (SLS) focused on specific mid-tier representation pools to target minute neural-codec quantization boundaries and phase-discontinuity anomalies.

Stacking Ensemble Engine: Combines individual views via linear feature projection networks and channels the unified multi-view representations into a dense classification head to emit binary predictions (Real vs. Fake).

#### Evaluation

The system was evaluated against an audio slice comprising 60 physical speaker tracks balanced equally between genuine vocals and synthetic clones.

1. Global Performance Matrix
    - Global System EER: 0.4667 (46.67%)
    - Vulnerability Target Profile: 0.4667 Equal Error Rate against the synthetic_clone generation paradigm.

2. FairVoice Demographic Bias Audit
To quantify algorithmic equity across specific speaker profiles, separate Equal Error Rates were computed for individual target identities to assess accuracy distribution variance.

Demographic Cohort Group	Equal Error Rate (EER)
Bernie_Sanders	0.3
Barack_Obama	0.3
Donald_Trump	0.5

Demographic Bias Disparity Score (MAD): 0.0889

*Algorithmic Audit Analysis*: While the system achieves an improved EER of 30.0% on the Obama and Sanders vocal profiles, performance deteriorates to a random-guess threshold of 50.0% on the Trump profile. This variance generates a Mean Absolute Deviation (MAD) score of 0.0889. This indicates a clear demographic footprint bias, showing that the underlying feature extractors over-index on certain idiosyncratic speech patterns, mistaking voice cloning anomalies for natural vocal variations.

#### Repository Blueprint

audio-deepfake-detector/
│
├── config.py                 # Hyperparameters, feature dimensions, and layer selection maps
├── dataset.py                # Dataset wrapper with dynamic 16kHz resampling and clip standardization
│
├── models/                   # Modular neural network architectures
│   ├── __init__.py           # Makes models a package directory
│   ├── whisper_feature.py    # Semantic View (Whisper Hidden States)
│   ├── xlsr_sls.py           # Structural/Quantization View (XLS-R + Sensitive Layer Selection)
│   └── ensemble.py           # Multi-View Stacking Ensemble Classifier Head
│
├── train.py                  # Multi-view training module with frozen feature extraction backbones
├── evaluate.py               # Forensic audit engine computing global/group EER and demographic MAD
├── run_pipeline.py           # Orchestration execution script for end-to-end data processing
├── requirements.txt          # Production environment dependencies
└── README.md                 # Project technical documentation

#### Virtual Environment & Dependency Installation

**Create and source environment**
python3 -m venv venv
source venv/bin/activate

**Install dependencies**
pip install -r requirements.txt

**Execute the full optimization loop, multi-view inference pass, and biometric safety audit:**
python3 run_pipeline.py

**Strategic Remediation Optimization Roadmap**
To reduce global error rates and drive the demographic MAD disparity score closer to 0.0000, three main optimization paths will be introduced:

- InfoNCE Contrastive Realignment: Applying a demographic-supervised contrastive loss to explicitly separate speaker identity variations from synthetic cloning traces.
- Learnable Layer Selection Weighting: Shifting from fixed layer selection to trainable attention weights across all XLS-R hidden states to automatically capture subtle quantization artifacts
- Spectro-Temporal Graph Integration: Incorporating an explicit physical signal monitoring layer via an SSL-AASIST graph model to balance the foundation network representations.
