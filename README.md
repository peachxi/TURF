Multi-Modal Sentiment Analysis with Reliability-Weighted Fusion
Project Overview

This project implements a multi-modal sentiment analysis system using text, audio, and visual modalities, incorporating uncertainty modeling and reliability-weighted feature fusion (RFF). Key features include:

BERT backbone for text encoding with optional partial layer unfreezing.
Audio and visual features modeled via BiLSTM or Learnable Temporal Pooling (LTP).
Token-level uncertainty estimation and modality reliability weighting.
Flexible training strategies including EMA weight updates, F1 optimization, contrastive and correlation-based losses.
Designed for the MOSI dataset, supporting multi-modal sentiment prediction and ablation studies.
File Structure
File	Description
config_mr.py	Defines the ConfigMR class with training hyperparameters, model architecture options, and loss weights.
model_mr.py	Implements the MRModel class: BERT text encoder, audio/vision feature processors, fusion modules, token-level uncertainty, and gating mechanisms.
trainer_mr.py	Implements HybridTrainer for training, evaluation, loss calculation, dynamic weighting, and reliability supervision.
run_train_mr.py	Entry point: loads data, initializes the model and trainer, and starts the training loop.
Environment
Python 3.8+
PyTorch 2.x
Transformers library (pip install transformers)
NumPy, SciPy for metrics computation
CUDA-enabled GPU for training (recommended)
Installation
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt

Note: Ensure the MOSI dataset is downloaded and the path is correctly set in config_mr.py under data_path.

Usage
Set up configuration
Edit config_mr.py to adjust hyperparameters, dataset paths, fusion weights, and training options.
Run training
python run_train_mr.py
Outputs
Model checkpoints: saved under save_dir specified in the config (default: ./ckpt_mosi_maeonly).
Diagnostics CSV: logs metrics and weights per epoch (diag.csv).
Optionally, visualization of dynamic gating and loss metrics can be added via the trainer.
Features
Reliability-Weighted Fusion (RFF): combines modality features according to dynamic reliability scores.
Token-level Uncertainty: estimates per-token variance to model prediction confidence.
Learnable Temporal Pooling (LTP): adaptive temporal pooling for audio and visual sequences.
Dynamic Modality Gating: modulates the contribution of each modality based on reliability.
Losses & Regularization:
Heteroscedastic sentence/token losses
Correlation and CCC losses
Focal and sign auxiliary losses
Optional contrastive and positional correlation losses
