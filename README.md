# Multi-Modal Sentiment Analysis with Reliability-Weighted Fusion

## Project Overview
This project implements a **multi-modal sentiment analysis** system using **text, audio, and visual modalities**, incorporating **uncertainty modeling** and **reliability-weighted feature fusion (RFF)**. Key features include:

- BERT backbone for text encoding with optional partial layer unfreezing.
- Audio and visual features modeled via BiLSTM or Learnable Temporal Pooling (LTP).
- Token-level uncertainty estimation and modality reliability weighting.
- Flexible training strategies including EMA weight updates, F1 optimization, contrastive and correlation-based losses.
- Designed for the **MOSI dataset**, supporting multi-modal sentiment prediction and ablation studies.

---

## File Structure

| File | Description |
|------|-------------|
| `config_mr.py` | Defines the `ConfigMR` class with training hyperparameters, model architecture options, and loss weights. |
| `model_mr.py` | Implements the `MRModel` class: BERT text encoder, audio/vision feature processors, fusion modules, token-level uncertainty, and gating mechanisms. |
| `trainer_mr.py` | Implements `HybridTrainer` for training, evaluation, loss calculation, dynamic weighting, and reliability supervision. |
| `run_train_mr.py` | Entry point: loads data, initializes the model and trainer, and starts the training loop. |

---

## Environment

- Python 3.8+
- PyTorch 2.x
- Transformers library (`pip install transformers`)
- NumPy, SciPy for metrics computation
- CUDA-enabled GPU for training (recommended)

---

## Installation

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
