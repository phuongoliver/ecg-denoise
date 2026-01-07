# Efficient CNN-SWT-Transformer for Fog-Based ECG Denoising

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Device](https://img.shields.io/badge/Device-Edge%2FCloud-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview
This project proposes a **lightweight Deep Learning architecture** designed to remove noise from Electrocardiogram (ECG) signals directly on **Fog/Edge computing devices** (e.g., Raspberry Pi, Jetson Nano). By combining **Convolutional Neural Networks (CNN)**, **Stationary Wavelet Transform (SWT)**, and **Transformers**, we achieve high-performance denoising with low latency, eliminating the need for heavy Cloud-based processing.

## 1. The Problem
Real-world ECG monitoring faces significant challenges:
* **Signal Noise:** ECG signals are highly susceptible to noise from muscle movement, breathing artifacts, and power line interference, often leading to cardiovascular misdiagnosis.
* **Limitations of Classical Filtering:** Traditional methods (like Band-pass filters) often fail to clean the signal thoroughly or result in signal distortion (losing critical clinical features).
* **Cloud-AI Limitations:** While modern Deep Learning models offer excellent denoising, they are typically too heavy. Deploying them on the Cloud incurs high costs, significant latency, and data privacy concerns.

## 2. Our Solution
We developed an **"Ultra-Lightweight"** AI model optimized for execution on Fog Computing devices.

### Architecture
The model utilizes a hybrid approach:
1.  **SWT (Stationary Wavelet Transform):** For time-frequency signal decomposition.
2.  **CNN (Convolutional Neural Network):** For local feature extraction.
3.  **Transformer:** Leveraging self-attention mechanisms to capture long-range dependencies in the signal.

## 3. Key Results
Our proposed method demonstrates superior performance compared to traditional and basic deep learning methods, validated through comprehensive ablation studies.

### 🏆 Benchmark Comparison
Comparison against baseline methods (Conducted by @TNAK2004):

| Method | Input SNR (dB) | SNR Output (dB) ↑ | ΔSNR (Gain) ↑ | MSE ↓ | Inference (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Filter** | 5.00 | 0.51 | -4.49 | 0.17 | ~0.00 |
| **Wavelet Transform** | 5.00 | 5.01 | +0.01 | 0.08 | 0.03 |
| **Simple CNN (AE)** | 5.00 | 11.65 | +6.65 | 0.03 | 0.04 |
| **Ours (SWT+Trans)** | 5.00 | **15.60** | **+10.60** | **0.02** | 0.02 |

> *Note: Results evaluated on Test Set with SNR_in = 5dB.*

### 🧩 Ablation Study
Investigating the contribution of each module (Conducted by @PhucCodee):

| Architecture Variant | Best SNR (dB) | Impact vs Full | Observation |
| :--- | :---: | :---: | :--- |
| **No-Transformer** (Ablation) | 15.77 | -3.92 dB | **Major Drop**. Transformer is critical for performance. |
| **No-SWT** (Ablation) | 19.35 | -0.34 dB | Slight Drop. SWT helps with fine detail stability. |
| **Full Model** (CNN-SWT-Trans) | **19.69** | - | Optimal performance. |

> *Note: Transformer contributes significantly (+3.92dB) to the denoising capability, while SWT adds refinement (+0.34dB).*

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* NumPy, SciPy

### Installation

```bash
git clone https://github.com/phuongoliver/ecg-denoise.git
cd ecg-denoise
python -m venv .venv
# Activate venv (Windows: .venv\Scripts\activate, Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

### Data Preparation
This project uses the **MIT-BIH Arrhythmia Database** and **NSTDB** (Noise Stress Test Database).
Since raw data is not included in the repo, you need to generate synthetic noisy data:

```bash
python scripts/download_data.py  # (Optional if you have the data)
python scripts/prepare_data.py --snr-db -5 0 5
```
This will create `train.npz`, `val.npz`, `test.npz` in `data/processed`.

### Usage

**1. Training**
To train the main model:
```bash
python scripts/train.py --epochs 50 --batch-size 64 --device cuda
```

**2. Training Variants (Ablation Studies)**
You can train different variants to verify the effectiveness of each module:
```bash
# Train without SWT (Transformer only)
python scripts/train.py --model-variant no-swt

# Train without Transformer (SWT only)
python scripts/train.py --model-variant no-transformer

# Train Baseline Simple CNN (Autoencoder)
python scripts/train.py --model-variant baseline-cnn
```

**3. Evaluation & ONNX Export**
```bash
# Export to ONNX for Edge deployment
python scripts/export_onnx.py --ckpt checkpoints/best_model.pth --out denoiser.onnx

# Evaluate on Test Set
python scripts/evaluate.py
```

---

## 🤝 Project Structure

```
├── configs/             # Configuration files (Hyperparameters)
├── scripts/             # Execution scripts (train, data gen, export)
├── src/                 # Source code
│   ├── models/          # Model architectures (Main, Variants, Baselines)
│   ├── datasets/        # Dataset loading logic
│   ├── training/        # Training & Evaluation loops
│   └── utils/           # Metrics & Helpers
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for experiments/visualization
└── results/             # Logs and Checkpoints
```

## 📜 Credits & References
This project is based on research into efficient deep learning for biomedical signal processing.
We extended the standard CS-TRANS methodologies by:
- Optimizing layers for Edge deployment.
- Introducing a weighted Peak-MSE loss for better QRS reconstruction.
- Conducting comprehensive ablation studies.

## ⚖️ Acknowledgements & License
This project is an **independent re-implementation and extension** of the architecture proposed in the research:
> *CS-TRANS: An efficient deep learning model for ECG denoising.* [DOI: 10.1016/j.bspc.2024.106441](https://doi.org/10.1016/j.bspc.2024.106441)

If you use this code, please credit the original authors and this repository.

**Contributors:**
- **[Tran Mai Phuong (@phuongoliver)](https://github.com/phuongoliver)** – *Lead Researcher & Implementation:* Core architecture re-implementation, Edge optimization, and weighted Peak-MSE loss.
- **[Trần Hoàng Phúc (@PhucCodee)](https://github.com/PhucCodee)** – *Baseline Comparison:* Implemented Wavelet Transform and Simple CNN Autoencoder baselines for performance benchmarking.
- **[Khoi Tran Nguyen Anh (@TNAK2004)](https://github.com/TNAK2004)** – *Ablation & Validation:* Conducted ablation studies (Effect of SWT/Transformer) and robustness testing on diverse noise levels.