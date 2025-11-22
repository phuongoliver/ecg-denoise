# Efficient CNN-SWT-Transformer for Fog-Based ECG Denoising

## 📌 Project Overview
This project proposes a lightweight Deep Learning architecture designed to remove noise from Electrocardiogram (ECG) signals directly on Fog/Edge computing devices (e.g., Raspberry Pi). By combining CNN, Stationary Wavelet Transform (SWT), and Transformers, we achieve high-performance denoising with low latency, eliminating the need for heavy Cloud-based processing.

## 1. The Problem
Real-world ECG monitoring faces significant challenges:
* **Signal Noise:** ECG signals are highly susceptible to noise from muscle movement, breathing artifacts, and power line interference, often leading to cardiovascular misdiagnosis.
* **Limitations of Classical Filtering:** Traditional methods (like Band-pass filters) often fail to clean the signal thoroughly or result in signal distortion (losing critical clinical features).
* **Cloud-AI Limitations:** While modern Deep Learning models offer excellent denoising, they are typically too heavy. Deploying them on the Cloud incurs high costs, significant latency, and data privacy concerns.

## 2. Our Solution
We developed an "Ultra-Lightweight" AI model optimized for execution on Fog Computing devices (specifically tested on **Raspberry Pi 3**).

### Architecture
The model utilizes a hybrid approach:
1.  **SWT (Stationary Wavelet Transform):** For time-frequency signal decomposition.
2.  **CNN (Convolutional Neural Network):** For local feature extraction.
3.  **Transformer:** Leveraging self-attention mechanisms to capture long-range dependencies in the signal.

## 3. Key Results
Our model achieves a balance between high accuracy and computational efficiency suitable for real-time IoT applications.

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Denoising Performance** | **+15.6 dB** | Improvement in SNR (Signal-to-Noise Ratio) compared to standard filters. |
| **Model Size** | **0.15 Million** | Extremely low parameter count. |
| **Inference Speed** | **~22 ms** | Running on Raspberry Pi 3 (Real-time capable). |

---

## 📄 HƯỚNG DẪN THỰC HIỆN MINI-PROJECT DATA MINING
**Topic:** Efficient CNN-SWT-Transformer for Fog-Based ECG Denoising
**Định hướng:** Type 1 (Propose Improvement to Existing Techniques)

Chào mọi người, để hoàn thiện bài Mini-project với mục tiêu đạt điểm tối đa ở các mục **Evaluation** và **Comparison**, mình (Phương) đã setup xong khung sườn dự án. Dưới đây là hướng dẫn cài đặt và phân chia nhiệm vụ cụ thể cho A và B.

### 🛠 PHẦN 1: CÀI ĐẶT MÔI TRƯỜNG (Làm đầu tiên)
Do dữ liệu ECG khá nặng, chúng ta sẽ **không push data lên GitHub**. Quy trình setup như sau:

**1. Clone Code:**
```bash
git clone [https://github.com/phuongoliver/ecg-denoise.git](https://github.com/phuongoliver/ecg-denoise.git)
cd ecg-denoise