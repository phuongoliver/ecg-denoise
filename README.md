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

Chào mọi người, để hoàn thiện bài Mini-project, mình đã setup xong khung sườn dự án. Dưới đây là hướng dẫn cài đặt và phân chia nhiệm vụ cụ thể cho A và B. Các bạn tự chọn A hoặc B nha.

### 🛠 PHẦN 1: CÀI ĐẶT MÔI TRƯỜNG (Làm đầu tiên)
Do dữ liệu ECG khá nặng, chúng ta sẽ **không push data lên GitHub**. Quy trình setup như sau:

**1. Clone Code:**
```bash
git clone https://github.com/phuongoliver/ecg-denoise.git
cd ecg-denoise
```
**2. Tải Dữ liệu:**

Vào link Google Drive này: [LINK](https://drive.google.com/file/d/1AmlfymWhjkB8n__mKqd8GE-PkzRuUTXq/view?usp=sharing)

Tải file data.zip và giải nén.

**Quan trọng:** Copy thư mục data vừa giải nén vào thư mục gốc của dự án (ngang hàng với scripts, models).

**3. Cài thư viện:**

Bash

# Tạo virtual environment
```bash
python -m venv .venv
```
# Windows:
```bash
.venv\Scripts\activate
```
# Mac/Linux:
```bash
source .venv/bin/activate
```
# Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**4. Test:** Chạy thử python scripts/00_EDA.py. Nếu hiện biểu đồ ECG là thành công.

### 👤 PHẦN 2: NHIỆM VỤ CỤ THỂ

**👉 Nhiệm vụ cho A: Baseline Comparison (So sánh)**
- Mục tiêu: "Compared with other methods" + "Discuss advantages"

- Nội dung: Bạn cần chạy 2 phương pháp lọc nhiễu khác để so sánh với model của mình proposed.

1. Phương pháp 1: Wavelet Transform (Truyền thống)
- Sử dụng thư viện PyWavelets (đã có trong requirements).
- Viết script dùng hàm pywt.threshold để lọc nhiễu trên tập test set giống như model chính đang dùng.
Lưu ý: Tham khảo các phương pháp truyền thống thường hạn chế trong việc loại bỏ đa dạng các loại nhiễu.

2. Phương pháp 2: Simple CNN Autoencoder (Deep Learning cơ bản)
- Dựng một model CNN 1D đơn giản (không có SWT, không có Transformer).
- Train nhanh trên data hiện có (khoảng 5-10 epochs). Model của mình là train trên 10 epochs.

**Output yêu cầu:**
- Một bảng so sánh 3 cột: Metric (SNR, MSE, Parameters, Inferenced Time) | Wavelet | Simple CNN | Ours.
- Nhận xét: Tại sao model của nhóm mình (kết hợp SWT + Transformer) lại tốt hơn CNN thường?
- Mở rộng: Nếu có thời gian thì có thể chạy một model DL khác nặng nhưng có kết quả có thể tốt hơn.

**👉 Nhiệm vụ cho bạn B: Ablation Study & Validation**
- Mục tiêu: "Evaluate with relevant data"  + "Improvement direction"

- Nội dung: Chứng minh độ hiệu quả và tính bền vững của kiến trúc model.

1. Ablation Study (Nghiên cứu loại bỏ):
- Thử bỏ module Transformer hoặc SWT ra khỏi kiến trúc code hiện tại.
- Train lại và xem chỉ số SNR giảm bao nhiêu.
Ý nghĩa: Chứng minh rằng các module này là cần thiết, không thể bỏ đi.

2. Robustness Test (Kiểm thử độ bền):
- Sử dụng model chính (Ours), chạy test trên một mức nhiễu khác như -5dB (ví dụ: nhiễu cực đại hoặc nhiễu cực tiểu).
- Hoặc/Và: Lấy 1 file ECG bất kỳ từ bộ dữ liệu khác (như QT Database trên PhysioNet) để chạy demo khử nhiễu.

**Output yêu cầu:**
- Biểu đồ so sánh SNR khi có/không có các module.
- Hình ảnh sóng ECG trước và sau khi lọc trên dữ liệu mới/mức nhiễu mới.

**📅 QUY ĐỊNH CHUNG**
- Code: Viết script mới trong thư mục scripts/, đặt tên là baseline_A.py và ablation_B.py để tránh đụng code chính.
- Commit: Khi push code, nhớ KHÔNG ADD folder data hay checkpoints.
- Deadline: Cố gắng xong code và có số liệu sơ bộ trước [Thứ 3, 25 tháng 11/14h Chiều] để có gì mình chỉnh sửa thêm và làm slide.

