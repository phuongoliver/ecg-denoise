import numpy as np
import torch
import pywt
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.datasets.dataset import NPZECGWindows

# Configuration
TEST_FILE = ROOT_DIR / "data/processed/test.npz"
SAVE_FIG_PATH = ROOT_DIR / "results/figs/wavelet_denoising_demo.png"
SAVE_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

def wavelet_transform(ecg_tensor, wavelet="db4", mode="soft", thresh=None):
    """
    Applies Wavelet Thresholding Denoising.
    Input: ecg_tensor (Batch, Channels, Length)
    Output: Denoised Tensor (Batch, Channels, Length)
    """
    ecg_np = ecg_tensor.detach().cpu().numpy()
    denoised = np.zeros_like(ecg_np)
    
    # Iterate over batch and channels (PyWavelets works on 1D arrays usually)
    for i in range(ecg_np.shape[0]):      
        for j in range(ecg_np.shape[1]):  
            sig = ecg_np[i, j]

            # 1. Decomposition
            coeffs = pywt.wavedec(sig, wavelet)

            # 2. Thresholding
            if thresh is None:
                # Universal threshold (VisuShrink) estimate
                # sigma = median(|detail_coeffs|) / 0.6745
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                thresh_val = sigma * np.sqrt(2 * np.log(len(sig)))
            else:
                thresh_val = thresh

            # Apply soft thresholding to detail coefficients (ignore approximation coeffs[0])
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(c, thresh_val, mode=mode) for c in coeffs[1:]]

            # 3. Reconstruction
            rec = pywt.waverec(coeffs_thresh, wavelet)
            
            # Fix potential length mismatch due to padding in waverec
            if len(rec) > len(sig):
                rec = rec[:len(sig)]
            elif len(rec) < len(sig):
                rec = np.pad(rec, (0, len(sig)-len(rec)), 'constant')
                
            denoised[i, j] = rec

    return torch.from_numpy(denoised).to(ecg_tensor.device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Wavelet Benchmark on {device}")

    # Load Data using existing Dataset class
    if not TEST_FILE.exists():
        print(f"[ERROR] Test file not found at {TEST_FILE}")
        return

    ds = NPZECGWindows(TEST_FILE, limit=64) # Load small batch for demo
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # Get one sample
    x, y_clean = next(iter(dl)) # x is noisy, y is clean
    x = x.to(device)

    # Run Wavelet Denoising
    y_denoised = wavelet_transform(x, wavelet="db4", mode="soft")

    # Plotting
    noisy_sig = x[0, 0, :].cpu().numpy()
    clean_sig = y_clean[0, 0, :].numpy()
    denoised_sig = y_denoised[0, 0, :].cpu().numpy()
    time_ax = np.arange(len(noisy_sig)) / 360.0

    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_ax, clean_sig, color="green", label="Clean Reference")
    plt.title("Clean ECG")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(time_ax, noisy_sig, color="red", label="Noisy Input")
    plt.title("Noisy Input")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(time_ax, denoised_sig, color="blue", label="Wavelet Denoised details")
    plt.title("Wavelet Denoised (db4)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=150)
    print(f"[SUCCESS] Plot saved to {SAVE_FIG_PATH}")

if __name__ == "__main__":
    main()
