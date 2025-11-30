import pywt, wfdb, os, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from baseline_A_autoencoderCNN import DenoiseCNN

ROOT = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = ROOT / "data/processed/test.npz"
save_dir = ROOT / "results/figs/Denoise_with_Wavelet_Thresholding.png"
checkpoint_dir = ROOT / "checkpoints/best.pt"
ts_dir = ROOT / "results/denoise.pth"

class NPZECGWindows(Dataset):
    """
    Expects an .npz with keys: noisy (N,L), clean (N,L).
    Returns tensors (noisy, clean) with shape (1,L), already z-scored per window
    (as you did during dataset synthesis).
    """
    def __init__(self, path: Path, limit: int = None):
        obj = np.load(path)
        self.noisy = obj["noisy"]  # (N,L)
        self.clean = obj["clean"]  # (N,L)
        if limit is not None:
            self.noisy = self.noisy[:limit]
            self.clean = self.clean[:limit]
        assert self.noisy.shape == self.clean.shape
        self.N, self.L = self.noisy.shape

    def __len__(self): return self.N

    def __getitem__(self, idx):
        x = torch.from_numpy(self.noisy[idx]).float().unsqueeze(0)  # (1,L)
        y = torch.from_numpy(self.clean[idx]).float().unsqueeze(0)  # (1,L)
        return x, y

def _make_loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=shuffle,
        num_workers=max(1, min(4, (os.cpu_count() or 2) // 2)),
        pin_memory="cuda",
        drop_last=False
)

def wavelet_transform(ecg_tensor, wavelet="db4", mode="soft", thresh=None):
    ecg_np = ecg_tensor.detach().cpu().numpy()
    denoised = np.zeros_like(ecg_np)
    
    for i in range(ecg_np.shape[0]):      # batch
        for j in range(ecg_np.shape[1]):  # channels (1)
            sig = ecg_np[i, j]

            # Wavelet decomposition
            coeffs = pywt.wavedec(sig, wavelet)

            # Compute threshold if not given
            if thresh is None:
                # Universal threshold (VisuShrink)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                thresh_val = sigma * np.sqrt(2 * np.log(len(sig)))
            else:
                thresh_val = thresh

            # Thresholding detail coefficients
            coeffs_thresh = coeffs.copy()
            coeffs_thresh[1:] = [pywt.threshold(c, thresh_val, mode=mode)
                                 for c in coeffs[1:]]

            # Reconstruct
            denoised[i, j] = pywt.waverec(coeffs_thresh, wavelet)

    return torch.from_numpy(denoised).to(ecg_tensor.device)

def main():
    model = DenoiseCNN().to(device).eval()
    if Path(checkpoint_dir).exists():
        sd = torch.load(checkpoint_dir, map_location=device).get("model", None)
        if sd is not None:
            model.load_state_dict(sd, strict=False)

    ds_t = NPZECGWindows(test_dir)
    ecg = _make_loader(ds_t, False)
    
    for x,y in ecg:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        original_ecg = x
        denoise_ecg = wavelet_transform(x)
        # denoise_ecg = model(x)
        torch.save(denoise_ecg, ts_dir)
        break

    print("[DONE]\n\nStart plotting")
    
    original = original_ecg[0, 0, :]
    denoise = denoise_ecg[0, 0, :]
    time = torch.arange(0, len(denoise)) / 1

    plt.figure(figsize=(12,4))
    plt.subplot(2,1,1)
    plt.plot(time.numpy(), original.numpy(), color="red")
    plt.title("Original ECG")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(time.numpy(), denoise.detach().numpy(), color="blue")
    plt.title("Denoised ECG")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=200)
    plt.close()
    print(f"[SAVED] {save_dir}")

if __name__ == "__main__":
    main()