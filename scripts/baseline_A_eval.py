# !/usr/bin/env python3
import numpy as np, pandas as pd, torch, math, time, pywt
from pathlib import Path
from baseline_A_autoencoderCNN import DenoiseCNN
from baseline_filter import baseline_filter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT  = SCRIPT_DIR.parent
DATA_DIR   = PROJ_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
RESULT_DIR = PROJ_ROOT / "results"

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

def snr_db(clean, est, eps=1e-12):
    num = np.sum(clean**2, axis=1)
    den = np.sum((clean - est)**2, axis=1) + eps
    return 10*np.log10(num/den)

def main(device="cpu", batch_size=32, limit=None):
    data_npz = DATA_DIR / "test.npz"
    ckpt = CHECKPOINT_DIR / "best.pt"

    # Load data
    obj   = np.load(data_npz, mmap_mode="r")
    noisy = obj["noisy"].astype(np.float32)   # (N,L)
    clean = obj["clean"].astype(np.float32)   # (N,L)
    fs    = int(obj["fs"][0])
    N, L  = noisy.shape
    if limit is not None:
        N = min(N, limit)
        noisy = noisy[:N]
        clean = clean[:N]

    print(f"[INFO] Test windows: {N}, length: {L}, fs={fs}")

    # Model
    model = DenoiseCNN().to(device).eval()
    if Path(ckpt).exists():
        sd = torch.load(ckpt, map_location=device).get("model", None)
        if sd is not None:
            model.load_state_dict(sd, strict=False)

    # Metrics running sum
    sum_snr_in = sum_snr_out_base = sum_snr_out_cnn = sum_snr_out_wt = 0.0
    sum_dsnr_base = sum_dsnr_cnn = sum_dsnr_wt = 0.0
    sum_mse_base = sum_mse_cnn = sum_mse_wt = 0.0
    nb = math.ceil(N / batch_size)

    with torch.no_grad():
        for bi in range(nb):
            s, e = bi * batch_size, min(N, (bi+1) * batch_size)
            clean_b, noisy_b = clean[s:e], noisy[s:e]

            # Baseline filter
            base_b = baseline_filter(noisy_b, fs=fs, notch_freq=50.0)
            
            # DL model
            x = torch.from_numpy(noisy_b).to(device).unsqueeze(1)
            start_cnn = time.time()
            y1 = model(x).squeeze(1).cpu().numpy()
            time_cnn = time.time() - start_cnn
            start_wt = time.time()
            y2 = wavelet_transform(x).squeeze(1).cpu().numpy()
            time_wt = time.time() - start_wt

            # Metrics batch
            snr_in_b   = snr_db(clean_b, noisy_b)
            snr_out_bf = snr_db(clean_b, base_b)
            snr_out_cnn = snr_db(clean_b, y1)
            snr_out_wt = snr_db(clean_b, y2)

            dsnr_bf = snr_out_bf - snr_in_b
            dsnr_cnn = snr_out_cnn - snr_in_b
            dsnr_wt = snr_out_wt - snr_in_b

            mse_bf = np.mean((clean_b - base_b)**2, axis=1)
            mse_cnn = np.mean((clean_b - y1)**2, axis=1)
            mse_wt = np.mean((clean_b - y2)**2, axis=1)

            sum_snr_in        += np.mean(snr_in_b)
            sum_snr_out_base  += np.mean(snr_out_bf)

            sum_snr_out_cnn    += np.mean(snr_out_cnn)
            sum_snr_out_wt    += np.mean(snr_out_wt)

            sum_dsnr_base     += np.mean(dsnr_bf)
            sum_dsnr_cnn       += np.mean(dsnr_cnn)
            sum_dsnr_wt       += np.mean(dsnr_wt)

            sum_mse_base      += np.mean(mse_bf)
            sum_mse_cnn        += np.mean(mse_cnn)
            sum_mse_wt        += np.mean(mse_wt)

            if (bi+1) % max(1, nb//10) == 0 or (bi+1)==nb:
                print(f"[{bi+1}/{nb}] done")

    # Averages
    def avg(x): return x / nb
    base_m = [avg(sum_snr_in), avg(sum_snr_out_base), avg(sum_dsnr_base), avg(sum_mse_base), 0]
    cnn_m   = [avg(sum_snr_in), avg(sum_snr_out_cnn),   avg(sum_dsnr_cnn),   avg(sum_mse_cnn), time_cnn]
    wt_m   = [avg(sum_snr_in), avg(sum_snr_out_wt),   avg(sum_dsnr_wt),   avg(sum_mse_wt), time_wt]

    df = pd.DataFrame([
        ["Baseline filter", *[f"{v:.2f}" for v in base_m]],
        ["CNN denoiser",     *[f"{v:.2f}" for v in cnn_m]],
        ["WT denoiser",     *[f"{v:.2f}" for v in wt_m]]
    ], columns=["Method", "SNR_in (dB)", "SNR_out (dB)", "ΔSNR (dB)", "MSE", "Inference Time"])
    print(df)
    df.to_csv(RESULT_DIR / "results_table_.csv", index=False)

if __name__ == "__main__":
    main()
