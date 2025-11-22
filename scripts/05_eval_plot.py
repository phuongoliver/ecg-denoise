#!/usr/bin/env python3
import numpy as np, pandas as pd, torch, math
from pathlib import Path
from sklearn.metrics import mean_squared_error
from model_train import DenoiseCNN_SWT_Transformer
from baseline_filter import baseline_filter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT  = SCRIPT_DIR.parent
DATA_DIR   = PROJ_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
RESULT_DIR = PROJ_ROOT / "results"

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
    model = DenoiseCNN_SWT_Transformer().to(device).eval()
    if Path(ckpt).exists():
        sd = torch.load(ckpt, map_location=device).get("model", None)
        if sd is not None:
            model.load_state_dict(sd, strict=False)

    # Metrics running sum
    sum_snr_in = sum_snr_out_base = sum_snr_out_dl = 0.0
    sum_dsnr_base = sum_dsnr_dl = 0.0
    sum_mse_base = sum_mse_dl = 0.0
    nb = math.ceil(N / batch_size)

    with torch.no_grad():
        for bi in range(nb):
            s, e = bi * batch_size, min(N, (bi+1) * batch_size)
            clean_b, noisy_b = clean[s:e], noisy[s:e]

            # Baseline filter
            base_b = baseline_filter(noisy_b, fs=fs, notch_freq=50.0)

            # DL model
            x = torch.from_numpy(noisy_b).to(device).unsqueeze(1)
            y = model(x).squeeze(1).cpu().numpy()

            # Metrics batch
            snr_in_b   = snr_db(clean_b, noisy_b)
            snr_out_bf = snr_db(clean_b, base_b)
            snr_out_dl = snr_db(clean_b, y)

            dsnr_bf = snr_out_bf - snr_in_b
            dsnr_dl = snr_out_dl - snr_in_b

            mse_bf = np.mean((clean_b - base_b)**2, axis=1)
            mse_dl = np.mean((clean_b - y)**2, axis=1)

            sum_snr_in        += np.mean(snr_in_b)
            sum_snr_out_base  += np.mean(snr_out_bf)
            sum_snr_out_dl    += np.mean(snr_out_dl)
            sum_dsnr_base     += np.mean(dsnr_bf)
            sum_dsnr_dl       += np.mean(dsnr_dl)
            sum_mse_base      += np.mean(mse_bf)
            sum_mse_dl        += np.mean(mse_dl)

            if (bi+1) % max(1, nb//10) == 0 or (bi+1)==nb:
                print(f"[{bi+1}/{nb}] done")

    # Averages
    def avg(x): return x / nb
    base_m = [avg(sum_snr_in), avg(sum_snr_out_base), avg(sum_dsnr_base), avg(sum_mse_base)]
    dl_m   = [avg(sum_snr_in), avg(sum_snr_out_dl),   avg(sum_dsnr_dl),   avg(sum_mse_dl)]

    df = pd.DataFrame([
        ["Baseline filter", *[f"{v:.2f}" for v in base_m]],
        ["DL denoiser",     *[f"{v:.2f}" for v in dl_m]],
    ], columns=["Method", "SNR_in (dB)", "SNR_out (dB)", "ΔSNR (dB)", "MSE"])
    print(df)
    df.to_csv(RESULT_DIR / "results_table1.csv", index=False)

if __name__ == "__main__":
    main()
