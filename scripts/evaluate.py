#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import math
import argparse
import pywt
from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import models from src
from src.models.architecture import DenoiseCNN_SWT_Transformer, DenoiseCNN, DenoiseCNN_NoSWT, DenoiseCNN_NoTransformer
from src.utils.baseline import baseline_filter

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Denoising Models")
    parser.add_argument("--data-dir", type=str, default="data/processed_more_noise_neg", help="Directory containing test.npz")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--model-type", type=str, default="full", 
                        choices=["full", "baseline-cnn", "no-swt", "no-transformer"],
                        help="Model architecture type")
    parser.add_argument("--result-dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()

def wavelet_transform(ecg_tensor, wavelet="db4", mode="soft", thresh=None):
    """
    Applies Wavelet Thresholding Denoising.
    Input: ecg_tensor (Batch, Channels, Length)
    Output: Denoised Tensor (Batch, Channels, Length)
    """
    ecg_np = ecg_tensor.detach().cpu().numpy()
    denoised = np.zeros_like(ecg_np)
    
    # Iterate over batch and channels
    for i in range(ecg_np.shape[0]):      
        for j in range(ecg_np.shape[1]):  
            sig = ecg_np[i, j]
            # 1. Decomposition
            coeffs = pywt.wavedec(sig, wavelet)
            # 2. Thresholding
            if thresh is None:
                # Universal threshold (VisuShrink)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                thresh_val = sigma * np.sqrt(2 * np.log(len(sig)))
            else:
                thresh_val = thresh
            # Apply soft thresholding
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(c, thresh_val, mode=mode) for c in coeffs[1:]]
            # 3. Reconstruction
            rec = pywt.waverec(coeffs_thresh, wavelet)
            # Fix length
            if len(rec) > len(sig): rec = rec[:len(sig)]
            elif len(rec) < len(sig): rec = np.pad(rec, (0, len(sig)-len(rec)), 'constant')
            denoised[i, j] = rec

    return torch.from_numpy(denoised).to(ecg_tensor.device)

def snr_db(clean, est, eps=1e-12):
    num = np.sum(clean**2, axis=1)
    den = np.sum((clean - est)**2, axis=1) + eps
    return 10*np.log10(num/den)

def load_model(model_type, checkpoint_path, device):
    if model_type == "baseline-cnn":
        model = DenoiseCNN(in_ch=1, base_ch=16)
    elif model_type == "no-swt":
        model = DenoiseCNN_NoSWT(in_ch=1, base_ch=16)
    elif model_type == "no-transformer":
        model = DenoiseCNN_NoTransformer(in_ch=1, base_ch=16)
    else:
        model = DenoiseCNN_SWT_Transformer(in_ch=1, base_ch=16)
    
    model = model.to(device).eval()
    
    if Path(checkpoint_path).exists():
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Handle if checkpoint wraps state_dict in 'model' key or is direct
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        
        # Helper to remove 'module.' prefix if saved with DataParallel
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[new_k] = v
            
        try:
            model.load_state_dict(new_sd, strict=False)
        except Exception as e:
            print(f"[WARN] Error loading state_dict: {e}")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}. Using random initialization.")
        
    return model

def main():
    args = get_args()
    
    DATA_DIR = (ROOT_DIR / args.data_dir).resolve()
    RESULT_DIR = (ROOT_DIR / args.result_dir).resolve()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    data_npz = DATA_DIR / "test.npz"
    if not data_npz.exists():
        print(f"[ERROR] Test data not found: {data_npz}")
        return

    # Load data
    print(f"[INFO] Loading data from {data_npz}...")
    obj   = np.load(data_npz, mmap_mode="r")
    noisy = obj["noisy"].astype(np.float32)   # (N,L)
    clean = obj["clean"].astype(np.float32)   # (N,L)
    fs    = int(obj["fs"][0]) if "fs" in obj else 360
    
    N, L  = noisy.shape
    if args.limit is not None:
        N = min(N, args.limit)
        noisy = noisy[:N]
        clean = clean[:N]

    print(f"[INFO] Test dataset: {N} windows, length: {L}, fs={fs}")

    # Load DL Model
    model = load_model(args.model_type, args.checkpoint, args.device)

    # Metrics accumulators
    metrics = {
        "Baseline Filter": {"snr_in": [], "snr_out": [], "dsnr": [], "mse": []},
        "Wavelet":         {"snr_in": [], "snr_out": [], "dsnr": [], "mse": []},
        f"DL ({args.model_type})":  {"snr_in": [], "snr_out": [], "dsnr": [], "mse": []}
    }

    batch_size = args.batch_size
    nb = math.ceil(N / batch_size)

    print("[INFO] Starting evaluation...")
    with torch.no_grad():
        for bi in range(nb):
            s, e = bi * batch_size, min(N, (bi+1) * batch_size)
            clean_b, noisy_b = clean[s:e], noisy[s:e]

            # 1. Baseline Filter
            base_b = baseline_filter(noisy_b, fs=fs, notch_freq=50.0)
            
            # 2. Wavelet
            x_tensor = torch.from_numpy(noisy_b).to(args.device).unsqueeze(1)
            wt_out = wavelet_transform(x_tensor).squeeze(1).cpu().numpy()
            
            # 3. DL Model
            dl_out = model(x_tensor).squeeze(1).cpu().numpy()

            # --- Metrics Calculation ---
            snr_in_b = snr_db(clean_b, noisy_b)
            
            # Helper to record metrics
            def record(name, est):
                snr_out = snr_db(clean_b, est)
                dsnr    = snr_out - snr_in_b
                mse     = np.mean((clean_b - est)**2, axis=1)
                
                metrics[name]["snr_in"].extend(snr_in_b)
                metrics[name]["snr_out"].extend(snr_out)
                metrics[name]["dsnr"].extend(dsnr)
                metrics[name]["mse"].extend(mse)

            record("Baseline Filter", base_b)
            record("Wavelet", wt_out)
            record(f"DL ({args.model_type})", dl_out)

            if (bi+1) % max(1, nb//5) == 0:
                print(f"  Batch [{bi+1}/{nb}] done")

    # Aggregate Results
    summary_rows = []
    for method, data in metrics.items():
        row = [
            method,
            np.mean(data["snr_in"]),
            np.mean(data["snr_out"]),
            np.mean(data["dsnr"]),
            np.mean(data["mse"])
        ]
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows, columns=["Method", "SNR_in (dB)", "SNR_out (dB)", "Gain (dB)", "MSE"])
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(df.to_string(float_format="{:.2f}".format))
    print("="*50)

    # Save to CSV
    csv_name = f"results_eval_{args.model_type}.csv"
    save_path = RESULT_DIR / csv_name
    df.to_csv(save_path, index=False)
    print(f"[INFO] Results saved to {save_path}")

if __name__ == "__main__":
    main()
