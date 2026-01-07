#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.architecture import DenoiseCNN_SWT_Transformer, DenoiseCNN
from src.utils.baseline import baseline_filter
from scripts.evaluate import wavelet_transform  # Reuse from evaluate

def get_args():
    parser = argparse.ArgumentParser(description="Visualize Denoising Results")
    parser.add_argument("--data-file", type=str, default="data/processed/test.npz")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best.pt")
    parser.add_argument("--model-type", type=str, default="full", choices=["full", "baseline-cnn"])
    parser.add_argument("--output", type=str, default="results/figs/denoising_comparison.png")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Load Data
    data_path = (ROOT_DIR / args.data_file).resolve()
    if not data_path.exists():
        print(f"[ERROR] Data not found: {data_path}")
        return

    obj = np.load(data_path)
    noisy_all = obj["noisy"]
    clean_all = obj["clean"]
    fs = int(obj["fs"][0]) if "fs" in obj else 360

    if args.sample_idx >= len(noisy_all):
        print(f"Sample index {args.sample_idx} out of bounds (max {len(noisy_all)-1})")
        return

    noisy_np = noisy_all[args.sample_idx]
    clean_np = clean_all[args.sample_idx]

    # Prepare inputs
    x_tensor = torch.from_numpy(noisy_np).float().unsqueeze(0).unsqueeze(0).to(args.device) # (1,1,L)

    # 1. Baseline Filter
    val_base = baseline_filter(noisy_np.reshape(1, -1), fs=fs, notch_freq=50.0).flatten()

    # 2. Wavelet
    val_wt = wavelet_transform(x_tensor).squeeze().cpu().numpy()

    # 3. DL Model
    if args.model_type == "baseline-cnn":
        model = DenoiseCNN(in_ch=1, base_ch=16)
    else:
        model = DenoiseCNN_SWT_Transformer(in_ch=1, base_ch=16)
    
    model = model.to(args.device).eval()
    
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[INFO] Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        # Fix keys
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}, using random weights")

    with torch.no_grad():
        val_dl = model(x_tensor).squeeze().cpu().numpy()

    # Plot
    t = np.arange(len(noisy_np)) / fs
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Original (Noisy + Clean ref)
    axs[0].plot(t, noisy_np, color='red', alpha=0.7, label='Noisy Input')
    axs[0].plot(t, clean_np, color='green', linestyle='--', linewidth=1, label='Clean Reference')
    axs[0].set_title(f"Input (Sample {args.sample_idx})")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # Baseline Filter
    axs[1].plot(t, val_base, color='orange', label='Butterworth + Notch')
    axs[1].plot(t, clean_np, color='green', linestyle='--', linewidth=0.8)
    axs[1].set_title("Baseline Filter")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)

    # Wavelet
    axs[2].plot(t, val_wt, color='blue', label='Wavelet (db4)')
    axs[2].plot(t, clean_np, color='green', linestyle='--', linewidth=0.8)
    axs[2].set_title("Wavelet Denoising")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    # DL Model
    axs[3].plot(t, val_dl, color='purple', label=f'DL Model ({args.model_type})')
    axs[3].plot(t, clean_np, color='green', linestyle='--', linewidth=0.8)
    axs[3].set_title(f"Deep Learning Model")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)
    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[SUCCESS] Figure saved to {out_path}")

if __name__ == "__main__":
    main()
