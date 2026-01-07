import os
import argparse
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
# This allows running the script from anywhere: python scripts/train.py
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Project Imports
from src.models.architecture import DenoiseCNN_SWT_Transformer, DenoiseCNN_NoSWT, DenoiseCNN_NoTransformer, DenoiseCNN
from src.models.losses import PeakWeightedMSE
from src.datasets.dataset import NPZECGWindows
from src.training.engine import train_one_epoch, evaluate
from src.utils.metrics import count_params, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/processed")
    ap.add_argument("--train-file", type=str, default="train.npz")
    ap.add_argument("--val-file",   type=str, default="val.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=5)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--wd",         type=float, default=1e-4)
    ap.add_argument("--seed",       type=int, default=123)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val",   type=int, default=None)
    ap.add_argument("--save-dir",   type=str, default="results/checkpoints")
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-variant", type=str, default="full", 
                    choices=["full", "no-swt", "no-transformer", "baseline-cnn"],
                    help="Model variant: full (default), no-swt, no-transformer, or baseline-cnn")
    args = ap.parse_args()

    set_seed(args.seed)

    # Resolve paths relative to repo root
    # script is in scripts/, so parent is root
    ROOT = Path(__file__).resolve().parents[1] 
    data_dir = (ROOT / args.data_dir).resolve()
    train_path = data_dir / args.train_file
    val_path   = data_dir / args.val_file
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not val_path.exists():
        # Fallback to local if running from somewhere else? 
        # But better to just warn.
        print(f"[WARN] Data files not found at {data_dir}. Please ensure data is generated.")
   
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Saving to: {save_dir}")
    print(f"[INFO] Model Variant: {args.model_variant}")

    # ---- Datasets ----
    # Check if files exist before loading
    if train_path.exists():
        ds_tr = NPZECGWindows(train_path, limit=args.limit_train)
        ds_va = NPZECGWindows(val_path,   limit=args.limit_val)
        
        num_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        pin_mem = ("cuda" in args.device.lower())

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    else:
        print("[ERROR] Train/Val files not found. Exiting...")
        return

    # ---- Model ----
    if args.model_variant == "no-swt":
        model = DenoiseCNN_NoSWT(in_ch=1, base_ch=16, nhead=4, ffn_mult=2, num_transformer_layers=2).to(args.device)
        print("[INFO] Using model WITHOUT SWT")
    elif args.model_variant == "no-transformer":
        model = DenoiseCNN_NoTransformer(in_ch=1, base_ch=16).to(args.device)
        print("[INFO] Using model WITHOUT Transformer")
    elif args.model_variant == "baseline-cnn":
        model = DenoiseCNN(in_ch=1, base_ch=16).to(args.device)
        print("[INFO] Using Baseline Simple CNN (Autoencoder)")
    else:
        model = DenoiseCNN_SWT_Transformer(in_ch=1, base_ch=16, nhead=4, ffn_mult=2, num_transformer_layers=2).to(args.device)
        print("[INFO] Using FULL model (SWT + Transformer)")

    print(f"[INFO] Model Params: {count_params(model):,}")

    # ---- Optimizer / Loss ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # Added WD later if needed
    
    # Use MSE for baseline if needed, but PeakWeightedMSE is generally better for ECG. 
    # The original baseline script used MSE. Let's stick to PeakWeightedMSE for fair comparison unless requested otherwise.
    # Actually, original code used MSE for baseline. Let's switch if baseline.
    if args.model_variant == "baseline-cnn":
         loss_fn = nn.MSELoss()
         print("[INFO] Using MSE Loss for Baseline")
    else:
         loss_fn = PeakWeightedMSE()
         print("[INFO] Using PeakWeightedMSE Loss")

    use_cuda_amp = ("cuda" in args.device.lower()) and torch.cuda.is_available()
    # Check if torch.cuda.amp exists (older pytorch) or torch.amp (newer)
    # Sticking to torch.cuda.amp for compatibility as in original script
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    best_val_snr = -1e9
    log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_snr_out = train_one_epoch(model, dl_tr, optimizer, scaler, args.device, loss_fn)
        va_loss, va_snr_out, va_snr_in, va_snr_gain = evaluate(model, dl_va, args.device, loss_fn)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | TrLoss: {tr_loss:.4f} | ValLoss: {va_loss:.4f} | ValSNR_Out: {va_snr_out:.2f}dB | Gain: {va_snr_gain:.2f}dB | {dt:.1f}s")
        
        log.append({
            "epoch": epoch, "train_loss": tr_loss, "val_snr_out": va_snr_out
        })

        if va_snr_out > best_val_snr:
            best_val_snr = va_snr_out
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  [CKPT] Saved best model: {best_val_snr:.2f} dB")

    # Save Last
    torch.save(model.state_dict(), save_dir / "last_model.pth")
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()
