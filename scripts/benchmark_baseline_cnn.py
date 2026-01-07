import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import shared modules
from src.datasets.dataset import NPZECGWindows
from src.training.engine import train_one_epoch, evaluate
from src.models.architecture import DenoiseCNN
from src.utils.metrics import count_params, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/processed")
    ap.add_argument("--train-file", type=str, default="train.npz")
    ap.add_argument("--val-file",   type=str, default="val.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=10) # Baseline train 10 epochs as requested
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--seed",       type=int, default=123)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--save-dir",   type=str, default="results/checkpoints/baseline_cnn")
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    # Paths
    data_dir = (ROOT_DIR / args.data_dir).resolve()
    train_path = data_dir / args.train_file
    val_path   = data_dir / args.val_file
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Training Baseline CNN (Simple Autoencoder)...")
    print(f"[INFO] Device: {args.device}")

    # Datasets
    if not train_path.exists():
        print(f"[ERROR] Data not found: {train_path}")
        return

    ds_tr = NPZECGWindows(train_path, limit=args.limit_train)
    ds_va = NPZECGWindows(val_path,   limit=1000) # Quick val

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model: Simple CNN Autoencoder (No SWT, No Transformer)
    model = DenoiseCNN(in_ch=1, base_ch=16).to(args.device)
    print(f"[INFO] Params: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() # Baseline often uses MSE
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device=="cuda"))

    best_val_snr = -1e9
    log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_snr_out = train_one_epoch(model, dl_tr, optimizer, scaler, args.device, loss_fn)
        va_loss, va_snr_out, va_snr_in, va_snr_gain = evaluate(model, dl_va, args.device, loss_fn)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | TrLoss: {tr_loss:.4f} | ValSNR_Out: {va_snr_out:.2f}dB | Gain: {va_snr_gain:.2f}dB | {dt:.1f}s")
        
        log.append({"epoch": epoch, "val_snr_out": va_snr_out})

        if va_snr_out > best_val_snr:
            best_val_snr = va_snr_out
            torch.save(model.state_dict(), save_dir / "best_baseline.pth")

    print(f"[DONE] Best Baseline SNR: {best_val_snr:.2f} dB")
    with open(save_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()
