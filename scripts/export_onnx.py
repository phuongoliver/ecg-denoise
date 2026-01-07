#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.architecture import DenoiseCNN_SWT_Transformer
from src.utils.metrics import count_params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    ap.add_argument("--out", type=str, default="denoiser.onnx", help="Output ONNX file")
    ap.add_argument("--seq-len", type=int, default=720, help="ECG window length (default 2s @ 360 Hz)")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    model = DenoiseCNN_SWT_Transformer(in_ch=1, base_ch=16, nhead=4, ffn_mult=2, num_transformer_layers=2)
    model.to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    print(f"[INFO] Loaded checkpoint {args.ckpt}")
    print(f"[INFO] Params: {count_params(model):,}")

    dummy = torch.randn(1, 1, args.seq_len).to(device)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["noisy_ecg"],
        output_names=["denoised_ecg"],
        dynamic_axes={
            "noisy_ecg": {0: "batch", 2: "length"},
            "denoised_ecg": {0: "batch", 2: "length"},
        },
        opset_version=17,
    )

    print(f"[DONE] Exported ONNX model to {args.out}")

if __name__ == "__main__":
    main()