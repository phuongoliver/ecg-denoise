import os, math, time, json, random, argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from losses import PeakWeightedMSE


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv1d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm1d(cout)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, use_dropout: bool = False, p: float = 0.1):
        super().__init__()
        p_pad = k // 2
        self.conv = ConvBNReLU(cin, cout, k=k, s=1, p=p_pad)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p) if use_dropout else nn.Identity()
        nn.init.kaiming_normal_(self.conv.conv.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        skip = x
        x = self.pool(x)
        return x, skip
    
class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBNReLU(cin, cout, k)

    def forward(self, x, skip):
        x = self.up(x)
        # pad/crop to match skip length
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff)) if diff > 0 else x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class DenoiseCNN(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4

        # Encoder k=3
        self.enc1 = EncoderBlock(in_ch, c1, k=3)
        self.enc2 = EncoderBlock(c1,  c2, k=3)
        self.enc3 = EncoderBlock(c2,  c3, k=3, use_dropout=True, p=0.5)
        self.pre_bottleneck = nn.Sequential(
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.Dropout(0.5)
        )
        # Bottleneck
        bott_ch = c3

        # Decoder
        self.dec4 = DecoderBlock(bott_ch + c3, c3, k=3)
        self.dec3 = DecoderBlock(c3 + c2, c2, k=3)   # skip enc2
        self.dec2 = DecoderBlock(c2 + c1, c1, k=3)   # skip enc1
        self.dec1 = ConvBNReLU(c1, c1, k=3)
        self.out  = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x1, skip1 = self.enc1(x)    # (B,c1,L/2),  skip1: (B,c1,L)
        x2, skip2 = self.enc2(x1)   # (B,c2,L/4),  skip2: (B,c2,L/2)
        x3, skip3 = self.enc3(x2)   # (B,c3,L/8),  skip3: (B,c3,L/4)

        # -------- Pre-bottleneck + Bottleneck --------
        b = self.pre_bottleneck(x3)           # ConvReLU+BN+Dropout(0.5), (B,c3,L/8)
        # b = self.swt(b)             # (B,c3,L/8)
        # b = self.transformer(b)     # (B,c3,L/8)

        # -------- Decoder (3 lần upsample đối xứng) --------
        y = self.dec4(b,   skip3)   # (B,c3,L/4)
        y = self.dec3(y,   skip2)   # (B,c2,L/2)
        y = self.dec2(y,   skip1)   # (B,c1,L)
        y = self.dec1(y)            # (B,c1,L)

        noise_hat = self.out(y)     # (B,1,L)
        return x - noise_hat
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def snr_db(clean: torch.Tensor, est: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    clean, est: (B,1,L) or (B,L)
    returns per-batch SNR in dB (tensor shape (B,))
    """
    if clean.dim() == 3:  # (B,1,L)
        clean = clean.squeeze(1)
        est   = est.squeeze(1)
    num = (clean ** 2).sum(dim=-1)
    den = ((clean - est) ** 2).sum(dim=-1) + eps
    return 10.0 * torch.log10(num / den)

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, grad_clip=1.0):
    model.train()
    loss_meter, snr_out_meter, n_total = 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)  # (B,1,L)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            y_hat = model(x)                  # denoised (B,1,L)
            loss  = loss_fn(y_hat, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            snr_batch = snr_db(y, y_hat)      # (B,)
            bsz = x.size(0)
            loss_meter   += loss.item() * bsz
            snr_out_meter+= snr_batch.mean().item() * bsz
            n_total      += bsz

    return loss_meter / n_total, snr_out_meter / n_total

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_meter, snr_out_meter, snr_in_meter, snr_gain_meter, n_total = 0.0, 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)   # noisy
        y = y.to(device, non_blocking=True)   # clean
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        snr_out = snr_db(y, y_hat)            # after denoise
        snr_in  = snr_db(y, x)                # before denoise
        snr_gain = snr_out - snr_in

        bsz = x.size(0)
        loss_meter    += loss.item() * bsz
        snr_out_meter += snr_out.mean().item() * bsz
        snr_in_meter  += snr_in.mean().item() * bsz
        snr_gain_meter+= snr_gain.mean().item() * bsz
        n_total       += bsz

    return (loss_meter / n_total,
            snr_out_meter / n_total,
            snr_in_meter / n_total,
            snr_gain_meter / n_total)
# -----------------------
# Dataset
# -----------------------
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
    ap.add_argument("--limit-train", type=int, default=200)
    ap.add_argument("--limit-val",   type=int, default=None)
    ap.add_argument("--save-dir",   type=str, default="checkpoints")
    ap.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- Resolve paths từ repo root để chạy ở đâu cũng đúng ----
    ROOT = Path(__file__).resolve().parents[1]
    data_dir = (ROOT / args.data_dir).resolve()
    train_path = data_dir / args.train_file
    val_path   = data_dir / args.val_file
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check dữ liệu
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file(s). Expected:\n  {train_path}\n  {val_path}\n"
            "Hãy chạy script tạo dữ liệu hoặc chỉnh --data-dir/--train-file/--val-file."
        )

    print(f"[INFO] Using data_dir   = {data_dir}")
    print(f"[INFO] Train file       = {train_path.name}")
    print(f"[INFO] Val file         = {val_path.name}")
    print(f"[INFO] Save dir         = {save_dir}")
    print(f"[INFO] Device           = {args.device}")

    # ---- Datasets & loaders ----
    ds_tr = NPZECGWindows(train_path, limit=args.limit_train)
    ds_va = NPZECGWindows(val_path,   limit=args.limit_val)

    # Worker/pin_memory an toàn cho CPU/WSL
    num_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    pin_mem = ("cuda" in args.device.lower())

    def _make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False
        )

    dl_tr = _make_loader(ds_tr, True)
    dl_va = _make_loader(ds_va, False)

    # ---- Model ----
    model = DenoiseCNN(in_ch=1, base_ch=16).to(args.device)
    
    # ---- Load Model for inference/resuming training
    # checkpoint = torch.load(save_dir / "last.pt", weights_only=True)
    # model.load_state_dict(checkpoint["model"])
    # epoch = checkpoint["epoch"]
    # vars = checkpoint["args"]
    
    # ---- Print params ----
    try:
        from inspect import signature  # noqa: F401
        print(f"[INFO] Params: {count_params(model):,}")
    except Exception:
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Params: {nparams:,}")

    # ---- Optimizer / loss ----
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = PeakWeightedMSE()

    # AMP chỉ khi thật sự dùng CUDA
    use_cuda_amp = ("cuda" in args.device.lower()) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    best_val_snr = -1e9
    log = []


    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_snr_out = train_one_epoch(model, dl_tr, optimizer, scaler, args.device, loss_fn)
        va_loss, va_snr_out, va_snr_in, va_snr_gain = evaluate(model, dl_va, args.device, loss_fn)

        dt = time.time() - t0
        msg = (f"Epoch {epoch:03d} | "
               f"train: loss={tr_loss:.4f}, SNR_out={tr_snr_out:.2f} dB | "
               f"val: loss={va_loss:.4f}, SNR_in={va_snr_in:.2f}, "
               f"SNR_out={va_snr_out:.2f}, gain={va_snr_gain:.2f} dB | "
               f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        print(msg)
        log.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_snr_out": tr_snr_out,
            "val_loss": va_loss,
            "val_snr_in": va_snr_in,
            "val_snr_out": va_snr_out,
            "val_snr_gain": va_snr_gain,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt
        })

        # checkpoint theo best val SNR_out
        if va_snr_out > best_val_snr:
            best_val_snr = va_snr_out
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_cuda_amp else None,
                "epoch": epoch,
                "args": vars(args),
                "best_val_snr": best_val_snr
            }
            torch.save(ckpt, save_dir / "best.pt")
            print(f"[CKPT] Saved best.pt (val SNR_out={best_val_snr:.2f} dB)")

    # save last + log
    torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, save_dir / "last.pt")
    with open(save_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("[DONE] Training finished.")

if __name__ == "__main__":
    main()