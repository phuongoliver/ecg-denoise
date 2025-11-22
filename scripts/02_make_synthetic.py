#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create synthetic noisy ECG windows from MIT-BIH Arrhythmia (clean reference) and NSTDB (noise).
- Windowing: 2s @ fs=360Hz (default), stride=1s (50% overlap)
- Normalization: z-score per window (zero-mean, unit-std; eps to avoid div0)
- Noise types: baseline wander (bw), motion artifact (ma), electrode motion (em)
- Target SNR_in: {0, 5, 10} dB (configurable)
- Split by record: train/val/test to avoid leakage

Outputs:
  data/processed/
    train.npz, val.npz, test.npz
  Each .npz contains:
    noisy: (N, L), clean: (N, L), snr_in_db: (N,), noise_type: (N,), rec: (N,)
"""

import os
from pathlib import Path
import argparse
import numpy as np
import wfdb
from typing import List, Tuple, Dict

# -----------------------
# Path resolution helpers
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT  = SCRIPT_DIR.parent
DATA_DIR   = PROJ_ROOT / "data"
OUT_DIR    = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MITDB_PATH = DATA_DIR / "mitdb/physionet.org/files/mitdb/1.0.0"
NSTDB_PATH = DATA_DIR / "nstdb/physionet.org/files/nstdb/1.0.0"

# NSTDB canonical file stems (common set)
NSTDB_NOISE_FILES = {
    "bw": "118e00",   # baseline wander
    "ma": "118e06",   # muscle / motion artifact
    "em": "118e12",   # electrode motion
    # "em_alt": "118e18"  # alternative electrode motion
}

# -----------------------
# Signal utilities
# -----------------------
def read_signal(basepath: Path, rec_name: str) -> Tuple[np.ndarray, int]:
    """Read WFDB record (returns signal [T, leads] in physical units and fs)."""
    rec = wfdb.rdrecord(str(basepath / rec_name))
    fs = int(rec.fs)
    sig = rec.p_signal  # float, physical units (mV typically)
    return sig, fs

def minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = x.min()
    mx = x.max()
    rng = mx - mn
    if rng < eps:
        # cửa sổ gần hằng -> trả về 0.5 (giữa khoảng 0..1) để tránh nổ số
        return np.full_like(x, 0.5, dtype=np.float32)
    return (x - mn) / rng

def window_1d(x: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    """Return shape (N, win_len) of overlapping windows from 1D signal."""
    L = len(x)
    if L < win_len:
        return np.empty((0, win_len), dtype=x.dtype)
    starts = np.arange(0, L - win_len + 1, stride)
    out = np.stack([x[s:s+win_len] for s in starts], axis=0)
    return out

def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, eps: float = 1e-12) -> np.ndarray:
    """Scale noise to achieve target SNR_in (dB) relative to clean."""
    Ps = np.mean(clean**2)
    Pn_raw = np.mean(noise**2) + eps
    alpha = np.sqrt(Ps / (Pn_raw * (10**(snr_db/10))))
    return clean + alpha * noise

def snr_db(clean: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sum(clean**2)
    den = np.sum((clean - est)**2) + eps
    return 10.0 * np.log10(num / den)

# -----------------------
# Dataset creation
# -----------------------
def build_record_splits(all_recs: List[str],
                        train_recs: List[str] = None,
                        val_recs: List[str] = None,
                        test_recs: List[str] = None) -> Dict[str, List[str]]:
    """Create default splits if not provided: train 70%, val 15%, test 15%."""
    if train_recs and val_recs and test_recs:
        return {"train": train_recs, "val": val_recs, "test": test_recs}
    # deterministic split
    rng = np.random.default_rng(42)
    perm = rng.permutation(all_recs)
    n = len(perm)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train = perm[:n_train].tolist()
    val   = perm[n_train:n_train+n_val].tolist()
    test  = perm[n_train+n_val:].tolist()
    return {"train": train, "val": val, "test": test}

def collect_windows_from_record(rec_name: str,
                                lead_idx: int,
                                win_sec: float,
                                stride_sec: float,
                                target_fs: int = 360) -> Tuple[np.ndarray, int]:
    """
    Read one MIT-BIH record, resample if needed (MIT-BIH is 360 Hz), take one lead,
    segment into windows and z-score per window.
    Returns windows (N, L) and fs.
    """
    sig, fs = read_signal(MITDB_PATH, rec_name)
    if sig.ndim == 1:
        ecg = sig
    else:
        # choose lead
        lead_idx = min(lead_idx, sig.shape[1]-1)
        ecg = sig[:, lead_idx]

    if fs != target_fs:
        raise ValueError(f"Record {rec_name} fs={fs} != target_fs={target_fs}. Resampling not implemented in this script.")

    L = int(win_sec * fs)
    S = int(stride_sec * fs)

    raw_windows = window_1d(ecg, L, S)  # (N, L)
    if raw_windows.size == 0:
        return raw_windows, fs

    # z-score per window
    win_norm = np.stack([minmax01(w) for w in raw_windows], axis=0)
    return win_norm, fs

def prepare_noise_windows(noise_key: str,
                          win_sec: float,
                          stride_sec: float,
                          target_fs: int = 360) -> np.ndarray:
    """Read NSTDB noise record and create normalized windows."""
    rec_name = NSTDB_NOISE_FILES[noise_key]
    sig, fs = read_signal(NSTDB_PATH, rec_name)
    if sig.ndim > 1:
        n = sig[:, 0]
    else:
        n = sig
    if fs != target_fs:
        raise ValueError(f"NSTDB {rec_name} fs={fs} != target_fs={target_fs}.")

    L = int(win_sec * fs)
    S = int(stride_sec * fs)
    noise_windows = window_1d(n, L, S)  # (M, L)
    # Normalize each noise window to unify scale before SNR mixing
    noise_windows = np.stack([minmax01(w) for w in noise_windows], axis=0)
    return noise_windows

def synthesize_split(rec_list: List[str],
                     noise_pools: Dict[str, np.ndarray],
                     snr_levels: List[float],
                     lead_idx: int,
                     win_sec: float,
                     stride_sec: float,
                     target_fs: int = 360,
                     max_windows_per_record: int = None,
                     seed: int = 123) -> Dict[str, np.ndarray]:
    """
    For each record in rec_list:
      - produce normalized clean windows
      - for each SNR level and noise type, randomly sample matching noise windows
      - mix to achieve target SNR_in
    Returns dict with arrays concatenated across records.
    """
    rng = np.random.default_rng(seed)
    all_noisy, all_clean, all_snr, all_noise_type, all_rec = [], [], [], [], []

    for rec in rec_list:
        clean_wins, fs = collect_windows_from_record(rec, lead_idx, win_sec, stride_sec, target_fs)
        if clean_wins.size == 0:
            continue

        # optionally downsample number of windows per record (for speed)
        idx = np.arange(len(clean_wins))
        if max_windows_per_record and len(idx) > max_windows_per_record:
            idx = rng.choice(idx, size=max_windows_per_record, replace=False)
        clean_wins = clean_wins[idx]

        # mix across snr_levels & noise types
        for snr in snr_levels:
            for nkey, nwin_pool in noise_pools.items():
                # sample same count from noise pool
                sel = rng.choice(len(nwin_pool), size=len(clean_wins), replace=True)
                noise_sel = nwin_pool[sel]
                # mix window-wise
                mixed = np.stack([mix_with_snr(cw, nw, snr) for cw, nw in zip(clean_wins, noise_sel)], axis=0)

                all_noisy.append(mixed.astype(np.float32))
                all_clean.append(clean_wins.astype(np.float32))
                all_snr.append(np.full((len(clean_wins),), snr, dtype=np.float32))
                all_noise_type.append(np.array([nkey]*len(clean_wins)))
                all_rec.append(np.array([rec]*len(clean_wins)))

    def cat(xs, axis=0):
        return np.concatenate(xs, axis=axis) if len(xs) > 0 else np.empty((0,))

    dataset = {
        "noisy":      cat(all_noisy),
        "clean":      cat(all_clean),
        "snr_in_db":  cat(all_snr),
        "noise_type": cat(all_noise_type),
        "rec":        cat(all_rec),
        "fs":         np.array([target_fs], dtype=np.int32),
        "win_len":    np.array([int(win_sec*target_fs)], dtype=np.int32)
    }
    return dataset

def save_npz(obj: Dict[str, np.ndarray], path: Path):
    np.savez_compressed(path, **obj)
    print(f"[SAVED] {path}  shapes: noisy{obj['noisy'].shape}, clean{obj['clean'].shape}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-sec", type=float, default=2.0, help="window length (s)")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="stride (s)")
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 5.0, 10.0], help="SNR_in levels (dB)")
    parser.add_argument("--lead-idx", type=int, default=0, help="ECG lead index to use")
    parser.add_argument("--max-win-per-rec", type=int, default=1000, help="limit windows per record for speed (None for all)")
    parser.add_argument("--seed", type=int, default=123, help="rng seed")
    args = parser.parse_args()

    print("[INFO] MITDB_PATH:", MITDB_PATH)
    print("[INFO] NSTDB_PATH:", NSTDB_PATH)

    # Enumerate available MIT-BIH records (file stems with .hea)
    mit_hea = sorted([p.stem for p in MITDB_PATH.glob("*.hea") if p.stem.isdigit()])
    if len(mit_hea) == 0:
        raise RuntimeError(f"No MIT-BIH .hea files found in {MITDB_PATH}")
    print(f"[INFO] Found {len(mit_hea)} MIT-BIH records.")

    # Create record splits (deterministic)
    splits = build_record_splits(mit_hea)
    print("[INFO] Splits:", {k: len(v) for k, v in splits.items()})

    # Prepare noise windows (z-scored)
    noise_pools = {}
    for k in ["bw", "ma", "em"]:
        print(f"[INFO] Preparing NSTDB noise windows: {k}")
        noise_pools[k] = prepare_noise_windows(k, win_sec=args.win_sec, stride_sec=args.stride_sec)

    # Synthesize each split
    for split_name in ["train", "val", "test"]:
        print(f"[INFO] Synthesizing {split_name}...")
        ds = synthesize_split(
            rec_list=splits[split_name],
            noise_pools=noise_pools,
            snr_levels=args.snr_db,
            lead_idx=args.lead_idx,
            win_sec=args.win_sec,
            stride_sec=args.stride_sec,
            max_windows_per_record=args.max_win_per_rec,
            seed=args.seed
        )
        out_path = OUT_DIR / f"{split_name}.npz"
        save_npz(ds, out_path)

    print("[DONE] Synthetic dataset created in", OUT_DIR)

if __name__ == "__main__":
    main()
