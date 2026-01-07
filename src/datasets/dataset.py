import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class NPZECGWindows(Dataset):
    """
    Expects an .npz with keys: noisy (N,L), clean (N,L).
    Returns tensors (noisy, clean) with shape (1,L), already z-scored per window
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
