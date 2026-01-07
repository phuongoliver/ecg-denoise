import random
import numpy as np
import torch

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # Optional, can be risky if no CUDA
    if torch.cuda.is_available():
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
