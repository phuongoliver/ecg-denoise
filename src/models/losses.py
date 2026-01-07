import torch
import torch.nn as nn

class PeakWeightedMSE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B,1,L) or (B,L)
        if pred.dim() == 3: pred = pred.squeeze(1)  # (B,L)
        if target.dim() == 3: target = target.squeeze(1)

        # median per batch
        med = target.median(dim=1, keepdim=True).values  # (B,1)
        Cp = 1.0 + torch.abs(target - med)              # (B,L)

        # Weighted MSE
        mse = (pred - target) ** 2
        loss = torch.mean(mse * Cp)
        return loss
