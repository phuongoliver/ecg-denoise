import torch
import torch.nn as nn
from src.utils.metrics import snr_db

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, grad_clip=1.0):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The data loader for the training set.
        optimizer (Optimizer): The optimizer to use.
        scaler (GradScaler): The gradient scaler for mixed precision training.
        device (torch.device): The device to train on.
        loss_fn (callable): The loss function.
        grad_clip (float, optional): Maximum norm for gradient clipping. Defaults to 1.0.

    Returns:
        tuple: (average_loss, average_snr)
    """
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
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The data loader for the evaluation set.
        device (torch.device): The device to evaluate on.
        loss_fn (callable): The loss function.

    Returns:
        tuple: (avg_loss, avg_snr_out, avg_snr_in, avg_snr_gain)
    """
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
