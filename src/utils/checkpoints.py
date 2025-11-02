# src/utils/checkpoints.py
from pathlib import Path
import shutil, torch

def save_checkpoint(state: dict, is_best: bool, ckpt_dir: str, filename: str = "last.pt"):
    """
    state: {
      "epoch": int,
      "model_state": ...,
      "optim_state": ...,
      "scaler_state": ...,
      "best_metric": float
    }
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / filename
    torch.save(state, last_path)
    # also keep a rolling epoch-named copy (optional)
    if "epoch" in state:
        torch.save(state, ckpt_dir / f"epoch_{state['epoch']:04d}.pt")
    if is_best:
        best_path = ckpt_dir / "best.pt"
        if last_path.resolve() != best_path.resolve():
            shutil.copy2(last_path, best_path)
def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    """
    Returns (start_epoch, best_metric). Restores model/opt/scaler if provided.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_metric = float(ckpt.get("best_metric", -1.0))
    return start_epoch, best_metric
