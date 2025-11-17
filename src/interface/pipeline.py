from pathlib import Path
import torch, torch.nn as nn
import numpy as np, cv2
from torchvision.models.video import r3d_18, R3D_18_Weights

def build_model(ckpt_path=None, device="cpu"):
    root = Path(__file__).resolve().parents[2]
    ckpt_path = ckpt_path or (root/"checkpoints"/"best.pt")
    state = torch.load(ckpt_path, map_location=device)
    num_classes = int(state.get("num_classes", 300))
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state["model"], strict=False)
    model.to(device).eval()

    # id2gloss
    manifest = root/"data"/"wlasl_preprocessed"/"manifest_nslt2000_roi_top300.csv"
    import pandas as pd
    m = pd.read_csv(manifest)
    id2gloss = m.drop_duplicates("label").sort_values("label")["gloss"].tolist()
    return model, {"id2gloss": id2gloss}

def kinetics_normalize(frames_tchw):
    mean = np.array([0.432,0.394,0.376], dtype=np.float32).reshape(1,3,1,1)
    std  = np.array([0.228,0.221,0.223], dtype=np.float32).reshape(1,3,1,1)
    return (frames_tchw - mean) / std

def preprocess_clip(frames_rgb, T=32, stride=2, size=112):
    # center sample (or you can add a temporal start offset)
    if len(frames_rgb)==0:
        raise ValueError("No frames")
    # Convert to T frames with stride
    idxs = np.arange(0, T*stride, stride)
    idxs = idxs % len(frames_rgb)
    fs = [cv2.resize(frames_rgb[i], (size,size)) for i in idxs]  # HWC RGB
    x = np.stack(fs, axis=0).astype(np.float32)/255.0            # [T,H,W,C]
    x = np.transpose(x, (0,3,1,2))                               # [T,C,H,W]
    x = kinetics_normalize(x)                                    # normalize
    x = np.transpose(x, (1,0,2,3))                               # [C,T,H,W]
    x = np.expand_dims(x, 0)                                     # [1,C,T,H,W]
    return torch.from_numpy(x)

def decode_topk(logits, id2gloss, k=5):
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = np.argsort(-prob)[:k]
    return [(id2gloss[i], 100*prob[i]) for i in idx]
