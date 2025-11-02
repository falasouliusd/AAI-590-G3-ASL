# src/data/wlasl_ds.py
import torch, numpy as np, cv2, decord
from torch.utils.data import Dataset
decord.bridge.set_bridge('torch')

def _resize_112(frame_tchw: torch.Tensor) -> torch.Tensor:
    T,C,H,W = frame_tchw.shape
    arr = frame_tchw.permute(0,2,3,1).cpu().numpy()
    out = np.empty((T,112,112,C), dtype=np.float32)
    for t in range(T):
        out[t] = cv2.resize(arr[t], (112,112), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(out).permute(0,3,1,2)

def _normalize(frame_tchw, mean=(0.45,)*3, std=(0.225,)*3):
    mean = torch.tensor(mean, dtype=frame_tchw.dtype, device=frame_tchw.device)[None,:,None,None]
    std  = torch.tensor(std,  dtype=frame_tchw.dtype, device=frame_tchw.device)[None,:,None,None]
    return (frame_tchw - mean) / std

def uniform_temporal_indices(n_total, clip_len, stride):
    if n_total <= 0: return [0]*clip_len
    wanted = (clip_len-1)*stride + 1
    if n_total >= wanted:
        start = (n_total - wanted)//2
        return [start + i*stride for i in range(clip_len)]
    idxs = [min(i*stride, n_total-1) for i in range(clip_len)]
    return idxs

class WLASLDataset(Dataset):
    def __init__(self, df, clip_len=32, stride=2, train=False):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.stride = stride
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row["path"]
        label = int(row["label"])
        vr = decord.VideoReader(path)
        n = len(vr)
        idxs = uniform_temporal_indices(n, self.clip_len, self.stride)
        batch = vr.get_batch(idxs)            # [T,H,W,C] uint8
        x = batch.float()/255.0
        x = x.permute(0,3,1,2)                # [T,C,H,W]
        x = _resize_112(x)
        x = _normalize(x)
        return x, label, path
