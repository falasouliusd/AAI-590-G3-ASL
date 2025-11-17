
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

def _sample_indices(n, T):
    if n <= 0: return []
    if n <= T: return list(range(n))
    step = n / float(T)
    return [int(i*step) for i in range(T)]

def _read_clip_cv(path, T=16, side=112):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = _sample_indices(length, T) if length > 0 else []
    if not idxs:
        allf=[]
        while True:
            ok, fr = cap.read()
            if not ok: break
            allf.append(fr)
        cap.release()
        if not allf:
            return None
        idxs = _sample_indices(len(allf), T)
        sel = [allf[j] for j in idxs]
    else:
        sel=[]
        i=0; wanted=set(idxs)
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i in wanted: sel.append(fr)
            i+=1
        cap.release()
        if len(sel) < 1:
            return None

    out=[]
    for fr in sel:
        fr = cv2.resize(fr, (side, side), interpolation=cv2.INTER_AREA)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        fr = (fr - MEAN) / STD
        out.append(fr)
    x = np.stack(out, axis=0)         # T,H,W,C
    x = np.transpose(x, (0,3,1,2))    # T,C,H,W
    return torch.from_numpy(x)        # float32

class VideoDataset(Dataset):
    def __init__(self, dataframe, frames_t=16, side=112):
        self.df = dataframe.reset_index(drop=True)
        self.frames_t = frames_t
        self.side = side

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = r["path"]
        y = int(r["label"])
        x = _read_clip_cv(p, T=self.frames_t, side=self.side)
        if x is None:
            x = torch.zeros((self.frames_t, 3, self.side, self.side), dtype=torch.float32)
        return x, y
