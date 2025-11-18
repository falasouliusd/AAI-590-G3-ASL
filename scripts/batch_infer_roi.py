"""Batch inference over ROI manifest samples.

Selects the newest checkpoint in `checkpoints/`, picks up to N ROI videos from
`data/wlasl_preprocessed/manifest_nslt2000_roi_top104_balanced_clean.csv`, and
runs inference on GPU (if available). Produces a per-sample CSV and a summary JSON
in `reports/`.

Usage: python scripts/batch_infer_roi.py --n 20 --threshold 0.10
"""
import argparse
from pathlib import Path
import time
import json
import csv
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge('torch')
    _USE_DECORd = True
except Exception:
    _USE_DECORd = False
    import cv2

# ---------- Helpers (kept minimal and self-contained) ----------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class R2Plus1D18WithPermute(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
    def forward(self, x):
        # x: [B,T,C,H,W] -> permute -> [B,C,T,H,W]
        x = x.permute(0,2,1,3,4).contiguous()
        return self.backbone(x)


def read_video_frames(path, clip_len=32, resize=(112,112)):
    """Read video and return [T,H,W,C] float32 in range 0..1.
    Uses decord if available, otherwise falls back to OpenCV.
    If resize is None, preserves original frame size.
    """
    if _USE_DECORd:
        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)
        if total == 0:
            raise RuntimeError(f'No frames in {path}')
        # pick indices uniformly
        if total < clip_len:
            idx = list(range(total)) + [total-1] * (clip_len - total)
        else:
            idx = np.linspace(0, total-1, clip_len, dtype=int).tolist()
        frames_batch = vr.get_batch(idx)
        # decord may return a numpy array or a tensor depending on bridge
        if hasattr(frames_batch, 'asnumpy'):
            frames = frames_batch.asnumpy()
        elif torch.is_tensor(frames_batch):
            frames = frames_batch.cpu().numpy()
        else:
            frames = np.array(frames_batch)
        frames = frames.astype(np.float32) / 255.0
        if resize is not None:
            import cv2 as _cv2
            out = []
            for f in frames:
                out.append(_cv2.resize(f, resize))
            frames = np.stack(out, axis=0)
        return frames
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError("No frames read")
        if len(frames) < clip_len:
            while len(frames) < clip_len:
                frames.append(frames[-1])
        else:
            indices = np.linspace(0, len(frames)-1, clip_len, dtype=int)
            frames = [frames[i] for i in indices]
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return arr


def training_normalize_tensor(x: torch.Tensor):
    mean = torch.tensor((0.45,0.45,0.45), dtype=x.dtype, device=x.device)[None,:,None,None]
    std  = torch.tensor((0.225,0.225,0.225), dtype=x.dtype, device=x.device)[None,:,None,None]
    return (x - mean) / std


def frames_to_tensor(frames_np, skip_normalize: bool = False):
    t = torch.from_numpy(frames_np).permute(0,3,1,2).contiguous()  # [T,C,H,W]
    if not skip_normalize:
        t = training_normalize_tensor(t)
    t = t.unsqueeze(0)
    return t


def robust_load_state_dict(model, raw):
    state_dict = None
    if isinstance(raw, dict):
        if 'model_state' in raw:
            state_dict = raw['model_state']
        elif 'state_dict' in raw:
            state_dict = raw['state_dict']
        else:
            keys = list(raw.keys())
            if keys and isinstance(raw[keys[0]], (torch.Tensor,)):
                state_dict = raw
    else:
        state_dict = raw
    if state_dict is None:
        raise RuntimeError('Could not extract state dict')

    def try_load(sd, strict=False):
        try:
            model.load_state_dict(sd, strict=strict)
            return True
        except Exception:
            return False

    if try_load(state_dict, strict=False):
        return model

    mk = list(model.state_dict().keys())
    ck = list(state_dict.keys())

    def remap(sd, remove_prefix=None, add_prefix=None):
        out = {}
        for k,v in sd.items():
            nk = k
            if remove_prefix and nk.startswith(remove_prefix):
                nk = nk[len(remove_prefix):]
            if add_prefix:
                nk = add_prefix + nk
            out[nk] = v
        return out

    attempts = []
    if ck and ck[0].startswith('_orig_mod.'):
        attempts.append(remap(state_dict, remove_prefix='_orig_mod.'))
    if ck and ck[0].startswith('module.'):
        attempts.append(remap(state_dict, remove_prefix='module.'))
    if mk and mk[0].startswith('module.') and not (ck and ck[0].startswith('module.')):
        attempts.append(remap(state_dict, add_prefix='module.'))
    if mk and mk[0].startswith('backbone.') and not (ck and ck[0].startswith('backbone.')):
        attempts.append(remap(state_dict, add_prefix='backbone.'))
    if ck and ck[0].startswith('backbone.') and not (mk and mk[0].startswith('backbone.')):
        attempts.append(remap(state_dict, remove_prefix='backbone.'))

    for cand in attempts:
        if try_load(cand, strict=False):
            return model

    model_sd = model.state_dict()
    intersect = {k:v for k,v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    if not intersect:
        raise RuntimeError('No intersecting keys')
    model.load_state_dict(intersect, strict=False)
    return model


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--threshold', type=float, default=0.10)
    p.add_argument('--manifest', type=str, default='data/wlasl_preprocessed/manifest_nslt2000_roi_top104_balanced_clean.csv')
    p.add_argument('--ckpt-dir', type=str, default='checkpoints')
    p.add_argument('--out-dir', type=str, default='reports')
    p.add_argument('--raw', action='store_true', help='If set, pass videos into the model with NO resize and NO normalization')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    cands = sorted(list(ckpt_dir.glob('*.pt')), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        print('No checkpoints found in', ckpt_dir)
        sys.exit(2)
    ckpt_path = cands[0]
    print('Using checkpoint:', ckpt_path)

    # simple heuristic for num_classes: inspect checkpoint fc shape if available
    raw = torch.load(str(ckpt_path), map_location='cpu')
    sd = None
    if isinstance(raw, dict):
        if 'model_state' in raw:
            sd = raw['model_state']
        elif 'state_dict' in raw:
            sd = raw['state_dict']
        else:
            keys = list(raw.keys())
            if keys and isinstance(raw[keys[0]], (torch.Tensor,)):
                sd = raw
    else:
        sd = raw

    num_classes = 104
    try:
        for key in ('backbone.fc.weight','fc.weight'):
            if sd is not None and key in sd:
                num_classes = sd[key].shape[0]
                break
    except Exception:
        pass

    print('Num classes chosen:', num_classes)

    m = R2Plus1D18WithPermute(num_classes=num_classes, pretrained=True)
    try:
        robust_load_state_dict(m, raw)
    except Exception as e:
        print('Warning: robust load failed, attempting partial load:', e)
        try:
            if isinstance(raw, dict) and 'state_dict' in raw:
                m.load_state_dict({k:v for k,v in raw['state_dict'].items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
        except Exception:
            pass

    m.to(DEVICE)
    m.eval()

    # read manifest and select ROI rows
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print('Manifest not found at', manifest_path)
        sys.exit(2)
    df = pd.read_csv(manifest_path)
    if 'path' not in df.columns:
        print('Manifest missing path column')
        sys.exit(2)

    videos_root = Path('data/wlasl_preprocessed/videos_roi')
    rows = []
    for _, r in df.iterrows():
        try:
            bn = Path(str(r['path'])).name
        except Exception:
            continue
        vp = videos_root / bn
        if vp.exists():
            rows.append((vp, r))
        if len(rows) >= args.n:
            break
    if not rows:
        print('No ROI videos found in manifest under', videos_root)
        sys.exit(2)

    results = []
    for i, (vp, r) in enumerate(rows):
        try:
            # If --raw is set, do not resize nor normalize (feed raw frames in 0..1)
            if args.raw:
                frames = read_video_frames(vp, clip_len=32, resize=None)
                tensor = frames_to_tensor(frames, skip_normalize=True)
            else:
                # Default behavior: ROI videos are already preprocessed; keep skip_normalize=True
                frames = read_video_frames(vp, clip_len=32, resize=None)
                tensor = frames_to_tensor(frames, skip_normalize=True)
            tensor = tensor.to(DEVICE)
            with torch.no_grad():
                logits = m(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            top1 = int(np.argmax(probs))
            conf = float(probs[top1])
            true_label = int(r.get('label_new')) if 'label_new' in r.index and not pd.isna(r.get('label_new')) else None
            match = (true_label is not None and top1 == true_label)
            results.append({'index': i, 'video': str(vp), 'true_label': true_label, 'pred': top1, 'conf': conf, 'match': bool(match)})
            print(f"{i+1}/{len(rows)}: {vp.name} pred={top1} conf={conf:.4f} true={true_label} match={match}")
        except Exception as e:
            print('Error processing', vp, e)
            results.append({'index': i, 'video': str(vp), 'true_label': None, 'pred': None, 'conf': 0.0, 'match': False, 'error': str(e)})

    # summary
    confs = [r['conf'] for r in results if r.get('conf') is not None]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    above_thresh = float(sum(1 for c in confs if c >= args.threshold))/len(confs) if confs else 0.0
    acc = float(sum(1 for r in results if r.get('match')))/len(results)

    timestamp = time.strftime('%Y%m%dT%H%M%S')
    base = f"batch_roi_{ckpt_path.stem}_{timestamp}"
    csv_path = out_dir / (base + '.csv')
    json_path = out_dir / (base + '.json')

    # write CSV
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['index','video','true_label','pred','conf','match','error'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in ['index','video','true_label','pred','conf','match','error']})

    summary = {'ckpt': str(ckpt_path), 'n': len(results), 'avg_conf': avg_conf, 'above_threshold_fraction': above_thresh, 'accuracy': acc, 'csv': str(csv_path)}
    with open(json_path, 'w') as jf:
        json.dump({'summary': summary, 'results': results}, jf, indent=2)

    print('\nSummary:')
    print(json.dumps(summary, indent=2))
    print('Saved CSV:', csv_path)
    print('Saved JSON:', json_path)

if __name__ == '__main__':
    main()
