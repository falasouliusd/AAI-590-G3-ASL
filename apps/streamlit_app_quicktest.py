"""
Streamlit sanity-check app for ASL inference with the VideoResNet baseline preset.

Usage:
    pip install -r requirements.txt
    streamlit run apps/streamlit_app_quicktest.py

This lightweight view keeps the same functionality as the main demo but preselects
the best VideoResNet checkpoint and adds a one-click sanity test on a known clip.
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
import os
import io
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    r2plus1d_18,
    R2Plus1D_18_Weights,
    r3d_18,
    R3D_18_Weights,
)

import cv2

# ---------------------- Utils & model wrapper ----------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class R2Plus1D18WithPermute(nn.Module):
    """Wrap r2plus1d_18 to accept [B, T, C, H, W] and permute internally."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        # permute to [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


class VideoResNetR3D18WithPermute(nn.Module):
    """Wrap r3d_18 VideoResNet to accept [B, T, C, H, W]."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r3d_18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


MODEL_FACTORIES = {
    "VideoResNet (r3d_18)": VideoResNetR3D18WithPermute,
    "R(2+1)D-18": R2Plus1D18WithPermute,
}


def kinetics_normalize_tensor(x: torch.Tensor):
    # x: [T, C, H, W] or [B, T, C, H, W]; values in [0,1]
    mean = torch.tensor((0.432, 0.394, 0.376), dtype=x.dtype, device=x.device)[None,:,None,None]
    std  = torch.tensor((0.228, 0.221, 0.223), dtype=x.dtype, device=x.device)[None,:,None,None]
    return (x - mean) / std


def training_normalize_tensor(x: torch.Tensor):
    # The training data pipeline (src/data/wlasl_ds.py) used mean=0.45 and std=0.225
    # for all channels. Provide this normalization for inference parity.
    mean = torch.tensor((0.45, 0.45, 0.45), dtype=x.dtype, device=x.device)[None,:,None,None]
    std  = torch.tensor((0.225, 0.225, 0.225), dtype=x.dtype, device=x.device)[None,:,None,None]
    return (x - mean) / std


def _temporal_indices(num_frames: int, clip_len: int, stride: int) -> list:
    """Mirror training-time sampling (contiguous clip with stride)."""
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")
    stride = max(1, int(stride))
    if num_frames <= 0:
        return [0] * clip_len
    wanted = (clip_len - 1) * stride + 1
    if num_frames >= wanted:
        start = (num_frames - wanted) // 2
        return [start + i * stride for i in range(clip_len)]
    last = num_frames - 1
    return [min(i * stride, last) for i in range(clip_len)]


def read_video_frames_opencv(path, clip_len=32, resize=(112, 112), stride=2):
    """Read video, return numpy array of shape [T, H, W, C] in RGB, values 0..1"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()

    num_frames = len(frames)
    if num_frames == 0:
        raise RuntimeError("No frames read from video")

    indices = _temporal_indices(num_frames, clip_len, stride)
    frames = [frames[i] for i in indices]

    arr = np.stack(frames, axis=0)  # [T, H, W, C]
    arr = arr.astype(np.float32) / 255.0
    return arr


def frames_to_tensor(frames_np, skip_normalize: bool = False):
    """Convert frames [T,H,W,C] -> tensor [1,T,C,H,W].
    If skip_normalize is True the function will not apply any normalization
    (useful when the clip is already preprocessed / ROI crops).
    """
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
    if not skip_normalize:
        if use_training_norm:
            t = training_normalize_tensor(t)
        else:
            t = kinetics_normalize_tensor(t)
    t = t.unsqueeze(0)  # [1, T, C, H, W]
    return t


def robust_load_checkpoint(model, ckpt_path):
    raw = torch.load(str(ckpt_path), map_location=DEVICE)
    state_dict = None
    if isinstance(raw, dict):
        if 'model_state' in raw:
            state_dict = raw['model_state']
        elif 'state_dict' in raw:
            state_dict = raw['state_dict']
        else:
            # maybe it's a plain state_dict
            keys = list(raw.keys())
            if keys and isinstance(raw[keys[0]], (torch.Tensor,)):
                state_dict = raw
    else:
        state_dict = raw

    if state_dict is None:
        raise RuntimeError(f'Could not extract model parameters from {ckpt_path}')

    def try_load(sd, strict=True):
        try:
            result = model.load_state_dict(sd, strict=strict)
            if strict:
                return True
            missing = getattr(result, 'missing_keys', [])
            unexpected = getattr(result, 'unexpected_keys', [])
            return not missing and not unexpected
        except Exception:
            return False

    if try_load(state_dict, strict=True):
        return model

    # attempt simple remaps
    mk = list(model.state_dict().keys())
    ck = list(state_dict.keys())

    def remap(sd, remove_prefix=None, add_prefix=None):
        out = {}
        for k, v in sd.items():
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
    # Some saved checkpoints use top-level keys (e.g. 'stem.*', 'fc.*') while
    # our runtime wrapper nests the torchvision model under `backbone` which
    # causes a mismatch like checkpoint having 'fc.weight' vs model expecting
    # 'backbone.fc.weight'. Try adding/removing a 'backbone.' prefix when
    # appropriate.
    if mk and mk[0].startswith('backbone.') and not (ck and ck[0].startswith('backbone.')):
        attempts.append(remap(state_dict, add_prefix='backbone.'))
    if ck and ck[0].startswith('backbone.') and not (mk and mk[0].startswith('backbone.')):
        attempts.append(remap(state_dict, remove_prefix='backbone.'))

    for cand in attempts:
        if try_load(cand, strict=False):
            return model

    # partial intersection
    model_sd = model.state_dict()
    intersect = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    if not intersect:
        raise RuntimeError('No intersecting keys between checkpoint and model.')
    model.load_state_dict(intersect, strict=False)
    return model


def predict_tensor(model, tensor):
    """tensor: [B, T, C, H, W] (common dataset format). Returns predicted label index and softmax scores."""
    model.eval()
    model.to(DEVICE)
    t = tensor.to(DEVICE)

    logits = None
    # try as-is first (many model wrappers will permute internally)
    for attempt in (0, 1):
        inp = t if attempt == 0 else t.permute(0, 2, 1, 3, 4).contiguous()
        try:
            with torch.no_grad():
                out = model(inp)
                logits = out
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if attempt == 0 and ("expected input" in msg or "channels" in msg or "but got" in msg or "shape" in msg):
                continue
            raise

    if logits is None:
        raise RuntimeError('Model forward failed on both original and permuted inputs.')

    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title='ASL Translator', layout='wide')

st.title('ASL Video Translator — upload or simple webcam')

sidebar = st.sidebar
sidebar.header('Settings')

# checkpoint selection
ckpt_dir = Path('checkpoints')
ckpt_dir.mkdir(exist_ok=True)
available_ckpts = sorted({p.name for pattern in ('*.pt', '*.pth') for p in ckpt_dir.glob(pattern)})

preferred_ckpt = '06_train_baseline_3dCNN-r3d18_k400_kenetics_best_cnn_Kinetics-400s.pt'
ckpt_options = ['-- none --'] + available_ckpts
default_ckpt_idx = ckpt_options.index(preferred_ckpt) if preferred_ckpt in ckpt_options else 0
ckpt_choice = sidebar.selectbox('Select checkpoint from `checkpoints/`', options=ckpt_options, index=default_ckpt_idx)
uploaded_ckpt = sidebar.file_uploader('Or upload a checkpoint (.pt/.pth)', type=['pt', 'pth'])

# optional manifest to map label->gloss
manifest_upload = sidebar.file_uploader('Optional: upload manifest CSV to map labels to gloss', type=['csv'])
label_map = None
manifest_df = None
if manifest_upload is not None:
    try:
        df_map = pd.read_csv(manifest_upload)
        # keep the full dataframe for path lookups
        manifest_df = df_map
        if 'label_new' in df_map.columns and 'gloss' in df_map.columns:
            label_map = df_map.groupby('label_new')['gloss'].first().to_dict()
    except Exception as e:
        st.sidebar.error('Could not read manifest CSV: ' + str(e))

# confidence threshold and top-k
conf_threshold = sidebar.slider('Confidence threshold (top-1) — below this shows "No confident translation"', min_value=0.0, max_value=1.0, value=0.10, step=0.01)
top_k = sidebar.number_input('Show top-k predictions', min_value=1, max_value=10, value=3)

# normalization / ROI handling options
use_training_norm = sidebar.checkbox('Use training normalization (mean=0.45,std=0.225)', value=True)
skip_roi_preproc = sidebar.checkbox('Skip resize/normalize for ROI clips (only if they have already been preprocessed)', value=False)

# canonical ROI videos directory used for skip-detection
videos_dir = Path('data/wlasl_preprocessed/videos_roi')
DEFAULT_MANIFEST_PATH = Path('data/wlasl_preprocessed/manifest_nslt2000_roi_top104_balanced_clean.csv')
SANITY_SAMPLE_PATH = videos_dir / '69219.mp4'
SANITY_SAMPLE_LABEL = 'bad'


def ensure_manifest_df_loaded():
    """Load the default manifest once so downstream lookups work."""
    global manifest_df, label_map
    if manifest_df is None and DEFAULT_MANIFEST_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_MANIFEST_PATH)
            manifest_df = df
            if label_map is None and 'label_new' in df.columns and 'gloss' in df.columns:
                label_map = df.groupby('label_new')['gloss'].first().to_dict()
        except Exception:
            pass


def lookup_manifest_entry(video_basename: str):
    """Return the manifest row (Series) matching a video basename, if available."""
    ensure_manifest_df_loaded()
    if manifest_df is None or 'path' not in manifest_df.columns:
        return None
    try:
        matches = manifest_df[manifest_df['path'].apply(lambda x: Path(str(x)).name == video_basename)]
        if matches.empty:
            matches = manifest_df[manifest_df['path'].apply(lambda x: str(x).endswith(video_basename))]
        if matches.empty:
            return None
        return matches.iloc[0]
    except Exception:
        return None


def is_path_in_roi_folder(path_like) -> bool:
    """Best-effort check whether a path resides in the ROI directory."""
    try:
        base = videos_dir.resolve()
        candidate = Path(path_like).resolve()
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            return False
    except Exception:
        try:
            return str(Path(path_like)).startswith(str(videos_dir))
        except Exception:
            return False


def manifest_entry_details(entry):
    """Extract useful manifest metadata from a pandas Series."""
    if entry is None:
        return False, None, None, None
    try:
        row_path = entry.get('path')
        if row_path is not None and pd.isna(row_path):
            row_path = None
        label_val = entry.get('label_new') if 'label_new' in entry.index else None
        label = int(label_val) if label_val is not None and pd.notna(label_val) else None
        gloss = entry.get('gloss') if 'gloss' in entry.index else None
        if gloss is not None and pd.isna(gloss):
            gloss = None
        return True, row_path, label, gloss
    except Exception:
        return False, None, None, None


def auto_load_manifest():
    """Try to auto-detect a manifest CSV in the repository data folder that contains label_new and gloss."""
    candidates = list(Path('data').rglob('*.csv')) if Path('data').exists() else []
    for p in candidates:
        try:
            dfc = pd.read_csv(p)
            if 'label_new' in dfc.columns and 'gloss' in dfc.columns:
                # return label_map dict and the path to the df so caller can reload full df if needed
                return dfc.groupby('label_new')['gloss'].first().to_dict(), p
        except Exception:
            continue
    return None, None

if label_map is None:
    lm, lm_path = auto_load_manifest()
    if lm is not None:
        label_map = lm
        try:
            manifest_df = pd.read_csv(lm_path)
        except Exception:
            manifest_df = None
        st.sidebar.markdown(f"Detected manifest: `{lm_path}` — mapping labels to gloss")

st.write('Model device:', DEVICE)

mode = st.radio('Mode', options=['Upload video', 'Live (simple snapshots)'])

# model loader state (persist model across reruns using session_state)
if 'model_obj' not in st.session_state:
    st.session_state['model_obj'] = None
    st.session_state['ckpt_loaded'] = None
if 'model_arch' not in st.session_state:
    st.session_state['model_arch'] = None

model_placeholder = st.empty()
num_classes = sidebar.number_input('Num classes (model output size)', min_value=2, value=104)
pretrained_flag = sidebar.checkbox('Assume Kinetics pretrained backbone (recommended)', value=True)
model_arch_options = list(MODEL_FACTORIES.keys())
default_arch_index = model_arch_options.index("VideoResNet (r3d_18)") if "VideoResNet (r3d_18)" in model_arch_options else 0
model_arch_choice = sidebar.selectbox('Model architecture', options=model_arch_options, index=default_arch_index)
if st.session_state.get('model_arch') and st.session_state['model_arch'] != model_arch_choice:
    st.session_state['model_obj'] = None
    st.session_state['ckpt_loaded'] = None

def _load_ckpt_and_store(ckpt_path, arch_choice):
    try:
        factory = MODEL_FACTORIES.get(arch_choice)
        if factory is None:
            raise ValueError(f'Unknown architecture selection: {arch_choice}')
        m = factory(num_classes=num_classes, pretrained=pretrained_flag)
        if ckpt_path is None:
            st.session_state['model_obj'] = m
            st.session_state['ckpt_loaded'] = None
            st.session_state['model_arch'] = arch_choice
            model_placeholder.info('Model created with random weights')
            # sanity-check if manifest present
            try:
                if label_map is not None:
                    max_label = max(int(k) for k in label_map.keys())
                    expected = max_label + 1
                    out = None
                    if hasattr(m, 'backbone') and hasattr(m.backbone, 'fc'):
                        out = getattr(m.backbone.fc, 'out_features', None)
                    if out is None and hasattr(m, 'fc'):
                        out = getattr(m.fc, 'out_features', None)
                    if out is not None and out != expected:
                        st.warning(f'Model output size ({out}) does not match manifest max label+1 ({expected}).')
            except Exception:
                pass
            return
        robust_load_checkpoint(m, ckpt_path)
        # attempt to compute a small diagnostics value: L2 difference between
        # checkpoint fc weights (if present) and the model's loaded fc weights.
        try:
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
            ck_fc = None
            for key in ('backbone.fc.weight', 'fc.weight'):
                if sd is not None and key in sd:
                    ck_fc = sd[key]
                    ck_fc_name = key
                    break
            mdl_sd = m.state_dict()
            mdl_fc = None
            for key in ('backbone.fc.weight','fc.weight'):
                if key in mdl_sd:
                    mdl_fc = mdl_sd[key].cpu()
                    mdl_fc_name = key
                    break
            if ck_fc is not None and mdl_fc is not None and tuple(ck_fc.shape) == tuple(mdl_fc.shape):
                diff = float((mdl_fc - ck_fc.cpu()).norm().item())
                model_placeholder.info(f'Loaded checkpoint: {Path(ckpt_path).name} — fc L2 diff={diff:.4f} ({ck_fc_name})')
                st.session_state['last_model_load_info'] = {'fc_l2_diff': diff, 'ck_fc_name': ck_fc_name, 'mdl_fc_name': mdl_fc_name}
            else:
                model_placeholder.info(f'Loaded checkpoint: {Path(ckpt_path).name} — fc comparison not available')
        except Exception:
            # non-fatal diagnostics
            pass
        st.session_state['model_obj'] = m
        st.session_state['ckpt_loaded'] = str(ckpt_path)
        st.session_state['model_arch'] = arch_choice
        model_placeholder.success(f'Loaded checkpoint: {Path(ckpt_path).name}')
        # sanity-check if manifest present
        try:
            if label_map is not None:
                max_label = max(int(k) for k in label_map.keys())
                expected = max_label + 1
                out = None
                if hasattr(m, 'backbone') and hasattr(m.backbone, 'fc'):
                    out = getattr(m.backbone.fc, 'out_features', None)
                if out is None and hasattr(m, 'fc'):
                    out = getattr(m.fc, 'out_features', None)
                if out is not None and out != expected:
                    st.warning(f'Model output size ({out}) does not match manifest max label+1 ({expected}).')
        except Exception:
            pass
    except Exception as e:
        st.session_state['model_obj'] = None
        st.session_state['ckpt_loaded'] = None
        st.exception(e)


def run_inference_on_path(video_path, describe='clip'):
    """Shared helper to run inference on a given video path."""
    m = st.session_state.get('model_obj')
    if m is None:
        st.warning('Model not loaded — please load a checkpoint (or create one).')
        return
    vp = Path(video_path)
    if not vp.exists():
        st.error(f'Video path does not exist: {vp}')
        return
    try:
        manifest_entry = lookup_manifest_entry(vp.name)
        in_manifest, manifest_row_path, manifest_label, true_gloss = manifest_entry_details(manifest_entry)
        with st.spinner(f'Reading and preprocessing {describe}...'):
            is_roi = False
            if skip_roi_preproc:
                is_roi = is_path_in_roi_folder(vp)
                if (not is_roi) and in_manifest and manifest_row_path:
                    try:
                        manifest_candidate = Path(str(manifest_row_path))
                        if manifest_candidate.exists() and is_path_in_roi_folder(manifest_candidate):
                            is_roi = True
                    except Exception:
                        pass
            read_resize = None if (skip_roi_preproc and is_roi) else (112,112)
            frames = read_video_frames_opencv(vp, clip_len=32, resize=read_resize, stride=2)
            tensor = frames_to_tensor(frames, skip_normalize=(skip_roi_preproc and is_roi))
        with st.spinner('Running inference...'):
            pred, conf, probs = predict_tensor(m, tensor)
        probs_arr = np.array(probs)
        topk_idx = probs_arr.argsort()[::-1][:top_k]
        topk = [(int(i), float(probs_arr[i])) for i in topk_idx]
        lines = []
        for i, pscore in topk:
            name = label_map.get(i) if label_map is not None else str(i)
            lines.append({'label': int(i), 'gloss': name, 'confidence': float(pscore)})
        st.session_state['last_debug'] = {
            'checkpoint': st.session_state.get('ckpt_loaded'),
            'clip_path': str(vp),
            'tensor_shape': tuple(tensor.shape),
            'tensor_min': float(tensor.min()),
            'tensor_max': float(tensor.max()),
            'topk': lines,
            'probs': probs_arr.tolist(),
            'in_manifest': in_manifest,
            'manifest_row_path': str(manifest_row_path) if manifest_row_path is not None else None,
            'manifest_label': int(manifest_label) if manifest_label is not None else None,
            'true_gloss': true_gloss
        }
        display_obj = {'top_k': lines}
        if true_gloss is not None:
            display_obj['true_gloss'] = true_gloss
            display_obj['in_manifest'] = in_manifest
        if float(topk[0][1]) < conf_threshold:
            result_box.warning('No confident translation found (top-1 confidence below threshold).')
            result_box.json(display_obj)
        else:
            result_box.success(f"Prediction: {lines[0]['gloss']} (label={lines[0]['label']}) — confidence {lines[0]['confidence']:.2f}")
            result_box.json(display_obj)
    except Exception as e:
        st.error('Error during inference: ' + str(e))

# Manual load button (still available)
if st.button('Load selected checkpoint'):
    try:
        if uploaded_ckpt is not None:
            tpath = Path(tempfile.gettempdir()) / f"uploaded_ckpt_{int(time.time())}.pt"
            with open(tpath, 'wb') as f:
                f.write(uploaded_ckpt.getbuffer())
            ckpt_path = tpath
        elif ckpt_choice != '-- none --':
            ckpt_path = ckpt_dir / ckpt_choice
        else:
            ckpt_path = None

        with st.spinner('Loading checkpoint...'):
            _load_ckpt_and_store(ckpt_path, model_arch_choice)
    except Exception as e:
        st.exception(e)

# Auto-load selected checkpoint if none is loaded in this session
if st.session_state.get('model_obj') is None:
    auto_ckpt = None
    if uploaded_ckpt is not None:
        tpath = Path(tempfile.gettempdir()) / f"uploaded_ckpt_{int(time.time())}.pt"
        with open(tpath, 'wb') as f:
            f.write(uploaded_ckpt.getbuffer())
        auto_ckpt = tpath
    elif ckpt_choice != '-- none --':
        auto_ckpt = ckpt_dir / ckpt_choice

    if auto_ckpt is not None:
        try:
            with st.spinner(f'Auto-loading checkpoint {Path(auto_ckpt).name} ...'):
                _load_ckpt_and_store(auto_ckpt, model_arch_choice)
        except Exception:
            # _load_ckpt_and_store already reports exceptions
            pass

# prediction area
result_box = st.empty()

if mode == 'Upload video':
    uploaded = st.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'mkv'])
    # Sample selector: let the user pick a known sample from the manifest (videos under data/wlasl_preprocessed/videos_roi)
    try:
        ensure_manifest_df_loaded()
        sample_chosen = None
        sample_candidates = []
        if manifest_df is not None and 'path' in manifest_df.columns and videos_dir.exists():
            # build candidates where the basename of manifest path exists in videos_dir
            for _, r in manifest_df.iterrows():
                try:
                    bn = Path(str(r['path'])).name
                except Exception:
                    continue
                vp = videos_dir / bn
                if vp.exists():
                    gloss = r.get('gloss') if 'gloss' in r.index else ''
                    lbl = int(r.get('label_new')) if 'label_new' in r.index and not pd.isna(r.get('label_new')) else None
                    sample_candidates.append((str(vp), gloss, lbl))
        if sample_candidates:
            options = [f"{i}: {c[1]} — {Path(c[0]).name}" for i, c in enumerate(sample_candidates)]
            sel = st.selectbox('Or pick a sample video from the manifest (existing files)', options=['-- none --'] + options)
            if sel != '-- none --':
                sel_idx = int(sel.split(':', 1)[0])
                sample_chosen = sample_candidates[sel_idx][0]
                st.write('Selected sample:', sample_chosen)
                if st.button('Use selected sample as uploaded video'):
                    st.session_state['selected_sample_path'] = sample_chosen
    except Exception:
        # non-fatal: sample selector is an optional convenience
        pass
    if SANITY_SAMPLE_PATH.exists():
        if st.button(f"Run built-in sanity check (gloss '{SANITY_SAMPLE_LABEL}')", key='btn_sanity_sample'):
            run_inference_on_path(SANITY_SAMPLE_PATH, f"sanity sample ({SANITY_SAMPLE_LABEL})")
    if uploaded is not None:
        tmp_path = Path(tempfile.gettempdir()) / f"uploaded_video_{int(time.time())}.{uploaded.name.split('.')[-1]}"
        with open(tmp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        st.info(f'File saved to {tmp_path}')

        if st.button('Process uploaded video'):
            run_inference_on_path(tmp_path, 'uploaded video')

    # If a manifest-selected sample is present in session_state, allow processing it
    if st.session_state.get('selected_sample_path') is not None and Path(st.session_state.get('selected_sample_path')).exists():
        if st.button('Process selected sample'):
            sample_path = Path(st.session_state.get('selected_sample_path'))
            run_inference_on_path(sample_path, 'selected sample')

else:
    st.write('Live (simple snapshots) mode: use the camera to capture multiple frames and then process as a clip.')
    st.write('This is a simple capture-based approach. For continuous live streaming, consider installing `streamlit-webrtc` and I can adapt the app.')

    if 'snapshot_frames' not in st.session_state:
        st.session_state['snapshot_frames'] = []

    cam_file = st.camera_input('Capture a frame')
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Add captured frame'):
            if cam_file is None:
                st.warning('No camera snapshot available. Take a snapshot first.')
            else:
                # read file bytes, decode with OpenCV
                bytes_data = cam_file.getvalue()
                arr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    st.error('Could not decode camera image')
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (112,112))
                    st.session_state['snapshot_frames'].append(img)
                    st.info(f'Captured frames: {len(st.session_state["snapshot_frames"]) }')
    with col2:
        if st.button('Clear captured frames'):
            st.session_state['snapshot_frames'] = []
            st.info('Cleared frames')
    with col3:
        if st.button('Process captured clip'):
            m = st.session_state.get('model_obj')
            if m is None:
                st.warning('Model not loaded — please load a checkpoint (or create one).')
            elif len(st.session_state['snapshot_frames']) == 0:
                st.warning('No captured frames — capture some images first')
            else:
                try:
                    frames = list(st.session_state['snapshot_frames'])
                    # pad/repeat to clip_len=32
                    clip_len = 32
                    while len(frames) < clip_len:
                        frames.append(frames[-1])
                    # sample evenly if more than clip_len
                    if len(frames) > clip_len:
                        indices = np.linspace(0, len(frames)-1, clip_len, dtype=int)
                        frames = [frames[i] for i in indices]

                    arr = np.stack(frames, axis=0).astype(np.float32)/255.0
                    tensor = frames_to_tensor(arr)
                    with st.spinner('Running inference on captured clip...'):
                        pred, conf, probs = predict_tensor(m, tensor)

                    probs_arr = np.array(probs)
                    topk_idx = probs_arr.argsort()[::-1][:top_k]
                    topk = [(int(i), float(probs_arr[i])) for i in topk_idx]
                    lines = []
                    for i, pscore in topk:
                        name = label_map.get(i) if label_map is not None else str(i)
                        lines.append({'label': int(i), 'gloss': name, 'confidence': float(pscore)})

                    # store debug info for expander (captured clip)
                    st.session_state['last_debug'] = {
                        'checkpoint': st.session_state.get('ckpt_loaded'),
                        'clip_path': 'captured_frames',
                        'tensor_shape': tuple(tensor.shape),
                        'tensor_min': float(tensor.min()),
                        'tensor_max': float(tensor.max()),
                        'topk': lines,
                        'probs': probs_arr.tolist()
                    }

                    if float(topk[0][1]) < conf_threshold:
                        result_box.warning('No confident translation found (top-1 confidence below threshold).')
                        result_box.json({'top_k': lines})
                    else:
                        result_box.success(f"Prediction: {lines[0]['gloss']} (label={lines[0]['label']}) — confidence {lines[0]['confidence']:.2f}")
                        result_box.json({'top_k': lines})
                except Exception as e:
                    st.error('Error during inference: ' + str(e))

# Footer / hints
st.markdown('---')
st.write('Notes:')
st.write('- This app uses a simple sampling strategy to build a fixed-length clip (32 frames) for the model.')
st.write('- For better results, provide videos where the signer occupies most of the frame and use the same preprocessing used during training (frame size 112x112, Kinetics normalization).')
st.write('- For a continuous real-time demo, I can adapt this app to use `streamlit-webrtc` which supports real webcam streams.')

# update todo list status
st.caption('App generated by helper — see apps/streamlit_app.py')

# Debug expander: show last processed clip debug info if available
if 'last_debug' in st.session_state:
    dbg = st.session_state['last_debug']
    with st.expander('Debug: last processed clip (show/hide)', expanded=False):
        st.write('Checkpoint loaded:', dbg.get('checkpoint'))
        st.write('Clip path:', dbg.get('clip_path'))
        st.write('Tensor shape:', dbg.get('tensor_shape'))
        st.write('Tensor min/max:', dbg.get('tensor_min'), dbg.get('tensor_max'))

        # show sample frames if available (only for uploaded videos we saved tmp_path)
        try:
            if dbg.get('clip_path') and dbg.get('clip_path') != 'captured_frames' and Path(dbg.get('clip_path')).exists():
                sample_frames = read_video_frames_opencv(dbg.get('clip_path'), clip_len=8, resize=(112,112))
                # convert to list of images [H,W,C] scaled 0..255
                imgs = (sample_frames * 255).astype('uint8')
                st.write('Sample frames:')
                st.image([imgs[i] for i in range(min(8, imgs.shape[0]))], width=140)
        except Exception:
            pass

        # show top-k table
        try:
            df_dbg = pd.DataFrame(dbg.get('topk'))
            st.write('Top-k predictions:')
            st.table(df_dbg)
        except Exception:
            st.write('No top-k info available')
        # show manifest match info if present
        try:
            if 'in_manifest' in dbg:
                st.write('Manifest match: ', dbg.get('in_manifest'))
                if dbg.get('in_manifest'):
                    st.write('Manifest row path:', dbg.get('manifest_row_path'))
                    st.write('Manifest label:', dbg.get('manifest_label'))
                    st.write('True gloss from manifest:', dbg.get('true_gloss'))
        except Exception:
            pass
