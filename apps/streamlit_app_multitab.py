"""
Two-tab Streamlit interface for ASL recognition demos.
- Tab 1: Quick preset demo using the best VideoResNet checkpoint
- Tab 2: Advanced customization for experimenting with other checkpoints/models
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
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

# ---------------------- Model wrappers & helpers ----------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class R2Plus1D18WithPermute(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.backbone(x)


class VideoResNetR3D18WithPermute(nn.Module):
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
    mean = torch.tensor((0.432, 0.394, 0.376), dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor((0.228, 0.221, 0.223), dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / std


def training_normalize_tensor(x: torch.Tensor):
    mean = torch.tensor((0.45, 0.45, 0.45), dtype=x.dtype, device=x.device)[None, :, None, None]
    std = torch.tensor((0.225, 0.225, 0.225), dtype=x.dtype, device=x.device)[None, :, None, None]
    return (x - mean) / std


def _temporal_indices(num_frames: int, clip_len: int, stride: int) -> list:
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
    num_frames = len(frames)
    if num_frames == 0:
        raise RuntimeError("No frames read from video")
    indices = _temporal_indices(num_frames, clip_len, stride)
    frames = [frames[i] for i in indices]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return arr


def frames_to_tensor(frames_np, skip_normalize: bool, norm_mode: str):
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
    if not skip_normalize:
        if norm_mode == 'training':
            t = training_normalize_tensor(t)
        else:
            t = kinetics_normalize_tensor(t)
    t = t.unsqueeze(0)
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
    if mk and mk[0].startswith('backbone.') and not (ck and ck[0].startswith('backbone.')):
        attempts.append(remap(state_dict, add_prefix='backbone.'))
    if ck and ck[0].startswith('backbone.') and not (mk and mk[0].startswith('backbone.')):
        attempts.append(remap(state_dict, remove_prefix='backbone.'))

    for cand in attempts:
        if try_load(cand, strict=True):
            return model

    model_sd = model.state_dict()
    intersect = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    if not intersect:
        raise RuntimeError('No intersecting keys between checkpoint and model.')
    model.load_state_dict(intersect, strict=False)
    return model


def predict_tensor(model, tensor):
    model.eval()
    model.to(DEVICE)
    t = tensor.to(DEVICE)
    logits = None
    for attempt in (0, 1):
        inp = t if attempt == 0 else t.permute(0, 2, 1, 3, 4).contiguous()
        try:
            with torch.no_grad():
                logits = model(inp)
            break
        except RuntimeError as e:
            if attempt == 0 and any(msg in str(e).lower() for msg in ('expected input', 'channels', 'shape', 'but got')):
                continue
            raise
    if logits is None:
        raise RuntimeError('Model forward failed on both original and permuted inputs.')
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf, probs


# ---------------------- Global data & manifest helpers ----------------------

CKPT_DIR = Path('checkpoints')
CKPT_DIR.mkdir(exist_ok=True)
DEFAULT_MANIFEST_PATH = Path('data/wlasl_preprocessed/manifest_nslt2000_roi_top104_balanced_clean.csv')
VIDEOS_DIR = Path('data/wlasl_preprocessed/videos_roi')
PER_CLASS_ACC_PATH = Path('reports/06_train_baseline_3dCNN-r3d18_k400_kenetics_per_class_accuracy.csv')
BEST_CKPT_NAME = '06_train_baseline_3dCNN-r3d18_k400_kenetics_best_cnn_Kinetics-400s.pt'
BEST_CKPT_PATH = CKPT_DIR / BEST_CKPT_NAME
BEST_ARCH = 'VideoResNet (r3d_18)'
BEST_NUM_CLASSES = 104
BEST_SANITY_CLIP = VIDEOS_DIR / '69219.mp4'
BEST_SANITY_GLOSS = 'bad'

manifest_df = None
label_map = None


def ensure_manifest_df_loaded():
    global manifest_df, label_map
    if manifest_df is None and DEFAULT_MANIFEST_PATH.exists():
        try:
            manifest_df = pd.read_csv(DEFAULT_MANIFEST_PATH)
        except Exception:
            manifest_df = None
    if label_map is None and manifest_df is not None and 'label_new' in manifest_df.columns and 'gloss' in manifest_df.columns:
        label_map = manifest_df.groupby('label_new')['gloss'].first().to_dict()


def load_manifest_from_upload(upload):
    global manifest_df, label_map
    try:
        df = pd.read_csv(upload)
        manifest_df = df
        if 'label_new' in df.columns and 'gloss' in df.columns:
            label_map = df.groupby('label_new')['gloss'].first().to_dict()
        return True
    except Exception as e:
        st.error(f'Failed to read manifest: {e}')
        return False


def auto_load_manifest_from_data_folder():
    global manifest_df, label_map
    if manifest_df is not None:
        return
    data_path = Path('data')
    if not data_path.exists():
        return
    for csv_path in data_path.rglob('*.csv'):
        try:
            df = pd.read_csv(csv_path)
            if 'label_new' in df.columns and 'gloss' in df.columns:
                manifest_df = df
                label_map = df.groupby('label_new')['gloss'].first().to_dict()
                st.sidebar.info(f"Detected manifest: {csv_path}")
                break
        except Exception:
            continue


def lookup_manifest_entry(video_basename: str):
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


def manifest_entry_details(entry):
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


def is_path_in_roi_folder(path_like) -> bool:
    try:
        base = VIDEOS_DIR.resolve()
        candidate = Path(path_like).resolve()
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            return False
    except Exception:
        try:
            return str(Path(path_like)).startswith(str(VIDEOS_DIR))
        except Exception:
            return False


def suggested_samples(limit=4):
    ensure_manifest_df_loaded()
    suggestions = []
    try:
        if PER_CLASS_ACC_PATH.exists():
            acc_df = pd.read_csv(PER_CLASS_ACC_PATH)
            acc_df = acc_df.sort_values('accuracy', ascending=False)
            for _, row in acc_df.iterrows():
                label_val = int(row['label']) if 'label' in row else None
                gloss = row.get('gloss')
                if gloss is None and label_val is not None and label_map is not None:
                    gloss = label_map.get(label_val)
                if manifest_df is None or label_val is None:
                    continue
                matches = manifest_df[(manifest_df['label_new'] == label_val)]
                for _, mrow in matches.iterrows():
                    p = Path(str(mrow['path']))
                    if p.exists():
                        suggestions.append({'path': p, 'label': label_val, 'gloss': gloss or p.stem})
                        break
                if len(suggestions) >= limit:
                    break
    except Exception:
        pass
    if not suggestions and BEST_SANITY_CLIP.exists():
        suggestions.append({'path': BEST_SANITY_CLIP, 'label': 4, 'gloss': BEST_SANITY_GLOSS})
    return suggestions[:limit]


# ---------------------- Session-state helpers ----------------------

def context_keys(prefix: str):
    return {
        'model': f'{prefix}_model_obj',
        'ckpt': f'{prefix}_ckpt_loaded',
        'arch': f'{prefix}_model_arch',
        'debug': f'{prefix}_last_debug',
        'load_info': f'{prefix}_last_model_load_info',
    }


def load_checkpoint_into_context(prefix, arch, ckpt_path, num_classes, pretrained, placeholder=None):
    keys = context_keys(prefix)
    try:
        factory = MODEL_FACTORIES.get(arch)
        if factory is None:
            raise ValueError(f'Unknown architecture: {arch}')
        model = factory(num_classes=num_classes, pretrained=pretrained)
        if ckpt_path is not None:
            robust_load_checkpoint(model, ckpt_path)
            message = f"Loaded checkpoint: {Path(ckpt_path).name}"
        else:
            message = 'Initialized model with random weights'
        st.session_state[keys['model']] = model
        st.session_state[keys['ckpt']] = str(ckpt_path) if ckpt_path is not None else None
        st.session_state[keys['arch']] = arch
        if placeholder is not None:
            placeholder.success(message)
        return True
    except Exception as e:
        st.session_state[keys['model']] = None
        st.session_state[keys['ckpt']] = None
        if placeholder is not None:
            placeholder.error(str(e))
        return False


def run_inference_for_context(prefix, video_path, describe, conf_threshold, top_k, skip_roi_preproc, norm_mode, result_box):
    keys = context_keys(prefix)
    model = st.session_state.get(keys['model'])
    if model is None:
        result_box.warning('Model not loaded yet.')
        return
    vp = Path(video_path)
    if not vp.exists():
        result_box.error(f'Video not found: {vp}')
        return
    try:
        manifest_entry = lookup_manifest_entry(vp.name)
        in_manifest, manifest_row_path, manifest_label, true_gloss = manifest_entry_details(manifest_entry)
        with st.spinner(f'Reading {describe}...'):
            is_roi = False
            if skip_roi_preproc:
                is_roi = is_path_in_roi_folder(vp)
                if (not is_roi) and in_manifest and manifest_row_path:
                    candidate = Path(str(manifest_row_path))
                    if candidate.exists() and is_path_in_roi_folder(candidate):
                        is_roi = True
            read_resize = None if (skip_roi_preproc and is_roi) else (112, 112)
            frames = read_video_frames_opencv(vp, clip_len=32, resize=read_resize, stride=2)
            tensor = frames_to_tensor(frames, skip_normalize=(skip_roi_preproc and is_roi), norm_mode=norm_mode)
        with st.spinner('Running inference...'):
            pred, conf, probs = predict_tensor(model, tensor)
        probs_arr = np.array(probs)
        topk_idx = probs_arr.argsort()[::-1][:top_k]
        topk = [(int(i), float(probs_arr[i])) for i in topk_idx]
        lines = []
        for idx, score in topk:
            name = label_map.get(idx) if label_map is not None else str(idx)
            lines.append({'label': idx, 'gloss': name, 'confidence': score})
        st.session_state[keys['debug']] = {
            'checkpoint': st.session_state.get(keys['ckpt']),
            'clip_path': str(vp),
            'tensor_shape': tuple(tensor.shape),
            'tensor_min': float(tensor.min()),
            'tensor_max': float(tensor.max()),
            'topk': lines,
            'probs': probs_arr.tolist(),
            'in_manifest': in_manifest,
            'manifest_row_path': str(manifest_row_path) if manifest_row_path is not None else None,
            'manifest_label': int(manifest_label) if manifest_label is not None else None,
            'true_gloss': true_gloss,
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
        result_box.error(f'Error during inference: {e}')


def debug_expander(prefix, title):
    keys = context_keys(prefix)
    dbg = st.session_state.get(keys['debug'])
    if not dbg:
        return
    with st.expander(title, expanded=False):
        st.write('Checkpoint:', dbg.get('checkpoint'))
        st.write('Clip path:', dbg.get('clip_path'))
        st.write('Tensor shape:', dbg.get('tensor_shape'))
        st.write('Tensor min/max:', dbg.get('tensor_min'), dbg.get('tensor_max'))
        try:
            df_dbg = pd.DataFrame(dbg.get('topk'))
            st.table(df_dbg)
        except Exception:
            pass
        if dbg.get('in_manifest'):
            st.write('Manifest path:', dbg.get('manifest_row_path'))
            st.write('Manifest label:', dbg.get('manifest_label'))
            st.write('True gloss:', dbg.get('true_gloss'))
        if dbg.get('clip_path') and Path(dbg.get('clip_path')).exists():
            try:
                frames = read_video_frames_opencv(dbg.get('clip_path'), clip_len=8, resize=(112, 112))
                imgs = (frames * 255).astype('uint8')
                st.image([imgs[i] for i in range(min(8, imgs.shape[0]))], width=140)
            except Exception:
                pass


# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title='ASL Translator (Tabs)', layout='wide')
PROJECT_MESSAGE = (
    "### AAI-590 Sign Language Translator\n"
    "This demo showcases our capstone ASL recognition models. Use the Quick Demo tab for a "
    "ready-to-run experience, or switch to Advanced Mode to experiment with other checkpoints "
    "and configurations."
)
st.markdown(PROJECT_MESSAGE)

ensure_manifest_df_loaded()
auto_load_manifest_from_data_folder()

quick_tab, advanced_tab = st.tabs(["Quick Demo", "Advanced Mode"])

# ---------------------- Quick Tab ----------------------
with quick_tab:
    st.subheader('Quick Demo — Best Checkpoint')
    st.write('Preloaded with the best VideoResNet (r3d_18) checkpoint. Choose a suggested clip or upload your own.')
    status_box = st.empty()
    if BEST_CKPT_PATH.exists():
        current_ckpt = st.session_state.get(context_keys('quick')['ckpt'])
        if current_ckpt != str(BEST_CKPT_PATH):
            with st.spinner('Loading best checkpoint...'):
                load_checkpoint_into_context('quick', BEST_ARCH, BEST_CKPT_PATH, BEST_NUM_CLASSES, True, status_box)
    else:
        status_box.error(f'Missing checkpoint: {BEST_CKPT_PATH}')
    suggestions = suggested_samples(limit=4)
    sample_options = ['-- none --'] + [f"{item['gloss']} — {item['path'].name}" for item in suggestions]
    selected = st.selectbox('Suggested ROI samples', options=sample_options, index=0)
    quick_result = st.empty()
    if selected != '-- none --':
        idx = sample_options.index(selected) - 1
        sample_entry = suggestions[idx]
        if st.button('Run selected sample', key='quick_sample_btn'):
            run_inference_for_context('quick', sample_entry['path'], sample_entry['gloss'], 0.10, 3, False, 'training', quick_result)
    if BEST_SANITY_CLIP.exists():
        if st.button(f"Run sanity clip ({BEST_SANITY_GLOSS})", key='quick_sanity_btn'):
            run_inference_for_context('quick', BEST_SANITY_CLIP, 'sanity clip', 0.10, 3, False, 'training', quick_result)
    uploaded = st.file_uploader('Or upload a video', type=['mp4', 'mov', 'avi', 'mkv'], key='quick_upload')
    if uploaded is not None:
        tmp_path = Path(tempfile.gettempdir()) / f"quick_uploaded_{int(time.time())}.{uploaded.name.split('.')[-1]}"
        with open(tmp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        st.info(f'Uploaded file saved to {tmp_path}')
        if st.button('Process uploaded video', key='quick_process_btn'):
            run_inference_for_context('quick', tmp_path, 'uploaded video', 0.10, 3, False, 'training', quick_result)
    debug_expander('quick', 'Quick Demo Debug Info')

# ---------------------- Advanced Tab ----------------------
with advanced_tab:
    st.subheader('Advanced Mode — Customize Everything')
    st.write('Choose different checkpoints, architectures, preprocessing options, and data sources.')
    manifest_upload = st.file_uploader('Optional manifest CSV (label_new, gloss, path)', type=['csv'], key='adv_manifest_upload')
    if manifest_upload is not None:
        load_manifest_from_upload(manifest_upload)
    conf_threshold = st.slider('Confidence threshold', min_value=0.0, max_value=1.0, value=0.10, step=0.01, key='adv_conf')
    top_k = st.number_input('Top-k predictions to display', min_value=1, max_value=10, value=3, key='adv_topk')
    norm_mode = st.radio('Normalization', options=['training', 'kinetics'], index=0, key='adv_norm')
    skip_roi_preproc = st.checkbox('Skip resize/normalize for ROI clips (already preprocessed)', value=False, key='adv_skip_roi')

    available_ckpts = sorted({p.name for p in CKPT_DIR.glob('*.pt')} | {p.name for p in CKPT_DIR.glob('*.pth')})
    ckpt_choice = st.selectbox('Checkpoint from `checkpoints/`', options=['-- none --'] + available_ckpts, key='adv_ckpt_choice')
    uploaded_ckpt = st.file_uploader('Or upload a checkpoint (.pt/.pth)', type=['pt', 'pth'], key='adv_ckpt_upload')

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        arch_choice = st.selectbox('Model architecture', options=list(MODEL_FACTORIES.keys()), index=0, key='adv_arch')
    with col_b:
        num_classes = st.number_input('Num classes', min_value=2, value=104, key='adv_num_classes')
    with col_c:
        pretrained_flag = st.checkbox('Use Kinetics pretrained backbone', value=True, key='adv_pretrained')

    load_placeholder = st.empty()
    if st.button('Load checkpoint/model', key='adv_load_btn'):
        if uploaded_ckpt is not None:
            tmp_path = Path(tempfile.gettempdir()) / f"adv_uploaded_ckpt_{int(time.time())}.pt"
            with open(tmp_path, 'wb') as f:
                f.write(uploaded_ckpt.getbuffer())
            ckpt_path = tmp_path
        elif ckpt_choice != '-- none --':
            ckpt_path = CKPT_DIR / ckpt_choice
        else:
            ckpt_path = None
        with st.spinner('Loading advanced model...'):
            load_checkpoint_into_context('advanced', arch_choice, ckpt_path, num_classes, pretrained_flag, load_placeholder)

    adv_result = st.empty()
    adv_uploaded = st.file_uploader('Upload a video for inference', type=['mp4', 'mov', 'avi', 'mkv'], key='adv_video_upload')
    if adv_uploaded is not None:
        tmp_path = Path(tempfile.gettempdir()) / f"adv_uploaded_video_{int(time.time())}.{adv_uploaded.name.split('.')[-1]}"
        with open(tmp_path, 'wb') as f:
            f.write(adv_uploaded.getbuffer())
        st.info(f'Advanced upload saved to {tmp_path}')
        if st.button('Process uploaded video', key='adv_process_btn'):
            run_inference_for_context('advanced', tmp_path, 'uploaded video', conf_threshold, top_k, skip_roi_preproc, norm_mode, adv_result)

    ensure_manifest_df_loaded()
    adv_sample = None
    if manifest_df is not None and 'path' in manifest_df.columns:
        existing = []
        for _, row in manifest_df.iterrows():
            try:
                p = Path(str(row['path']))
            except Exception:
                continue
            if p.exists():
                gloss = row.get('gloss') if 'gloss' in row.index else p.stem
                existing.append((str(p), gloss))
        if existing:
            options = ['-- none --'] + [f"{i}: {Path(item[0]).name} — {item[1]}" for i, item in enumerate(existing)]
            sel = st.selectbox('Select a manifest sample', options=options, key='adv_manifest_select')
            if sel != '-- none --':
                sel_idx = int(sel.split(':', 1)[0])
                adv_sample = existing[sel_idx][0]
    if adv_sample and st.button('Process manifest sample', key='adv_manifest_btn'):
        run_inference_for_context('advanced', adv_sample, 'manifest sample', conf_threshold, top_k, skip_roi_preproc, norm_mode, adv_result)

    st.markdown('---')
    st.write('Live capture (simple snapshots)')
    if 'advanced_snapshot_frames' not in st.session_state:
        st.session_state['advanced_snapshot_frames'] = []
    cam_file = st.camera_input('Capture a frame', key='adv_camera')
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Add captured frame', key='adv_add_frame'):
            if cam_file is None:
                st.warning('Take a snapshot first.')
            else:
                bytes_data = cam_file.getvalue()
                arr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (112, 112))
                    st.session_state['advanced_snapshot_frames'].append(img)
                    st.info(f'Captured frames: {len(st.session_state["advanced_snapshot_frames"])}')
    with col2:
        if st.button('Clear captured frames', key='adv_clear_frames'):
            st.session_state['advanced_snapshot_frames'] = []
    with col3:
        if st.button('Process captured clip', key='adv_process_clip'):
            frames = st.session_state.get('advanced_snapshot_frames', [])
            if not frames:
                st.warning('No captured frames yet.')
            else:
                clip_len = 32
                frames_copy = list(frames)
                while len(frames_copy) < clip_len:
                    frames_copy.append(frames_copy[-1])
                if len(frames_copy) > clip_len:
                    indices = np.linspace(0, len(frames_copy) - 1, clip_len, dtype=int)
                    frames_copy = [frames_copy[i] for i in indices]
                arr = np.stack(frames_copy, axis=0).astype(np.float32) / 255.0
                tensor = frames_to_tensor(arr, skip_normalize=False, norm_mode=norm_mode)
                model = st.session_state.get(context_keys('advanced')['model'])
                if model is None:
                    st.warning('Load a model before running live capture.')
                else:
                    with st.spinner('Running inference...'):
                        pred, conf, probs = predict_tensor(model, tensor)
                    probs_arr = np.array(probs)
                    topk_idx = probs_arr.argsort()[::-1][:top_k]
                    lines = []
                    for idx in topk_idx:
                        label = int(idx)
                        lines.append({'label': label, 'gloss': label_map.get(label) if label_map else str(label), 'confidence': float(probs_arr[idx])})
                    st.session_state[context_keys('advanced')['debug']] = {
                        'checkpoint': st.session_state.get(context_keys('advanced')['ckpt']),
                        'clip_path': 'captured_frames',
                        'tensor_shape': tuple(tensor.shape),
                        'tensor_min': float(tensor.min()),
                        'tensor_max': float(tensor.max()),
                        'topk': lines,
                        'probs': probs_arr.tolist(),
                    }
                    display_obj = {'top_k': lines}
                    if float(lines[0]['confidence']) < conf_threshold:
                        adv_result.warning('No confident translation found (top-1 confidence below threshold).')
                        adv_result.json(display_obj)
                    else:
                        adv_result.success(f"Prediction: {lines[0]['gloss']} (label={lines[0]['label']}) — confidence {lines[0]['confidence']:.2f}")
                        adv_result.json(display_obj)

    debug_expander('advanced', 'Advanced Debug Info')
