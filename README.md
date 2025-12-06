## Sign Language Recognition using Deep Learning

### Project Description

This project implements a real-time American Sign Language (ASL) word-level recognition system using deep learning and computer vision. Built from scratch as part of the University of San Diego’s AAI-590 Capstone, the system processes short video clips or live webcam streams to classify ASL glosses using a 3D convolutional neural network architecture.

The pipeline includes full data preprocessing, MediaPipe-based region-of-interest extraction, temporal frame sampling, augmentation, model training, and evaluation. Multiple architectures were tested, with R(2+1)D-18 pretrained on Kinetics-400 achieving the highest validation performance.
The project provides an interactive demo interface and lays the groundwork for future expansion toward full ASL-to-English sentence translation.

---

### Repository Layout

```
AAI-590-G3-ASL/
├─ apps/                          # Streamlit + API front ends
│  ├─ streamlit_app_multitab_live.py   #  latest Streamlit entry point
│  ├─ streamlit_app.py                # legacy prototype
│  └─ api_flask.py
├─ configs/
│  └─ nslt2000.yaml
├─ data/wlasl_preprocessed/
│  ├─ videos/, videos_clean/, videos_trim/, videos_roi/   # raw → cleaned → trimmed → ROI clips
│  ├─ manifest_nslt2000_*.csv                             # manifests at each stage
│  └─ metadata json files (WLASL_v0.3.json, nslt_2000.json, class maps, etc.)
├─ notebooks/                    # preprocessing + modeling notebooks (see summary below)
├─ scripts/                      # helper utilities for automation (details in “Helper Utilities”)
├─ src/
│  ├─ data_utils/                # reusable data/ROI tooling
│  ├─ training/                  # dataset wrappers, model zoo, losses, loops
│  └─ inference/                 # deployment pipelines
├─ runs/                         # audits, logs, generated shell scripts
├─ checkpoints/                  # saved model weights
├─ reports/                      # metrics, per-class CSVs, confusion matrices
├─ requirements.txt, environment.yml
└─ README.md
```

---

### Streamlit App

The current demo lives in `apps/streamlit_app_multitab_live.py`. Earlier files (`app_streamlit.py`, `streamlit_app.py`) remain for reference but may lag features/data bindings.

```bash
conda activate <env>   # or pip/venv equivalent
cd /home/falasoul/notebooks/USD/AAI-590/Capstone/AAI-590-G3-ASL
pip install -r requirements.txt  # first-time setup
streamlit run apps/streamlit_app_multitab_live.py --server.port 8501
```

#### Local Windows setup (Python 3.11 + venv)

The repository has also been tested on Windows 11 with Python 3.11 in a virtual environment. A minimal setup looks like:

```powershell
cd C:\Temp\USD\590-G3-ASL\AAI-590-G3-ASL

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run apps/streamlit_app_multitab_live.py --server.port 8501
```

---

### Helper Utilities

Outside the notebooks, several helpers live under `scripts/` (CLI-friendly) and `notebooks/` (iterative experiments):

| Location | Helper | Purpose |
| --- | --- | --- |
| `scripts/build_manifest_wlasl_preprocessed.py` | Manifest builder | Mirrors `01_manifest_nslt2000.ipynb` logic to regenerate CSVs headlessly. |
| `scripts/preprocess_segments.py` | Re-encode + trim | Batch ffmpeg/re-encoding aligned with `02_preprocess_reencode_trim.ipynb`. |
| `scripts/clean_roi_manifests.py` & `scripts/batch_infer_roi.py` | ROI repair/dedupe | Complements `03_roi_mediapipe_resume.ipynb` by fixing broken crops and deduplicating outputs. |
| `scripts/detect_encoding_errors.py`, `scripts/find_missing.py` | Audits | Surface corrupt clips or broken manifest entries. |
| Notebook helpers | `02_preprocess_reencode_trim.ipynb`, `03_roi_mediapipe_resume.ipynb`, `05_select_top_and_balance.ipynb`, etc., include embedded functions (ffprobe wrappers, MediaPipe ROI logic, sampler/balancer utilities) that inspired the corresponding scripts. |

Use the scripts for reproducible batch runs; refer to the notebooks when you need step-by-step context or to tweak thresholds interactively.

---

### Notebook & Model Summary

| Notebook | Focus & Model | Notes |
| --- | --- | --- |
| `01_manifest_nslt2000.ipynb` | Manifest creation with NSLT metadata | Normalizes JSON metadata and produces strict local-path manifests. |
| `02_preprocess_reencode_trim.ipynb` | Audit → ffmpeg re-encode → trimming | H.264/yuv420p/30 fps standardization, active-window trims, merged manifests. |
| `03_roi_mediapipe_resume.ipynb` | MediaPipe ROI crops + manifests | CPU-only MediaPipe Hands/Pose pass, ffmpeg piping, ROI audits and manifest rebuilds. |
| `04_visualize_and_motion.ipynb` | Dataset EDA | Split distributions, motion energy visualization pre/post ROI. |
| `05_select_top_and_balance.ipynb` | Balanced subset selection | Enforces per-split minima, caps per-class counts, exports class maps/weights. |
| `06_train_baseline_3dCNN-r3d18_k400_kenetics-R(2+1)D-18.ipynb` | R(2+1)D‑18 baseline (acc ≈ 0.453) | Uses `R2Plus1D_18_Weights.KINETICS400_V1`; each 3D conv is factorized into a 2D spatial + 1D temporal step, providing faster convergence vs. plain R3D while ingesting `[B,3,T,H,W]` clips. |
| `06_train_baseline_3dCNN-r3d18_k400_kenetics.ipynb` | R3D‑18 baseline (acc ≈ 0.432) | Loads `R3D_18_Weights.KINETICS400_V1` (3D ResNet‑18 pretrained on Kinetics‑400). Full 3D convolutions capture spatial + temporal cues simultaneously. |
| `06_train_baseline_CNN.ipynb` | Scratch 3D CNN baseline (acc ≈ 0.083) | Instantiates `r3d_18(weights=None)` with a custom FC head. Pure 3D CNN learns from scratch and, unsurprisingly, underperforms without pretrained priors. |
| `06_train_baseline_CNN_BiGRU_aug-RestNet18.ipynb` | 2D ResNet‑18 encoder + BiGRU head (acc ≈ 0.130) | Optionally loads `ResNet18_Weights.IMAGENET1K_V1` (standard 2D ImageNet backbone) before temporal aggregation with BiGRUs for ROI clip sequences. |
| Additional `06_*` notebooks | Variants (augmentation sweeps, tuning runs) | Each notebook’s `reports/` artifacts capture the matching checkpoints, confusion matrices, and per-class accuracy CSVs. |

---

### Authors

- Fuad Al Asouli — `falasouli@sandiego.edu`
- Mythreyi Thirumalai — `mthirumalai@sandiego.edu`

---

For questions or contributions, open an issue/PR or reach out via the emails above. When extending the pipeline, update the relevant notebook and a matching helper script so batch workflows stay in sync.

---
### Checkpoints
Due to size limit these are saved on drive
https://drive.google.com/drive/folders/1GioVPayxsfmPp2a-9FztuoI1wAiHW1DT?usp=drive_link
its shared with instructor, if you have any questions please email authors. 


### Acknowledgments

We would like to thank the University of San Diego’s Applied Artificial Intelligence program for providing the foundation, guidance, and motivation behind this project. Special thanks to our instructor, Anna Marbut, M.S., for her mentorship and continuous support throughout the capstone.
We also acknowledge the contributors of the WLASL dataset, whose work made this research and model development possible.
