AAI-590-G3-ASL/
├─ configs/
│  └─ nslt2000.yaml
├─ data/
│  └─ wlasl_preprocessed/
│     ├─ WLASL_v0.3.json
│     ├─ nslt_2000.json
│     ├─ videos/                # original mp4s (existing)
│     ├─ videos_clean/          # re-encoded (created)
│     ├─ videos_trim/           # trimmed (created)
│     ├─ videos_roi/            # ROI-cropped (created)
│     ├─ manifest_nslt2000_raw.csv
│     ├─ manifest_nslt2000_clean.csv
│     ├─ manifest_nslt2000_trim.csv
│     ├─ manifest_nslt2000_roi.csv
│     └─ class_map_nslt2000.csv
├─ notebooks/
│  ├─ 01_manifest_nslt2000.ipynb            # build manifest from nslt_2000.json
│  ├─ 02_preprocess_reencode_trim.ipynb     # audit → re-encode → trim
│  ├─ 03_roi_mediapipe.ipynb                # hands/upper-body ROI crop (MediaPipe)
│  ├─ 04_visualize_and_motion.ipynb         # distributions & motion viz (before/after)
│  ├─ 05_select_top_and_balance.ipynb       # top classes & imbalance handling
│  └─ 06_train_r3d18_nslt2000.ipynb         # full training (separate notebook)
├─ src/
│  ├─ data_utils/
│  │  ├─ manifest_utils.py
│  │  ├─ video_audit.py
│  │  ├─ trim.py
│  │  └─ roi_mediapipe.py
│  ├─ training/
│  │  ├─ datasets.py
│  │  ├─ model_zoo.py
│  │  ├─ losses.py
│  │  └─ train_loop.py
│  └─ inference/
│     ├─ pipeline.py
│     └─ decode.py
├─ apps/
│  ├─ app_streamlit.py
│  └─ api_flask.py
├─ runs/                    # reports, scripts, figures
├─ checkpoints/
├─ requirements.txt
└─ README.md
