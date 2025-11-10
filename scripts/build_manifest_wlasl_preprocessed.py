#!/usr/bin/env python3
"""
Build manifest CSV for the Kaggle WLASL-Processed NSLT format.

NSLT file shape:
{
  "05237": {"subset": "train", "action": [class_idx, start, end]},
  "69422": {"subset": "val",   "action": [class_idx, start, end]},
  ...
}

We map class_idx -> gloss via a class list (one gloss per line).
Output columns: video_path, gloss, label, split
"""

import argparse, json, csv
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}

def load_class_list(path: Path) -> list:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def vid_to_filename(vid_key: str) -> str:
    # keys may already be zero-padded; normalize safely to 5 digits
    digits = ''.join(ch for ch in str(vid_key) if ch.isdigit())
    if not digits:
        return None
    return f"{int(digits):05d}.mp4"

def build_manifest_nslt_indexed(root: Path, nslt_json: Path, class_list_path: Path, out_csv: Path):
    videos_dir = root / "videos"
    assert videos_dir.exists(), f"Missing videos dir: {videos_dir}"
    assert nslt_json.exists(), f"Missing nslt file: {nslt_json}"
    assert class_list_path.exists(), f"Missing class list: {class_list_path}"

    classes = load_class_list(class_list_path)

    data = json.load(open(nslt_json, "r"))
    if not (isinstance(data, dict) and data):
        raise ValueError("Expected NSLT-indexed dict with video_id keys.")

    # Quick shape check using one value
    any_val = next(iter(data.values()))
    if not (isinstance(any_val, dict) and "subset" in any_val and "action" in any_val):
        raise ValueError("NSLT file does not look like the expected indexed format with 'subset' and 'action'.")

    rows, missing = [], []
    for vid, info in data.items():
        split = info.get("subset")
        if split not in VALID_SPLITS:
            continue
        act = info.get("action")
        if not (isinstance(act, list) and len(act) >= 1):
            continue
        label_idx = int(act[0])
        if not (0 <= label_idx < len(classes)):
            continue
        gloss = classes[label_idx]
        fn = vid_to_filename(vid)
        if not fn:
            continue
        vp = videos_dir / fn
        if vp.exists():
            rows.append({
                "video_path": str(vp),
                "gloss": gloss,
                "label": label_idx,
                "split": split
            })
        else:
            missing.append(fn)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_path","gloss","label","split"])
        w.writeheader()
        w.writerows(rows)

    miss_path = root / f"missing_in_{nslt_json.name.replace('.json','')}.txt"
    with open(miss_path, "w") as f:
        for m in sorted(set(missing)):
            f.write(m + "\n")

    print(f"[OK] {nslt_json.name} -> {out_csv} | samples={len(rows)} | missing_list={miss_path} ({len(missing)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., data/wlasl_preprocessed")
    ap.add_argument("--nslt", required=True, help="e.g., nslt_100.json")
    ap.add_argument("--out",  required=True, help="output CSV path")
    ap.add_argument("--class-list", default=None, help="path to class list txt (one gloss per line)")
    args = ap.parse_args()

    root = Path(args.root)
    nslt_json = root / args.nslt
    class_list = Path(args.class_list) if args.class_list else (root / "wlasl_class_list.txt")

    build_manifest_nslt_indexed(root, nslt_json, class_list, Path(args.out))
