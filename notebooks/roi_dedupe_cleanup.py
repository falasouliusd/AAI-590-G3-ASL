#!/usr/bin/env python3
from pathlib import Path
import subprocess, json, os, shutil

root = Path(".").resolve()
roi_dir = root / "data" / "wlasl_preprocessed" / "videos_roi"
assert roi_dir.exists(), f"Missing {roi_dir}"

def ffprobe_ok(p: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-print_format","json","-show_streams","-show_format",str(p)],
            stderr=subprocess.STDOUT
        ).decode()
        j = json.loads(out)
        v = [s for s in j.get("streams",[]) if s.get("codec_type")=="video"]
        if not v: return False
        dur = float((j.get("format") or {}).get("duration", 0) or 0)
        return dur > 0
    except Exception:
        return False

# Gather candidates grouped by 5-digit id
by_id = {}
for p in roi_dir.glob("*.mp4"):
    stem = p.stem
    id5  = stem[:5] if stem[:5].isdigit() else None
    if not id5: continue
    by_id.setdefault(id5, []).append(p)

kept = 0; deleted = 0
for id5, files in by_id.items():
    # rank: ffprobe_ok first, then size descending
    ranked = sorted(files, key=lambda x: (ffprobe_ok(x), x.stat().st_size), reverse=True)
    best = ranked[0]
    target = roi_dir / f"{id5}.mp4"
    # Ensure best is named {id}.mp4
    if best != target:
        try:
            # overwrite target if exists and is worse
            if target.exists():
                t_ok = ffprobe_ok(target)
                if not t_ok or target.stat().st_size < best.stat().st_size:
                    target.unlink()
                else:
                    # target is already better; mark best for deletion instead
                    best = target
            if best != target:
                shutil.move(str(best), str(target))
        except Exception:
            pass

    # Delete all others for this id (including *_fix.mp4, *.remux.mp4)
    for extra in files:
        if extra == target:
            continue
        try:
            extra.unlink()
            deleted += 1
        except Exception:
            pass
    kept += 1

print(f"Done. Kept {kept} canonical files; deleted {deleted} extras.")
