from pathlib import Path
import subprocess, json

root = Path("../data/wlasl_preprocessed")
src_dir = root / "videos_clean"
roi_dir = root / "videos_roi"

# find broken ROI files (<10 KB)
broken = [p for p in roi_dir.glob("*.mp4") if p.stat().st_size < 10_000]
print(f"Found {len(broken)} broken ROI clips")

# re-encode them using original videos as source
for b in broken:
    vid = b.stem                      # e.g. "03542"
    src = src_dir / f"{vid}.mp4"
    if not src.exists():
        print(f"⚠️ Missing source for {vid}")
        continue

    out_tmp = roi_dir / f"{vid}_fix.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-crf", "23",
        "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_tmp)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_tmp.rename(b)  # replace old file
        print(f"✅ Re-encoded {vid}")
    except Exception as e:
        print(f"❌ Failed {vid}: {e}")
