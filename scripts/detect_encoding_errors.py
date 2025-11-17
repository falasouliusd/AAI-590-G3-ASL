#!/usr/bin/env python3
import argparse
from pathlib import Path
import decord
from tqdm.auto import tqdm

decord.bridge.set_bridge("torch")

def check_videos(video_dir: Path, delete: bool = False):
    assert video_dir.exists(), f"Directory not found: {video_dir}"

    mp4_files = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(mp4_files)} videos\n")

    bad = []
    good = 0

    for path in tqdm(mp4_files, desc="Checking videos"):
        try:
            vr = decord.VideoReader(str(path))
            n = len(vr)
            if n <= 0:
                bad.append(path)
            else:
                good += 1
        except Exception:
            bad.append(path)

    print("\n===== SUMMARY =====")
    print(f"Good videos:      {good}")
    print(f"Corrupted videos: {len(bad)}")

    if not bad:
        print("No corrupted videos detected. Nothing to delete.")
        return

    print("\nExample corrupted files:")
    for p in bad[:10]:
        print(" -", p)

    if delete:
        print("\nDeleting corrupted videos...\n")
        deleted = 0
        for p in bad:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {p}: {e}")

        print(f"Deleted {deleted}/{len(bad)} corrupted videos.")
    else:
        print("\nDry run mode: NO files were deleted.")
        print("Run again with --delete to remove corrupted videos.")


def main():
    parser = argparse.ArgumentParser(
        description="Check ROI videos for corruption. "
                    "By default, no deletion occurs (dry run)."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="../data/wlasl_preprocessed/videos_roi",
        help="Directory containing ROI mp4 files."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete corrupted videos instead of dry-run."
    )

    args = parser.parse_args()
    video_dir = Path(args.video_dir).resolve()

    check_videos(video_dir, delete=args.delete)


if __name__ == "__main__":
    main()
