import os
from pathlib import Path
import pandas as pd

root = Path("..").resolve()
data_dir = root / "data" / "wlasl_preprocessed"

for man_path in sorted(data_dir.glob("manifest_nslt2000_roi_*.csv")):
    print(f"\n=== Cleaning {man_path.name} ===")
    df = pd.read_csv(man_path)
    n_before = len(df)

    df["exists"] = df["path"].apply(os.path.exists)
    n_missing = (~df["exists"]).sum()

    print(f"Rows before: {n_before} | missing files: {n_missing}")

    df_clean = df[df["exists"]].reset_index(drop=True)
    n_after = len(df_clean)

    # Save with _clean suffix to be safe
    clean_path = man_path.with_name(man_path.stem + "_clean.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"Rows after:  {n_after} | saved: {clean_path.name}")
