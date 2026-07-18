#!/usr/bin/env bash
set -euo pipefail

cd /home/liufangyi/proj/SLotSPE/SlotSPE

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("results_train/kirc/SlotSPE")
rows = []
for summary in root.glob("*/summary.csv"):
    df = pd.read_csv(summary)
    mean = df[df["folds"].astype(str) == "mean"]
    std = df[df["folds"].astype(str) == "std"]
    if mean.empty:
        continue
    rows.append({
        "run": summary.parent.name,
        "mean_cindex": float(mean["val_cindex"].iloc[0]),
        "std_cindex": float(std["val_cindex"].iloc[0]) if not std.empty else float("nan"),
        "mean_ipcw": float(mean["val_cindex_ipcw"].iloc[0]),
        "mean_ibs": float(mean["val_IBS"].iloc[0]),
        "mean_iauc": float(mean["val_iauc"].iloc[0]),
        "summary": str(summary),
    })

if not rows:
    print("No completed summaries found.")
else:
    out = pd.DataFrame(rows).sort_values("mean_cindex", ascending=False)
    pd.set_option("display.max_colwidth", 160)
    print(out.to_string(index=False))
PY
