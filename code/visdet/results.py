import glob
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_RESULTS = CODE_DIR / "results"

COLS = ["exp", "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR500", "ms_per_img",
        "epochs", "train_minutes"]


def search_roots(extra=()):
    roots = [*extra, os.environ.get("VISDET_WORK", ""), "/kaggle/working/runs", str(REPO_RESULTS), "/kaggle/input"]
    return [r for r in roots if r and os.path.isdir(r)]


def find_run(exp, extra=()):
    for r in search_roots(extra):
        hits = sorted(glob.glob(os.path.join(r, "**", exp, "metrics.json"), recursive=True))
        if hits:
            return Path(hits[0]).parent
    return None


def collect(prefixes, extra=()):
    found = {}
    for r in search_roots(extra):
        for p in glob.glob(os.path.join(r, "**", "metrics.json"), recursive=True):
            exp = Path(p).parent.name
            if exp.startswith(tuple(prefixes)) and exp not in found:
                found[exp] = json.load(open(p))
    if not found:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame([{k: v for k, v in m.items() if not isinstance(v, (dict, list))} for m in found.values()])
    cols = [c for c in COLS if c in df] + sorted(c for c in df if c.startswith("AP_") and c not in COLS)
    return df[cols].sort_values("exp").reset_index(drop=True)


def load_preds(run_dir):
    p = Path(run_dir) / "val_preds.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in d.items()}


def export_results(run_root, out_zip, figures_dir=None, include_preds=True):
    run_root = Path(run_root)
    keep = {"metrics.json", "history.csv", "results.csv", "args.yaml", "strategies.csv"} | ({"val_preds.json"} if include_preds else set())
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_root.glob("*/*"):
            if p.name in keep:
                z.write(p, f"code/results/{p.parent.name}/{p.name}")
        for p in run_root.glob("*.csv"):
            z.write(p, f"code/results/{p.name}")
        if figures_dir and Path(figures_dir).is_dir():
            for p in Path(figures_dir).glob("*.png"):
                z.write(p, f"images/{p.name}")
    print(f"saved {out_zip} ({Path(out_zip).stat().st_size / 1e6:.1f} MB): unzip it at the root of the repo (code/results/ + images/)")
