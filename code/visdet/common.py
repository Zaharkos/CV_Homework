import glob
import json
import os
import random
from pathlib import Path

import numpy as np

CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NC = len(CLASSES)

LONG_SIDE = 1280
TRAIN_STEP = 9
FINAL_STEP = 3
VAL_STEP = 9
MAX_DET = 300
SEED = 0


def is_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def is_debug() -> bool:
    return os.environ.get("VISDET_DEBUG", "0") == "1"


def find_comp_root(hint=None) -> Path:
    for c in (hint, os.environ.get("VISDET_DATA")):
        if c and (Path(c) / "test_images.csv").exists():
            return Path(c)
    for depth in range(1, 5):
        pattern = "/kaggle/input/" + "*/" * depth + "test_images.csv"
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError(
        "Competition data not found. Attach the competition to the notebook "
        "or set VISDET_DATA=/path/to/data")


def work_dir() -> Path:
    d = Path(os.environ.get("VISDET_WORK", "/kaggle/working" if is_kaggle() else "./work"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = Path(os.environ.get("VISDET_DATA_DIR", "/tmp/visdet_data" if is_kaggle() else str(work_dir() / "data")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_file(pattern: str, roots=None):
    for r in roots or ("/kaggle/input", str(work_dir())):
        hits = sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def setup_wandb_from_kaggle_secret():
    if not os.environ.get("WANDB_API_KEY"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            pass
    if os.environ.get("WANDB_API_KEY"):
        os.environ.pop("WANDB_MODE", None)
        print("W&B logging: ON")
    else:
        os.environ["WANDB_MODE"] = "disabled"
        print("W&B logging: OFF (add a Kaggle secret named WANDB_API_KEY to enable)")
    os.environ.setdefault("WANDB_PROJECT", "ucu-visdrone-hw2")
    os.environ["WANDB_SILENT"] = "true"


def _fp16():
    try:
        import torch
        return {"quantize": 16} if torch.cuda.is_available() else {}
    except ImportError:
        return {}


def compact_preds(preds, min_score=0.05):
    import numpy as np
    out = {}
    for k, v in preds.items():
        m = np.asarray(v["scores"]) >= min_score
        out[k] = dict(boxes=np.round(np.asarray(v["boxes"])[m], 1).tolist(),
                      scores=np.round(np.asarray(v["scores"])[m], 4).tolist(),
                      labels=np.asarray(v["labels"])[m].astype(int).tolist())
    return out
