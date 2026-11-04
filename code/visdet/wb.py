import os

import cv2
import numpy as np

from .common import CLASSES
from .evaluate import gt_per_image


def enabled():
    return bool(os.environ.get("WANDB_API_KEY")) and os.environ.get("WANDB_MODE") != "disabled"


def init(name, config, job_type="train"):
    if not enabled():
        return None
    try:
        import wandb
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "ucu-visdrone-hw2"), name=name,
                          group=name.split("_")[0], job_type=job_type, config=config)
    except Exception as e:
        print("W&B init failed:", e)
        return None


def log(run, d, step=None):
    if run is not None:
        try:
            run.log(d, step=step)
        except Exception as e:
            print("W&B log failed:", e)


def log_summary(run, metrics):
    if run is not None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                run.summary[k] = v


def log_val_images(run, val_manifest, preds, gt, n=6, score_thr=0.3, max_side=1280):
    if run is None:
        return
    try:
        import wandb
        G = gt_per_image(gt)
        labels = {i: c for i, c in enumerate(CLASSES)}
        idx = np.linspace(0, len(val_manifest) - 1, min(n, len(val_manifest))).astype(int)
        ims = []
        for r in val_manifest.iloc[idx].itertuples():
            img = cv2.cvtColor(cv2.imread(r.src_img), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            s = min(1.0, max_side / max(H, W))
            img = cv2.resize(img, (round(W * s), round(H * s)))
            p = preds.get(r.stem, {"boxes": [], "scores": [], "labels": []})

            def boxes(bb, cc, ss=None):
                out = []
                for k, (b, c) in enumerate(zip(bb, cc)):
                    d = dict(position=dict(minX=float(b[0] * s), minY=float(b[1] * s),
                                           maxX=float(b[2] * s), maxY=float(b[3] * s)),
                             domain="pixel", class_id=int(c), box_caption=CLASSES[int(c)])
                    if ss is not None:
                        d["scores"] = {"score": float(ss[k])}
                    out.append(d)
                return out
            keep = np.asarray(p["scores"]) >= score_thr
            gb, gl, _ = G[r.stem]
            ims.append(wandb.Image(img, caption=r.stem, boxes={
                "predictions": {"box_data": boxes(np.asarray(p["boxes"])[keep], np.asarray(p["labels"])[keep],
                                                  np.asarray(p["scores"])[keep]), "class_labels": labels},
                "ground_truth": {"box_data": boxes(gb, gl), "class_labels": labels}}))
        run.log({"val/predictions": ims})
    except Exception as e:
        print("W&B image logging failed:", e)


def log_images(run, key, paths):
    if run is None:
        return
    try:
        import wandb
        run.log({key: [wandb.Image(str(p), caption=os.path.basename(str(p))) for p in paths]})
    except Exception as e:
        print("W&B image logging failed:", e)


def finish(run):
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
