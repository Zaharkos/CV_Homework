import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import compact_preds, _fp16, MAX_DET, SEED, VAL_STEP, is_debug, save_json, seed_everything
from .configs import BUDGET_H, FINAL_EXPS, NWD_ALPHA, NWD_C, UL_COMMON, UL_EXPS, UL_OPTIM
from .evaluate import build_gt, coco_eval, time_inference
from . import wb

ALL_EXPS = {**UL_EXPS, **FINAL_EXPS}


def _pretrained():
    return os.environ.get("VISDET_NO_PRETRAINED", "0") != "1"


def quiet_integrations():
    from ultralytics import settings
    for k in ("wandb", "mlflow", "comet", "clearml", "dvc", "neptune", "raytune", "tensorboard", "hub"):
        try:
            settings.update({k: False})
        except Exception:
            pass


P2_HEAD = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""


def model_yaml(cfg, out_dir) -> Path:
    import ultralytics
    root = Path(ultralytics.__file__).parent / "cfg" / "models"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["family"] == "yolo":
        txt = (root / "11" / "yolo11.yaml").read_text()
        name = cfg["model"]
        if cfg.get("p2"):
            txt = txt[:txt.index("head:")] + P2_HEAD
            name += "-p2"
    else:
        txt = (root / "rt-detr" / f"{cfg['model']}.yaml").read_text()
        name = cfg["model"]
        if cfg.get("nq"):
            txt, n = re.subn(r"RTDETRDecoder, \[nc\]\]", f"RTDETRDecoder, [nc, 256, {cfg['nq']}]]", txt)
            assert n == 1, "could not set the number of queries"
            name += f"-q{cfg['nq']}"
    p = out_dir / f"{name}.yaml"
    p.write_text(txt)
    return p


def build_model(cfg, out_dir):
    from ultralytics import RTDETR, YOLO
    cls = YOLO if cfg["family"] == "yolo" else RTDETR
    weights = f"{cfg['model']}.pt"
    custom = cfg.get("p2") or cfg.get("nq")
    if not custom and _pretrained():
        return cls(weights)
    m = cls(str(model_yaml(cfg, out_dir)))
    if _pretrained():
        m.load(weights)
    return m


def nwd_similarity(b1, b2, C=NWD_C, eps=1e-7):
    cx1, cy1 = (b1[..., 0] + b1[..., 2]) / 2, (b1[..., 1] + b1[..., 3]) / 2
    cx2, cy2 = (b2[..., 0] + b2[..., 2]) / 2, (b2[..., 1] + b2[..., 3]) / 2
    w1, h1 = b1[..., 2] - b1[..., 0], b1[..., 3] - b1[..., 1]
    w2, h2 = b2[..., 2] - b2[..., 0], b2[..., 3] - b2[..., 1]
    w2_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-torch.sqrt(w2_dist.clamp(min=0) + eps) / C)


NWD_CALLS = {"n": 0}


def enable_nwd_loss(alpha=NWD_ALPHA):
    import ultralytics.utils.loss as L
    orig = L.bbox_iou

    def bbox_iou_nwd(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        iou = orig(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
        b1, b2 = box1, box2
        if xywh:
            from ultralytics.utils.ops import xywh2xyxy
            b1, b2 = xywh2xyxy(box1), xywh2xyxy(box2)
        NWD_CALLS["n"] += 1
        return (1 - alpha) * iou + alpha * nwd_similarity(b1, b2).unsqueeze(-1).to(iou.dtype)

    L.bbox_iou = bbox_iou_nwd


def predict_ul(model, paths, stems, imgsz, conf=0.001, iou=0.6, augment=False, batch=8, max_det=MAX_DET):
    out = {}
    for i in range(0, len(paths), batch):
        res = model.predict(paths[i:i + batch], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                            augment=augment, verbose=False, **_fp16())
        for s, r in zip(stems[i:i + batch], res):
            b = r.boxes
            out[s] = dict(boxes=b.xyxy.cpu().numpy(), scores=b.conf.cpu().numpy(), labels=b.cls.cpu().numpy().astype(int))
    return out


def load_trained(path):
    from ultralytics import RTDETR, YOLO
    return RTDETR(str(path)) if "rtdetr" in str(path) else YOLO(str(path))


def train(exp, data_yaml, out_root, budget_h=None, val_manifest=None, override=None):
    base = {**ALL_EXPS.get(exp, {}), **(override or {})}
    cfg = {**UL_COMMON, **UL_OPTIM[base["family"]], **base}
    budget_h = budget_h if budget_h is not None else BUDGET_H
    imgsz = cfg["imgsz"]
    if is_debug():
        budget_h, imgsz = 0.01, 320
    seed_everything(SEED)
    quiet_integrations()
    out_root = Path(out_root)
    res_dir = out_root / exp
    res_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("nwd"):
        enable_nwd_loss()
    model = build_model(cfg, res_dir)
    device = 0 if torch.cuda.is_available() else "cpu"
    kw = dict(data=str(data_yaml), imgsz=imgsz, epochs=300, time=budget_h, batch=cfg["batch"],
              workers=cfg["workers"], device=device, seed=SEED, project=str(out_root / "ul"), name=exp,
              exist_ok=True, optimizer=cfg["optimizer"], lr0=cfg["lr0"], patience=1000, close_mosaic=3,
              amp=True, plots=True, verbose=False, cache=False)
    if cfg.get("freeze"):
        kw["freeze"] = cfg["freeze"]
    model.train(**kw)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    for f in ("results.csv", "args.yaml", "results.png"):
        if (save_dir / f).exists():
            shutil.copy(save_dir / f, res_dir / f)
    if cfg.get("nwd"):
        print(f"[{exp}] NWD loss used in {NWD_CALLS['n']} loss calls", flush=True)

    va = pd.read_csv(val_manifest or Path(data_yaml).parent / f"manifest_val_s{VAL_STEP}.csv")
    gt = build_gt(va)
    m = load_trained(best)
    preds = predict_ul(m, va["src_img"].tolist(), va["stem"].tolist(), imgsz, cfg["conf"], cfg["nms_iou"])
    metrics = coco_eval(gt, preds)
    import cv2
    imgs = [cv2.imread(p) for p in va["src_img"].tolist()[:30]]
    metrics["ms_per_img"] = time_inference(
        lambda x: m.predict(x, imgsz=imgsz, conf=cfg["conf"], iou=cfg["nms_iou"], max_det=MAX_DET,
                            verbose=False, **_fp16()), imgs, warmup=5, n=25)
    hist = pd.read_csv(res_dir / "results.csv") if (res_dir / "results.csv").exists() else pd.DataFrame()
    metrics.update(exp=exp, budget_h=budget_h, epochs=int(len(hist)), weights=str(best), config=cfg,
                   train_minutes=float(hist["time"].iloc[-1] / 60) if "time" in hist and len(hist) else None,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   nwd_calls=NWD_CALLS["n"])
    shutil.copy(best, res_dir / "best.pt")
    save_json(metrics, res_dir / "metrics.json")
    save_json(compact_preds(preds), res_dir / "val_preds.json")

    run = wb.init(exp, {**cfg, "budget_h": budget_h})
    if run is not None and len(hist):
        hist.columns = [c.strip() for c in hist.columns]
        for _, r in hist.iterrows():
            wb.log(run, {k: float(v) for k, v in r.items() if k != "epoch"}, step=int(r["epoch"]))
    wb.log_summary(run, metrics)
    wb.log_val_images(run, va, preds, gt, n=6)
    wb.log_images(run, "plots", [p for p in [save_dir / "results.png", save_dir / "BoxPR_curve.png",
                                             save_dir / "confusion_matrix_normalized.png"] if p.exists()])
    wb.finish(run)
    print(f"[{exp}] FINAL mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} APs {metrics['AP_small']:.4f} "
          f"{metrics['ms_per_img']:.1f} ms/img", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True, help="Ultralytics data yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-h", type=float, default=None)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--cfg-json", default=None, help="JSON dict merged into the experiment config")
    a = ap.parse_args()
    import json
    train(a.exp, a.data, a.out, a.budget_h, a.val_manifest, json.loads(a.cfg_json) if a.cfg_json else None)


if __name__ == "__main__":
    main()
