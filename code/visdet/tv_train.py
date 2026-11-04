import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .common import compact_preds, MAX_DET, SEED, TRAIN_STEP, VAL_STEP, is_debug, save_json, seed_everything
from .configs import BUDGET_H, TV_COMMON, TV_EXPS
from .evaluate import build_gt, coco_eval, time_inference
from .prep import read_yolo
from .tv_models import build_tv_model, label_offset
from . import wb


class DetDataset(Dataset):

    def __init__(self, manifest: pd.DataFrame, offset: int, hflip=0.0):
        self.rows = manifest.to_dict("records")
        self.offset, self.hflip = offset, hflip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(r["img"]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lab = read_yolo(r["label"])
        b = np.zeros((0, 4), np.float32)
        cls = np.zeros(0, np.int64)
        if len(lab):
            cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
            b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).clip(0, [w, h, w, h])
            ok = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
            b, cls = b[ok], lab[ok, 0].astype(np.int64)
        if self.hflip and np.random.rand() < self.hflip:
            img = img[:, ::-1].copy()
            b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255)
        target = dict(boxes=torch.from_numpy(b).float(), labels=torch.from_numpy(cls + self.offset))
        return t, target


def collate(batch):
    return tuple(zip(*batch))


def load_rgb(path):
    return torch.from_numpy(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)


@torch.no_grad()
def predict_tv(model, paths, stems, offset, device, batch=4):
    model.eval()
    out = {}
    for i in range(0, len(paths), batch):
        imgs = [load_rgb(p).to(device) for p in paths[i:i + batch]]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            res = model(imgs)
        for s, r in zip(stems[i:i + batch], res):
            k = r["scores"].float().argsort(descending=True)[:MAX_DET]
            out[s] = dict(boxes=r["boxes"][k].float().cpu().numpy(), scores=r["scores"][k].float().cpu().numpy(),
                          labels=(r["labels"][k] - offset).cpu().numpy())
    return out


def train(exp, data_root, out_root, budget_h=None, batch=None):
    cfg = {**TV_COMMON, **TV_EXPS[exp]}
    if batch:
        cfg["batch"] = batch
    budget_h = budget_h if budget_h is not None else BUDGET_H
    if is_debug():
        budget_h, cfg["max_epochs"], cfg["warmup_iters"] = 0.02, 2, 5
        cfg["min_size"], cfg["max_size"] = 320, 512
    seed_everything(SEED)
    out = Path(out_root) / exp
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)
    tr = pd.read_csv(data_root / f"manifest_train_s{TRAIN_STEP}.csv")
    va = pd.read_csv(data_root / f"manifest_val_s{VAL_STEP}.csv")
    gt = build_gt(va)
    off = label_offset(cfg["arch"])

    model = build_tv_model(cfg).to(device)
    ds = DetDataset(tr, off, hflip=cfg["hflip"])
    g = torch.Generator(); g.manual_seed(SEED)
    dl = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=cfg["workers"],
                    collate_fn=collate, drop_last=True, generator=g, persistent_workers=cfg["workers"] > 0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run = wb.init(exp, {**cfg, "budget_h": budget_h, "train_frames": len(tr), "val_frames": len(va)})

    budget_s = budget_h * 3600
    total_iters = None
    calib_start, calib_n = (1, 2) if is_debug() else (10, 40)
    it, t_train, t_cal, hist = 0, 0.0, 0.0, []
    warm = cfg["warmup_iters"]

    def lr_at(i):
        if i < warm:
            return cfg["lr"] * (0.001 + 0.999 * i / warm)
        T = total_iters or 10 ** 9
        p = min(1.0, (i - warm) / max(T - warm, 1))
        return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))

    done = False
    for epoch in range(cfg["max_epochs"]):
        model.train()
        losses = []
        t_prev = time.perf_counter()
        for imgs, tgts in dl:
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            for gp in opt.param_groups:
                gp["lr"] = lr_at(it)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                ld = model(imgs, tgts)
                loss = sum(ld.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at iter {it}: { {k: v.item() for k, v in ld.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["clip_grad"])
            scaler.step(opt)
            scaler.update()
            now = time.perf_counter()
            t_train += now - t_prev
            t_prev = now
            it += 1
            losses.append({k: v.item() for k, v in ld.items()})
            if it == calib_start:
                t_cal = t_train
            if total_iters is None and it == calib_start + calib_n:
                sec_it = (t_train - t_cal) / calib_n
                total_iters = min(max(int(budget_s / sec_it), it + 1), cfg["max_epochs"] * len(dl))
                print(f"[{exp}] {sec_it:.3f}s/iter -> {total_iters} iters "
                      f"(~{total_iters / len(dl):.1f} epochs) in {budget_h}h", flush=True)
            if it % 50 == 0:
                print(f"[{exp}] ep {epoch} it {it}/{total_iters} loss {loss.item():.3f} "
                      f"lr {lr_at(it):.5f} t {t_train / 60:.1f}m", flush=True)
                wb.log(run, {"train/loss": loss.item(), **{f"train/{k}": v.item() for k, v in ld.items()},
                             "train/lr": lr_at(it), "train/minutes": t_train / 60}, step=it)
            if (total_iters and it >= total_iters) or t_train > budget_s * 1.15:
                done = True
                break
        preds = predict_tv(model, va["src_img"].tolist(), va["stem"].tolist(), off, device)
        m = coco_eval(gt, preds, per_class=False)
        mean_l = pd.DataFrame(losses).mean().to_dict() if losses else {}
        row = dict(epoch=epoch + 1, iter=it, train_minutes=t_train / 60, loss=sum(mean_l.values()),
                   mAP=m["mAP"], AP50=m["AP50"], AP_small=m["AP_small"])
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        wb.log(run, {f"val/{k}": v for k, v in row.items() if k not in ("epoch", "iter")}, step=it)
        print(f"[{exp}] epoch {epoch + 1}: mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} APs {m['AP_small']:.4f}", flush=True)
        if done:
            break

    torch.save(model.state_dict(), out / "model_final.pth")
    preds = predict_tv(model, va["src_img"].tolist(), va["stem"].tolist(), off, device)
    metrics = coco_eval(gt, preds)
    imgs = [load_rgb(p).to(device) for p in va["src_img"].tolist()[:30]]
    model.eval()

    def _f(x):
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            model([x])
    metrics["ms_per_img"] = time_inference(_f, imgs, warmup=5, n=25)
    metrics.update(exp=exp, train_minutes=t_train / 60, iters=it, epochs=len(hist), budget_h=budget_h,
                   config=cfg, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    save_json(metrics, out / "metrics.json")
    save_json(compact_preds(preds), out / "val_preds.json")
    wb.log_summary(run, metrics)
    wb.log_val_images(run, va, preds, gt, n=6)
    wb.finish(run)
    print(f"[{exp}] FINAL mAP {metrics['mAP']:.4f} AP50 {metrics['AP50']:.4f} APs {metrics['AP_small']:.4f} "
          f"{metrics['ms_per_img']:.1f} ms/img", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-h", type=float, default=None)
    a = ap.parse_args()
    batch = {**TV_COMMON, **TV_EXPS[a.exp]}["batch"]
    while True:
        try:
            train(a.exp, a.data, a.out, a.budget_h, batch)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                break
            batch //= 2
            print(f"[{a.exp}] CUDA OOM -> retry with batch {batch}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
