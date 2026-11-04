import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import CLASSES, LONG_SIDE, is_debug

GRAY = (114, 114, 114)


def read_yolo(path) -> np.ndarray:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 5), np.float32)
    a = np.loadtxt(path, ndmin=2, dtype=np.float32)
    return a.reshape(-1, 5) if a.size else np.zeros((0, 5), np.float32)


def read_ignore(path) -> np.ndarray:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 4), np.float32)
    a = np.loadtxt(path, ndmin=2, dtype=np.float32)
    return a.reshape(-1, 4) if a.size else np.zeros((0, 4), np.float32)


def cxcywh_to_xyxy(a, W, H):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    x1 = (a[:, 0] - a[:, 2] / 2) * W
    y1 = (a[:, 1] - a[:, 3] / 2) * H
    x2 = (a[:, 0] + a[:, 2] / 2) * W
    y2 = (a[:, 1] + a[:, 3] / 2) * H
    return np.stack([x1, y1, x2, y2], 1)


def list_frames(comp_root, split) -> pd.DataFrame:
    comp_root = Path(comp_root)
    img_root = comp_root / split / "images"
    rows = []
    for clip_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        frames = sorted(clip_dir.glob("*.jpg"))
        for i, f in enumerate(frames):
            rows.append(dict(
                split=split, clip=clip_dir.name, order=i, frame=f.stem,
                stem=f"{clip_dir.name}_{f.stem}", src_img=str(f),
                src_label=str(comp_root / split / "labels" / clip_dir.name / f"{f.stem}.txt"),
                src_ignore=str(comp_root / split / "ignore" / clip_dir.name / f"{f.stem}.txt")))
    return pd.DataFrame(rows)


def clip_sizes(frames: pd.DataFrame) -> dict:
    from PIL import Image
    out = {}
    for clip, g in frames.groupby("clip"):
        with Image.open(g.iloc[0]["src_img"]) as im:
            out[clip] = im.size
    return out


def subsample(frames: pd.DataFrame, step: int) -> pd.DataFrame:
    sel = frames[frames["order"] % step == 0]
    if is_debug():
        clips = sorted(sel["clip"].unique())[:3]
        sel = sel[sel["clip"].isin(clips)].groupby("clip").head(4)
    return sel.reset_index(drop=True)


def box_inside_fraction(boxes, regions):
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), np.float32)
    ix1 = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    iy1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    ix2 = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    iy2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)
    return (inter / area[:, None]).max(1)


def _export_one(row, out_root, long_side, mask):
    img = cv2.imread(row["src_img"])
    H, W = img.shape[:2]
    lab = read_yolo(row["src_label"])
    ign = read_ignore(row["src_ignore"])
    n_dropped = 0
    if len(ign):
        reg = cxcywh_to_xyxy(ign, W, H)
        if mask:
            for x1, y1, x2, y2 in reg.round().astype(int):
                img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = GRAY
        if len(lab):
            keep = box_inside_fraction(cxcywh_to_xyxy(lab[:, 1:], W, H), reg) < 0.5
            n_dropped = int((~keep).sum())
            lab = lab[keep]
    s = min(1.0, long_side / max(H, W))
    if s < 1.0:
        img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    dst_img = Path(out_root) / "images" / row["split"] / f"{row['stem']}.jpg"
    dst_lab = Path(out_root) / "labels" / row["split"] / f"{row['stem']}.txt"
    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if len(lab):
        lab[:, 1:] = np.clip(lab[:, 1:], 0, 1)
    with open(dst_lab, "w") as f:
        for c, cx, cy, bw, bh in lab:
            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return dict(stem=row["stem"], img=str(dst_img), label=str(dst_lab), orig_w=W, orig_h=H,
                w=w, h=h, scale=w / W, n_boxes=len(lab), n_ignore=len(ign), n_dropped=n_dropped)


def build_split(comp_root, split, step, out_root, long_side=LONG_SIDE, mask=True, workers=None):
    out_root = Path(out_root)
    man_path = out_root / f"manifest_{split}_s{step}.csv"
    if man_path.exists():
        return pd.read_csv(man_path)
    (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    sel = subsample(list_frames(comp_root, split), step)
    fn = partial(_export_one, out_root=out_root, long_side=long_side, mask=mask)
    with Pool(workers or os.cpu_count()) as p:
        res = p.map(fn, sel.to_dict("records"), chunksize=8)
    man = sel.merge(pd.DataFrame(res), on="stem")
    man.to_csv(man_path, index=False)
    with open(out_root / f"{split}_s{step}.txt", "w") as f:
        f.write("\n".join(man["img"]) + "\n")
    return man


def write_yolo_yaml(out_root, train_lists, val_list, name):
    out_root = Path(out_root)
    lines = [f"path: {out_root}", "train:"] + [f"  - {t}" for t in train_lists]
    lines += [f"val: {val_list}", "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    p = out_root / name
    p.write_text("\n".join(lines) + "\n")
    return p


def prepare_data(comp_root, out_root, train_step, val_step, include_val_in_train=False):
    tr = build_split(comp_root, "train", train_step, out_root)
    va = build_split(comp_root, "val", val_step, out_root)
    lists = [f"train_s{train_step}.txt"]
    if include_val_in_train:
        lists.append(f"val_s{val_step}.txt")
    name = f"data_s{train_step}{'_trval' if include_val_in_train else ''}.yaml"
    y = write_yolo_yaml(out_root, lists, f"val_s{val_step}.txt", name)
    return tr, va, y
