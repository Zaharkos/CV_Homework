import contextlib
import io

import numpy as np
import pandas as pd

from .common import CLASSES, NC
from .prep import cxcywh_to_xyxy, read_ignore, read_yolo

SIZE_BINS = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def build_gt(val_manifest: pd.DataFrame) -> dict:
    images, anns = [], []
    aid = 1
    for i, r in enumerate(val_manifest.itertuples()):
        W, H = int(r.orig_w), int(r.orig_h)
        images.append(dict(id=i, file_name=r.stem, width=W, height=H))
        lab = read_yolo(r.src_label)
        for c, *box in lab:
            x1, y1, x2, y2 = cxcywh_to_xyxy(box, W, H)[0]
            w, h = x2 - x1, y2 - y1
            anns.append(dict(id=aid, image_id=i, category_id=int(c) + 1, bbox=[float(x1), float(y1), float(w), float(h)],
                             area=float(w * h), iscrowd=0))
            aid += 1
        for x1, y1, x2, y2 in cxcywh_to_xyxy(read_ignore(r.src_ignore), W, H):
            w, h = x2 - x1, y2 - y1
            for c in range(NC):
                anns.append(dict(id=aid, image_id=i, category_id=c + 1, bbox=[float(x1), float(y1), float(w), float(h)],
                                 area=float(w * h), iscrowd=1))
                aid += 1
    cats = [dict(id=i + 1, name=n) for i, n in enumerate(CLASSES)]
    return dict(images=images, annotations=anns, categories=cats)


def preds_to_coco(preds: dict, gt: dict) -> list:
    sid = {im["file_name"]: im["id"] for im in gt["images"]}
    out = []
    for stem, p in preds.items():
        if stem not in sid:
            continue
        b = np.asarray(p["boxes"], np.float32).reshape(-1, 4)
        for (x1, y1, x2, y2), s, c in zip(b, p["scores"], p["labels"]):
            out.append(dict(image_id=sid[stem], category_id=int(c) + 1,
                            bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(s)))
    return out


def coco_eval(gt: dict, preds: dict, max_dets=500, per_class=True) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dets = preds_to_coco(preds, gt)
    keys = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR100", f"AR{max_dets}", "AR_small", "AR_medium", "AR_large"]
    if not dets:
        return {k: 0.0 for k in keys}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, "bbox")
        E.params.maxDets = [1, 100, max_dets]
        E.evaluate()
        E.accumulate()
        E.summarize()
    m = {k: float(v) for k, v in zip(keys, E.stats)}
    if per_class:
        prec = E.eval["precision"]
        for k, name in enumerate(CLASSES):
            p = prec[:, :, k, 0, -1]
            p = p[p > -1]
            m[f"AP_{name}"] = float(p.mean()) if p.size else float("nan")
    return m


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None] - inter, 1e-6, None)


def gt_per_image(gt: dict):
    by = {im["id"]: ([], [], []) for im in gt["images"]}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        if a["iscrowd"]:
            if a["category_id"] == 1:
                by[a["image_id"]][2].append([x, y, x + w, y + h])
        else:
            by[a["image_id"]][0].append([x, y, x + w, y + h])
            by[a["image_id"]][1].append(a["category_id"] - 1)
    out = {}
    for im in gt["images"]:
        b, l, ig = by[im["id"]]
        out[im["file_name"]] = (np.array(b, np.float32).reshape(-1, 4), np.array(l, int),
                                np.array(ig, np.float32).reshape(-1, 4))
    return out


def error_breakdown(gt: dict, preds: dict, score_thr=0.3, iou_thr=0.5):
    G = gt_per_image(gt)
    counts = dict(TP=0, Dup=0, Cls=0, Loc=0, Both=0, Bkg=0, Ignored=0)
    missed = {k: 0 for k in SIZE_BINS}; total = {k: 0 for k in SIZE_BINS}
    confusion = np.zeros((NC, NC), int)
    rows = []
    for stem, (gb, gl, ig) in G.items():
        p = preds.get(stem, dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), labels=np.zeros(0)))
        keep = np.asarray(p["scores"]) >= score_thr
        pb = np.asarray(p["boxes"], np.float32).reshape(-1, 4)[keep]
        ps = np.asarray(p["scores"])[keep]; pl = np.asarray(p["labels"]).astype(int)[keep]
        order = np.argsort(-ps); pb, pl = pb[order], pl[order]
        ious = _iou(pb, gb)
        matched = np.zeros(len(gb), bool)
        img_counts = dict(TP=0, Dup=0, Cls=0, Loc=0, Both=0, Bkg=0, Ignored=0)
        for i in range(len(pb)):
            same = gl == pl[i]
            iou_same = np.where(same, ious[i], 0) if len(gb) else np.zeros(0)
            iou_any = ious[i] if len(gb) else np.zeros(0)
            if len(gb) and iou_same.max(initial=0) >= iou_thr:
                cand = np.where((iou_same >= iou_thr) & ~matched)[0]
                if len(cand):
                    j = cand[np.argmax(iou_same[cand])]; matched[j] = True; t = "TP"
                else:
                    t = "Dup"
            elif len(gb) and iou_any.max(initial=0) >= iou_thr:
                t = "Cls"; confusion[gl[np.argmax(iou_any)], pl[i]] += 1
            elif len(gb) and iou_same.max(initial=0) >= 0.1:
                t = "Loc"
            elif len(gb) and iou_any.max(initial=0) >= 0.1:
                t = "Both"
            else:
                inside = False
                if len(ig):
                    b = pb[i]
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    inside = bool(((ig[:, 0] <= cx) & (cx <= ig[:, 2]) & (ig[:, 1] <= cy) & (cy <= ig[:, 3])).any())
                t = "Ignored" if inside else "Bkg"
            img_counts[t] += 1
        area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) if len(gb) else np.zeros(0)
        for k, (lo, hi) in SIZE_BINS.items():
            sel = (area >= lo) & (area < hi)
            total[k] += int(sel.sum()); missed[k] += int((sel & ~matched).sum())
        for k in counts:
            counts[k] += img_counts[k]
        rows.append(dict(stem=stem, n_gt=len(gb), n_missed=int((~matched).sum()), **img_counts))
    summary = dict(counts=counts, missed_by_size=missed, gt_by_size=total,
                   recall_by_size={k: (1 - missed[k] / total[k]) if total[k] else float("nan") for k in SIZE_BINS},
                   confusion=confusion.tolist())
    return summary, pd.DataFrame(rows)


def time_inference(fn, inputs, warmup=5, n=50):
    import time
    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    inputs = list(inputs)
    for x in inputs[:warmup]:
        fn(x)
    sync()
    xs = inputs[:n]
    t0 = time.perf_counter()
    for x in xs:
        fn(x)
    sync()
    return 1000 * (time.perf_counter() - t0) / max(len(xs), 1)
