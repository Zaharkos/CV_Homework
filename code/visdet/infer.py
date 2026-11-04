import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms

from .common import MAX_DET, _fp16


def _empty():
    return dict(boxes=np.zeros((0, 4), np.float32), scores=np.zeros(0, np.float32), labels=np.zeros(0, int))


def _cat(parts):
    parts = [p for p in parts if len(p["scores"])]
    if not parts:
        return _empty()
    return dict(boxes=np.concatenate([p["boxes"] for p in parts]).astype(np.float32),
                scores=np.concatenate([p["scores"] for p in parts]).astype(np.float32),
                labels=np.concatenate([p["labels"] for p in parts]).astype(int))


def nms_merge(p, iou=0.6, max_det=MAX_DET):
    if not len(p["scores"]):
        return p
    k = batched_nms(torch.from_numpy(p["boxes"]), torch.from_numpy(p["scores"]),
                    torch.from_numpy(p["labels"]), iou).numpy()[:max_det]
    return dict(boxes=p["boxes"][k], scores=p["scores"][k], labels=p["labels"][k])


def wbf_merge(preds_list, W, H, iou=0.55, skip=0.001, weights=None, conf_type="avg", max_det=MAX_DET):
    from ensemble_boxes import weighted_boxes_fusion
    bl, sl, ll = [], [], []
    for p in preds_list:
        b = np.asarray(p["boxes"], np.float32).reshape(-1, 4) / np.array([W, H, W, H], np.float32)
        bl.append(b.clip(0, 1).tolist()); sl.append(np.asarray(p["scores"]).tolist()); ll.append(np.asarray(p["labels"]).tolist())
    if not any(len(s) for s in sl):
        return _empty()
    b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou, skip_box_thr=skip, conf_type=conf_type)
    o = np.argsort(-s)[:max_det]
    return dict(boxes=(b[o] * np.array([W, H, W, H])).astype(np.float32), scores=s[o].astype(np.float32),
                labels=l[o].astype(int))


def tiles(W, H, grid=(2, 2), overlap=0.2):
    nx, ny = grid
    tw, th = W / (nx - (nx - 1) * overlap), H / (ny - (ny - 1) * overlap)
    xs = np.linspace(0, W - tw, nx) if nx > 1 else [0]
    ys = np.linspace(0, H - th, ny) if ny > 1 else [0]
    return [(int(x), int(y), int(min(W, x + tw)), int(min(H, y + th))) for y in ys for x in xs]


def _ul_one(model, img, imgsz, conf, iou, augment=False):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=MAX_DET, augment=augment,
                      verbose=False, **_fp16())[0]
    b = r.boxes
    return dict(boxes=b.xyxy.cpu().numpy(), scores=b.conf.cpu().numpy(), labels=b.cls.cpu().numpy().astype(int))


def predict_strategy(model, path, strategy: dict, conf=0.001, iou=0.6):
    img = cv2.imread(str(path))
    H, W = img.shape[:2]
    parts = [_ul_one(model, img, strategy.get("imgsz", 1280), conf, iou, strategy.get("augment", False))]
    if strategy.get("tile_grid"):
        for x1, y1, x2, y2 in tiles(W, H, strategy["tile_grid"], strategy.get("overlap", 0.2)):
            p = _ul_one(model, img[y1:y2, x1:x2], strategy.get("tile_imgsz", 1280), conf, iou)
            if len(p["scores"]):
                p["boxes"] = p["boxes"] + np.array([x1, y1, x1, y1], np.float32)
                b = p["boxes"]; m = 2.0
                cut = (((b[:, 0] <= x1 + m) & (x1 > 0)) | ((b[:, 1] <= y1 + m) & (y1 > 0)) |
                       ((b[:, 2] >= x2 - m) & (x2 < W)) | ((b[:, 3] >= y2 - m) & (y2 < H)))
                p = {k: v[~cut] for k, v in p.items()}
            parts.append(p)
    if len(parts) == 1:
        return parts[0]
    if strategy.get("merge", "nms") == "wbf":
        return wbf_merge(parts, W, H, iou=0.55)
    return nms_merge(_cat(parts), iou=iou)


def predict_many(model, paths, stems, strategy, conf=0.001, iou=0.6, log_every=500):
    out = {}
    for i, (p, s) in enumerate(zip(paths, stems)):
        out[s] = predict_strategy(model, p, strategy, conf, iou)
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    return out


def ensemble(preds_by_model, sizes, weights=None, iou=0.55):
    out = {}
    for s, (W, H) in sizes.items():
        out[s] = wbf_merge([pm.get(s, _empty()) for pm in preds_by_model], W, H, iou=iou, weights=weights)
    return out


def write_submission(preds: dict, test_df: pd.DataFrame, path, max_det=MAX_DET, min_score=0.001):
    rows = []
    n_boxes = 0
    for r in test_df.itertuples():
        p = preds.get(r.image_id, _empty())
        s = np.asarray(p["scores"])
        k = np.argsort(-s)[:max_det]
        k = k[s[k] >= min_score]
        b = np.asarray(p["boxes"]).reshape(-1, 4)[k]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, r.width); b[:, [1, 3]] = b[:, [1, 3]].clip(0, r.height)
        toks = [f"{int(c)} {sc:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}"
                for c, sc, (x1, y1, x2, y2) in zip(np.asarray(p["labels"])[k], s[k], b) if x2 > x1 and y2 > y1]
        n_boxes += len(toks)
        rows.append((r.image_id, " ".join(toks)))
    sub = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    sub.to_csv(path, index=False)
    print(f"submission: {len(sub)} images, {n_boxes} boxes ({n_boxes / max(len(sub), 1):.1f}/img) -> {path}")
    return sub


def check_submission(sub: pd.DataFrame, test_df: pd.DataFrame, nc=8):
    assert list(sub.columns) == ["image_id", "PredictionString"]
    assert len(sub) == len(test_df) and set(sub["image_id"]) == set(test_df["image_id"]), "image ids mismatch"
    sizes = test_df.set_index("image_id")[["width", "height"]]
    for iid, ps in sub.sample(min(200, len(sub)), random_state=0).itertuples(index=False):
        if not isinstance(ps, str) or not ps.strip():
            continue
        v = np.array(ps.split(), float).reshape(-1, 6)
        W, H = sizes.loc[iid]
        assert ((v[:, 0] >= 0) & (v[:, 0] < nc)).all() and ((v[:, 1] >= 0) & (v[:, 1] <= 1)).all()
        assert (v[:, 2] >= 0).all() and (v[:, 2] + v[:, 4] <= W + 1).all() and (v[:, 3] + v[:, 5] <= H + 1).all()
    print("submission format OK")
