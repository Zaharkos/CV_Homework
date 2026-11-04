import cv2
import numpy as np
import pandas as pd

from .common import CLASSES, LONG_SIDE
from .prep import clip_sizes, read_ignore, read_yolo


def load_boxes(frames: pd.DataFrame) -> tuple:
    sizes = clip_sizes(frames)
    rows, frows = [], []
    for r in frames.itertuples():
        W, H = sizes[r.clip]
        lab = read_yolo(r.src_label)
        ign = read_ignore(r.src_ignore)
        for c, cx, cy, w, h in lab:
            rows.append((r.split, r.clip, r.stem, int(c), cx * W, cy * H, w * W, h * H, W, H))
        ign_area = float((ign[:, 2] * ign[:, 3]).sum()) if len(ign) else 0.0
        frows.append((r.split, r.clip, r.stem, r.order, len(lab), len(ign), ign_area, W, H))
    b = pd.DataFrame(rows, columns=["split", "clip", "stem", "cls", "cx", "cy", "w", "h", "W", "H"])
    b["class"] = b["cls"].map(dict(enumerate(CLASSES)))
    b["area"] = b["w"] * b["h"]
    b["sqrt_area"] = np.sqrt(b["area"])
    b["in_scale"] = LONG_SIDE / b[["W", "H"]].max(1)
    b["w_in"], b["h_in"] = b["w"] * b["in_scale"], b["h"] * b["in_scale"]
    b["sqrt_area_in"] = b["sqrt_area"] * b["in_scale"]
    b["aspect"] = b["w"] / b["h"].clip(lower=1e-3)
    b["size_bin"] = pd.cut(b["area"], [0, 32 ** 2, 96 ** 2, 1e12], labels=["small", "medium", "large"], right=False)
    f = pd.DataFrame(frows, columns=["split", "clip", "stem", "order", "n_obj", "n_ignore", "ignore_frac", "W", "H"])
    return b, f


def anchor_shapes(sizes, ratios=(0.5, 1.0, 2.0)):
    out = []
    for s in sizes:
        for r in ratios:
            hr = np.sqrt(r); out.append((s / hr, s * hr))
    return np.array(out, np.float32)


def best_anchor_iou(wh, anchors):
    w, h = wh[:, None, 0], wh[:, None, 1]
    aw, ah = anchors[None, :, 0], anchors[None, :, 1]
    inter = np.minimum(w, aw) * np.minimum(h, ah)
    return (inter / (w * h + aw * ah - inter)).max(1)


def kmeans_iou(wh, k=9, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = wh[rng.choice(len(wh), k, replace=False)].copy()
    for _ in range(iters):
        inter = np.minimum(wh[:, None, 0], C[None, :, 0]) * np.minimum(wh[:, None, 1], C[None, :, 1])
        iou = inter / (wh[:, None].prod(-1) + C[None].prod(-1) - inter)
        a = iou.argmax(1)
        newC = np.array([np.median(wh[a == j], 0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C[np.argsort(C.prod(1))]


def frame_similarity(frames: pd.DataFrame, clips, gaps=(1, 3, 5, 9, 15, 30), n_per_clip=20, size=320):
    res = []
    for clip in clips:
        g = frames[frames["clip"] == clip].sort_values("order")
        paths = g["src_img"].tolist()
        starts = np.linspace(0, max(len(paths) - max(gaps) - 1, 0), n_per_clip).astype(int)
        cache = {}

        def im(i):
            if i not in cache:
                x = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
                cache[i] = cv2.resize(x, (size, int(size * x.shape[0] / x.shape[1]))).astype(np.float32)
            return cache[i]
        for gap in gaps:
            d = [np.abs(im(s) - im(s + gap)).mean() for s in starts if s + gap < len(paths)]
            res.append(dict(clip=clip, gap=gap, mad=float(np.mean(d)) if d else np.nan))
    return pd.DataFrame(res)
