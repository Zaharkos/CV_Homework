import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import roi_heads as tv_roi_heads
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import generalized_box_iou_loss

from .common import NC
from .configs import FRCNN_ANCHORS, RETINA_BASE, TV_COMMON


def _pretrained():
    return os.environ.get("VISDET_NO_PRETRAINED", "0") != "1"


_GIOU_WEIGHT = 10.0
_unit_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
_orig_fastrcnn_loss = tv_roi_heads.fastrcnn_loss


def fastrcnn_loss_giou(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    if pos.numel() == 0:
        return classification_loss, box_regression.sum() * 0
    N, _ = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    pred = box_regression[pos, labels[pos]].float()
    tgt = regression_targets[pos].float()
    ref = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=pred.device).expand(len(pos), 4)
    pb = _unit_coder.decode_single(pred, ref)
    tb = _unit_coder.decode_single(tgt, ref)
    box_loss = generalized_box_iou_loss(pb, tb, reduction="sum") / labels.numel()
    return classification_loss, _GIOU_WEIGHT * box_loss


def set_frcnn_box_loss(kind: str):
    tv_roi_heads.fastrcnn_loss = fastrcnn_loss_giou if kind == "giou" else _orig_fastrcnn_loss


def build_tv_model(cfg: dict):
    c = {**TV_COMMON, **cfg}
    pre = _pretrained()
    if c["arch"] == "frcnn":
        ag = AnchorGenerator(sizes=FRCNN_ANCHORS[c["anchors"]], aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        kw = dict(min_size=c["min_size"], max_size=c["max_size"], rpn_anchor_generator=ag,
                  box_detections_per_img=c["det_per_img"], box_score_thresh=c["score_thresh"],
                  box_batch_size_per_image=c.get("roi_batch", 512),
                  box_positive_fraction=c.get("roi_pos_fraction", 0.25),
                  rpn_pre_nms_top_n_test=c.get("rpn_pre_test", 1000),
                  rpn_post_nms_top_n_test=c.get("rpn_post_test", 1000))
        W = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pre else None
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=W, weights_backbone=None, trainable_backbone_layers=3, **kw)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NC + 1)
        set_frcnn_box_loss(c.get("box_loss", "smooth_l1"))
    elif c["arch"] == "retina":
        sizes = tuple((x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3))) for x in RETINA_BASE[c["anchors"]])
        ag = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
        kw = dict(min_size=c["min_size"], max_size=c["max_size"], anchor_generator=ag,
                  detections_per_img=c["det_per_img"], score_thresh=c["score_thresh"],
                  fg_iou_thresh=c.get("fg_iou", 0.5), bg_iou_thresh=c.get("bg_iou", 0.4),
                  topk_candidates=1000)
        W = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pre else None
        m = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=W, weights_backbone=None, trainable_backbone_layers=3, **kw)
        na = m.head.classification_head.num_anchors
        m.head.classification_head = RetinaNetClassificationHead(256, na, NC)
        m.head.regression_head._loss_type = c.get("box_loss", "l1")
    else:
        raise ValueError(c["arch"])
    return m


def label_offset(arch: str) -> int:
    return 1 if arch == "frcnn" else 0


@torch.no_grad()
def rpn_proposal_recall(model, img_paths, gt_per_img, top_ns=(100, 300, 1000), iou_thr=0.5, device="cpu"):
    from torchvision.ops import box_iou
    from .tv_train import load_rgb
    model.eval()
    hits = {n: {"small": [0, 0], "medium": [0, 0], "large": [0, 0]} for n in top_ns}
    for path, (gb, gl, _) in zip(img_paths, gt_per_img):
        if not len(gb):
            continue
        img = load_rgb(path).to(device)
        H, W = img.shape[1:]
        images, _ = model.transform([img])
        feats = model.backbone(images.tensors)
        props = model.rpn(images, feats)[0][0]
        h, w = images.image_sizes[0]
        props = props * torch.tensor([W / w, H / h, W / w, H / h], device=props.device)
        g = torch.as_tensor(gb, device=props.device)
        area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        bucket = np.where(area.cpu() < 32 ** 2, "small", np.where(area.cpu() < 96 ** 2, "medium", "large"))
        for n in top_ns:
            best = box_iou(g, props[:n]).max(1).values.cpu().numpy() if len(props) else np.zeros(len(g))
            for b, ok in zip(bucket, best >= iou_thr):
                hits[n][b][0] += int(ok); hits[n][b][1] += 1
    return {n: {b: v[0] / max(v[1], 1) for b, v in d.items()} for n, d in hits.items()}


def retina_anchor_stats(anchor_cfg, gt_boxes_list, img_hw_list, fg=0.5, bg=0.4, max_size=1280, min_size=720):
    from torchvision.ops import box_iou
    sizes = tuple((x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3))) for x in RETINA_BASE[anchor_cfg])
    ag = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    out = []
    for gb, (H, W) in zip(gt_boxes_list, img_hw_list):
        s = min(min_size / min(H, W), max_size / max(H, W))
        h, w = int(H * s), int(W * s)
        hp, wp = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        feats = [torch.zeros(1, 1, int(np.ceil(hp / st)), int(np.ceil(wp / st))) for st in (8, 16, 32, 64, 128)]

        class _IL:
            tensors = torch.zeros(1, 3, hp, wp); image_sizes = [(h, w)]
        A = ag(_IL, feats)[0]
        g = torch.as_tensor(np.asarray(gb, np.float32).reshape(-1, 4) * s)
        if len(g) == 0:
            continue
        iou = box_iou(g, A)
        best_per_anchor = iou.max(0).values
        out.append(dict(anchors=len(A), pos=int((best_per_anchor >= fg).sum()),
                        neg=int((best_per_anchor < bg).sum()), ignored=int(((best_per_anchor >= bg) & (best_per_anchor < fg)).sum()),
                        gt=len(g), gt_no_anchor=int((iou.max(1).values < fg).sum())))
    return pd.DataFrame(out)
