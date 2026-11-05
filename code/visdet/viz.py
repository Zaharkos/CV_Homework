import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .common import CLASSES
from .evaluate import _iou

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
           (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def draw_boxes(img, boxes, labels=None, color=None, thick=2, scores=None):
    img = img.copy()
    for i, b in enumerate(np.asarray(boxes).reshape(-1, 4)):
        c = color or PALETTE[int(labels[i]) % len(PALETTE)]
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    return img


def draw_regions(img, regions, alpha=0.35):
    over = img.copy()
    for x1, y1, x2, y2 in np.asarray(regions).reshape(-1, 4).round().astype(int):
        cv2.rectangle(over, (x1, y1), (x2, y2), (128, 128, 128), -1)
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


def legend(ax):
    for i, c in enumerate(CLASSES):
        ax.plot([], [], color=np.array(PALETTE[i]) / 255, lw=3, label=c)
    ax.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.8)


def error_overlay(img, gt_boxes, gt_labels, pb, ps, pl, score_thr=0.3, iou_thr=0.5):
    keep = ps >= score_thr
    pb, ps, pl = pb[keep], ps[keep], pl[keep]
    ious = _iou(pb, gt_boxes)
    matched = np.zeros(len(gt_boxes), bool)
    out = img.copy()
    for i in np.argsort(-ps):
        col = (255, 0, 0)
        if len(gt_boxes):
            same = np.where(gt_labels == pl[i], ious[i], 0)
            j = int(np.argmax(same))
            if same[j] >= iou_thr and not matched[j]:
                matched[j] = True; col = (0, 200, 0)
            elif ious[i].max() >= iou_thr:
                col = (255, 0, 255)
        out = draw_boxes(out, pb[i:i + 1], color=col, thick=2)
    out = draw_boxes(out, gt_boxes[~matched], color=(255, 220, 0), thick=1)
    return out, int((~matched).sum())


def crop_zoom(img, box, pad=40, size=220):
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img.shape[:2]
    c = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST) if c.size else np.zeros((size, size, 3), np.uint8)


class FRCNNGradCAM:

    def __init__(self, model):
        self.model = model.eval()
        self.acts = {}
        self.hooks = [model.backbone.register_forward_hook(self._fpn_hook)]
        for i, name in enumerate(["layer1", "layer2", "layer3", "layer4"]):
            self.hooks.append(getattr(model.backbone.body, name).register_forward_hook(self._mk(f"C{i + 2}")))

    def _fpn_hook(self, m, inp, out):
        for k, v in out.items():
            if k in ("0", "1", "2", "3"):
                if v.requires_grad:
                    v.retain_grad()
                self.acts[f"P{int(k) + 2}"] = v

    def _mk(self, name):
        def h(m, inp, out):
            if out.requires_grad:
                out.retain_grad()
            self.acts[name] = out
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()

    def __call__(self, img_tensor, det_index=None, box=None):
        self.acts = {}
        self.model.zero_grad(set_to_none=True)
        img_tensor = img_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model([img_tensor])[0]
            if det_index is None:
                ious = _iou(np.asarray([box], np.float32), out["boxes"].detach().cpu().numpy())[0]
                det_index = int(np.argmax(ious))
            out["scores"][det_index].backward()
        H, W = img_tensor.shape[1:]
        cams = {}
        for k, a in self.acts.items():
            g = a.grad
            if g is None:
                cams[k] = np.zeros((H, W), np.float32); continue
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * a).sum(1, keepdim=True))[0, 0].detach()
            ch, cw = self._valid_hw(H, W, a)
            cam = F.interpolate(cam[:ch, :cw][None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cams[k] = cam / cam.max() if cam.max() > 0 else cam
        det = {k: v.detach().cpu().numpy() for k, v in out.items()}
        return det, cams, det_index

    def _valid_hw(self, H, W, a):
        t = self.model.transform
        s = min(t.min_size[-1] / min(H, W), t.max_size / max(H, W))
        h, w = int(H * s), int(W * s)
        Hp, Wp = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        fh, fw = a.shape[2], a.shape[3]
        return max(1, round(fh * h / Hp)), max(1, round(fw * w / Wp))


def cam_overlay(img, cam, alpha=0.45):
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (img * (1 - alpha) + hm * alpha).astype(np.uint8)


@torch.no_grad()
def roi_embeddings(model, img_tensor, boxes):
    model.eval()
    images, _ = model.transform([img_tensor])
    feats = model.backbone(images.tensors)
    H, W = img_tensor.shape[1:]
    h, w = images.image_sizes[0]
    b = torch.as_tensor(boxes, dtype=torch.float32, device=img_tensor.device).reshape(-1, 4)
    b = b * torch.tensor([w / W, h / H, w / W, h / H], device=b.device)
    pooled = model.roi_heads.box_roi_pool(feats, [b], images.image_sizes)
    emb = model.roi_heads.box_head(pooled)
    logits, _ = model.roi_heads.box_predictor(emb)
    return emb.cpu().numpy(), logits.softmax(-1).cpu().numpy()


class RTDETRAttention:

    def __init__(self):
        import ultralytics.nn.modules.transformer as T
        self.T = T
        self.orig = T.multi_scale_deformable_attn_pytorch
        self.records = []

        def rec(value, value_shapes, sampling_locations, attention_weights):
            nl = len(value_shapes)
            loc = sampling_locations.detach().float().cpu()
            w = attention_weights.detach().float().cpu()
            if loc.dim() == 5:
                loc = loc.reshape(*loc.shape[:3], nl, -1, 2)
            if w.dim() == 4:
                w = w.reshape(*w.shape[:3], nl, -1)
            self.records.append(dict(loc=loc, w=w, shapes=[tuple(int(x) for x in s) for s in value_shapes]))
            return self.orig(value, value_shapes, sampling_locations, attention_weights)
        T.multi_scale_deformable_attn_pytorch = rec

    def remove(self):
        self.T.multi_scale_deformable_attn_pytorch = self.orig

    @torch.no_grad()
    def run(self, net, img_rgb, imgsz):
        self.records = []
        H, W = img_rgb.shape[:2]
        x = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        p = next(net.parameters())
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(p.device, p.dtype) / 255
        out = net(x)
        raw = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else out
        bb, sc = raw[0], raw[1]
        bb = bb.reshape(-1, bb.shape[-2], 4)[-1].float().cpu()
        sc = sc.reshape(-1, sc.shape[-2], sc.shape[-1])[-1].float().sigmoid().cpu()
        s, l = sc.max(-1)
        cx, cy, w, h = bb.unbind(-1)
        boxes = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)
        return boxes.numpy(), s.numpy(), l.numpy(), self.records
