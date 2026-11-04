BUDGET_H = 0.75

FRCNN_ANCHORS = {
    "default": ((32,), (64,), (128,), (256,), (512,)),
    "small":   ((16,), (32,), (64,), (128,), (256,)),
}
RETINA_BASE = {"default": [32, 64, 128, 256, 512], "small": [16, 32, 64, 128, 256]}

TV_COMMON = dict(batch=4, workers=2, min_size=720, max_size=1280, lr=0.01, momentum=0.9,
                 weight_decay=1e-4, warmup_iters=300, max_epochs=40, clip_grad=10.0, hflip=0.5,
                 det_per_img=300, score_thresh=0.001)

TV_EXPS = {
    "frcnn_E0_baseline":   dict(arch="frcnn", anchors="default"),
    "frcnn_E1_anchors":    dict(arch="frcnn", anchors="small"),
    "frcnn_E2_roi_pos50":  dict(arch="frcnn", anchors="small", roi_batch=512, roi_pos_fraction=0.5),
    "frcnn_E3_giou":       dict(arch="frcnn", anchors="small", box_loss="giou"),
    "retina_E0_baseline":  dict(arch="retina", anchors="default", lr=0.005),
    "retina_E1_anchors":   dict(arch="retina", anchors="small", lr=0.005),
    "retina_E2_iou_thr":   dict(arch="retina", anchors="small", lr=0.005, fg_iou=0.4, bg_iou=0.3),
    "retina_E3_giou":      dict(arch="retina", anchors="small", lr=0.005, box_loss="giou"),
}

UL_COMMON = dict(imgsz=1280, workers=2, seed=0, conf=0.001, nms_iou=0.6)
UL_OPTIM = {"yolo": dict(optimizer="AdamW", lr0=0.001), "rtdetr": dict(optimizer="AdamW", lr0=1e-4)}

UL_EXPS = {
    "yolo_E0_baseline": dict(family="yolo", model="yolo11s", batch=8),
    "yolo_E1_p2":       dict(family="yolo", model="yolo11s", p2=True, batch=6),
    "yolo_E2_nwd":      dict(family="yolo", model="yolo11s", nwd=True, batch=8),
    "yolo_E3_p2_nwd":   dict(family="yolo", model="yolo11s", p2=True, nwd=True, batch=6),
    "rtdetr_E0_baseline": dict(family="rtdetr", model="rtdetr-l", batch=4),
    "rtdetr_E1_q500":     dict(family="rtdetr", model="rtdetr-l", nq=500, batch=4),
    "rtdetr_E2_freeze":   dict(family="rtdetr", model="rtdetr-l", nq=500, freeze=10, batch=4),
    "rtdetr_E3_lr3x":     dict(family="rtdetr", model="rtdetr-l", nq=500, batch=4, lr0=3e-4),
}

NWD_ALPHA = 0.5
NWD_C = 2.0

FINAL_EXPS = {
    "final_A_yolo11s": dict(family="yolo", model="yolo11s", batch=8),
    "final_B_yolo11m": dict(family="yolo", model="yolo11m", batch=6),
}
FINAL_BUDGET_H = 3.0
