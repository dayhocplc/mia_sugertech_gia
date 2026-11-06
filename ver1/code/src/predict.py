#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Một giao diện suy luận chung cho cả năm mô hình.

`build_predictor` trả về một hàm nhận ảnh BGR (numpy) và trả về
`Prediction(boxes_xyxy, classes, scores, masks)` — nhờ vậy evaluate.py và
make_figures.py không cần biết mô hình chạy trên Detectron2 hay Ultralytics.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import RUNS, load_cfg


@dataclasses.dataclass
class Prediction:
    boxes: np.ndarray            # (N, 4) xyxy, toạ độ theo ảnh gốc
    classes: np.ndarray          # (N,) 0 = sugarcane, 1 = weed
    scores: np.ndarray           # (N,)
    masks: np.ndarray | None = None   # (N, H, W) bool, chỉ mô hình segment


def default_weights(model_key: str, framework: str) -> pathlib.Path:
    """Checkpoint dùng để đánh giá: EPOCH CUỐI cho mọi mô hình.

    Trước đây nhánh Detectron2 lấy model_final.pth (epoch cuối) còn nhánh
    Ultralytics lấy weights/best.pt (tốt nhất trên validation). Bất đối xứng đó
    tặng YOLOv10 khoảng +2,5 AP50 và còn làm phương sai theo seed của nó bị
    phóng đại sáu lần — phần lớn "độ bất ổn huấn luyện" của YOLOv10 chỉ là hệ
    quả của việc chọn checkpoint. Ngân sách huấn luyện cố định 50 epoch, không
    dừng sớm, nên epoch cuối là lựa chọn đối xứng và không cần giả định gì thêm.
    """
    if framework == "detectron2":
        return RUNS / model_key / "model_final.pth"
    return RUNS / model_key / "weights/last.pt"


def _detectron2_predictor(model_key: str, cfg: dict, weights: pathlib.Path, conf: float,
                          run_dir: pathlib.Path | None = None):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    run_cfg = (run_dir or (RUNS / model_key)) / "config.yaml"
    d = get_cfg()
    d.set_new_allowed(True)
    if run_cfg.exists():
        d.merge_from_file(str(run_cfg))
    else:                                   # chưa huấn luyện: dựng từ model zoo
        from detectron2 import model_zoo
        spec = cfg["models"][model_key]
        d.merge_from_file(model_zoo.get_config_file(spec["config"]))
        d.MODEL.ROI_HEADS.NUM_CLASSES = len(cfg["dataset"]["classes"])
        d.MODEL.RETINANET.NUM_CLASSES = len(cfg["dataset"]["classes"])
        d.INPUT.MIN_SIZE_TEST = d.INPUT.MAX_SIZE_TEST = spec["imgsz"]
    d.MODEL.WEIGHTS = str(weights)
    # Ngưỡng ở đây đặt thấp; lọc theo conf_threshold của bài báo làm ở evaluate.py,
    # để cùng một lần suy luận có thể tính lại ở nhiều ngưỡng khác nhau.
    d.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(conf, 0.05)
    d.MODEL.RETINANET.SCORE_THRESH_TEST = min(conf, 0.05)
    predictor = DefaultPredictor(d)

    def run(img_bgr: np.ndarray) -> Prediction:
        inst = predictor(img_bgr)["instances"].to("cpu")
        masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        return Prediction(
            boxes=inst.pred_boxes.tensor.numpy().astype(float),
            classes=inst.pred_classes.numpy().astype(int),
            scores=inst.scores.numpy().astype(float),
            masks=masks,
        )

    return run


def _ultralytics_predictor(model_key: str, cfg: dict, weights: pathlib.Path, conf: float):
    from ultralytics import YOLO

    spec = cfg["models"][model_key]
    model = YOLO(str(weights))

    def run(img_bgr: np.ndarray) -> Prediction:
        r = model.predict(img_bgr, imgsz=spec["imgsz"], conf=min(conf, 0.05),
                          verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Prediction(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
        masks = None
        if getattr(r, "masks", None) is not None:
            masks = r.masks.data.cpu().numpy().astype(bool)
        return Prediction(
            boxes=r.boxes.xyxy.cpu().numpy().astype(float),
            classes=r.boxes.cls.cpu().numpy().astype(int),
            scores=r.boxes.conf.cpu().numpy().astype(float),
            masks=masks,
        )

    return run


def build_predictor(model_key: str, cfg: dict | None = None,
                    weights: str | pathlib.Path | None = None,
                    conf: float = 0.05, run_dir: str | pathlib.Path | None = None):
    """run_dir: thư mục của một lượt chạy khác (vd runs/retinanet_seed1)."""
    cfg = cfg or load_cfg()
    framework = cfg["models"][model_key]["framework"]
    rd = pathlib.Path(run_dir) if run_dir else None
    if weights:
        w = pathlib.Path(weights)
    elif rd:
        w = rd / ("model_final.pth" if framework == "detectron2" else "weights/last.pt")
    else:
        w = default_weights(model_key, framework)
    if not w.exists():
        raise SystemExit(f"Không thấy trọng số {w} — huấn luyện {model_key} trước đã.")
    if framework == "detectron2":
        return _detectron2_predictor(model_key, cfg, w, conf, rd)
    return _ultralytics_predictor(model_key, cfg, w, conf)
