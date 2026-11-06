#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đánh giá một mô hình trên tập test và ghi kết quả cho Bảng 2, Bảng 3.

    python3 src/evaluate.py --model yolov10
    python3 src/evaluate.py --all

Sinh ra results/<model>.json gồm:
  * ma trận nhầm lẫn 3×3 (sugarcane, weed, background) trên cả tập test;
  * Precision/Recall/F1/Accuracy từng lớp — công thức (1)–(4);
  * AP/AP50/AP75 theo chuẩn COCO, để đối chiếu với các bài báo khác;
  * tốc độ suy luận trên GPU máy trạm (ms/ảnh và FPS).

Ngưỡng conf 0,30 và IoU 0,50 lấy từ configs/paper.yaml — đúng mặc định của
supervision.ConfusionMatrix.benchmark mà khoá luận gốc đã dùng.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import (MODEL_LABELS, MODEL_ORDER, RESULTS, coco_json, dataset_root,
                    load_cfg, log, remap_categories, save_json)
from data_prep import _source_image
from metrics import class_metrics, confusion_matrix_image
from predict import build_predictor


def common_test_images(cfg: dict) -> set[str]:
    """Tập test dùng chung cho cả năm mô hình.

    Hai bản xuất Roboflow có tập test gần như rời nhau (124 ảnh và 50 ảnh, chỉ 3
    ảnh chung), nên so năm mô hình với nhau bằng số đo trên hai tập đó là không
    hợp lệ. Tập dùng chung ở đây là tập test của bản nhãn hộp, sau khi loại các
    ảnh đã xuất hiện trong tập train của bản nhãn đa giác — nếu không loại thì
    Mask R-CNN và YOLOv8-seg được chấm trên ảnh chúng đã nhìn thấy khi huấn luyện.

    Nhãn thật dùng chung là hộp; cả năm mô hình đều sinh hộp, nên phép so là
    cùng một thước đo trên cùng một tập ảnh.
    """
    test_bbox = {_source_image(i["file_name"])
                 for i in json.loads(coco_json(cfg, "bbox", "test").read_text())["images"]}
    train_poly = {_source_image(i["file_name"])
                  for i in json.loads(coco_json(cfg, "polygon", "train").read_text())["images"]}
    return test_bbox - train_poly


def load_test_set(cfg: dict, annotation: str, only_sources: set[str] | None = None):
    coco = remap_categories(cfg, json.loads(coco_json(cfg, annotation, "test").read_text()))
    if only_sources is not None:
        keep_ids = {i["id"] for i in coco["images"]
                    if _source_image(i["file_name"]) in only_sources}
        coco = dict(coco,
                    images=[i for i in coco["images"] if i["id"] in keep_ids],
                    annotations=[a for a in coco["annotations"] if a["image_id"] in keep_ids])
    img_dir = dataset_root(cfg, annotation) / "test"
    by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append((np.array([x, y, x + w, y + h], float), a["category_id"]))
    items = []
    for img in coco["images"]:
        anns = by_img.get(img["id"], [])
        boxes = np.array([b for b, _ in anns], float).reshape(-1, 4)
        classes = np.array([c for _, c in anns], int)
        items.append((img, img_dir / img["file_name"], boxes, classes))
    return items, coco


def coco_ap(cfg: dict, annotation: str, detections: list,
            only_sources: set[str] | None = None) -> dict:
    """AP theo pycocotools, trên bản COCO đã ánh xạ category id."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {}
    if not detections:
        return {}
    import contextlib
    import io
    import tempfile

    gt = remap_categories(cfg, json.loads(coco_json(cfg, annotation, "test").read_text()))
    if only_sources is not None:
        keep = {i["id"] for i in gt["images"] if _source_image(i["file_name"]) in only_sources}
        gt = dict(gt, images=[i for i in gt["images"] if i["id"] in keep],
                  annotations=[a for a in gt["annotations"] if a["image_id"] in keep])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt, f)
        gt_path = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = COCO(gt_path)
        dt = c.loadRes(detections)
        e = COCOeval(c, dt, "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    pathlib.Path(gt_path).unlink(missing_ok=True)
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    return {k: float(v) * 100 for k, v in zip(keys, e.stats[:6])}


def evaluate(model_key: str, cfg: dict, weights=None, limit: int | None = None,
             common: bool = False, run_dir=None, tag: str | None = None) -> dict:
    import cv2

    spec = cfg["models"][model_key]
    ev = cfg["eval"]
    names = cfg["dataset"]["classes"]
    # chế độ dùng chung: mọi mô hình chấm trên tập test của bản nhãn hộp
    annotation = "bbox" if common else spec["annotation"]
    sources = common_test_images(cfg) if common else None
    items, _ = load_test_set(cfg, annotation, sources)
    if limit:
        items = items[:limit]

    predictor = build_predictor(model_key, cfg, weights, conf=ev["conf_threshold"],
                                run_dir=run_dir)
    matrix = np.zeros((len(names) + 1, len(names) + 1), dtype=np.int64)
    detections, latencies = [], []
    area_hit, area_miss = [], []      # diện tích cỏ (% khung hình), ghép được / bỏ sót
    # Ma trận nhầm lẫn của TỪNG ảnh — cần cho bootstrap lấy mẫu lại theo ảnh.
    per_image = []

    for n, (img_meta, path, gt_boxes, gt_classes) in enumerate(items, 1):
        img = cv2.imread(str(path))
        if img is None:
            log(f"bỏ qua ảnh đọc không được: {path}")
            continue
        t0 = time.perf_counter()
        pred = predictor(img)
        latencies.append(time.perf_counter() - t0)

        m_img, hit = confusion_matrix_image(
            gt_boxes, gt_classes, pred.boxes, pred.classes, pred.scores,
            num_classes=len(names),
            conf_threshold=ev["conf_threshold"], iou_threshold=ev["iou_threshold"],
            return_matched=True)
        matrix += m_img
        per_image.append({"image_id": img_meta["id"], "matrix": m_img.tolist()})
        frame = float(img_meta["width"] * img_meta["height"])
        for b, c, ok in zip(gt_boxes, gt_classes, hit):
            if c != 1:                       # chỉ xét lớp cỏ
                continue
            pct = (b[2] - b[0]) * (b[3] - b[1]) / frame * 100
            (area_hit if ok else area_miss).append(float(pct))

        for b, c, s in zip(pred.boxes, pred.classes, pred.scores):
            detections.append({
                "image_id": img_meta["id"], "category_id": int(c),
                "bbox": [float(b[0]), float(b[1]),
                         float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(s),
            })
        if n % 25 == 0:
            log(f"{model_key}: {n}/{len(items)} ảnh")

    m = class_metrics(matrix, names)
    # 5 ảnh đầu thường gánh chi phí khởi động CUDA — bỏ khi tính tốc độ
    warm = latencies[5:] if len(latencies) > 10 else latencies
    result = {
        "model": model_key,
        "run_dir": str(run_dir) if run_dir else None,
        "label": MODEL_LABELS[model_key],
        "annotation": annotation,
        "test_set": "common" if common else "native",
        "imgsz": spec["imgsz"],
        "n_test_images": len(latencies),
        "conf_threshold": ev["conf_threshold"],
        "iou_threshold": ev["iou_threshold"],
        "confusion_matrix": matrix.tolist(),
        "per_image": per_image,
        "confusion_matrix_labels": names + ["background"],
        "metrics": m,
        "coco": coco_ap(cfg, annotation, detections, sources),
        "weed_area_pct": {
            "detected_median": float(np.median(area_hit)) if area_hit else None,
            "missed_median": float(np.median(area_miss)) if area_miss else None,
            "n_detected": len(area_hit),
            "n_missed": len(area_miss),
        },
        # Số đo một lượt bên trong vòng lặp đánh giá, kèm I/O và tải dữ liệu, nên
        # thấp hơn bộ đo thông lượng 2,5-4 lần và KHÔNG dùng được cho bài. Tên
        # trường nói rõ điều đó vì nó nằm trong phụ lục công bố. Số của bài lấy
        # từ results/edge_<model>.json qua common.clean_fps().
        "speed_eval_loop_single_pass": {
            "ms_per_image": float(np.mean(warm) * 1000),
            "fps": float(1.0 / np.mean(warm)),
            "device": "workstation GPU",
            "note": ("Một lượt trong vòng lặp đánh giá, gồm cả I/O; không phải "
                     "phép đo thông lượng. Số thông lượng của bài nằm ở "
                     "results/edge_<model>.json (lặp 10×40 ảnh, bỏ 10 ảnh khởi "
                     "động, batch 1, FP32, đồng bộ CUDA từng ảnh)."),
        },
    }
    stem = tag or model_key
    out = (RESULTS / "common" / f"{stem}.json") if common else (RESULTS / f"{stem}.json")
    save_json(result, out)
    # Dự đoán thô lưu riêng: cần để tính lại AP trên tập ảnh lấy mẫu lại,
    # và quá lớn để nhét vào file kết quả chính.
    save_json({"model": model_key, "annotation": annotation,
               "test_set": "common" if common else "native",
               "detections": detections},
              (out.parent / "dets") / f"{stem}.json")
    f = m["filtered"]
    log(f"{MODEL_LABELS[model_key]}: "
        + "  ".join(f"{c} P={f['per_class'][c]['precision']:.3f} "
                    f"R={f['per_class'][c]['recall']:.3f} "
                    f"F1={f['per_class'][c]['f1']:.3f}" for c in names)
        + f"  |  Acc={f['overall_accuracy']:.3f}  {result['speed_eval_loop_single_pass']['fps']:.1f} FPS một lượt")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODEL_ORDER)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu (thử nhanh)")
    ap.add_argument("--common", action="store_true",
                    help="chấm cả năm mô hình trên MỘT tập test dùng chung "
                         "(bắt buộc nếu muốn so sánh năm mô hình với nhau)")
    a = ap.parse_args()
    if not (a.model or a.all):
        ap.error("chọn --model <tên> hoặc --all")

    cfg = load_cfg()
    keys = MODEL_ORDER if a.all else [a.model]
    for k in keys:
        try:
            evaluate(k, cfg, a.weights, a.limit, a.common)
        except SystemExit as e:
            log(f"bỏ qua {k}: {e}")


if __name__ == "__main__":
    main()
