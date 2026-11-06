#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra các khối tính toán trước khi tin vào số của bài báo.

    python3 src/selftest.py

Không cần GPU, không cần trọng số. Chạy trong vài giây.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import coco_json, load_cfg, remap_categories
from data_prep import _source_image
from metrics import box_iou_batch, class_metrics, confusion_matrix_image

FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    iou = box_iou_batch(a, b)[0]
    check("IoU trùng khít = 1", abs(iou[0] - 1) < 1e-9)
    check("IoU chồng một nửa = 1/3", abs(iou[1] - 1 / 3) < 1e-9, f"{iou[1]}")
    check("IoU rời nhau = 0", iou[2] == 0)


def test_confusion_perfect():
    """Dự đoán đúng hoàn toàn → ma trận chỉ có đường chéo, mọi chỉ số = 1."""
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], float)
    cls = np.array([0, 1, 1])
    m = confusion_matrix_image(boxes, cls, boxes, cls, np.ones(3), 2)
    check("đoán đúng hết: không có ô ngoài đường chéo",
          int(m.sum() - np.trace(m)) == 0, str(m.tolist()))
    r = class_metrics(m, ["sugarcane", "weed"])["filtered"]
    check("đoán đúng hết: F1 = 1 cả hai lớp",
          all(abs(r["per_class"][c]["f1"] - 1) < 1e-9 for c in ("sugarcane", "weed")))


def test_confusion_known_case():
    """Trường hợp tính tay: 2 mía + 2 cỏ; 1 cỏ bị đoán thành mía, 1 mía bị bỏ sót,
    thêm 1 báo nhầm cỏ trên nền."""
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10],
                   [0, 20, 10, 30], [20, 20, 30, 30]], float)
    gt_c = np.array([0, 0, 1, 1])
    det = np.array([[0, 0, 10, 10],            # mía → đúng
                    [0, 20, 10, 30],           # cỏ  → đoán thành mía
                    [20, 20, 30, 30],          # cỏ  → đúng
                    [90, 90, 99, 99]], float)  # báo nhầm trên nền
    det_c = np.array([0, 0, 1, 1])
    m = confusion_matrix_image(gt, gt_c, det, det_c, np.ones(4), 2)
    want = np.array([[1, 0, 1],    # mía: 1 đúng, 1 bỏ sót
                     [1, 1, 0],    # cỏ : 1 thành mía, 1 đúng
                     [0, 1, 0]])   # nền: 1 báo nhầm thành cỏ
    check("ma trận khớp tính tay", np.array_equal(m, want), f"\n{m}\nmong đợi\n{want}")

    r = class_metrics(m, ["sugarcane", "weed"])
    f = r["filtered"]["per_class"]
    # sau khi bỏ background: ma trận [[1,0],[1,1]] → mía P=1/2, R=1; cỏ P=1, R=1/2
    check("bỏ background: Precision mía = 0,5", abs(f["sugarcane"]["precision"] - 0.5) < 1e-9)
    check("bỏ background: Recall mía = 1,0", abs(f["sugarcane"]["recall"] - 1.0) < 1e-9)
    check("bỏ background: Precision cỏ = 1,0", abs(f["weed"]["precision"] - 1.0) < 1e-9)
    check("bỏ background: Recall cỏ = 0,5", abs(f["weed"]["recall"] - 0.5) < 1e-9)
    check("bỏ background: Accuracy = 2/3",
          abs(r["filtered"]["overall_accuracy"] - 2 / 3) < 1e-9)
    # giữ background thì bỏ sót và báo nhầm mới được tính vào
    w = r["with_background"]["per_class"]
    check("giữ background: Recall mía = 0,5", abs(w["sugarcane"]["recall"] - 0.5) < 1e-9)
    check("giữ background: Precision cỏ = 0,5", abs(w["weed"]["precision"] - 0.5) < 1e-9)


def test_one_match_per_box():
    """Hai dự đoán chồng lên cùng một nhãn thật: chỉ cái IoU cao hơn được ghép."""
    gt = np.array([[0, 0, 10, 10]], float)
    det = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float)
    m = confusion_matrix_image(gt, np.array([1]), det, np.array([1, 1]), np.ones(2), 2)
    check("dự đoán thừa thành báo nhầm trên nền", m[2, 1] == 1 and m[1, 1] == 1, str(m.tolist()))


def test_conf_threshold():
    gt = np.array([[0, 0, 10, 10]], float)
    m = confusion_matrix_image(gt, np.array([0]), gt, np.array([0]),
                               np.array([0.2]), 2, conf_threshold=0.3)
    check("dự đoán dưới ngưỡng conf bị bỏ → thành bỏ sót", m[0, 2] == 1)


def test_category_remap():
    cfg = load_cfg()
    raw = json.loads(coco_json(cfg, "bbox", "test").read_text())
    out = remap_categories(cfg, raw)
    ids = {c["id"] for c in out["categories"]}
    check("category id ánh xạ về {0, 1}", ids == {0, 1})
    check("tên lớp đúng thứ tự",
          [c["name"] for c in out["categories"]] == cfg["dataset"]["classes"])
    check("lớp gộp giữ chỗ (id 0 của Roboflow) đã bị bỏ",
          all(a["category_id"] in (0, 1) for a in out["annotations"]))
    n_raw = sum(1 for a in raw["annotations"] if a["category_id"] in (1, 2))
    check("không mất đối tượng nào khi ánh xạ", len(out["annotations"]) == n_raw,
          f"{len(out['annotations'])} vs {n_raw}")


def test_yolo_roundtrip():
    """Nhãn YOLO sinh ra phải dựng lại được hộp gốc (sai số < 1 px)."""
    from common import YOLO_DATA
    lab_dir = YOLO_DATA / "bbox/labels/test"
    if not lab_dir.is_dir() or not any(lab_dir.iterdir()):
        print("  · bỏ qua roundtrip YOLO (chưa chạy data_prep.py --to-yolo)")
        return
    cfg = load_cfg()
    coco = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", "test").read_text()))
    by_name = {pathlib.Path(i["file_name"]).stem: i for i in coco["images"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    random.seed(0)
    worst = 0.0
    for stem in random.sample(sorted(by_name), min(20, len(by_name))):
        img = by_name[stem]
        w, h = img["width"], img["height"]
        want = sorted(tuple(round(v, 2) for v in a["bbox"]) for a in anns.get(img["id"], []))
        got = []
        for line in (lab_dir / f"{stem}.txt").read_text().splitlines():
            _c, cx, cy, bw, bh = (float(v) for v in line.split())
            got.append((round((cx - bw / 2) * w, 2), round((cy - bh / 2) * h, 2),
                        round(bw * w, 2), round(bh * h, 2)))
        got.sort()
        check(f"số hộp khớp ({stem[:24]})", len(want) == len(got), f"{len(want)} vs {len(got)}")
        for a, b in zip(want, got):
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    check(f"sai số dựng lại hộp < 1 px (lớn nhất {worst:.3f})", worst < 1.0)


def test_source_image_naming():
    check("cắt hash Roboflow đúng",
          _source_image("IMG_20250320_103646_jpg.rf.df06dd5c.jpg") == "IMG_20250320_103646")


def main():
    print("Kiểm tra metrics.py")
    test_iou(); test_confusion_perfect(); test_confusion_known_case()
    test_one_match_per_box(); test_conf_threshold()
    print("Kiểm tra dữ liệu")
    test_source_image_naming(); test_category_remap(); test_yolo_roundtrip()
    print()
    if FAIL:
        print(f"HỎNG {len(FAIL)} mục: " + ", ".join(FAIL))
        sys.exit(1)
    print("Tất cả đều đạt.")


if __name__ == "__main__":
    main()
