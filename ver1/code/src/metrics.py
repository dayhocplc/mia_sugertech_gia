#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ma trận nhầm lẫn và bốn chỉ số của bài báo (công thức 1–4).

Thuật toán ở đây tái hiện đúng `supervision.ConfusionMatrix.benchmark` —
công cụ mà khoá luận gốc đã dùng để sinh Bảng 2 và Bảng 3 — nhưng không
phụ thuộc vào thư viện đó, để kết quả tái lập được và kiểm chứng được:

  1. Bỏ dự đoán có confidence <= conf_threshold (mặc định 0,30).
  2. Ghép dự đoán với nhãn thật khi IoU > iou_threshold (mặc định 0,50);
     mỗi nhãn thật và mỗi dự đoán chỉ được ghép một lần, ưu tiên IoU cao.
  3. Ma trận M[thật, đoán] cỡ (C+1)×(C+1); chỉ số cuối cùng là background:
       - nhãn thật không ghép được   → M[lớp_thật, background]  (bỏ sót)
       - dự đoán không ghép được     → M[background, lớp_đoán]  (báo nhầm)

Bảng 2/3 của bản thảo được tính trên ma trận ĐÃ BỎ hàng và cột background
(`drop_background: true` trong configs/paper.yaml) — đây chính là bước
"lọc chỉ số hàng và cột ... nhằm loại bỏ lớp background" trong khoá luận.
Vì vậy FP/FN chỉ đếm nhầm lẫn mía↔cỏ, không đếm bỏ sót và báo nhầm trên nền.
Hàm `class_metrics` trả về cả hai phiên bản để bản thảo ghi rõ mình dùng cái nào.
"""

from __future__ import annotations

import numpy as np


def box_iou_batch(boxes_true: np.ndarray, boxes_det: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp (nhãn thật, dự đoán). Hộp ở dạng xyxy."""
    if boxes_true.size == 0 or boxes_det.size == 0:
        return np.zeros((len(boxes_true), len(boxes_det)), dtype=float)

    def area(b):
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    a_true, a_det = area(boxes_true), area(boxes_det)
    top_left = np.maximum(boxes_true[:, None, :2], boxes_det[None, :, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_det[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = a_true[:, None] + a_det[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _drop_extra_matches(matches: np.ndarray) -> np.ndarray:
    """Giữ lại ghép IoU cao nhất: mỗi dự đoán một lần, mỗi nhãn thật một lần."""
    if matches.shape[0] == 0:
        return matches
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]   # theo dự đoán
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]   # theo nhãn thật
    return matches


def confusion_matrix_image(
    true_boxes: np.ndarray,
    true_classes: np.ndarray,
    det_boxes: np.ndarray,
    det_classes: np.ndarray,
    det_conf: np.ndarray,
    num_classes: int,
    conf_threshold: float = 0.30,
    iou_threshold: float = 0.50,
    return_matched: bool = False,
):
    """Ma trận nhầm lẫn (C+1)×(C+1) cho MỘT ảnh. Hàng = thật, cột = đoán.

    `return_matched=True` trả thêm mảng bool cho biết nhãn thật nào ghép được
    với một dự đoán — dùng để đo xem đối tượng bị bỏ sót có nhỏ hơn đối tượng
    phát hiện được hay không.
    """
    m = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    bg = num_classes

    keep = det_conf > conf_threshold
    det_boxes, det_classes = det_boxes[keep], det_classes[keep]

    iou = box_iou_batch(true_boxes, det_boxes)
    hit = np.asarray(iou > iou_threshold).nonzero()
    if hit[0].shape[0]:
        matches = np.stack((hit[0], hit[1], iou[hit]), axis=1)
        matches = _drop_extra_matches(matches)
    else:
        matches = np.zeros((0, 3))

    matched_true = matches[:, 0].astype(int)
    matched_det = matches[:, 1].astype(int)

    for i, c_true in enumerate(true_classes):
        sel = matched_true == i
        if sel.sum() == 1:
            m[int(c_true), int(det_classes[matched_det[sel][0]])] += 1
        else:
            m[int(c_true), bg] += 1                 # bỏ sót

    for i, c_det in enumerate(det_classes):
        if not (matched_det == i).any():
            m[bg, int(c_det)] += 1                  # báo nhầm trên nền

    if return_matched:
        hit_mask = np.zeros(len(true_classes), dtype=bool)
        hit_mask[matched_true] = True
        return m, hit_mask
    return m


def class_metrics(matrix: np.ndarray, class_names: list[str]) -> dict:
    """Precision/Recall/F1/Accuracy theo công thức (1)–(4) của bản thảo.

    Trả về hai nhánh:
      "filtered" — đã bỏ hàng+cột background (số liệu Bảng 2 và Bảng 3);
      "with_background" — giữ nguyên, tức có tính bỏ sót và báo nhầm trên nền.
    """
    n = len(class_names)
    out = {}
    for tag, mat in (("filtered", matrix[:n, :n]), ("with_background", matrix)):
        mat = np.asarray(mat, dtype=float)
        tp = np.diag(mat)[:n]
        fp = mat.sum(axis=0)[:n] - tp                # cột = dự đoán
        fn = mat.sum(axis=1)[:n] - tp                # hàng = nhãn thật
        total = mat.sum()
        tn = total - tp - fp - fn                    # công thức (4)

        per_class = {}
        for i, name in enumerate(class_names):
            p = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0
            r = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            denom = tp[i] + tn[i] + fp[i] + fn[i]
            per_class[name] = {
                "TP": int(tp[i]), "FP": int(fp[i]), "FN": int(fn[i]), "TN": int(tn[i]),
                "precision": p, "recall": r, "f1": f1,
                "accuracy": float((tp[i] + tn[i]) / denom) if denom else 0.0,
            }

        out[tag] = {
            "matrix": mat.astype(int).tolist(),
            "per_class": per_class,
            # Bảng 3 — trung bình không trọng số giữa hai lớp (macro average)
            "macro": {
                k: float(np.mean([per_class[c][k] for c in class_names]))
                for k in ("precision", "recall", "f1", "accuracy")
            },
            # Độ chính xác tổng thể: tỷ lệ ô nằm trên đường chéo
            "overall_accuracy": float(np.trace(mat) / total) if total else 0.0,
        }
    return out


def accumulate(matrices) -> np.ndarray:
    """Cộng dồn ma trận từng ảnh thành ma trận của cả tập test."""
    it = iter(matrices)
    total = next(it).copy()
    for m in it:
        total += m
    return total
