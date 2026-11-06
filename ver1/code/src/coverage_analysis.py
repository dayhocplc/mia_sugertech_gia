#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diện tích cỏ thực sự được xử lý — ba cách đo thay cho phép cộng diện tích hộp.

    python3 src/coverage_analysis.py

Phản biện vòng Sugar Tech chỉ ra ba chỗ hỏng trong con số "diện tích phun" của
bài:

  1. Trọng số theo diện tích hộp CỘNG DỒN các hộp chồng nhau, nên mẫu số không
     phải diện tích cỏ thật mà là tổng diện tích hộp, đếm nhiều lần phần giao.
  2. Hộp bao không phải phần phủ thật của cỏ; bộ dữ liệu có bản xuất polygon
     dùng được để đo độ phủ theo pixel.
  3. Bình phun không làm việc theo từng instance mà theo vùng béc: chỉ cần một
     phát hiện trong vùng là cả vùng được phun, nên recall theo instance bi
     quan hơn thực tế.

Script tính lại cả ba, từ detection thô đã lưu ở results/dets/ nên không cần GPU:

  C1  recall theo diện tích HỢP (union) của các hộp cỏ — bỏ phần đếm trùng.
  C2  recall theo pixel cỏ thật, trên các ảnh của tập test bbox mà bản xuất
      polygon cũng có nhãn. Số ảnh ít nên đây là phép kiểm có giới hạn, không
      phải con số chính, và script in rõ cỡ mẫu.
  C3  recall và precision theo Ô LƯỚI, mô phỏng vùng béc phun: một ô là "có cỏ"
      nếu giao với hộp cỏ thật, là "được phun" nếu giao với hộp cỏ dự đoán.

Kết quả ghi ra results/coverage_analysis.json.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RESULTS, ROOT, load_cfg, log
from evaluate import load_test_set
from review_analysis import PRIMARY, WEED, load_runs, match_image

# Ô lưới tính theo phần bề ngang khung hình. Không quy ra mét được vì bài không
# đo tầm phủ mặt đất của khung hình; bài phải nêu giả định đó chứ không bịa số.
GRID_FRACTIONS = [1 / 4, 1 / 8, 1 / 16]
CONF, IOU = 0.30, 0.50


def _source(name: str) -> str:
    """Tên ảnh gốc, bỏ hậu tố Roboflow."""
    return re.sub(r"[._]rf\.[0-9a-f]+\.jpg$", "", name)


def union_area(boxes, shape) -> tuple[np.ndarray, float]:
    """Mặt nạ hợp của một tập hộp, và diện tích của nó tính bằng pixel."""
    m = np.zeros(shape, dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(max(x1, x0 + 1), shape[1]); y1 = min(max(y1, y0 + 1), shape[0])
        m[y0:y1, x0:x1] = True
    return m, float(m.sum())


def c1_union_area_recall(items, runs):
    """Recall theo diện tích hợp của hộp cỏ, thay cho tổng diện tích từng hộp."""
    out = {}
    for m, rs in runs.items():
        per_run = []
        for r in rs:
            num = den = 0.0
            for meta, _p, gb, gc in items:
                shape = (int(meta["height"]), int(meta["width"]))
                w_idx = np.nonzero(gc == WEED)[0]
                if not len(w_idx):
                    continue
                d = r["by_img"].get(meta["id"], ([], [], []))
                db = np.array(d[0], float).reshape(-1, 4)
                dc = np.array(d[1], int); ds = np.array(d[2], float)
                _h, pred_of_true, _kc, _u = match_image(gb, gc, db, dc, ds, CONF, IOU)
                _all_m, a_all = union_area(gb[w_idx], shape)
                found = [i for i in w_idx if pred_of_true[i] == WEED]
                _hit_m, a_hit = union_area(gb[found], shape) if found else (None, 0.0)
                num += a_hit; den += a_all
            per_run.append(num / den if den else 0.0)
        out[m] = {"mean": float(np.mean(per_run)),
                  "sd": float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
                  "n_runs": len(per_run)}
    return out


def _polygon_masks(cfg):
    """Mặt nạ pixel cỏ theo bản xuất polygon, khoá theo tên ảnh gốc."""
    import cv2
    root = ROOT / cfg["dataset"]["polygon_root"]
    cls = cfg["dataset"]["classes"]
    masks = {}
    for split in ("train", "valid", "test"):
        f = root / split / "_annotations.coco.json"
        if not f.exists():
            continue
        coco = json.loads(f.read_text())
        name_of = {c["id"]: c["name"].lower() for c in coco["categories"]}
        weed_ids = {i for i, n in name_of.items() if "weed" in n or n == cls[WEED].lower()}
        by_img = collections.defaultdict(list)
        for a in coco["annotations"]:
            if a["category_id"] in weed_ids and a.get("segmentation"):
                by_img[a["image_id"]].append(a["segmentation"])
        for im in coco["images"]:
            segs = by_img.get(im["id"])
            if not segs:
                continue
            m = np.zeros((im["height"], im["width"]), np.uint8)
            for seg in segs:
                for poly in (seg if isinstance(seg, list) else []):
                    if isinstance(poly, list) and len(poly) >= 6:
                        pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
                        cv2.fillPoly(m, [pts], 1)
            masks.setdefault(_source(im["file_name"]), m.astype(bool))
    return masks


def _box_area_coverage(items, runs):
    """Phần diện tích hộp cỏ thật bị hộp cỏ dự đoán phủ, trên TOÀN tập test.

    Cùng một phép đo với cột "box area" của C2 nhưng chạy trên cả 124 ảnh, để
    biết 26 ảnh có nhãn polygon dễ hay khó hơn phần còn lại. Phản biện vòng hai
    nêu đúng rằng nếu không có con số này thì không nói được tập con có đại diện
    hay không.
    """
    out = {}
    for m, rs in runs.items():
        vals = []
        for r in rs:
            num = den = 0.0
            for meta, _p, gb, gc in items:
                shape = (int(meta["height"]), int(meta["width"]))
                d = r["by_img"].get(meta["id"], ([], [], []))
                db = np.array(d[0], float).reshape(-1, 4)
                dc = np.array(d[1], int); ds = np.array(d[2], float)
                keep = (ds > CONF) & (dc == WEED)
                det_mask, _ = union_area(db[keep], shape)
                gt_mask, a_all = union_area(gb[gc == WEED], shape)
                num += float((gt_mask & det_mask).sum()); den += a_all
            vals.append(num / den if den else 0.0)
        out[m] = {"mean": float(np.mean(vals)),
                  "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                  "n_runs": len(vals)}
    return out


def c2_pixel_coverage(items, runs, cfg):
    """Độ phủ pixel cỏ thật, trên phần tập test có nhãn polygon."""
    import cv2
    masks = _polygon_masks(cfg)
    pairs = [(meta, gb, gc, masks[_source(meta["file_name"])])
             for meta, _p, gb, gc in items if _source(meta["file_name"]) in masks]
    log(f"C2: {len(pairs)}/{len(items)} ảnh test có nhãn polygon")
    if not pairs:
        return {"n_images": 0}
    out = {"n_images": len(pairs), "models": {}}
    for m, rs in runs.items():
        by_px, by_box = [], []
        for r in rs:
            num_px = den_px = num_bx = den_bx = 0.0
            for meta, gb, gc, pmask in pairs:
                shape = (int(meta["height"]), int(meta["width"]))
                if pmask.shape != shape:
                    pmask = cv2.resize(pmask.astype(np.uint8), (shape[1], shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                d = r["by_img"].get(meta["id"], ([], [], []))
                db = np.array(d[0], float).reshape(-1, 4)
                dc = np.array(d[1], int); ds = np.array(d[2], float)
                keep = (ds > CONF) & (dc == WEED)
                det_mask, _ = union_area(db[keep], shape)
                num_px += float((pmask & det_mask).sum()); den_px += float(pmask.sum())
                w_idx = np.nonzero(gc == WEED)[0]
                gt_mask, a_all = union_area(gb[w_idx], shape)
                num_bx += float((gt_mask & det_mask).sum()); den_bx += a_all
            by_px.append(num_px / den_px if den_px else 0.0)
            by_box.append(num_bx / den_bx if den_bx else 0.0)
        out["models"][m] = {
            "pixel_coverage_recall": {"mean": float(np.mean(by_px)),
                                      "sd": float(np.std(by_px, ddof=1)) if len(by_px) > 1 else None},
            "box_area_recall_same_images": {"mean": float(np.mean(by_box)),
                                            "sd": float(np.std(by_box, ddof=1)) if len(by_box) > 1 else None},
            "n_runs": len(by_px)}
    return out


def c3_grid_recall(items, runs):
    """Recall/precision theo ô lưới — mô phỏng vùng béc phun."""
    out = {}
    for frac in GRID_FRACTIONS:
        per_model = {}
        for m, rs in runs.items():
            rec, prec = [], []
            for r in rs:
                tp = fp = fn = 0
                for meta, _p, gb, gc in items:
                    H, W = int(meta["height"]), int(meta["width"])
                    g = max(1, int(round(W * frac)))
                    ny, nx = int(np.ceil(H / g)), int(np.ceil(W / g))
                    def cells(boxes):
                        c = np.zeros((ny, nx), bool)
                        for b in boxes:
                            x0 = max(int(b[0] // g), 0); x1 = min(int((b[2] - 1e-9) // g), nx - 1)
                            y0 = max(int(b[1] // g), 0); y1 = min(int((b[3] - 1e-9) // g), ny - 1)
                            if x1 >= x0 and y1 >= y0:
                                c[y0:y1 + 1, x0:x1 + 1] = True
                        return c
                    d = r["by_img"].get(meta["id"], ([], [], []))
                    db = np.array(d[0], float).reshape(-1, 4)
                    dc = np.array(d[1], int); ds = np.array(d[2], float)
                    keep = (ds > CONF) & (dc == WEED)
                    truth = cells(gb[gc == WEED]); pred = cells(db[keep])
                    tp += int((truth & pred).sum())
                    fn += int((truth & ~pred).sum())
                    fp += int((~truth & pred).sum())
                rec.append(tp / (tp + fn) if tp + fn else 0.0)
                prec.append(tp / (tp + fp) if tp + fp else 0.0)
            per_model[m] = {
                "cell_recall": {"mean": float(np.mean(rec)),
                                "sd": float(np.std(rec, ddof=1)) if len(rec) > 1 else None},
                "cell_precision": {"mean": float(np.mean(prec)),
                                   "sd": float(np.std(prec, ddof=1)) if len(prec) > 1 else None},
                "n_runs": len(rec)}
        out[f"{frac:.4g}"] = {"cell_fraction_of_frame_width": frac, "models": per_model}
    return out


def main():
    cfg = load_cfg()
    items, _ = load_test_set(cfg, "bbox", None)
    runs = load_runs(cfg)
    log(f"{len(items)} ảnh test · {sum(len(v) for v in runs.values())} lượt chạy")

    out = {"conf": CONF, "iou": IOU, "n_test_images": len(items),
           "note": ("C1 bỏ phần đếm trùng của các hộp chồng nhau; C2 đo pixel cỏ thật "
                    "trên phần tập test có nhãn polygon; C3 tính theo ô lưới để mô phỏng "
                    "vùng béc phun. Ô lưới tính theo phần bề ngang khung hình vì bài "
                    "không đo tầm phủ mặt đất của khung hình.")}
    log("C1 recall theo diện tích hợp…")
    out["union_area_recall"] = c1_union_area_recall(items, runs)
    log("C2 độ phủ pixel theo nhãn polygon…")
    out["pixel_coverage"] = c2_pixel_coverage(items, runs, cfg)
    out["box_area_coverage_all_images"] = _box_area_coverage(items, runs)
    log("C3 recall theo ô lưới…")
    out["grid_recall"] = c3_grid_recall(items, runs)

    p = RESULTS / "coverage_analysis.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"→ {p}")
    for m in PRIMARY:
        u = out["union_area_recall"][m]["mean"]
        g = out["grid_recall"]["0.125"]["models"][m]["cell_recall"]["mean"]
        log(f"{m:14s} diện tích hợp {u:.3f}   ô 1/8 khung {g:.3f}")


if __name__ == "__main__":
    main()
