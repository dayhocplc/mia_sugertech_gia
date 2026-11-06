#!/usr/bin/env python3
"""Đối chứng không học sâu: chỉ số xanh dư (ExG) trên cùng tập test (mục A12).

Vì sao cần
----------
Bài kết luận rằng độ đầy đủ phát hiện là ràng buộc quyết định, nhưng không có
mốc nào cho biết ba mạng học sâu *hơn* một cách làm đơn giản bao nhiêu. Thiếu
mốc đó, người đọc không biết con số recall 0,39-0,50 là tốt hay tệ.

Phương pháp
-----------
ExG = 2G − R − B trên ảnh đã chuẩn hoá, ngưỡng Otsu, dọn hình thái học, rồi lấy
thành phần liên thông làm hộp phát hiện. Hai biến thể:

    exg_all   mọi thành phần thực vật là một phát hiện cỏ
    exg_size  lọc thêm theo khoảng diện tích, khoảng này chọn trên tập KIỂM ĐỊNH

Biến thể thứ hai cho đối chứng cơ hội công bằng nhất: ngưỡng diện tích được
chọn trên tập kiểm định rồi áp nguyên lên tập test, không hề nhìn tập test.

Điều cần nói trước
------------------
Cả mía và cỏ đều xanh, nên một chỉ số màu về nguyên tắc không tách được hai lớp.
Con số thu được chính là lượng mà "green-on-green" đòi học sâu phải bù.

Dùng
----
    python3 src/baseline_exg.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RESULTS, load_cfg, log


def exg_mask(img: np.ndarray) -> np.ndarray:
    """Mặt nạ thực vật từ chỉ số xanh dư, ngưỡng Otsu, dọn hình thái học."""
    import cv2

    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    tot = np.maximum(b + g + r, 1e-6)
    exg = 2.0 * (g / tot) - (r / tot) - (b / tot)
    u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _t, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def boxes_from_mask(m: np.ndarray, min_px: int = 64) -> np.ndarray:
    """Hộp bao của từng thành phần liên thông, bỏ thành phần quá nhỏ."""
    import cv2

    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_px or w < 4 or h < 4:
            continue
        out.append([float(x), float(y), float(x + w), float(y + h)])
    return np.array(out, float).reshape(-1, 4)


def _score(items, lo: float | None, hi: float | None, conf: float, iou: float) -> dict:
    """Chấm các hộp ExG như phát hiện lớp cỏ, cùng quy ước với các mô hình."""
    import cv2

    from review_analysis import WEED, match_image, prf

    tp = fp = fn = 0
    n_det = 0
    for _meta, path, gb, gc in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        db = boxes_from_mask(exg_mask(img))
        if len(db) and (lo is not None or hi is not None):
            a = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
            keep = np.ones(len(db), bool)
            if lo is not None:
                keep &= a >= lo
            if hi is not None:
                keep &= a <= hi
            db = db[keep]
        n_det += len(db)
        dc = np.full(len(db), WEED, int)
        ds = np.ones(len(db), float)          # không có điểm tin cậy
        _h, pot, kept, used = match_image(gb, gc, db, dc, ds, conf, iou)
        is_w = gc == WEED
        hit = int(((pot == WEED) & is_w).sum())
        tp += hit
        fn += int(is_w.sum()) - hit
        fp += int(((kept == WEED) & (~used)).sum())
        fp += int(((pot == WEED) & (~is_w)).sum())
    p, r, f = prf(tp, fp, fn)
    return {"weed_precision": p, "weed_recall": r, "weed_f1": f,
            "tp": tp, "fp": fp, "fn": fn,
            "detections_per_image": n_det / max(len(items), 1)}


def load_split(cfg, split: str):
    """Nạp một split như load_test_set, nhưng cho split bất kỳ.

    Cần tập kiểm định để chọn ngưỡng diện tích; load_test_set trong evaluate.py
    chỉ đọc split test.
    """
    import collections

    from common import coco_json, dataset_root, remap_categories

    coco = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", split).read_text()))
    img_dir = dataset_root(cfg, "bbox") / split
    by = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by[a["image_id"]].append((np.array([x, y, x + w, y + h], float), a["category_id"]))
    items = []
    for im in coco["images"]:
        aa = by.get(im["id"], [])
        items.append((im, img_dir / im["file_name"],
                      np.array([b for b, _ in aa], float).reshape(-1, 4),
                      np.array([c for _, c in aa], int)))
    return items


def _tune_on_valid(cfg, conf: float, iou: float):
    """Chọn khoảng diện tích trên tập KIỂM ĐỊNH, không nhìn tập test."""
    from review_analysis import WEED

    items = load_split(cfg, "valid")
    areas = [float((b[2] - b[0]) * (b[3] - b[1]))
             for _m, _p, gb, gc in items for b, c in zip(gb, gc) if c == WEED]
    if not areas:
        return None, None, {"note": "tập kiểm định không có nhãn cỏ"}
    lo, hi = float(np.percentile(areas, 5)), float(np.percentile(areas, 95))
    best = None
    for cand in ((lo, hi), (lo, None), (None, hi)):
        sc = _score(items, cand[0], cand[1], conf, iou)
        if best is None or sc["weed_f1"] > best[1]["weed_f1"]:
            best = (cand, sc)
    log(f"chọn trên tập kiểm định ({len(items)} ảnh): diện tích "
        f"{best[0][0]} .. {best[0][1]} px², F1 kiểm định {best[1]['weed_f1']:.3f}")
    return best[0][0], best[0][1], {
        "chosen_on": "valid", "n_valid_images": len(items),
        "area_lo_px2": best[0][0], "area_hi_px2": best[0][1],
        "valid_f1": best[1]["weed_f1"],
        "weed_area_px2_p5_p95_on_valid": [lo, hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    from evaluate import load_test_set

    cfg = load_cfg()
    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    items, _ = load_test_set(cfg, "bbox", None)
    log(f"{len(items)} ảnh test")

    out = {"conf": conf, "iou": iou, "n_test_images": len(items),
           "method": ("ExG = 2G − R − B trên ảnh chuẩn hoá, ngưỡng Otsu, mở rồi đóng "
                      "hình thái học với nhân ellipse 7×7, thành phần liên thông ≥ 64 px "
                      "làm hộp. Mọi hộp được chấm như phát hiện lớp cỏ, cùng ngưỡng "
                      "tin cậy và IoU với ba mô hình học sâu."),
           "caveat": ("Mía và cỏ đều xanh, nên một chỉ số màu về nguyên tắc không tách "
                      "được hai lớp; đối chứng này đo phần việc mà học sâu phải làm, "
                      "không phải một phương pháp cạnh tranh."),
           "variants": {}}

    out["variants"]["exg_all"] = _score(items, None, None, conf, iou)
    log("exg_all   recall={weed_recall:.3f} precision={weed_precision:.3f} "
        "F1={weed_f1:.3f} {detections_per_image:.1f} hộp/ảnh".format(
            **out["variants"]["exg_all"]))

    lo, hi, tune = _tune_on_valid(cfg, conf, iou)
    out["size_filter_tuning"] = tune
    if lo is not None or hi is not None:
        out["variants"]["exg_size"] = _score(items, lo, hi, conf, iou)
        log("exg_size  recall={weed_recall:.3f} precision={weed_precision:.3f} "
            "F1={weed_f1:.3f} {detections_per_image:.1f} hộp/ảnh".format(
                **out["variants"]["exg_size"]))
    else:
        log("bỏ biến thể lọc cỡ: " + str(tune.get("note")))

    p = RESULTS / "baseline_exg.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"→ {p}")
    return out


if __name__ == "__main__":
    main()
