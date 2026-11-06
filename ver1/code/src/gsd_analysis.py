#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recall theo độ phân giải mặt đất (GSD) — kiểm trực tiếp giả thuyết thiếu pixel.

    python3 src/gsd_analysis.py

Phản biện vòng Sugar Tech nêu đúng một điểm: bài kết luận "đòn bẩy kỹ thuật là
hình học thu ảnh" nhưng chưa làm thí nghiệm nào thay đổi hình học thu ảnh. Suy
luận hiện có chỉ dựa vào một kết quả âm tính (tăng kích thước đầu vào của một
mô hình không đổi gì) cộng với việc recall tăng theo kích thước đối tượng, mà
điều sau gần như hiển nhiên. Phản biện cũng nêu một giải thích thay thế: các
instance nhỏ có thể là mảnh cỏ trong cụm dày, tức mơ hồ do cách gán nhãn, chứ
không phải do thiếu pixel.

Thí nghiệm này tách hai giải thích đó ra.

CÁCH LÀM. Mỗi ảnh test được giảm mẫu xuống s×1280 rồi phóng ngược về 1280. Kích
thước đầu vào của mô hình, toạ độ nhãn và tiên nghiệm tỷ lệ của bộ phát hiện đều
KHÔNG đổi; thứ duy nhất mất đi là lượng chi tiết thật trên mỗi đối tượng. Đó
đúng là thứ mà việc nâng camera lên cao làm mất: độ phân giải mặt đất thô đi
1/s lần, mỗi đối tượng còn s lần số pixel thật.

CÁCH ĐỌC. Nếu recall sụt nhanh khi s giảm thì số pixel trên đối tượng thật sự là
ràng buộc, và hạ camera hay cắt ô (tiling) là hướng đi đúng. Nếu recall gần như
đứng yên khi chi tiết bị phá đến một phần tư thì ràng buộc nằm ở chỗ khác —
nhiều khả năng ở độ hạt và tính nhất quán của nhãn, đúng như phản biện ngờ.

Thí nghiệm chỉ đi được về phía THÔ hơn. Ảnh nguồn đã là 1280×1280 do Roboflow ép
từ bản gốc 4080×3060, nên không có chiều ngược lại để thử, và bài phải nói rõ
điều đó thay vì ngoại suy.

Kết quả ghi ra results/gsd_analysis.json.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RESULTS, load_cfg, log
from evaluate import load_test_set
from review_analysis import PRIMARY, WEED, match_image

# s = 1 là ảnh như đã dùng trong bài; s = 0,1875 là camera cao gấp hơn năm lần.
SCALES = [1.0, 0.75, 0.50, 0.375, 0.25, 0.1875]
CAMERA_HEIGHT_CM = 140.0


def converged_runs(cfg) -> dict[str, list[tuple[int, str]]]:
    """(seed, thư mục chạy) của mọi lượt đã hội tụ, theo từng mô hình."""
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    out = {}
    for m in PRIMARY:
        rs = []
        for r in sv[m]["runs"]:
            if r.get("status") != "ok":
                continue
            seed = r["seed"]
            rs.append((seed, m if seed == 0 else f"{m}_seed{seed}"))
        out[m] = rs
    return out


def degrade(img, s: float):
    """Giảm mẫu rồi phóng ngược: mất chi tiết, giữ nguyên lưới pixel."""
    import cv2
    if s >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def quartile_edges(items) -> list[float]:
    """Cùng bộ mốc phân vị diện tích mà R7 dùng, để hai phân tích so được."""
    areas = [float((b[2] - b[0]) * (b[3] - b[1]))
             for _m, _p, gb, gc in items for b, c in zip(gb, gc) if c == WEED]
    return [float(x) for x in np.percentile(areas, [25, 50, 75])]


def run_one(predictor, items, s, conf, iou, edges):
    """Một lượt chạy ở một mức độ phân giải: recall/precision/F1 và theo phân vị."""
    import cv2
    tp = fp = fn = 0
    band_hit = collections.Counter()
    band_tot = collections.Counter()
    for _meta, path, gb, gc in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        pred = predictor(degrade(img, s))
        db = np.asarray(pred.boxes, float).reshape(-1, 4)
        dc = np.asarray(pred.classes, int)
        ds = np.asarray(pred.scores, float)
        # pred_of_true[i] là LỚP của dự đoán ghép với nhãn i (-1 nếu không ghép)
        _hit, pred_of_true, kept_cls, used = match_image(gb, gc, db, dc, ds, conf, iou)
        is_w = gc == WEED
        for i, b in enumerate(gb):
            if not is_w[i]:
                continue
            # "tìm thấy" nghĩa là ghép được VÀ mang nhãn cỏ — cùng định nghĩa
            # với Bảng 2, chứ không phải chỉ ghép được hộp.
            a = float((b[2] - b[0]) * (b[3] - b[1]))
            k = int(np.searchsorted(edges, a, side="right"))
            band_tot[k] += 1
            if pred_of_true[i] == WEED:
                tp += 1
                band_hit[k] += 1
            else:
                fn += 1
        # FP như counts_full: hộp cỏ không ghép được, cộng hộp cỏ ghép vào mía
        fp += int(((kept_cls == WEED) & (~used)).sum())
        fp += int(((pred_of_true == WEED) & (~is_w)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {"weed_recall": rec, "weed_precision": prec,
            "weed_f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "recall_by_quartile": [band_hit[k] / band_tot[k] if band_tot[k] else None
                                   for k in range(4)]}


def main():
    from predict import build_predictor

    cfg = load_cfg()
    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    items, _ = load_test_set(cfg, "bbox", None)
    edges = quartile_edges(items)
    runs = converged_runs(cfg)
    log(f"{len(items)} ảnh test · {sum(len(v) for v in runs.values())} lượt chạy · "
        f"{len(SCALES)} mức độ phân giải")

    out = {"scales": SCALES, "camera_height_cm": CAMERA_HEIGHT_CM,
           "quartile_edges_px2": edges, "conf": conf, "iou": iou,
           "note": ("Ảnh được giảm mẫu xuống s×1280 rồi phóng ngược về 1280, nên "
                    "kích thước đầu vào và toạ độ nhãn không đổi; chỉ chi tiết "
                    "thật trên mỗi đối tượng mất đi. GSD hiệu dụng thô đi 1/s lần, "
                    "tương đương camera đặt ở độ cao 140/s cm."),
           "models": {}}

    for m, rs in runs.items():
        per_scale = collections.defaultdict(list)
        for seed, stem in rs:
            rd = pathlib.Path("runs") / stem
            try:
                pr = build_predictor(m, cfg, conf=conf,
                                     run_dir=rd if rd.is_dir() else None)
            except SystemExit as e:
                log(f"bỏ qua {stem}: {e}")
                continue
            for s in SCALES:
                r = run_one(pr, items, s, conf, iou, edges)
                per_scale[s].append(r)
                log(f"  {stem:22s} s={s:<6.4g} recall={r['weed_recall']:.3f} "
                    f"precision={r['weed_precision']:.3f}")
        out["models"][m] = {
            f"{s:g}": {
                "n_runs": len(v),
                "equivalent_camera_height_cm": CAMERA_HEIGHT_CM / s,
                "effective_gsd_multiplier": 1 / s,
                **{k: {"mean": float(np.mean([r[k] for r in v])),
                       "sd": (float(np.std([r[k] for r in v], ddof=1))
                              if len(v) > 1 else None)}
                   for k in ("weed_recall", "weed_precision", "weed_f1")},
                "recall_by_quartile": [
                    float(np.mean([r["recall_by_quartile"][q] for r in v
                                   if r["recall_by_quartile"][q] is not None]))
                    for q in range(4)],
            }
            for s, v in sorted(per_scale.items(), reverse=True) if v
        }

    p = RESULTS / "gsd_analysis.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"→ {p}")
    for m, d in out["models"].items():
        line = "  ".join(f"s={s}:{v['weed_recall']['mean']:.3f}" for s, v in d.items())
        log(f"{m:14s} {line}")


if __name__ == "__main__":
    main()
