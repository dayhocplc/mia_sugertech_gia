#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Suy luận theo ô cắt trên ảnh gốc — kiểm giả thuyết thiếu pixel theo chiều MỊN.

    python3 src/tiling_analysis.py

Thí nghiệm GSD (src/gsd_analysis.py) chỉ đi được về phía THÔ hơn, vì ảnh dùng để
huấn luyện đã bị Roboflow ép xuống 1280×1280. Phản biện vòng hai nêu đúng rằng
chiều ngược lại mới là chiều quyết định cho thông điệp của bài, và nêu đúng rằng
ảnh gốc 4080×3072 vẫn còn. Kiểm lại: 123 trong 124 ảnh của tập test còn bản gốc
trong kho ảnh của nhóm, nên thí nghiệm làm được mà không cần huấn luyện lại.

CÁCH LÀM. Mỗi ảnh gốc được cắt thành lưới k×k ô chồng lấn 20%; mỗi ô được ép về
1280×1280 — cùng kiểu ép tỷ lệ mà Roboflow đã làm với cả khung, và vì mọi ô có
cùng tỷ lệ khung nên hệ số ép giống hệt lúc huấn luyện. Dự đoán của các ô được
đưa về toạ độ ảnh gốc rồi gộp bằng NMS theo lớp. Nhãn được đưa từ toạ độ 1280
sang toạ độ gốc bằng tỷ lệ từng trục.

  k = 1  cả khung ép về 1280×1280 — tái lập đúng đường ống của bài, làm đối chứng
  k = 2  mỗi đối tượng có gấp đôi số pixel theo mỗi chiều
  k = 3  gần như độ phân giải gốc

GSD hiệu dụng mịn đi k lần, tương đương hạ camera từ 140 cm xuống 140/k cm.

ĐIỀU THÍ NGHIỆM NÀY KHÔNG CHỨNG MINH. Mô hình được huấn luyện ở một thang tỷ lệ
và được chạy ở thang khác, nên một phần thay đổi là lệch miền chứ không phải
giới hạn nội tại của số pixel. Nhưng đó đúng là điều xảy ra khi hạ camera trên
một máy đã triển khai mà không huấn luyện lại, nên con số này là con số có nghĩa
với người vận hành. Mô hình huấn luyện lại ở chính thang đó có thể tốt hơn.

Kết quả ghi ra results/tiling_analysis.json.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RESULTS, load_cfg, log
from evaluate import load_test_set
from review_analysis import PRIMARY, WEED, match_image

# Kho ảnh gốc của nhóm. Không nằm trong paper.yaml vì nó không thuộc đường ống
# chính; đây là thí nghiệm bổ sung đọc thêm từ đó.
RAW_ROOT = pathlib.Path("/home/thanhtran/Gia-mia/mia/TayNinh/Nhan/lo_anh")
TILES = [1, 2, 3]
OVERLAP = 0.20
NMS_IOU = 0.55
EXPORT_PX = 1280
CAMERA_HEIGHT_CM = 140.0


def raw_index() -> dict[str, pathlib.Path]:
    out = {}
    for p in RAW_ROOT.rglob("*.jpg"):
        out.setdefault(p.stem, p)
    return out


def raw_path(index: dict, source: str) -> pathlib.Path | None:
    """Ảnh gốc của một ảnh nguồn Roboflow; tên gốc có thể thiếu tiền tố IMG_."""
    for k in (source, re.sub(r"^IMG_", "", source)):
        if k in index:
            return index[k]
    return None


def tile_boxes(W: int, H: int, k: int) -> list[tuple[int, int, int, int]]:
    """Lưới k×k ô chồng lấn, trả (x0, y0, x1, y1) trên ảnh gốc."""
    if k == 1:
        return [(0, 0, W, H)]
    tw, th = W / k, H / k
    ox, oy = tw * OVERLAP, th * OVERLAP
    out = []
    for i in range(k):
        for j in range(k):
            x0 = max(0.0, i * tw - ox / 2); x1 = min(float(W), x0 + tw + ox)
            y0 = max(0.0, j * th - oy / 2); y1 = min(float(H), y0 + th + oy)
            out.append((int(x0), int(y0), int(round(x1)), int(round(y1))))
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """NMS tiêu chuẩn; chỉ dùng để gộp dự đoán trùng ở vùng ô chồng lấn."""
    idx = np.argsort(-scores)
    keep = []
    while len(idx):
        i = idx[0]; keep.append(int(i))
        if len(idx) == 1:
            break
        b, r = boxes[i], boxes[idx[1:]]
        ix = np.maximum(0, np.minimum(b[2], r[:, 2]) - np.maximum(b[0], r[:, 0]))
        iy = np.maximum(0, np.minimum(b[3], r[:, 3]) - np.maximum(b[1], r[:, 1]))
        inter = ix * iy
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_r = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
        iou = inter / np.maximum(area_b + area_r - inter, 1e-9)
        idx = idx[1:][iou <= thr]
    return keep


def infer_tiled(predictor, img, k: int):
    """Chạy suy luận trên lưới k×k rồi gộp về toạ độ ảnh gốc."""
    import cv2
    H, W = img.shape[:2]
    bb, cc, ss = [], [], []
    for x0, y0, x1, y1 in tile_boxes(W, H, k):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        res = predictor(cv2.resize(crop, (EXPORT_PX, EXPORT_PX),
                                   interpolation=cv2.INTER_AREA))
        b = np.asarray(res.boxes, float).reshape(-1, 4)
        if not len(b):
            continue
        sx, sy = (x1 - x0) / EXPORT_PX, (y1 - y0) / EXPORT_PX
        b[:, [0, 2]] = b[:, [0, 2]] * sx + x0
        b[:, [1, 3]] = b[:, [1, 3]] * sy + y0
        bb.append(b)
        cc.append(np.asarray(res.classes, int))
        ss.append(np.asarray(res.scores, float))
    if not bb:
        return (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    bb = np.concatenate(bb); cc = np.concatenate(cc); ss = np.concatenate(ss)
    keep = []
    for c in np.unique(cc):
        m = np.nonzero(cc == c)[0]
        keep += [m[i] for i in nms(bb[m], ss[m], NMS_IOU)]
    keep = np.array(sorted(keep), int)
    return bb[keep], cc[keep], ss[keep]


def converged_runs(seeds: list[int] | None = None) -> dict[str, list[tuple[int, str]]]:
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    out = {}
    for m in PRIMARY:
        out[m] = [(r["seed"], m if r["seed"] == 0 else f"{m}_seed{r['seed']}")
                  for r in sv[m]["runs"]
                  if r.get("status") == "ok" and (seeds is None or r["seed"] in seeds)]
    return out


def main():
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="chỉ chạy các seed này (mặc định: mọi lượt đã hội tụ)")
    args = ap.parse_args()
    from data_prep import _source_image
    from predict import build_predictor

    cfg = load_cfg()
    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    items, _ = load_test_set(cfg, "bbox", None)
    index = raw_index()

    # chỉ giữ ảnh test còn bản gốc; nhãn đưa sang toạ độ gốc
    work, sizes = [], collections.Counter()
    for meta, _p, gb, gc in items:
        rp = raw_path(index, _source_image(meta["file_name"]))
        if rp is None:
            continue
        img = cv2.imread(str(rp))
        if img is None:
            continue
        H, W = img.shape[:2]
        sizes[f"{W}x{H}"] += 1
        g = gb.copy()
        g[:, [0, 2]] *= W / EXPORT_PX
        g[:, [1, 3]] *= H / EXPORT_PX
        work.append((rp, g, gc, gb))
    log(f"{len(work)}/{len(items)} ảnh test còn bản gốc · cỡ {dict(sizes)}")

    # mốc phân vị tính trên TOẠ ĐỘ 1280 để so được với R7 và gsd_analysis
    areas = [float((b[2] - b[0]) * (b[3] - b[1]))
             for _rp, _g, gc, gb in work for b, c in zip(gb, gc) if c == WEED]
    edges = [float(x) for x in np.percentile(areas, [25, 50, 75])]

    runs = converged_runs(args.seeds)
    out = {"tiles": TILES, "overlap": OVERLAP, "nms_iou": NMS_IOU,
           "seeds": args.seeds,
           "conf": conf, "iou": iou, "export_px": EXPORT_PX,
           "camera_height_cm": CAMERA_HEIGHT_CM,
           "n_test_images_with_raw": len(work), "n_test_images": len(items),
           "raw_sizes": dict(sizes), "quartile_edges_px2_on_1280": edges,
           "note": ("Mỗi ảnh gốc cắt thành lưới k×k ô chồng lấn 20%, mỗi ô ép về "
                    "1280×1280 rồi gộp dự đoán bằng NMS. k=1 tái lập đường ống của "
                    "bài và là đối chứng. Mô hình không được huấn luyện lại, nên một "
                    "phần thay đổi là lệch thang tỷ lệ giữa huấn luyện và suy luận — "
                    "đúng như khi hạ camera trên một máy đã triển khai."),
           "models": {}}

    for m, rs in runs.items():
        per_k = collections.defaultdict(list)
        for _seed, stem in rs:
            rd = pathlib.Path("runs") / stem
            try:
                pr = build_predictor(m, cfg, conf=0.05,
                                     run_dir=rd if rd.is_dir() else None)
            except SystemExit as e:
                log(f"bỏ qua {stem}: {e}")
                continue
            for k in TILES:
                tp = fp = fn = 0
                hit_q = collections.Counter(); tot_q = collections.Counter()
                for rp, g, gc, gb in work:
                    img = cv2.imread(str(rp))
                    db, dc, ds = infer_tiled(pr, img, k)
                    _h, pot, kept, used = match_image(g, gc, db, dc, ds, conf, iou)
                    is_w = gc == WEED
                    for i, b in enumerate(gb):
                        if not is_w[i]:
                            continue
                        a = float((b[2] - b[0]) * (b[3] - b[1]))
                        q = int(np.searchsorted(edges, a, side="right"))
                        tot_q[q] += 1
                        if pot[i] == WEED:
                            tp += 1; hit_q[q] += 1
                        else:
                            fn += 1
                    fp += int(((kept == WEED) & (~used)).sum())
                    fp += int(((pot == WEED) & (~is_w)).sum())
                rec = tp / (tp + fn) if tp + fn else 0.0
                prec = tp / (tp + fp) if tp + fp else 0.0
                per_k[k].append({
                    "weed_recall": rec, "weed_precision": prec,
                    "weed_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "tp": tp, "fp": fp, "fn": fn,
                    "recall_by_quartile": [hit_q[q] / tot_q[q] if tot_q[q] else None
                                           for q in range(4)]})
                log(f"  {stem:22s} k={k} recall={rec:.3f} precision={prec:.3f}")
        out["models"][m] = {
            str(k): {
                "n_runs": len(v),
                "effective_gsd_multiplier": 1.0 / k,
                "equivalent_camera_height_cm": CAMERA_HEIGHT_CM / k,
                **{f: {"mean": float(np.mean([r[f] for r in v])),
                       "sd": (float(np.std([r[f] for r in v], ddof=1))
                              if len(v) > 1 else None)}
                   for f in ("weed_recall", "weed_precision", "weed_f1")},
                "recall_by_quartile": [
                    float(np.mean([r["recall_by_quartile"][q] for r in v
                                   if r["recall_by_quartile"][q] is not None]))
                    for q in range(4)],
            }
            for k, v in sorted(per_k.items()) if v}

    p = RESULTS / "tiling_analysis.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"→ {p}")
    for m, d in out["models"].items():
        line = "  ".join(f"k={k}:{v['weed_recall']['mean']:.3f}" for k, v in d.items())
        log(f"{m:14s} {line}")


if __name__ == "__main__":
    main()
