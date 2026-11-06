#!/usr/bin/env python3
"""Huấn luyện lại ở ĐÚNG thang ô chia, rồi chấm ở thang đó (mục A13 của audit).

Vì sao cần thí nghiệm này
-------------------------
`tiling_analysis.py` cho thấy chia ô 2×2 chỉ thêm dưới 0,06 recall mà làm
precision giảm quá nửa. Nhưng ở đó mô hình được huấn luyện ở một thang rồi chạy
ở thang khác, nên kết quả lẫn hai nguyên nhân: *độ nhạy với thang kích thước* và
*lệch miền giữa huấn luyện và suy luận*. Bài không được phép khẳng định nguyên
nhân thứ nhất khi chưa loại được nguyên nhân thứ hai.

Thí nghiệm này loại nó: tập huấn luyện, kiểm định và kiểm tra đều được cắt từ
ảnh gốc thành lưới 2×2 giống hệt `tiling_analysis.py`, nên mô hình thấy đối
tượng ở đúng thang mà nó sẽ gặp khi suy luận.

Đối chứng cùng ngân sách — chỗ lượt chạy đầu đã sai
---------------------------------------------------
Lượt đầu chỉ huấn luyện điều kiện "ô → ô" rồi đặt cạnh Bảng 2 và cạnh
`tiling_analysis.json`. Cách đó không hợp lệ: hai điều kiện kia dùng tập huấn
luyện Roboflow đã tăng cường (2.892 ảnh, 31.672 đối tượng), còn điều kiện ô chỉ
có một bản cho mỗi ảnh gốc (3.856 ô, 13.530 đối tượng). Recall thấp hơn khi đó
có thể chỉ là *ít dữ liệu hơn*, không phải thang kích thước — tức là thí nghiệm
thay một yếu tố gây nhiễu bằng một yếu tố gây nhiễu khác.

Nên thí nghiệm dựng thêm điều kiện đối chứng `orig1`: **đúng 964 ảnh gốc đó, cũng
không tăng cường, cũng công thức huấn luyện đó**, chỉ khác là không chia ô — mỗi
ảnh gốc thành một ảnh 1280×1280 y như bản xuất của Roboflow.

Bốn ô của ma trận, hai cái so được trực tiếp
--------------------------------------------
                         chấm trên test gốc   chấm trên test ô 2×2
    huấn luyện trên gốc        orig1→orig1          orig1→tile2
    huấn luyện trên ô          tile2→orig1          tile2→tile2

Cặp quyết định là **cột phải**: cùng một tập nhãn kiểm tra, cùng số ảnh nguồn,
cùng công thức; khác duy nhất ở thang mà mô hình được huấn luyện. Nếu
`tile2→tile2` phục hồi recall so với `orig1→tile2` thì phần mất mát ở k=2 của
`tiling_analysis.py` là lệch miền huấn luyện–suy luận, và câu "độ đầy đủ bị chặn
bởi thang huấn luyện" phải phát biểu lại. Nếu không phục hồi thì giới hạn là thật.

Hai con số của hàng trên, `orig1→orig1` so với Bảng 2, đo riêng phần mà việc
tăng cường dữ liệu mang lại; nó giải thích vì sao giá trị tuyệt đối của cả thí
nghiệm này thấp hơn bài, và không được đọc như một kết quả về thang.

Dùng
----
    python3 src/tile_scale_training.py --build
    python3 src/tile_scale_training.py --train
    python3 src/tile_scale_training.py --eval
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RAE, RESULTS, dataset_root, load_cfg, log, remap_categories
from session_split import FOLDS, IMGDIR, YFOLDS, fold_ap, load_fold_test
from tiling_analysis import EXPORT_PX, OVERLAP, raw_index, raw_path, tile_boxes

TAG = "tile2"
K = 2
# tag điều kiện → số ô mỗi chiều. orig1 (k=1) là đối chứng cùng ngân sách.
CONDS = {"tile2": 2, "orig1": 1}
MODELS = ["faster_rcnn", "retinanet", "yolov10"]
# RetinaNet phân kỳ (loss_cls = NaN) ở lr=1e-3 của bài trên cả hai điều kiện,
# kể cả khi đã cắt chuẩn gradient ở 1,0. Hạ lr là biện pháp ổn định số học, và
# phải áp CÙNG giá trị cho cả hai điều kiện, nếu không thì chính nó thành yếu tố
# gây nhiễu cho so sánh.
RETINANET_LR = 3e-4
# Một hộp bị ô cắt mất phần lớn diện tích thì không còn là nhãn của vật đó nữa.
MIN_AREA_KEPT = 0.50
MIN_SIDE_PX = 8


def _gt_on_native(gb: np.ndarray, W: int, H: int) -> np.ndarray:
    """Nhãn từ toạ độ xuất 1280 sang toạ độ ảnh gốc, theo từng trục.

    Roboflow ép ảnh về vuông mà không giữ tỷ lệ khung, nên hệ số hai trục khác
    nhau — đúng cách `tiling_analysis.py` làm, để hai thí nghiệm so được.
    """
    g = gb.astype(float).copy()
    g[:, [0, 2]] *= W / EXPORT_PX
    g[:, [1, 3]] *= H / EXPORT_PX
    return g


def _boxes_in_tile(g: np.ndarray, gc: np.ndarray, tile, sx: float, sy: float):
    """Nhãn nằm trong một ô, cắt theo biên ô rồi đưa sang toạ độ ô đã ép cỡ."""
    x0, y0, x1, y1 = tile
    out = []
    for b, c in zip(g, gc):
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
        cx0, cy0 = max(b[0], x0), max(b[1], y0)
        cx1, cy1 = min(b[2], x1), min(b[3], y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) / area < MIN_AREA_KEPT:
            continue
        nx0, ny0 = (cx0 - x0) * sx, (cy0 - y0) * sy
        nx1, ny1 = (cx1 - x0) * sx, (cy1 - y0) * sy
        if nx1 - nx0 < MIN_SIDE_PX or ny1 - ny0 < MIN_SIDE_PX:
            continue
        out.append(([nx0, ny0, nx1 - nx0, ny1 - ny0], int(c)))
    return out


def build(cfg, tag: str = TAG) -> dict:
    """Cắt ba split từ ảnh gốc thành lưới k×k và ghi COCO + bố cục YOLO.

    k = 1 cho điều kiện đối chứng `orig1`: không chia ô, mỗi ảnh gốc thành một
    ảnh 1280×1280 — cùng hình học với bản xuất của Roboflow, chỉ bỏ phần tăng
    cường, nên nó giữ đúng ngân sách dữ liệu của điều kiện chia ô.
    """
    import cv2

    from data_prep import _source_image

    k = CONDS[tag]
    index = raw_index()
    IMGDIR.mkdir(parents=True, exist_ok=True)
    (FOLDS / tag).mkdir(parents=True, exist_ok=True)
    classes = cfg["dataset"]["classes"]
    stats = {"tag": tag, "k": k, "overlap": OVERLAP, "export_px": EXPORT_PX,
             "min_area_kept": MIN_AREA_KEPT, "min_side_px": MIN_SIDE_PX,
             "splits": {}}

    for split in ("train", "valid", "test"):
        # Phải ánh xạ lại id Roboflow (1=mía, 2=cỏ) về 0/1 và bỏ lớp gộp giữ
        # chỗ, đúng như load_test_set làm; nếu không, nhãn mang id 2 trong khi
        # danh mục chỉ có 0 và 1 và bộ huấn luyện báo KeyError.
        src = remap_categories(cfg, json.loads(
            (dataset_root(cfg, "bbox") / split / "_annotations.coco.json").read_text()))
        by = collections.defaultdict(list)
        for a in src["annotations"]:
            by[a["image_id"]].append(a)
        # một bản cho mỗi ảnh gốc: bản tăng cường không có ảnh gốc riêng
        seen, chosen = set(), []
        for im in sorted(src["images"], key=lambda x: x["file_name"]):
            s = _source_image(im["file_name"])
            if s in seen:
                continue
            seen.add(s)
            chosen.append((s, im))

        images, anns, missing = [], [], 0
        for s, im in chosen:
            rp = raw_path(index, s)
            if rp is None:
                missing += 1
                continue
            raw = cv2.imread(str(rp))
            if raw is None:
                missing += 1
                continue
            H, W = raw.shape[:2]
            gb = np.array([[a["bbox"][0], a["bbox"][1],
                            a["bbox"][0] + a["bbox"][2],
                            a["bbox"][1] + a["bbox"][3]] for a in by.get(im["id"], [])],
                          float).reshape(-1, 4)
            gc = np.array([a["category_id"] for a in by.get(im["id"], [])], int)
            g = _gt_on_native(gb, W, H)
            for ti, tile in enumerate(tile_boxes(W, H, k)):
                x0, y0, x1, y1 = tile
                tw, th = x1 - x0, y1 - y0
                if tw < 2 or th < 2:
                    continue
                sx, sy = EXPORT_PX / tw, EXPORT_PX / th
                keep = _boxes_in_tile(g, gc, tile, sx, sy)
                name = f"{s}_k{k}t{ti}.jpg"
                dst = IMGDIR / name
                if not dst.exists():
                    crop = cv2.resize(raw[y0:y1, x0:x1],
                                      (EXPORT_PX, EXPORT_PX),
                                      interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                iid = len(images)
                images.append({"id": iid, "file_name": name,
                               "width": EXPORT_PX, "height": EXPORT_PX})
                for bbox, c in keep:
                    anns.append({"id": len(anns) + 1, "image_id": iid,
                                 "category_id": c, "bbox": bbox,
                                 "area": float(bbox[2] * bbox[3]), "iscrowd": 0})

        out = {"info": {}, "licenses": [],
               "categories": [{"id": j, "name": n, "supercategory": "none"}
                              for j, n in enumerate(classes)],
               "images": images, "annotations": anns}
        (FOLDS / tag / f"{split}.json").write_text(json.dumps(out))
        stats["splits"][split] = {"source_images": len(chosen),
                                  "source_images_without_raw": missing,
                                  "tiles": len(images), "instances": len(anns)}
        log(f"[{tag}] {split:6s} {len(chosen)} ảnh gốc → {len(images)} ảnh, "
            f"{len(anns)} đối tượng (thiếu bản gốc: {missing})")

    _to_yolo_tag(tag, cfg)
    (FOLDS / tag / "build.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    return stats


def _to_yolo_tag(tag: str, cfg) -> None:
    """Bố cục YOLO cho fold này; nhãn sinh lại từ COCO của fold."""
    d = YFOLDS / tag
    for split in ("train", "valid", "test"):
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)
        coco = json.loads((FOLDS / tag / f"{split}.json").read_text())
        by = collections.defaultdict(list)
        for a in coco["annotations"]:
            by[a["image_id"]].append(a)
        for im in coco["images"]:
            w, h = im["width"], im["height"]
            link = d / "images" / split / im["file_name"]
            if not (link.exists() or link.is_symlink()):
                link.symlink_to((IMGDIR / im["file_name"]).resolve())
            lines = []
            for a in by.get(im["id"], []):
                x, y, bw, bh = a["bbox"]
                lines.append(f"{a['category_id']} {(x + bw / 2) / w:.6f} "
                             f"{(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            (d / "labels" / split / (pathlib.Path(im["file_name"]).stem + ".txt")
             ).write_text("\n".join(lines) + ("\n" if lines else ""))
    names = cfg["dataset"]["classes"]
    (d / "data.yaml").write_text(
        f"path: {d.resolve()}\ntrain: images/train\nval: images/valid\n"
        f"test: images/test\nnc: {len(names)}\nnames: [{', '.join(names)}]\n")


def train(cfg, tag: str = TAG, only=None, redo=False) -> None:
    """Huấn luyện tuần tự ba kiến trúc trên một điều kiện, seed mặc định."""
    py = sys.executable
    for m in (only or MODELS):
        rd = RAE / "runs" / f"{m}_fold{tag}"
        done = (rd / "model_final.pth").exists() or (rd / "weights" / "last.pt").exists()
        if done and not redo:
            log(f"bỏ qua {m} [{tag}]: đã có trọng số")
            continue
        script = "train_ultralytics.py" if m == "yolov10" else "train_detectron2.py"
        cmd = [py, str(RAE / "src" / script), "--model", m, "--fold", tag]
        # Cùng biện pháp ổn định số học cho MỌI điều kiện, nếu không thì chính
        # nó thành yếu tố gây nhiễu cho so sánh giữa các điều kiện.
        if m == "retinanet":
            cmd += ["--clip-grad", "1.0", "--base-lr", repr(RETINANET_LR)]
        log("$ " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=RAE)
        if r.returncode:
            log(f"LỖI {m} [{tag}]: mã trả về {r.returncode}")


def _score_on(cfg, model_key: str, train_tag: str, test_tag: str) -> dict | None:
    """Chấm trọng số của một điều kiện trên tập test của một điều kiện."""
    import cv2

    from predict import build_predictor
    from review_analysis import WEED, match_image, prf

    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    rd = RAE / "runs" / f"{model_key}_fold{train_tag}"
    try:
        pr = build_predictor(model_key, cfg, conf=0.05, run_dir=rd)
    except SystemExit as e:
        log(f"thiếu trọng số {model_key} [{train_tag}]: {e}")
        return None
    items, _ = load_fold_test(test_tag)
    tp = fp = fn = 0
    dets = []
    for meta, path, gb, gc in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        pd = pr(img)
        db = np.asarray(pd.boxes, float).reshape(-1, 4)
        dc = np.asarray(pd.classes, int)
        ds = np.asarray(pd.scores, float)
        for b, c_, s_ in zip(db, dc, ds):
            dets.append({"image_id": meta["id"], "category_id": int(c_),
                         "bbox": [float(b[0]), float(b[1]),
                                  float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s_)})
        _h, pot, kept, used = match_image(gb, gc, db, dc, ds, conf, iou)
        is_w = gc == WEED
        hit = int(((pot == WEED) & is_w).sum())
        tp += hit
        fn += int(is_w.sum()) - hit
        fp += int(((kept == WEED) & (~used)).sum())
        fp += int(((pot == WEED) & (~is_w)).sum())
    p_, r_, f_ = prf(tp, fp, fn)
    out = {"weed_precision": p_, "weed_recall": r_, "weed_f1": f_,
           "tp": tp, "fp": fp, "fn": fn, "n_test_images": len(items),
           **fold_ap(test_tag, dets)}
    log(f"{model_key:12s} {train_tag}→{test_tag}  recall={r_:.3f} "
        f"precision={p_:.3f} F1={f_:.3f} AP50={out.get('AP50', float('nan')):.1f}")
    return out


def evaluate(cfg) -> dict:
    """Chấm cả bốn ô của ma trận điều kiện × tập test."""
    out = {"conds": CONDS, "overlap": OVERLAP, "export_px": EXPORT_PX,
           "conf": cfg["eval"]["conf_threshold"], "iou": cfg["eval"]["iou_threshold"],
           "retinanet_base_lr": RETINANET_LR,
           "build": {t: json.loads((FOLDS / t / "build.json").read_text())
                     for t in CONDS if (FOLDS / t / "build.json").exists()},
           "note": ("Hai điều kiện dùng CÙNG 964 ảnh gốc, cùng công thức, đều "
                    "không tăng cường; khác duy nhất ở thang huấn luyện. Vì bỏ "
                    "phần tăng cường của Roboflow nên giá trị tuyệt đối thấp hơn "
                    "Bảng 2 của bài; chỉ so trong nội bộ ma trận là có nghĩa. "
                    "RetinaNet dùng lr thấp hơn ở CẢ HAI điều kiện vì lr của bài "
                    "làm loss_cls thành NaN; giá trị của nó không so được với Bảng 2."),
           "matrix": {}}

    for train_tag in CONDS:
        out["matrix"][train_tag] = {}
        for test_tag in CONDS:
            row = {}
            for m in MODELS:
                sc = _score_on(cfg, m, train_tag, test_tag)
                if sc is not None:
                    row[m] = sc
            out["matrix"][train_tag][test_tag] = row

    out["scale_contrast"] = _scale_contrast(out)
    out["augmentation_gap"] = _augmentation_gap(out)
    p = RESULTS / "tile_scale_training.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"→ {p}")
    for m, d in out["scale_contrast"].items():
        log(f"{m:12s} trên test ô: gốc→ô {d['orig_trained']['recall']:.3f}/"
            f"{d['orig_trained']['precision']:.3f}  →  ô→ô "
            f"{d['tile_trained']['recall']:.3f}/{d['tile_trained']['precision']:.3f}"
            f"   Δrecall {d['delta_recall']:+.3f}  Δprecision {d['delta_precision']:+.3f}")
    return out


def _scale_contrast(out: dict) -> dict:
    """Cặp quyết định: cùng tập nhãn test ô, khác duy nhất thang huấn luyện."""
    a = out["matrix"].get("orig1", {}).get("tile2", {})
    b = out["matrix"].get("tile2", {}).get("tile2", {})
    cmp_ = {}
    for m in MODELS:
        if m not in a or m not in b:
            continue
        cmp_[m] = {
            "test_set": "tile2",
            "orig_trained": {"recall": a[m]["weed_recall"],
                             "precision": a[m]["weed_precision"],
                             "f1": a[m]["weed_f1"], "AP50": a[m].get("AP50")},
            "tile_trained": {"recall": b[m]["weed_recall"],
                             "precision": b[m]["weed_precision"],
                             "f1": b[m]["weed_f1"], "AP50": b[m].get("AP50")},
            "delta_recall": b[m]["weed_recall"] - a[m]["weed_recall"],
            "delta_precision": b[m]["weed_precision"] - a[m]["weed_precision"],
            "delta_f1": b[m]["weed_f1"] - a[m]["weed_f1"],
        }
    return cmp_


def _augmentation_gap(out: dict) -> dict:
    """orig1→orig1 so với k=1 của tiling_analysis.json: phần của tăng cường.

    Cùng thang, cùng công thức, cùng ảnh nguồn; khác ở chỗ bài huấn luyện trên
    bản đã tăng cường của Roboflow. Con số này KHÔNG phải kết quả về thang, nó
    chỉ để giải thích vì sao ma trận trên có giá trị tuyệt đối thấp hơn bài.
    """
    f = RESULTS / "tiling_analysis.json"
    if not f.exists():
        return {}
    til = json.loads(f.read_text())
    gap = {}
    for m, rec in out["matrix"].get("orig1", {}).get("orig1", {}).items():
        row = til.get("models", {}).get(m, {}).get("1") or {}
        if not row:
            continue
        # tiling_analysis lưu mỗi chỉ số dưới dạng {"mean": …, "sd": …}
        pr = _mean(row.get("weed_recall"))
        pp = _mean(row.get("weed_precision"))
        gap[m] = {
            "paper_augmented": {"recall": pr, "precision": pp,
                                "f1": _mean(row.get("weed_f1"))},
            "this_unaugmented": {"recall": rec["weed_recall"],
                                 "precision": rec["weed_precision"],
                                 "f1": rec["weed_f1"]},
            "delta_recall": None if pr is None else rec["weed_recall"] - pr,
            "delta_precision": None if pp is None else rec["weed_precision"] - pp,
        }
    return gap


def _mean(v):
    """tiling_analysis.json gói mỗi chỉ số trong {"mean", "sd"}; lấy mean."""
    if isinstance(v, dict):
        v = v.get("mean")
    return None if v is None else float(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--models", nargs="*", choices=MODELS)
    ap.add_argument("--conds", nargs="*", choices=sorted(CONDS),
                    help="điều kiện cần dựng/huấn luyện (mặc định: cả hai)")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if not (a.build or a.train or a.eval):
        raise SystemExit("Chọn --build, --train hoặc --eval")
    conds = a.conds or list(CONDS)
    if a.build:
        for t in conds:
            build(cfg, t)
    if a.train:
        for t in conds:
            train(cfg, t, a.models, a.redo)
    if a.eval:
        evaluate(cfg)


if __name__ == "__main__":
    main()
