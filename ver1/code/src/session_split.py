#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đánh giá chia theo phiên chụp — đo trực tiếp phần recall đến từ rò rỉ.

    python3 src/session_split.py --build
    python3 src/session_split.py --train
    python3 src/session_split.py --eval

Phản biện vòng hai nêu đúng một điều mà bài chưa trả lời được: tập test được
chia NGẪU NHIÊN THEO ẢNH, trong khi ảnh được chụp thành từng chuỗi cách nhau
vài giây. Kiểm lại results/leakage_analysis.json: cả 27 burst có ảnh trong tập
test đều có ảnh trong tập train, nên không có một ảnh test nào thuộc burst mà
mô hình chưa từng thấy. Phép thử ORB/RANSAC chỉ loại được trùng lặp hình học,
không loại được tương quan về điều kiện. Vì vậy không thể ước lượng độ lớn của
rò rỉ trên các lượt chạy đã có — phải huấn luyện lại với cách chia khác.

THIẾT KẾ. Bộ ảnh được chụp trong ba phiên trên ba thửa (09h thửa 40 ngày, 10h
thửa 60 ngày, 13h thửa trộn). Với từng phiên s:

  grouped (hold<s>)  huấn luyện trên hai phiên còn lại, kiểm trên toàn bộ phiên s
  control (ctl<s>)   huấn luyện và kiểm trên tập chia NGẪU NHIÊN, cùng cỡ

Đối chứng là phần không thể thiếu: tập huấn luyện của một lượt grouped nhỏ hơn
tập huấn luyện của bài, nên nếu chỉ chạy grouped thì không tách được "ít dữ liệu
hơn" khỏi "chia theo nhóm". Hai điều kiện chỉ khác nhau ở cách gán ảnh, mọi thứ
còn lại giữ nguyên, nên hiệu giữa chúng mới là phần quy được cho rò rỉ.

Mỗi ảnh gốc chỉ lấy MỘT bản (Roboflow nhân ba bản tăng cường cho các ảnh thuộc
tập train của bài, còn valid/test chỉ có một bản; giữ cả ba sẽ làm hai điều kiện
không còn cùng cỡ). Vì vậy con số tuyệt đối của thí nghiệm này thấp hơn con số
trong bài và chỉ so sánh được NỘI BỘ giữa hai điều kiện.

Một ảnh gốc không có EXIF nên không gán được phiên; nó bị loại khỏi thí nghiệm.

Kết quả ghi ra results/session_split.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, RAE, RESULTS, RUNS, coco_json, dataset_root, load_cfg, log, remap_categories
from data_prep import _source_image

FOLDS = RAE / "data_coco" / "folds"
YFOLDS = RAE / "data_yolo" / "folds"
IMGDIR = FOLDS / "images"
MODELS = ["faster_rcnn", "retinanet", "yolov10"]
VALID_FRACTION = 0.10
SPLIT_SEED = 0

SESSION_TAG = {"09h (40-day plot)": "09", "10h (60-day plot)": "10",
               "13h (mixed plot)": "13"}


# ───────────────────────────── dựng các fold ────────────────────────────────

def base_images(cfg) -> dict[str, tuple[str, dict, list]]:
    """Một bản duy nhất cho mỗi ảnh gốc: (thư mục split, metadata, annotation).

    Lấy bản có tên sắp xếp đầu tiên. Roboflow không đặt tên phân biệt bản gốc
    với bản tăng cường, nên "bản đầu tiên" là quy tắc duy nhất khả thi và nó
    áp dụng như nhau cho mọi fold.
    """
    best = {}
    for split in ("train", "valid", "test"):
        coco = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", split).read_text()))
        by_img = collections.defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in coco["images"]:
            src = _source_image(im["file_name"])
            cur = best.get(src)
            if cur is None or im["file_name"] < cur[1]["file_name"]:
                best[src] = (split, im, by_img.get(im["id"], []))
    return best


def _write_coco(tag: str, split: str, srcs: list[str], base, cfg) -> pathlib.Path:
    """COCO cho một split của một fold, id đánh lại từ 0, ảnh trỏ vào IMGDIR."""
    images, anns = [], []
    for i, s in enumerate(sorted(srcs)):
        _sp, im, aa = base[s]
        images.append({**im, "id": i})
        for a in aa:
            anns.append({**a, "id": len(anns) + 1, "image_id": i})
    out = {"info": {}, "licenses": [],
           "categories": [{"id": j, "name": n, "supercategory": "none"}
                          for j, n in enumerate(cfg["dataset"]["classes"])],
           "images": images, "annotations": anns}
    p = FOLDS / tag / f"{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return p


def _link_images(base, cfg) -> None:
    """Một thư mục phẳng chứa symlink tới mọi ảnh gốc được dùng."""
    IMGDIR.mkdir(parents=True, exist_ok=True)
    for src, (split, im, _a) in base.items():
        dst = IMGDIR / im["file_name"]
        if dst.exists() or dst.is_symlink():
            continue
        s = dataset_root(cfg, "bbox") / split / im["file_name"]
        if s.exists():
            dst.symlink_to(s.resolve())


def _to_yolo(tag: str, cfg) -> None:
    """Bố cục YOLO cho một fold, nhãn sinh lại từ COCO của fold."""
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


def build(cfg) -> dict:
    base = base_images(cfg)
    lk = json.loads((RESULTS / "leakage_analysis.json").read_text())["image_clusters"]
    sess = {s: SESSION_TAG[lk[s]["session"]] for s in base if s in lk}
    dropped = sorted(set(base) - set(sess))
    log(f"{len(sess)} ảnh gốc có phiên chụp, loại {len(dropped)} ảnh không có EXIF")

    by_sess = collections.defaultdict(list)
    for s, t in sess.items():
        by_sess[t].append(s)
    allsrc = sorted(sess)

    out = {"valid_fraction": VALID_FRACTION, "split_seed": SPLIT_SEED,
           "n_sources_used": len(sess), "dropped_sources": dropped,
           "session_sizes": {t: len(v) for t, v in sorted(by_sess.items())},
           "folds": {}}
    for t in sorted(by_sess):
        test_g = sorted(by_sess[t])
        pool_g = [s for s in allsrc if s not in set(test_g)]
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(pool_g)
        n_val = max(1, round(len(pool_g) * VALID_FRACTION))
        folds = {f"hold{t}": {"condition": "grouped", "held_out_session": t,
                              "test": test_g, "valid": sorted(pool_g[:n_val]),
                              "train": sorted(pool_g[n_val:])}}
        # đối chứng: cùng cỡ ba split, nhưng gán ngẫu nhiên trên toàn bộ ảnh
        rng2 = random.Random(SPLIT_SEED + 1000 + int(t))
        shuf = list(allsrc)
        rng2.shuffle(shuf)
        folds[f"ctl{t}"] = {"condition": "random", "held_out_session": None,
                            "test": sorted(shuf[:len(test_g)]),
                            "valid": sorted(shuf[len(test_g):len(test_g) + n_val]),
                            "train": sorted(shuf[len(test_g) + n_val:])}
        out["folds"].update(folds)

    _link_images(base, cfg)
    for tag, f in out["folds"].items():
        for split in ("train", "valid", "test"):
            _write_coco(tag, split, f[split], base, cfg)
        _to_yolo(tag, cfg)
        log(f"{tag:9s} {f['condition']:7s} train {len(f['train']):4d} "
            f"valid {len(f['valid']):3d} test {len(f['test']):4d}")
    for f in out["folds"].values():
        for split in ("train", "valid", "test"):
            f[f"n_{split}"] = len(f[split])
    p = FOLDS / "splits.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"→ {p}")
    return out


# ──────────────────────────── huấn luyện lại ────────────────────────────────

def train(cfg, only=None, redo=False) -> None:
    tags = sorted(json.loads((FOLDS / "splits.json").read_text())["folds"])
    for tag in tags:
        for m in MODELS:
            if only and m not in only:
                continue
            rd = RUNS / f"{m}_fold{tag}"
            done = (rd / "model_final.pth").exists() or (rd / "weights/last.pt").exists()
            if done and not redo:
                log(f"bỏ qua {m} {tag}: đã có trọng số")
                continue
            script = ("train_ultralytics.py" if cfg["models"][m]["framework"] == "ultralytics"
                      else "train_detectron2.py")
            cmd = [sys.executable, f"src/{script}", "--model", m, "--fold", tag]
            log("$ " + " ".join(cmd))
            r = subprocess.run(cmd, cwd=str(RAE))
            if r.returncode:
                log(f"LỖI {m} {tag}: mã trả về {r.returncode}")


# ───────────────────────────── đánh giá fold ────────────────────────────────

def load_fold_test(tag: str):
    coco = json.loads((FOLDS / tag / "test.json").read_text())
    by = collections.defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by[a["image_id"]].append((np.array([x, y, x + w, y + h], float), a["category_id"]))
    items = []
    for im in coco["images"]:
        aa = by.get(im["id"], [])
        items.append((im, IMGDIR / im["file_name"],
                      np.array([b for b, _ in aa], float).reshape(-1, 4),
                      np.array([c for _, c in aa], int)))
    return items, coco


def fold_ap(tag: str, dets: list) -> dict:
    """AP50/AP trên COCO của fold. Trả {} nếu không có pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {}
    if not dets:
        return {}
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = COCO(str(FOLDS / tag / "test.json"))
        e = COCOeval(c, c.loadRes(dets), "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    return {"AP": float(e.stats[0]) * 100, "AP50": float(e.stats[1]) * 100}


def evaluate_folds(cfg) -> dict:
    import cv2
    from predict import build_predictor
    from review_analysis import WEED, match_image, prf

    sp = json.loads((FOLDS / "splits.json").read_text())
    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    out = {"conf": conf, "iou": iou, "valid_fraction": VALID_FRACTION,
           "note": ("Mỗi ảnh gốc một bản, không dùng bản tăng cường, nên con số tuyệt "
                    "đối thấp hơn trong bài; chỉ hiệu giữa grouped và control có nghĩa."),
           "session_sizes": sp["session_sizes"], "folds": {}}
    for tag, f in sorted(sp["folds"].items()):
        items, _coco = load_fold_test(tag)
        out["folds"][tag] = {"condition": f["condition"],
                             "held_out_session": f["held_out_session"],
                             "n_train": f["n_train"], "n_test": f["n_test"],
                             "models": {}}
        for m in MODELS:
            rd = RUNS / f"{m}_fold{tag}"
            try:
                pr = build_predictor(m, cfg, conf=0.05, run_dir=rd)
            except SystemExit as e:
                log(f"thiếu trọng số {m} {tag}: {e}")
                continue
            tp = fp = fn = 0
            dets = []
            for meta, path, gb, gc in items:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                p = pr(img)
                db = np.asarray(p.boxes, float).reshape(-1, 4)
                dc = np.asarray(p.classes, int)
                ds = np.asarray(p.scores, float)
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
            rec = {"weed_precision": p_, "weed_recall": r_, "weed_f1": f_,
                   "tp": tp, "fp": fp, "fn": fn, **fold_ap(tag, dets)}
            out["folds"][tag]["models"][m] = rec
            log(f"{tag:9s} {m:12s} recall={r_:.3f} precision={p_:.3f} "
                f"F1={f_:.3f} AP50={rec.get('AP50', float('nan')):.1f}")

    aggregate(out)
    return out


def _and_list_en(xs: list[str]) -> str:
    return (xs[0] if len(xs) == 1
            else " and ".join([", ".join(xs[:-1]), xs[-1]]))


def _degenerate(rec: dict) -> str | None:
    """Lượt huấn luyện sập, không phải kết quả đo được.

    RetinaNet dùng focal loss với AdamW lr=1e-3 nằm sát ngưỡng ổn định: ba
    fold đối chứng phân kỳ ngay khi hết warmup (NaN ở iter 185 và 267) hoặc
    thu về mô hình không dự đoán gì. Một lượt như thế không được coi là số
    đo, vì hiệu grouped − control tính trên nó sẽ dương một cách giả tạo.
    """
    if rec.get("tp", 0) == 0:
        return "mô hình không tìm được cỏ nào — lượt huấn luyện sập"
    ap = rec.get("AP50")
    if ap is None or ap != ap:
        return "AP50 không xác định — lượt huấn luyện sập"
    return None


def _clipped_models(out: dict) -> list[str]:
    """Mô hình nào đã bật cắt gradient, đọc từ config.yaml của chính lượt chạy.

    Đọc từ tệp thay vì viết tay để ghi chú trong bài không thể lệch với thứ
    đã thực sự chạy.
    """
    import yaml
    hit = []
    for m in MODELS:
        tags = [t for t in out["folds"] if m in out["folds"][t].get("models", {})]
        if not tags:
            continue
        on = []
        for t in tags:
            f = RUNS / f"{m}_fold{t}" / "config.yaml"
            if not f.exists():
                on.append(False)
                continue
            d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            on.append(bool(d.get("SOLVER", {}).get("CLIP_GRADIENTS", {}).get("ENABLED")))
        if on and all(on):
            hit.append(m)
    return hit


def aggregate(out: dict) -> dict:
    """Hiệu grouped − control, ghép theo cỡ tập huấn luyện."""
    for tag, f in out["folds"].items():
        for m, rec in f.get("models", {}).items():
            why = _degenerate(rec)
            if why:
                rec["failed"] = why
                log(f"bỏ {tag} {m}: {why}")
    out["grouped_minus_control"] = {}
    for t in sorted(out["session_sizes"]):
        g = out["folds"].get(f"hold{t}", {})
        c = out["folds"].get(f"ctl{t}", {})
        d = {}
        for m in MODELS:
            gm = g.get("models", {}).get(m)
            cm = c.get("models", {}).get(m)
            if not gm or not cm or gm.get("failed") or cm.get("failed"):
                continue
            d[m] = {k: gm[k] - cm[k] for k in ("weed_recall", "weed_precision", "weed_f1")
                    if k in gm and k in cm}
            if "AP50" in gm and "AP50" in cm:
                d[m]["AP50"] = gm["AP50"] - cm["AP50"]
        out["grouped_minus_control"][t] = d
    mean, npairs = {}, {}
    for m in MODELS:
        vals = collections.defaultdict(list)
        for t, d in out["grouped_minus_control"].items():
            for k, v in d.get(m, {}).items():
                vals[k].append(v)
        if vals:
            mean[m] = {k: float(np.mean(v)) for k, v in vals.items()}
            npairs[m] = len(vals["weed_recall"])
    out["mean_grouped_minus_control"] = mean
    out["n_pairs"] = npairs
    out["excluded"] = {f"{tag}/{m}": rec["failed"]
                       for tag, f in sorted(out["folds"].items())
                       for m, rec in sorted(f.get("models", {}).items())
                       if rec.get("failed")}
    clipped = [m for m in _clipped_models(out) if m in mean]
    out["clipped_models"] = clipped
    out["recipe_note"] = ("" if not clipped else
        _and_list_en([MODEL_LABELS[m] for m in clipped])
        + (" was" if len(clipped) == 1 else " were")
        + " trained with gradient-norm clipping in all six folds of this experiment, "
        "because without it the focal-loss objective diverged as the learning rate "
        "reached its full value on the smaller fold partitions; clipping was applied to "
        "both members of every pair, so the difference within a pair is unaffected, but "
        f"the absolute values for {_and_list_en([MODEL_LABELS[m] for m in clipped])} "
        "are not directly comparable with those of the other architectures here.")

    p = RESULTS / "session_split.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"→ {p}")
    for m, d in mean.items():
        log(f"{m:14s} {npairs[m]}/3 cặp · hiệu trung bình "
            f"recall {d.get('weed_recall', float('nan')):+.3f} "
            f"F1 {d.get('weed_f1', float('nan')):+.3f} "
            f"AP50 {d.get('AP50', float('nan')):+.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--models", nargs="*", choices=MODELS)
    ap.add_argument("--redo", action="store_true", help="huấn luyện lại dù đã có trọng số")
    ap.add_argument("--reagg", action="store_true",
                    help="tính lại hiệu từ results/session_split.json, không suy luận lại")
    a = ap.parse_args()
    cfg = load_cfg()
    if not (a.build or a.train or a.eval or a.reagg):
        raise SystemExit("Chọn --build, --train, --eval hoặc --reagg")
    if a.build:
        build(cfg)
    if a.train:
        train(cfg, a.models, a.redo)
    if a.eval:
        evaluate_folds(cfg)
    if a.reagg:
        aggregate(json.loads((RESULTS / "session_split.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
