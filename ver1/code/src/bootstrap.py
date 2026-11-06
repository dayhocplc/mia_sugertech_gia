#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Khoảng tin cậy bootstrap và kiểm định ghép cặp giữa các mô hình.

    python3 src/bootstrap.py                    # ba mô hình bounding-box
    python3 src/bootstrap.py --n-boot 2000

Vì sao phải GHÉP CẶP: các mô hình được chấm trên CÙNG một tập ảnh, nên sai số
của chúng tương quan mạnh — ảnh khó thì mô hình nào cũng sai. So hai khoảng tin
cậy độc lập sẽ kết luận "không phân biệt được" ngay cả khi một mô hình thực sự
tốt hơn một cách nhất quán. Phép làm đúng là lấy mẫu lại ẢNH, tính lại chỉ số
cho mọi mô hình trên cùng mẫu đó, rồi lấy HIỆU SỐ — phần phương sai chung triệt
tiêu, chỉ còn phần khác biệt thật giữa hai mô hình.

Đây là phương sai do TẬP TEST HỮU HẠN. Phương sai do huấn luyện (khởi tạo, thứ
tự dữ liệu) là chuyện khác và phải đo bằng cách huấn luyện lại nhiều seed —
xem run_seeds.sh.

Ghi results/bootstrap.json.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import MODEL_LABELS, RESULTS, coco_json, load_cfg, log, remap_categories, save_json
from metrics import class_metrics

BOX_MODELS = ["faster_rcnn", "retinanet", "yolov10"]


def load_per_image(model_key: str) -> dict[int, np.ndarray]:
    r = json.loads((RESULTS / f"{model_key}.json").read_text())
    if "per_image" not in r:
        raise SystemExit(f"{model_key}.json chưa có 'per_image' — chạy lại src/evaluate.py")
    return {int(p["image_id"]): np.array(p["matrix"], dtype=np.int64) for p in r["per_image"]}


def load_dets(model_key: str) -> dict[int, list]:
    p = RESULTS / "dets" / f"{model_key}.json"
    if not p.exists():
        return {}
    by_img = collections.defaultdict(list)
    for d in json.loads(p.read_text())["detections"]:
        by_img[int(d["image_id"])].append(d)
    return by_img


def metrics_from_matrices(mats: list[np.ndarray], names: list[str]) -> dict:
    total = np.sum(mats, axis=0)
    m = class_metrics(total, names)
    return {
        "f1": m["filtered"]["macro"]["f1"],
        "accuracy": m["filtered"]["overall_accuracy"],
        "weed_recall_full": m["with_background"]["per_class"]["weed"]["recall"],
    }


def ap50_on_subset(cfg: dict, annotation: str, dets_by_img: dict, img_ids: list[int],
                   gt_cache: dict) -> float | None:
    """AP50 tính lại trên đúng tập ảnh được lấy mẫu (có lặp lại ảnh)."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None

    # Ảnh lặp lại phải được nhân bản thành id mới, nếu không COCOeval chỉ đếm một lần
    images, anns, dets, next_img, next_ann = [], [], [], 1, 1
    for k, iid in enumerate(img_ids, start=1):
        meta, gt_anns = gt_cache[iid]
        new_id = next_img; next_img += 1
        images.append(dict(meta, id=new_id))
        for a in gt_anns:
            anns.append(dict(a, id=next_ann, image_id=new_id)); next_ann += 1
        for d in dets_by_img.get(iid, []):
            dets.append(dict(d, image_id=new_id))
    if not dets or not anns:
        return None

    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in enumerate(cfg["dataset"]["classes"])]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt, f); path = f.name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = COCO(path); dt = c.loadRes(dets)
        e = COCOeval(c, dt, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    pathlib.Path(path).unlink(missing_ok=True)
    return float(e.stats[1]) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000, help="số lần lấy mẫu lại")
    # Vòng phản biện trước yêu cầu nâng từ 200 lên 1000; để mặc định 200 thì mỗi
    # lần chạy lại pipeline lại âm thầm hạ xuống. Mặc định nay là 1000.
    ap.add_argument("--n-boot-ap", type=int, default=1000,
                    help="số lần lấy mẫu lại cho AP50 (chậm hơn nhiều)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_cfg()
    names = cfg["dataset"]["classes"]
    rng = np.random.default_rng(a.seed)

    per = {k: load_per_image(k) for k in BOX_MODELS}
    img_ids = sorted(set.intersection(*(set(v) for v in per.values())))
    log(f"{len(img_ids)} ảnh test dùng chung cho cả ba mô hình")

    # nhãn thật, nạp một lần để dùng lại cho mọi lần lấy mẫu
    coco = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", "test").read_text()))
    by_img = collections.defaultdict(list)
    for an in coco["annotations"]:
        by_img[an["image_id"]].append(an)
    gt_cache = {i["id"]: (i, by_img.get(i["id"], [])) for i in coco["images"]}
    dets = {k: load_dets(k) for k in BOX_MODELS}

    point = {k: metrics_from_matrices([per[k][i] for i in img_ids], names) for k in BOX_MODELS}

    log(f"bootstrap {a.n_boot} lần cho chỉ số ma trận nhầm lẫn…")
    draws = {k: collections.defaultdict(list) for k in BOX_MODELS}
    idx_all = []
    for _ in range(a.n_boot):
        idx = rng.choice(len(img_ids), size=len(img_ids), replace=True)
        idx_all.append(idx)
        ids = [img_ids[j] for j in idx]
        for k in BOX_MODELS:
            for name, v in metrics_from_matrices([per[k][i] for i in ids], names).items():
                draws[k][name].append(v)

    log(f"bootstrap {a.n_boot_ap} lần cho AP50 (chậm hơn)…")
    ap_draws = {k: [] for k in BOX_MODELS}
    for n, idx in enumerate(idx_all[:a.n_boot_ap], 1):
        ids = [img_ids[j] for j in idx]
        for k in BOX_MODELS:
            v = ap50_on_subset(cfg, "bbox", dets[k], ids, gt_cache)
            if v is not None:
                ap_draws[k].append(v)
        if n % 50 == 0:
            log(f"  AP50 {n}/{a.n_boot_ap}")

    def ci(vals):
        v = np.asarray(vals, float)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {"n_images": len(img_ids), "n_boot": a.n_boot, "n_boot_ap50": a.n_boot_ap,
           "note": "Khoảng tin cậy 95% do TẬP TEST HỮU HẠN. Phương sai do huấn luyện "
                   "được đo riêng bằng nhiều seed.",
           "models": {}, "pairwise": {}}
    for k in BOX_MODELS:
        out["models"][k] = {"label": MODEL_LABELS[k],
                            **{m: {"point": point[k][m], "ci95": ci(draws[k][m])}
                               for m in point[k]}}
        if ap_draws[k]:
            out["models"][k]["ap50"] = {
                "point": json.loads((RESULTS / f"{k}.json").read_text())["coco"]["AP50"],
                "ci95": ci(ap_draws[k])}

    # Hiệu số ghép cặp: tính trên CÙNG mẫu, nên phương sai chung bị triệt tiêu
    for i, ka in enumerate(BOX_MODELS):
        for kb in BOX_MODELS[i + 1:]:
            d = {}
            for m in point[ka]:
                diff = np.array(draws[ka][m]) - np.array(draws[kb][m])
                lo, hi = ci(diff)
                d[m] = {"diff": point[ka][m] - point[kb][m], "ci95": [lo, hi],
                        "significant": bool(lo > 0 or hi < 0)}
            if ap_draws[ka] and ap_draws[kb]:
                n = min(len(ap_draws[ka]), len(ap_draws[kb]))
                diff = np.array(ap_draws[ka][:n]) - np.array(ap_draws[kb][:n])
                lo, hi = ci(diff)
                d["ap50"] = {"diff": out["models"][ka]["ap50"]["point"]
                             - out["models"][kb]["ap50"]["point"],
                             "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
            out["pairwise"][f"{ka}_vs_{kb}"] = d

    save_json(out, RESULTS / "bootstrap.json")
    log(f"→ {RESULTS / 'bootstrap.json'}")

    print(f"\n{'mô hình':<14}{'F1 [KTC 95%]':>26}{'AP50 [KTC 95%]':>26}")
    for k in BOX_MODELS:
        m = out["models"][k]
        f = m["f1"]; s = f"{f['point']:.3f} [{f['ci95'][0]:.3f}, {f['ci95'][1]:.3f}]"
        aa = m.get("ap50")
        t = f"{aa['point']:.1f} [{aa['ci95'][0]:.1f}, {aa['ci95'][1]:.1f}]" if aa else "–"
        print(f"{MODEL_LABELS[k]:<14}{s:>26}{t:>26}")
    print("\nHiệu số ghép cặp (KTC 95% không chứa 0 ⇒ khác biệt có ý nghĩa):")
    for pair, d in out["pairwise"].items():
        a_, b_ = pair.split("_vs_")
        for m in ("f1", "ap50"):
            if m in d:
                x = d[m]
                mark = "CÓ Ý NGHĨA" if x["significant"] else "không"
                print(f"  {MODEL_LABELS[a_]:<13} − {MODEL_LABELS[b_]:<13} {m:<6}"
                      f"{x['diff']:+8.3f}  [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]  {mark}")


if __name__ == "__main__":
    main()
