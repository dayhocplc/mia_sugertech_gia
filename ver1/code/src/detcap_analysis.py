#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trần số phát hiện mỗi ảnh có bó buộc RetinaNet không?

    python3 src/detcap_analysis.py

Detectron2 mặc định chỉ giữ 100 phát hiện cho mỗi ảnh (TEST.DETECTIONS_PER_IMAGE).
RetinaNet chạm trần đó ở 14-18 trên 124 ảnh test, nên bài đã nêu rằng average
precision của nó bị trần chặn. Phản biện vòng Sugar Tech nói đúng rằng nêu xong
thì phải chạy lại với trần cao hơn chứ không để đó: nếu AP50 tăng thì con số
đang báo cáo là con số của trần, không phải của kiến trúc.

Script dùng lại trọng số đã có, chỉ đổi trần rồi suy luận lại, nên không huấn
luyện gì thêm. Kết quả ghi ra results/detcap_analysis.json.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, RESULTS, load_cfg, log
from evaluate import load_test_set
from review_analysis import PRIMARY, WEED, match_image, prf

CAPS = [100, 300]
MODEL = "retinanet"


def predictor_with_cap(model_key, cfg, run_dir, conf, cap):
    """DefaultPredictor như predict.py dựng, nhưng đặt lại trần số phát hiện."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from common import RUNS

    run_cfg = (run_dir or (RUNS / model_key)) / "config.yaml"
    d = get_cfg()
    d.set_new_allowed(True)
    d.merge_from_file(str(run_cfg))
    w = (run_dir or (RUNS / model_key)) / "model_final.pth"
    d.MODEL.WEIGHTS = str(w)
    d.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(conf, 0.05)
    d.MODEL.RETINANET.SCORE_THRESH_TEST = min(conf, 0.05)
    d.TEST.DETECTIONS_PER_IMAGE = cap
    d.MODEL.RETINANET.TOPK_CANDIDATES_TEST = max(1000, cap * 10)
    return DefaultPredictor(d)


def score(predictor, items, conf, iou):
    """Recall/precision/F1 lớp cỏ, số phát hiện, và detection thô để tính AP."""
    import cv2
    tp = fp = fn = 0
    n_det = []
    raw = []
    n_at_cap = 0
    for _meta, path, gb, gc in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        inst = predictor(img)["instances"].to("cpu")
        db = inst.pred_boxes.tensor.numpy()
        dc = inst.pred_classes.numpy()
        ds = inst.scores.numpy()
        n_det.append(int((ds > conf).sum()))
        for b, c_, sc in zip(db, dc, ds):
            raw.append({"image_id": _meta["id"], "category_id": int(c_),
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(sc)})
        _h, pot, kept, used = match_image(gb, gc, db, dc, ds, conf, iou)
        is_w = gc == WEED
        tp += int(((pot == WEED) & is_w).sum())
        fn += int(is_w.sum()) - int(((pot == WEED) & is_w).sum())
        fp += int(((kept == WEED) & (~used)).sum())
        fp += int(((pot == WEED) & (~is_w)).sum())
    p, r, f = prf(tp, fp, fn)
    return ({"weed_precision": p, "weed_recall": r, "weed_f1": f,
             "tp": tp, "fp": fp, "fn": fn,
             "median_detections_per_image": float(np.median(n_det)),
             "max_detections_per_image": int(max(n_det)) if n_det else 0}, raw)


def main():
    cfg = load_cfg()
    ev = cfg["eval"]
    conf, iou = ev["conf_threshold"], ev["iou_threshold"]
    items, _ = load_test_set(cfg, "bbox", None)

    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    runs = [(r["seed"], MODEL if r["seed"] == 0 else f"{MODEL}_seed{r['seed']}")
            for r in sv[MODEL]["runs"] if r.get("status") == "ok"]
    log(f"{MODEL}: {len(runs)} lượt đã hội tụ · trần {CAPS}")

    out = {"model": MODEL, "caps": CAPS, "conf": conf, "iou": iou,
           "n_test_images": len(items),
           "note": ("Detectron2 mặc định giữ 100 phát hiện mỗi ảnh. Chạy lại cùng "
                    "trọng số với trần cao hơn để biết average precision và recall "
                    "đang bị trần chặn hay không. Không huấn luyện lại."),
           "by_cap": {}}
    for cap in CAPS:
        per = collections.defaultdict(list)
        for seed, stem in runs:
            rd = pathlib.Path("runs") / stem
            pr = predictor_with_cap(MODEL, cfg, rd, conf, cap)
            s, raw = score(pr, items, conf, iou)
            from evaluate import coco_ap
            ap = coco_ap(cfg, "bbox", raw)
            s["AP50"] = ap.get("AP50", float("nan"))
            s["AP"] = ap.get("AP", float("nan"))
            s["n_raw_detections"] = len(raw)
            for k, v in s.items():
                per[k].append(v)
            log(f"  trần {cap:4d} · {stem:18s} recall={s['weed_recall']:.3f} "
                f"precision={s['weed_precision']:.3f} "
                f"max/ảnh={s['max_detections_per_image']} AP50={s['AP50']:.2f}")
        out["by_cap"][str(cap)] = {
            "n_runs": len(runs),
            **{k: {"mean": float(np.mean(v)),
                   "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
               for k, v in per.items()}}

    a, b = (out["by_cap"][str(c)] for c in CAPS)
    out["delta_recall"] = b["weed_recall"]["mean"] - a["weed_recall"]["mean"]
    out["delta_f1"] = b["weed_f1"]["mean"] - a["weed_f1"]["mean"]
    out["delta_ap50"] = b["AP50"]["mean"] - a["AP50"]["mean"]

    p = RESULTS / "detcap_analysis.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"→ {p}")
    for c in CAPS:
        d = out["by_cap"][str(c)]
        log(f"trần {c:4d}: recall {d['weed_recall']['mean']:.3f} "
            f"precision {d['weed_precision']['mean']:.3f} "
            f"F1 {d['weed_f1']['mean']:.3f} AP50 {d['AP50']['mean']:.2f} "
            f"({d['n_raw_detections']['mean']:.0f} hộp thô)")
    log(f"chênh lệch recall {out['delta_recall']:+.4f}, F1 {out['delta_f1']:+.4f}, "
        f"AP50 {out['delta_ap50']:+.3f}")


if __name__ == "__main__":
    main()
