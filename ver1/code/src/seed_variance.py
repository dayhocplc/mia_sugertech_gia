#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo phương sai do HUẤN LUYỆN: chấm mọi lượt seed trên đúng tập test.

    python3 src/seed_variance.py              # chấm các lượt chưa chấm rồi tổng hợp
    python3 src/seed_variance.py --summary    # chỉ tổng hợp từ kết quả đã có

Vì sao cần script riêng: số AP50 in trong log huấn luyện là trên tập VALID
(Detectron2 đặt DATASETS.TEST = valid; Ultralytics in mAP50 của tập val).
Đem những số đó so với AP50 trên tập test là so hai tập khác nhau. Ở đây mọi
lượt đều được chấm lại trên cùng tập test bằng cùng giao thức.

Hai nguồn phương sai phải báo cáo RIÊNG:
  * tập test hữu hạn  → src/bootstrap.py  (lấy mẫu lại ảnh)
  * huấn luyện        → script này        (lấy mẫu lại seed)
Cộng chúng lại cho khoảng bất định tổng, dùng khi so sánh hai mô hình.

Ghi results/seed_variance.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import MODEL_LABELS, RESULTS, RUNS, load_cfg, log, save_json

BOX_MODELS = ["faster_rcnn", "retinanet", "yolov10"]
STATUS = pathlib.Path("logs/seeds_status.tsv")


def find_runs(model_key: str, cfg: dict) -> list[tuple[int, pathlib.Path | None, str]]:
    """Mọi lượt của một mô hình: (seed, thư mục hoặc None nếu hỏng, trạng thái)."""
    fw = cfg["models"][model_key]["framework"]
    final = "model_final.pth" if fw == "detectron2" else "weights/best.pt"
    out = [(0, RUNS / model_key, "ok")] if (RUNS / model_key / final).exists() else []
    for d in sorted(RUNS.glob(f"{model_key}_seed*")):
        m = re.search(r"_seed(\d+)$", d.name)
        if not m:
            continue
        seed = int(m.group(1))
        if (d / final).exists():
            out.append((seed, d, "ok"))
        else:
            out.append((seed, None, _status_of(model_key, seed)))
    return sorted(out)


def _status_of(model_key: str, seed: int) -> str:
    """Trạng thái thật của một lượt, đọc từ logs/seeds_status.tsv.

    Không có trọng số cuối KHÔNG đồng nghĩa với phân kỳ — lượt đang chạy cũng
    chưa có. Gộp hai thứ đó lại từng làm bảng tóm tắt báo sai số lượt phân kỳ.
    """
    if STATUS.exists():
        for line in STATUS.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 3 and f[0] == model_key and f[1] == str(seed):
                return f[2]
    return "in_progress_or_unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="store_true", help="không chấm lại, chỉ tổng hợp")
    a = ap.parse_args()

    cfg = load_cfg()
    from evaluate import evaluate

    data = {}
    for k in BOX_MODELS:
        runs = find_runs(k, cfg)
        entries = []
        for seed, d, status in runs:
            if d is None:
                entries.append({"seed": seed, "status": status})
                continue
            tag = k if seed == 0 else f"{k}_seed{seed}"
            res_path = RESULTS / f"{tag}.json"
            if not res_path.exists() and not a.summary:
                log(f"chấm {tag} trên tập test…")
                evaluate(k, cfg, run_dir=d, tag=tag)
            if res_path.exists():
                r = json.loads(res_path.read_text())
                pc = r["metrics"]["filtered"]["per_class"]
                e = {
                    "seed": seed, "status": "ok",
                    "ap50": r["coco"]["AP50"], "ap": r["coco"]["AP"],
                    "ap75": r["coco"].get("AP75"),
                    "f1": r["metrics"]["filtered"]["macro"]["f1"],
                    "accuracy": r["metrics"]["filtered"]["overall_accuracy"],
                    "weed_recall_full":
                        r["metrics"]["with_background"]["per_class"]["weed"]["recall"],
                    "sugarcane_recall_full":
                        r["metrics"]["with_background"]["per_class"]["sugarcane"]["recall"],
                }
                # Bảng 3 phải trung bình cùng số seed như Bảng 2; thiếu các trường
                # này thì hai bảng cạnh nhau lại đọc hai nguồn khác nhau.
                for c in ("sugarcane", "weed"):
                    e[f"{c}_precision"] = pc[c]["precision"]
                    e[f"{c}_recall"] = pc[c]["recall"]
                entries.append(e)
        data[k] = {"label": MODEL_LABELS[k], "runs": entries}

    # thống kê trên các lượt hội tụ
    for k, d in data.items():
        ok = [e for e in d["runs"] if e.get("status") == "ok"]
        d["n_converged"] = len(ok)
        d["n_diverged"] = sum(1 for e in d["runs"]
                              if str(e.get("status", "")).startswith("diverged"))
        d["n_incomplete"] = sum(1 for e in d["runs"]
                                if e.get("status") not in ("ok",)
                                and not str(e.get("status", "")).startswith("diverged"))
        for m in ("ap50", "ap", "ap75", "f1", "accuracy", "weed_recall_full",
                  "sugarcane_recall_full", "sugarcane_precision", "sugarcane_recall",
                  "weed_precision", "weed_recall"):
            v = np.array([e[m] for e in ok if m in e], float)
            if v.size:
                d[m] = {"mean": float(v.mean()),
                        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
                        "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}

    # gộp với phương sai tập test (bootstrap) — hai nguồn độc lập, cộng phương sai
    bpath = RESULTS / "bootstrap.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        for k, d in data.items():
            bm = b["models"].get(k, {})
            for m in ("ap50", "f1"):
                if m in d and d[m].get("sd") is not None and m in bm:
                    lo, hi = bm[m]["ci95"]
                    sd_test = (hi - lo) / 3.92          # KTC 95% ≈ ±1,96 sd
                    sd_seed = d[m]["sd"]
                    d[m]["sd_test_set"] = sd_test
                    d[m]["sd_training"] = sd_seed
                    d[m]["sd_combined"] = float(np.hypot(sd_test, sd_seed))

    save_json(data, RESULTS / "seed_variance.json")
    log(f"→ {RESULTS / 'seed_variance.json'}")

    print(f"\n{'mô hình':<14}{'lượt':>6}{'AP50 trung bình ± sd':>24}{'khoảng':>16}")
    for k, d in data.items():
        if "ap50" not in d:
            print(f"{d['label']:<14}{'—':>6}  chưa đủ dữ liệu"); continue
        s = d["ap50"]
        sd = f"{s['sd']:.2f}" if s["sd"] is not None else "–"
        note = f"  ({d['n_diverged']} lượt phân kỳ)" if d["n_diverged"] else ""
        print(f"{d['label']:<14}{s['n']:>6}{s['mean']:>17.2f} ± {sd:<5}"
              f"{s['min']:>7.1f}–{s['max']:<6.1f}{note}")
    print("\nPhân rã phương sai AP50 (điểm phần trăm):")
    for k, d in data.items():
        s = d.get("ap50", {})
        if "sd_combined" in s:
            print(f"  {d['label']:<13} tập test {s['sd_test_set']:.2f} | "
                  f"huấn luyện {s['sd_training']:.2f} | gộp {s['sd_combined']:.2f}")


if __name__ == "__main__":
    main()
