#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rà soát độc lập toàn bộ nghiên cứu: tính lại từ dữ liệu gốc, không tin file kết quả.

    python3 src/audit.py

Khác với check_consistency.py (kiểm bản thảo có tự mâu thuẫn không), script này
kiểm xem BẢN THÂN CÁC SỐ có hợp lý không: ma trận nhầm lẫn có cộng đúng không,
recall tính lại có khớp không, AP50 có mâu thuẫn với trần recall không, huấn
luyện có chạy đủ epoch không, các seed có thật sự khác nhau không.

Mã thoát 1 nếu có mục HỎNG.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import (MODEL_LABELS, RAE, RESULTS, RUNS, coco_json, load_cfg,
                    remap_categories, scope)

FAIL, WARN = [], []


def ok(m):   print(f"  ✓ {m}")
def bad(m):  FAIL.append(m); print(f"  ✗ {m}")
def warn(m): WARN.append(m); print(f"  ⚠ {m}")


def a1_dataset(cfg):
    """Nhãn: số ảnh, tỷ lệ chia, cân bằng lớp, rò rỉ."""
    print("\nA1 — bộ dữ liệu")
    from data_prep import _source_image
    for ann in ("bbox", "polygon"):
        src, n_img, n_ann = {}, {}, {}
        cls = collections.Counter()
        for sp in ("train", "valid", "test"):
            c = remap_categories(cfg, json.loads(coco_json(cfg, ann, sp).read_text()))
            src[sp] = {_source_image(i["file_name"]) for i in c["images"]}
            n_img[sp] = len(c["images"]); n_ann[sp] = len(c["annotations"])
            if sp == "test":
                cls = collections.Counter(a["category_id"] for a in c["annotations"])
        # rò rỉ
        leak = (src["test"] & src["train"]) | (src["valid"] & src["train"])
        if leak:
            bad(f"{ann}: {len(leak)} ảnh gốc xuất hiện ở cả train và valid/test")
        else:
            ok(f"{ann}: không rò rỉ train↔valid/test")
        # tỷ lệ theo ảnh GỐC
        tot = len(src["train"] | src["valid"] | src["test"])
        r = [len(src[s]) / tot * 100 for s in ("train", "valid", "test")]
        if abs(r[0] - 80) > 3 or abs(r[1] - 10) > 3 or abs(r[2] - 10) > 3:
            warn(f"{ann}: tỷ lệ ảnh gốc {r[0]:.0f}:{r[1]:.0f}:{r[2]:.0f}, lệch 80:10:10")
        else:
            ok(f"{ann}: tỷ lệ ảnh gốc {r[0]:.0f}:{r[1]:.0f}:{r[2]:.0f}")
        # tăng cường ×3 chỉ ở train
        exp = len(src["train"]) * 3
        if abs(n_img["train"] - exp) > len(src["train"]) * 0.05:
            warn(f"{ann}: train {n_img['train']} ảnh, kỳ vọng ×3 ảnh gốc = {exp}")
        else:
            ok(f"{ann}: train nhân đúng ×3 ({len(src['train'])}→{n_img['train']})")
        if n_img["valid"] != len(src["valid"]) or n_img["test"] != len(src["test"]):
            bad(f"{ann}: valid/test bị tăng cường — không được phép")
        else:
            ok(f"{ann}: valid/test KHÔNG tăng cường")
        bal = cls[1] / max(cls[0], 1)
        ok(f"{ann}: test có {cls[0]} mía / {cls[1]} cỏ (tỷ lệ cỏ:mía = {bal:.1f})")


def a2_matrix(cfg):
    """Ma trận nhầm lẫn: tổng hàng phải bằng số nhãn thật; recall tính lại phải khớp."""
    print("\nA2 — ma trận nhầm lẫn tính lại")
    primary, _ = scope(cfg)
    c = remap_categories(cfg, json.loads(coco_json(cfg, "bbox", "test").read_text()))
    gt = collections.Counter(a["category_id"] for a in c["annotations"])
    for k in primary:
        r = json.loads((RESULTS / f"{k}.json").read_text())
        M = np.array(r["confusion_matrix"], dtype=float)
        for ci, name in enumerate(cfg["dataset"]["classes"]):
            row = M[ci].sum()
            if abs(row - gt[ci]) > 0.5:
                bad(f"{MODEL_LABELS[k]}: hàng '{name}' cộng {row:.0f}, "
                    f"nhãn thật có {gt[ci]} — lệch {row-gt[ci]:+.0f}")
            else:
                ok(f"{MODEL_LABELS[k]}: hàng '{name}' cộng đúng {row:.0f} = số nhãn thật")
        # recall đầy đủ tính lại
        rec = M[1, 1] / M[1].sum()
        rep = r["metrics"]["with_background"]["per_class"]["weed"]["recall"]
        if abs(rec - rep) > 1e-6:
            bad(f"{MODEL_LABELS[k]}: recall cỏ tính lại {rec:.4f} ≠ báo cáo {rep:.4f}")
        else:
            ok(f"{MODEL_LABELS[k]}: recall cỏ tính lại khớp ({rec:.3f})")


def a3_ap_vs_recall(cfg):
    """AP50 không thể vượt xa trần recall — nếu vượt là dấu hiệu tính sai."""
    print("\nA3 — AP50 so với trần recall")
    primary, _ = scope(cfg)
    for k in primary:
        r = json.loads((RESULTS / f"{k}.json").read_text())
        wb = r["metrics"]["with_background"]["per_class"]
        ceil = (wb["sugarcane"]["recall"] + wb["weed"]["recall"]) / 2 * 100
        ap50 = r["coco"]["AP50"]
        if ap50 > ceil + 5:
            bad(f"{MODEL_LABELS[k]}: AP50 {ap50:.1f} vượt trần recall trung bình "
                f"{ceil:.1f} — không nhất quán")
        else:
            ok(f"{MODEL_LABELS[k]}: AP50 {ap50:.1f} ≤ trần recall {ceil:.1f} (hợp lý)")


def _run_status(run_name: str) -> str:
    """Trạng thái lượt chạy, đọc từ logs/seeds_status.tsv."""
    f = pathlib.Path("logs/seeds_status.tsv")
    if not f.exists():
        return "unknown"
    m = re.match(r"(.+)_seed(\d+)$", run_name)
    if not m:
        return "unknown"
    for line in f.read_text().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 3 and c[0] == m.group(1) and c[1] == m.group(2):
            return c[2]
    return "unknown"


def a4_training(cfg):
    """Huấn luyện có chạy đủ epoch, batch hiệu dụng có đúng như tuyên bố."""
    print("\nA4 — huấn luyện")
    primary, _ = scope(cfg)
    want_ep = cfg["train"]["epochs"]
    for k in primary:
        for d in sorted(RUNS.glob(f"{k}*")):
            if not d.is_dir():
                continue
            log = d / "log.txt"
            if log.exists():
                txt = log.read_text(errors="ignore")
                m = re.findall(r"iter: (\d+)", txt)
                mx = int(m[-1]) if m else 0
                mi = re.search(r"MAX_ITER: (\d+)", (d / "config.yaml").read_text()) \
                    if (d / "config.yaml").exists() else None
                target = int(mi.group(1)) if mi else None
                if target and mx < target * 0.95:
                    st = _run_status(d.name)
                    if st.startswith("diverged"):
                        ok(f"{d.name}: dừng ở iter {mx}/{target} — PHÂN KỲ đã ghi nhận "
                           f"({st}), loại khỏi phân tích và thay bằng seed khác")
                    else:
                        bad(f"{d.name}: dừng ở iter {mx}/{target}, không rõ lý do")
                elif target:
                    ok(f"{d.name}: chạy hết {mx}/{target} iter")
                acc = re.search(r"ACCUM_STEPS: (\d+)", (d / "config.yaml").read_text()) \
                    if (d / "config.yaml").exists() else None
                if acc:
                    micro = re.search(r"IMS_PER_BATCH: (\d+)", (d / "config.yaml").read_text())
                    eff = int(acc.group(1)) * int(micro.group(1))
                    if eff != cfg["train"]["batch_size"]:
                        bad(f"{d.name}: batch hiệu dụng {eff} ≠ {cfg['train']['batch_size']}")
            csv = d / "results.csv"
            if csv.exists():
                n = len(csv.read_text().strip().split("\n")) - 1
                if n < want_ep:
                    bad(f"{d.name}: chỉ {n}/{want_ep} epoch")
                else:
                    ok(f"{d.name}: đủ {n} epoch")


def a5_seeds(cfg):
    """Các seed phải thật sự khác nhau — nếu giống hệt là seed không có tác dụng."""
    print("\nA5 — seed có thật sự khác nhau")
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    for k, d in sv.items():
        v = [e["ap50"] for e in d["runs"] if e.get("status") == "ok"]
        if len(v) < 2:
            warn(f"{d['label']}: chỉ {len(v)} lượt, không đánh giá được")
        elif len(set(round(x, 4) for x in v)) == 1:
            bad(f"{d['label']}: {len(v)} lượt cho AP50 GIỐNG HỆT {v[0]:.2f} — "
                f"seed có thể không tác dụng")
        else:
            ok(f"{d['label']}: {len(v)} lượt khác nhau, AP50 {min(v):.1f}–{max(v):.1f}")


def a6_complexity(cfg):
    """Tham số/GFLOPs so với giá trị đã công bố của kiến trúc."""
    print("\nA6 — chi phí tính toán so với tài liệu")
    ref = {"faster_rcnn": (41.5, "R50-FPN ~41,5 M"),
           "retinanet": (37.9, "R50-FPN ~37,9 M"),
           "yolov10": (8.04, "YOLOv10-S ~8,0 M")}
    comp = json.loads((RESULTS / "complexity.json").read_text())
    for k, (want, note) in ref.items():
        if k not in comp:
            continue
        got = comp[k]["params"] / 1e6
        if abs(got - want) / want > 0.15:
            warn(f"{MODEL_LABELS[k]}: {got:.1f} M tham số, tài liệu {note} — lệch "
                 f"{abs(got-want)/want*100:.0f}%")
        else:
            ok(f"{MODEL_LABELS[k]}: {got:.1f} M tham số, khớp {note}")


def a7_speed(cfg):
    """FPS và ms/ảnh phải là nghịch đảo của nhau."""
    print("\nA7 — tốc độ")
    primary, _ = scope(cfg)
    for k in primary:
        p = RESULTS / f"edge_{k}.json"
        if not p.exists():
            warn(f"{MODEL_LABELS[k]}: chưa có phép đo lặp")
            continue
        d = json.loads(p.read_text())
        ms, fps = d["latency_ms"]["mean"], d["fps"]["mean"]
        if abs(1000 / ms - fps) > 0.5:
            bad(f"{MODEL_LABELS[k]}: {ms:.1f} ms nhưng {fps:.1f} FPS — không nghịch đảo")
        else:
            ok(f"{MODEL_LABELS[k]}: {ms:.1f} ms ↔ {fps:.1f} FPS nhất quán "
               f"({d['repeats']}×{d['n_images']} ảnh)")


def a8_bootstrap(cfg):
    """Điểm ước lượng phải nằm trong khoảng tin cậy của chính nó."""
    print("\nA8 — bootstrap")
    b = json.loads((RESULTS / "bootstrap.json").read_text())
    for k, m in b["models"].items():
        for metric in ("f1", "ap50", "accuracy"):
            if metric not in m:
                continue
            pt, (lo, hi) = m[metric]["point"], m[metric]["ci95"]
            if not (lo <= pt <= hi):
                bad(f"{m['label']} {metric}: điểm {pt:.3f} NGOÀI khoảng [{lo:.3f}, {hi:.3f}]")
            else:
                ok(f"{m['label']} {metric}: {pt:.3f} trong [{lo:.3f}, {hi:.3f}]")


def a9_same_testset(cfg):
    """Ba mô hình phải được chấm trên ĐÚNG cùng những ảnh — nếu không thì so sai."""
    print("\nA9 — ba mô hình có chấm trên cùng tập test không")
    primary, _ = scope(cfg)
    sets = {}
    for k in primary:
        r = json.loads((RESULTS / f"{k}.json").read_text())
        if "per_image" not in r:
            warn(f"{MODEL_LABELS[k]}: chưa lưu per_image, không kiểm được"); return
        sets[k] = {p["image_id"] for p in r["per_image"]}
    ref = sets[primary[0]]
    if all(sets[k] == ref for k in primary):
        ok(f"cả {len(primary)} mô hình chấm trên đúng {len(ref)} ảnh giống nhau")
    else:
        for k in primary:
            d = sets[k] ^ ref
            if d:
                bad(f"{MODEL_LABELS[k]}: lệch {len(d)} ảnh so với {MODEL_LABELS[primary[0]]}")


def a11_bootstrap_resamples(cfg):
    """Số lần lấy mẫu lại phải đủ, kể cả cho AP50.

    Vòng phản biện trước yêu cầu nâng AP50 từ 200 lên 1000 lần. Mặc định của
    script vẫn là 200, nên một lần chạy lại pipeline đã âm thầm hạ xuống và
    không có gì báo. Kiểm ở đây để lần sau hỏng là hỏng ồn ào.
    """
    print("\nA11 — số lần lấy mẫu lại của bootstrap")
    f = RESULTS / "bootstrap.json"
    if not f.exists():
        warn("chưa có bootstrap.json"); return
    b = json.loads(f.read_text())
    for key, want in (("n_boot", 1000), ("n_boot_ap50", 1000)):
        n = b.get(key, 0)
        if n < want:
            bad(f"{key} = {n}, cần ít nhất {want}")
        else:
            ok(f"{key} = {n}")


def a10_manuscript(cfg):
    """Số then chốt trong bản thảo phải khớp results/."""
    print("\nA10 — số trong bản thảo so với results/")
    # Gói nộp chỉ mang bản thảo của tạp chí đích, nên thử lần lượt; và bố cục
    # gói đặt mã ở code/ với bài ở gốc, nên thử cả hai gốc.
    names = ("AIIA_manuscript.md", "SugarTech_manuscript.md", "RAE_manuscript.md")
    ms = next((f for root in (RAE, RAE.parent) for n in names
               if (f := root / "manuscript" / n).exists()), None)
    if ms is None:
        warn("chưa có bản thảo"); return
    t = ms.read_text()
    import discussion as D
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    D.use_seed_variance(sv)
    primary, _ = scope(cfg)
    for k in primary:
        for metric, fmt in (("ap50", "{:.2f}"), ("weed_recall_full", "{:.3f}")):
            v = fmt.format(D.seed_mean(k, metric))
            if v in t:
                ok(f"{MODEL_LABELS[k]} {metric} = {v} có trong bài")
            else:
                bad(f"{MODEL_LABELS[k]} {metric} = {v} KHÔNG thấy trong bài")


def main():
    cfg = load_cfg()
    print("RÀ SOÁT ĐỘC LẬP — tính lại từ dữ liệu gốc")
    for f in (a1_dataset, a2_matrix, a3_ap_vs_recall, a4_training, a5_seeds,
              a6_complexity, a7_speed, a8_bootstrap, a9_same_testset, a10_manuscript,
              a11_bootstrap_resamples):
        try:
            f(cfg)
        except Exception as e:
            bad(f"{f.__name__} lỗi: {type(e).__name__}: {e}")
    print(f"\n{'='*62}\nHỎNG: {len(FAIL)}   CẢNH BÁO: {len(WARN)}")
    for m in FAIL:
        print(f"  ✗ {m}")
    for m in WARN:
        print(f"  ⚠ {m}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
