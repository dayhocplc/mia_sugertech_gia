#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo mức rò rỉ do chia tập ngẫu nhiên trên ảnh chụp liên tục.

    python3 src/leakage_analysis.py

Phản biện nêu nguy cơ rò rỉ vì ảnh cùng trang trại, cùng lô, cùng đợt chụp rất
tương quan. Dấu thời gian trong tên tệp (xác nhận bằng EXIF của ảnh gốc) cho
thấy toàn bộ 1 210/1 211 ảnh nguồn được chụp trong MỘT ngày (2025-03-20, từ
09:09 đến 13:44) và trung vị khoảng cách giữa hai lần bấm máy là 2 giây. Mọi
ảnh test đều có một ảnh train chụp cách dưới 30 giây.

Nhưng khoảng cách thời gian KHÔNG phải bằng chứng trùng lặp. Phép đo chồng lấn
hình học (ORB + homography RANSAC trên ảnh gốc chưa augment) cho thấy các khung
cách nhau ≤5 giây hầu như không chia sẻ cấu trúc nào: 56/60 cặp lấy mẫu có tỷ
lệ inlier dưới 5%, ngang với các cặp cách nhau hơn 10 phút. Ở góc chụp thẳng
đứng cách mặt đất 140 cm, một hai bước chân đã cho khuôn hình hoàn toàn khác.
Vậy KHÔNG có bằng chứng rò rỉ dạng ảnh gần-trùng; hạn chế thật nằm ở chỗ khác:
một trang trại, một ngày, ba buổi chụp, nên các ảnh tương quan về điều kiện
chiếu sáng và về lô — điều này giới hạn khả năng khái quát, không phải trùng lặp.

Script này:
  1. dựng lại dấu thời gian cho mọi ảnh nguồn của cả ba phần chia;
  2. với mỗi ảnh test, tính khoảng cách thời gian tới ảnh train gần nhất;
  3. đo chồng lấn hình học giữa các khung liền kề thời gian, so với nền;
  4. báo cáo chỉ số theo từng BUỔI CHỤP (phản biện nêu bài thiếu kết quả theo
     điều kiện), kèm số ảnh mỗi buổi trong từng phần chia.

Kết quả ghi ra results/leakage_analysis.json.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, RESULTS, load_cfg, coco_json, log

PRIMARY = ["faster_rcnn", "retinanet", "yolov10"]
WEED = 1
TS = re.compile(r'_?(\d{8})_(\d{6})')
# ngưỡng khoảng cách thời gian tới ảnh train gần nhất (giây)
GAPS = [0, 5, 15, 30, 60, 120, 300]
# khoảng trống mở một cụm chụp mới
BURST_GAP_S = 30


def source_name(file_name: str) -> str:
    return re.sub(r'_jpg\.rf\.[0-9a-f]+\.jpg$', '', file_name)


def timestamps(cfg) -> dict[str, dict]:
    """Dấu thời gian và phần chia của mọi ảnh nguồn trong bản nhãn hộp."""
    out = {}
    for split in ("train", "valid", "test"):
        coco = json.loads(coco_json(cfg, "bbox", split).read_text())
        for im in coco["images"]:
            src = source_name(im["file_name"])
            m = TS.search(src)
            ts = (dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                  if m else None)
            out.setdefault(src, {"split": split, "ts": ts, "image_ids": []})
            out[src]["image_ids"].append(im["id"])
    return out


def per_image_counts(model_stem: str, cfg):
    """TP/FP/FN lớp cỏ theo từng ảnh test, lấy từ ma trận đã lưu."""
    d = json.loads((RESULTS / f"{model_stem}.json").read_text())
    out = {}
    for rec in d["per_image"]:
        m = np.array(rec["matrix"])
        tp = m[WEED, WEED]
        fp = m[:, WEED].sum() - tp
        fn = m[WEED, :].sum() - tp
        out[rec["image_id"]] = (int(tp), int(fp), int(fn))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def augmentation_consistency(cfg) -> dict:
    """Ba bản augment của cùng một ảnh nguồn có cùng số instance không?

    Random crop 0-20% làm hộp vượt biên bị loại khi xuất, nên số instance lệch
    nhau — đây là bằng chứng cho thấy recipe augmentation không chỉ là lật và
    quay 90° như bản thảo trước viết.
    """
    coco = json.loads(coco_json(cfg, "bbox", "train").read_text())
    cnt = collections.Counter(a["image_id"] for a in coco["annotations"])
    grp = collections.defaultdict(list)
    for im in coco["images"]:
        grp[source_name(im["file_name"])].append(im["id"])
    differ = sum(1 for ids in grp.values() if len({cnt.get(i, 0) for i in ids}) > 1)
    sizes = collections.Counter(len(v) for v in grp.values())
    return {"n_source_images": len(grp),
            "versions_per_source": dict(sizes),
            "n_sources_count_differs": differ,
            "pct_sources_count_differs": 100 * differ / len(grp) if grp else 0.0,
            "total_train_instances": int(sum(cnt.values())),
            "divisible_by_3": int(sum(cnt.values())) % 3 == 0}


def session_of(t: dt.datetime) -> str:
    """Ba buổi chụp của ngày 2025-03-20, khớp ba thư mục ảnh gốc:
    09h → lô mía 40 ngày, 10h → lô 60 ngày, 13h → lô hỗn hợp."""
    return {9: "09h (40-day plot)", 10: "10h (60-day plot)"}.get(t.hour, "13h (mixed plot)")


def frame_overlap(cfg, n_pairs: int = 60, adj_s: int = 5, far_s: int = 600) -> dict:
    """Chồng lấn hình học giữa các khung liền kề thời gian, trên ảnh GỐC.

    Dùng ORB + homography RANSAC. So cặp cách ≤adj_s giây với cặp cách >far_s
    giây: nếu hai phân bố như nhau thì gần kề thời gian không kéo theo trùng
    lặp nội dung, và nguy cơ rò rỉ mà phản biện nêu là nhỏ.
    """
    import random
    import cv2
    base = pathlib.Path(cfg["dataset"].get("raw_picture_root", ""))
    if not base.is_dir():
        return {"note": "không tìm thấy thư mục ảnh gốc; bỏ qua phép đo chồng lấn"}
    items = []
    for f in sorted(base.rglob("*.jpg")):
        m = re.search(r"(\d{8})_(\d{6})", f.name)
        if m:
            items.append((dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), f))
    items.sort()
    if len(items) < 20:
        return {"note": f"chỉ thấy {len(items)} ảnh gốc; bỏ qua"}

    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cache = {}

    def feat(p):
        if p not in cache:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            cache[p] = orb.detectAndCompute(cv2.resize(im, (1020, 765)), None) if im is not None else (None, None)
        return cache[p]

    def ratio(p1, p2):
        k1, d1 = feat(p1); k2, d2 = feat(p2)
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
            return 0.0
        ms = bf.match(d1, d2)
        if len(ms) < 10:
            return 0.0
        a = np.float32([k1[x.queryIdx].pt for x in ms]).reshape(-1, 1, 2)
        b = np.float32([k2[x.trainIdx].pt for x in ms]).reshape(-1, 1, 2)
        _H, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
        return float(mask.sum()) / min(len(k1), len(k2)) if mask is not None else 0.0

    rng = random.Random(0)
    adj = [(items[i][1], items[i + 1][1]) for i in range(len(items) - 1)
           if (items[i + 1][0] - items[i][0]).total_seconds() <= adj_s]
    far = []
    for _ in range(n_pairs):
        i = rng.randrange(len(items))
        cand = [j for j in range(len(items))
                if abs((items[j][0] - items[i][0]).total_seconds()) > far_s]
        if cand:
            far.append((items[i][1], items[rng.choice(cand)][1]))
    A = np.array([ratio(*x) for x in rng.sample(adj, min(n_pairs, len(adj)))])
    B = np.array([ratio(*x) for x in far])
    return {"adjacent_max_s": adj_s, "far_min_s": far_s,
            "n_adjacent": int(len(A)), "n_far": int(len(B)),
            "adjacent_median": float(np.median(A)), "far_median": float(np.median(B)),
            "adjacent_gt_005": int((A > 0.05).sum()), "far_gt_005": int((B > 0.05).sum()),
            "adjacent_gt_015": int((A > 0.15).sum()), "far_gt_015": int((B > 0.15).sum())}


def session_metrics(cfg, have: dict) -> dict:
    """Số ảnh và chỉ số lớp cỏ theo từng buổi chụp."""
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    test_session = {}
    counts = collections.defaultdict(lambda: collections.Counter())
    for k, v in have.items():
        nm = session_of(v["ts"])
        counts[nm][v["split"]] += 1
        if v["split"] == "test":
            for i in v["image_ids"]:
                test_session[i] = nm

    out = {}
    for nm in sorted(counts):
        keep = {i for i, s in test_session.items() if s == nm}
        models = {}
        for m in PRIMARY:
            seeds = [r["seed"] for r in sv[m]["runs"] if r.get("status") == "ok"]
            R, P, F = [], [], []
            for s in seeds:
                stem = m if s == 0 else f"{m}_seed{s}"
                if not (RESULTS / f"{stem}.json").exists():
                    continue
                cnt = per_image_counts(stem, cfg)
                tp = fp = fn = 0
                for iid, (a, b, c) in cnt.items():
                    if iid in keep:
                        tp += a; fp += b; fn += c
                p, r, f = prf(tp, fp, fn)
                P.append(p); R.append(r); F.append(f)
            models[m] = {"precision": float(np.mean(P)), "recall": float(np.mean(R)),
                         "f1": float(np.mean(F)), "n_runs": len(R)}
        out[nm] = {"n_train": counts[nm]["train"], "n_valid": counts[nm]["valid"],
                   "n_test": counts[nm]["test"], "n_test_images": len(keep),
                   "models": models}
    return out


def main():
    cfg = load_cfg()
    ts = timestamps(cfg)
    have = {k: v for k, v in ts.items() if v["ts"]}
    log(f"ảnh nguồn: {len(ts)}, có dấu thời gian: {len(have)}")

    days = collections.Counter(v["ts"].date() for v in have.values())
    allt = sorted(v["ts"] for v in have.values())
    log(f"số ngày chụp: {len(days)}; từ {allt[0]} đến {allt[-1]}")

    train_t = sorted(v["ts"] for v in have.values() if v["split"] == "train")
    test_src = {k: v for k, v in have.items() if v["split"] == "test"}
    train_arr = np.array([t.timestamp() for t in train_t])

    # khoảng cách thời gian từ mỗi ảnh test tới ảnh train gần nhất
    dist = {}
    for k, v in test_src.items():
        d = np.abs(train_arr - v["ts"].timestamp()).min()
        dist[k] = float(d)
    dvals = np.array(sorted(dist.values()))
    log(f"khoảng cách test→train gần nhất: trung vị {np.median(dvals):.0f}s, "
        f"≤2s: {(dvals <= 2).sum()}, ≤5s: {(dvals <= 5).sum()}, "
        f"≤30s: {(dvals <= 30).sum()} / {len(dvals)} ảnh test")

    # cụm chụp: khoảng trống > BURST_GAP_S mở cụm mới
    order = sorted(have.items(), key=lambda kv: kv[1]["ts"])
    burst, b = {}, 0
    for i, (k, v) in enumerate(order):
        if i and (v["ts"] - order[i - 1][1]["ts"]).total_seconds() > BURST_GAP_S:
            b += 1
        burst[k] = b
    n_burst = b + 1
    # cụm nào bị cắt ngang giữa các phần chia
    bysplit = collections.defaultdict(set)
    for k, v in have.items():
        bysplit[burst[k]].add(v["split"])
    split_bursts = sum(1 for s in bysplit.values() if len(s) > 1)
    log(f"số cụm chụp (khoảng trống > {BURST_GAP_S}s): {n_burst}; "
        f"bị cắt ngang giữa các phần chia: {split_bursts}")

    out = {
        "capture_days": {str(d): n for d, n in sorted(days.items())},
        "capture_window": {"start": str(allt[0]), "end": str(allt[-1]),
                           "span_hours": (allt[-1] - allt[0]).total_seconds() / 3600},
        "n_source_images": len(ts),
        "n_with_timestamp": len(have),
        "consecutive_gap_s": {
            "median": float(np.median(np.diff([t.timestamp() for t in allt]))),
            "le_2s": int((np.diff([t.timestamp() for t in allt]) <= 2).sum()),
            "le_5s": int((np.diff([t.timestamp() for t in allt]) <= 5).sum()),
            "n_pairs": len(allt) - 1,
        },
        "test_to_train_distance_s": {
            "median": float(np.median(dvals)),
            "le_2s": int((dvals <= 2).sum()), "le_5s": int((dvals <= 5).sum()),
            "le_30s": int((dvals <= 30).sum()), "n_test": int(len(dvals)),
        },
        "bursts": {"gap_threshold_s": BURST_GAP_S, "n_bursts": n_burst,
                   "n_bursts_spanning_splits": split_bursts},
        # Gán cụm cho TỪNG ảnh, để bootstrap theo cụm dùng lại được. Phản biện
        # nêu đúng rằng bootstrap coi 124 ảnh test là độc lập thì khoảng tin cậy
        # quá hẹp, vì ảnh trong cùng một cụm chụp cách nhau vài giây.
        "image_clusters": {k: {"burst": int(burst[k]),
                               "session": session_of(v["ts"]),
                               "split": v["split"]}
                           for k, v in have.items()},
        "metrics_by_isolation": {},
    }

    # ───── hệ quả của random crop: số instance lệch giữa ba bản ─────
    out["augmentation"] = augmentation_consistency(cfg)

    # ───── mô tả ngày và buổi chụp, dùng trực tiếp trong Methods ─────
    sess = collections.Counter(session_of(v["ts"]) for v in have.values())
    out["capture_date"] = str(allt[0].date())
    span = collections.defaultdict(list)
    for v in have.values():
        span[session_of(v["ts"])].append(v["ts"])
    out["sessions_desc"] = ", ".join(
        f"{min(t).strftime('%H:%M')}–{max(t).strftime('%H:%M')}"
        for _name, t in sorted(span.items()))
    out["session_counts"] = dict(sorted(sess.items()))

    # ───── chồng lấn hình học giữa các khung liền kề thời gian ─────
    out["frame_overlap"] = frame_overlap(cfg)

    # ───── chỉ số theo buổi chụp ─────
    out["sessions"] = session_metrics(cfg, have)

    p = RESULTS / "leakage_analysis.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"→ {p}")

    ov = out["frame_overlap"]
    print()
    print(f"Chồng lấn hình học (tỷ lệ inlier ORB+homography) trên ảnh gốc:")
    print(f"  cặp cách ≤{ov['adjacent_max_s']}s   (n={ov['n_adjacent']}): trung vị "
          f"{ov['adjacent_median']:.3f}, trên 5%: {ov['adjacent_gt_005']}")
    print(f"  cặp cách >{ov['far_min_s']}s (n={ov['n_far']}): trung vị "
          f"{ov['far_median']:.3f}, trên 5%: {ov['far_gt_005']}")
    print()
    print("Số ảnh nguồn và chỉ số lớp cỏ theo buổi chụp")
    for name, d in out["sessions"].items():
        print(f"  {name}")
        print(f"    ảnh nguồn train/valid/test: {d['n_train']}/{d['n_valid']}/{d['n_test']}")
        for m in PRIMARY:
            v = d["models"][m]
            print(f"    {MODEL_LABELS[m]:<14} R {v['recall']:.3f}  P {v['precision']:.3f}  F1 {v['f1']:.3f}")

if __name__ == "__main__":
    main()
