#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đường dẫn, cấu hình và tiện ích dùng chung cho toàn bộ pipeline RAE."""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys

RAE = pathlib.Path(__file__).resolve().parent.parent      # .../mia/rae
ROOT = RAE.parent                                          # .../mia
CFG_FILE = RAE / "configs/paper.yaml"



def _dir(name: str, create: bool = False) -> pathlib.Path:
    """Thư mục dữ liệu ở cạnh mã nguồn, hoặc ở cấp trên nó.

    Trong repo, mọi thứ nằm dưới .../mia/rae/ nên nhánh đầu là đúng. Trong gói
    bổ sung gửi kèm bài báo, mã nằm ở code/ còn results/, logs/, figures/,
    tables/ nằm ở gốc gói — nhánh thứ hai bắt trường hợp đó. Trước đây chỗ này
    gọi mkdir vô điều kiện nên nó tự tạo code/results/ rỗng và mọi kiểm tra
    chạy từ trong gói đều báo thiếu tệp.
    """
    env = os.environ.get("RAE_" + name.upper() + "_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    here, up = RAE / name, RAE.parent / name
    if not here.is_dir() and up.is_dir():
        return up
    if create:
        here.mkdir(parents=True, exist_ok=True)
    return here


RESULTS = _dir("results", create=True)
FIGURES = _dir("figures", create=True)
TABLES = _dir("tables", create=True)
LOGS = _dir("logs", create=True)
RUNS = _dir("runs")            # checkpoint + log huấn luyện
YOLO_DATA = RAE / "data_yolo"  # bộ dữ liệu YOLO sinh ra từ COCO


def ensure_dirs() -> None:
    """Tạo thư mục đầu ra. Chỉ gọi từ các script thực sự ghi tệp."""
    for _d in (RESULTS, FIGURES, TABLES, LOGS, RUNS, YOLO_DATA):
        _d.mkdir(parents=True, exist_ok=True)


def load_cfg(path: pathlib.Path | str = CFG_FILE) -> dict:
    """Đọc configs/paper.yaml. Dùng pyyaml nếu có, không thì parser tối giản."""
    try:
        import yaml
        return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    except ImportError:
        raise SystemExit(
            "Thiếu pyyaml. Chạy:  pip install pyyaml  (hoặc dùng env do setup_env.sh tạo)"
        )


def dataset_root(cfg: dict, annotation: str) -> pathlib.Path:
    """Thư mục bộ dữ liệu COCO tương ứng kiểu nhãn ('bbox' hoặc 'polygon').

    Tìm theo thứ tự: biến môi trường RAE_DATA_ROOT, đường dẫn trong paper.yaml,
    rồi bố cục của gói nộp kèm bài báo (data/annotations/<kiểu>/). Nhờ bước cuối,
    gói supplementary chạy được phần không cần ảnh mà không phải sửa cấu hình.
    """
    key = "bbox_root" if annotation == "bbox" else "polygon_root"
    candidates = []
    env = os.environ.get("RAE_DATA_ROOT")
    if env:
        candidates.append(pathlib.Path(env) / annotation)
    candidates.append(ROOT / cfg["dataset"][key])
    candidates.append(RAE.parent / "data/annotations" / annotation)
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit("Không thấy bộ dữ liệu, đã thử:\n  "
                     + "\n  ".join(str(c) for c in candidates))


def coco_json(cfg: dict, annotation: str, split: str) -> pathlib.Path:
    return dataset_root(cfg, annotation) / split / "_annotations.coco.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def remap_categories(cfg: dict, coco: dict) -> dict:
    """Ánh xạ id Roboflow (1=sugarcane, 2=weed) về 0/1 và bỏ lớp gộp giữ chỗ.

    Trả về dict COCO mới; không sửa dict gốc.
    """
    cmap = {int(k): int(v) for k, v in cfg["dataset"]["roboflow_category_map"].items()}
    names = cfg["dataset"]["classes"]
    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": coco["images"],
        "categories": [{"id": i, "name": n, "supercategory": "none"}
                       for i, n in enumerate(names)],
        "annotations": [],
    }
    for a in coco["annotations"]:
        if a["category_id"] not in cmap:
            continue                      # bỏ lớp gộp giữ chỗ id 0
        b = dict(a)
        b["category_id"] = cmap[a["category_id"]]
        out["annotations"].append(b)
    return out


def save_json(obj, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def log(msg: str) -> None:
    print(f"[rae] {msg}", flush=True, file=sys.stderr)


MODEL_LABELS = {
    "faster_rcnn": "Faster R-CNN",
    "mask_rcnn": "Mask R-CNN",
    "retinanet": "RetinaNet",
    "yolov8_seg": "YOLOv8-seg",
    "yolov10": "YOLOv10",
}
MODEL_ORDER = ["faster_rcnn", "mask_rcnn", "retinanet", "yolov8_seg", "yolov10"]


def scope(cfg: dict) -> tuple[list[str], list[str]]:
    """(mô hình chính, mô hình phụ) theo configs/paper.yaml."""
    sc = cfg.get("scope", {})
    primary = sc.get("primary", MODEL_ORDER)
    secondary = sc.get("secondary", [])
    return list(primary), list(secondary)


def clean_fps(model_key: str) -> tuple[float, str]:
    """FPS đáng tin nhất đang có, kèm nguồn.

    edge_<model>.json là phép đo lặp nhiều lượt lấy trung vị, làm lúc GPU rảnh —
    dùng nó trước. results/<model>.json chỉ đo một lượt trong lúc đánh giá, và
    nếu lúc đó GPU bận thì con số sai lệch nhiều lần.
    """
    e = RESULTS / f"edge_{model_key}.json"
    if e.exists():
        d = json.loads(e.read_text())
        # Phép đo phải đúng kích thước đầu vào của mô hình trong cấu hình, nếu
        # không ta đang gán tốc độ của một biến thể khác cho mô hình này.
        want = load_cfg()["models"].get(model_key, {}).get("imgsz")
        if want and d.get("imgsz") and int(d["imgsz"]) != int(want):
            raise SystemExit(
                f"edge_{model_key}.json đo ở {d['imgsz']} px nhưng cấu hình nói "
                f"{want} px — đo lại:  python3 src/edge_benchmark.py --model {model_key}")
        return float(d["fps"]["mean"]), f"đo lặp {d['repeats']}×{d['n_images']} ảnh"
    # Không rơi về trường speed trong tệp theo seed: đó là số đo một lượt trong
    # vòng lặp đánh giá, kèm I/O, thấp hơn bộ đo lặp 2,5-4 lần. Một con số như
    # thế không được phép vào bài, nên thiếu bộ đo là lỗi chứ không phải để lùi.
    raise SystemExit(
        f"Không có results/edge_{model_key}.json — chạy:  "
        f"python3 src/edge_benchmark.py --model {model_key}\n"
        f"Trường 'speed' trong results/{model_key}.json là số đo một lượt và "
        f"không dùng được cho bài.")


def bench_fps_detail(model_key: str) -> dict:
    """Thông lượng từ bộ đo lặp, cả trung bình và phân vị 90.

    Phép sàng lọc triển khai dùng trung bình, nhưng một hệ thống thời gian thực
    phải đạt hạn chót đều đặn chứ không đạt trung bình, nên p90 cũng phải nêu.
    """
    e = RESULTS / f"edge_{model_key}.json"
    if not e.exists():
        raise SystemExit(f"Không có results/edge_{model_key}.json")
    d = json.loads(e.read_text())
    f = d["fps"]
    return {"mean": float(f["mean"]), "p90": float(f["p90"]),
            "latency_ms_p90": float(d["latency_ms"]["p90"]),
            "latency_ms_p99": float(d["latency_ms"]["p99"]),
            "repeats": d["repeats"], "n_images": d["n_images"]}


def throughput_requirement(cfg: dict) -> dict:
    """Ngưỡng FPS suy từ yêu cầu nông học — đặt trước, không phụ thuộc kết quả."""
    d = cfg.get("deployment", {})
    v = float(d.get("sprayer_speed_kmh", 6.0))
    adv = float(d.get("max_frame_advance_m", 0.30))
    lo, hi = d.get("embedded_slowdown", [5, 10])
    need_device = (v / 3.6) / adv
    return {"sprayer_speed_kmh": v, "max_frame_advance_m": adv,
            "fps_required_on_device": need_device,
            "embedded_slowdown": [lo, hi],
            "fps_required_on_workstation": need_device * float(hi)}
