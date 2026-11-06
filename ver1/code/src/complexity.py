#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo chi phí tính toán của năm mô hình — số liệu KHÔNG phụ thuộc phần cứng.

    python3 src/complexity.py

Vì sao cần: bài báo nhắm tới triển khai trên thiết bị biên nhưng nhóm không có
thiết bị đó. Số FPS đo trên RTX 3080 Ti không chuyển sang Jetson Orin NX được.
Số tham số và GFLOPs thì chuyển được — chúng là thuộc tính của kiến trúc, và
người đọc có phần cứng khác có thể tự ước lượng từ đó.

Ghi results/complexity.json: tham số, GFLOPs, VRAM đỉnh khi suy luận, kích thước
trọng số trên đĩa. FLOPs của Detectron2 phụ thuộc dữ liệu (số vùng đề xuất thay
đổi theo ảnh) nên lấy trung vị trên nhiều ảnh và ghi rõ điều đó.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import MODEL_LABELS, MODEL_ORDER, RESULTS, load_cfg, log, save_json

N_IMAGES = 10          # số ảnh lấy mẫu để tính trung vị FLOPs


def detectron2_complexity(model_key: str, cfg: dict) -> dict:
    import torch
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog
    from detectron2.modeling import build_model
    from detectron2.utils.analysis import FlopCountAnalysis

    from common import RUNS
    from evaluate import load_test_set

    d = get_cfg()
    d.set_new_allowed(True)
    d.merge_from_file(str(RUNS / model_key / "config.yaml"))
    d.MODEL.WEIGHTS = str(RUNS / model_key / "model_final.pth")
    model = build_model(d)
    from detectron2.checkpoint import DetectionCheckpointer
    DetectionCheckpointer(model).load(d.MODEL.WEIGHTS)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    import cv2
    items, _ = load_test_set(cfg, cfg["models"][model_key]["annotation"])
    imgsz = cfg["models"][model_key]["imgsz"]
    flops = []
    for _img_meta, path, _b, _c in items[:N_IMAGES]:
        im = cv2.imread(str(path))
        if im is None:
            continue
        im = cv2.resize(im, (imgsz, imgsz))
        t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": t, "height": imgsz, "width": imgsz}]
        try:
            flops.append(FlopCountAnalysis(model, (inputs,)).total() / 1e9)
        except Exception as e:                      # một số op chưa có công thức
            log(f"{model_key}: bỏ qua một ảnh khi đếm FLOPs ({type(e).__name__})")
    # fvcore/Detectron2 đếm MAC (nhân-cộng) và gọi đó là "flops". Ultralytics
    # báo GFLOPs = 2 × MAC. Trộn hai nguồn làm hai mô hình Detectron2 trông nhẹ
    # đi một nửa so với YOLOv10 — phản biện đã chỉ ra. Ở đây quy về MAC cho mọi
    # mô hình và ghi rõ quy ước.
    return {"params": n_params,
            "gmacs_median": float(np.median(flops)) if flops else None,
            "gflops_median": float(np.median(flops)) * 2 if flops else None,
            "flops_tool": "fvcore.FlopCountAnalysis (đếm MAC)",
            "gflops_n_images": len(flops)}


def ultralytics_complexity(model_key: str, cfg: dict) -> dict:
    from ultralytics import YOLO

    from predict import default_weights
    m = YOLO(str(default_weights(model_key, "ultralytics")))
    # Ultralytics chỉ TRẢ VỀ bộ số khi verbose=True; verbose=False trả None.
    # Vì vậy bật verbose rồi nuốt phần in ra màn hình.
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        _layers, params, _grad, gflops = m.info(detailed=False, verbose=True)
    # Ultralytics: GFLOPs = 2 × MAC (thop đếm MAC rồi nhân 2). Chia 2 để cùng
    # quy ước MAC với nhánh Detectron2 ở trên.
    return {"params": int(params),
            "gmacs_median": float(gflops) / 2,
            "gflops_median": float(gflops),
            "flops_tool": "ultralytics.model.info / thop (GFLOPs = 2 × MAC)",
            "gflops_n_images": None}


def peak_vram(model_key: str, cfg: dict) -> float | None:
    """VRAM đỉnh khi chạy suy luận một ảnh."""
    import cv2
    import torch

    from evaluate import load_test_set
    from predict import build_predictor

    if not torch.cuda.is_available():
        return None
    items, _ = load_test_set(cfg, cfg["models"][model_key]["annotation"])
    im = cv2.imread(str(items[0][1]))
    run = build_predictor(model_key, cfg, conf=0.25)
    run(im)                                   # làm nóng, cấp phát bộ đệm
    torch.cuda.reset_peak_memory_stats()
    run(im)
    return torch.cuda.max_memory_allocated() / 1024**2


def weight_size_mb(model_key: str, cfg: dict) -> float | None:
    from predict import default_weights
    p = default_weights(model_key, cfg["models"][model_key]["framework"])
    return p.stat().st_size / 1024**2 if p.exists() else None


def main():
    cfg = load_cfg()
    out = {}
    for k in MODEL_ORDER:
        log(f"đo {k}…")
        try:
            fw = cfg["models"][k]["framework"]
            c = (detectron2_complexity if fw == "detectron2" else ultralytics_complexity)(k, cfg)
            c["label"] = MODEL_LABELS[k]
            c["imgsz"] = cfg["models"][k]["imgsz"]
            c["peak_vram_mb"] = peak_vram(k, cfg)
            c["weights_mb"] = weight_size_mb(k, cfg)
            out[k] = c
        except SystemExit as e:
            log(f"bỏ qua {k}: {e}")
    save_json(out, RESULTS / "complexity.json")

    print(f"\n{'mô hình':<14}{'tham số':>14}{'GFLOPs':>10}{'VRAM đỉnh':>12}{'trọng số':>11}")
    for k, c in out.items():
        g = f"{c['gflops_median']:.1f}" if c["gflops_median"] else "–"
        v = f"{c['peak_vram_mb']:.0f} MB" if c["peak_vram_mb"] else "–"
        w = f"{c['weights_mb']:.0f} MB" if c["weights_mb"] else "–"
        print(f"{c['label']:<14}{c['params']:>14,}{g:>10}{v:>12}{w:>11}")


if __name__ == "__main__":
    main()
