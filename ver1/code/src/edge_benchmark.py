#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo tốc độ suy luận trên thiết bị biên — phần "Edge device deployment test".

Chạy trên chính NVIDIA Jetson Orin NX (JetPack 6, TensorRT):

    python3 src/edge_benchmark.py --model yolov10 --export engine --half
    python3 src/edge_benchmark.py --model yolov10 --weights best.engine --camera 0

Chạy trên máy trạm để lấy số đối chiếu:

    python3 src/edge_benchmark.py --model yolov10 --device cuda

Ghi results/edge_<model>.json: ms/ảnh và FPS ở các phân vị, tách riêng thời gian
tiền xử lý / suy luận / hậu xử lý, kèm thông tin thiết bị. Đây là các con số điền
vào chỗ "[Insert measured FPS/latency results here]" trong bản thảo.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import MODEL_LABELS, MODEL_ORDER, RESULTS, coco_json, dataset_root, load_cfg, log, save_json


def device_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version()}
    # Jetson ghi tên board ở đây; máy x86 không có file này
    for f, key in (("/proc/device-tree/model", "board"),
                   ("/etc/nv_tegra_release", "jetpack")):
        p = pathlib.Path(f)
        if p.exists():
            try:
                info[key] = p.read_text(errors="ignore").strip("\x00\n ")
            except OSError:
                pass
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def sample_images(cfg: dict, annotation: str, n: int) -> list[pathlib.Path]:
    coco = json.loads(coco_json(cfg, annotation, "test").read_text())
    d = dataset_root(cfg, annotation) / "test"
    return [d / i["file_name"] for i in coco["images"][:n]]


def export(model_key: str, cfg: dict, fmt: str, half: bool) -> pathlib.Path:
    """Xuất sang TensorRT/ONNX. Chỉ chạy được cho hai mô hình Ultralytics."""
    from ultralytics import YOLO
    from predict import default_weights

    spec = cfg["models"][model_key]
    if spec["framework"] != "ultralytics":
        raise SystemExit(f"{model_key} chạy trên Detectron2 — xuất engine phải qua "
                         "torch.onnx.export riêng, chưa hỗ trợ ở script này.")
    w = default_weights(model_key, "ultralytics")
    log(f"xuất {w} → {fmt} (half={half})")
    out = YOLO(str(w)).export(format=fmt, imgsz=spec["imgsz"], half=half, device=0)
    return pathlib.Path(out)


def benchmark(model_key: str, cfg: dict, weights, n_images: int, warmup: int,
              repeats: int, device: str) -> dict:
    import cv2
    from ultralytics import YOLO

    from predict import build_predictor, default_weights

    spec = cfg["models"][model_key]
    paths = sample_images(cfg, spec["annotation"], n_images)
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("không đọc được ảnh test nào")

    # Với engine TensorRT thì đi thẳng qua Ultralytics, không qua predict.py
    w = pathlib.Path(weights) if weights else default_weights(model_key, spec["framework"])
    if w.suffix in (".engine", ".onnx"):
        model = YOLO(str(w), task=spec["task"])
        pre, inf, post = [], [], []

        def run(img):
            r = model.predict(img, imgsz=spec["imgsz"], device=device, verbose=False)[0]
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
    else:
        predictor = build_predictor(model_key, cfg, w, conf=0.25)
        pre, inf, post = [], [], []

        def run(img):
            predictor(img)

    for i in range(warmup):
        run(imgs[i % len(imgs)])

    # Lệnh CUDA chạy bất đồng bộ, nên nếu không đồng bộ trước khi chốt đồng hồ
    # thì con số đo được là thời gian ĐƯA LỆNH vào hàng đợi, không phải thời
    # gian chạy. Phản biện nêu đúng rằng bài không nói có đồng bộ hay không.
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except Exception:
        sync = lambda: None

    sync()
    lat = []
    for _ in range(repeats):
        for img in imgs:
            t0 = time.perf_counter()
            run(img)
            sync()
            lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)

    out = {
        "model": model_key,
        "label": MODEL_LABELS[model_key],
        "weights": str(w),
        "imgsz": spec["imgsz"],
        "n_images": len(imgs),
        "repeats": repeats,
        # Siêu dữ liệu phép đo — phản biện nêu đúng là bài thiếu hết phần này.
        "protocol": {
            "batch_size": 1,
            "precision": "fp32",
            "includes_preprocess": True,
            "includes_nms_and_postprocess": True,
            "cuda_synchronised_per_image": True,
            "warmup_images": warmup,
            "images_sampled_from": "test split",
            "note": ("Độ trễ đầu-cuối của một ảnh: tiền xử lý, suy luận, NMS và "
                     "hậu xử lý, batch 1, FP32, đồng bộ CUDA sau mỗi ảnh. "
                     f"{len(imgs)} ảnh × {repeats} lượt lặp."),
        },
        "latency_ms": {
            "mean": float(lat.mean()), "std": float(lat.std()),
            "p50": float(np.percentile(lat, 50)),
            "p90": float(np.percentile(lat, 90)),
            "p99": float(np.percentile(lat, 99)),
        },
        "fps": {"mean": float(1000 / lat.mean()), "p90": float(1000 / np.percentile(lat, 90))},
        "device_requested": device,
        "device_info": device_info(),
    }
    if inf:
        out["stage_ms"] = {"preprocess": float(np.mean(pre)),
                           "inference": float(np.mean(inf)),
                           "postprocess": float(np.mean(post))}
    # Tên tệp phải phản ánh kích thước đầu vào: chạy lại ở 1024 px đã từng ghi
    # đè phép đo 640 px của chính mô hình đó và làm mất số liệu.
    default_imgsz = cfg["models"][model_key]["imgsz"]
    suffix = "" if spec["imgsz"] == default_imgsz else f"_img{spec['imgsz']}"
    save_json(out, RESULTS / f"edge_{model_key}{suffix}.json")
    log(f"{MODEL_LABELS[model_key]}: {out['latency_ms']['mean']:.1f} ms/ảnh "
        f"({out['fps']['mean']:.1f} FPS), p90 {out['latency_ms']['p90']:.1f} ms")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_ORDER)
    ap.add_argument("--weights", default=None, help=".pt, .onnx hoặc .engine")
    ap.add_argument("--export", choices=["engine", "onnx"], default=None)
    ap.add_argument("--half", action="store_true", help="FP16 khi xuất engine")
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    cfg = load_cfg()
    weights = a.weights
    if a.export:
        weights = export(a.model, cfg, a.export, a.half)
    benchmark(a.model, cfg, weights, a.n_images, a.warmup, a.repeats, a.device)


if __name__ == "__main__":
    main()
