#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện YOLOv8-seg và YOLOv10 bằng Ultralytics.

    python3 src/train_ultralytics.py --model yolov8_seg
    python3 src/train_ultralytics.py --model yolov10 --smoke

Bám Bảng 1: ảnh 640×640, batch 16, 50 epoch, AdamW, lr0 0,001, cosine annealing.

Ultralytics mặc định bật một loạt augmentation riêng (mosaic, HSV, scale...).
Bản thảo nói augmentation đã làm sẵn ở Roboflow, nên ở đây tắt hết phần trùng
lặp và chỉ giữ lật ngang — để năm mô hình nhìn thấy cùng một bộ dữ liệu.
Muốn dùng augmentation mặc định của Ultralytics thì thêm --ultralytics-aug.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RUNS, YOLO_DATA, load_cfg, log, set_seed

# augmentation của Ultralytics bị tắt để không chồng lên augmentation Roboflow
NO_AUG = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
              scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
              mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["yolov8_seg", "yolov10"])
    ap.add_argument("--batch", type=int, default=None, help="ghi đè batch (mặc định 16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--smoke", action="store_true", help="chạy 2 epoch để kiểm tra")
    ap.add_argument("--ultralytics-aug", action="store_true",
                    help="giữ augmentation mặc định của Ultralytics")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="ghi đè seed; kết quả ra runs/<model>_seed<N>")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="ghi đè kích thước đầu vào; kết quả ra runs/<model>_img<N>[_seed<M>]. "
                         "Dùng để bỏ confound độ phân giải giữa YOLOv10 (640) và hai mô "
                         "hình Detectron2 (1024) mà phản biện nêu.")
    ap.add_argument("--fold", default=None,
                    help="fold chia theo phiên chụp do src/session_split.py --build dựng; "
                         "kết quả ra runs/<model>_fold<tag>")
    ap.add_argument("--nbs", type=int, default=None,
                    help="batch danh nghĩa cho tích lũy gradient (mặc định của Ultralytics là 64)")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    set_seed(cfg["train"]["seed"])
    spec = dict(cfg["models"][args.model])
    if args.imgsz:
        spec["imgsz"] = args.imgsz
    tr = cfg["train"]

    if args.fold:
        data_yaml = YOLO_DATA / "folds" / args.fold / "data.yaml"
        hint = "python3 src/session_split.py --build"
    else:
        data_yaml = YOLO_DATA / spec["annotation"] / "data.yaml"
        hint = "python3 src/data_prep.py --to-yolo"
    if not data_yaml.exists():
        raise SystemExit(f"Chưa có {data_yaml}. Chạy trước:  {hint}")

    from ultralytics import YOLO

    model = YOLO(spec["config"])
    kw = dict(
        data=str(data_yaml),
        epochs=2 if args.smoke else tr["epochs"],
        imgsz=spec["imgsz"],
        batch=args.batch or tr["batch_size"],
        optimizer=tr["optimizer"],
        lr0=tr["lr"],
        lrf=0.01,                      # lr cuối = lr0 × lrf
        cos_lr=tr["lr_schedule"] == "cosine",
        weight_decay=tr["weight_decay"],
        warmup_epochs=3.0,
        seed=tr["seed"],
        deterministic=True,
        workers=args.workers,
        device=args.device,
        project=str(RUNS),
        name=(args.model
              + (f"_fold{args.fold}" if args.fold else "")
              + ("" if args.imgsz is None else f"_img{args.imgsz}")
              + ("" if args.seed is None else f"_seed{args.seed}")),
        exist_ok=True,
        resume=args.resume,
        plots=True,
        val=True,
    )
    if not args.ultralytics_aug:
        kw.update(NO_AUG)
    if args.nbs:
        kw["nbs"] = args.nbs

    log(f"{args.model}: {kw['epochs']} epoch, imgsz {kw['imgsz']}, batch {kw['batch']}, "
        f"{tr['optimizer']} lr0={tr['lr']} cosine={kw['cos_lr']}")
    model.train(**kw)
    log(f"xong → {RUNS / kw['name'] / 'weights/last.pt'}")


if __name__ == "__main__":
    main()
