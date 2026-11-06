#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện Faster R-CNN / Mask R-CNN / RetinaNet bằng Detectron2.

    python3 src/train_detectron2.py --model faster_rcnn
    python3 src/train_detectron2.py --model mask_rcnn --micro-batch 2
    python3 src/train_detectron2.py --model retinanet --smoke     # 2 epoch, thử nhanh

Bám đúng Bảng 1 của bản thảo: ảnh 1024×1024, batch 16, 50 epoch, AdamW,
lr 0,001, cosine annealing, backbone ResNet-50 + FPN.

Một điểm phải nói rõ: batch 16 ở 1024×1024 với R50-FPN cần khoảng 40 GB VRAM,
trong khi máy trong bài báo là RTX 3080 Ti 12 GB. Script này giữ nguyên batch
HIỆU DỤNG 16 bằng cách tích luỹ gradient: chạy `--micro-batch k` ảnh mỗi lượt
forward rồi cộng gradient 16/k lượt mới cập nhật trọng số một lần. Về mặt toán
học đây là batch 16; `--micro-batch 16` cho hành vi y hệt một máy đủ VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RAE, RUNS, coco_json, dataset_root, load_cfg, log, remap_categories, set_seed

CACHE = RAE / "data_coco"          # bản COCO đã ánh xạ lại category id


def prepare_coco(cfg: dict, annotation: str) -> dict:
    """Ghi bản COCO đã ánh xạ 0=sugarcane, 1=weed và trả về đường dẫn từng split."""
    out = {}
    for split in ("train", "valid", "test"):
        src = coco_json(cfg, annotation, split)
        dst = CACHE / annotation / f"{split}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            dst.write_text(json.dumps(remap_categories(cfg, json.loads(src.read_text()))))
        out[split] = (dst, dataset_root(cfg, annotation) / split)
    return out


def register(cfg: dict, annotation: str, prefix: str) -> dict:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    names = cfg["dataset"]["classes"]
    paths = prepare_coco(cfg, annotation)
    registered = {}
    for split, (js, imgdir) in paths.items():
        name = f"{prefix}_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(js), str(imgdir))
        MetadataCatalog.get(name).thing_classes = names
        registered[split] = name
    return registered


def register_fold(cfg: dict, prefix: str, tag: str) -> dict:
    """Đăng ký ba split của một fold chia theo phiên chụp (src/session_split.py).

    Fold dùng một bản duy nhất cho mỗi ảnh gốc và một thư mục ảnh phẳng, nên
    không đi qua prepare_coco của bộ dữ liệu chính.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    folds = RAE / "data_coco" / "folds"
    imgdir = folds / "images"
    registered = {}
    for split in ("train", "valid", "test"):
        js = folds / tag / f"{split}.json"
        if not js.exists():
            raise SystemExit(f"Chưa có {js}. Chạy trước:  python3 src/session_split.py --build")
        name = f"{prefix}_fold{tag}_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        register_coco_instances(name, {}, str(js), str(imgdir))
        MetadataCatalog.get(name).thing_classes = cfg["dataset"]["classes"]
        registered[split] = name
    registered["_train_json"] = str(folds / tag / "train.json")
    return registered


def build_cfg(cfg: dict, model_key: str, datasets: dict, args) -> "CfgNode":
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    spec = cfg["models"][model_key]
    tr = cfg["train"]
    n_classes = len(cfg["dataset"]["classes"])

    d = get_cfg()
    d.merge_from_file(model_zoo.get_config_file(spec["config"]))
    d.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(spec["config"])

    d.DATASETS.TRAIN = (datasets["train"],)
    d.DATASETS.TEST = (datasets["valid"],)
    d.DATALOADER.NUM_WORKERS = args.workers

    imgsz = spec["imgsz"]
    d.INPUT.MIN_SIZE_TRAIN = (imgsz,)
    d.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    d.INPUT.MAX_SIZE_TRAIN = imgsz
    d.INPUT.MIN_SIZE_TEST = imgsz
    d.INPUT.MAX_SIZE_TEST = imgsz
    d.INPUT.MASK_FORMAT = "polygon"

    d.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    d.MODEL.RETINANET.NUM_CLASSES = n_classes
    d.MODEL.MASK_ON = spec["task"] == "segment"

    # batch hiệu dụng 16 = micro-batch × số lượt tích luỹ
    eff = tr["batch_size"]
    micro = args.micro_batch or eff
    if eff % micro:
        raise SystemExit(f"--micro-batch {micro} phải là ước của batch {eff}")
    accum = eff // micro
    d.SOLVER.IMS_PER_BATCH = micro

    n_train = len(json.loads(pathlib.Path(datasets["_train_json"]).read_text())["images"])
    iters_per_epoch = max(1, -(-n_train // eff))       # làm tròn lên
    epochs = 2 if args.smoke else tr["epochs"]
    d.SOLVER.MAX_ITER = iters_per_epoch * epochs
    if getattr(args, "max_iter", 0):
        d.SOLVER.MAX_ITER = int(args.max_iter)
    d.SOLVER.BASE_LR = float(getattr(args, "base_lr", 0) or tr["lr"])
    d.SOLVER.WEIGHT_DECAY = tr["weight_decay"]
    d.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    warm = int(getattr(args, "warmup_iters", 0) or tr["warmup_iters"])
    d.SOLVER.WARMUP_ITERS = min(warm, d.SOLVER.MAX_ITER // 10)
    d.SOLVER.CHECKPOINT_PERIOD = max(1, d.SOLVER.MAX_ITER // 5)
    d.SOLVER.AMP.ENABLED = not args.no_amp
    d.TEST.EVAL_PERIOD = max(1, iters_per_epoch * max(1, epochs // 10))
    d.SEED = tr["seed"]
    # RetinaNet với focal loss + AdamW lr=1e-3 phân kỳ ngay khi hết warmup trên
    # các tập huấn luyện nhỏ của thí nghiệm leave-one-session-out. Cắt gradient
    # là biện pháp ổn định số học tiêu chuẩn, không đổi sức chứa hay lịch lr.
    if getattr(args, "clip_grad", 0):
        d.SOLVER.CLIP_GRADIENTS.ENABLED = True
        d.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
        d.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = float(args.clip_grad)
        d.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    suffix = "" if args.seed is None else f"_seed{args.seed}"
    if getattr(args, "fold", None):
        suffix = f"_fold{args.fold}" + suffix
    d.OUTPUT_DIR = str(RUNS / f"{model_key}{suffix}")
    pathlib.Path(d.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # thông tin cho trainer, không thuộc schema Detectron2
    d.set_new_allowed(True)
    d.RAE = type(d)()
    d.RAE.ACCUM_STEPS = accum
    d.RAE.ITERS_PER_EPOCH = iters_per_epoch
    d.RAE.EPOCHS = epochs
    d.RAE.OPTIMIZER = tr["optimizer"]
    return d


def make_trainer(d):
    """DefaultTrainer + AdamW + tích luỹ gradient."""
    import torch
    from detectron2.engine import DefaultTrainer, SimpleTrainer
    from detectron2.evaluation import COCOEvaluator
    from detectron2.solver.build import (get_default_optimizer_params,
                                         maybe_add_gradient_clipping)
    from detectron2.utils.events import get_event_storage

    class AccumTrainer(SimpleTrainer):
        """Như SimpleTrainer nhưng cộng gradient qua nhiều micro-batch."""

        def __init__(self, model, data_loader, optimizer, accum_steps, amp):
            super().__init__(model, data_loader, optimizer)
            self.accum_steps = accum_steps
            self.grad_scaler = torch.cuda.amp.GradScaler() if amp else None

        def run_step(self):
            assert self.model.training, "model phải ở chế độ train"
            import time
            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            last_loss_dict, data_time = {}, 0.0

            for _ in range(self.accum_steps):
                t1 = time.perf_counter()
                data = next(self._data_loader_iter)
                data_time += time.perf_counter() - t1
                if self.grad_scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model(data)
                        losses = sum(loss_dict.values()) / self.accum_steps
                    self.grad_scaler.scale(losses).backward()
                else:
                    loss_dict = self.model(data)
                    losses = sum(loss_dict.values()) / self.accum_steps
                    losses.backward()
                last_loss_dict = {k: v.detach() for k, v in loss_dict.items()}

            self._write_metrics(last_loss_dict, data_time)
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            get_event_storage().put_scalar("accum_step_time", time.perf_counter() - t0)

    class RAETrainer(DefaultTrainer):
        @classmethod
        def build_optimizer(cls, cfg, model):
            params = get_default_optimizer_params(
                model,
                base_lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
                bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
                weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            )
            if cfg.RAE.OPTIMIZER.lower() == "adamw":
                opt = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR,
                                        weight_decay=cfg.SOLVER.WEIGHT_DECAY)
            else:
                opt = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR,
                                      momentum=cfg.SOLVER.MOMENTUM,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
            # hàm này nhận thẳng một optimizer instance và vá lớp của nó
            return maybe_add_gradient_clipping(cfg, opt)

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, output_dir=output_folder or cfg.OUTPUT_DIR)

        def __init__(self, cfg):
            super().__init__(cfg)
            if cfg.RAE.ACCUM_STEPS > 1:
                self._trainer = AccumTrainer(
                    self._trainer.model, self._trainer.data_loader,
                    self._trainer.optimizer, cfg.RAE.ACCUM_STEPS, cfg.SOLVER.AMP.ENABLED)

    return RAETrainer(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["faster_rcnn", "mask_rcnn", "retinanet"])
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="ảnh mỗi lượt forward; batch hiệu dụng vẫn là 16 (mặc định 2)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="chạy 2 epoch để kiểm tra")
    ap.add_argument("--no-amp", action="store_true", help="tắt mixed precision")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="ghi đè seed; kết quả ra runs/<model>_seed<N>")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="cắt chuẩn gradient ở ngưỡng này (0 = tắt)")
    ap.add_argument("--base-lr", type=float, default=0.0,
                    help="ghi đè tốc độ học của config (0 = dùng config). Chỉ dùng "
                         "khi lịch học của bài phân kỳ trên một tập cụ thể, và phải "
                         "áp cùng giá trị cho mọi điều kiện được so sánh.")
    ap.add_argument("--warmup-iters", type=int, default=0,
                    help="ghi đè số iter warmup (0 = dùng config)")
    ap.add_argument("--max-iter", type=int, default=0,
                    help="ghi đè MAX_ITER, chỉ để dò ổn định số học")
    ap.add_argument("--fold", default=None,
                    help="fold chia theo phiên chụp do src/session_split.py --build dựng; "
                         "kết quả ra runs/<model>_fold<tag>")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    set_seed(cfg["train"]["seed"])
    spec = cfg["models"][args.model]

    from detectron2.utils.logger import setup_logger
    run_name = (args.model + (f"_fold{args.fold}" if args.fold else "")
                + ("" if args.seed is None else f"_seed{args.seed}"))
    setup_logger(output=str(RUNS / run_name), name="detectron2")

    if args.fold:
        datasets = register_fold(cfg, args.model, args.fold)
    else:
        datasets = register(cfg, spec["annotation"], args.model)
        datasets["_train_json"] = str(CACHE / spec["annotation"] / "train.json")

    d = build_cfg(cfg, args.model, datasets, args)
    log(f"{args.model}: {d.SOLVER.MAX_ITER} iter = {d.RAE.EPOCHS} epoch × "
        f"{d.RAE.ITERS_PER_EPOCH} iter, micro-batch {d.SOLVER.IMS_PER_BATCH} "
        f"× tích luỹ {d.RAE.ACCUM_STEPS} = batch hiệu dụng {cfg['train']['batch_size']}")
    (pathlib.Path(d.OUTPUT_DIR) / "config.yaml").write_text(d.dump())

    trainer = make_trainer(d)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()
    log(f"xong → {d.OUTPUT_DIR}/model_final.pth")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
