#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra bộ dữ liệu Tây Ninh và sinh bản YOLO từ COCO.

    python3 src/data_prep.py --stats            # thống kê, ghi results/dataset_stats.json
    python3 src/data_prep.py --to-yolo          # sinh data_yolo/{bbox,polygon}
    python3 src/data_prep.py --stats --to-yolo

Nguồn: hai bản xuất Roboflow của cùng đợt chụp tại nông trại Tân Hưng,
Châu Thành, Tây Ninh (Bảng "Study area and image dataset" của bản thảo).

  Dataset_Bounding_Box — nhãn hộp, dùng cho Faster R-CNN, RetinaNet, YOLOv10
  Dataset_Polygon      — nhãn đa giác, dùng cho Mask R-CNN, YOLOv8-seg

Bản YOLO ghi ra data_yolo/ và không đụng vào dữ liệu gốc.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil

from common import (RESULTS, YOLO_DATA, coco_json, dataset_root, load_cfg, log,
                    remap_categories, save_json)

SPLITS = ("train", "valid", "test")


def _source_image(name: str) -> str:
    """Tên ảnh gốc trước khi Roboflow nhân bản tăng cường.

    Roboflow đặt tên `<goc>_jpg.rf.<hash>.jpg`; ba phiên bản tăng cường của
    cùng một ảnh chỉ khác phần hash. Cắt hash đi để đếm số ảnh gốc.
    """
    return name.split("_jpg.rf.")[0].split(".rf.")[0]


def stats(cfg: dict) -> dict:
    names = cfg["dataset"]["classes"]
    out = {}
    for annotation in ("bbox", "polygon"):
        root = dataset_root(cfg, annotation)
        per_split = {}
        for split in SPLITS:
            coco = remap_categories(cfg, json.loads(coco_json(cfg, annotation, split).read_text()))
            cnt = collections.Counter(a["category_id"] for a in coco["annotations"])
            sizes = {(i["width"], i["height"]) for i in coco["images"]}
            sources = {_source_image(i["file_name"]) for i in coco["images"]}
            wh = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
            weed_area = sorted(a["bbox"][2] * a["bbox"][3]
                               / (wh[a["image_id"]][0] * wh[a["image_id"]][1]) * 100
                               for a in coco["annotations"] if a["category_id"] == 1)
            per_split[split] = {
                "images": len(coco["images"]),
                "instances_per_image": round(len(coco["annotations"]) / max(1, len(coco["images"])), 2),
                "weed_area_pct_median": round(weed_area[len(weed_area) // 2], 3) if weed_area else None,
                "source_images": len(sources),
                "instances": {names[c]: cnt.get(c, 0) for c in range(len(names))},
                "instances_total": sum(cnt.values()),
                "image_sizes": sorted(f"{w}x{h}" for w, h in sizes),
            }
        all_sources = set()
        for split in SPLITS:
            coco = json.loads(coco_json(cfg, annotation, split).read_text())
            all_sources |= {_source_image(i["file_name"]) for i in coco["images"]}
        per_split["_total"] = {
            "images": sum(per_split[s]["images"] for s in SPLITS),
            "source_images": len(all_sources),
            "instances_total": sum(per_split[s]["instances_total"] for s in SPLITS),
            "split_ratio": [round(per_split[s]["images"]
                                  / sum(per_split[x]["images"] for x in SPLITS), 4)
                            for s in SPLITS],
            "root": str(root),
        }
        out[annotation] = per_split

    # Ảnh gốc được gán nhãn ở CẢ hai bản xuất bị đếm hai lần nếu cộng thẳng,
    # nên đếm riêng phần hợp và phần chung.
    src = {}
    for annotation in ("bbox", "polygon"):
        s_ = set()
        for split in SPLITS:
            coco = json.loads(coco_json(cfg, annotation, split).read_text())
            s_ |= {_source_image(i["file_name"]) for i in coco["images"]}
        src[annotation] = s_
    out["_combined"] = {
        "source_images_union": len(src["bbox"] | src["polygon"]),
        "source_images_shared": len(src["bbox"] & src["polygon"]),
        "source_images_bbox_only": len(src["bbox"] - src["polygon"]),
        "source_images_polygon_only": len(src["polygon"] - src["bbox"]),
        "instances_total": sum(out[a]["_total"]["instances_total"] for a in ("bbox", "polygon")),
    }
    save_json(out, RESULTS / "dataset_stats.json")
    log(f"thống kê → {RESULTS / 'dataset_stats.json'}")
    return out


def to_yolo(cfg: dict, annotation: str, link: bool = True) -> pathlib.Path:
    """Chuyển COCO → bố cục YOLO (images/ + labels/ + data.yaml).

    annotation='bbox'    → nhãn `cls cx cy w h`        (YOLOv10)
    annotation='polygon' → nhãn `cls x1 y1 x2 y2 ...`  (YOLOv8-seg)
    Ảnh được symlink chứ không chép, để khỏi nhân đôi vài GB trên đĩa.
    """
    names = cfg["dataset"]["classes"]
    dst = YOLO_DATA / annotation
    for split in SPLITS:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_skipped = 0
    for split in SPLITS:
        src_dir = dataset_root(cfg, annotation) / split
        coco = remap_categories(cfg, json.loads(coco_json(cfg, annotation, split).read_text()))
        by_img = collections.defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)

        for img in coco["images"]:
            w, h = img["width"], img["height"]
            src = src_dir / img["file_name"]
            if not src.exists():
                n_skipped += 1
                continue
            dst_img = dst / "images" / split / img["file_name"]
            if not dst_img.exists():
                if link:
                    dst_img.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst_img)

            lines = []
            for a in by_img.get(img["id"], []):
                cid = a["category_id"]
                if annotation == "polygon" and a.get("segmentation"):
                    poly = a["segmentation"][0]
                    if len(poly) < 6:
                        continue
                    # kẹp về [0, 1]: đa giác Roboflow đôi khi lố ra ngoài mép ảnh
                    xy = [f"{min(max(p / (w if i % 2 == 0 else h), 0.0), 1.0):.6f}"
                          for i, p in enumerate(poly)]
                    lines.append(f"{cid} " + " ".join(xy))
                else:
                    x, y, bw, bh = a["bbox"]
                    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                    nw, nh = min(bw / w, 1.0), min(bh / h, 1.0)
                    if nw <= 0 or nh <= 0:
                        continue
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stem = pathlib.Path(img["file_name"]).stem
            (dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    yaml_txt = (
        f"# sinh bởi rae/src/data_prep.py — bộ {annotation} Tây Ninh\n"
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (dst / "data.yaml").write_text(yaml_txt)
    log(f"YOLO {annotation} → {dst}" + (f" (bỏ qua {n_skipped} ảnh thiếu file)" if n_skipped else ""))
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--to-yolo", action="store_true")
    ap.add_argument("--copy", action="store_true", help="chép ảnh thay vì symlink")
    a = ap.parse_args()
    if not (a.stats or a.to_yolo):
        ap.error("chọn ít nhất --stats hoặc --to-yolo")

    cfg = load_cfg()
    if a.stats:
        s = stats(cfg)
        c = s["_combined"]
        for annotation, d in s.items():
            if annotation.startswith("_"):
                continue
            t = d["_total"]
            print(f"{annotation:8s} {t['images']:5d} ảnh (từ {t['source_images']} ảnh gốc), "
                  f"{t['instances_total']:6d} đối tượng, tỷ lệ chia {t['split_ratio']}")
        print(f"gộp     {c['source_images_union']:5d} ảnh gốc riêng biệt "
              f"({c['source_images_shared']} ảnh có nhãn ở cả hai bản xuất), "
              f"{c['instances_total']} đối tượng")
    if a.to_yolo:
        for annotation in ("bbox", "polygon"):
            to_yolo(cfg, annotation, link=not a.copy)


if __name__ == "__main__":
    main()
