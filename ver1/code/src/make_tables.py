#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh Bảng 1, 2, 3 của bản thảo từ results/*.json.

    python3 src/make_tables.py

Ghi ra tables/table1_training_config.{csv,md}, table2_per_class.{csv,md},
table3_overall.{csv,md} và tables/check_vs_manuscript.md — bảng đối chiếu
số đo chạy lại với số đang in trong bản thảo, để biết chỗ nào phải sửa.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (MODEL_LABELS, MODEL_ORDER, RESULTS, TABLES, clean_fps, load_cfg,
                    log, scope, throughput_requirement)

# Số đang in trong bản thảo (Bảng 2 và Bảng 3), dùng để đối chiếu — KHÔNG dùng
# để sinh bảng. Nguồn: Full_Manuscript_RAE.docx.
MANUSCRIPT = {
    "faster_rcnn": {"sugarcane": (0.959, 0.948, 0.953), "weed": (0.964, 0.972, 0.968),
                    "overall": (0.962, 0.960, 0.960, 0.962)},
    "mask_rcnn":   {"sugarcane": (0.829, 0.928, 0.867), "weed": (0.959, 0.897, 0.927),
                    "overall": (0.894, 0.912, 0.900, 0.908)},
    "retinanet":   {"sugarcane": (0.901, 0.885, 0.893), "weed": (0.928, 0.938, 0.933),
                    "overall": (0.914, 0.911, 0.913, 0.917)},
    "yolov8_seg":  {"sugarcane": (0.950, 0.946, 0.948), "weed": (0.969, 0.967, 0.968),
                    "overall": (0.960, 0.957, 0.958, 0.960)},
    "yolov10":     {"sugarcane": (0.950, 0.946, 0.948), "weed": (0.969, 0.967, 0.968),
                    "overall": (0.960, 0.957, 0.958, 0.980)},
}


def write(name: str, header: list[str], rows: list[list], note: str = "") -> None:
    with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", note]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"{name} → tables/{name}.csv, tables/{name}.md")


def table1(cfg: dict) -> None:
    """Bảng 1 — cấu hình huấn luyện, dựng thẳng từ configs/paper.yaml."""
    tr = cfg["train"]
    keys = MODEL_ORDER
    fw = {"detectron2": f"Detectron2, PyTorch, CUDA 11.8", "ultralytics": "Ultralytics"}
    stats_file = RESULTS / "dataset_stats.json"
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    rows = [
        ["Framework"] + [fw[cfg["models"][k]["framework"]] for k in keys],
        ["Backbone"] + [("ResNet-50 + FPN" if cfg["models"][k]["framework"] == "detectron2"
                         else cfg["models"][k]["config"].replace(".pt", "")) for k in keys],
        ["Image size (px)"] + [f"{cfg['models'][k]['imgsz']} × {cfg['models'][k]['imgsz']}" for k in keys],
        ["Batch size"] + [tr["batch_size"]] * len(keys),
        ["Epochs"] + [tr["epochs"]] * len(keys),
        ["Label format"] + [".json" if cfg["models"][k]["framework"] == "detectron2" else ".txt"
                            for k in keys],
        ["Annotation"] + [cfg["models"][k]["annotation"] for k in keys],
        ["Learning rate"] + [f"{tr['lr']} ({tr['optimizer']}, {tr['lr_schedule']} annealing)"] * len(keys),
    ]
    if stats:
        rows.append(["Training images"] +
                    [stats[cfg["models"][k]["annotation"]]["train"]["images"] for k in keys])
        rows.append(["Test images"] +
                    [stats[cfg["models"][k]["annotation"]]["test"]["images"] for k in keys])
    write("table1_training_config", ["Setting"] + [MODEL_LABELS[k] for k in keys], rows)


def _load(model_key: str):
    p = RESULTS / f"{model_key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def table2(cfg: dict) -> list[str]:
    names = cfg["dataset"]["classes"]
    rows, missing = [], []
    for k in MODEL_ORDER:
        r = _load(k)
        if r is None:
            missing.append(k)
            continue
        pc = r["metrics"]["filtered"]["per_class"]
        for c in names:
            rows.append([MODEL_LABELS[k], c.capitalize(),
                         f"{pc[c]['precision']:.3f}", f"{pc[c]['recall']:.3f}",
                         f"{pc[c]['f1']:.3f}"])
    write("table2_per_class", ["Model", "Class", "Precision", "Recall", "F1-score"], rows,
          note=f"Ngưỡng conf {cfg['eval']['conf_threshold']}, IoU {cfg['eval']['iou_threshold']}; "
               "ma trận nhầm lẫn đã bỏ hàng và cột background.")
    return missing


def _bench_fps(model_key: str) -> str:
    """FPS từ bộ đo lặp, hoặc gạch nếu mô hình đó chưa được đo.

    Không bao giờ lấy trường speed_eval_loop_single_pass trong tệp theo seed:
    đó là số đo một lượt kèm I/O, thấp hơn 2,5-4 lần, và các bảng này nằm trong
    phụ lục công bố nên không được mang hai con số cho cùng một đại lượng.
    """
    try:
        return f"{clean_fps(model_key)[0]:.1f}"
    except SystemExit:
        return "–"


def table3(cfg: dict) -> None:
    rows = []
    for k in MODEL_ORDER:
        r = _load(k)
        if r is None:
            continue
        f = r["metrics"]["filtered"]
        coco = r.get("coco", {})
        rows.append([
            MODEL_LABELS[k],
            f"{f['macro']['precision']:.3f}", f"{f['macro']['recall']:.3f}",
            f"{f['macro']['f1']:.3f}", f"{f['overall_accuracy']:.3f}",
            f"{coco.get('AP50', float('nan')):.1f}" if coco else "–",
            _bench_fps(k),
        ])
    write("table3_overall",
          ["Model", "Precision", "Recall", "F1-score", "Accuracy", "AP50 (COCO)",
           "FPS (RTX 3080 Ti, repeated benchmark)"],
          rows,
          note="Precision/Recall/F1 là trung bình không trọng số của hai lớp; "
               "Accuracy là tỷ lệ đường chéo của ma trận 2×2 (công thức 4). "
               "AP50 tính theo chuẩn COCO trên cùng tập test, không dùng ngưỡng conf 0,30.")


def _ci(boot: dict, model: str, metric: str) -> str:
    m = (boot.get("models", {}) or {}).get(model, {})
    if metric not in m:
        return ""
    lo, hi = m[metric]["ci95"]
    fmt = "{:.1f}" if metric == "ap50" else "{:.3f}"
    return f" [{fmt.format(lo)}, {fmt.format(hi)}]"


def table_detection_quality(cfg: dict) -> None:
    """Bảng chính: chất lượng phát hiện. AP theo chuẩn COCO + recall đầy đủ.

    Đây là bộ chỉ số phạt cả bỏ sót lẫn báo nhầm trên nền, nên nó trả lời được
    câu hỏi "mô hình tìm được bao nhiêu phần cỏ ngoài đồng" — điều mà bộ chỉ số
    ma trận nhầm lẫn (đã bỏ background) không trả lời.
    """
    primary, _ = scope(cfg)
    bpath = RESULTS / "bootstrap.json"
    boot = json.loads(bpath.read_text()) if bpath.exists() else {}
    rows = []
    for k in primary:
        r = _load(k)
        if r is None:
            continue
        c = r.get("coco", {})
        wb = r["metrics"]["with_background"]["per_class"]
        rows.append([
            MODEL_LABELS[k],
            f"{c.get('AP', float('nan')):.1f}",
            f"{c.get('AP50', float('nan')):.1f}" + _ci(boot, k, "ap50"),
            f"{c.get('AP75', float('nan')):.1f}",
            f"{wb['weed']['recall']:.3f}",
            f"{wb['sugarcane']['recall']:.3f}",
        ])
    write("table2_detection_quality",
          ["Model", "AP", "AP50 [95% CI]", "AP75", "Weed recall", "Sugarcane recall"], rows,
          note="AP theo chuẩn COCO trên tập test chung của ba mô hình. Recall tính trên "
               "ma trận nhầm lẫn CÒN hàng background, nên có tính cả đối tượng bỏ sót. "
               "Khoảng tin cậy 95% từ bootstrap lấy mẫu lại ảnh test.")


def table_class_discrimination(cfg: dict) -> None:
    """Bảng phụ: phân biệt lớp trên những đối tượng ĐÃ được định vị."""
    primary, _ = scope(cfg)
    names = cfg["dataset"]["classes"]
    bpath = RESULTS / "bootstrap.json"
    boot = json.loads(bpath.read_text()) if bpath.exists() else {}
    rows = []
    for k in primary:
        r = _load(k)
        if r is None:
            continue
        f = r["metrics"]["filtered"]
        pc = f["per_class"]
        rows.append([MODEL_LABELS[k]]
                    + [f"{pc[c][m]:.3f}" for c in names for m in ("precision", "recall")]
                    + [f"{f['macro']['f1']:.3f}" + _ci(boot, k, "f1"),
                       f"{f['overall_accuracy']:.3f}"])
    write("table3_class_discrimination",
          ["Model", "Precision (sugarcane)", "Recall (sugarcane)",
           "Precision (weed)", "Recall (weed)", "F1 [95% CI]", "Accuracy"], rows,
          note="Tính SAU KHI bỏ hàng và cột background của ma trận nhầm lẫn, nên chỉ đo "
               "việc gán đúng lớp cho những đối tượng đã ghép được với nhãn thật; không "
               "phạt bỏ sót và báo nhầm trên nền. Không dùng để xếp hạng mô hình — xem "
               "Bảng 2.")


def table_throughput(cfg: dict) -> None:
    """Sàng lọc thông lượng: điều kiện CẦN cho triển khai trên thiết bị biên."""
    primary, _ = scope(cfg)
    t = throughput_requirement(cfg)
    cpath = RESULTS / "complexity.json"
    comp = json.loads(cpath.read_text()) if cpath.exists() else {}
    rows = []
    for k in primary:
        r = _load(k)
        if r is None:
            continue
        fps, _src = clean_fps(k)
        c = comp.get(k, {})
        rows.append([
            MODEL_LABELS[k],
            f"{c.get('params', 0) / 1e6:.1f}" if c else "–",
            f"{c.get('gmacs_median', float('nan')):.1f}" if c else "–",
            f"{fps:.1f}",
            "đạt" if fps >= t["fps_required_on_workstation"] else "không đạt",
        ])
    write("table4_throughput",
          ["Model", "Params (M)", "GMACs", "FPS (RTX 3080 Ti)",
           f"Vượt ngưỡng sàng lọc {t['fps_required_on_workstation']:.0f} FPS?"], rows,
          note=f"Ngưỡng suy từ yêu cầu nông học đặt trước: máy phun "
               f"{t['sprayer_speed_kmh']:.0f} km/h, hai khung hình liên tiếp lệch không quá "
               f"{t['max_frame_advance_m']:.2f} m ⇒ cần "
               f"{t['fps_required_on_device']:.1f} FPS TRÊN THIẾT BỊ; nhân hệ số quy đổi "
               f"{t['embedded_slowdown'][1]}× (giả định thiết bị biên chậm hơn GPU máy "
               f"trạm) ⇒ {t['fps_required_on_workstation']:.0f} FPS trên máy trạm. "
               f"GMACs (nhân-cộng, không phải FLOPs=2×MAC) đo ở kích thước đầu vào riêng "
               f"của từng mô hình, không quy về "
               f"cùng một cỡ — xem chú thích trong bài.")


def table_secondary(cfg: dict) -> None:
    """Hai mô hình quy ước đa giác — báo cáo riêng, KHÔNG so với bảng trên."""
    _, secondary = scope(cfg)
    names = cfg["dataset"]["classes"]
    rows = []
    for k in secondary:
        r = _load(k)
        if r is None:
            continue
        f = r["metrics"]["filtered"]
        c = r.get("coco", {})
        wb = r["metrics"]["with_background"]["per_class"]
        rows.append([MODEL_LABELS[k], f"{c.get('AP50', float('nan')):.1f}",
                     f"{wb['weed']['recall']:.3f}", f"{f['macro']['f1']:.3f}",
                     _bench_fps(k)])
    write("table5_polygon_models",
          ["Model", "AP50", "Weed recall", "F1", "FPS (repeated benchmark)"], rows,
          note="Hai mô hình này huấn luyện trên bản xuất nhãn đa giác và đánh giá trên tập "
               "test riêng của bản đó, với độ mịn nhãn khác hẳn (≈29 so với ≈11 đối tượng "
               "mỗi ảnh). Các giá trị ở đây KHÔNG so sánh được với Bảng 2–4.")


def check_vs_manuscript(cfg: dict) -> None:
    names = cfg["dataset"]["classes"]
    lines = ["# Đối chiếu số chạy lại với số in trong bản thảo", "",
             "Cột *bản thảo* lấy từ Bảng 2 và Bảng 3 của Full_Manuscript_RAE.docx.",
             "Chênh lệch > 0,02 được đánh dấu ⚠ và cần xem lại trước khi nộp.", ""]
    header = ["Mô hình", "Lớp", "Chỉ số", "Bản thảo", "Chạy lại", "Chênh"]
    lines += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for k in MODEL_ORDER:
        r = _load(k)
        if r is None:
            lines.append(f"| {MODEL_LABELS[k]} | – | – | – | *chưa chạy* | – |")
            continue
        pc = r["metrics"]["filtered"]["per_class"]
        for c in names:
            for i, metric in enumerate(("precision", "recall", "f1")):
                want = MANUSCRIPT[k][c][i]
                got = pc[c][metric]
                d = got - want
                flag = " ⚠" if abs(d) > 0.02 else ""
                lines.append(f"| {MODEL_LABELS[k]} | {c} | {metric} | {want:.3f} | "
                             f"{got:.3f} | {d:+.3f}{flag} |")
        mac = r["metrics"]["filtered"]["macro"]
        got_all = (mac["precision"], mac["recall"], mac["f1"],
                   r["metrics"]["filtered"]["overall_accuracy"])
        for i, metric in enumerate(("precision", "recall", "f1", "accuracy")):
            want = MANUSCRIPT[k]["overall"][i]
            d = got_all[i] - want
            flag = " ⚠" if abs(d) > 0.02 else ""
            lines.append(f"| {MODEL_LABELS[k]} | *trung bình* | {metric} | {want:.3f} | "
                         f"{got_all[i]:.3f} | {d:+.3f}{flag} |")
    lines += ["", "## Hai điểm phải xử lý trong bản thảo", "",
              "1. Bảng 2 đang cho YOLOv8-seg và YOLOv10 **đúng cùng bộ ba số** "
              "(0,950/0,946/0,948 và 0,969/0,967/0,968). Hai mô hình khác nhau, "
              "chạy trên hai kiểu nhãn khác nhau, gần như không thể trùng khớp đến "
              "chữ số thứ ba — nhiều khả năng một dòng bị chép nhầm.",
              "2. Bảng 3 cho YOLOv10 Accuracy 0,980 trong khi Precision/Recall/F1 "
              "trùng YOLOv8-seg. Với ma trận 2×2 đã bỏ background, Accuracy là hàm "
              "của chính ma trận đó, nên P/R/F1 trùng nhau thì Accuracy bắt buộc "
              "cũng trùng. Con số 0,980 không thể sinh ra từ cùng một ma trận.",
              ]
    (TABLES / "check_vs_manuscript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("đối chiếu → tables/check_vs_manuscript.md")


def main():
    cfg = load_cfg()
    table1(cfg)
    # Bảng theo phạm vi mới: chất lượng phát hiện là CHÍNH, ma trận nhầm lẫn là PHỤ
    table_detection_quality(cfg)
    table_class_discrimination(cfg)
    table_throughput(cfg)
    table_secondary(cfg)
    # giữ hai bảng cũ để đối chiếu với bản thảo gốc
    missing = table2(cfg)
    table3(cfg)
    check_vs_manuscript(cfg)
    if missing:
        log("chưa có kết quả cho: " + ", ".join(missing)
            + " — chạy src/evaluate.py cho các mô hình này")


if __name__ == "__main__":
    main()
