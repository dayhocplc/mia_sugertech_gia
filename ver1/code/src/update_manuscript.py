#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cập nhật bản thảo .docx từ kết quả chạy lại, dưới dạng Track Changes.

    python3 src/update_manuscript.py \
        --docx ~/Downloads/Full_Manuscript_RAE.docx \
        --out  ~/Downloads/Full_Manuscript_RAE_revised.docx

Nguyên tắc: script chỉ sửa những gì nó ĐO ĐƯỢC — các ô số của Bảng 2, Bảng 3,
danh sách độ chính xác trong Abstract và Conclusion, và mấy câu mô tả dữ liệu
kiểm chứng được từ file trên đĩa. Những câu mang lập luận định tính ("Faster
R-CNN phát hiện cỏ chính xác hơn mía"...) thì KHÔNG bị viết lại — vì lập luận
là của nhóm tác giả; thay vào đó script chèn một đoạn ghi chú nêu rõ phát biểu
nào cần đọc lại theo số mới.

Mọi thay đổi đều là Track Changes: mở trong Word, Review → Accept/Reject từng cái.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import MODEL_LABELS, MODEL_ORDER, RESULTS, load_cfg, log
import numpy as np

import discussion as D
from docx_edit import Document

# chỉ số bảng trong document.xml (đếm cả hai bảng bố cục ảnh ở đầu bài)
TBL_CONFIG, TBL_PER_CLASS, TBL_OVERALL = 2, 3, 4

# số cũ trong bản thảo, dùng để dựng lại câu liệt kê trong Abstract/Conclusion


def _review() -> dict:
    """review_analysis.json, cho các mục cần số liệu phân tích bổ sung."""
    return json.loads((RESULTS / "review_analysis.json").read_text())


def load_aux() -> tuple[dict, dict, dict]:
    """seed_variance, bootstrap, complexity — ba file phụ trợ cho phần thống kê."""
    out = []
    for name in ("seed_variance", "bootstrap", "complexity"):
        p = RESULTS / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"Thiếu {p} — chạy src/{name.split('_')[0]}*.py trước.")
        out.append(json.loads(p.read_text()))
    return tuple(out)


def load_results(sub: str = "") -> dict:
    out = {}
    base = RESULTS / sub if sub else RESULTS
    for k in MODEL_ORDER:
        p = base / f"{k}.json"
        if not p.exists():
            hint = "--all --common" if sub else "--all"
            raise SystemExit(f"Thiếu {p} — chạy src/evaluate.py {hint} trước.")
        out[k] = json.loads(p.read_text())
    return out


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def rebuild_tables(d: Document, res: dict, cfg: dict, seedvar: dict,
                   boot: dict, comp: dict) -> None:
    """Dựng lại Bảng 2 và Bảng 3 theo bố cục mới.

    Văn bản phần Kết quả hứa "Table 2 reports average precision and class-wise
    Recall", nhưng bảng trong file Word vẫn là bảng Precision/Recall/F1 cũ —
    `set_cell` chỉ đổi được chữ trong ô, không đổi được số cột. Ở đây dùng
    `replace_table` để thay cả cấu trúc.
    """
    from common import clean_fps, scope, throughput_requirement
    primary, _sec = scope(cfg)
    t = throughput_requirement(cfg)

    def ci(model, metric, fmt):
        m = (boot.get("models", {}) or {}).get(model, {})
        if metric not in m:
            return ""
        lo, hi = m[metric]["ci95"]
        return f" [{fmt.format(lo)}, {fmt.format(hi)}]"

    def seed_mean_sd(model, metric):
        v = [e[metric] for e in seedvar[model]["runs"]
             if e.get("status") == "ok" and metric in e]
        return (float(np.mean(v)), float(np.std(v, ddof=1))) if len(v) > 1 else \
               (float(np.mean(v)) if v else float("nan"), float("nan"))

    # --- Bảng 2: chất lượng phát hiện ---
    rows2 = []
    for k in primary:
        ap50, sd50 = seed_mean_sd(k, "ap50")
        ap, _ = seed_mean_sd(k, "ap")
        rec, rsd = seed_mean_sd(k, "weed_recall_full")
        r = res[k]
        rows2.append([
            MODEL_LABELS[k], f"{ap:.1f}",
            f"{ap50:.2f} ± {sd50:.2f}" + ci(k, "ap50", "{:.1f}"),
            f"{r['coco'].get('AP75', float('nan')):.1f}",
            f"{rec:.3f} ± {rsd:.3f}",
            f"{r['metrics']['with_background']['per_class']['sugarcane']['recall']:.3f}",
        ])
    d.replace_table(TBL_PER_CLASS,
                    ["Model", "AP", "AP50 (mean ± SD) [95% CI]", "AP75",
                     "Weed recall (mean ± SD)", "Sugarcane recall"], rows2)

    # --- Bảng 3: phân biệt lớp, đổi nhãn cho khỏi bị đọc như bảng xếp hạng ---
    names = cfg["dataset"]["classes"]
    rows3 = []
    for k in primary:
        f = res[k]["metrics"]["filtered"]
        pc = f["per_class"]
        rows3.append([MODEL_LABELS[k]]
                     + [f"{pc[c][m]:.3f}" for c in names for m in ("precision", "recall")]
                     + [f"{f['macro']['f1']:.3f}" + ci(k, "f1", "{:.3f}"),
                        f"{f['overall_accuracy']:.3f}",
                        f"{clean_fps(k)[0]:.1f}"])
    d.replace_table(TBL_OVERALL,
                    ["Model", "Precision (sugarcane)", "Recall (sugarcane)",
                     "Precision (weed)", "Recall (weed)", "F1 [95% CI]", "Accuracy",
                     "FPS"], rows3)

    # Đ1: chú thích bảng phải mô tả đúng nội dung MỚI và đúng phạm vi 3 mô hình,
    # nếu không người đọc thấy "the evaluated models" mà bảng chỉ có 3 hàng.
    d.replace_text(
        "Table 2. Detection performance of the evaluated models by class",
        "Table 2. Detection quality of the three bounding-box models, averaged over three "
        "training runs", required=False)
    d.replace_text(
        "Table 3. Overall average performance of the evaluated models",
        "Table 3. Class discrimination among matched detections, and inference throughput, "
        "for the three bounding-box models", required=False)

    # Đ2: bài viện dẫn "Table 4" nhưng bảng đó chưa từng tồn tại — chèn thật.
    t = throughput_requirement(cfg)
    rows4 = []
    for k in primary:
        c = comp.get(k, {})
        f = clean_fps(k)[0]
        rows4.append([
            MODEL_LABELS[k],
            f"{c.get('params', 0) / 1e6:.1f}",
            f"{c.get('gflops_median', float('nan')):.1f}",
            f"{cfg['models'][k]['imgsz']}",
            f"{c.get('peak_vram_mb', float('nan')):.0f}",
            f"{f:.1f}",
            "yes" if f >= t["fps_required_on_workstation"] else "no",
        ])
    d.insert_table_after(
        "Table 3. Class discrimination among matched detections",
        ["Model", "Params (M)", "GFLOPs", "Input size (px)", "Peak VRAM (MB)",
         "FPS (RTX 3080 Ti)", "Clears screening threshold"],
        rows4,
        caption=(f"Table 4. Inference cost and deployment screening. GFLOPs are measured at "
                 f"each model's own input resolution and are therefore not directly "
                 f"comparable between rows. The screening threshold of "
                 f"{t['fps_required_on_workstation']:.0f} frames per second on the "
                 f"workstation follows from the field-speed requirement stated in the "
                 f"Methods and the assumed embedded conversion factor."),
        model_table_index=TBL_PER_CLASS)


def update_tables(d: Document, res: dict, cfg: dict) -> None:
    names = cfg["dataset"]["classes"]

    # Bảng 2 — mỗi mô hình hai hàng (sugarcane, weed), cột 2/3/4 = P/R/F1
    row = 1
    for k in MODEL_ORDER:
        pc = res[k]["metrics"]["filtered"]["per_class"]
        for c in names:
            for col, metric in ((2, "precision"), (3, "recall"), (4, "f1")):
                d.set_cell(TBL_PER_CLASS, row, col, fmt(pc[c][metric]))
            row += 1

    # Bảng 3 — mỗi mô hình một hàng: P, R, F1, Accuracy
    for i, k in enumerate(MODEL_ORDER, start=1):
        m = res[k]["metrics"]["filtered"]
        for col, val in ((1, m["macro"]["precision"]), (2, m["macro"]["recall"]),
                         (3, m["macro"]["f1"]), (4, m["overall_accuracy"])):
            d.set_cell(TBL_OVERALL, i, col, fmt(val))


def update_numbers_in_text(d: Document, res: dict, stats: dict) -> None:
    # Abstract — số ảnh: thay bằng số đếm được từ hai bản xuất Roboflow trên đĩa.
    # (câu tương ứng trong Conclusion được sinh lại cùng cả đoạn, xem rewrite_conclusion)
    n_src = stats["_combined"]["source_images_union"]
    d.replace_text("2,300 field images", f"{n_src:,} annotated field images",
                   count=1, required=False)

    # (số liệu Abstract được thay trọn cặp câu ở rewrite_abstract)

    # Conclusion — hai cụm số độ chính xác
    # (các con số độ chính xác trong Conclusion được thay cùng lúc với cả đoạn)


def add_method_clarifications(d: Document, res: dict, cfg: dict, stats: dict) -> None:
    ev = cfg["eval"]
    r0 = res["faster_rcnn"]

    # 1. Giao thức so khớp — thiếu chỗ này thì Bảng 2/3 không tái lập được
    d.insert_paragraph_after(
        "TN – true negative",
        ["Detections were assigned to ground-truth instances before the confusion "
         f"matrix was formed: predictions with a confidence score of at most "
         f"{ev['conf_threshold']:.2f} were discarded, and each remaining prediction was "
         f"matched to at most one ground-truth instance with an intersection-over-union "
         f"above {ev['iou_threshold']:.2f}, matches being assigned in order of decreasing "
         "IoU. Unmatched ground-truth instances and unmatched predictions were assigned "
         "to a background row and column, which were then excluded before computing "
         "Equations (1)-(4). Precision, Recall and F1-score therefore quantify "
         "sugarcane-versus-weed discrimination among matched instances and do not "
         "penalise missed detections or background false positives; the corresponding "
         "COCO-style average precision values, which do account for both, are reported "
         "alongside the confusion-matrix metrics in Table 3."])

    # 2. Batch hiệu dụng — batch 16 ở 1024 px không vừa 12 GB nếu forward một lần
    d.insert_paragraph_after(
        "Training configurations for the five models are summarised in Table 1",
        ["For the three Detectron2 models, the batch size of 16 reported in Table 1 is "
         "an effective batch size realised through gradient accumulation: two images "
         "were processed per forward pass and gradients were accumulated over eight "
         "passes before each weight update, which is mathematically equivalent to a "
         "single pass over 16 images and keeps peak memory within the 12 GB available "
         "on the RTX 3080 Ti."])

    # 3. Cấu trúc bộ dữ liệu đo được từ chính hai bản xuất Roboflow
    bb, pg = stats["bbox"], stats["polygon"]
    d.insert_paragraph_after(
        "The dataset was split into training, validation, and test sets at a ratio of 80:10:10",
        [f"The 80:10:10 split was applied at the level of source images, before "
         f"augmentation, and augmentation was applied to the training partition only. "
         f"The bounding-box annotation set used for Faster R-CNN, RetinaNet and YOLOv10 "
         f"therefore comprises {bb['train']['images']:,} training, "
         f"{bb['valid']['images']:,} validation and {bb['test']['images']:,} test images "
         f"({bb['_total']['source_images']:,} source images; "
         f"{bb['_total']['instances_total']:,} annotated instances), and the polygon "
         f"annotation set used for Mask R-CNN and YOLOv8-seg comprises "
         f"{pg['train']['images']:,} training, {pg['valid']['images']:,} validation and "
         f"{pg['test']['images']:,} test images ({pg['_total']['source_images']:,} source "
         f"images; {pg['_total']['instances_total']:,} annotated instances)."])

    # (đoạn "Practical implications" được viết lại trọn vẹn ở rewrite_remaining,
    #  nên không chèn thêm gì vào đó nữa)


def rewrite_discussion(d: Document, res: dict, common: dict, stats: dict,
                       n_common: int, cfg: dict) -> None:
    """Viết lại ba đoạn thảo luận và câu chốt Conclusion theo số mới.

    Văn bản do src/discussion.py sinh: mỗi con số đọc thẳng từ results/*.json,
    mỗi nhận định rút ra bằng luật tường minh trong file đó.
    """
    # "Basis of comparison" phải đứng trước mọi con số so sánh, nên chèn vào đầu
    seedvar, boot, comp = load_aux()
    # Phạm vi + chất lượng phát hiện thay cho đoạn mô tả từng mô hình theo F1
    d.replace_paragraph(
        "Detection performance by model",
        D.scope_and_basis(stats, cfg) + D.detection_quality(res, seedvar, cfg))
    d.replace_paragraph("Comparison with weed detection results in sugarcane reported in "
                        "the literature", D.literature_comparison(res))
    # "Detection completeness" đi ngay sau phần so sánh tổng thể
    # Mục Statistics thay cho đoạn so sánh tổng thể cũ
    d.replace_paragraph("Overall model comparison",
                        D.statistics_section(seedvar, boot, cfg)
                        + D.detection_completeness(res))


def rewrite_abstract(d: Document, res: dict, common: dict, stats: dict,
                     n_common: int, cfg: dict) -> None:
    """Abstract cũ xếp hạng cả năm mô hình và nói mô hình phát hiện cỏ 'accurately
    and effectively' — cả hai đều mâu thuẫn với phần Kết quả của chính bài."""
    d.replace_text(
        "The models achieved overall detection accuracies of 96.2, 90.8, 91.7, 96.0, and "
        "98.0%, respectively, with YOLOv10 achieving the highest accuracy. These findings "
        "demonstrate that computer vision and deep learning techniques can accurately and "
        "effectively detect weeds in sugarcane fields under practical field conditions, "
        "providing a viable basis for automated, precision weed management.",
        D.abstract_v2(res, load_aux()[0], stats, cfg))

    # Đ5: câu liệt kê năm mô hình của bản cũ trùng ý với câu mở đầu Abstract mới
    d.replace_text(
        "Five deep learning object detection models — Faster R-CNN, Mask R-CNN, RetinaNet, "
        "YOLOv8-seg, and YOLOv10 — were trained and evaluated on this dataset. ", "",
        required=False)

    # Tiêu đề hứa "comparative evaluation" — thiết kế hai bộ nhãn không đỡ được
    d.replace_text("A comparative evaluation of deep learning object detection models",
                   "performance and detection limits of five deep learning object "
                   "detection models", count=2, required=False)


def fix_botanical_and_captions(d: Document, cfg: dict) -> None:
    """Sửa những chỗ sai về thực vật học và chú thích hình."""
    # (tên khoa học của mía đã được sửa sẵn trong văn bản Introduction mới)

    # Chú thích Hình 3 nói hộp = mía, đa giác = cỏ; thực tế CẢ HAI định dạng đều
    # gán nhãn CẢ HAI lớp — đếm được từ chính file nhãn.
    d.replace_text(
        "Figure 3. Examples of image annotation using bounding boxes (sugarcane, yellow) "
        "and polygons (weeds, purple): (a)–(d) representative annotated samples",
        "Figure 3. Examples of image annotation: (a)-(b) bounding-box format and (c)-(d) "
        "polygon format. Both classes were annotated in both formats; sugarcane is shown "
        "in yellow and weeds in purple",
        required=False)

    # Tên loài cỏ: chỉ đánh dấu, không tự đặt tên khoa học
    d.replace_text(
        "namely purple nutsedge, Bermuda grass, crowfoot grass, and ginger-leaf grass",
        "namely purple nutsedge (Cyperus rotundus L.), Bermuda grass (Cynodon dactylon "
        "(L.) Pers.), crowfoot grass (Dactyloctenium aegyptium (L.) Willd.) and a fourth "
        "species recorded locally as ginger-leaf grass [scientific name to be confirmed by "
        "the authors]",
        required=False)


def rewrite_introduction(d: Document) -> None:
    """Thay 9 đoạn Introduction bằng 6 đoạn ngắn hơn, giữ đủ 16 trích dẫn.

    Thay đoạn đầu bằng toàn bộ văn bản mới, rồi xoá tám đoạn còn lại. Mỗi mốc
    dưới đây đã kiểm tra là chỉ khớp đúng một đoạn.
    """
    d.replace_paragraph("Sugarcane (Saccharum officinarum L.) is a major crop",
                        D.introduction())
    for anchor in (
        "To ensure healthy sugarcane growth, weed management is commonly carried out",
        "Consequently, developing accurate weed-control methods with reduced environmental",
        "Computer vision is a promising approach for this task",
        "Furthermore, variations in field conditions",
        "Beyond classification, recent studies have increasingly focused",
        "In another approach, Moazzam et al. (2021) used a multispectral camera",
        "For sugarcane specifically, weed recognition is particularly challenging",
        "Overall, these studies demonstrate that computer vision and deep learning have",
    ):
        d.replace_paragraph(anchor, [])


def rewrite_conclusion(d: Document, res: dict, stats: dict, cfg: dict) -> None:
    """Đoạn đầu của CONCLUSION gộp cả mô tả dữ liệu lẫn xếp hạng — thay trọn đoạn."""
    d.replace_paragraph(
        "Faster R-CNN, YOLOv8-seg, and YOLOv10 achieved the highest and most balanced "
        "performance", D.conclusion_v2(res, load_aux()[0], stats, cfg))


def training_times(cfg: dict) -> dict:
    """Đọc thời gian huấn luyện thật từ log, thay cho chỗ '[to be confirmed]'."""
    import re
    from common import RAE
    out = {}
    for model, spec in cfg["models"].items():
        if spec["framework"] == "detectron2":
            log_file = RAE / "runs" / model / "log.txt"
            if log_file.exists():
                m = re.findall(r"Total training time: ([0-9:]+)", log_file.read_text())
                if m:
                    h, mi, _s = m[-1].split(":")
                    out[model] = f"{int(h)} h {int(mi)} min"
        else:
            log_file = RAE / "logs" / f"train_{model}.log"
            if log_file.exists():
                m = re.findall(r"epochs completed in ([0-9.]+) hours", log_file.read_text())
                if m:
                    out[model] = f"{float(m[-1]) * 60:.0f} min"
    return out


def rewrite_remaining(d: Document, res: dict, cfg: dict, stats: dict) -> None:
    """Vòng sửa thứ hai: những chỗ còn lại chưa khớp dữ liệu đã chạy."""
    # 1. Số ảnh trong phần Phương pháp (Abstract đã sửa riêng)
    comb = stats["_combined"]
    bb, pg = stats["bbox"], stats["polygon"]
    d.replace_text(
        "A total of 2,300 images were collected and used for model development.",
        f"A total of {comb['source_images_union']:,} images were annotated and used for "
        f"model development: {bb['_total']['source_images']:,} with bounding boxes and "
        f"{pg['_total']['source_images']:,} with polygons, "
        f"{comb['source_images_shared']:,} of them in both formats.")

    # 2. Chú thích Bảng 1 — thời gian huấn luyện thật
    times = training_times(cfg)
    if times:
        d.replace_text("[to be confirmed] – Mask R-CNN training time to be confirmed by "
                       "the authors.", D.training_time_note(times))

    _seedvar, _boot, comp = load_aux()
    # 3–4. Hai đoạn triển khai biên còn nguyên ô giữ chỗ
    d.replace_paragraph("Edge device deployment test. To evaluate the feasibility",
                        D.edge_methods(res))
    d.replace_paragraph("Practical implications and edge deployment test",
                        D.inference_cost(res, cfg, comp))

    # 5. Mô tả lỗi định tính — ma trận cho thấy bản cũ sai trọng tâm
    d.replace_paragraph("Qualitative results. Representative detection results",
                        D.qualitative_results(res, _review()))

    # 6. Đoạn hạn chế — bỏ điều đã làm, thêm điều thật sự còn thiếu
    d.replace_paragraph(
        "providing a viable basis for site-specific weed management",
        D.limitations(res, stats))

    # 7. Bài không có Hình 4; đánh số lại Hình 5 thành Hình 4 cho liền mạch
    d.replace_text("shown in Figure 5", "shown in Figure 4", required=False)
    d.replace_text("Figure 5. Representative detection results",
                   "Figure 4. Representative detection results", required=False)


def add_review_note(d: Document, res: dict) -> None:
    """Một đoạn ghi chú liệt kê phát biểu định tính cần đọc lại theo số mới."""
    acc = {k: res[k]["metrics"]["filtered"]["overall_accuracy"] for k in MODEL_ORDER}
    f1 = {k: res[k]["metrics"]["filtered"]["macro"]["f1"] for k in MODEL_ORDER}
    best_acc = max(acc, key=acc.get)
    best_f1 = max(f1, key=f1.get)
    lines = [
        "[GHI CHÚ CHO NHÓM TÁC GIẢ — xoá trước khi nộp] Các con số trong Bảng 2, Bảng 3, "
        "Abstract và Conclusion đã được thay bằng kết quả chạy lại toàn bộ pipeline "
        "(mã nguồn: rae/, cấu hình: rae/configs/paper.yaml). Bảng đối chiếu từng ô với "
        "số cũ nằm ở rae/tables/check_vs_manuscript.md.",
        f"Theo số mới: Accuracy cao nhất thuộc về {MODEL_LABELS[best_acc]} "
        f"({acc[best_acc]:.3f}), F1 trung bình cao nhất thuộc về {MODEL_LABELS[best_f1]} "
        f"({f1[best_f1]:.3f}). Ba đoạn 'Detection performance by model', 'Comparison with "
        "weed detection results ... in the literature' và 'Overall model comparison' đã "
        "được VIẾT LẠI theo số mới (mã sinh: rae/src/discussion.py) — mời nhóm tác giả "
        "đọc lại phần lập luận và chỉnh giọng văn cho khớp với phần còn lại của bài.",
        "Những chỗ đã sửa ở vòng hai: số ảnh trong phần Phương pháp; chú thích Bảng 1 "
        "(thời gian huấn luyện thật, đọc từ log); hai đoạn về triển khai biên (viết lại "
        "theo số FPS đo được trên GPU máy trạm — số Jetson Orin NX vẫn phải đo trên chính "
        "thiết bị); đoạn Qualitative results (ma trận cho thấy lỗi chính là BỎ SÓT, nhiều "
        "gấp ~50 lần nhầm lớp, chứ không phải nhầm cỏ thành mía như bản cũ viết); đoạn hạn "
        "chế; và đánh số lại Hình 5 thành Hình 4 vì bài không có Hình 4.",
        "Còn một việc cần nhóm tác giả quyết: con số '2.300 ảnh' trong bản gốc không kiểm "
        "chứng được từ dữ liệu trên đĩa (hai bản xuất Roboflow cộng lại có 1.363 ảnh gốc). "
        "Nếu nhóm thực sự chụp 2.300 ảnh rồi chỉ gán nhãn một phần thì nên ghi cả hai con "
        "số. Ngoài ra rae/figures/fig4_training.pdf là đường học của năm mô hình — nếu "
        "muốn đưa vào bài thì nó thành Hình 4 và phần kết quả phát hiện lùi lại thành Hình 5.",
    ]
    # neo vào đoạn hạn chế do chính pipeline sinh ra ở bước trước
    d.insert_paragraph_after(
        "integrating the selected model into a complete precision-spraying prototype",
        lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--author", default="Claude (repro pipeline)")
    ap.add_argument("--merge-runs", default=None,
                    help="đường dẫn merge_runs.py của skill docx (nên có)")
    a = ap.parse_args()

    cfg = load_cfg()
    res = load_results()
    common = load_results("common")
    # Nguồn AP50 duy nhất cho cả bài: trung bình các seed (xem discussion._ap50)
    D.use_seed_variance(load_aux()[0])
    n_common = common["faster_rcnn"]["n_test_images"]
    stats_path = RESULTS / "dataset_stats.json"
    if not stats_path.exists():
        raise SystemExit("Thiếu results/dataset_stats.json — chạy src/data_prep.py --stats")
    stats = json.loads(stats_path.read_text())

    d = Document(a.docx, author=a.author, merge_runs_script=a.merge_runs)
    seedvar, boot, comp = load_aux()
    rebuild_tables(d, res, cfg, seedvar, boot, comp)
    update_numbers_in_text(d, res, stats)
    rewrite_abstract(d, res, common, stats, n_common, cfg)
    fix_botanical_and_captions(d, cfg)
    rewrite_introduction(d)
    rewrite_discussion(d, res, common, stats, n_common, cfg)
    rewrite_conclusion(d, res, stats, cfg)
    rewrite_remaining(d, res, cfg, stats)
    add_method_clarifications(d, res, cfg, stats)
    add_review_note(d, res)
    out = d.save(a.out)
    log(f"{len(d.log)} thay đổi (Track Changes) → {out}")
    for line in d.log:
        print("  ·", line)
    d.cleanup()


if __name__ == "__main__":
    main()
