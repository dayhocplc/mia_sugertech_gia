#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bản thảo cho Sugar Tech (Springer) — cùng số liệu, khác khung lập luận.

    python3 src/build_manuscript_sugartech.py

Vì sao cần một bản riêng chứ không chỉ đổi định dạng. Bản gửi Research in
Agricultural Engineering dẫn bằng một phê phán phương pháp luận: quy ước loại
nền trong ma trận nhầm lẫn làm F1 nhảy từ 0,47 lên 0,98 trên cùng dự đoán. Đó
là lập luận gửi tới cộng đồng thị giác máy nông nghiệp. Độc giả Sugar Tech là
nhà khoa học cây mía, và năm bài phát hiện đối tượng trên mía mà tạp chí đăng
gần đây đều có dạng "mô hình cải tiến cho một nhiệm vụ mía cụ thể".

Nên ở bản này thứ tự đảo lại: dẫn bằng câu hỏi ứng dụng — máy phun theo điểm
cần gì, và mô hình nào đáp ứng được — rồi đặt phê phán quy ước vào mục Thảo
luận như LỜI GIẢI THÍCH cho việc vì sao con số đã công bố đẹp hơn thực địa.
Không con số nào thay đổi; chỉ thứ tự trình bày thay đổi.

Khác biệt hình thức so với bản RAE:
  * bài nguyên tác ≤ 25 trang GIÃN ĐÔI, lề 2,5 cm, kể cả bảng và hình;
  * abstract 150-250 từ; 4-6 từ khoá;
  * Results và Discussion tách riêng (RAE gộp làm một);
  * tham chiếu kiểu Springer, chuyển bằng src/journal_style.py;
  * mục bắt buộc "Statements and Declarations".
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import (AFFIL, AUTHORS, CORRESP, REFERENCES, figure_block,
                              load_review, md_table, methods, to_docx)
from common import (MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope,
                    throughput_requirement)
from journal_style import to_springer

OUT = RAE / "manuscript"

# Phản biện vòng hai nêu đúng: tiêu đề cũ nói "Onboard Throughput" nhưng không
# có phép đo nào trên thiết bị nhúng. Thông lượng ở đây đo trên máy trạm và chỉ
# là một phép sàng, nên tiêu đề phải nói đúng thế.
TITLE = ("Detection Completeness and Throughput Screening of Three Deep Learning "
         "Detectors for Site-Specific Weed Control in Sugarcane")

KEYWORDS = ("Sugarcane; Weed detection; Site-specific herbicide application; "
            "Deep learning; Detection recall")

# Khi đổi đích sang Artificial Intelligence in Agriculture, hai trong ba bài
# Sugar Tech 2026 trước đây không còn đúng hướng: một bài về bệnh lá, một bài về
# hốc mía lưu gốc, không bài nào là phát hiện cỏ. Giữ lại bài phát hiện đối
# tượng trên mía làm bối cảnh cây trồng.
# Giữ ở dạng chuẩn "Tác-giả (năm): …" như REFERENCES, để journal_style.py đổi
# sang kiểu của tạp chí đích; nhập sẵn theo một kiểu thì bộ chuyển bỏ qua mục đó.
EXTRA_REFS = [
    "Li H., Yang J., Song J., Sun M., Wei W., Zhang S., Wu T., Li Y. (2026): Stem base "
    "detection in sugarcane plants using improved YOLOv5m model. Sugar Tech, 28: "
    "467-477. https://doi.org/10.1007/s12355-025-01590-6",
]

DECLARATIONS = [
    "## Statements and Declarations",
    "**Competing Interests.** The authors declare that they have no competing financial "
    "or non-financial interests that are directly or indirectly related to the work "
    "submitted for publication.",
    "**Funding.** This work was supported by the Ministry of Education and Training, "
    "Vietnam, under project code B2025-NLS-02. The funder had no role in the design of "
    "the study, in the collection, analysis or interpretation of the data, or in the "
    "writing of this article.",
    "**Author Contributions.** Nguyen Huynh Truong Gia: Conceptualization, Methodology, "
    "Software, Investigation, Data curation, Formal analysis, Visualization, Writing – "
    "original draft. Nguyen Duc Khuyen: Investigation, Data curation, Writing – review & "
    "editing. Vo Van Qui: Investigation, Data curation, Writing – review & editing. "
    "Tran Quoc Khanh: Investigation, Validation, Writing – review & editing. "
    "Tran Thi Kim Nga: Conceptualization, Methodology, Supervision, Project "
    "administration, Resources, Writing – review & editing. All authors read and "
    "approved the manuscript submitted for publication.",
    "**Data Availability.** The annotation files in COCO format, the per-image results "
    "and the per-seed training logs are provided in Online Resource 2, together with "
    "SHA-256 checksums of every image file, so that the image set — supplied by the "
    "corresponding author on request because of its size — can be verified as the one "
    "used here.",
    "**Code Availability.** The complete training, evaluation and analysis code, "
    "including the scripts that regenerate every table and figure in this article and "
    "in Online Resource 1, is provided in Online Resource 2.",
    "**Supplementary Information.** Online Resource 1 (.docx): supplementary tables and "
    "figure supporting the Methods and Results, comprising the per-seed results, the "
    "threshold and IoU sweeps, the size-stratified recall, the checkpoint-protocol "
    "comparison, the resolution control experiment, the polygon-trained models and the "
    "training curves. Online Resource 2 (.zip): source code, COCO annotations, raw "
    "per-model results, generated tables and figures, training and evaluation logs, and "
    "a SHA-256 manifest of all files including the images.",
    "**Ethics Approval.** This study did not involve human participants or animals.",
]


# ───────────────────────────── phần viết riêng ──────────────────────────────

def abstract(stats: dict, rev: dict, cfg: dict) -> str:
    """Abstract 150-250 từ, dẫn bằng ứng dụng thay vì bằng phương pháp luận.

    Phản biện vòng hai nêu bốn chỗ phải sửa ở đây, và cả bốn đều đúng:
      * "one input resolution" trái với Methods (640 px cho YOLOv10, 1024 px cho
        hai mô hình Detectron2);
      * câu "no pairwise AP50 difference survived ... jointly" không có phân tích
        nào đỡ, vì average precision không nằm trong bootstrap hai tầng;
      * chỉ nêu cách đọc theo instance là bi quan một chiều, phải nêu cả recall
        theo vùng béc;
      * "evidence points to acquisition geometry" là ngoại suy, phải hạ xuống
        mức mà thí nghiệm làm mờ ảnh thật sự chứng minh.
    Thêm một câu về việc bốn loài cỏ đều là một lá mầm và bị gộp một lớp, vì đó
    là giới hạn nông học của mô hình chứ không phải chi tiết kỹ thuật.
    """
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm, ar = rev["full_metrics"], rev["area_recall"]
    ss = rev["size_strata"]["models"]
    cov = json.loads((RESULTS / "coverage_analysis.json").read_text())
    un = cov["union_area_recall"]
    grid = cov["grid_recall"]["0.125"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    best = max(prim, key=lambda k: ap50[k])
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    bb = stats["bbox"]
    lo = lambda d, m: min(d[k]["weed"][m]["mean"] for k in prim)
    hi = lambda d, m: max(d[k]["weed"][m]["mean"] for k in prim)
    sz = {k: cfg["models"][k]["imgsz"] for k in prim}
    return (
        "A site-specific sprayer needs a detector that finds the weed present and runs "
        "fast enough to act. Faster R-CNN, "
        f"RetinaNet and YOLOv10 were trained on {bb['_total']['source_images']:,} "
        "close-range field images from Tay Ninh, Vietnam, and tested on a common "
        f"{bb['test']['images']}-image partition under one protocol — "
        f"each framework at its default input size, {cfg['train']['epochs']} "
        f"epochs, {D.runs_phrase(prim)} per model. Under the "
        "confusion-matrix convention that discards undetected instances, all "
        "three separated the classes reliably (F1-score "
        f"{min(f1s.values()):.2f}-{max(f1s.values()):.2f}). Retaining undetected weed "
        "and spurious detections, the same predictions gave weed-class precision "
        f"{lo(fm, 'precision'):.2f}-{hi(fm, 'precision'):.2f} and recall "
        f"{lo(fm, 'recall'):.2f}-{hi(fm, 'recall'):.2f}; "
        f"{min(un[k]['mean'] for k in prim):.2f}-{max(un[k]['mean'] for k in prim):.2f} "
        "of weed-covered ground would be treated, and "
        f"{min(grid[k]['cell_recall']['mean'] for k in prim):.2f}-"
        f"{max(grid[k]['cell_recall']['mean'] for k in prim):.2f} of nozzle zones one "
        "eighth of the frame wide. Recall rose with weed size, "
        f"{min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} to "
        f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} across area "
        "quartiles. Degrading image detail lowered recall sharply, most steeply for the "
        "smallest weed; tiled inference on the native frames, which raises detail "
        "instead, gained under 0.06 of recall and roughly halved precision, so neither "
        "a coarser nor a finer acquisition improves completeness. Training at the tile "
        "scale, against a control matched on source images and recipe, did not recover "
        "it either: recall fell for all three architectures. "
        + _leakage_clause() +
        f"{MODEL_LABELS[best]} reached the highest AP50 ({ap50[best]:.1f}), a difference "
        "whose interval from resampling test images overlaps that of the next model"
        + (f", and only {MODEL_LABELS[fast[0]]} met the throughput requirement adopted "
           "here, screened on a workstation rather than embedded hardware" if fast else "")
        + ". An excess-green baseline with no learned parameters reached a weed-class "
          "F1-score some twenty-four times lower on the same images, so these figures "
          "measure a hard task rather than weak models"
        + ". Detection completeness, not class discrimination, limits this application "
          "on these data; the weed flora is pooled into one class, "
          "so the models indicate where to spray, not what to spray.")


def _leakage_clause() -> str:
    """Hiệu grouped − control, nếu đã đo được; tóm tắt không được bỏ qua nó."""
    import json as _json
    from common import RESULTS as _R
    f = _R / "session_split.json"
    if not f.exists():
        return ""
    mean = _json.loads(f.read_text(encoding="utf-8")).get("mean_grouped_minus_control", {})
    d = [v["weed_recall"] for v in mean.values() if "weed_recall" in v]
    if not d or not all(x < 0 for x in d):
        return ""
    # trung bình, không phải khoảng: làm tròn hai chữ số khiến đầu khoảng cắt
    # mất giá trị 0,135 của RetinaNet mà phần Thảo luận nêu ở ba chữ số
    return ("Holding out a capture session rather than at random cost a "
            f"further {abs(sum(d) / len(d)):.2f} of recall on average, so these "
            "figures are optimistic. ")


def _exg_baseline_paragraph(fm, order) -> list[str]:
    """Đối chứng chỉ số xanh: cho người đọc biết recall 0,39-0,50 là tốt hay tệ.

    Trả danh sách rỗng nếu chưa chạy src/baseline_exg.py, để bài không nhắc tới
    một con số không có.
    """
    f = RESULTS / "baseline_exg.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    v = d["variants"]
    if "exg_all" not in v:
        return []
    best = max(v.values(), key=lambda x: x["weed_f1"])
    name = next(k for k, x in v.items() if x is best)
    dl_lo = min(fm[k]["weed"]["f1"]["mean"] for k in order)
    dl_hi = max(fm[k]["weed"]["f1"]["mean"] for k in order)
    ratio = dl_lo / max(best["weed_f1"], 1e-9)
    tuned = ("" if name == "exg_all" else
             " The area filter was chosen on the validation partition and applied "
             "unchanged to the test partition, so the baseline is given its best "
             "honest setting rather than one fitted to the test images.")
    return [
        "**What the same images yield without a learned model.** The figures above are "
        "low in absolute terms, and a reader cannot tell from them whether the task is "
        "hard or the detectors are weak. A baseline that uses no learning answers that: "
        "an excess-green index on the same 124 images, thresholded by Otsu's method and "
        "reduced to bounding boxes of its connected components, scored under the same "
        "convention and the same thresholds. It recovers a weed-class F1-score of "
        f"{best['weed_f1']:.3f} "
        f"(recall {best['weed_recall']:.3f}, precision {best['weed_precision']:.3f}, "
        f"{best['detections_per_image']:.1f} boxes per image), against "
        f"{dl_lo:.3f} to {dl_hi:.3f} for the three detectors — about "
        f"{ratio:.0f} times lower (Supplementary Table S18)." + tuned + " The reason is structural rather than a "
        "matter of tuning: sugarcane and its weed flora are both green, so a colour "
        "index separates vegetation from soil and not weed from crop, which is the "
        "green-on-green problem the learned models are there to solve. The shortfall "
        "measured in this article is therefore a property of a hard task on which the "
        "detectors deliver most of what is currently achievable, not evidence that they "
        "are poorly chosen.",
    ]


def introduction(stats: dict, rev: dict, cfg: dict) -> list[str]:
    """Dẫn bằng bài toán của máy phun, không bằng phê phán phương pháp."""
    t = throughput_requirement(cfg)
    return [
        "Weeds are among the principal constraints on sugarcane yield and quality "
        "(Suganthi et al. 2019), competing with the crop for nutrients, light, water and "
        "space during establishment and acting as hosts for pests and diseases. "
        "Breeding has raised the yield potential of the crop substantially (Mirajkar "
        "et al. 2019), but that potential is only realised where competition is "
        "controlled. The "
        "dominant control is herbicide, applied uniformly across the field, and the cost "
        "of that uniformity is both economic and environmental (Donga and Eklo 2018). "
        "Site-specific application — treating only the locations where weeds actually "
        "occur — reduces the quantity applied without reducing efficacy, and it is the "
        "motivation for putting a camera and a detection model on a sprayer.",

        "That shifts the problem upstream, onto the detector, and it imposes two "
        "requirements that a laboratory accuracy figure does not express. The first is "
        "completeness: a weed patch the model does not find is a patch the implement "
        "drives past untreated, so what matters is the proportion of the weed present "
        "that is detected, not the proportion of detections that are correct. The second "
        "is throughput: detections must arrive fast enough that the implement does not "
        "travel past the target before the nozzle can act. At a working speed of "
        f"{t['sprayer_speed_kmh']:.0f} km h⁻¹, with successive frames required to advance "
        f"no more than {t['max_frame_advance_m']:.2f} m, the onboard hardware must supply "
        f"at least {t['fps_required_on_device']:.1f} frames per second.",

        "Deep learning object detectors are the natural candidates, and implements built "
        "on them are already in the field: Azghadi et al. (2025) report a robotic "
        "spot-sprayer that detects weeds in sugarcane with a deep network and treats "
        "them selectively, evaluated against broadcast spraying on commercial farms, "
        "and detectors of the same family have been applied to other tasks in this crop "
        "such as stem base detection for intelligent harvesting (Li et al. 2026). For "
        "weed detection specifically, "
        "Sun et al. (2024) combined multi-scale convolution with a transformer backbone "
        "in sugarcane fields, Singh et al. (2024) performed pixel-level classification of "
        "weed patches from drone imagery, and Mini et al. (2020) applied Mask R-CNN to "
        "aerial sugarcane imagery. Earlier work in the same crop separated sugarcane "
        "from weeds spectrally rather than by shape (de Souza et al. 2020), and the "
        "machine-vision approaches to weed detection across crops are reviewed by "
        "Sohail et al. (2021). Reported accuracies in the wider weed-detection "
        "literature are often high, frequently above 98% (dos Santos Ferreira et al. "
        "2017; Yu et al. 2019; Moazzam et al. 2021; Zhang et al. 2023), which would "
        "suggest the detection half of the problem is close to solved. Those four "
        "figures are not all of one kind: the first three score classification of "
        "whole images or patches, where every item is by construction present and "
        "the only question is its label, while a detector must also find the item. "
        "The comparison drawn in this article is therefore not that such figures are "
        "wrong, but that a number of that magnitude cannot be read as a statement "
        "about field performance unless the convention behind it is given.",

        "Whether that impression survives the two requirements above has not been tested "
        "for sugarcane on close-range imagery of the kind a boom-mounted camera would "
        "produce. The question is practical rather than architectural: not which detector "
        "is most accurate in the abstract, but how much of the weed in a sugarcane field "
        "a current detector actually finds, what limits that figure, and whether any of "
        "the candidates could run at the rate the implement demands.",

        "This study addresses those questions on imagery collected in a standing crop at "
        "a commercial farm in Tay Ninh province, Vietnam, by hand rather than from a "
        "moving implement. It (i) builds and "
        "annotates a close-range dataset of sugarcane and its associated weed "
        "flora; (ii) trains and compares three detectors spanning the two-stage, "
        "single-stage dense and anchor-free real-time families — Faster R-CNN, RetinaNet "
        "and YOLOv10 — under a single protocol, each at its framework's default input "
        "size, with a "
        "single checkpoint rule and repeated training runs; (iii) reports detection "
        "completeness both by instance and by the ground area a sprayer would treat, "
        "together with the size dependence that governs it; and (iv) screens the three "
        "against the throughput the implement requires. Two further models trained on a "
        "polygon annotation of the same images are reported separately and are not ranked "
        "against the three.",
    ]


def _union_recall() -> dict:
    """Recall theo diện tích hợp của hộp cỏ, từ coverage_analysis.json."""
    return json.loads((RESULTS / "coverage_analysis.json").read_text())["union_area_recall"]


def _grid_cell(model: str, frac: str = "0.125") -> float:
    """Recall theo ô lưới của một mô hình, ở một cỡ ô."""
    cov = json.loads((RESULTS / "coverage_analysis.json").read_text())
    return cov["grid_recall"][frac]["models"][model]["cell_recall"]["mean"]


def _cell_range(cov: dict, order: list, frac: str) -> str:
    """Dải recall theo ô lưới trên ba kiến trúc, ở một cỡ ô."""
    v = [cov["grid_recall"][frac]["models"][k]["cell_recall"]["mean"] for k in order]
    return f"{min(v):.3f}-{max(v):.3f}"


def _shared_panel_note() -> str:
    """Nếu hai mô hình rơi vào cùng một ảnh thì chú thích phải nói ra.

    Quy tắc chọn ảnh có thể trả về cùng một ảnh khó cho hai mô hình, và khi đó
    hai ô cạnh nhau trông y hệt. Phản biện đã hỏi đúng chỗ này, nên câu dưới
    đây sinh thẳng từ bản ghi của make_figures chứ không viết tay.
    """
    f = RESULTS / "qualitative_picks.json"
    if not f.exists():
        return ""
    sh = json.loads(f.read_text()).get("shared_images") or {}
    if not sh:
        return ""
    who = sorted({m for v in sh.values() for m in v})
    return (". The selection rule returned the same test image as the failure case for "
            + " and ".join(who)
            + ", so their panels in the third and fourth rows show one image scored by "
              "two models rather than two different images")


def results_blocks(cfg, stats, res, seedvar, boot, comp, rev, lk) -> list[str]:
    """Mục Results: chỉ số liệu, phần diễn giải rộng để sang Discussion."""
    primary, secondary = scope(cfg)
    order = sorted(primary, key=lambda k: -D.seed_mean(k, "ap50"))
    fm = rev["full_metrics"]
    B = []

    # --- Bảng 1: chất lượng phát hiện ---
    rows = []
    for k in order:
        cells = [MODEL_LABELS[k]]
        for metric, dec in (("ap", 1), ("ap50", 2), ("ap75", 1)):
            m_, sd_ = D.seed_mean(k, metric), D.seed_sd(k, metric)
            cells.append(f"{m_:.{dec}f} ± {sd_:.{dec}f}" if sd_ is not None
                         else f"{m_:.{dec}f}")
        lo, hi = boot["models"][k]["ap50"]["ci95"]
        cells.append(f"{lo:.1f}-{hi:.1f}")
        cells.append(str(len([e for e in seedvar[k]["runs"] if e.get("status") == "ok"])))
        rows.append(cells)
    B.append(
        "**Detection quality.** Average precision on the common test partition is given "
        "in Table 1. Ranked by AP50 the order was "
        + D._and_list([f"{MODEL_LABELS[k]} ({D.seed_mean(k, 'ap50'):.2f} ± "
                       f"{D.seed_sd(k, 'ap50'):.2f})" for k in order])
        + ". The stricter criteria do not reproduce that ordering: on AP averaged over "
          "intersection-over-union thresholds from 0.50 to 0.95 the values were "
        + D._and_list([f"{D.seed_mean(k, 'ap'):.1f} for {MODEL_LABELS[k]}" for k in order])
        + ", and on AP75 "
        + D._and_list([f"{D.seed_mean(k, 'ap75'):.1f}" for k in order])
        + ", so the architecture leading on AP50 does not lead once localisation is "
          "required to be more precise.")
    B.append(md_table(
        ["Model", "AP (mean ± SD)", "AP50 (mean ± SD)", "AP75 (mean ± SD)",
         "AP50 95% CI", "Runs"], rows,
        "**Table 1** Average precision of the three bounding-box detectors on the common "
        f"test partition of {stats['bbox']['test']['images']} images under the COCO "
        "protocol. Means and standard deviations are over each model's converged training "
        "runs; the confidence interval is a separate 95% percentile interval from "
        "resampling test images."))

    # --- Bảng 2: hai biến thể ---
    # Khoảng tin cậy theo BURST chụp, không theo ảnh: ảnh trong cùng burst cách
    # nhau vài giây nên không độc lập. Phản biện vòng hai nêu đúng rằng khoảng
    # rộng hơn này mới là khoảng nên đứng trong bảng chính, chứ không chỉ ở phụ lục.
    cb = rev.get("cluster_bootstrap") or {}
    rows = []
    for k in order:
        w, c = fm[k]["weed"], fm[k]["sugarcane"]
        ci = ""
        if cb.get("available"):
            lo, hi = cb["models"][k]["by_cluster"]["recall"]["ci95"]
            ci = f"{lo:.3f}-{hi:.3f}"
        rows.append([
            MODEL_LABELS[k],
            f"{D.seed_mean(k, 'weed_precision'):.3f}",
            f"{D.seed_mean(k, 'weed_recall'):.3f}",
            f"{w['precision']['mean']:.3f} ± {w['precision']['sd']:.3f}",
            f"{w['recall']['mean']:.3f} ± {w['recall']['sd']:.3f}",
            ci,
            f"{w['f1']['mean']:.3f} ± {w['f1']['sd']:.3f}",
            f"{c['recall']['mean']:.3f} ± {c['recall']['sd']:.3f}",
        ])
    f1s = {k: D.seed_mean(k, "f1") for k in primary}
    B.append(
        "**How much weed is found.** The confusion matrix over matched detections can be "
        "scored in two ways, and Table 2 gives both for the same predictions. Deleting "
        "the background row and column before computing the metrics — one of the two "
        "conventions in use — counts only instances given the wrong one of "
        "the two classes, and on that basis all three detectors separated sugarcane from "
        f"weeds reliably, with two-class mean F1-scores from {min(f1s.values()):.3f} to "
        f"{max(f1s.values()):.3f}. Retaining the background row and column, so that an "
        "undetected instance counts as a false negative and a detection with no matching "
        "annotation as a false positive, gives for the weed class Precision of "
        + D._and_list([f"{fm[k]['weed']['precision']['mean']:.3f}" for k in order])
        + ", Recall of "
        + D._and_list([f"{fm[k]['weed']['recall']['mean']:.3f}" for k in order])
        + " and F1-score of "
        + D._and_list([f"{fm[k]['weed']['f1']['mean']:.3f}" for k in order])
        + " for "
        + D._and_list([MODEL_LABELS[k] for k in order])
        + " respectively. Expressed in the terms that matter to a sprayer: counted by "
          "instance, about half the weed annotated in these images was not found, and "
          "about half of what the models reported as weed had no annotated weed under it. "
          "Counted as a nozzle would act on it, over zones one eighth of the frame wide, "
          "the proportion found is "
        + D._and_list([f"{_grid_cell(k):.3f}" for k in order])
        + " instead; the two readings bracket what an implement would do, and both are "
          "developed below. The confidence intervals in Table 2 resample whole capture "
          "bursts rather than single images, because images taken seconds apart are not "
          "independent observations; they are roughly half again as wide as the "
          "image-level intervals.")
    B.append(md_table(
        ["Model", "Precision (weed, bg-excl.)", "Recall (weed, bg-excl.)",
         "Precision (weed, full)", "Recall (weed, full)",
         "Recall 95% CI (burst)", "F1 (weed, full)",
         "Recall (sugarcane, full)"], rows,
        "**Table 2** The same predictions scored under both confusion-matrix conventions, "
        f"at a confidence threshold of {cfg['eval']['conf_threshold']:.2f} and an "
        f"intersection-over-union threshold of {cfg['eval']['iou_threshold']:.2f}. "
        "Background-excluded columns describe class discrimination among localised "
        "objects; full columns retain undetected instances and spurious detections. Values "
        "are means over each model's converged runs."))

    # --- Hình 2: ma trận nhầm lẫn, hình trung tâm của bài ---
    cm = {k: res[k]["confusion_matrix"] for k in order}
    k0 = order[0]
    tp0, fp_c0, fn_c0 = cm[k0][1][1], cm[k0][0][1], cm[k0][1][0]
    fp_b0, fn_b0 = cm[k0][2][1], cm[k0][1][2]
    f1 = lambda fp, fn: 2 * tp0 / (2 * tp0 + fp + fn)
    B.append(
        "**Where the difference comes from.** Fig. 2 shows the confusion matrices "
        "behind both columns of Table 2, with the band that the background-excluded "
        f"convention removes marked. Taking {MODEL_LABELS[k0]} as the worked example: of "
        f"the {tp0 + fn_c0 + fn_b0} annotated weed instances in the test partition, "
        f"{tp0} were detected and classified as weed, {fn_c0} were detected but called "
        f"sugarcane, and {fn_b0} produced no detection at all. In the other direction the "
        f"model placed {fp_b0} weed boxes where no instance was annotated and {fp_c0} on "
        "sugarcane. Scoring with the whole matrix gives a weed-class F1-score of "
        f"{f1(fp_c0 + fp_b0, fn_c0 + fn_b0):.2f}. Deleting the background row and column "
        f"first removes {fn_b0} false negatives and {fp_b0} false positives from the "
        f"arithmetic — the two largest off-diagonal entries — and the same predictions "
        f"then score {f1(fp_c0, fn_c0):.2f}.")
    B.append(
        "The deleted band is not a technicality. Its column holds the weed the implement "
        "would drive past untreated and its row holds the ground it would spray for "
        "nothing, and these are closer proxies for what a sprayer does than anything the "
        "remaining cells measure; no field trial was run here, so they remain proxies. "
        "What survives deletion is whether an object already found is labelled correctly, "
        "which these models answer almost perfectly. "
        "The three matrices also show the behavioural difference reported earlier: the "
        "background row grows from "
        + D._and_list([f"{cm[k][2][0] + cm[k][2][1]} for {MODEL_LABELS[k]}" for k in order])
        + " spurious detections, in the same order as the number of boxes each model "
          "proposes.")
    B.append(figure_block(
        "fig6_confusion",
        "**Fig. 2** Confusion matrices on the test partition for the first training run "
        "(seed 0), rows ground truth and columns prediction. The dashed outline marks the "
        "background row and column, which the background-excluded convention deletes "
        "before computing its metrics; the background column counts annotated instances "
        "that produced no detection and the background row counts detections that match "
        "no annotation. Weed-class F1-scores under both conventions are printed beneath "
        "each matrix. Counts are from a single run, so they differ slightly from the "
        "seed-averaged values in Table 2"))

    # --- mốc không học sâu: cho biết ba mạng mua được bao nhiêu ---
    B.extend(_exg_baseline_paragraph(fm, order))

    # --- diện tích phủ: con số có nghĩa với máy phun ---
    ar = rev["area_recall"]
    cov = json.loads((RESULTS / "coverage_analysis.json").read_text())
    grid = cov["grid_recall"]["0.125"]["models"]
    B.append(
        "**What a sprayer would actually treat.** Counting instances is not the same as "
        "covering ground, and for an implement the second is the operative quantity. "
        "Three measures are reported, because they answer three different questions and "
        "they do not agree. Weighting each annotated weed instance by the area of its "
        "box, and taking the union of those boxes so that overlapping annotations are "
        "not counted twice, gives a recall of "
        + D._and_list([f"{cov['union_area_recall'][k]['mean']:.3f} for {MODEL_LABELS[k]}"
                       for k in order])
        + ", against "
        + D._and_list([f"{ar[k]['instance_recall']['mean']:.3f}" for k in order])
        + " by instance count, because the instances that go undetected are the small "
          "ones."
        + (f" A bounding box is not the plant, however. On the {cov['pixel_coverage']['n_images']} "
           "test images that also carry polygon annotations, the fraction of true weed "
           "pixels falling inside a predicted weed box is "
           + D._and_list([f"{cov['pixel_coverage']['models'][k]['pixel_coverage_recall']['mean']:.3f}"
                          f" for {MODEL_LABELS[k]}" for k in order])
           + ", against "
           + D._and_list([f"{cov['pixel_coverage']['models'][k]['box_area_recall_same_images']['mean']:.3f}"
                          for k in order])
           + " by box area on those same images, so the box-area figure overstates "
             "true cover by roughly ten percentage points. Whether those 26 images "
             "stand for the partition can be checked on the box-area measure, which is "
             "defined on all of them: it reads "
           + D._and_list([f"{cov['box_area_coverage_all_images'][k]['mean']:.3f} for "
                          f"{MODEL_LABELS[k]}" for k in order])
           + " over the full test partition against "
           + D._and_list([f"{cov['pixel_coverage']['models'][k]['box_area_recall_same_images']['mean']:.3f}"
                          for k in order])
           + " on the subset, so the subset is close to representative for the two "
             "Detectron2 models and distinctly easier for YOLOv10. The polygon figures "
             "are therefore a bounded check and not a headline number."
           if cov["pixel_coverage"].get("n_images") else ""))

    B.append(
        "**Nozzle-zone resolution changes the picture.** A boom sprayer does not treat "
        "instances; it opens a nozzle over a zone, and one detection anywhere in that "
        "zone treats all of it. Scoring the same predictions on a square grid, counting "
        "a cell as correctly treated if it contains both an annotated and a predicted "
        "weed, gives a recall of "
        + D._and_list([f"{grid[k]['cell_recall']['mean']:.3f} for {MODEL_LABELS[k]}"
                       for k in order])
        + " at a cell one eighth of the frame width, with cell precision of "
        + D._and_list([f"{grid[k]['cell_precision']['mean']:.3f}" for k in order])
        + ". The figure moves with the cell, so the cell size is reported rather than "
          "assumed: across the three architectures recall is "
        + D._and_list([f"{_cell_range(cov, order, f)} at {name} of the frame width"
                       for f, name in (("0.25", "one quarter"),
                                       ("0.125", "one eighth"),
                                       ("0.0625", "one sixteenth"))])
        + ", and the ordering of the "
          "architectures does not change across the three cell sizes (Supplementary "
          "Table S12, which lists all five readings side by side). The instance figure "
          "is therefore the pessimistic reading of these detectors and the cell figure "
          "the optimistic one; which applies depends on how coarse the implement is. "
          "The cell measure flatters the detectors in a second way: a large annotated "
          "box covers several cells, so a single detection inside it marks all of them "
          "treated. The frame's ground footprint was not measured, "
          "so cell size is expressed as a fraction of the frame rather than in "
          "centimetres, and the correspondence to a real nozzle spacing is left to the "
          "reader's own geometry. What does not change between the two readings is the "
          "comparison this article is about. The background-excluded convention is "
          "silent on all of them, because it removes undetected weed from the "
          "arithmetic before any of these quantities can be formed, and so reports the "
          "same near-perfect score whether the implement is fine or coarse.")

    # --- Hình 2: PR / FROC / ROC ---
    cur = rev.get("curves")
    if cur:
        at30 = {k: min(cur["froc"][k], key=lambda d: abs(d["conf"] - 0.30))
                for k in order}
        mx = {k: max(cur["froc"][k], key=lambda d: d["recall"]) for k in order}
        B.append(
            "**The whole operating curve, not one point.** Every figure above is taken at "
            f"a single confidence threshold of {cfg['eval']['conf_threshold']:.2f}. "
            "Fig. 3 shows the complete trade-off. Fig. 3a gives the precision-recall "
            "curve of the weed class in the full variant: the three architectures trace "
            "closely spaced curves, differing in precision by a median of 0.075 over the "
            "recall range they share, and they differ mainly in where on that common "
            "trade-off each one sits rather than in how favourable the trade-off is. "
            "Fig. 3b restates it in the units a sprayer operator would use, recall "
            "against false weed detections per image. At the working threshold the models "
            "produce "
            + D._and_list([f"{at30[k]['fp_per_image']:.1f} for {MODEL_LABELS[k]}"
                           for k in order])
            + " spurious weed detections per image, and buying further recall is "
              "expensive: "
            + D._and_list([f"{MODEL_LABELS[k]} reaches {mx[k]['recall']:.2f} only at "
                           f"{mx[k]['fp_per_image']:.1f} false detections per image"
                           for k in order])
            + ".")
        B.append(
            "Fig. 3c asks whether the confidence score itself can separate correct weed "
            "detections from spurious ones. This is the one place in a detection problem "
            "where a receiver operating characteristic is well defined, because the "
            "negatives are a finite, identifiable set — the weed detections that match no "
            "annotated weed — rather than the unbounded set of image locations containing "
            "no object. The areas under the curve are "
            + D._and_list([f"{cur['roc'][k]['auc']:.2f} for {MODEL_LABELS[k]}"
                           for k in order])
            + ". Separation is moderate, not strong, and that is the reason the spurious "
              "detections cannot simply be thresholded away: raising the confidence "
              "threshold discards correct detections at almost the same rate as incorrect "
              "ones, which is why the F1-score in Supplementary Table S2 is nearly flat "
              "across the threshold range for two of the three models.")
        B.append(figure_block(
            "fig8_curves",
            "**Fig. 3** Operating characteristics of the three detectors for the weed "
            "class in the full variant, first training run (seed 0). **a** "
            "precision-recall curve, with the filled circle marking the working threshold "
            f"of {cfg['eval']['conf_threshold']:.2f}. **b** free-response operating "
            "characteristic: recall against the mean number of false weed detections per "
            "image, the form of operating characteristic that is defined for detection "
            "tasks. **c** receiver operating characteristic for the confidence score as a "
        "discriminator between weed detections that match an annotated weed and those "
            "that do not, with the area under each curve; the dotted diagonal is chance. "
            "All three curves use the same confidence grid, 0.05 to 0.95 in steps of "
            "0.01, and all detections were retained down to a score of 0.05 at "
            "inference, so the curves end at different points because the models reach "
            "different recall, not because they were sampled differently"))

    # --- cái gì giới hạn: kích thước, định vị, điều kiện ---
    B.extend(D.what_limits_detection(rev, lk, cfg, coverage_elsewhere=True))
    B.extend(D.resolution_control(rev, cfg))
    if (RESULTS / "gsd_analysis.json").exists():
        _g = json.loads((RESULTS / "gsd_analysis.json").read_text())
        _tf = RESULTS / "tiling_analysis.json"
        _t = json.loads(_tf.read_text()) if _tf.exists() else {
            "n_test_images_with_raw": 0, "n_test_images": 0}
        _k0 = next(iter(_g["models"]))
        _lo, _hi = [f"{x:g}" for x in _g["scales"]][0], [f"{x:g}" for x in _g["scales"]][-1]
        B.append(figure_block(
            "fig9_gsd",
            "**Fig. 4** Weed-class recall and precision against the scale at which "
            "objects reach the detector. Two experiments share the abscissa. Solid "
            "lines with filled circles: each test image is downsampled and resampled "
            "back to its original size, so the network input, the annotations and the "
            "detector are unchanged and only the detail carried by each object is lost "
            "— the effect of mounting the camera further from the canopy. Dashed lines "
            "with open triangles: the native frames retained for "
            f"{_t['n_test_images_with_raw']} of the {_t['n_test_images']} test images "
            "are divided into a grid of overlapping tiles, each tile rescaled to the "
            "export size and the predictions merged, which raises the detail per object "
            "instead; this series is the first training run only. The two meet at "
            "1.00×, where both reduce to the pipeline of the main text, and the small "
            "offset there is the disagreement between the two resampling paths. The "
            "lower axis gives the resulting ground sample distance as a multiple of "
            "that of the acquisition and the upper axis the camera height that would "
            "produce it, assuming ground sample distance scales linearly with height "
            f"from the {_g['camera_height_cm']:.0f} cm used here. **a** weed-class "
            "recall and **b** weed-class precision for all three detectors, error bars "
            "one standard deviation over the same converged training runs as the main "
            f"text. **c** recall for {MODEL_LABELS[_k0]} split by quartile of annotated "
            "weed area, Q1 smallest, for the detail-removal experiment only"))

    # --- Bảng 3: chi phí suy luận ---
    t = throughput_requirement(cfg)
    rows = []
    for k in order:
        c = comp[k]
        fps = clean_fps(k)[0]
        det = D.bench_fps_detail(k)
        rows.append([MODEL_LABELS[k], f"{c['params'] / 1e6:.1f}",
                     f"{c['gmacs_median']:.1f}", f"{c['peak_vram_mb']:.0f}",
                     f"{fps:.1f}", f"{det['p90']:.1f}",
                     "yes" if fps >= t["fps_required_on_workstation"] else "no"])
    B.extend(D.inference_cost(res, cfg, comp, rev, cost_table=3))
    B.append(md_table(
        ["Model", "Parameters (M)", "GMACs", "Peak VRAM (MB)", "FPS (mean)",
         "FPS (p90)", "Screen passed"],
        rows,
        "**Table 3** Inference cost on the workstation GPU and the outcome of the "
        f"throughput screen. The threshold is the {t['fps_required_on_workstation']:.0f} "
        "frames per second derived in the Materials and Methods, and the screen is "
        "applied to the mean rate; the 90th-percentile rate over the same repeated "
        "passes is given beside it because a sprayer must meet its deadline on most "
        "frames rather than on average, and the outcome is the same under either. "
        "Compute is given as multiply-accumulate operations at each model's own input "
        "resolution and is therefore not directly comparable between rows."))

    # Bảng độ nhạy ngưỡng thông lượng, trước đây chỉ ở phụ lục S13. Phản biện
    # vòng hai nêu đúng rằng kết luận throughput của bài đứng trên đúng bảng này,
    # nên nó phải ở bài chính chứ không phải ở phụ lục.
    ts = rev.get("throughput_sensitivity")
    if ts:
        speed = t["sprayer_speed_kmh"]
        rows = []
        for g in ts["grid"]:
            if abs(g["speed_kmh"] - speed) > 1e-9 or g["embedded_penalty"] != 0.1:
                continue
            fifth = next(x for x in ts["grid"]
                         if abs(x["speed_kmh"] - speed) < 1e-9
                         and x["advance_m"] == g["advance_m"]
                         and x["embedded_penalty"] == 0.2)
            rows.append([
                f"{g['advance_m']:.2f}",
                f"{g['required_fps_onboard']:.1f}",
                f"{g['required_fps_workstation']:.0f}",
                ", ".join(MODEL_LABELS[m] for m in g["models_meeting"]) or "none",
                f"{fifth['required_fps_workstation']:.0f}",
                ", ".join(MODEL_LABELS[m] for m in fifth["models_meeting"]) or "none",
            ])
        B.append(
            "**How much the throughput conclusion depends on its assumptions.** The "
            "screen rests on two numbers that were assumed rather than measured: how far "
            "the implement may advance between frames, and how much slower the embedded "
            "accelerator is than the workstation. Table 4 varies both at the working "
            f"speed of {speed:.0f} km h⁻¹. The conclusion that only the lightest model "
            "qualifies holds in the stricter part of the range; at a frame advance of "
            "0.50 m with the more favourable hardware ratio all three architectures "
            "qualify. The conclusion is therefore conditional, and the condition is "
            "stated rather than buried: it is the frame advance, not the architecture, "
            "that decides the outcome over most of this range.")
        B.append(md_table(
            ["Frame advance (m)", "Required FPS onboard",
             "Required FPS on workstation at 1/10", "Models meeting it at 1/10",
             "Required FPS on workstation at 1/5", "Models meeting it at 1/5"],
            rows,
            "**Table 4** Sensitivity of the throughput screen to its two assumptions, at "
            f"a working speed of {speed:.0f} km h⁻¹. The required onboard rate is the "
            "speed divided by the permitted frame advance; the workstation rate is that "
            "divided by the assumed embedded penalty, taken from vendor specifications "
            "and not measured here. Measured workstation rates are in Table 3. The full "
            "grid over three working speeds is in Supplementary Table S13."))

    # --- Bảng 4: mọi thang đo cạnh nhau ---
    mr = rev.get("metric_ranking")
    if mr:
        rows, prev = [], None
        for m in mr["metrics"]:
            grp = m["group"] if m["group"] != prev else ""
            prev = m["group"]
            cells = [grp, m["label"] + ("" if m["better"] == "hi" else " \u2193")]
            for k in order:
                v = m["values"][k]
                txt = f"{v:,.0f}" if abs(v) >= 100 else (
                    f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")
                cells.append(f"**{txt}**" if m["rank"][k] == 1 else txt)
            rows.append(cells)
        B.append(
            "**Which model is best depends on the metric.** Table 5 places every measure "
            "used in this study side by side. "
            + (f"All {mr['n_models_winning']} architectures rank first on at least one of "
               f"the {mr['n_metrics']} measures" if mr["n_models_winning"] == len(order)
               else f"{mr['n_models_winning']} of the {len(order)} architectures rank "
                    f"first on at least one of the {mr['n_metrics']} measures")
            + ", and the two that lead most often do so on "
            + D._and_list([f"{mr['wins'][k]} measure" + ("s" if mr["wins"][k] != 1 else "")
                           + f" for {MODEL_LABELS[k]}"
                           for k in sorted(order, key=lambda x: -mr['wins'][x])])
            + ". Counting wins across measures would overstate how independent they "
              "are — three of them are forms of COCO average precision and two are "
              "forms of computational cost — so the same table is also read by family: "
            + D._and_list([f"{mr['family_wins'][k]} of the {mr['n_families']} metric "
                           f"families for {MODEL_LABELS[k]}" for k in
                           sorted(order, key=lambda x: -mr['family_wins'][x])])
            + (f", with {mr['n_families_split']} family split between architectures"
               if mr["n_families_split"] else "")
            + ". Neither count should be read as a score. The measure-by-measure count "
              "depends on how many measures of each kind happen to be reported, and the "
              "family count depends on how the families are drawn and on how a split "
              "family is handled; both are given here to show that the ranking moves "
              "with the yardstick, not to declare a winner. The pattern behind the "
              "movement is what matters. "
            + f"{MODEL_LABELS[max(order, key=lambda k: mr['wins'][k])]} leads wherever "
              "finding objects is what is rewarded, and the lighter architecture leads "
              "wherever precise placement, restraint in reporting, or computational cost "
              "is what is rewarded. This follows directly from how many detections each "
              "returns: "
            + D.det_count_phrase(rev, cfg, order)
            + ". A detector that proposes more finds more and "
              "is wrong more often; one that proposes less is right more often and misses "
              "more. The gap is far the wider one at the retention floor, so it bears "
              "most on the average-precision measures and least on the figures quoted at "
              "the working threshold.")
        B.append(md_table(
            ["Metric family", "Measure"] + [MODEL_LABELS[k] for k in order], rows,
            "**Table 5** Every measure reported in this study, for the three "
            "bounding-box models. Bold marks the best value in each row; \u2193 marks a "
            "measure for which lower is better. Values are means over each model's "
            "converged training runs, except throughput and compute, which are "
            "architectural properties. The table is given because no single row of it "
            "determines the ranking of the others."))

    # --- thống kê ---
    B.extend(D.statistics_section(seedvar, boot, cfg, rev))

    # --- hai mô hình đa giác ---
    B.append(
        "**Models trained on the polygon convention.** For completeness, "
        + " and ".join(MODEL_LABELS[k] for k in secondary)
        + " were evaluated on the polygon test partition of "
        + f"{stats['polygon']['test']['images']} images under the same protocol, reaching "
          "AP50 of "
        + D._and_list([f"{res[k]['coco']['AP50']:.1f}" for k in secondary])
        + " respectively (Supplementary Table S10). Those values are computed against "
          "ground truth that resolves weed patches into constituent parts and describe a "
          "different task; each model was trained once, so no training variability is "
          "available for them.")
    return B


def discussion(cfg, stats, res, rev, lk) -> list[str]:
    """Thảo luận. Phê phán quy ước đặt Ở ĐÂY, làm lời giải thích."""
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm = rev["full_metrics"]
    ar = rev["area_recall"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap50[k] or 0))
    cm = {k: res[k]["confusion_matrix"] for k in prim}
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    out = []

    out.append(
        "**Why published figures look better than field performance.** The two sets of "
        "columns in Table 2 come from one set of predictions on one set of images and "
        "differ by roughly a factor of two. The difference is a reporting convention. "
        "When the background row and column of the confusion matrix are deleted, an "
        "annotated weed that produced no detection leaves the denominator with them, "
        "and so does a detection that matched no annotation. What remains measures one "
        "thing: given that an object has been localised, is it labelled correctly? "
        "That is a reasonable question and the answer here is encouraging, a two-class "
        f"mean F1-score between {min(f1s.values()):.2f} and {max(f1s.values()):.2f}. "
        "It is not the question a sprayer asks, and a figure of that magnitude quoted "
        "without stating which convention produced it will be read as though it were. "
        "On the same predictions the full variant gives weed-class F1-scores between "
        f"{min(fm[k]['weed']['f1']['mean'] for k in prim):.2f} and "
        f"{max(fm[k]['weed']['f1']['mean'] for k in prim):.2f}. The "
        "effect of the choice is large enough to change what a reader concludes about "
        "whether the problem is solved. This study did not survey how often each "
        "convention is used and makes no claim about which is the more common. The "
        "point rests on the arithmetic above: a published figure that does not say "
        "which convention produced it cannot be interpreted either way.")

    out.append(
        "**Comparison with detection results reported in sugarcane.** Published "
        "figures for this crop are mostly not commensurable with the present ones. "
        "Singh et al. (2024) report 90.5% accuracy for pixel-level classification of "
        "weed patches from drone imagery, a per-pixel figure for a segmentation task "
        "at a different acquisition scale; Mini et al. (2020) report an AP50 of 65.5% "
        "with Mask R-CNN against mask annotations on aerial imagery. By the same rule "
        "this study applies to its own two annotation conventions, neither can be "
        "placed on one scale with the AP50 of "
        + D._and_list([f"{ap50[k]:.1f}% for {MODEL_LABELS[k]}" for k in order])
        + " obtained here. No direction can be read from the difference either, "
          "because the annotation convention, the object scale and the task all "
          "differ at once; the two scale experiments reported in the Results are "
          "internal to this study and are the only evidence here that bears on "
          "acquisition geometry.")

    out.append(
        "One published result is close enough in kind to be informative. Partel et al. "
        "(2019) built and field-tested a camera-guided selective sprayer and reported, "
        "for the trial on real plants rather than artificial ones and on the embedded "
        "accelerator rather than the larger GPU, a weed-detection precision of 0.59 and "
        "a recall of 0.44. Those figures retain undetected weed "
        "in the arithmetic, as the full variant does here, and they fall inside the "
        "range obtained in this study ("
        + D._and_list([f"{fm[k]['weed']['precision']['mean']:.2f}" for k in order])
        + " for precision and "
        + D._and_list([f"{fm[k]['weed']['recall']['mean']:.2f}" for k in order])
        + " for recall). A working implement reporting much the same completeness "
          "supports reading the shortfall measured here as a property of the task at "
          "this scale rather than of these three models or this dataset. Azghadi et al. "
          "(2025) go further and report a deep-learning spot-sprayer operating in "
          "sugarcane itself, evaluated against broadcast spraying on commercial farms; "
          "that an implement reaches the field on detection of this completeness is the "
          "practical counterpart of the readings reported above, and it is the "
          "treated-area and nozzle-zone figures rather than the instance figure that "
          "such a system acts on.")

    out.append(
        "Two families of detection metric that would sharpen the diagnosis were not "
        "computed here and are named for the work that follows. Localization recall "
        "precision (Oksuz et al. 2018) folds localisation error together with the "
        "false negative and false positive rates and selects an operating threshold "
        "per class, which is closer to what an implement needs than average precision "
        "is. The TIDE decomposition (Bolya et al. 2020) attributes a loss of average "
        "precision to its causes — classification, localisation, duplicate detections, "
        "missed objects — and would separate the imprecise-localisation and the "
        "never-detected halves of the shortfall more cleanly than the "
        "intersection-over-union sweep used here.")

    out.append(
        "**Implications for sprayer design.** Three consequences follow for anyone "
        "building the implement. First, the binding constraint is detection, not "
        "classification: once a weed patch is localised its class is almost always "
        "assigned correctly, and the counts of sugarcane reported as weed ("
        + D._and_list([str(cm[k][0][1]) for k in order])
        + ") and weed reported as sugarcane ("
        + D._and_list([str(cm[k][1][0]) for k in order])
        + ") are negligible beside the numbers undetected; of those two, sugarcane "
          "reported as weed is the more harmful, because that plant is sprayed. "
          "Second, the shortfall has two halves that cost differently — weed left "
          "untreated, and herbicide applied where no weed was annotated — and a single "
          "accuracy figure hides the trade-off between them. Third, the evidence "
          "points at acquisition geometry rather than architecture: recall falls "
          "monotonically with object size, retraining the lightest model at the input "
          "resolution of the other two did not change its detection quality, and "
          "removing detail from the imagery lowers recall sharply and size-selectively. "
          "Lowering the camera and tiling the frame were therefore the indicated "
          "routes, and both were tested: neither relieved the shortfall, and training "
          "at the tile scale against a matched control did not either. The evidence "
          "points at the size at which weeds appear in this imagery rather than at a "
          "geometry a designer could correct by moving the camera.")

    mr = rev.get("metric_ranking")
    if mr:
        wins = mr["wins"]
        top = sorted(prim, key=lambda k: -wins[k])
        out.append(
            "**The ranking is a property of the yardstick, not only of the model.** "
            f"The {mr['n_metrics']} measures of Table 5, grouped into "
            f"{mr['n_families']} families in the Results, do not agree on an ordering. "
            "A reader who takes AP50 as the criterion concludes that "
            f"{MODEL_LABELS[top[0]]} is the better detector; one who takes AP75 or "
            "weed-class precision concludes the opposite; one who takes weed-class "
            "F1-score in the full variant concludes that neither leads. All three "
            "readings are correct about their own measure, and none of them is correct "
            "about the field.")
        out.append(
            "The pattern behind the split is mechanical rather than mysterious, and it "
            "is visible in how many detections each model returns: "
            + D.det_count_phrase(rev, cfg, order)
            + ". A detector that "
            "proposes freely finds more of what is present and is wrong more often, so "
            "it leads on recall, on average recall and on the area-weighted measure, and "
            "trails on precision. A detector that proposes sparingly is right more often "
            "and misses more, so it leads on precision, on the background-excluded "
            "F1-score and on the stricter overlap threshold. Neither behaviour is better "
            "in the abstract; which is better depends on whether a weed left untreated "
            "or herbicide applied to bare ground is the more expensive error on a given "
            "farm. That is an agronomic and economic question, not a question about "
            "architectures, and it cannot be settled by reporting one number. The "
            "practical recommendation that follows is to state the operating point and "
            "the convention alongside any figure quoted for this task, and to choose the "
            "model against the cost structure of the intended operation rather than "
            "against a leaderboard.")

    out.append(
        "**Choosing between the three.** No architecture was best on every criterion, "
        "and the outcome is a stated trade-off. "
        + (f"{MODEL_LABELS[fast[0]]} was the only model to clear the throughput "
           "screen, and it also recovered the fewest weed instances and the smallest "
           "treated area; the two architectures that found more are, on this hardware, "
           "too slow at the assumed working speed. " if fast else
           "No model cleared the throughput screen. ")
        + "Where completeness governs, the heavier architectures are indicated and the "
          "throughput problem must be solved another way — a slower working speed, a "
          "wider field of view, or a dedicated accelerator. Where throughput "
          "binds, the lighter architecture is the only candidate and the weed it "
          "misses must be accepted or handled by a second pass. The treated-area "
          "figures are the ones to use in that calculation: "
        + D._and_list([f"{_union_recall()[k]['mean']:.2f} for {MODEL_LABELS[k]}"
                       for k in order])
        + " of the weed-covered ground would be treated, or, over nozzle zones one "
          "eighth of the frame wide, "
        + D._and_list([f"{_grid_cell(k):.2f}" for k in order])
        + " of the zones holding weed.")

    # Đoạn "shortfall ... two halves" của limitations_v2 nay trùng với mục
    # Implications viết riêng ở trên, nên bỏ khỏi bản này. Bản RAE không có mục
    # Discussion riêng nên vẫn cần nó.
    lim = D.limitations_v2(res, json.loads((RESULTS / "seed_variance.json").read_text()),
                           stats, cfg, rev, lk)
    out.extend(x for x in lim if "two halves that should not be conflated" not in x)
    return out


def conclusion(cfg, stats, res, rev) -> list[str]:
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm = rev["full_metrics"]
    ar = rev["area_recall"]
    ss = rev["size_strata"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    ap50 = {k: D.seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap50[k] or 0))
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    return [
        "Three deep learning detectors were trained on close-range field imagery of "
        "sugarcane and weeds from Tay Ninh province, Vietnam, and evaluated against the "
        "two requirements a site-specific sprayer imposes. All three assigned the correct "
        "class to almost every object they localised, with two-class mean F1-scores from "
        f"{min(f1s.values()):.2f} to {max(f1s.values()):.2f} under the convention commonly "
        "reported in this literature. Scored so that undetected weed and spurious "
        "detections are retained, the same predictions gave weed-class Precision of "
        f"{min(fm[k]['weed']['precision']['mean'] for k in prim):.2f} to "
        f"{max(fm[k]['weed']['precision']['mean'] for k in prim):.2f} and Recall of "
        f"{min(fm[k]['weed']['recall']['mean'] for k in prim):.2f} to "
        f"{max(fm[k]['weed']['recall']['mean'] for k in prim):.2f}; weighted by the ground "
        "area a sprayer would treat, recall was "
        f"{min(_union_recall()[k]['mean'] for k in prim):.2f} to "
        f"{max(_union_recall()[k]['mean'] for k in prim):.2f}, and counted over nozzle "
        "zones one eighth of the frame wide, "
        f"{min(_grid_cell(k) for k in prim):.2f} to {max(_grid_cell(k) for k in prim):.2f}. "
        "Which of those three readings applies depends on how coarse the implement is.",

        "Detection completeness, not class discrimination, is therefore what limits this "
        "application, and it is governed by object size: recall rose from "
        f"{min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f}-"
        f"{max(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the smallest "
        "quartile of annotated weed instances to "
        f"{min(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f}-"
        f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the largest. "
        "Three experiments bear on why. Retraining the lightest detector at 2.56 times "
        "the pixel count left its detection quality unchanged. Degrading the imagery to "
        "simulate a higher camera lowered recall monotonically and most steeply for the "
        "smallest weed. Dividing the retained native frames into tiles, which raises "
        "the detail per object instead of lowering it, bought at most 0.05 of recall at "
        "a two-by-two grid and cost more than half of the precision, and at a "
        "three-by-three grid lowered recall below the untiled control for all three "
        "architectures. Neither a coarser nor a finer acquisition therefore improves "
        "completeness on these data. Because both experiments apply a detector trained "
        "at one object scale to imagery at another, a fourth experiment trained the "
        "three architectures at the tile scale against a control matched on source "
        "images, recipe and augmentation: "
        "scored on the same tiled imagery, the tile-trained models recovered less weed "
        "than the frame-trained ones, for all three architectures. Removing the "
        "augmentation puts that matrix well below the main results, so it bounds what "
        "retraining buys at this budget rather than at the full one; training across "
        "ground sample distances at the full budget is the experiment this work most "
        "needs next and was not done here. What the four establish together is useful "
        "to a designer and less convenient than a reading in terms of geometry: moving "
        "the camera is not a lever on completeness, and neither is retraining at the "
        "moved geometry. "
        + (f"Of the three, only {MODEL_LABELS[fast[0]]} met the throughput requirement "
           f"adopted here for {t['sprayer_speed_kmh']:.0f} km h⁻¹; that requirement "
           "rests on an assumed frame advance and an assumed embedded-hardware penalty, "
           "and relaxing either admits the other two. It is also the model that treats "
           "the least ground, so the choice is a trade-off rather than a ranking. "
           if fast else "")
        + "Confirming feasibility on a working implement requires measurement on the "
          "target hardware, which this study did not have. An excess-green baseline on "
          "the same images, carrying no learned parameters, reached a weed-class "
          "F1-score some twenty-four times lower than the three detectors, so the "
          "shortfall reported here measures the difficulty of the task and not a poor "
          "choice of model. The weed flora recorded here "
          "comprises two grasses, a sedge and a fourth species not identified to species "
          "but described locally as a grass, all pooled into a single class, so a detector "
          "built this way supports a decision about where to spray and not a choice of "
          "herbicide.",
    ]


# ───────────────────────────────── dựng bài ─────────────────────────────────

def build(cfg, stats, res, seedvar, boot, comp, journal: str = "sugartech") -> str:
    D.use_seed_variance(seedvar)
    rev, lk = load_review()
    primary, _ = scope(cfg)
    tr = cfg["train"]

    # Hướng dẫn của tạp chí: trang bìa mang tiêu đề, thông tin tác giả, abstract,
    # từ khoá; lời cảm ơn cũng đặt ở trang bìa. Không có yêu cầu làm mù, nên tên
    # tác giả nằm trong chính tệp bản thảo (khác RAE).
    L = [f"# {TITLE}", "",
         AUTHORS, "",
         AFFIL, "",
         CORRESP, "",
         "**Abstract** " + abstract(stats, rev, cfg), "",
         f"**Keywords** {KEYWORDS}", "",
"**Acknowledgements** The authors thank the management and staff of Tan Hung farm, "
         "Chau Thanh district, Tay Ninh province, for access to the sugarcane fields and "
         "for their assistance during image acquisition, and Nong Lam University Ho Chi "
         "Minh City for the provision of computing facilities.", "",
         "## Introduction", ""]
    L += [p + "\n" for p in introduction(stats, rev, cfg)]

    L += ["## Materials and Methods", ""]
    meth = methods(cfg, stats, res, seedvar, boot, journal=journal)
    # Hình 1 (quy trình) đi ngay sau đoạn mô tả địa điểm
    blocks = []
    for p in meth:
        blocks.append(p)
        if p.startswith("**Study area and image dataset."):
            blocks.append(figure_block(
                "fig1_workflow",
                "**Fig. 1** Workflow of the study, from image acquisition and annotation "
                "through model training to evaluation. The two annotation conventions run "
                "in parallel to separate test partitions and are never ranked against "
                "each other"))
    L += [b + "\n" for b in blocks]

    L += ["## Results", ""]
    for b in results_blocks(cfg, stats, res, seedvar, boot, comp, rev, lk):
        L.append(b + "\n")
    for b in D.qualitative_results(res, rev, fig="Fig. 5"):
        L.append(b + "\n")
    L.append(figure_block(
        "fig5_qualitative",
        "**Fig. 5** Detection results on the test set, first training run (seed 0), for "
        "Faster R-CNN (first column), RetinaNet (second column) and YOLOv10 (third "
        "column). **a**-**c** ground-truth annotation of the success case for each "
        "model; **d**-**f** the corresponding predictions; **g**-**i** ground-truth "
        "annotation of the failure case; **j**-**l** the corresponding predictions. "
        "Green boxes sugarcane, red boxes weed; predicted boxes carry the confidence "
        "score, and a yellow dashed box on a prediction panel marks an annotated weed "
        "that the model did not detect, so the misses can be read without comparing "
        "panels. For each model the success case is the test image at the median of "
        "per-image weed-class recall and the failure case the image at the tenth "
        "percentile, so neither is the most favourable nor the most unfavourable "
        "example available"
        + _shared_panel_note()
        + ". No scale bar is given because the ground footprint of the frame was not "
          "measured; the camera was held about 140 cm above the ground with the optical "
          "axis vertical") + "\n")

    L += ["## Discussion", ""]
    for b in discussion(cfg, stats, res, rev, lk):
        L.append(b + "\n")

    L += ["## Conclusion", ""]
    for b in conclusion(cfg, stats, res, rev):
        L.append(b + "\n")

    L += [""] + [d + "\n" for d in DECLARATIONS]

    src_refs = REFERENCES + EXTRA_REFS
    if journal == "aiia":
        from journal_style import to_elsevier
        refs, bad, unknown = to_elsevier(src_refs)
        if unknown:
            log(f"cảnh báo: {len(unknown)} tên tạp chí chưa có viết tắt LTWA "
                f"nên giữ tên đầy đủ — {unknown}")
    else:
        refs, bad = to_springer(src_refs)
        refs = sorted(refs, key=str.lower)
    if bad:
        log(f"cảnh báo: {len(bad)} tài liệu giữ nguyên kiểu RAE — {bad[:2]}")
    L += ["## References", ""] + [r + "\n" for r in refs]

    out = "\n".join(L)

    # Kiểm tại chỗ: bảng phải đánh số liên tục và không rơi mất.
    caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d)\*\*", out, re.M)]
    n_tbl = out.count("\n|---")
    if caps != list(range(1, len(caps) + 1)):
        raise SystemExit(f"Đánh số bảng không liên tục: {caps}")
    if n_tbl != len(caps):
        raise SystemExit(f"{len(caps)} chú thích bảng nhưng {n_tbl} bảng thật")
    log(f"kiểm tra bảng: {len(caps)} bảng, đánh số liên tục 1–{len(caps)}")
    return out


def main():
    cfg = load_cfg()
    res = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in MODEL_LABELS}
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    boot = json.loads((RESULTS / "bootstrap.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())

    md = build(cfg, stats, res, seedvar, boot, comp)
    p = OUT / "SugarTech_manuscript.md"
    p.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {p}")
    # Sugar Tech: Times Roman 10 pt, giãn đôi, lề 2,5 cm, đánh số trang tự động.
    d = to_docx(p, double_spaced=True, font_pt=10, page_numbers=True)
    if d:
        final = OUT / "SugarTech_manuscript.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
