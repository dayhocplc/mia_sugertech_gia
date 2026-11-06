#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phụ lục trực tuyến của bài RAE.

    python3 src/build_supplementary.py

Tạp chí giới hạn bài nguyên tác ở 15 trang chuẩn KỂ CẢ bảng và hình, nhưng vòng
phản biện yêu cầu nhiều phân tích độ nhạy mà bài không thể chứa hết. Mục
"ix. Supplementary Material" của hướng dẫn tác giả cho phép đưa bảng và dữ liệu
bổ sung ra một tệp riêng, chỉ xuất bản ở bản điện tử, đánh số S1, S2, …

Nguyên tắc tạp chí nêu rõ: bản thảo chính phải ĐỨNG ĐỘC LẬP được. Vì vậy mỗi
bảng ở đây đều có một câu tóm tắt tương ứng trong bài, và phụ lục chỉ chứa chi
tiết đầy đủ chứ không chứa kết luận mới.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_manuscript import AFFIL, AUTHORS, CORRESP, md_table, to_docx
from common import MODEL_LABELS, RAE, RESULTS, load_cfg, log, scope

OUT = RAE / "manuscript"
TITLE = "Supplementary file – for online publication only"

# Tiêu đề bài, theo tạp chí. Trước đây tệp này mang tiêu đề bản RAE kể cả trong
# bộ nộp Sugar Tech, và còn nói "S1 to S6" khi phụ lục đã có 15 bảng.
PAPER_TITLE = {
    "RAE": ("Detection completeness limits deep learning weed detection in "
            "field-grown sugarcane"),
    "SugarTech": ("Detection Completeness and Throughput Screening of Three Deep "
                  "Learning Detectors for Site-Specific Weed Control in Sugarcane"),
    # Bản gửi Artificial Intelligence in Agriculture dùng cùng tiêu đề; phụ lục
    # được Elsevier đăng dưới dạng "Appendix A. Supplementary data".
    "AIIA": ("Detection Completeness and Throughput Screening of Three Deep "
             "Learning Detectors for Site-Specific Weed Control in Sugarcane"),
}


def _header(prefix: str, n_tables: int) -> list[str]:
    """Phần đầu tệp phụ lục.

    Springer đòi mỗi tệp Electronic Supplementary Material tự mang tiêu đề bài,
    tên tạp chí, danh sách tác giả, đơn vị và email liên hệ, vì tệp được đăng
    nguyên dạng và người đọc có thể tải riêng nó.
    """
    head = [f"# {TITLE}", ""]
    if prefix == "SugarTech":
        head += [f"**Online Resource 1** for *{PAPER_TITLE[prefix]}*", "",
                 "Submitted to *Sugar Tech*", "",
                 AUTHORS, "", AFFIL, "", CORRESP, ""]
    elif prefix == "AIIA":
        head += [f"**Appendix A. Supplementary data** for *{PAPER_TITLE[prefix]}*", "",
                 "Submitted to *Artificial Intelligence in Agriculture*", "",
                 AUTHORS, "", AFFIL, "", CORRESP, ""]
    else:
        head += [f"Supplementary material for: *{PAPER_TITLE[prefix]}*.", ""]
    head += [f"These tables give in full the sensitivity analyses summarised in the main "
             f"text. They are referred to there as Supplementary Table S1 to "
             f"S{n_tables}. The main text stands alone without them.", ""]
    return head


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="RAE",
                    help="tiền tố tên tệp; dùng SugarTech cho bản nộp Sugar Tech")
    args = ap.parse_args()
    cfg = load_cfg()
    primary, _ = scope(cfg)
    rev = json.loads((RESULTS / "review_analysis.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    sv = json.loads((RESULTS / "seed_variance.json").read_text())
    order = sorted(primary, key=lambda k: -rev["coco_detail"]["models"][k]["AP50"]["mean"])

    L = []        # phần đầu được chèn sau, khi đã biết số bảng thực có

    # ── S1: cấu hình huấn luyện và suy luận đầy đủ ──
    rows = []
    for k in order:
        spec = cfg["models"][k]
        rows.append([
            MODEL_LABELS[k],
            "Detectron2 v0.6" if spec["framework"] == "detectron2" else "Ultralytics v8.3",
            f"{spec['imgsz']} × {spec['imgsz']}",
            "COCO-pretrained",
            "100" if spec["framework"] == "detectron2" else "300",
            "0.50 (head), 0.70 (RPN)" if k == "faster_rcnn"
            else ("0.50" if spec["framework"] == "detectron2" else "none (end-to-end head)"),
            "final epoch",
        ])
    L.append(md_table(
        ["Model", "Framework", "Input size", "Initialisation",
         "Max detections per image", "NMS IoU threshold", "Checkpoint evaluated"], rows,
        "**Supplementary Table S1. Training and inference configuration of the three "
        "bounding-box models, in full. All were trained for 50 epochs with AdamW, an "
        "initial learning rate of 0.001, cosine annealing, weight decay 0.0001, a "
        "three-epoch linear warm-up and an effective batch of 16 realised by gradient "
        "accumulation. Predictions scoring below 0.05 were discarded before any metric was "
        "computed. A seed fixes the Python, NumPy and PyTorch generators, and therefore the "
        "initialisation of the detection heads, the data ordering and the sampling of the "
        "training-time horizontal flip; the data partition and the augmented images are "
        "identical across seeds.**"))

    # ── S2: quét ngưỡng tin cậy ──
    cs = rev["conf_sweep"]
    rows = [[f"{r['conf']:.2f}"] + [f"{cs[k][i]['recall']:.3f} / {cs[k][i]['precision']:.3f}"
                                    f" / {cs[k][i]['f1']:.3f}" for k in order]
            for i, r in enumerate(cs[order[0]])]
    L.append(md_table(["Confidence threshold"] + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order],
                      rows,
                      "**Supplementary Table S2. Weed-class Recall, Precision and F1-score "
                      "in the full variant of the confusion matrix, as a function of the "
                      "confidence threshold, at a fixed IoU threshold of 0.50. Values are "
                      "means over each model's converged training runs. Confidence scores "
                      "are not calibrated to a common scale between frameworks, so a single "
                      "threshold does not represent the same operating point for every "
                      "model.**"))

    # ── S3: quét ngưỡng IoU ──
    iou = rev["iou_sweep"]
    rows = [[f"{r['iou']:.2f}"] + [f"{iou[k][i]['recall']:.3f} / {iou[k][i]['precision']:.3f}"
                                   f" / {iou[k][i]['f1']:.3f}" for k in order]
            for i, r in enumerate(iou[order[0]])]
    L.append(md_table(["IoU threshold"] + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order],
                      rows,
                      "**Supplementary Table S3. The same metrics as a function of the IoU "
                      "threshold at which a prediction is accepted as matching an "
                      "annotation, at a fixed confidence threshold of 0.30. The difference "
                      "between the rows measures how much of the apparent shortfall in "
                      "Recall is imprecise localisation rather than failure to detect.**"))

    # ── S4: recall phân tầng kích thước ──
    ss = rev["size_strata"]
    rows = []
    for band, label in (("small", "COCO small (< 32 × 32 px)"),
                        ("medium", "COCO medium (32–96 px)"),
                        ("large", "COCO large (> 96 × 96 px)")):
        rows.append([label, str(ss["coco_band_counts"][band])]
                    + [f"{ss['models'][k]['coco'][band]['recall_mean']:.3f}" for k in order])
    q = ss["quartile_edges_px2"]
    edges = ["< {:,.0f}".format(q[0]), "{:,.0f}–{:,.0f}".format(q[0], q[1]),
             "{:,.0f}–{:,.0f}".format(q[1], q[2]), "> {:,.0f}".format(q[2])]
    for i, nm in enumerate(("Q1", "Q2", "Q3", "Q4")):
        rows.append([f"{nm} ({edges[i]} px²)", str(ss["quartile_band_counts"][nm])]
                    + [f"{ss['models'][k]['quartile'][nm]['recall_mean']:.3f}" for k in order])
    L.append(md_table(["Size band", "Weed instances"] + [MODEL_LABELS[k] for k in order], rows,
                      "**Supplementary Table S4. Weed-class Recall in the full variant, "
                      "stratified by the area of the annotated bounding box, at a "
                      "confidence threshold of 0.30 and an IoU threshold of 0.50. The upper "
                      "block uses the absolute size bands of the COCO protocol; the lower "
                      "block uses quartiles of the area distribution of this test "
                      "partition, which divide the data evenly and are therefore the more "
                      "reliable instrument here. Areas are measured on the 1280 px exported "
                      "frame.**"))

    # ── S5: COCO AP/AR theo dải kích thước ──
    cd = rev["coco_detail"]
    keys = [("AP", "AP"), ("AP50", "AP50"), ("AP75", "AP75"),
            ("AP_small", "AP small"), ("AP_medium", "AP medium"), ("AP_large", "AP large"),
            ("AR100", "AR@100"), ("AR_small", "AR small"),
            ("AR_medium", "AR medium"), ("AR_large", "AR large")]
    rows = [[lab] + [f"{cd['models'][k][key]['mean']:.1f}" for k in order] for key, lab in keys]
    L.append(md_table(["Metric"] + [MODEL_LABELS[k] for k in order], rows,
                      "**Supplementary Table S5. Average precision and average recall under "
                      "the COCO protocol, by size band, averaged over each model's "
                      "converged training runs. The ground-truth test partition contains "
                      f"{cd['gt_area_bands']['small']} small, "
                      f"{cd['gt_area_bands']['medium']} medium and "
                      f"{cd['gt_area_bands']['large']} large instances across both classes, "
                      "so the small-band figures rest on few objects and are reported for "
                      "the trend rather than as values.**"))

    # ── S6: độ nhạy với giao thức checkpoint ──
    cs6 = rev.get("checkpoint_sensitivity", {}).get("yolov10", {})
    if cs6.get("last") and cs6.get("best_on_val"):
        rows = [["Final epoch (protocol used here)",
                 f"{cs6['last']['mean']:.2f} ± {cs6['last']['sd']:.2f}", str(cs6['last']['n'])],
                ["Best epoch on the validation partition",
                 f"{cs6['best_on_val']['mean']:.2f} ± {cs6['best_on_val']['sd']:.2f}",
                 str(cs6['best_on_val']['n'])]]
        L.append(md_table(["Checkpoint rule", "AP50 (mean ± SD across seeds)", "Runs"], rows,
                          "**Supplementary Table S6. Sensitivity of YOLOv10 to the choice "
                          "of evaluated checkpoint. Detectron2 writes only a final "
                          "checkpoint, so the two Detectron2 models admit no equivalent "
                          "choice; taking each framework's default would therefore have "
                          f"given YOLOv10 an advantage of {cs6['delta_mean']:.1f} AP50 and "
                          f"inflated its between-seed standard deviation by a factor of "
                          f"{cs6['sd_ratio']:.0f}. The final-epoch rule is used throughout "
                          "the main text.**"))

    # ── S7: kiểm định theo seed, đầy đủ ba chỉ số × ba cặp ──
    rows = []
    for metric, lab in (("ap50", "AP50"),
                        ("weed_recall_full", "Weed Recall (full)"),
                        ("weed_f1_full", "Weed F1 (full)")):
        for key, v in rev["tests"][metric].items():
            a_, b_ = key.split("__")
            rows.append([lab, f"{MODEL_LABELS[a_]} − {MODEL_LABELS[b_]}",
                         f"{v['diff']:+.3f}", f"{v['p_raw']:.3f}", f"{v['p_holm']:.3f}",
                         f"{v['hedges_g']:+.2f} [{v['g_ci95'][0]:+.2f}, {v['g_ci95'][1]:+.2f}]"])
    L.append(md_table(["Metric", "Comparison", "Difference", "P (raw)", "P (Holm)",
                       "Hedges' g [95% CI]"], rows,
                      "**Supplementary Table S7. Welch's t-test on the seed-level values "
                      "of each metric, for all three pairwise comparisons, with P-values "
                      "adjusted across the three comparisons by the Holm–Bonferroni "
                      "procedure. These tests hold the test partition fixed and therefore "
                      "reflect training variability only; the two-level bootstrap reported "
                      "in the main text releases both sources at once and is the basis for "
                      "the paper's statements about whether architectures differ.**"))

    # ── S8: bootstrap hai tầng, đầy đủ ──
    tl = rev["two_level_bootstrap"]
    rows = []
    for key, v in tl["differences"].items():
        a_, b_ = key.split("__")
        for met, lab in (("recall", "Recall"), ("f1", "F1-score")):
            d = v[met]
            rows.append([f"Weed {lab}", f"{MODEL_LABELS[a_]} − {MODEL_LABELS[b_]}",
                         f"{d['mean']:+.3f}",
                         f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]",
                         "yes" if d["excludes_zero"] else "no"])
    L.append(md_table(["Metric", "Comparison", "Difference", "95% interval",
                       "Excludes zero"], rows,
                      f"**Supplementary Table S8. Two-level bootstrap over {tl['n_boot']} "
                      f"iterations, resampling the {tl['n_images']} test images and the "
                      "converged training runs together and taking each difference within "
                      "an iteration. This is the interval the main text refers to wherever "
                      "it states that two architectures do or do not differ.**"))

    # ── S9: kết quả theo buổi chụp (chuyển từ bài chính để giữ giới hạn trang) ──
    lk = json.loads((RESULTS / "leakage_analysis.json").read_text())
    se = lk.get("sessions") or {}
    if se:
        rows = []
        for name, d in se.items():
            cells = [name, f"{d['n_train']} / {d['n_valid']} / {d['n_test']}"]
            for k in order:
                v = d["models"][k]
                cells.append(f"{v['recall']:.3f} / {v['precision']:.3f} / {v['f1']:.3f}")
            rows.append(cells)
        L.append(md_table(
            ["Capture session", "Source images (train / valid / test)"]
            + [f"{MODEL_LABELS[k]} (R / P / F1)" for k in order], rows,
            "**Supplementary Table S9. Weed-class Recall, Precision and F1-score in the "
            "full variant, computed separately on each capture session of the test "
            "partition, with the number of source images each session contributed to the "
            "three partitions. Values are means over each model's converged runs. The "
            "sessions differ simultaneously in plot, crop age and illumination, so the "
            "spread between them cannot be attributed to any single factor. Note that the "
            "random partitioning left the validation partition with no images from the "
            "first session.**"))

    # ── S10: hai mô hình huấn luyện trên nhãn đa giác ──
    res = {k: json.loads((RESULTS / f"{k}.json").read_text())
           for k in ("mask_rcnn", "yolov8_seg")
           if (RESULTS / f"{k}.json").exists()}
    if res:
        from common import clean_fps

        def _fps(key):
            # Hai mô hình đa giác chưa có bộ đo thông lượng riêng; để gạch chứ
            # không lấy số đo một lượt trong vòng lặp đánh giá.
            try:
                return f"{clean_fps(key)[0]:.1f}"
            except SystemExit:
                return "–"

        rows = []
        for k, d in res.items():
            f_ = d["metrics"]["filtered"]
            wb = d["metrics"]["with_background"]["per_class"]
            rows.append([MODEL_LABELS[k], f"{d['coco']['AP50']:.1f}",
                         f"{wb['weed']['recall']:.3f}", f"{f_['macro']['f1']:.3f}",
                         f"{f_['overall_accuracy']:.3f}", _fps(k)])
        L.append(md_table(
            ["Model", "AP50", "Weed recall (full)", "F1 (two-class mean)", "Accuracy",
             "FPS (repeated benchmark)"], rows,
            "**Supplementary Table S10. The two polygon-trained models on the polygon test "
            "partition (50 images) under the same protocol. These values are not "
            "comparable with the main-text tables, which are computed on a different test "
            "partition under a different annotation convention, and each model was trained "
            "once only.**"))

    # ── S11: bootstrap theo cụm chụp so với theo ảnh ──
    cb = (rev or {}).get("cluster_bootstrap") or {}
    if cb.get("available"):
        rows = []
        for k, v in cb["models"].items():
            for met, nm in (("recall", "Weed recall"), ("f1", "Weed F1-score")):
                a, b = v["by_image"][met], v["by_cluster"][met]
                rows.append([MODEL_LABELS[k], nm,
                             f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]", f"{a['ci_width']:.3f}",
                             f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]", f"{b['ci_width']:.3f}",
                             f"{b['ci_width'] / a['ci_width']:.2f}"])
        L.append(md_table(
            ["Model", "Measure", "95% CI, images resampled", "Width",
             "95% CI, bursts resampled", "Width", "Ratio"], rows,
            f"**Supplementary Table S11. Bootstrap confidence intervals when test images "
            f"are resampled independently and when whole capture bursts are resampled "
            f"instead ({cb['n_clusters']} bursts, largest {cb['cluster_sizes'][0]} images). "
            "Frames within a burst are seconds apart and share illumination, plot and "
            "canopy, so the independent-image interval is too narrow. No comparison in "
            "the main text reverses under the wider interval.**"))

    # ── S12: ba cách đo diện tích được xử lý ──
    cf = RESULTS / "coverage_analysis.json"
    if cf.exists():
        cov = json.loads(cf.read_text())
        ar = (rev or {}).get("area_recall", {})
        rows = []
        for k in MODEL_LABELS:
            if k not in cov["union_area_recall"]:
                continue
            px = cov["pixel_coverage"].get("models", {}).get(k, {})
            rows.append([
                MODEL_LABELS[k],
                f"{ar.get(k, {}).get('instance_recall', {}).get('mean', float('nan')):.3f}",
                f"{cov['union_area_recall'][k]['mean']:.3f}",
                (f"{px['pixel_coverage_recall']['mean']:.3f}" if px else "—"),
                *[f"{cov['grid_recall'][g]['models'][k]['cell_recall']['mean']:.3f}"
                  for g in sorted(cov["grid_recall"], key=float, reverse=True)]])
        heads = ["Model", "By instance", "By box-union area", "By weed pixel"] + [
            f"Grid cell {g} frame"
            for g in sorted(cov["grid_recall"], key=float, reverse=True)]
        L.append(md_table(heads, rows,
            "**Supplementary Table S12. Weed recall under five readings of what counts as "
            "treated, same predictions and same threshold throughout. By instance is the "
            "main-text figure. By box-union area weights each annotation by its area and "
            "removes the double counting of overlapping boxes. By weed pixel uses the "
            f"polygon annotations available for {cov['pixel_coverage'].get('n_images', 0)} of "
            "the test images. The grid readings treat a cell as covered if it contains "
            "both an annotated and a predicted weed, as a nozzle zone would. Cell size is "
            "given as a fraction of the frame width because the ground footprint of the "
            "frame was not measured.**"))

    # ── S13: độ nhạy của ngưỡng thông lượng ──
    ts = (rev or {}).get("throughput_sensitivity") or {}
    if ts.get("grid"):
        v_ref = ts["reference_speed_kmh"]
        rows = []
        for g in ts["grid"]:
            if g["speed_kmh"] != v_ref:
                continue
            met = ", ".join(MODEL_LABELS[k] for k in g["models_meeting"]) or "none"
            rows.append([f"{g['advance_m']:.2f}", f"1/{1 / g['embedded_penalty']:.0f}",
                         f"{g['required_fps_onboard']:.1f}",
                         f"{g['required_fps_workstation']:.1f}", met])
        L.append(md_table(
            ["Frame advance (m)", "Embedded penalty", "Required onboard FPS",
             "Required workstation FPS", "Models meeting it"], rows,
            f"**Supplementary Table S13. Which architectures meet the throughput "
            f"requirement at {v_ref:g} km/h as the two assumptions behind it are varied. "
            "The frame advance sets how far the implement may travel between frames and "
            "is an input to the requirement, not a measurement; the embedded penalty is "
            "the assumed ratio of embedded to workstation throughput, taken from vendor "
            "specifications. The main text uses "
            f"{[g for g in ts['grid'] if g['speed_kmh'] == v_ref][0]['advance_m']:.2f} m "
            "with the more conservative penalty. The conclusion that only one "
            "architecture qualifies does not hold across the whole grid.**"))

    # ── S14: recall theo độ phân giải mặt đất ──
    gf = RESULTS / "gsd_analysis.json"
    if gf.exists():
        g = json.loads(gf.read_text())
        rows = []
        for s_ in [f"{x:g}" for x in g["scales"]]:
            first = next(iter(g["models"].values()))
            if s_ not in first:
                continue
            rows.append([f"{1 / float(s_):.2f}×",
                         f"{first[s_]['equivalent_camera_height_cm']:.0f}"]
                        + [f"{g['models'][k][s_]['weed_recall']['mean']:.3f}"
                           for k in g["models"]])
        L.append(md_table(
            ["Effective GSD", "Equivalent camera height (cm)"]
            + [MODEL_LABELS[k] for k in g["models"]], rows,
            "**Supplementary Table S14. Weed recall when image detail is reduced to "
            "simulate a camera mounted further from the canopy. Each test image is "
            "downsampled and resampled back to its original size, so the input size, the "
            "annotations and the detector are unchanged and only the detail per object is "
            "lost. Equivalent camera height assumes ground sample distance scales linearly "
            "with height from the 140 cm used here. Means over the same converged runs as "
            "the main text.**"))

    # ── S15: trần số phát hiện mỗi ảnh ──
    df = RESULTS / "detcap_analysis.json"
    if df.exists():
        dc = json.loads(df.read_text())
        rows = []
        for c in dc["caps"]:
            v = dc["by_cap"][str(c)]
            rows.append([str(c), f"{v['n_raw_detections']['mean']:.0f}",
                         f"{v['max_detections_per_image']['mean']:.0f}",
                         f"{v['AP50']['mean']:.2f}", f"{v['weed_recall']['mean']:.3f}",
                         f"{v['weed_precision']['mean']:.3f}",
                         f"{v['weed_f1']['mean']:.3f}"])
        L.append(md_table(
            ["Detections per image allowed", "Raw detections retained",
             "Most returned above threshold", "AP50", "Weed recall", "Weed precision",
             "Weed F1-score"], rows,
            f"**Supplementary Table S15. {MODEL_LABELS[dc['model']]} re-evaluated with "
            "the framework's per-image detection cap raised, using the same weights and "
            f"the same {dc['by_cap'][str(dc['caps'][0])]['n_runs']} converged runs. "
            "Raising the cap retains more detections in the low-confidence tail but "
            "leaves every reported figure unchanged, so the cap does not bound the "
            "results of this model.**"))

    # ── S16: suy luận theo ô cắt trên ảnh gốc ──
    tf = RESULTS / "tiling_analysis.json"
    if tf.exists():
        tl = json.loads(tf.read_text())
        rows = []
        for k_ in [str(x) for x in tl["tiles"]]:
            first = next(iter(tl["models"].values()))
            if k_ not in first:
                continue
            rows.append([f"{k_} × {k_}",
                         f"{first[k_]['effective_gsd_multiplier']:.2f}×",
                         f"{first[k_]['equivalent_camera_height_cm']:.0f}"]
                        + [f"{tl['models'][m][k_]['weed_recall']['mean']:.3f} / "
                           f"{tl['models'][m][k_]['weed_precision']['mean']:.3f}"
                           for m in tl["models"]])
        L.append(md_table(
            ["Tile grid", "Effective GSD", "Equivalent camera height (cm)"]
            + [f"{MODEL_LABELS[m]} (R / P)" for m in tl["models"]], rows,
            "**Supplementary Table S16. Weed recall and precision when the native "
            f"{tl['n_test_images_with_raw']} test frames retained at 4080 × 3072 pixels "
            "are divided into a grid of overlapping tiles, each tile rescaled to the "
            "1280 px export size and the predictions merged by non-maximum suppression. "
            "The one-tile row rescales the whole frame and therefore reproduces the "
            "pipeline of the main text, serving as the internal control. The models were "
            "not retrained, so each row presents them with objects at a different scale "
            "from the one they were trained on.**"))

    # ── S17: đánh giá chia theo phiên chụp ──
    sf = RESULTS / "session_split.json"
    if sf.exists():
        sp = json.loads(sf.read_text())
        rows = []
        for tag, f_ in sorted(sp["folds"].items()):
            for m in order:
                v = f_["models"].get(m)
                head = [tag, f_["condition"], f_["held_out_session"] or "-",
                        str(f_["n_train"]), str(f_["n_test"]), MODEL_LABELS[m]]
                # một lượt sập không phải số đo: để trống thay vì in số 0 của nó
                if v is None or v.get("failed"):
                    rows.append(head + ["--"] * 4)
                    continue
                rows.append(head + [f"{v['weed_recall']:.3f}",
                                    f"{v['weed_precision']:.3f}",
                                    f"{v['weed_f1']:.3f}",
                                    f"{v.get('AP50', float('nan')):.1f}"])
        gap = ("" if not sp.get("excluded") else
               " A dash marks a fold whose training diverged under the fixed recipe and "
               "which therefore yields no measurement; the matched member of such a pair "
               "is reported but contributes no difference.")
        L.append(md_table(
            ["Fold", "Split", "Held-out session", "Train images", "Test images",
             "Model", "Weed recall", "Weed precision", "Weed F1-score", "AP50"], rows,
            "**Supplementary Table S17. Leave-one-session-out evaluation against a "
            "matched random split. Each grouped fold trains on two capture sessions and "
            "tests on the third; each control fold draws the same number of training and "
            "test images at random from all three sessions, so the only difference "
            "between a pair is whether the test images come from a session the model has "
            "seen. One version of each source image is used and the augmented copies are "
            "omitted, so the absolute values are below those of the main text and only "
            "the difference within a pair is interpretable." + gap + "**"))

    # ── S18: đối chứng không học sâu ──
    bf = RESULTS / "baseline_exg.json"
    if bf.exists():
        bd = json.loads(bf.read_text())
        rows = []
        for key, lab in (("exg_all", "ExG, every component"),
                         ("exg_size", "ExG, area filter from validation")):
            v = bd["variants"].get(key)
            if not v:
                continue
            rows.append([lab, f"{v['weed_recall']:.3f}", f"{v['weed_precision']:.3f}",
                         f"{v['weed_f1']:.3f}", f"{v['detections_per_image']:.1f}"])
        for k in order:
            w = rev["full_metrics"][k]["weed"]
            rows.append([MODEL_LABELS[k], f"{w['recall']['mean']:.3f}",
                         f"{w['precision']['mean']:.3f}", f"{w['f1']['mean']:.3f}",
                         "—"])
        tune = bd.get("size_filter_tuning", {})
        lo, hi = tune.get("area_lo_px2"), tune.get("area_hi_px2")
        band = ("" if lo is None and hi is None else
                f" The area filter keeps components between {lo:.0f} and {hi:.0f} px² on "
                "the 1280 px exported frame, the 5th to 95th "
                "percentile of annotated weed-box area on the validation partition; it "
                "was chosen there and applied unchanged to the test partition.")
        L.append(md_table(
            ["Method", "Weed recall", "Weed precision", "Weed F1-score",
             "Detections per image"], rows,
            "**Supplementary Table S18. A baseline that uses no learning, on the same "
            "test partition and under the same convention as Table 2 of the main text. "
            "The excess-green index is computed as 2G − R − B on the channel-normalised "
            "image, thresholded by Otsu's method, opened and closed with a 7 × 7 "
            "elliptical kernel, and every connected component of at least 64 px becomes "
            "one weed-class box with a confidence of one." + band + " Sugarcane and the "
            "weed flora present are both green, so a colour index separates vegetation "
            "from soil rather than weed from crop; the row is given to show the margin "
            "the learned detectors add, not as a competing method.**"))

    # ── S19: ma trận điều kiện huấn luyện theo thang ──
    tf = RESULTS / "tile_scale_training.json"
    if tf.exists():
        ts = json.loads(tf.read_text())
        mx = ts.get("matrix") or {}
        label = {"orig1": "whole frame", "tile2": "2 × 2 tiles"}
        rows = []
        for k in order:
            for tr in ("orig1", "tile2"):
                for te in ("tile2", "orig1"):
                    v = (mx.get(tr) or {}).get(te, {}).get(k)
                    if not v:
                        continue
                    rows.append([MODEL_LABELS[k], label[tr], label[te],
                                 f"{v['weed_recall']:.3f}",
                                 f"{v['weed_precision']:.3f}",
                                 f"{v['weed_f1']:.3f}",
                                 f"{v.get('AP50', float('nan')):.1f}"])
        if rows:
            n_src = ts["build"]["tile2"]["splits"]["train"]["source_images"]
            lr = ts.get("retinanet_base_lr")
            lr_note = ("" if not lr else
                       " RetinaNet diverged at the learning rate of the main text on "
                       f"the tiled partitions, so {lr:g} was used for both of its "
                       "conditions; its absolute values are therefore not comparable "
                       "with Table 2 either.")
            L.append(md_table(
                ["Model", "Trained on", "Scored on", "Weed recall", "Weed precision",
                 "Weed F1-score", "AP50"], rows,
                "**Supplementary Table S19. Training scale against test scale. Both "
                f"training conditions are built from the same {n_src} training images "
                "that retain native frames and share one recipe; neither uses the "
                "augmented copies, so every value here is below the main text and only "
                "comparisons inside this table are interpretable. The rows scored on "
                "the tiled partition isolate training scale, because ground truth, "
                "source images and recipe are then held fixed. The tiled condition "
                "receives four times as many optimiser updates as its control, since "
                "the recipe fixes epochs and tiling quadruples the image count, so an "
                "adverse result for it is conservative." + lr_note + "**"))

    # ── hình hỗ trợ: đường học (ma trận nhầm lẫn ở bài chính) ──
    from build_manuscript import figure_block
    L.append(figure_block(
        "fig4_training",
        "Supplementary Figure S1. Training curves for the three bounding-box models, "
        "first training run (seed 0). Left axis: training loss; right axis, for YOLOv10: "
        "validation mAP@50. The abscissa is epochs for all three panels, the Detectron2 "
        "logs converted at 181 iterations per epoch; that framework computes no "
        "validation loss under this configuration. The training loss of all three is "
        "still falling at the final epoch, so none had reached a plateau, and the "
        "validation mAP@50 of YOLOv10 stops improving after roughly the tenth epoch. The "
        "break in the YOLOv10 curve marks two epochs at which the logged box-regression "
        "loss was non-finite; training recovered in the following epoch."))

    import re
    n_tables = max((int(m) for m in re.findall(r"Supplementary Table S(\d+)\.",
                                               "\n".join(L))), default=0)
    md = "\n\n".join(_header(args.prefix, n_tables) + L) + "\n"
    stem = args.prefix
    p = OUT / f"{stem}_supplementary.md"
    p.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {p}")
    d = to_docx(p)
    if d:
        final = OUT / f"{stem}_supplementary.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
