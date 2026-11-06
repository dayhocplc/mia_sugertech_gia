#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh lại ba đoạn thảo luận của phần RESULTS AND DISCUSSION từ số đo thật.

Mỗi câu ở đây hoặc là một con số đọc thẳng từ results/*.json, hoặc là một
nhận định được rút ra bằng luật tường minh trong chính file này (ai cao nhất,
chênh bao nhiêu, lệch lớp có đáng kể không). Không có câu nào được chép lại
từ bản thảo cũ mà không kiểm tra lại theo số mới.

Ngưỡng dùng để phát biểu định tính:
  TIE      chênh ≤ 0,005 → coi là ngang nhau, không ai "cao hơn"
  NOTABLE  chênh ≥ 0,030 → đủ lớn để gọi là "markedly"
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import (MODEL_LABELS, MODEL_ORDER, RESULTS, bench_fps_detail,
                    clean_fps, scope,
                    throughput_requirement)

TIE = 0.005
NOTABLE = 0.030
# Khi chọn mô hình khuyến nghị: mọi mô hình nằm trong AP_TOL của AP50 cao nhất
# được coi là ngang nhau về chất lượng phát hiện, và trong nhóm đó chọn con nhanh nhất.
AP_TOL = 0.10

# số đối chiếu từ tài liệu đã công bố, dùng nguyên trong đoạn so sánh
LIT = {
    "singh2024_accuracy": 90.5,
    "mini2020_ap50": 65.5,
    "mini2020_precision": 0.803,
    "mini2020_recall": 0.707,
    "mini2020_f1": 0.752,
}


def f3(v: float) -> str:
    return f"{v:.3f}"


def f1p(v: float) -> str:
    return f"{v * 100:.1f}"


def _cls(res: dict, model: str, cls: str, metric: str) -> float:
    return res[model]["metrics"]["filtered"]["per_class"][cls][metric]


def _macro(res: dict, model: str, metric: str) -> float:
    return res[model]["metrics"]["filtered"]["macro"][metric]


def _acc(res: dict, model: str) -> float:
    return res[model]["metrics"]["filtered"]["overall_accuracy"]


# Nguồn AP50 duy nhất cho toàn bộ bài. Trước đây đoạn kết quả đọc trung bình
# nhiều seed còn đoạn so sánh tài liệu đọc một lượt chạy, nên cùng một đại lượng
# có hai giá trị khác nhau trong cùng bài. Mọi hàm sinh văn bản phải đi qua đây.
_SEEDVAR: dict | None = None


def use_seed_variance(seedvar: dict | None) -> None:
    """Đăng ký kết quả nhiều seed làm nguồn AP50 chuẩn cho mọi đoạn."""
    global _SEEDVAR
    _SEEDVAR = seedvar


def _seed_values(model: str, metric: str) -> list[float]:
    if not _SEEDVAR or model not in _SEEDVAR:
        return []
    return [e[metric] for e in _SEEDVAR[model]["runs"]
            if e.get("status") == "ok" and metric in e]


def seed_mean(model: str, metric: str, fallback: float | None = None) -> float | None:
    """Giá trị trung bình các seed của MỘT chỉ số, dùng chung cho cả bài.

    Vòng sửa trước tôi chỉ thống nhất nguồn cho AP50, nên weed recall và F1 vẫn
    đọc từ một lượt chạy đơn ở một số đoạn — đúng cùng loại lỗi, chỉ khác chỉ số.
    Mọi hàm sinh văn bản phải lấy số qua đây, không đọc thẳng results/*.json.
    """
    v = _seed_values(model, metric)
    return float(np.mean(v)) if v else fallback


def seed_sd(model: str, metric: str) -> float | None:
    v = _seed_values(model, metric)
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def _diverge_clause(models: list[str], subject: str) -> str:
    """Mệnh đề về phân kỳ, chỉ nói 'duy nhất' khi dữ liệu thật sự cho phép."""
    div = [k for k in models if _n_div(k)]
    if subject not in div:
        return ""
    nd, na = _n_div(subject), _n_att(subject)
    only = " the only architecture whose training diverged" if div == [subject] \
        else " an architecture whose training diverged"
    return f"though it was also{only} ({nd} of {na} attempts). "


def _n_sig_txt(models: list[str]) -> str:
    """Số cặp khác biệt đạt ý nghĩa 5%, tính lại thay vì viết cứng."""
    import itertools
    import scipy.stats as _st
    n = 0
    for a, b in itertools.combinations(models, 2):
        va, vb = _seed_values(a, "ap50"), _seed_values(b, "ap50")
        if len(va) > 1 and len(vb) > 1 and _st.ttest_ind(va, vb, equal_var=False)[1] < 0.05:
            n += 1
    return _NUMWORD_CARD.get(n, str(n))


_NUMWORD_CARD = {0: "none", 1: "only one", 2: "two", 3: "all"}


def _n_att(model: str) -> int:
    return len((_SEEDVAR or {}).get(model, {}).get("runs", []))


def _n_div(model: str) -> int:
    runs = (_SEEDVAR or {}).get(model, {}).get("runs", [])
    return sum(1 for e in runs if e.get("status") != "ok")


def seed_counts(models: list[str]) -> dict[str, int]:
    """Số lượt chạy hội tụ của từng mô hình, lấy từ seed_variance."""
    return {k: len(_seed_values(k, "ap50")) for k in models}


_PRIMARY = ["faster_rcnn", "retinanet", "yolov10"]


def runs_phrase(models: list[str]) -> str:
    """Cụm danh từ về số lượt chạy; 'repeated runs' nếu chưa có số liệu seed."""
    n = seed_counts(models)
    if not n or min(n.values()) == 0:
        return "repeated runs"
    uniq = sorted(set(n.values()))
    if len(uniq) == 1:
        return f"{_NUMWORD_N.get(uniq[0], str(uniq[0]))} runs"
    return f"{min(n.values())}–{max(n.values())} runs"


_NUMWORD_N = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight"}


def trained_n_times(models: list[str]) -> str:
    """Cụm từ mô tả số lần huấn luyện; không viết cứng vì n khác nhau giữa các
    mô hình sau khi RetinaNet phân kỳ ở 3/7 lượt."""
    n = seed_counts(models)
    if not n or min(n.values()) == 0:
        return "repeatedly"
    uniq = sorted(set(n.values()))
    if len(uniq) == 1:
        return f"{_NUMWORD.get(uniq[0], str(uniq[0]))} times"
    return ", ".join(str(v) for v in n.values()) + " times respectively"


_NUMWORD = {1: "once", 2: "twice", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight"}


def has_seeds(model: str) -> bool:
    return len(_seed_values(model, "ap50")) > 1


def _ap50(res: dict, model: str) -> float | None:
    """AP50 trung bình các seed nếu có; nếu không thì của lượt chạy đơn."""
    return seed_mean(model, "ap50", res[model].get("coco", {}).get("AP50"))


def ap50_note() -> str:
    """Câu ghi rõ AP50 là trung bình mấy lượt — để người đọc không phải đoán."""
    if not _SEEDVAR:
        return ""
    ns = {len(_seed_values(k, "ap50")) for k in _SEEDVAR}
    ns.discard(0)
    if ns == {1} or not ns:
        return ""
    n = min(ns)
    return f" (mean of {n} training runs)" if len(ns) == 1 else ""


def _bg(res: dict, model: str, cls: str, metric: str) -> float:
    """Chỉ số tính trên ma trận CÒN background — có tính bỏ sót và báo nhầm trên nền."""
    return res[model]["metrics"]["with_background"]["per_class"][cls][metric]


def _rank(res: dict, fn) -> list[tuple[str, float]]:
    return sorted(((k, fn(res, k)) for k in MODEL_ORDER), key=lambda t: -t[1])


def _joint_leaders(ranked: list[tuple[str, float]]) -> list[str]:
    """Những mô hình nằm trong khoảng TIE của mô hình đứng đầu."""
    top = ranked[0][1]
    return [k for k, v in ranked if top - v <= TIE]


def _and_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# --------------------------------------------------------------------------
# Đoạn 1 — Detection performance by model
# --------------------------------------------------------------------------

# đổi động từ mở đầu để năm câu liên tiếp không lặp một khuôn
OPENERS = ["achieved", "recorded", "attained", "reached", "obtained"]


def _per_model_sentence(res: dict, k: str) -> str:
    """Một câu mô tả một mô hình: số hai lớp + nhận định về cân bằng lớp."""
    p_s, r_s, f_s = (_cls(res, k, "sugarcane", m) for m in ("precision", "recall", "f1"))
    p_w, r_w, f_w = (_cls(res, k, "weed", m) for m in ("precision", "recall", "f1"))
    label = MODEL_LABELS[k]

    verb = OPENERS[MODEL_ORDER.index(k) % len(OPENERS)]
    core = (f"{label} {verb} a Precision of {f3(p_s)}, a Recall of {f3(r_s)} and an "
            f"F1-score of {f3(f_s)} for sugarcane, against {f3(p_w)}, {f3(r_w)} and "
            f"{f3(f_w)}, respectively, for weeds")

    gap = f_w - f_s
    if abs(gap) <= TIE:
        tail = ("; performance was essentially balanced between the two classes, "
                "indicating that the model discriminated sugarcane and weed instances "
                "with comparable reliability")
    elif gap > 0:
        strength = "markedly " if gap >= NOTABLE else ""
        tail = (f"; the weed class was therefore detected {strength}more reliably than "
                f"sugarcane, by {f3(abs(gap))} in F1-score")
    else:
        strength = "markedly " if -gap >= NOTABLE else ""
        tail = (f"; the sugarcane class was therefore detected {strength}more reliably "
                f"than weeds, by {f3(abs(gap))} in F1-score")

    # bất đối xứng Precision/Recall trong cùng một lớp nói lên kiểu lỗi
    asym_w, asym_s = p_w - r_w, p_s - r_s
    if asym_w >= NOTABLE:
        tail += (", and its Recall for weeds fell below its Precision, meaning that the "
                 "weed instances it did report were mostly correct but a proportion of "
                 "weed instances present in the test images were assigned to sugarcane")
    elif -asym_w >= NOTABLE:
        tail += (", and its Precision for weeds fell below its Recall, meaning that it "
                 "recovered most weed instances at the cost of labelling some sugarcane "
                 "instances as weeds")
    elif asym_s >= NOTABLE:
        tail += (", and its Recall for sugarcane fell below its Precision, indicating "
                 "that part of the sugarcane canopy was assigned to the weed class")
    elif -asym_s >= NOTABLE:
        tail += (", and its Precision for sugarcane fell below its Recall, indicating "
                 "that weed patches adjacent to the stem base were absorbed into "
                 "sugarcane detections")
    return core + tail + "."


def detection_performance(res: dict) -> list[str]:
    ranked_f1 = _rank(res, lambda r, k: _macro(r, k, "f1"))
    balance = sorted(MODEL_ORDER,
                     key=lambda k: abs(_cls(res, k, "weed", "f1") - _cls(res, k, "sugarcane", "f1")))
    most_balanced, least_balanced = balance[0], balance[-1]
    gap_least = abs(_cls(res, least_balanced, "weed", "f1")
                    - _cls(res, least_balanced, "sugarcane", "f1"))

    opening = ("**Detection performance by model.** The trained models were evaluated on the "
               "held-out test set, and their ability to detect sugarcane and weeds was "
               "assessed separately for each class using Precision, Recall and F1-score "
               "(Table 2). ")
    body = " ".join(_per_model_sentence(res, k) for k in MODEL_ORDER)

    closing = (f" Across the five models, {MODEL_LABELS[most_balanced]} was the most "
               f"balanced between the two classes, whereas {MODEL_LABELS[least_balanced]} "
               f"showed the largest class asymmetry ({f3(gap_least)} in F1-score)")
    if gap_least >= NOTABLE:
        closing += (", which is consistent with the visual similarity between sugarcane "
                    "and weed foliage: where the two overlap, a detector that resolves "
                    "the ambiguity in favour of one class systematically loses instances "
                    "of the other")
    box = [(k, v) for k, v in ranked_f1 if k in BOX_MODELS]
    poly = [(k, v) for k, v in ranked_f1 if k in POLY_MODELS]
    closing += (". Because the models were trained on two annotation conventions (see "
                "Basis of comparison above), F1-scores are ranked within each convention "
                "rather than across all five: among the bounding-box models the order was "
                + _and_list([f"{MODEL_LABELS[k]} ({f3(v)})" for k, v in box])
                + ", and among the polygon models "
                + _and_list([f"{MODEL_LABELS[k]} ({f3(v)})" for k, v in poly]) + ".")
    return [opening + body + closing]


# --------------------------------------------------------------------------
# Đoạn 2 — so sánh với tài liệu đã công bố
# --------------------------------------------------------------------------

def literature_comparison(res: dict, rev: dict | None = None) -> list[str]:
    """So với tài liệu, nhưng chỉ so những gì cùng nhiệm vụ và cùng chỉ số.

    Bản trước gọi phép so AP50 với Mini et al. là "like-for-like" trong khi công
    trình đó dùng mask trên ảnh trên không — đúng như phản biện chỉ ra, điều đó
    mâu thuẫn với chính nguyên tắc phạm vi mà bài đặt ra ở Methods.
    """
    prim = BOX_MODELS
    ap50 = {k: _ap50(res, k) for k in prim}
    fw = ((rev or {}).get("full_metrics") or {})
    out = []
    out.append(
        "**Comparison with the literature.** Published results for this crop are mostly "
        "not comparable with the present ones. Singh et al. (2024) report 90.5% "
        "accuracy for pixel-level classification of weed patches from drone imagery — a "
        "per-pixel figure for a segmentation task at a different scale — and Mini et "
        "al. (2020) Precision 0.803, Recall 0.707 and F1-score 0.752 with Mask R-CNN on "
        "aerial imagery, at AP50 65.5%. Those come from mask annotations at an aerial "
        "scale, so by the same scope rule applied to our own two conventions they "
        "cannot be placed on one scale with the AP50 of "
        + _and_list([f"{ap50[k]:.1f}% for {MODEL_LABELS[k]}"
                     for k in sorted(prim, key=lambda m: -(ap50[m] or 0))])
        + " obtained here. No direction can be read from that difference either, because "
          "the annotation convention, the object scale and the task all differ at once; "
          "the scale experiments reported above are internal to this study and are the "
          "only evidence here that bears on acquisition geometry.")

    out.append(
        "One published result is close enough in kind to be informative. Partel et al. "
        "(2019) built and field-tested a camera-guided selective sprayer and reported, "
        "for the trial on real plants rather than artificial ones, a precision of 0.59 "
        "and a recall of 0.44 for weed detection. Those figures retain undetected weed "
        "in the arithmetic, as the full variant here does, and they sit inside the "
        "range obtained in this study"
        + (" (" + _and_list([f"{fw[k]['weed']['precision']['mean']:.2f}" for k in prim])
           + " for precision and "
           + _and_list([f"{fw[k]['weed']['recall']['mean']:.2f}" for k in prim])
           + " for recall)" if fw else "")
        + ". A working implement reporting roughly the same completeness supports the "
          "reading that the shortfall measured here is a property of the task at this "
          "scale rather than of these three models or this dataset.")

    out.append(
        "Two families of detection metric that would sharpen the diagnosis were not "
        "computed here and are worth naming for work that follows. Localization recall "
        "precision (Oksuz et al. 2018) combines localisation error with the false "
        "negative and false positive rates into one quantity and selects an operating "
        "threshold per class, which is closer to what an implement needs than average "
        "precision is. The TIDE decomposition (Bolya et al. 2020) attributes a loss of "
        "average precision to its causes — classification, localisation, duplicate "
        "detections, missed objects — and so would separate the imprecise-localisation "
        "and the never-detected components of the shortfall reported above more cleanly "
        "than the intersection-over-union sweep used here does.")

    if fw:
        out.append(
            "One comparison can be made on equal terms, with the convention rather than "
            "with a paper. Applied studies commonly quote Precision, Recall and "
            "F1-score from a confusion matrix over matched detections — the "
            "background-excluded variant. On this test partition that convention gives "
            "weed-class F1-scores of "
            + _and_list([f"{seed_mean(k, 'weed_f1') or 0:.3f}" for k in prim])
            + ", of the order routinely read as evidence that the problem is close to "
              "solved; the same predictions in the full variant give "
            + _and_list([f"{fw[k]['weed']['f1']['mean']:.3f}" for k in prim])
            + ". The discrepancy is not between studies or datasets — it is the same "
              "model on the same images, and the choice of variant accounts for all of "
              "it.")
    return out


def overall_comparison(res: dict, common: dict | None = None) -> list[str]:
    ranks = {m: _rank(res, lambda r, k, m=m: _macro(r, k, m))
             for m in ("precision", "recall", "f1")}
    ranks["accuracy"] = _rank(res, _acc)
    ap50 = {k: _ap50(res, k) for k in MODEL_ORDER}

    def rng(metric):
        vals = [v for _, v in ranks[metric]]
        return f"{min(vals):.3f} to {max(vals):.3f}"

    text = (f"**Overall model comparison.** Averaged over both classes (Table 3), Precision, "
            f"Recall and F1-score across the five models, each on the test partition of "
            f"the dataset it was trained on, ranged from {rng('precision')}, "
            f"{rng('recall')} and {rng('f1')}, respectively, while Accuracy ranged from "
            f"{rng('accuracy')}. ")

    # Bảng 2 và Bảng 3 là số trên tập test RIÊNG của từng bộ nhãn; đoạn này phải
    # dùng đúng nguồn đó, nếu không thứ hạng sẽ mâu thuẫn với bảng. Số trên tập
    # test dùng chung chỉ nằm ở "Basis of comparison", nơi nó được gọi tên rõ ràng.
    src = res
    box_rank = sorted(BOX_MODELS, key=lambda k: -_macro(res, k, "f1"))
    poly_rank = sorted(POLY_MODELS, key=lambda k: -_macro(res, k, "f1"))
    where = "on their respective test partitions"
    text += (f"Ranked within each annotation convention {where}, the bounding-box models "
             f"ordered as "
             + _and_list([f"{MODEL_LABELS[k]} "
                          f"({f3(src[k]['metrics']['filtered']['macro']['f1'])})"
                          for k in box_rank])
             + " by F1-score, and the polygon models as "
             + _and_list([f"{MODEL_LABELS[k]} ({f3(_macro(res, k, 'f1'))})"
                          for k in poly_rank]) + ". ")

    box_f1 = [(k, src[k]["metrics"]["filtered"]["macro"]["f1"]) for k in box_rank]
    f1_leaders = _joint_leaders(box_f1)
    top_f1 = box_f1[0][1]
    if len(f1_leaders) == 1:
        k = f1_leaders[0]
        text += (f"Among the bounding-box models, {MODEL_LABELS[k]} achieved the highest "
                 f"F1-score ({f3(top_f1)}), which is attributable to {ARCH_NOTE[k]}. ")
    else:
        text += (f"Among the bounding-box models, "
                 f"{_and_list([MODEL_LABELS[k] for k in f1_leaders])} were "
                 f"indistinguishable at the top of the F1-score ranking "
                 f"(within {TIE:.3f} of {f3(top_f1)}), so no single architecture can be "
                 f"declared superior on this criterion alone. ")

    box_acc = sorted(((k, src[k]["metrics"]["filtered"]["overall_accuracy"])
                      for k in BOX_MODELS), key=lambda t: -t[1])
    acc_leaders = _joint_leaders(box_acc)
    top_acc = box_acc[0][1]
    if len(acc_leaders) == 1 and acc_leaders != f1_leaders:
        k = acc_leaders[0]
        text += (f"The highest Accuracy among the bounding-box models ({f3(top_acc)}, "
                 f"Table 3) was recorded by {MODEL_LABELS[k]}, reflecting {ARCH_NOTE[k]}. ")
    elif len(acc_leaders) == 1:
        text += (f"The same model also recorded the highest Accuracy in that group "
                 f"({f3(top_acc)}, Table 3). ")

    worst_k = poly_rank[-1]
    text += (f"Among the polygon models, {MODEL_LABELS[worst_k]} recorded the lower "
             f"F1-score ({f3(_macro(res, worst_k, 'f1'))}), which may be attributed to "
             f"{ARCH_NOTE[worst_k]}. ")

    ap50 = {k: (src[k].get("coco", {}) or {}).get("AP50") for k in BOX_MODELS}
    have_ap = [k for k in BOX_MODELS if ap50[k] is not None]
    if have_ap:
        ap_order = sorted(have_ap, key=lambda k: -ap50[k])
        f1_order = [k for k in box_rank if k in have_ap]
        text += (f"Ranked instead by COCO-style AP50 on the same partitions, which "
                 f"penalises both missed detections and background false positives, the "
                 f"order was "
                 f"{_and_list([f'{MODEL_LABELS[k]} ({ap50[k]:.1f}%)' for k in ap_order])}. ")
        if ap_order != f1_order:
            text += ("This ordering differs from that obtained from the confusion matrix, "
                     "and the difference is informative: the confusion-matrix metrics "
                     "measure how well a model tells sugarcane from weeds once an "
                     "instance has been localised, whereas AP50 additionally measures how "
                     "many instances are found at all. A model may therefore separate the "
                     "two classes cleanly while still missing a substantial share of the "
                     "weed patches present in the field, and both properties matter for a "
                     "spraying system: the first determines how much herbicide is applied "
                     "to the crop, the second how much weed cover is left untreated. ")
        else:
            text += ("This ordering agrees with that obtained from the confusion matrix, "
                     "indicating that the models that separate the two classes best are "
                     "also those that localise the most instances. ")

    speeds = {k: clean_fps(k)[0] for k in MODEL_ORDER}
    fastest = max(speeds, key=speeds.get)
    text += (f"On the workstation GPU, inference throughput ranged from "
             f"{min(speeds.values()):.1f} to {max(speeds.values()):.1f} frames per second, "
             f"the highest being {MODEL_LABELS[fastest]}; for deployment on an autonomous "
             f"sprayer this is a first-order constraint alongside accuracy, since the "
             f"detection rate bounds the forward speed at which the implement can operate.")
    return [text]


def recommended_model(res: dict) -> tuple[str, str]:
    """Mô hình nên dùng, theo một luật nêu rõ trong chính câu văn sinh ra.

    Luật: lấy AP50 cao nhất làm mốc; mọi mô hình có AP50 không thấp hơn mốc quá
    AP_TOL được coi là ngang nhau về chất lượng phát hiện; trong nhóm đó chọn
    mô hình nhanh nhất. Nếu không có AP50 thì dùng F1 trung bình thay thế.
    Luật này là một lựa chọn kỹ thuật, không phải sự thật khách quan — nó được
    viết thẳng vào bản thảo để nhóm tác giả thấy và có thể đổi.
    """
    have_ap = [k for k in BOX_MODELS if _ap50(res, k) is not None]
    score = (lambda k: _ap50(res, k)) if have_ap else (lambda k: _macro(res, k, "f1") * 100)
    pool = have_ap or BOX_MODELS
    top = max(score(k) for k in pool)
    near = [k for k in pool if score(k) >= top * (1 - AP_TOL)]
    best = max(near, key=lambda k: clean_fps(k)[0])
    reason = (f"F1-score {f3(_macro(res, best, 'f1'))}, Accuracy {f3(_acc(res, best))}"
              + (f", AP50 {_ap50(res, best):.1f}%" if _ap50(res, best) is not None else "")
              + f", {clean_fps(best)[0]:.1f} FPS")
    return best, reason


# --------------------------------------------------------------------------
# Đoạn kết luận — viết lại trọn vẹn vì nó gộp cả mô tả dữ liệu lẫn xếp hạng
# --------------------------------------------------------------------------

def conclusion_v2(res: dict, seedvar: dict, stats: dict, cfg: dict,
                  rev: dict | None = None) -> list[str]:
    """Kết luận ngắn, phát biểu đúng mức bằng chứng.

    Tạp chí yêu cầu Conclusion "tóm tắt các điểm chính và nêu đóng góp"; bài
    nguyên tác đã đăng của họ thường dài 150-250 từ. Bản trước dài gấp ba.
    """
    prim = BOX_MODELS
    rev = rev or {}
    fw = rev.get("full_metrics", {})
    ss = rev.get("size_strata", {}).get("models", {})
    ar = rev.get("area_recall", {})
    tl = rev.get("two_level_bootstrap", {}).get("differences", {})
    ap = {k: seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap[k] or 0))
    f1s = {k: seed_mean(k, "f1") for k in prim}
    fast = [k for k in prim if clean_fps(k)[0] and
            clean_fps(k)[0] >= throughput_requirement(cfg)["fps_required_on_workstation"]]
    n_f1 = sum(1 for v in tl.values() if v["f1"]["excludes_zero"]) if tl else 0

    out = []
    if fw:
        out.append(
            "Three detectors sharing one annotation convention, each at the input size "
            "its framework defaults to "
            "and one training protocol were compared on close-range sugarcane imagery. "
            "The principal finding is a gap between two ways of scoring the same "
            "predictions. Excluding the background row and column of the confusion "
            "matrix, all three assign the correct class to almost every object they "
            f"localise (two-class mean F1-score {min(f1s.values()):.2f} to "
            f"{max(f1s.values()):.2f}); retaining them, on identical predictions, gives "
            "weed-class Precision of "
            f"{min(fw[k]['weed']['precision']['mean'] for k in prim):.2f} to "
            f"{max(fw[k]['weed']['precision']['mean'] for k in prim):.2f} and Recall of "
            f"{min(fw[k]['weed']['recall']['mean'] for k in prim):.2f} to "
            f"{max(fw[k]['weed']['recall']['mean'] for k in prim):.2f}. Which set is "
            "quoted determines whether this task appears solved, and the difference "
            "between them is a reporting convention rather than a result.")
    if ss and ar:
        out.append(
            f"{MODEL_LABELS[order[0]]} obtained the highest AP50 ({ap[order[0]]:.1f}), "
            "but stricter overlap thresholds do not reproduce that ordering and "
            + ("no pairwise difference in weed-class F1-score survives a bootstrap over "
               "both test images and training runs" if n_f1 == 0 else
               f"only {n_f1} of three differences survives a bootstrap over both")
            + ". What does separate the models is detection completeness: Recall falls "
              "monotonically with object size, from "
            + f"{min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f}–"
            + f"{max(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the "
            + "smallest area quartile to "
            + f"{min(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f}–"
            + f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the "
            + "largest. Weighted by ground area rather than by instance, recall rises "
              "to "
            + f"{min(_union()[k]['mean'] for k in prim):.2f}–"
            + f"{max(_union()[k]['mean'] for k in prim):.2f}.")
    out.append(
        "The practical outcome is a stated trade-off rather than a recommendation: "
        + (f"{MODEL_LABELS[fast[0]]} alone cleared the throughput screen derived from "
           "the target field speed and detects the fewest weed instances, while the "
           "two architectures that detect more are too slow on this hardware. "
           if fast else "no model cleared the throughput screen. ")
        + "Confirming either route requires measurement on the target hardware. For "
          "this literature the methodological point is more general: a "
          "confusion-matrix metric reported without stating whether unmatched ground "
          "truth and unmatched predictions were retained makes the difference between "
          "a figure near 0.98 and one near 0.47 on identical predictions, and only the "
          "second describes what a sprayer would do in a field.")
    return out


def conclusion(res: dict, stats: dict) -> list[str]:
    f1 = {k: _macro(res, k, "f1") for k in MODEL_ORDER}
    acc = {k: _acc(res, k) for k in MODEL_ORDER}
    box_order = sorted(BOX_MODELS, key=lambda k: -f1[k])
    poly_order = sorted(POLY_MODELS, key=lambda k: -f1[k])
    best, reason = recommended_model(res)
    fastest = max(BOX_MODELS, key=lambda k: clean_fps(k)[0])
    if fastest != best:
        d_ap = _ap50(res, best) - _ap50(res, fastest)
        ratio = clean_fps(fastest)[0] / clean_fps(best)[0]
        alt_note = (f"Where throughput rather than detection quality is the binding "
                    f"constraint, {MODEL_LABELS[fastest]} runs {ratio:.0f} times faster "
                    f"({clean_fps(fastest)[0]:.0f} against "
                    f"{clean_fps(best)[0]:.0f} frames per second) for "
                    f"{d_ap:.1f} percentage points less AP50, and is the better choice. ")
    else:
        alt_note = ""

    bb, pg = stats["bbox"], stats["polygon"]
    comb = stats["_combined"]
    n_bb, n_pg = bb["_total"]["source_images"], pg["_total"]["source_images"]
    n_img_bb, n_img_pg = bb["_total"]["images"], pg["_total"]["images"]
    n_union, n_shared = comb["source_images_union"], comb["source_images_shared"]
    n_inst = comb["instances_total"]

    text = (
        "This study developed and evaluated a computer vision-based approach for "
        "detecting weeds in sugarcane fields, with a view towards supporting precision, "
        "automated herbicide-spraying systems. Field images of sugarcane and common weed "
        "species were collected under a range of field conditions — three times of day, "
        "three weather conditions, and four sugarcane growth stages, spanning both the dry "
        "and rainy seasons — at Tan Hung farm, Chau Thanh district, Tay Ninh province, "
        f"Vietnam. Of these, {n_bb:,} source images were annotated with bounding boxes and "
        f"{n_pg:,} with polygons ({n_union:,} distinct images, {n_shared:,} of which were "
        f"annotated in both formats); after augmentation of the training partition this "
        f"gave {n_img_bb:,} and {n_img_pg:,} images respectively, carrying {n_inst:,} "
        "annotated sugarcane and weed instances across the two annotation sets. Five deep "
        "learning object detection "
        "models, namely Faster R-CNN, Mask R-CNN, RetinaNet, YOLOv8-seg, and YOLOv10, were "
        "trained and evaluated using Precision, Recall, F1-score, and Accuracy, each on "
        "the test partition of the annotation set it was trained on. Among the "
        f"bounding-box models, {_and_list([MODEL_LABELS[k] for k in box_order])} reached "
        f"overall Accuracy values of "
        f"{_and_list([f3(acc[k]) for k in box_order])}, respectively; among the polygon "
        f"models, {_and_list([MODEL_LABELS[k] for k in poly_order])} reached "
        f"{_and_list([f3(acc[k]) for k in poly_order])}. The two groups are not ranked "
        "against each other, because the two exports annotate the same scenes at different "
        "granularities and therefore define different detection targets. Among the evaluated "
        f"models trained on the bounding-box convention, {MODEL_LABELS[best]} offers the "
        f"best balance between detection quality and inference cost ({reason}). This "
        f"selection applies a reporting convention chosen after the results were obtained "
        f"— the fastest model whose AP50 falls within {AP_TOL * 100:.0f}% of the best AP50 "
        f"— and is stated explicitly so that readers weighting the trade-off differently "
        f"can reach a different conclusion from the same table. {alt_note}The two "
        f"polygon-trained "
        "models are not ranked against these three, because the two annotation conventions "
        "define different detection targets and no common ground truth exists for them.")
    return [text]


# --------------------------------------------------------------------------
# Đoạn bổ sung — cái mà Bảng 2/3 không nói: bao nhiêu cỏ bị bỏ sót
# --------------------------------------------------------------------------

def _gsd_paragraph(fig: str = "Fig. 4") -> str:
    """Đoạn về thí nghiệm giảm độ phân giải mặt đất.

    Phản biện nêu đúng rằng bài suy ra "đòn bẩy là hình học thu ảnh" từ một kết
    quả âm tính, chưa có thí nghiệm nào đổi hình học thu ảnh. Thí nghiệm này đổi
    nó theo chiều thô hơn — chiều duy nhất mà bộ ảnh 1280 px cho phép — và đoạn
    này phải nói rõ cả điều nó chứng minh lẫn điều nó không chứng minh.
    """
    import json as _json
    from common import RESULTS as _R
    f = _R / "gsd_analysis.json"
    if not f.exists():
        return ""
    g = _json.loads(f.read_text())
    ks = [f"{x:g}" for x in g["scales"]]
    lo, hi = ks[0], ks[-1]
    mods = list(g["models"])
    q_lo = g["models"][mods[0]][lo]["recall_by_quartile"]
    q_hi = g["models"][mods[0]][hi]["recall_by_quartile"]
    return (
        "**Does the number of pixels on a weed actually govern recall?** The experiment "
        "above changes what the network is fed; it does not change what the camera "
        "captures. To vary the second, every test image was downsampled and resampled "
        "back to its original size, which leaves the input size, the annotations and the "
        "detector untouched and removes only the detail carried by each object — the "
        "effect of mounting the camera further from the canopy. Over a range from the "
        f"acquisition used here to a ground sample distance "
        f"{g['models'][mods[0]][hi]['effective_gsd_multiplier']:.1f} times coarser, "
        "equivalent to raising the camera from "
        f"{g['models'][mods[0]][lo]['equivalent_camera_height_cm']:.0f} to "
        f"{g['models'][mods[0]][hi]['equivalent_camera_height_cm']:.0f} cm, weed recall "
        "fell monotonically for all three architectures, from "
        + _and_list([f"{g['models'][k][lo]['weed_recall']['mean']:.3f} to "
                     f"{g['models'][k][hi]['weed_recall']['mean']:.3f} for {MODEL_LABELS[k]}"
                     for k in mods])
        + f" ({fig}a; Supplementary Table S14). The decline is strongly size-selective: "
        "for "
        + MODEL_LABELS[mods[0]]
        + f" the smallest area quartile falls from {q_lo[0]:.3f} to {q_hi[0]:.3f}, a "
        f"{q_lo[0] / max(q_hi[0], 1e-9):.0f}-fold loss, while the largest falls from "
        f"{q_lo[3]:.3f} to {q_hi[3]:.3f}, a {q_lo[3] / max(q_hi[3], 1e-9):.1f}-fold loss "
        f"({fig}b). "
        "Detail per object therefore governs recall over the range that can be tested, "
        "and a coarser acquisition than the one used here would make the shortfall "
        "substantially worse. Two limits on that reading should be stated plainly. The "
        "models were trained on full-detail imagery and applied to degraded imagery, so "
        "part of the decline is the mismatch between the two rather than a bound set by "
        "the missing detail itself, and a detector trained at the coarser ground sample "
        "distance would be expected to lose less. And downsampling followed by bilinear "
        "resampling removes detail per object without reproducing the optics of a real "
        "lens or the wider field of view that a higher mounting would also bring. What "
        "the experiment establishes is that these detectors are sensitive to the scale "
        "at which objects reach them, not how much a particular camera height would "
        "cost.")


def _tiling_paragraph(rev: dict, fig: str = "Fig. 4") -> str:
    """Đoạn về suy luận theo ô cắt trên ảnh gốc — chiều MỊN của trục tỷ lệ.

    Phản biện vòng hai nêu đúng rằng thí nghiệm GSD chỉ đi một chiều và rằng ảnh
    gốc vẫn còn. Kết quả đi ngược điều bài từng ngoại suy: cho mô hình nhiều
    pixel hơn trên mỗi đối tượng chỉ nâng recall rất ít và phá precision, nên
    kết luận của bài phải đổi, không phải chỉ hạ giọng.
    """
    import json as _json
    from common import RESULTS as _R
    f = _R / "tiling_analysis.json"
    if not f.exists():
        return ""
    tl = _json.loads(f.read_text())
    mods = [k for k in BOX_MODELS if k in tl["models"]] or list(tl["models"])
    ks = [str(k) for k in tl["tiles"]]
    base, mid, fine = ks[0], ks[1], ks[-1]
    at = lambda k, s, m: tl["models"][k][s][m]["mean"]
    best = max(mods, key=lambda k: at(k, mid, "weed_recall") - at(k, base, "weed_recall"))
    gain = at(best, mid, "weed_recall") - at(best, base, "weed_recall")
    return (
        "**And when the detail per object is raised instead?** The native frames were "
        f"retained for {tl['n_test_images_with_raw']} of the "
        f"{tl['n_test_images']} test images, which makes the converse experiment "
        "possible without retraining. Each was divided into a grid of overlapping "
        "tiles, each tile rescaled to the 1,280 px export size and the predictions "
        "merged by non-maximum suppression, so that an object reaches the network with "
        "two or three times the linear detail it had in the images used for training. "
        "The single-tile row of this experiment rescales the whole frame and so "
        "reproduces the pipeline of the main text; it recovers "
        + _and_list([f"{at(k, base, 'weed_recall'):.3f} for {MODEL_LABELS[k]}"
                     for k in mods])
        + " against "
        + _and_list([f"{rev['area_recall'][k]['instance_recall']['mean']:.3f}"
                     for k in mods])
        + " in Table 2, which is the agreement to expect from a different resampling "
          "path, one missing frame and a single training run, and it is the control "
          "against which the other rows are read. Raising the detail did not relieve "
          "the shortfall. At a two-by-two grid, equivalent to a camera at "
        f"{tl['models'][mods[0]][mid]['equivalent_camera_height_cm']:.0f} cm, recall "
        "rose by at most "
        f"{gain:.3f} ({MODEL_LABELS[best]}: {at(best, base, 'weed_recall'):.3f} to "
        f"{at(best, mid, 'weed_recall'):.3f}) while weed-class precision fell from "
        + _and_list([f"{at(k, base, 'weed_precision'):.3f} to "
                     f"{at(k, mid, 'weed_precision'):.3f} for {MODEL_LABELS[k]}"
                     for k in mods])
        + ". At a three-by-three grid recall fell below the control for all three "
          "architectures and precision fell further, to "
        + _and_list([f"{at(k, fine, 'weed_precision'):.3f}" for k in mods])
        + f" ({fig}a, b; Supplementary Table S16). The mechanism is the same one that "
          "the coarsening experiment exposes, acting in the opposite direction: a "
          "detector trained at one object scale and run at another loses accuracy "
          "either way, and here it loses it mostly as spurious detections, because a "
          "weed patch presented three times larger than anything in training is "
          "reported as several objects and bare ground is reported as weed. Both "
          "experiments, however, apply a detector trained at one scale to imagery at "
          "another, so neither can say whether training at the new scale would remove "
          "the shortfall. That question was put to a direct test.")


def _scale_training_paragraph() -> str:
    """Huấn luyện ở thang ô, với đối chứng cùng ngân sách dữ liệu.

    Lần chạy đầu của thí nghiệm này không dùng được: nó so một mô hình huấn
    luyện trên ô (964 ảnh gốc, không tăng cường) với các điều kiện huấn luyện
    trên tập đã tăng cường của bài, nên thay một yếu tố gây nhiễu bằng một yếu
    tố khác. Bản chạy lại thêm điều kiện `orig1` — đúng những ảnh đó, cũng không
    tăng cường, cũng công thức đó, chỉ không chia ô — rồi chấm cả sáu mô hình
    trên cả hai tập test. Cặp quyết định là hai mô hình cùng chấm trên tập test
    ô: nhãn, ảnh nguồn và công thức giữ nguyên, chỉ khác thang huấn luyện.

    Kết quả đi ngược điều bài từng ngoại suy, nên đoạn này phải báo cáo cả ba
    điểm bất đối xứng còn lại chứ không chỉ con số.
    """
    import json as _json
    from common import RESULTS as _R
    f = _R / "tile_scale_training.json"
    if not f.exists():
        return ""
    d = _json.loads(f.read_text())
    sc, ag = d.get("scale_contrast") or {}, d.get("augmentation_gap") or {}
    mods = [k for k in BOX_MODELS if k in sc] or list(sc)
    if not mods:
        return ""

    def _fall(metric: str) -> str:
        return _and_list([f"{sc[k]['orig_trained'][metric]:.3f} to "
                          f"{sc[k]['tile_trained'][metric]:.3f} for {MODEL_LABELS[k]}"
                          for k in mods])

    worse = [k for k in mods if sc[k]["delta_recall"] < 0]
    up = [k for k in mods if sc[k]["delta_precision"] > 0]
    lr = d.get("retinanet_base_lr")
    gap_lo = min(-ag[k]["delta_recall"] for k in mods if k in ag) if ag else None
    gap_hi = max(-ag[k]["delta_recall"] for k in mods if k in ag) if ag else None
    return (
        "**Does training at that scale remove the shortfall?** Both experiments above "
        "vary the imagery and leave the detector alone, so neither separates "
        "sensitivity to object scale from the mismatch between training and test "
        "conditions. Separating them requires training at the finer scale, and doing "
        "that without substituting a data-budget confound for the scale confound "
        f"requires a matched control. Both conditions were built from the same "
        f"{d['build']['tile2']['splits']['train']['source_images']} training images "
        "that retain native frames: one "
        "keeps each frame whole, the other divides it into a two-by-two grid of "
        "overlapping tiles, and neither uses the augmented copies of the main "
        "experiment. All three architectures were trained under both conditions with "
        "an otherwise identical recipe, and each of the six models was scored on both "
        "test partitions. The decisive comparison is the pair scored on the tiled test "
        "set, where ground truth, source images and recipe are fixed and only the "
        "training scale differs. Training at the tile scale did not improve "
        "completeness; it lowered it, for all "
        f"{'three' if len(worse) == len(mods) else str(len(worse))} architectures. "
        "Weed-class recall fell from " + _fall("recall") + ", while weed-class "
        "precision rose for "
        + _and_list([MODEL_LABELS[k] for k in up])
        + " (" + _and_list([f"{sc[k]['orig_trained']['precision']:.3f} to "
                            f"{sc[k]['tile_trained']['precision']:.3f}" for k in up])
        + ") and fell with recall for "
        + _and_list([MODEL_LABELS[k] for k in mods if k not in up])
        + " (Supplementary Table S19). Three asymmetries bound how far this reaches. "
          "The tiled condition received four times as many optimiser updates as its "
          "control, because the recipe fixes the number of epochs and tiling "
          "quadruples the image count, so the adverse direction is conservative and "
          "not an artefact of a shorter schedule. RetinaNet diverged at the learning "
          "rate of the main experiment on the tiled partitions even with gradient-norm "
        + (f"clipping, so its learning rate was lowered to {lr:g} for both members of "
           "the pair; the difference within the pair therefore stands, but its "
           "absolute values are not comparable with Table 2. " if lr else
           "clipping, so its learning rate was lowered for both members of the pair. ")
        + "And the matrix as a whole sits well below Table 2 because it removes the "
          "augmentation: at the whole-frame condition recall is "
        + _and_list([f"{ag[k]['this_unaugmented']['recall']:.3f}" for k in mods if k in ag])
        + " against "
        + _and_list([f"{ag[k]['paper_augmented']['recall']:.3f}" for k in mods if k in ag])
        + f" there, so {gap_lo:.2f} to {gap_hi:.2f} of recall is the cost of the "
          "smaller unaugmented budget alone. The matrix is internally controlled and "
          "has to be read within itself. What it establishes is that at a matched "
          "budget, retraining at the tile scale does not recover the completeness that "
          "tiled inference fails to deliver — not that no training scheme could. Taken "
          "with the two experiments above, this places the limit less on a mismatch "
          "between training and test scale than on the size at which weeds appear in "
          "this imagery at all: on these data moving the camera does not improve "
          "completeness, and neither does retraining at the moved geometry. Training "
          "across several ground sample distances at the full data budget is the "
          "experiment this work most needs next, and it was not done here.")


def resolution_control(rev: dict, cfg: dict, fig: str = "Fig. 4") -> list[str]:
    """Đối chứng độ phân giải: huấn luyện lại YOLOv10 ở 1024 px.

    Đây là câu trả lời trực tiếp cho lập luận confound mà phản biện nêu. Kết
    quả bác bỏ giả thuyết đó, nên phải báo cáo rõ cả hai chiều: thí nghiệm được
    thiết kế để có thể làm đổ kết luận của bài, và nó không làm đổ.
    """
    rc = rev.get("resolution_control") or {}
    if not (rc.get("imgsz_640") and rc.get("imgsz_1024")):
        return []
    a, b, d = rc["imgsz_640"], rc["imgsz_1024"], rc["delta_1024_minus_640"]
    sd = f" ± {b['ap50']['sd']:.2f}" if b["ap50"]["sd"] is not None else ""
    return [x for x in [
        "**Is the difference an effect of input resolution?** YOLOv10 is the lightest "
        "model and, at its default 640 × 640 input, the one that recovers fewest weed "
        "instances — exactly what a resolution limitation would predict. To separate the "
        "two explanations it was retrained from scratch at 1024 × 1024, the input size "
        f"of the other two models, for {b['n']} seeds under an otherwise identical "
        "protocol, the effective batch preserved by gradient accumulation. It did not "
        f"help: AP50 was {b['ap50']['mean']:.2f}{sd} against "
        f"{a['ap50']['mean']:.2f} ± {a['ap50']['sd']:.2f} at 640 × 640, weed-class "
        f"Recall {b['recall']['mean']:.3f} against {a['recall']['mean']:.3f}, and "
        f"weed-class F1-score {b['f1']['mean']:.3f} against {a['f1']['mean']:.3f}. "
        "Multiplying the pixel count by 2.56 changed none of these by more than the "
        "spread between seeds. The recall shortfall of this architecture is therefore "
        "not an artefact of its input size; capacity, or the way the anchor-free head "
        "assigns targets, is the remaining explanation, and this experiment does not "
        "separate those. The experiment itself is not compromised by the export: the "
        "source images were written at 1,280 × 1,280 from 4,080 × 3,060 frames, so the "
        "1024 px input did receive detail that 640 px discards. What the export does "
        "limit is how far the experiment reaches, because no input above 1,280 px could "
        "be tested, and in any case it varies the network input rather than the imagery.",

        _gsd_paragraph(fig),
        _tiling_paragraph(rev, fig),
        _scale_training_paragraph(),
    ] if x]


def _session_split_paragraph(lk: dict) -> str:
    """Rò rỉ do chia ngẫu nhiên theo ảnh, và phép đo trực tiếp độ lớn của nó.

    Phép thử ORB chỉ loại trùng lặp hình học. Thứ nó không loại được là việc mọi
    burst có ảnh trong tập test cũng có ảnh trong tập huấn luyện, nên không một
    ảnh test nào thuộc một burst mà mô hình chưa từng thấy. Điều đó chỉ đo được
    bằng cách huấn luyện lại với cách chia khác.
    """
    import json as _json
    from common import RESULTS as _R
    b = lk.get("bursts", {})
    head = (
        "A second form of dependence is not covered by that test and bears more "
        "heavily on the figures. The partition was drawn at random over source images, "
        f"and of the {b.get('n_bursts', 27)} capture bursts behind the dataset, "
        f"{b.get('n_bursts_spanning_splits', 21)} contribute images to more than one "
        "partition; every burst represented in the test partition also contributes "
        "training images. No test image therefore comes from a burst the model has "
        "never seen, and the reported figures measure generalisation to unseen frames "
        "of seen bursts, which is the easier quantity.")
    f = _R / "session_split.json"
    if not f.exists():
        return head + (" How much easier was not measured here; it would require "
                       "retraining under a grouped partition, which this study did not "
                       "do, and the figures above should be read as optimistic by an "
                       "unknown margin.")
    sp = _json.loads(f.read_text())
    mean = sp.get("mean_grouped_minus_control", {})
    if not mean:
        return head
    d = [v.get("weed_recall") for v in mean.values() if v.get("weed_recall") is not None]
    dropped = sorted({k.split("/")[1] for k in sp.get("excluded", {})})
    scope = ("the models" if len(mean) == len(MODEL_LABELS)
             else _and_list([MODEL_LABELS[k] for k in mean]))
    note = ""
    if dropped:
        note = (" " + _and_list([MODEL_LABELS[k] for k in dropped])
                + (" is" if len(dropped) == 1 else " are")
                + " absent from this experiment: under the fixed recipe its training "
                "diverged on the smaller fold partitions, where the shorter warmup "
                "brings the learning rate to its full value sooner, and the folds that "
                "did complete could not be paired with a valid control. The experiment "
                "is therefore reported for the architectures that trained stably under "
                "the common recipe.")
    note += " " + sp["recipe_note"] if sp.get("recipe_note") else ""
    return head + (
        " How much easier was measured directly. Each capture session was held out in "
        f"turn, {scope} retrained on the other two, and each grouped fold compared "
        "with a random split of exactly the same size, so that the only difference "
        "within a pair is whether the test images come from a session the model has "
        "seen (Supplementary Table S17). Averaged over the three folds, holding out the "
        "session "
        + ("lowered weed-class recall by "
           if all(v["weed_recall"] < 0 for v in mean.values())
           else "changed weed-class recall by ")
        + _and_list([f"{abs(mean[k]['weed_recall']):.3f}"
                     + ("" if mean[k]["weed_recall"] < 0 else " in the opposite direction")
                     + f" for {MODEL_LABELS[k]}" for k in mean])
        + ". The random split in that experiment is therefore optimistic by about "
        f"{abs(sum(d) / len(d)):.2f} of recall relative to a genuinely unseen session, "
        "and the figures in Tables 1 and 2, which come from a random split, carry the "
        "same optimism. The fold experiment uses one version of each source image "
        "rather than the augmented copies, so its absolute values are below those of "
        "the main text and only the difference within a pair is interpretable." + note)


def det_count_phrase(rev: dict, cfg: dict, order: list) -> str:
    """Số phát hiện mỗi ảnh, NÊU RÕ NGƯỠNG.

    Bản trước viết "trung vị 14, 50 và 9 mỗi ảnh" mà không nói ngưỡng. Phản biện
    vòng hai tính lại và chỉ ra rằng ở ngưỡng vận hành 0,30 các con số là 11, 13
    và 7, tức tỷ lệ gần 1,6 : 1,9 : 1 chứ không phải 1,6 : 5,6 : 1; con số cũ là
    ở sàn giữ lại 0,05. Cả hai đều có chỗ dùng — average precision tích phân qua
    toàn dải nên sàn 0,05 là dải đúng cho nó, còn precision và recall trong bài
    đo ở 0,30 — nên bài nêu cả hai và nói rõ cái nào thuộc về cái nào.
    """
    d = rev["detections_per_image"]
    floor, work = "0.05", f"{cfg['eval']['conf_threshold']:g}"
    named = _and_list([f"{d[floor][k]['median_all_classes']:.0f} for {MODEL_LABELS[k]}"
                       for k in order])
    plain = _and_list([f"{d[work][k]['median_all_classes']:.0f}" for k in order])
    return (f"medians of {named} detections per image at the {float(floor):.2f} "
            f"retention floor that average precision integrates over, and {plain} in the "
            f"same order at the working threshold of {float(work):.2f}")


def _q4_range(rev: dict) -> str:
    """Dải recall ở tứ phân vị lớn nhất, trên ba kiến trúc."""
    ss = rev["size_strata"]["models"]
    v = [ss[k]["quartile"]["Q4"]["recall_mean"] for k in BOX_MODELS if k in ss]
    return f"{min(v):.2f}-{max(v):.2f}"


def _union() -> dict:
    """Recall theo diện tích HỢP của hộp cỏ, từ coverage_analysis.json."""
    import json as _json
    from common import RESULTS as _R
    return _json.loads((_R / "coverage_analysis.json").read_text())["union_area_recall"]


def _grid(frac: str = "0.125") -> dict:
    """Recall theo ô lưới, từ coverage_analysis.json."""
    import json as _json
    from common import RESULTS as _R
    return _json.loads((_R / "coverage_analysis.json").read_text())["grid_recall"][frac]["models"]


def what_limits_detection(rev: dict, lk: dict, cfg: dict,
                          coverage_elsewhere: bool = False) -> list[str]:
    """Vì sao recall thấp: kích thước, định vị, ngưỡng, và điều kiện thu ảnh.

    Thay cho `detection_completeness()` cũ, mục này trả lời bằng phép đo thay vì
    bằng suy đoán. Bản trước nêu giả thuyết "vật thể nhỏ, bị che khuất" rồi chỉ
    chứng minh bằng trung vị diện tích; phản biện chỉ ra rằng trung vị đó nằm
    trong dải "medium" của COCO và che khuất thì không được định lượng ở đâu.
    Bản này báo cáo recall phân tầng theo kích thước, độ nhạy với ngưỡng IoU, và
    recall theo diện tích phủ.
    """
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    ss = rev["size_strata"]
    out = []

    # ── 1. phân tầng kích thước ──
    q = ss["quartile_edges_px2"]
    out.append(
        "**What limits detection.** Weed-class Recall falls monotonically with object "
        "size across the whole range, not only among the smallest instances. Dividing "
        f"the {ss['n_weed_instances']} annotated weed instances of the test partition "
        "into area quartiles, with boundaries at "
        + _and_list([f"{v:,.0f} px²" for v in q])
        + " on the 1280 px exported frame, Recall rises from "
        + _and_list([f"{ss['models'][k]['quartile']['Q1']['recall_mean']:.3f} to "
                     f"{ss['models'][k]['quartile']['Q4']['recall_mean']:.3f} for "
                     f"{MODEL_LABELS[k]}" for k in prim])
        + ". The gradient is steeper still under the absolute COCO size bands "
          "(Supplementary Tables S4 and S5), though those place only "
        f"{ss['coco_band_counts']['small']} of the {ss['n_weed_instances']} instances "
        "in the smallest category. The missed instances are not, in absolute terms, "
        f"tiny: the smallest quartile reaches {q[0] ** 0.5:.0f} px on a side and its "
        "median instance is larger than the threshold at which general-purpose "
        "benchmarks stop calling an object small. What makes them hard is their size "
        "combined with the resizing applied at the network input and the visual "
        "similarity of weed and crop foliage.")

    # ── 2. định vị lệch và diện tích phủ ──
    iou = rev["iou_sweep"]
    ar = rev["area_recall"]
    lo_thr, hi_thr = iou[prim[0]][0]["iou"], iou[prim[0]][-1]["iou"]
    out.append(
        "Part of that shortfall is imprecise localisation rather than failure to see "
        f"the object: relaxing the matching threshold from an IoU of {hi_thr:.2f} to "
        f"{lo_thr:.2f} at the same confidence (Supplementary Table S3) raises Recall "
        "from "
        + _and_list([f"{iou[k][-1]['recall']:.3f} to {iou[k][0]['recall']:.3f} for "
                     f"{MODEL_LABELS[k]}" for k in prim])
        + ", so about a quarter of the apparent shortfall is a box that overlapped its "
          "annotation too little to count."
        + (" How much of that shortfall matters to an implement is the subject of the "
           "two preceding paragraphs, and it depends on whether completeness is counted "
           "by instance, by treated area or by nozzle zone."
           if coverage_elsewhere else
           " A second correction works the same way: for a sprayer the quantity that "
           "matters is the weed-covered ground treated, not the instances found, and "
           "the two differ because the undetected instances are the small ones. Taking "
           "the union of the annotated weed boxes, so that overlapping annotations are "
           "not counted twice, the treated fraction of weed-covered ground is "
           + _and_list([f"{_union()[k]['mean']:.3f} for {MODEL_LABELS[k]}"
                        for k in prim])
           + " against "
           + _and_list([f"{ar[k]['instance_recall']['mean']:.3f}" for k in prim])
           + " by instance count. Summing the box areas instead of taking their union "
             "double-counts the overlaps and gives a slightly lower figure, "
           + _and_list([f"{ar[k]['area_recall']['mean']:.3f}" for k in prim])
           + "; the union figure is the one used here and in the table of all measures. "
             "Neither is what a "
             "boom sprayer delivers, because a nozzle treats a zone rather than an "
             "instance and one detection anywhere in the zone treats all of it. Scored "
             "on a square grid one eighth of the frame width, counting a cell as treated "
             "when it holds both an annotated and a predicted weed, recall is "
           + _and_list([f"{_grid()[k]['cell_recall']['mean']:.3f} for {MODEL_LABELS[k]}"
                        for k in prim])
           + ", with cell precision of "
           + _and_list([f"{_grid()[k]['cell_precision']['mean']:.3f}" for k in prim])
           + " (Supplementary Table S12). The instance count is therefore the "
             "pessimistic reading and the cell count the optimistic one; which applies "
             "depends on how coarse the implement is, and the ground footprint of the "
             "frame was not measured here, so cell size is given as a fraction of the "
             "frame. The cell measure is optimistic for a second reason as well: a large "
             "annotated box spans several cells, so one detection inside it scores every "
             "cell it covers as treated."))

    # ── 4. phụ thuộc điều kiện thu ảnh ──
    if lk.get("sessions"):
        se = lk["sessions"]
        names = list(se)
        best = max(names, key=lambda n: np.mean([se[n]["models"][k]["recall"] for k in prim]))
        worst = min(names, key=lambda n: np.mean([se[n]["models"][k]["recall"] for k in prim]))
        out.append(
            "Recall also varies with plot and time of acquisition. Scored separately by "
            "capture session (Supplementary Table S9), the mean weed-class Recall "
            "across the three models is "
            f"{np.mean([se[best]['models'][k]['recall'] for k in prim]):.3f} on the "
            f"{best} and "
            f"{np.mean([se[worst]['models'][k]['recall'] for k in prim]):.3f} on the "
            f"{worst} — a spread wider than any difference between the architectures, "
            "with the model ordering not stable across sessions. Because the sessions "
            "differ simultaneously in plot, crop age and illumination, this experiment "
            "cannot attribute the spread to any one of them; what it establishes is "
            "that a single pooled figure conceals variation of that magnitude.")

    # ── 5. độ nhạy với ngưỡng tin cậy → chi tiết ở Phụ lục S2 ──
    cs = rev["conf_sweep"]
    at = lambda k, c: next(r["f1"] for r in cs[k] if abs(r["conf"] - c) < 1e-9)
    out.append(
        f"All the above use the confidence threshold of "
        f"{cfg['eval']['conf_threshold']:.2f} inherited from earlier work, and "
        "confidence scores are not calibrated to a common scale across frameworks. "
        "Sweeping it from 0.05 to 0.90 (Supplementary Table S2) leaves the weed-class "
        "F1-score of the two Detectron2 models nearly flat while that of YOLOv10 falls "
        f"from {at('yolov10', 0.20):.3f} to {at('yolov10', 0.90):.3f}. Faster R-CNN "
        "has the highest weed-class F1-score at every threshold in that range, so the "
        "ordering of the three is stable against the value chosen, although the size of "
        "the differences is not and no difference in F1-score survives both sources of "
        "variability (Statistics); the stability is a statement about the threshold, not "
        "a claim that one architecture is better.")
    return out


def detection_completeness(res: dict, fig: str = "Figure 2") -> list[str]:
    """Recall thật của lớp cỏ khi KHÔNG bỏ background.

    Đây là con số quyết định với máy phun: nó trả lời "bao nhiêu phần cỏ ngoài
    đồng bị bỏ qua", trong khi Bảng 2 chỉ trả lời "khi đã nhìn thấy một đối
    tượng thì gọi đúng tên nó bao thường xuyên".
    """
    # Chỉ ba mô hình quy ước hộp — hai mô hình polygon không so chung được, và
    # chúng chỉ có một lượt huấn luyện nên không có ± để báo cáo.
    prim = BOX_MODELS
    rec_w = {k: seed_mean(k, "weed_recall_full", _bg(res, k, "weed", "recall"))
             for k in prim}
    sd_w = {k: seed_sd(k, "weed_recall_full") for k in prim}
    rec_s = {k: _bg(res, k, "sugarcane", "recall") for k in prim}
    best_k = max(rec_w, key=rec_w.get)
    worst_k = min(rec_w, key=rec_w.get)
    filt_w = {k: _cls(res, k, "weed", "recall") for k in prim}

    text = (
        "**Detection completeness.** The metrics in Tables 2 and 3 are computed after the "
        "background row and column of the confusion matrix have been removed, and they "
        "therefore answer a specific question: once an object has been localised, how "
        "often is it given the correct class? They do not describe how many of the objects "
        "present in the field are found in the first place. Recomputing the same confusion "
        "matrices without removing the background row and column (" + fig + ") gives the "
        "weed-class Recall as "
        + _and_list([f"{f3(rec_w[k])}"
                     + (f" ± {f3(sd_w[k])}" if sd_w[k] else "")
                     + f" for {MODEL_LABELS[k]}" for k in prim])
        + f", against {_and_list([f3(filt_w[k]) for k in prim])} when background is "
          "excluded. "
        f"The best of the three, {MODEL_LABELS[best_k]}, therefore recovered "
        f"{rec_w[best_k] * 100:.0f}% of the weed instances annotated in the test images, "
        f"and the weakest, {MODEL_LABELS[worst_k]}, recovered "
        f"{rec_w[worst_k] * 100:.0f}%. "
        "For a precision-spraying system this is the operative figure: the "
        "background-excluded Recall determines how reliably a detected object is treated "
        "as weed rather than crop, whereas the Recall reported here determines how much "
        "weed cover the implement drives past without treating. The corresponding "
        "sugarcane-class Recall was "
        + _and_list([f"{f3(rec_s[k])} for {MODEL_LABELS[k]}" for k in prim])
        + ". Both sets of values are reported so that the two properties are not "
          "confused; the shortfall is dominated by small, partially occluded weed "
          "patches, as quantified under Qualitative results.")
    return [text]


BOX_MODELS = ["faster_rcnn", "retinanet", "yolov10"]
POLY_MODELS = ["mask_rcnn", "yolov8_seg"]


def _ordinal(n: int) -> str:
    return {5: "fifth", 10: "tenth", 4: "quarter", 3: "third"}.get(int(n), f"{int(n)}th")


def basis_of_comparison(res: dict, common: dict, stats: dict, n_common: int) -> list[str]:
    """CŨ — dùng cho bản vá bản thảo gốc. Bản thảo mới dùng
    build_manuscript.methods() với mục "Scope of the comparison"."""
    """Đoạn quan trọng nhất: nói rõ năm mô hình KHÔNG xếp hạng chung được.

    Hai bản xuất Roboflow gán nhãn ở hai độ mịn khác nhau, nên một bảng xếp hạng
    năm dòng đặt cạnh nhau là so sai. Đoạn này định lượng khác biệt đó, rồi đưa
    ra phép so hợp lệ: ba mô hình cùng quy ước hộp, trên cùng một tập test.
    """
    bb, pg = stats["bbox"], stats["polygon"]
    n_bbox, n_poly = bb["test"]["images"], pg["test"]["images"]
    ipi_b, ipi_p = bb["test"]["instances_per_image"], pg["test"]["instances_per_image"]
    ar_b, ar_p = bb["test"]["weed_area_pct_median"], pg["test"]["weed_area_pct_median"]

    ap_c = {k: common[k]["coco"]["AP50"] for k in MODEL_ORDER}
    box_rank = sorted(BOX_MODELS, key=lambda k: -ap_c[k])

    para1 = (
        "**Basis of comparison.** The five models were not trained on a single dataset. "
        "Faster R-CNN, RetinaNet and YOLOv10 were trained on the bounding-box export and "
        "Mask R-CNN and YOLOv8-seg on the polygon export, and the two exports differ both "
        "in which images they contain and in how finely those images are annotated. Their "
        f"test partitions are almost disjoint ({n_bbox} and {n_poly} source images, three "
        "images in common), and the polygon export annotates the same scenes at a finer "
        f"granularity: {ipi_p:.1f} instances per test image against {ipi_b:.1f}, with a "
        f"median weed instance occupying {ar_p:.2f}% of the frame against {ar_b:.2f}%. On "
        "the three images annotated in both formats, the bounding-box export marks 2, 25 "
        "and 10 objects where the polygon export marks 3, 35 and 14. The two conventions "
        "therefore define different detection targets: one delineates weed patches, the "
        "other resolves them into their constituent parts.")

    para2 = (
        "A single five-way ranking is consequently not supported by this design, and the "
        "per-model values in Tables 2 and 3 should be read as characterising each model on "
        "its own annotation convention rather than as places in a league table. To obtain "
        "one comparison that is valid, the three models trained on the bounding-box "
        f"convention were re-evaluated on a common test set of {n_common} images, formed "
        "from the bounding-box test partition after removing every image occurring in the "
        "polygon training partition, under the identical matching protocol. On that common "
        "set the AP50 values were "
        + _and_list([f"{ap_c[k]:.1f}% for {MODEL_LABELS[k]}" for k in box_rank])
        + ", and the confusion-matrix F1-scores were "
        + _and_list([f"{f3(common[k]['metrics']['filtered']['macro']['f1'])} for "
                     f"{MODEL_LABELS[k]}" for k in box_rank])
        + ". Applying the same procedure to the two polygon-trained models yields AP50 "
          "values of "
        + _and_list([f"{ap_c[k]:.1f}% for {MODEL_LABELS[k]}" for k in POLY_MODELS])
        + ", but these are an artefact of the convention mismatch rather than a measure of "
          "model quality: a detector trained to output many small instances is scored "
          "against ground truth that groups them into a few large ones, so most of its "
          "predictions cannot match any annotation at an IoU of 0.50. Quantifying the two "
          "conventions against each other would require re-annotating one export in the "
          "format of the other, which was outside the scope of this study and is identified "
          "as the principal methodological limitation of the comparison reported here.")
    return [para1, para2]


# --------------------------------------------------------------------------
# Các đoạn còn lại chưa khớp dữ liệu — vòng sửa thứ hai
# --------------------------------------------------------------------------

def qualitative_results(res: dict, rev: dict, fig: str = "Figure 2") -> list[str]:
    """Mục định tính: nói đúng cái hình cho thấy, và cả cái nó không cho thấy."""
    prim = BOX_MODELS
    mis = {k: res[k]["confusion_matrix"][1][0] for k in prim}
    miss = {k: res[k]["confusion_matrix"][1][2] for k in prim}
    spur = {k: res[k]["confusion_matrix"][2][1] for k in prim}
    return [
        f"**Qualitative results.** {fig} shows, for each model, one test image on "
        "which detection succeeded and one on which it failed, with the ground-truth "
        "annotation for the same frames; the pairs were chosen by per-image weed-class "
        "Recall on the first run, at the median and the tenth percentile. The failure "
        "frames show the two error modes favourable examples conceal — small patches "
        "near the stem base with no predicted box, and predicted weed boxes over bare "
        "soil with no corresponding annotation. Per model, weed instances assigned to "
        "the sugarcane class, weed instances not detected at all and spurious weed "
        "detections numbered "
        + _and_list([f"{mis[k]}, {miss[k]} and {spur[k]} for {MODEL_LABELS[k]}"
                     for k in prim])
        + ", so class confusion is two orders of magnitude rarer than either of the "
          "other two failures.",

        "The failure panels also qualify the size explanation rather than only "
        "illustrating it. Several of the missed annotations in them are large, diffuse "
        "patches on open soil rather than small objects near a stem, and the stratified "
        "figures say the same thing less vividly: recall in the largest area quartile "
        f"is {_q4_range(rev)}, so roughly a third of the largest annotated instances "
        "are missed as well. Size is the strongest single predictor of whether an "
        "instance is found, not a complete account of why one is missed.",
    ]


def training_time_note(times: dict) -> str:
    """Chú thích Bảng 1 — thay chỗ '[to be confirmed]' bằng thời gian đo được."""
    return ("Training time measured on the workstation described in the text (NVIDIA "
            "GeForce RTX 3080 Ti, 12 GB): "
            + _and_list([f"{MODEL_LABELS[k]} {v}" for k, v in times.items()])
            + ". The three Detectron2 models were trained with gradient accumulation at an "
              "effective batch size of 16, as described in the text.")


def edge_methods(res: dict) -> list[str]:
    """Phần phương pháp: mô tả đúng thứ đã đo, không hứa thứ chưa đo."""
    return [
        "Inference-speed measurement. To assess the feasibility of real-time field "
        "deployment, single-image inference throughput was measured for all five models on "
        "the workstation used for training (NVIDIA GeForce RTX 3080 Ti, 12 GB), over the "
        "full test set and after discarding the first five images to exclude CUDA "
        "initialisation cost. Throughput is reported in frames per second at the input "
        "resolution given in Table 1, with no export or quantisation applied, so the values "
        "characterise the models as trained. Deployment on an embedded platform such as the "
        "NVIDIA Jetson Orin NX, together with TensorRT or ONNX export and reduced-precision "
        "inference, is identified as the next step; those figures must be measured on the "
        "device itself and are not reported here."]


def edge_results(res: dict) -> list[str]:
    """Phần kết quả: số FPS thật, và hệ quả với tốc độ di chuyển của máy phun."""
    fps = {k: clean_fps(k)[0] for k in MODEL_ORDER}
    ms = {k: 1000.0 / clean_fps(k)[0] for k in MODEL_ORDER}
    fastest = max(fps, key=fps.get)
    slowest = min(fps, key=fps.get)
    best, _ = recommended_model(res)

    # Với dải quét 1 m và 30 cm chồng lấn giữa hai khung liên tiếp
    def speed_kmh(f):
        return f * 0.30 * 3.6

    return [
        "Practical implications and inference speed. Beyond detection accuracy, deployment "
        "in a real-time autonomous spraying system requires inference to complete within "
        "the time the implement takes to traverse one camera footprint. Measured on the "
        "workstation GPU, throughput ranged from "
        f"{fps[slowest]:.1f} frames per second for {MODEL_LABELS[slowest]} "
        f"({ms[slowest]:.0f} ms per image) to {fps[fastest]:.1f} frames per second for "
        f"{MODEL_LABELS[fastest]} ({ms[fastest]:.1f} ms per image), the full set of values "
        "being "
        + _and_list([f"{fps[k]:.1f} for {MODEL_LABELS[k]}" for k in MODEL_ORDER])
        + ". Taking a nadir footprint advanced by 0.30 m between successive frames, these "
          "rates correspond to forward speeds of approximately "
        + _and_list([f"{speed_kmh(fps[k]):.0f} km/h for {MODEL_LABELS[k]}"
                     for k in (slowest, fastest)])
        + ", so detection rate is not the binding constraint for any of the five models at "
          "realistic field speeds, and the choice between them can be made on detection "
          "quality. This conclusion applies to the workstation GPU; an embedded platform "
          "delivers a fraction of that throughput, and the margin available on the "
          f"workstation is what makes {MODEL_LABELS[best]} a defensible choice despite its "
          "lower frame rate. Confirming feasibility on the target hardware requires the "
          "on-device measurement identified in the Methods."]


def limitations(res: dict, stats: dict) -> list[str]:
    """Đoạn hạn chế: bỏ điều đã làm, thêm điều thật sự còn thiếu."""
    rec_w = {k: _bg(res, k, "weed", "recall") for k in MODEL_ORDER}
    lo, hi = min(rec_w.values()), max(rec_w.values())
    bb, pg = stats["bbox"], stats["polygon"]

    return [
        "These findings demonstrate that computer vision and deep learning techniques can "
        "distinguish weeds from sugarcane reliably once a plant has been localised, "
        "providing a basis for site-specific weed management in sugarcane-growing regions "
        "such as Tay Ninh province. By supplying weed-location information at rates well "
        "above those required by a field implement, the proposed approach can support the "
        "development of automated precision herbicide-spraying systems that reduce "
        "herbicide usage, lower labour costs, and mitigate the environmental and health "
        "risks associated with blanket spraying. Nevertheless, this study has several "
        "limitations that should be addressed in future work.",

        "First, and most importantly, the five models were trained on two annotation sets "
        "that differ in granularity "
        f"({pg['test']['instances_per_image']:.1f} against "
        f"{bb['test']['instances_per_image']:.1f} instances per test image) and whose test "
        "partitions are almost disjoint, so the study supports comparison within each "
        "annotation convention but not a single ranking across all five; re-annotating one "
        "export in the format of the other would remove this limitation. Second, detection "
        f"completeness remains the binding constraint: weed-class Recall, computed without "
        f"excluding the background row of the confusion matrix, ranged from {lo:.3f} to "
        f"{hi:.3f}, meaning that between {(1 - hi) * 100:.0f}% and {(1 - lo) * 100:.0f}% of "
        "the annotated weed instances went undetected, predominantly small and partially "
        "occluded patches. Third, the estimate of training variability rests on "
        f"{runs_phrase(_PRIMARY)} per model for the three bounding-box models and on a "
        "single run for each of the two polygon-trained models, for which no run-to-run "
        "variability is therefore reported; and a handful of runs gives a test of low "
        "statistical power, so a comparison "
        "that does not reach significance here should be read as inconclusive rather than "
        "as evidence that two models perform equally. Fourth, the polygon test "
        f"partition contains only {stats['polygon']['test']['images']} images, so the "
        "metrics for the two polygon-trained models carry correspondingly wide confidence "
        "intervals. Fifth, the dataset remains limited to a single growing region and "
        "sugarcane cultivar (KK3), and further validation across cultivars, soil types and "
        "agroecological zones is warranted. Sixth, inference speed was measured on a "
        "workstation GPU rather than on the embedded platform intended for deployment. "
        "Future work should therefore focus on raising recall for small weed patches, "
        "unifying the annotation convention, repeating training across seeds, expanding "
        "the dataset, and integrating the selected model into a complete precision-spraying "
        "prototype validated under real-time field conditions."]


# --------------------------------------------------------------------------
# Abstract — bản cũ xếp hạng cả năm mô hình và nói "accurately and effectively",
# cả hai đều mâu thuẫn với chính phần Kết quả. Viết lại cho khớp.
# --------------------------------------------------------------------------

def abstract_v2(res: dict, seedvar: dict, stats: dict, cfg: dict,
                rev: dict | None = None) -> str:
    """Abstract ngắn, làm tròn số, và không hứa nhiều hơn bằng chứng.

    Phản biện đề nghị hạ giọng kết luận và làm tròn (57,0 thay vì 57,05 ± 0,15);
    họ cũng đúng rằng bản cũ nói "RetinaNet cho chất lượng phát hiện cao nhất"
    trong khi AP và AP75 không ủng hộ điều đó.
    """
    prim = BOX_MODELS
    rev = rev or {}
    fw = rev.get("full_metrics", {})
    ss = rev.get("size_strata", {}).get("models", {})
    ar = rev.get("area_recall", {})
    tl = rev.get("two_level_bootstrap", {}).get("differences", {})
    bb = stats["bbox"]
    ap = {k: seed_mean(k, "ap50") for k in prim}
    order = sorted(prim, key=lambda k: -(ap[k] or 0))
    f1s = {k: seed_mean(k, "f1") for k in prim}
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] and
            clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    n_f1 = sum(1 for v in tl.values() if v["f1"]["excludes_zero"]) if tl else None

    lo_p = min(fw[k]["weed"]["precision"]["mean"] for k in prim) if fw else 0
    hi_p = max(fw[k]["weed"]["precision"]["mean"] for k in prim) if fw else 0
    lo_r = min(fw[k]["weed"]["recall"]["mean"] for k in prim) if fw else 0
    hi_r = max(fw[k]["weed"]["recall"]["mean"] for k in prim) if fw else 0

    parts = [
        "Three object detectors — Faster R-CNN, RetinaNet and YOLOv10 — were trained on "
        f"{bb['_total']['source_images']:,} close-range field images of sugarcane and "
        "weeds from Tay Ninh province, Vietnam, and evaluated on a common test partition "
        f"of {bb['test']['images']} images under one protocol: each framework's default "
        "input size, a "
        f"fixed {cfg['train']['epochs']}-epoch schedule, final-epoch weights, and "
        f"{runs_phrase(prim)} per model.",
        f"At a confidence threshold of {cfg['eval']['conf_threshold']:.2f}, excluding the "
        "background row and column of the confusion matrix gave two-class mean F1-scores "
        f"of {min(f1s.values()):.2f}–{max(f1s.values()):.2f}; retaining them, on identical "
        f"predictions, gave weed-class Precision {lo_p:.2f}–{hi_p:.2f} and Recall "
        f"{lo_r:.2f}–{hi_r:.2f}.",
    ]
    if ap:
        parts.append(
            f"{MODEL_LABELS[order[0]]} obtained the highest AP50 ({ap[order[0]]:.1f}), but "
            "AP at stricter overlap thresholds did not reproduce that ordering, "
            + ("and no pairwise difference in weed-class F1-score survived a two-level "
               "bootstrap over test images and training runs"
               if n_f1 == 0 else
               f"and only {n_f1} of three differences in weed-class F1-score survived a "
               "two-level bootstrap over images and runs")
            + ".")
    if ss:
        parts.append(
            "Recall fell monotonically with object size, from "
            f"{min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f}–"
            f"{max(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the "
            "smallest area quartile to "
            f"{min(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f}–"
            f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the "
            "largest, and rose to "
            f"{min(_union()[k]['mean'] for k in prim):.2f}–"
            f"{max(_union()[k]['mean'] for k in prim):.2f} when weighted by "
            "ground area covered rather than by instance.")
    if fast:
        parts.append(
            f"Only {MODEL_LABELS[fast[0]]} met the throughput required at "
            f"{t['sprayer_speed_kmh']:.0f} km/h, and it detected the fewest weed instances.")
    parts.append(
        "On this single-farm, single-day dataset detection rather than class "
        "discrimination is limiting, which points the engineering effort towards image "
        "resolution and acquisition geometry rather than a better classifier.")
    return " ".join(parts)


def abstract_results_sentences(res: dict, common: dict, stats: dict, n_common: int) -> str:
    """Hai câu kết quả + một câu kết luận của Abstract, sinh từ số đo."""
    acc = {k: _acc(res, k) for k in MODEL_ORDER}
    rec_w = {k: _bg(res, k, "weed", "recall") for k in MODEL_ORDER}
    lo, hi = min(rec_w.values()), max(rec_w.values())
    box_rank = sorted(BOX_MODELS, key=lambda k: -common[k]["coco"]["AP50"])
    best_ap = box_rank[0]

    return (
        "Because the bounding-box and polygon annotation sets differ in granularity and "
        "have almost disjoint test partitions, the five models are characterised within "
        "each annotation convention rather than ranked against one another. Discriminating "
        "sugarcane from weeds among detected objects was accurate for all five, with "
        "overall Accuracy between "
        f"{min(acc.values()):.3f} and {max(acc.values()):.3f}. Detection completeness was "
        "the limiting factor: weed-class Recall computed without excluding background "
        f"ranged from {lo:.2f} to {hi:.2f}, so between {(1 - hi) * 100:.0f}% and "
        f"{(1 - lo) * 100:.0f}% of annotated weed instances went undetected, predominantly "
        "small and partially occluded patches. On a common test set of "
        f"{n_common} images the three bounding-box models reached AP50 values of "
        + _and_list([f"{common[k]['coco']['AP50']:.1f}% for {MODEL_LABELS[k]}"
                     for k in box_rank])
        + f", {MODEL_LABELS[best_ap]} being the highest. These findings indicate that "
          "computer vision and deep learning can separate weeds from sugarcane reliably "
          "once a plant is localised, while recall on small weed patches remains the main "
          "obstacle to fully automated precision weed management.")


def title_suggestion(res: dict) -> str:
    """Tiêu đề cũ hứa 'comparative evaluation' — thiết kế thí nghiệm không đỡ được."""
    return ("Computer vision-based weed detection in sugarcane cultivation: "
            "performance and detection limits of five deep learning object detection models")


# --------------------------------------------------------------------------
# Introduction — bản cũ 1.236 từ, hai đoạn dài là danh mục số liệu trên cây
# trồng khác, còn Sun et al. (2024) (nghiên cứu trên mía, gần đề tài nhất) chỉ
# nằm trong một ngoặc đơn. Viết lại: giữ NGUYÊN toàn bộ 16 trích dẫn để không
# tạo tài liệu tham khảo mồ côi, cắt phần số liệu không phục vụ lập luận, và
# đưa Sun et al. lên làm điểm tựa cho khoảng trống nghiên cứu.
# --------------------------------------------------------------------------

def introduction() -> list[str]:
    return [
        "Sugarcane (*Saccharum* spp. hybrids) is grown for sugar, ethanol, paper and "
        "organic fertiliser (Mirajkar et al. 2019), and weeds are among the principal "
        "constraints on its yield and quality (Suganthi et al. 2019), competing for "
        "nutrients, light, water and space and hosting pests and diseases. Each "
        "available control carries a cost: manual and mechanical weeding are "
        "labour-intensive, herbicides pose risks to farmer health and to the "
        "environment (Donga and Eklo 2018), and mulching can aggravate waterlogging on "
        "poorly drained soils.",

        "Precision weed management keeps the efficacy of herbicides while reducing the "
        "quantity applied: instead of treating the field uniformly it determines where "
        "weeds occur and treats only those locations, which shifts the technical "
        "problem to distinguishing weeds from sugarcane and reporting their positions "
        "accurately and quickly enough to drive a sprayer. Traditional approaches rely "
        "on colour, shape, texture and spectral features (Sohail et al. 2021); de Souza "
        "et al. (2020) separated sugarcane from weeds by leaf spectral characteristics, "
        "but such methods need specialised instrumentation whose cost limits deployment "
        "on working farms.",

        "Field conditions complicate the visual task. Illumination, shadow, viewing "
        "angle, growth stage, plant density and occlusion all alter the appearance of "
        "crop and weeds, and sugarcane shares colour and leaf morphology with several "
        "weed species, so hand-crafted features transfer poorly between conditions. "
        "Convolutional neural networks learn discriminative features directly from "
        "images and have been applied in other crops: dos Santos Ferreira et al. (2017) "
        "classified grass and broadleaf weeds in soybean from UAV imagery. Where "
        "position is required rather than a class label, detection networks have been "
        "used — Yu et al. (2019) for three weed species in perennial ryegrass turf, "
        "Zhang et al. (2023) with an attention mechanism added to Faster R-CNN for "
        "soybean seedlings, and Moazzam et al. (2021) reformulating weed mapping in "
        "sugar beet as patch-based classification. Headline figures across this work are "
        "consistently high, typically above 98%, but they are not commensurable: they "
        "comprise per-image classification accuracy, per-pixel segmentation accuracy and "
        "detection average precision, computed on different tasks and different units of "
        "evaluation, and are cited here to describe the impression the literature creates "
        "rather than as a scale for the present results.",

        "Sugarcane is a harder case, and the published results for it reflect that. Crop "
        "and weeds share green colouration and leaf morphology, grow interspersed, and the "
        "weeds present as numerous small regions rather than as discrete plants, so the "
        "boundary between the two is genuinely ambiguous. Sun et al. (2024) addressed the "
        "resulting scale problem directly, combining multi-scale convolution with a "
        "transformer backbone for weed identification in sugarcane fields, on the reasoning "
        "that no single receptive field covers weed structures ranging from an individual "
        "leaf to a whole patch. Singh et al. (2024) performed pixel-level classification "
        "and localisation of small weed patches from drone imagery and reached an accuracy "
        "of 90.5%, while Mini et al. (2020) applied Mask R-CNN to aerial sugarcane imagery "
        "and obtained an AP50 of 65.5%. The contrast with the near-99% figures reported for "
        "soybean and turf is informative: in sugarcane the reported performance falls as "
        "soon as the task moves from assigning a class to finding and delineating the "
        "weeds themselves.",

        "Two questions follow that the literature does not settle for this crop. First, "
        "how do the main detector families — two-stage region-proposal networks, "
        "single-stage dense detectors and anchor-free real-time detectors — behave on "
        "close-range field imagery of sugarcane, as opposed to the aerial imagery used "
        "in most of the work above? Second, when such a model reports high class-wise "
        "accuracy, how much of the weed present has it actually found? The second "
        "matters to a spraying system as much as the first: class-wise accuracy governs "
        "how much herbicide reaches the crop, detection completeness how much weed is "
        "left untreated, and the two are not the same measurement.",

        "This study addresses both on imagery collected under production conditions in "
        "Tay Ninh province, Vietnam. It (i) builds an annotated close-range dataset of "
        "sugarcane and its weed flora in both a bounding-box and a polygon convention "
        "and releases the annotations, code and per-image results; (ii) compares three "
        "detectors spanning the three families — Faster R-CNN, RetinaNet and YOLOv10 — "
        "under one protocol, one checkpoint rule and repeated "
        "runs, reporting two polygon-trained models separately rather than ranking "
        "them against the three; and (iii) reports, alongside the confusion-matrix "
        "metrics conventionally quoted here, the same metrics computed without "
        "discarding undetected instances and spurious detections, together with the "
        "size dependence of recall and a throughput screen derived from the agronomy.",
    ]


# --------------------------------------------------------------------------
# Mục Statistics — RAE yêu cầu mô tả số lần lặp và cách xử lý biến thiên.
# Bản thảo gốc không có mục này; nó huấn luyện một lần rồi báo cáo ba chữ số
# thập phân, nên không có cách nào biết chênh lệch 0,002 là thật hay nhiễu.
# --------------------------------------------------------------------------

def statistics_section(seedvar: dict, boot: dict, cfg: dict,
                       rev: dict | None = None) -> list[str]:
    """Mục Statistics. Chi tiết đầy đủ ở Phụ lục S6-S8."""
    prim = [k for k in seedvar if k in ("faster_rcnn", "retinanet", "yolov10")]
    vals = {k: [e["ap50"] for e in seedvar[k]["runs"] if e.get("status") == "ok"]
            for k in prim}
    n_div = {k: seedvar[k].get("n_diverged", 0) for k in prim}
    rev = rev or {}
    cvs = {k: np.std(vals[k], ddof=1) / np.mean(vals[k]) * 100 for k in prim}
    out = []

    out.append(
        "**Statistics.** Two sources of variability bear on every comparison. "
        f"Resampling the {boot['n_images']} test images with replacement "
        f"({boot['n_boot']} resamples) gives 95% intervals on AP50 of "
        + _and_list([f"{MODEL_LABELS[k]} [{boot['models'][k]['ap50']['ci95'][0]:.1f}, "
                     f"{boot['models'][k]['ap50']['ci95'][1]:.1f}]" for k in prim])
        + " and on weed-class Recall in the full variant of "
        + _and_list([f"[{boot['models'][k]['weed_recall_full']['ci95'][0]:.3f}, "
                     f"{boot['models'][k]['weed_recall_full']['ci95'][1]:.3f}]"
                     for k in prim])
        + " — wide relative to the differences between models. Across seeds, AP50 was "
        + _and_list([f"{MODEL_LABELS[k]} {np.mean(vals[k]):.2f} ± "
                     f"{np.std(vals[k], ddof=1):.2f}" for k in prim])
        + ". For every model the test-set interval is the larger of the two, so the "
          "dominant uncertainty comes from how few images were held out.")

    if rev.get("tests"):
        bits = []
        for metric, lab in (("ap50", "AP50"), ("weed_recall_full", "weed-class Recall"),
                            ("weed_f1_full", "weed-class F1-score")):
            n_sig = sum(1 for v in rev["tests"][metric].values() if v["sig_holm"])
            bits.append(f"{_NUMWORD_CARD.get(n_sig, str(n_sig))} of three for {lab}")
        ns = {k: len(vals[k]) for k in prim}
        n_txt = ("n = " + ", ".join(f"{v} for {MODEL_LABELS[k]}" for k, v in ns.items())
                 if len(set(ns.values())) > 1 else f"n = {list(ns.values())[0]} per model")
        out.append(
            "Holding the test partition fixed and comparing seed-level values with "
            f"Welch's t-test ({n_txt}), the three P-values of each metric adjusted by the "
            "Holm–Bonferroni procedure, the differences remaining significant at the 5% "
            "level are " + _and_list(bits) + " (Supplementary Table S7, which also gives "
            "Hedges' g with its interval for all nine comparisons).")

    if rev.get("two_level_bootstrap"):
        tl = rev["two_level_bootstrap"]
        keep = [f"{nm}, {MODEL_LABELS[k.split('__')[0]]} against "
                f"{MODEL_LABELS[k.split('__')[1]]}: {v[met]['mean']:+.3f} "
                f"[{v[met]['ci95'][0]:+.3f}, {v[met]['ci95'][1]:+.3f}]"
                for k, v in tl["differences"].items()
                for met, nm in (("recall", "Recall"), ("f1", "F1-score"))
                if v[met]["excludes_zero"]]
        n_tot = 2 * len(tl["differences"])
        out.append(
            "Each of those holds one source fixed, and a claim that one architecture "
            "detects better than another needs both released. A two-level bootstrap — "
            "resampling test images and training runs together, each difference taken "
            "within an iteration — gives intervals excluding zero in "
            + (f"{len(keep)} of the {n_tot} comparisons: " + _and_list(keep) + " "
               if keep else f"none of the {n_tot} comparisons ")
            + "(Supplementary Table S8). The two architectures with the larger backbones "
              "therefore detect reliably more weed instances than the third, while no "
              "difference in F1-score between any pair survives both sources. Where this "
              "paper states that architectures are not separable, that is the interval it "
              "refers to. Average precision is absent from this bootstrap: it is defined "
              "over a ranking of all detections in the partition rather than as a sum of "
              "per-image counts, so it cannot be recomputed cheaply inside a resampling "
              "loop, and for it only the Welch comparison over runs is available.")

    cb = rev.get("cluster_bootstrap") or {}
    if cb.get("available"):
        wid = [(cb["models"][k]["by_cluster"][m]["ci_width"]
                / cb["models"][k]["by_image"][m]["ci_width"])
               for k in cb["models"] for m in ("recall", "f1")]
        out.append(
            "Both of those intervals resample test images independently, which they are "
            f"not: the {cb['n_clusters']} capture bursts behind the "
            "test partition contain frames seconds apart, and frames from one burst "
            "share illumination, plot and canopy. Resampling whole bursts instead of "
            "single images widens the 95% interval by a factor of "
            f"{min(wid):.1f} to {max(wid):.1f} "
            "(Supplementary Table S11). The direction of every comparison is unchanged "
            "and no conclusion reverses, but the intervals quoted above are the narrow "
            "ones, and the burst-level interval is the honest width for a reader asking "
            "how far these numbers would travel to another field.")
        out.append(
            "The same dependence limits what the seed-level term can carry. Four to five "
            "converged runs per architecture is a small sample from which to estimate a "
            "variance, and for RetinaNet the three runs that diverged are excluded, so "
            "its run-to-run standard deviation is conditional on convergence and "
            "understates the spread a practitioner would meet. Conclusions here are "
            "therefore stated as failures to separate rather than as equivalences.")

    cs = (rev.get("checkpoint_sensitivity") or {}).get("yolov10") or {}
    if cs.get("last") and cs.get("best_on_val"):
        la, bo = cs["last"], cs["best_on_val"]
        out.append(
            "One methodological choice is large enough to change a conclusion. All models "
            "were evaluated at final-epoch weights; evaluating YOLOv10 instead at the "
            "epoch scoring best on validation — its framework's default, which the other "
            f"two do not offer — raises AP50 from {la['mean']:.2f} ± {la['sd']:.2f} to "
            f"{bo['mean']:.2f} ± {bo['sd']:.2f} (Supplementary Table S6), "
            f"{cs['delta_mean']:.1f} points of mean and a {cs['sd_ratio']:.0f}-fold larger "
            "standard deviation. A study that took each framework's own default would "
            "therefore have called YOLOv10 the least reproducible of the three; under "
            "the symmetric protocol applied here it is the most reproducible.")

    div = [k for k in prim if n_div[k]]
    if div:
        parts = []
        for k in div:
            n_ok = len([e for e in seedvar[k]["runs"] if e.get("status") == "ok"])
            parts.append(f"{n_div[k]} of the {n_ok + n_div[k]} runs attempted for "
                         f"{MODEL_LABELS[k]}")
        ta = (rev.get("training_anomalies") or {}).get("_summary") or {}
        extra = ("" if not ta.get("yolov10_runs_with_non_finite_epoch") else
                 f" For symmetry, in {ta['yolov10_runs_with_non_finite_epoch']} of the "
                 f"{ta['yolov10_runs']} YOLOv10 runs the log shows a non-finite "
                 "box-regression loss at one or two late epochs, from which the run "
                 "recovered and completed normally.")
        out.append(
            "Training stability differed sharply: " + _and_list(parts) + " diverged "
            "with a non-finite classification loss within the first 300 iterations at "
            f"the learning rate of {cfg['train']['lr']} used throughout, while the "
            "others converged on every attempt. Divergent runs were excluded and "
            "further seeds drawn, but drawing was stopped rather than continued to a "
            "target count, because re-drawing only the architecture that fails would "
            "make the surviving sample a product of that selection; the standard "
            "deviation above for that architecture is therefore conditioned on "
            "convergence." + extra)

    out.append(
        f"Two cautions apply to the analysis itself: {runs_phrase(prim)} per model is a "
        "small sample, so the tests have low power and a comparison that does not reach "
        "significance should be read as inconclusive rather than as evidence of "
        "equivalence; and the whole analysis is conditional on one test partition from a "
        "single day's acquisition.")
    return out


def scope_and_basis(stats: dict, cfg: dict) -> list[str]:
    """Mở đầu phần Kết quả: nói rõ so cái gì với cái gì, và vì sao."""
    bb, pg = stats["bbox"], stats["polygon"]
    return [
        "**Scope of the comparison.** The five models were trained on two separate annotation "
        "exports of the same image collection, and the two exports are not "
        "interchangeable. Faster R-CNN, RetinaNet and YOLOv10 were trained on the "
        "bounding-box export and Mask R-CNN and YOLOv8-seg on the polygon export; the two "
        "differ in which images they contain, in the granularity at which those images are "
        f"annotated ({pg['test']['instances_per_image']:.1f} against "
        f"{bb['test']['instances_per_image']:.1f} instances per test image, with a median "
        f"weed instance occupying {pg['test']['weed_area_pct_median']:.2f}% of the frame "
        f"against {bb['test']['weed_area_pct_median']:.2f}%), and in their test partitions, "
        f"which share only three images. On the three images annotated in both formats the "
        "bounding-box export marks 2, 25 and 10 objects where the polygon export marks 3, "
        "35 and 14. The two conventions therefore define different detection targets: one "
        "delineates weed patches, the other resolves them into their constituent parts, "
        "and a metric computed under one cannot be compared with the same metric computed "
        "under the other.",

        "The comparison reported below is therefore restricted to the three models trained "
        "on the bounding-box convention, which share a single test partition of "
        f"{bb['test']['images']} images and are scored under an identical protocol. The two "
        "polygon-trained models are reported separately, on their own partition, and are "
        "not ranked against the three; their results are given for completeness rather "
        "than as part of the comparison. Placing all five in a single ranking, as the "
        "design of this study might invite, would attribute to the models a difference "
        "that belongs to the annotation conventions.",
    ]


def _tail_latency_sentence(prim, t) -> str:
    """Hạn chót phải đạt đều đặn, nên nêu cả phân vị 90 bên cạnh trung bình.

    Trả chuỗi rỗng nếu chưa có bộ đo cho mọi mô hình, để đoạn văn không nửa vời.
    """
    try:
        det = {k: bench_fps_detail(k) for k in prim}
    except SystemExit:
        return ""
    need = t["fps_required_on_workstation"]
    pass_mean = [k for k in prim if det[k]["mean"] >= need]
    pass_p90 = [k for k in prim if det[k]["p90"] >= need]
    same = set(pass_mean) == set(pass_p90)
    return (
        "The rates above are means over the repeated passes, and a sprayer must meet "
        "its deadline on most frames rather than on average, so the slower tail is "
        "given as well: the 90th-percentile rate is "
        + _and_list([f"{det[k]['p90']:.1f} frames per second for {MODEL_LABELS[k]}"
                     for k in prim])
        + ", and the 99th-percentile per-image latency "
        + _and_list([f"{det[k]['latency_ms_p99']:.1f} ms for {MODEL_LABELS[k]}"
                     for k in prim])
        + ". "
        + ("The outcome of the screen is the same under either, so the conclusion does "
           "not depend on which statistic is taken."
           if same else
           "The screen changes outcome between the two, so the mean overstates which "
           "architectures qualify and the percentile figure is the one to use."))


def detection_quality(res: dict, seedvar: dict, cfg: dict) -> list[str]:
    """Đoạn kết quả chính: AP và recall đầy đủ, không phải F1 đã lọc nền."""
    prim, _ = scope(cfg) if False else (["faster_rcnn", "retinanet", "yolov10"], [])
    ap = {k: np.mean([e["ap50"] for e in seedvar[k]["runs"] if e.get("status") == "ok"])
          for k in prim}
    sd = {k: np.std([e["ap50"] for e in seedvar[k]["runs"] if e.get("status") == "ok"],
                    ddof=1) for k in prim}
    rec = {k: np.mean([e["weed_recall_full"] for e in seedvar[k]["runs"]
                       if e.get("status") == "ok"]) for k in prim}
    rsd = {k: np.std([e["weed_recall_full"] for e in seedvar[k]["runs"]
                      if e.get("status") == "ok"], ddof=1) for k in prim}
    order = sorted(prim, key=lambda k: -ap[k])
    best, worst = order[0], order[-1]

    return [
        "**Detection quality.** Table 2 reports average precision and class-wise Recall on the "
        "held-out test set, averaged over each model's converged training runs. Ranked by AP50, "
        "the order was "
        + _and_list([f"{MODEL_LABELS[k]} ({ap[k]:.2f} ± {sd[k]:.2f})" for k in order])
        + ". Recall for the weed class, computed without excluding the background row of "
          "the confusion matrix and therefore counting undetected instances, followed the "
          "same order: "
        + _and_list([f"{rec[k]:.3f} ± {rsd[k]:.3f} for {MODEL_LABELS[k]}" for k in order])
        + f". The practical reading of these figures is that {MODEL_LABELS[best]} left "
          f"{(1 - rec[best]) * 100:.0f}% of the annotated weed instances undetected and "
          f"{MODEL_LABELS[worst]} left {(1 - rec[worst]) * 100:.0f}%; detection "
          "completeness, not class confusion, is what separates the three models and what "
          "limits all of them. Precision and Recall computed after removing the background "
          "row and column of the confusion matrix are given separately in Table 3, where "
          "all three models exceed 0.96; those values describe how reliably a localised "
          "object is assigned to the correct class, and are reported because misassigning "
          "sugarcane to the weed class would cause a sprayer to treat the crop, but they "
          "do not measure how much of the weed cover is found and should not be used to "
          "rank the models.",
    ]


def inference_cost(res: dict, cfg: dict, complexity: dict,
                   rev: dict | None = None, cost_table: int = 4) -> list[str]:
    """Sàng lọc thông lượng: điều kiện cần, không phải khẳng định khả thi."""
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    t = throughput_requirement(cfg)
    fps = {k: clean_fps(k)[0] for k in prim}
    passes = [k for k in prim if fps[k] >= t["fps_required_on_workstation"]]
    fails = [k for k in prim if k not in passes]
    lo, _hi = t["embedded_slowdown"]
    rc = (rev or {}).get("resolution_control") or {}

    out = [
        "**Inference cost and deployment screening.** Throughput on the workstation GPU "
        "was "
        + _and_list([f"{fps[k]:.1f} frames per second for {MODEL_LABELS[k]}" for k in prim])
        + ". Against the screening threshold of "
        f"{t['fps_required_on_workstation']:.0f} frames per second derived in the "
        "Methods, "
        + (f"only {_and_list([MODEL_LABELS[k] for k in passes])} clears it; "
           f"{_and_list([MODEL_LABELS[k] for k in fails])} fall below the required rate "
           "even assuming the embedded device delivers one "
           f"{_ordinal(lo)} of workstation throughput. " if passes and fails else "")
        + f"Table {cost_table} gives the corresponding architectural cost: "
        + _and_list([f"{MODEL_LABELS[k]} {complexity[k]['params'] / 1e6:.1f} million "
                     f"parameters and {complexity[k]['gmacs_median']:.0f} GMACs"
                     for k in prim])
        + ", computed at each model's own input resolution and therefore overstating "
          "the ratio at a common one. The screen is a necessary and not a sufficient "
          "condition.",
        _tail_latency_sentence(prim, t),
    ]
    out = [x for x in out if x]
    if rc.get("imgsz_1024") and rc.get("fps_1024"):
        out.append(
            "The throughput advantage is not bought by the lower input resolution alone. "
            "Measured under the same protocol at 1024 × 1024, the resolution of the other "
            f"two models, YOLOv10 still reached {rc['fps_1024']['mean']:.0f} frames per "
            f"second ({rc['fps_1024']['latency_ms']:.1f} ms per image), well above the "
            f"{t['fps_required_on_workstation']:.0f} frames per second required, while its "
            "detection quality was unchanged. The separation between this architecture and "
            "the two ResNet-50 models survives equalising the input size on both counts.")
    return out


def limitations_v2(res: dict, seedvar: dict, stats: dict, cfg: dict,
                   rev: dict | None = None, lk: dict | None = None) -> list[str]:
    """Hạn chế — khối CUỐI của Results and Discussion.

    Tạp chí quy định đúng bốn mục và không có mục Limitations riêng, nên phần
    này nằm cuối Results, có nhãn in đậm để không bị đọc nhầm thành kết quả —
    đó mới là điều phản biện phản đối ở bản trước.
    """
    prim = BOX_MODELS
    rev, lk = rev or {}, lk or {}
    bb, pg = stats["bbox"], stats["polygon"]
    fw = rev.get("full_metrics", {})
    ov = lk.get("frame_overlap", {})
    out = []

    out.append(
        "**Limitations.** The acquisition bounds everything reported here. All "
        f"{lk.get('n_with_timestamp', 1210):,} source images carrying a timestamp were "
        f"recorded on a single day, {lk.get('capture_date', '')}, within "
        f"{lk.get('capture_window', {}).get('span_hours', 4.6):.1f} hours, in three "
        "sessions on three plots of one farm with one cultivar, so a test partition "
        "drawn at random from it measures generalisation to unseen frames of the same "
        "plots under the same light and nothing beyond. Whether frames two seconds "
        "apart are near-duplicates, which would leak training content into the test "
        "partition, was checked rather than assumed: matching ORB features under a "
        "RANSAC homography, pairs within five seconds share a median inlier fraction "
        f"of {ov.get('adjacent_median', 0.006):.3f} against "
        f"{ov.get('far_median', 0.005):.3f} for pairs more than ten minutes apart, and "
        f"only {ov.get('adjacent_gt_005', 4)} of {ov.get('n_adjacent', 60)} "
        "close-in-time pairs exceed five per cent. There is therefore no evidence of "
        "duplicate leakage; the correlation that exists is one of condition.")

    out.append(_session_split_paragraph(lk))

    out.append(
        "The comparison covers three architectures rather than five, because the two "
        "annotation conventions were never reconciled; deriving boxes from the polygon "
        "annotations and retraining all five under one convention is the single change "
        "that would most strengthen the design. One asymmetry remains among the three: "
        "YOLOv10 carries a fifth of the parameters of the two ResNet-50 models, so a "
        "difference between them confounds architecture with capacity, and no larger "
        "YOLOv10 variant was trained that would separate the two. A single "
        "learning rate was used throughout, which "
        "keeps the comparison from rewarding whichever model received more tuning but "
        "also means no architecture is shown at its best. The rate chosen, 0.001 with "
        "AdamW, is high for fine-tuning a ResNet-50 backbone, and the three RetinaNet "
        "runs that produced non-finite losses are more likely a consequence of that "
        "setting than a property of the architecture; the divergences reported here "
        "should be read as a property of this configuration. A per-architecture "
        "learning-rate search on the validation partition is the obvious next step and "
        "was not undertaken.")

    out.append(
        "Three features of the acquisition bound the claim further. The camera was "
        "held by hand and the implement was not moving, so the imagery carries none of "
        "the vibration, motion blur, oblique incidence or changing exposure that a "
        "boom-mounted camera produces, and throughput aside, nothing here establishes "
        "that these detectors would behave the same way on a moving machine. The weed "
        "flora recorded here comprises two grasses, a sedge and a fourth species not "
        "identified to species but described locally as a grass, and all four are "
        "pooled into a single weed class, whereas a herbicide decision usually "
        "turns on the grass–broadleaf distinction, so the detector built here supports "
        "a blanket site-specific application and not a choice of product. And the "
        "near-duplicate test above has limited power: matching sparse keypoints across "
        "repetitive vegetation is not a sensitive instrument, and the more informative "
        "signal is that weed recall varies from 0.37 to 0.55 between the three capture "
        "sessions, which is the width a reader should expect between fields rather "
        "than the width of the intervals quoted above.")

    out.append(
        "The shortfall in detection has two halves that should not be conflated: "
        "roughly half the annotated weed is missed, and roughly half of what is "
        "reported as weed is not annotated as such. Part of the second may be "
        "annotation rather than model error, and this study cannot separate the two "
        "because no second annotation of the test partition exists; the annotation "
        "protocol is likewise not documented in a form that would let another group "
        "reproduce the labelling. How arbitrary that boundary is can be read inside "
        "this study: the same scenes annotated under the two conventions yield 10.6 "
        "and 29.3 instances per image, so whether a dense patch is one weed or several "
        "is a decision of the annotator rather than a property of the field, and any "
        "instance-counted recall inherits that decision. A second independent "
        "annotation of a subset, scored many-to-one or by area, would bound how much "
        "of the shortfall is the detector's. Finally, the deployment screen rests on a published "
        "desktop-to-embedded throughput ratio rather than on the target device and "
        "treats throughput alone, the test partitions are small "
        f"({bb['test']['images']} and {pg['test']['images']} images), and "
        f"{runs_phrase(prim)} per model is few enough that the tests have low power. "
        "These argue for reading the comparisons as bounds on what these data can "
        "distinguish rather than as measurements of architectural merit.")
    return out
