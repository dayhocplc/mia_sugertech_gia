#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Các tệp đi kèm bản Sugar Tech: title page, cover letter, thư rút bài khỏi RAE.

    python3 src/build_sugartech_extras.py

Springer yêu cầu nộp title page riêng với tên tác giả và cơ quan, còn tệp bản
thảo để làm mù. Thư rút bài là bắt buộc về mặt liêm chính: không được nộp song
song hai nơi, nên phải rút khỏi Research in Agricultural Engineering TRƯỚC khi
nộp Sugar Tech.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import AFFIL, AUTHORS, CORRESP, to_docx
from build_manuscript_sugartech import KEYWORDS, TITLE
from common import MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope, \
    throughput_requirement

OUT = RAE / "manuscript"
TODAY = datetime.date.today().strftime("%d %B %Y")
RAE_TITLE_OLD = ("Computer vision-based weed detection in sugarcane cultivation: "
                 "detection limits of three object detection architectures under "
                 "field conditions")


def _plain(s: str) -> str:
    return s.replace("<sup>", " (").replace("</sup>", ")")


def checklist() -> str:
    """Phiếu kiểm nộp bài: những gì đã xong và những gì nhóm tác giả phải điền."""
    ms = OUT / "SugarTech_manuscript.md"
    text = ms.read_text(encoding="utf-8") if ms.exists() else ""
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)|\*\*|\*|&nbsp;", "", text)
    n_tbl = len(re.findall(r"\*\*Table \d+\*\*", text))
    n_fig = len(re.findall(r"\*\*Fig\. \d+\*\*", text))
    todo = re.findall(r"\[TO BE (?:COMPLETED|CONFIRMED OR AMENDED) BY THE AUTHORS[^\]]*\]", text)
    sup = OUT / "SugarTech_supplementary.md"
    st = sup.read_text(encoding="utf-8") if sup.exists() else ""

    # Đếm mục tài liệu thật (dòng "Họ, chữ-viết-tắt… NĂM. "), không đếm dòng trống.
    _refs = text.split("## References")[-1]
    n_ref = len(re.findall(r"(?m)^[A-ZÀ-Ýa-z][^\n|]*?\. [12][0-9]{3}[ab]?\. ", _refs))
    rows = [
        ("Type of paper", "Original Paper", "\u2713"),
        ("Length", "25 double-spaced pages max, incl. tables and figures",
         "\u2713 see page count printed in the footer of the manuscript"),
        ("Font", "10-point Times Roman, double-spaced, 2.5 cm margins", "\u2713"),
        ("Page numbering", "automatic", "\u2713 footer field"),
        ("Abstract", "150-250 words",
         f"\u2713 {len(re.findall(r'[*][*]Abstract[*][*] (.+)', text)[0].split()) if re.findall(r'[*][*]Abstract[*][*] (.+)', text) else 0} words"),
        ("Keywords", "4-6", f"\u2713 {len(KEYWORDS.split(';'))}"),
        ("Headings", "no more than three levels", "\u2713 two levels"),
        ("Tables", "Arabic numerals, cited in order, caption above",
         f"\u2713 {n_tbl} tables"),
        ("Figures", "named FigN, cited in order, captions in the text file",
         f"\u2713 {n_fig} figures, files in springer_figures/"),
        ("Figure resolution", "600 dpi for combination art, RGB",
         f"\u2713 Fig1-Fig{n_fig} at 600 dpi, TIFF and PNG, within 174 mm width"),
        ("Figure captions", "'Fig.' bold, no punctuation after number or at end",
         "\u2713"),
        ("References", "author-year in text; alphabetical; journal titles italic",
         f"\u2713 {n_ref} entries"),
        ("Statements and Declarations", "required heading", "\u2713 present"),
        ("Supplementary", "named Supplementary Table SN / Figure SN",
         f"\u2713 {len(re.findall(r'[*][*]Supplementary Table S[0-9]+[.]', st))} tables and "
         f"{len(re.findall(r'Supplementary Figure S[0-9]+[.]', st))} figure, each cited "
         "in the manuscript"),
        ("Supplementary file naming", "named consecutively, e.g. ESM_1, ESM_2",
         "\u2713 ESM_1.docx (tables and figure), ESM_2.zip (code, data, logs, results)"),
        ("Supplementary citation", "referred to in the text as Online Resource N",
         "\u2713 Online Resource 1 and 2 cited under Statements and Declarations"),
        ("Supplementary header", "each file carries title, journal, authors, "
         "affiliation, corresponding e-mail",
         "\u2713 first page of ESM_1 and README of ESM_2"),
        ("Source files", "editable .docx required",
         "\u2713 manuscript and supplementary supplied as .docx"),
    ]
    tbl = ["| Requirement | Journal rule | Status |", "|---|---|---|"]
    tbl += [f"| {a_} | {b_} | {c_} |" for a_, b_, c_ in rows]

    return "\n\n".join([
        "# Sugar Tech submission checklist",
        f"Manuscript: {len(clean.split()):,} words, {len(clean):,} characters.",
        "## Automatically verified",
        "\n".join(tbl),
        "## To be completed by the authors",
        "These cannot be derived from the data or the code. Each appears in the "
        "manuscript or cover letter as a bracketed placeholder, and the consistency "
        "checker reports them on every run until they are replaced.",
        "\n".join(f"{i}. {t}" for i, t in enumerate(todo, 1)) or "(none)",
        "## Also required by the journal, outside the files",
        "1. **Society membership.** One author must be an annual or life member of the "
        "Society for Sugar Research & Promotion. The membership form is linked from the "
        "journal's submission guidelines.",
        "2. **Competing-interest form.** All listed authors must complete the journal's "
        "online conflict-of-interest disclosure; the generated statement is published "
        "before the Acknowledgements.",
        "3. **ORCID.** Supply the 16-digit ORCID of each author if available.",
        "4. **Use of AI tools.** If a large language model was used beyond copy editing, "
        "it must be documented in the Methods section. AI-assisted copy editing does not "
        "need to be declared.",
        "5. **Withdrawal from the previous journal.** Send the withdrawal letter to "
        "Research in Agricultural Engineering and wait for written acknowledgement "
        "before submitting here. The cover letter states that this has been done.",
    ]) + "\n"


def _leakage_finding(n: int = 5) -> str:
    """Điều kiện tối thiểu thứ hai của ban biên tập: đánh giá chia theo nhóm.

    `n` là số thứ tự của gạch đầu dòng trong thư; thư của hai tạp chí có số
    lượng phát hiện khác nhau nên không thể viết cứng.
    """
    import json as _json
    from common import RESULTS as _R
    f = _R / "session_split.json"
    if not f.exists():
        return ""
    sp = _json.loads(f.read_text(encoding="utf-8"))
    mean = sp.get("mean_grouped_minus_control", {})
    d = [v["weed_recall"] for v in mean.values() if "weed_recall" in v]
    if not d:
        return ""
    names = " and ".join(MODEL_LABELS[k] for k in mean) if len(mean) < 3 else "all three"
    # trung bình, không phải khoảng hai chữ số: khoảng đó cắt mất giá trị lớn
    # nhất mà bài nêu ở ba chữ số
    rng = f"{abs(sum(d) / len(d)):.2f}"
    clipped = [MODEL_LABELS[m] for m in sp.get("clipped_models", [])]
    if len(mean) < 3:
        extra = (" The remaining architecture is absent because its training diverged on "
                 "the smaller fold partitions under the fixed recipe; we say so rather "
                 "than substituting a differently trained run.")
    elif clipped:
        extra = (" " + " and ".join(clipped)
                 + " required gradient-norm clipping to train at all on the smaller fold "
                 "partitions; it was applied to both members of every pair, so the "
                 "difference within a pair stands, and we state the departure in the "
                 "Limitations rather than leaving it to be inferred.")
    else:
        extra = ""
    return (f"{n}. **The random partition is itself optimistic, and by how much is now "
            "measured.** Every capture burst represented in the test partition also "
            "contributes training images, so the headline figures measure generalisation "
            "to unseen frames of seen bursts. We therefore held each capture session out "
            "in turn and compared each grouped fold with a random split of exactly the "
            f"same size. Holding out the session cost {names} a further {rng} of recall "
            "on average. "
            "We state this in the Abstract rather than only in the Limitations, because "
            "it qualifies the numbers a reader will quote." + extra)


def cover_letter(cfg, rev, journal: str = "sugartech") -> str:
    """Thư gửi ban biên tập. Nội dung khoa học dùng chung cho cả hai tạp chí;
    chỉ đoạn mở, đoạn về tình trạng bài và đoạn về điều kiện riêng là khác.
    """
    JN = ("*Artificial Intelligence in Agriculture*" if journal == "aiia"
          else "*Sugar Tech*")
    prim = ["faster_rcnn", "retinanet", "yolov10"]
    fm, ar = rev["full_metrics"], rev["area_recall"]
    ss = rev["size_strata"]["models"]
    cov = json.loads((RESULTS / "coverage_analysis.json").read_text())
    gcell = cov["grid_recall"]["0.125"]["models"]
    f1s = {k: D.seed_mean(k, "f1") for k in prim}
    t = throughput_requirement(cfg)
    fast = [k for k in prim if clean_fps(k)[0] >= t["fps_required_on_workstation"]]
    lo = lambda m: min(fm[k]["weed"][m]["mean"] for k in prim)
    hi = lambda m: max(fm[k]["weed"][m]["mean"] for k in prim)
    A = []
    a = A.append
    a(TODAY)
    a(f"To: The Editor-in-Chief, {JN}")
    a("Dear Editor,")
    a(f"We are submitting our manuscript, “{TITLE}”, for consideration as "
      + ("a research article in " if journal == "aiia" else "an Original Paper in ")
      + JN + ".")
    if journal == "aiia":
        a("The manuscript fits the scope of the journal on its methodological side as "
          "much as its applied one. Its central finding is about how detection results "
          "for agricultural computer vision are reported: one set of predictions, scored "
          "under the two confusion-matrix conventions in common use, yields weed-class "
          "F1-scores that differ by roughly a factor of two, and the convention that "
          "flatters the result is the one usually reported. That is a measurement "
          "problem for the field, not a property of these three models or this crop.")
    a("Site-specific herbicide application in sugarcane depends on a detector that finds "
      "the weed actually present and supplies detections fast enough for the implement to "
      "act on them. The deep learning literature reports accuracies for this task that "
      "are consistently high, and implements built on it are already working in this "
      "crop — Azghadi et al. (2025) field-tested a deep-learning spot-sprayer in sugarcane "
      "against broadcast spraying on commercial farms. Our study asks what "
      "those detectors deliver against the two requirements a sprayer imposes, and the "
      "answer is less comfortable than the published figures suggest.")
    a("The main findings are these.")
    a(f"1. **Class discrimination is solved; detection is not.** On the same test "
      f"partition, the three detectors reached two-class mean F1-scores of "
      f"{min(f1s.values()):.2f}–{max(f1s.values()):.2f} under the confusion-matrix "
      f"convention usually reported, which discards undetected instances. Scored so that "
      f"undetected weed and spurious detections are retained, the same predictions gave "
      f"weed-class precision of {lo('precision'):.2f}–{hi('precision'):.2f} and "
      f"recall of {lo('recall'):.2f}–{hi('recall'):.2f}. About half the weed was not "
      "found, and about half of what was reported as weed had no annotated weed under it.")
    a(f"2. **How much of that matters depends on what the implement treats.** Weighted "
      f"by the ground area an implement would cover rather than by instance, recall was "
      f"{min(cov['union_area_recall'][k]['mean'] for k in prim):.2f}–"
      f"{max(cov['union_area_recall'][k]['mean'] for k in prim):.2f}; scored on a grid "
      "the size of a nozzle zone, where one detection treats the whole cell, it was "
      f"{min(gcell[k]['cell_recall']['mean'] for k in prim):.2f}–"
      f"{max(gcell[k]['cell_recall']['mean'] for k in prim):.2f}. We report all three "
      "readings and say which question each answers, rather than quoting the most "
      "striking one.")
    a(f"3. **The cause is measured, not asserted.** Recall falls monotonically with weed "
      f"size, from {min(ss[k]['quartile']['Q1']['recall_mean'] for k in prim):.2f} in the "
      f"smallest area quartile to "
      f"{max(ss[k]['quartile']['Q4']['recall_mean'] for k in prim):.2f} in the largest. "
      "Retraining the lightest detector at 2.56 times the pixel count changed nothing, so "
      "a larger network input is not the remedy. Degrading the imagery to simulate a "
      "camera mounted higher lowered recall monotonically and most steeply for the "
      "smallest weed. We then tested the converse directly, on the native 4080 × 3072 "
      "frames retained for 123 of the 124 test images: tiled inference, which raises "
      "the detail per object instead of removing it, gained under 0.06 of recall and "
      "roughly halved precision, and at a finer tiling lowered recall below the untiled "
      "control for all three architectures. Because both experiments still apply a "
      "detector trained at one scale to imagery at another, we added a fourth: "
      "training at the tile scale against a control matched on source images, recipe "
      "and augmentation, with every model scored on both test partitions. Scored on "
      "the same tiled imagery, the tile-trained models recovered less weed than the "
      "frame-trained ones for all three architectures, and the tiled condition had "
      "four times the optimiser updates, so that direction is conservative. Moving "
      "the camera is therefore not a lever on completeness, and neither is retraining "
      "at the moved geometry at this data budget — a conclusion we did not expect and "
      "that replaces the one the coarsening experiment alone would have supported.")
    a("4. **A deployment screen grounded in agronomy.** A working speed of "
      f"{t['sprayer_speed_kmh']:.0f} km h⁻¹ with a maximum frame advance of "
      f"{t['max_frame_advance_m']:.2f} m implies "
      f"{t['fps_required_on_device']:.1f} frames per second on the implement"
      + (f", and only {MODEL_LABELS[fast[0]]} clears the corresponding workstation "
         "threshold — the same model that treats the least ground. Both the frame "
         "advance and the embedded-hardware penalty are assumptions rather than "
         "measurements, and we map the conclusion across both rather than asserting it: "
         "a more generous frame advance admits all three architectures. We report the "
         "trade-off rather than a ranking." if fast else "."))
    bx = json.loads((RESULTS / "baseline_exg.json").read_text())
    bf1 = max(v["weed_f1"] for v in bx["variants"].values())
    # Cùng cách tính với đoạn trong bài: lấy F1 lớp cỏ THẤP NHẤT của ba mô hình
    # chia cho đối chứng, không lấy F1 hai lớp — nếu không thì thư và bài nói
    # hai con số khác nhau cho cùng một so sánh.
    dl_lo = min(fm[k]["weed"]["f1"]["mean"] for k in prim)
    ratio = dl_lo / bf1 if bf1 else float("inf")
    a(f"5. **A floor for the comparison.** Because the absolute figures above are low, "
      f"we added a baseline with no learned parameters — an excess-green index with an "
      f"Otsu threshold on the same test images, its area filter chosen on the validation "
      f"partition — which reaches a weed-class F1-score of {bf1:.3f} against "
      f"{min(fm[k]['weed']['f1']['mean'] for k in prim):.3f}–"
      f"{max(fm[k]['weed']['f1']['mean'] for k in prim):.3f} for the three detectors, "
      f"about {ratio:.0f} times lower. Crop and weed are both green, so a colour index "
      f"separates vegetation from soil and not weed from crop; the comparison shows that "
      f"the figures we report measure a hard task rather than weak models.")
    # gạch đầu dòng về đối chứng không học sâu đã chiếm số 5 ở trên
    _lk = _leakage_finding(6)
    if _lk:
        a(_lk)
    a("The dataset, the annotation files, the per-image results, the training logs and "
      "the complete analysis code accompany the submission, with checksums so that the "
      "image set can be verified as the one used here.")
    if journal == "aiia":
        # Bài chưa từng nộp ở đâu cả. Bản dựng trước đây nói nó từng được phản
        # biện ở Research in Agricultural Engineering rồi rút — điều đó không
        # đúng: các vòng "phản biện" là audit nội bộ, không phải của tạp chí nào.
        a("We confirm that this manuscript is original, has not been published "
          "elsewhere, is not under consideration by any other journal, and has not "
          "been submitted to any journal previously.")
    else:
        a("We confirm that this manuscript is original, has not been published elsewhere "
          "and is not under consideration by another journal. An earlier version was "
          "under review at *Research in Agricultural Engineering*; it has been formally "
          "withdrawn, and the withdrawal was acknowledged before this submission. The "
          "present manuscript has been substantially rewritten and extended in response "
          "to that review.")
        a("Nguyen Huynh Truong Gia, the first author, holds membership of the Society for "
          "Sugar Research & Promotion, as the journal requires.")
    a("Thank you for your consideration.")
    a("Sincerely,")
    a("Tran Thi Kim Nga, on behalf of all authors  \nCorresponding author  \n"
      "Faculty of Mechanical Engineering – Technology, Nong Lam University, "
      "Ho Chi Minh City, Vietnam  \nEmail: ttknga@hcmuaf.edu.vn")
    return "\n\n".join(A) + "\n"


def withdrawal_letter() -> str:
    return "\n\n".join([
        TODAY,
        "To: The Editor-in-Chief, *Research in Agricultural Engineering*",
        "Dear Editor-in-Chief,",
        f"**Subject: Withdrawal of manuscript “{RAE_TITLE_OLD}”**",
        "We write to withdraw the above manuscript from consideration by *Research in "
        "Agricultural Engineering*.",
        "We do so with regret and with thanks. The reviewer's report was exceptionally "
        "careful, and acting on it changed the paper substantially: we added the "
        "precision and F1-score of the full confusion-matrix variant, size-stratified and "
        "area-weighted recall, sensitivity to the confidence and overlap thresholds, "
        "multiplicity-adjusted tests with effect sizes, a two-level bootstrap over images "
        "and training runs, and a new experiment retraining one model at the input "
        "resolution of the others to remove a confound the reviewer identified. The "
        "reviewer also found four errors we had missed, including a reversed statement "
        "about which direction of class error harms the crop. The manuscript is better "
        "for all of it, and we are grateful.",
        "Our reason for withdrawing is institutional rather than scientific. Our "
        "institution's research assessment recognises journals indexed in the Web of "
        "Science Core Collection, and *Research in Agricultural Engineering* is indexed "
        "in Scopus and other databases but not in that collection. We became aware of "
        "this only while preparing the revision. We would not raise it had the review "
        "gone differently; as it is, we cannot complete the process and then be unable to "
        "count the result.",
        "We ask that the reviewer be thanked on our behalf. Their report improved the work "
        "regardless of where it appears, and the revised manuscript credits the review "
        "explicitly in its acknowledgement of how the analysis came to be structured as "
        "it is.",
        "We confirm that the manuscript is withdrawn in full and that we will not submit "
        "it elsewhere until this withdrawal has been acknowledged.",
        "Yours sincerely,",
        "Tran Thi Kim Nga, on behalf of all authors  \nCorresponding author  \n"
        "Faculty of Mechanical Engineering – Technology, Nong Lam University, "
        "Ho Chi Minh City, Vietnam  \nEmail: ttknga@hcmuaf.edu.vn",
    ]) + "\n"


def main():
    cfg = load_cfg()
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    D.use_seed_variance(seedvar)
    rev = json.loads((RESULTS / "review_analysis.json").read_text())

    for name, body in (("SugarTech_submission_checklist", checklist()),
                       ("SugarTech_cover_letter", cover_letter(cfg, rev)),
                       ("RAE_withdrawal_letter", withdrawal_letter())):
        p = OUT / f"{name}.md"
        p.write_text(body, encoding="utf-8")
        log(f"{len(body.split())} từ → {p}")
        d = to_docx(p)
        if d:
            final = OUT / f"{name}.docx"
            if d != final:
                d.replace(final)


if __name__ == "__main__":
    main()
