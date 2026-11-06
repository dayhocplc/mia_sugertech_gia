#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh thư gửi toà soạn (cover letter) cho bản thảo RAE.

    python3 src/build_cover_letter.py

Mọi con số lấy từ results/ qua cùng các accessor mà bản thảo dùng, nên thư và
bài không thể lệch nhau. Đầu ra: manuscript/RAE_cover_letter.{md,docx}
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import TITLE, md_inline, to_docx
from common import MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope, throughput_requirement

JOURNAL = "Research in Agricultural Engineering"
CORRESPONDING = {
    "name": "Tran Thi Kim Nga",
    "role": "Corresponding author",
    "affil": ("Faculty of Mechanical Engineering – Technology, "
              "Nong Lam University Ho Chi Minh City"),
    "city": "Ho Chi Minh City, Vietnam",
    "email": "ttknga@hcmuaf.edu.vn",
}


def build() -> str:
    cfg = load_cfg()
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    D.use_seed_variance(seedvar)

    primary, secondary = scope(cfg)
    t = throughput_requirement(cfg)
    ap = {k: D.seed_mean(k, "ap50") for k in primary}
    rec = {k: D.seed_mean(k, "weed_recall_full") for k in primary}
    cv = {k: D.seed_sd(k, "ap50") / ap[k] * 100 for k in primary}
    fps = {k: clean_fps(k)[0] for k in primary}
    best = max(primary, key=lambda k: ap[k])
    fast = max(primary, key=lambda k: fps[k])
    n_img = stats["_combined"]["source_images_union"]
    n_test = stats["bbox"]["test"]["images"]
    n_seeds = len([e for e in seedvar[best]["runs"] if e.get("status") == "ok"])
    f1_lo = min(D.seed_mean(k, "f1") or 0 for k in primary)
    # Precision lớp cỏ ở biến thể giữ nền — lấy từ review_analysis, cùng nguồn
    # với bài, để thư và bài không bao giờ lệch nhau.
    _rev = json.loads((RESULTS / "review_analysis.json").read_text())
    fp = {k: _rev["full_metrics"][k]["weed"]["precision"]["mean"] for k in primary}

    L = []
    A = L.append
    A(datetime.date.today().strftime("%d-%m-%Y"))
    A("")
    A("To: The Editor-in-Chief and Editorial Board")
    A(JOURNAL)
    A("")
    A(f'Subject: Submission of manuscript "{TITLE}"')
    A("")
    A("Dear Editor-in-Chief and Editorial Board Members,")
    A("")
    A(f'We are submitting our manuscript, "{TITLE}", for consideration as a research '
      f"article in {JOURNAL}.")
    A("")
    A("Weed detection in sugarcane is a prerequisite for site-specific herbicide "
      "application, and the deep learning literature on it reports accuracies close to "
      "99%. Those figures are almost always derived from a confusion matrix over matched "
      "detections, with the background row and column removed. That construction measures "
      "how reliably a detector assigns the correct class to an object it has already "
      "found; it says nothing about how many of the weeds present in the field are found "
      "at all. Our work began as a straightforward comparison of detector architectures on "
      "field imagery and became, in the course of the analysis, an examination of that "
      "gap.")
    A("")
    A("The main contributions of the manuscript are the following.")
    A("")
    A(f"1. **A field dataset acquired under production conditions.** "
      f"{n_img:,} source images of sugarcane (cultivar KK3) and its associated weed flora "
      f"were collected at Tan Hung farm, Chau Thanh district, Tay Ninh province, Vietnam, "
      f"at close range and nadir view, across three times of day, three weather conditions "
      f"and four growth stages. The annotation files, evaluation code, per-image results "
      f"and training logs are provided as supplementary material, together with SHA-256 "
      f"checksums of every image so that the image set, which is supplied separately on "
      f"request because of its size, can be verified as the one used here.")
    A("")
    A(f"2. **A quantified separation between class discrimination and detection "
      f"completeness.** On the same test partition the three detectors reached "
      f"two-class mean F1-scores of {f1_lo:.3f} and above when the background row and "
      f"column were excluded. The same predictions scored with those retained give "
      f"weed-class Precision of {min(fp.values()):.3f} to {max(fp.values()):.3f} and "
      f"Recall of {min(rec.values()):.3f} to {max(rec.values()):.3f}: about half the "
      f"annotated weed is not found, and about half of what is reported as weed is not "
      f"annotated weed. Which set of figures is quoted decides whether this task looks "
      f"solved, and the difference between them is a reporting convention, not a "
      f"result. We locate the cause by measurement — recall falls monotonically with "
      f"object size — rather than by assertion.")
    A("")
    div = {k: (D._n_div(k), D._n_att(k)) for k in primary if D._n_div(k)}
    div_txt = ("; " + ", ".join(
        f"{n} of the {a} attempted runs for {MODEL_LABELS[k]} diverged"
        for k, (n, a) in div.items())) if div else ""
    A(f"3. **A statistical treatment that separates test-set variability from training "
      f"variability.** Each model was trained repeatedly from different random seeds "
      f"({D.trained_n_times(primary)}{div_txt}) "
      f"and evaluated with a paired bootstrap over test images. {D._n_sig_txt(primary).capitalize()} of the three "
      f"pairwise differences in AP50 reach significance once training variability is "
      f"included, and the coefficient of variation across seeds ranged from "
      f"{min(cv.values()):.1f}% to {max(cv.values()):.1f}%. We believe this bears on how "
      f"single-run results reported to three decimal places elsewhere in this literature "
      f"should be read.")
    A("")
    A(f"4. **A deployment screening grounded in agronomy rather than in the results.** "
      f"A field speed of {t['sprayer_speed_kmh']:.0f} km/h with a maximum frame advance of "
      f"{t['max_frame_advance_m']:.2f} m implies {t['fps_required_on_device']:.1f} frames "
      f"per second on the deployment hardware, a requirement fixed before evaluation. "
      f"Applying the corresponding workstation threshold, only {MODEL_LABELS[fast]} "
      f"remains a candidate. We state explicitly that this is a necessary and not a "
      f"sufficient condition, since on-device measurement was not available to us.")
    A("")
    A(f"The study reaches a trade-off rather than a single recommendation. "
      f"{MODEL_LABELS[best]} obtained the highest AP50 ({ap[best]:.1f}), but stricter "
      f"overlap thresholds do not reproduce that ordering and no pairwise difference "
      f"in weed-class F1-score survives a bootstrap over both test images and training "
      f"runs. {MODEL_LABELS[fast]} was the only model fast enough for the intended "
      f"platform, at the lowest weed Recall ({rec[fast]:.3f}); retraining it at the "
      f"input resolution of the other two left its detection quality unchanged, so "
      f"that gap is not an artefact of input size. We report this openly because the "
      f"honest form of the result is more useful to readers building spraying "
      f"equipment than a ranking would "
      f"be.")
    A("")
    A(f"We also draw the Editor's attention to a limitation that shaped the design. Five "
      f"models were trained, but two of them "
      f"({' and '.join(MODEL_LABELS[k] for k in secondary)}) were trained on a polygon "
      f"annotation set whose granularity and test partition differ from those of the "
      f"bounding-box set. We therefore restrict the comparison to the three models sharing "
      f"the bounding-box convention and a common {n_test}-image test partition, and report "
      f"the other two separately without ranking them against the three. This is stated in "
      f"the Methods and in the Conclusion.")
    A("")
    A(f"{JOURNAL} is, in our view, the appropriate venue for this work. The question we "
      f"address is not the design of a new architecture but whether existing detectors "
      f"meet the requirements of a field implement, expressed in the terms an agricultural "
      f"engineer would use: detection completeness, throughput at working speed, and "
      f"reproducibility between training runs. The manuscript is written for readers "
      f"developing precision spraying equipment rather than for a computer vision "
      f"audience.")
    A("")
    A("We confirm that this manuscript is original, has not been published elsewhere, is "
      "not under consideration by another journal, and that all authors have read and "
      "approved the submission. The authors declare no conflict of interest. The "
      "annotation files, source code, per-image results and training logs needed to "
      "reproduce every number in the manuscript accompany this submission; the image set "
      "itself is available from the corresponding author on request, and checksums are "
      "included so that it can be verified against the one used here.")
    A("")
    # Giải trình độ dài: bài vượt giới hạn 15 trang, và phần vượt gần như toàn
    # bộ là phân tích do phản biện yêu cầu. Nêu thẳng và xin Editor cân nhắc.
    A("One matter requires the Editor's judgement. The instructions limit an original "
      "paper to 15 standard pages including tables and figures. In the formatting the "
      "instructions specify — Times New Roman 12, 1.5 line spacing, 2.5 cm margins — "
      f"this manuscript runs to {_pdf_pages('RAE_manuscript.pdf') or 22} pages. The "
      "excess is almost entirely analysis the "
      "reviewer asked for: precision and F1 in the full confusion-matrix variant, "
      "size-stratified and area-weighted recall, sensitivity to the confidence and IoU "
      "thresholds, Holm-adjusted tests with effect sizes, a two-level bootstrap, "
      "per-session results, and a new experiment retraining one model at the input "
      "resolution of the other two to remove a confound the reviewer identified. We "
      f"have cut the manuscript from 27 pages to "
      f"{_pdf_pages('RAE_manuscript.pdf') or 22}, moved ten tables and two figures "
      "into a supplementary file for online publication only, and verified that the "
      "main text stands alone without it. We would rather ask whether the remaining "
      "length is acceptable than drop analysis the reviewer requested, and will cut "
      "further on your instruction.")
    A("")
    A("Thank you for your time and consideration.")
    A("")
    A("Sincerely,")
    A("")
    A(f"{CORRESPONDING['name']}")
    A(f"{CORRESPONDING['role']}")
    A(f"{CORRESPONDING['affil']}")
    A(f"{CORRESPONDING['city']}")
    A(f"Email: {CORRESPONDING['email']}")
    return "\n\n".join(L)


def _pdf_pages(name: str) -> int | None:
    """Số trang thật của bản thảo đã render. Viết cứng con số này trong thư là
    cách chắc chắn để nó sai sau lần sửa tiếp theo."""
    import re
    import subprocess
    f = RAE / "manuscript" / name
    if not f.exists():
        return None
    try:
        out = subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True).stdout
        return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))
    except Exception:
        return None


def main():
    out = RAE / "manuscript" / "RAE_cover_letter.md"
    text = build()
    out.write_text(text, encoding="utf-8")
    log(f"{len(text.split())} từ → {out}")
    d = to_docx(out)
    if d:
        final = out.with_suffix(".docx")
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
