#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Title Page nộp riêng, kèm bản thảo đã làm mù.

    python3 src/build_title_page.py

Hướng dẫn tác giả của tạp chí yêu cầu nộp ba tệp: (i) Title Page có loại bài,
tiêu đề, tên đầy đủ các tác giả theo thứ tự sẽ in, cơ quan, email tác giả liên
hệ, SỐ KÝ TỰ của bản thảo, và phần ghi công; (ii) tệp bản thảo đã làm mù; (iii)
tệp hình. Vì vậy tên tác giả không còn nằm trong RAE_manuscript.docx.

Số ký tự lấy trực tiếp từ bản thảo đã dựng, nên không thể lệch với tệp thật.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_manuscript import AFFIL, AUTHORS, CORRESP, KEYWORDS, TITLE, to_docx
from common import RAE, log

OUT = RAE / "manuscript"


def main():
    ms = OUT / "RAE_manuscript.md"
    if not ms.exists():
        raise SystemExit("Chưa có RAE_manuscript.md — chạy src/build_manuscript.py trước")
    text = ms.read_text(encoding="utf-8")
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    clean = re.sub(r"\*\*|\*|&nbsp;", "", clean)
    n_chars = len(clean)
    n_words = len(clean.split())
    n_tbl = len(re.findall(r"^\*\*Table \d+\.", text, re.M))
    n_fig = len(re.findall(r"^Figure \d+\.", text, re.M))
    # Đếm phụ lục từ chính tệp phụ lục, không viết cứng.
    sup = OUT / "RAE_supplementary.md"
    stext = sup.read_text(encoding="utf-8") if sup.exists() else ""
    n_stbl = len(re.findall(r"\*\*Supplementary Table S\d+\.", stext))
    n_sfig = len(re.findall(r"Supplementary Figure S\d+\.", stext))

    L = [
        "# Title page", "",
        "**Type of document:** original paper", "",
        f"**Title:** {TITLE}", "",
        f"**Running title:** Detection completeness in sugarcane weed detection", "",
        f"**Authors:** {AUTHORS.replace('<sup>', ' (').replace('</sup>', ')')}", "",
        f"**Affiliation:** {AFFIL.replace('<sup>1</sup>', '1 — ')}", "",
        f"**Corresponding author:** {CORRESP.replace('*Corresponding author*: ', '')}", "",
        f"**Keywords:** {KEYWORDS}", "",
        f"**Number of characters (manuscript file, including spaces, excluding "
        f"figures):** {n_chars:,}", "",
        f"**Approximate word count:** {n_words:,}", "",
        f"**Tables:** {n_tbl}  ·  **Figures:** {n_fig}  ·  "
        f"**Supplementary tables:** {n_stbl}  ·  "
        f"**Supplementary figures:** {n_sfig}", "",
        "**Acknowledgement:** The authors thank the management and staff of Tan Hung "
        "farm, Chau Thanh district, Tay Ninh province, for access to the sugarcane "
        "fields and for their assistance during image acquisition, and Nong Lam "
        "University Ho Chi Minh City for the provision of computing facilities.", "",
        "**Funding:** This work was supported by the Ministry of Education and Training, "
        "Vietnam, under project code B2025-NLS-02.", "",
        "**Author contributions:** Nguyen Huynh Truong Gia: Conceptualization, "
        "Methodology, Software, Investigation, Data curation, Formal analysis, "
        "Visualization, Writing – original draft. Nguyen Duc Khuyen: Investigation, "
        "Data curation, Writing – review & editing. Vo Van Qui: Investigation, Data "
        "curation, Writing – review & editing. Tran Quoc Khanh: Investigation, "
        "Validation, Writing – review & editing. Tran Thi Kim Nga: Conceptualization, "
        "Methodology, Supervision, Project administration, Resources, Writing – review "
        "& editing. All authors read and approved the manuscript submitted for "
        "publication.", "",
        "**Conflict of interest:** The authors declare no conflict of interest.", "",
        "**Use of artificial intelligence:** A large language model (Claude, Anthropic) "
        "was used as a programming and drafting assistant; the declaration is given in "
        "the Materials and Methods, because the tool was used as part of the research "
        "process and not only in preparing the manuscript.", "",
    ]
    md = "\n".join(L)
    p = OUT / "RAE_title_page.md"
    p.write_text(md, encoding="utf-8")
    log(f"bản thảo: {n_chars:,} ký tự, {n_words:,} từ, {n_tbl} bảng, {n_fig} hình")
    log(f"→ {p}")
    d = to_docx(p)
    if d:
        final = OUT / "RAE_title_page.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
