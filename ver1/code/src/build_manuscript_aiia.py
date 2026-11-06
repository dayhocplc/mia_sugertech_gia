#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dựng bản thảo cho Artificial Intelligence in Agriculture (KeAi/Elsevier).

Toàn bộ nội dung dùng chung với bản Sugar Tech — cùng bộ dựng, cùng các con số,
cùng mã sinh ra chúng — nên hai bản không thể lệch nhau về nội dung. Khác biệt
chỉ nằm ở lớp trình bày mà `aiia_layout.py` áp lên văn bản đã dựng, và ở danh
mục tài liệu mà `journal_style.to_elsevier` sinh ra.

Dùng
----
    python3 src/build_manuscript_aiia.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import aiia_layout as A
from build_manuscript import MODEL_LABELS, to_docx
from build_manuscript_sugartech import OUT, build
from common import RESULTS, load_cfg, log

# Elsevier đặt khai báo về trí tuệ nhân tạo thành một mục riêng ở cuối bài,
# ngoài đoạn đã có trong Phương pháp. Câu chữ lấy đúng từ đoạn đó để hai chỗ
# không thể nói khác nhau.
AI_SECTION_HEAD = ("### Declaration of generative AI and AI-assisted technologies "
                   "in the writing process")


def _ai_declaration(md: str) -> str:
    """Lấy lại nguyên văn đoạn khai báo AI trong Phương pháp làm mục khai báo."""
    m = re.search(r"(?m)^### \d+\.\d+\. Use of artificial-intelligence tools\n\n(.+?)\n",
                  md)
    if not m:
        raise SystemExit("không thấy tiểu mục khai báo công cụ AI trong Phương pháp")
    return AI_SECTION_HEAD + "\n\n" + m.group(1)


def main():
    cfg = load_cfg()
    res = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in MODEL_LABELS}
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    boot = json.loads((RESULTS / "bootstrap.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())

    md = build(cfg, stats, res, seedvar, boot, comp, journal="aiia")

    md = A.number_sections(md)
    md, n_t, n_f = A.captions(md)
    log(f"chú thích kiểu Elsevier: {n_t} bảng, {n_f} hình")
    if (n_t, n_f) != (5, 5):
        raise SystemExit(f"chờ 5 bảng và 5 hình, nhận {n_t} và {n_f}")
    md = A.declarations(md, _ai_declaration(md))
    md = A.appendices(md)
    md = A.acknowledgements(md)

    n_sec = len(re.findall(r"(?m)^## \d+\. ", md))
    n_sub = len(re.findall(r"(?m)^### \d+\.\d+\. ", md))
    log(f"{n_sec} mục chính được đánh số, {n_sub} tiểu mục")

    p = OUT / "AIIA_manuscript.md"
    p.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {p}")
    # AIIA nhận bản thảo một cột, không giới hạn số trang; để giãn đôi cho
    # người phản biện dễ ghi chú, như bản gửi Sugar Tech.
    d = to_docx(p, double_spaced=True, font_pt=11, page_numbers=True)
    if d:
        final = OUT / "AIIA_manuscript.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
