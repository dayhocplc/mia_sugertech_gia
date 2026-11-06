#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Các tệp đi kèm bản gửi Artificial Intelligence in Agriculture.

    python3 src/build_aiia_extras.py

Nội dung khoa học của thư gửi ban biên tập dùng chung với bản Sugar Tech, qua
`cover_letter(..., journal="aiia")`, nên hai thư không thể lệch nhau về số liệu.
AIIA phản biện đơn mù và không đòi title page riêng, nên ở đây chỉ có thư gửi
ban biên tập và danh mục tự kiểm.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from build_manuscript import to_docx
from build_sugartech_extras import OUT, cover_letter
from common import RESULTS, load_cfg, log


def main():
    cfg = load_cfg()
    D.use_seed_variance(json.loads((RESULTS / "seed_variance.json").read_text()))
    rev = json.loads((RESULTS / "review_analysis.json").read_text())

    body = cover_letter(cfg, rev, journal="aiia")
    p = OUT / "AIIA_cover_letter.md"
    p.write_text(body, encoding="utf-8")
    log(f"{len(body.split())} từ → {p}")
    d = to_docx(p)
    if d:
        final = OUT / "AIIA_cover_letter.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")


if __name__ == "__main__":
    main()
