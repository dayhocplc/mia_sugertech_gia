#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gom toàn bộ thông tin của bài vào một chỗ, theo quy ước của dự án.

    python3 src/assemble_paper.py

Dự án mia/ov dùng hai thư mục: `paper/` giữ nguồn bản thảo, và
`submission_<TẠP CHÍ>/` giữ đúng bộ tệp đem nộp với tên gọn. Script này dựng
cùng cấu trúc cho bài RAE/Sugar Tech:

    rae/paper/SugarTech/      bản thảo, phụ lục, thư, phiếu kiểm, hình
    rae/paper/RAE/            bản gửi Research in Agricultural Engineering
    rae/paper/review/         tài liệu rà soát tiếng Việt
    rae/paper/README.md       bản đồ: tệp nào là gì, sinh bằng lệnh nào
    rae/submission_SugarTech/ bộ nộp sạch, tên theo kiểu Springer

Script CHÉP chứ không di chuyển. Các tệp gốc do những script khác ghi ra ở
đường dẫn cố định (manuscript/, figures/, kết quả ở gốc rae/); di chuyển chúng
sẽ làm hỏng chuỗi dựng lại. Chạy lại script này sau mỗi lần dựng bản thảo thì
thư mục bài luôn khớp với bản mới nhất.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RAE, log

SRC = RAE / "manuscript"
PAPER = RAE / "paper"
SUBMIT = RAE / "submission_SugarTech"

# (tệp nguồn ở manuscript/, thư mục đích trong paper/)
SUGARTECH = ["SugarTech_manuscript", "SugarTech_supplementary",
             "SugarTech_cover_letter", "SugarTech_submission_checklist",
             "SugarTech_response_to_reviewers"]
RAE_FILES = ["RAE_manuscript", "RAE_supplementary", "RAE_title_page",
             "RAE_cover_letter", "RAE_response_to_reviewers",
             "RAE_withdrawal_letter"]
EXTS = (".docx", ".md", ".pdf")          # .html là tệp trung gian, không chép

# Hai tài liệu này do script sinh ra ở gốc rae/ (make_report.py và
# build_knowledge_graph.py ghi vào đó, make_package.py đọc từ đó), nên chúng ở
# lại gốc và được CHÉP sang. Các tài liệu viết tay đã CHUYỂN hẳn vào
# paper/review/ và không còn bản ở gốc.
REVIEW = ["BAO_CAO_KET_QUA.md", "KNOWLEDGE_GRAPH.md"]

# tên tệp khi nộp: Springer đặt tên theo vai trò, không theo tên nội bộ
# Thư phản hồi phản biện KHÔNG có trong bảng này: đây là lần nộp đầu cho
# Sugar Tech, bản rà soát mà nó trả lời là bản đọc trước khi nộp. Thư nằm ở
# paper/SugarTech/ làm hồ sơ nội bộ.
SUBMIT_MAP = {
    "SugarTech_manuscript.docx": "Manuscript.docx",
    "SugarTech_supplementary.docx": "ESM_1.docx",
    "SugarTech_cover_letter.docx": "Cover_letter.docx",
    "SugarTech_submission_checklist.docx": "Submission_checklist.docx",
}


def _copy(src: pathlib.Path, dst: pathlib.Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _group(names: list[str], out: pathlib.Path) -> int:
    n = 0
    for stem in names:
        for ext in EXTS:
            if _copy(SRC / f"{stem}{ext}", out / f"{stem}{ext}"):
                n += 1
    return n


def readme(counts: dict) -> str:
    return f"""# Bài báo: phát hiện cỏ dại trong ruộng mía

Thư mục này gom mọi thứ thuộc về bài. Nội dung ở đây là BẢN CHÉP; tệp gốc do
các script trong `../src/` sinh ra ở `../manuscript/`, `../figures/` và
`../results/`. Chạy lại `python3 src/assemble_paper.py` sau mỗi lần dựng bản
thảo để thư mục này khớp với bản mới nhất.

## Hai phiên bản của cùng một nghiên cứu

Cùng số liệu, khác khung lập luận và khác tạp chí.

| | `SugarTech/` | `RAE/` |
|---|---|---|
| Tạp chí | Sugar Tech (Springer) | Research in Agricultural Engineering |
| Chỉ mục | SCIE, Scopus Q2, IF 1.9 | Scopus Q3, không có WoS |
| Trạng thái | bản đang nhắm nộp | đã qua phản biện vòng 1, cần rút nếu nộp Sugar Tech |
| Dẫn bài bằng | bài toán của máy phun | phê phán quy ước đo lường |
| Khổ | 10 pt, giãn đôi, ≤25 trang | 12 pt, giãn 1,5, ≤15 trang |
| Số tệp | {counts['sugartech']} | {counts['rae']} |

Bản Sugar Tech đặt phê phán quy ước ma trận nhầm lẫn vào mục Thảo luận thay vì
mở đầu, vì độc giả tạp chí đó là nhà khoa học cây mía chứ không phải cộng đồng
thị giác máy. Không con số nào khác nhau giữa hai bản.

## Thư mục

- `SugarTech/` — bản thảo, phụ lục, thư gửi toà soạn, phiếu kiểm, hình 600 dpi
- `RAE/` — bản gửi RAE, kèm thư phản hồi phản biện và thư rút bài
- `review/` — tài liệu rà soát tiếng Việt ({counts['review']} tệp)
- `../submission_SugarTech/` — bộ tệp đem nộp, đã đổi sang tên Springer

## Dựng lại

```bash
python3 src/build_manuscript_sugartech.py     # bản Sugar Tech
python3 src/build_supplementary.py --prefix SugarTech
python3 src/build_sugartech_extras.py         # thư + phiếu kiểm + thư rút bài
RAE_FIG_DPI=600 RAE_FIG_LANG=en python3 src/make_figures.py --workflow --qualitative
python3 src/export_springer_figures.py        # Fig1/Fig2 TIFF+PNG 600 dpi
python3 src/build_manuscript.py               # bản RAE
python3 src/assemble_paper.py                 # gom vào đây
```

## Kiểm tra trước khi nộp

```bash
python3 src/selftest.py
python3 src/audit.py
python3 src/check_consistency.py --docx manuscript/SugarTech_manuscript.docx \\
        --journal sugartech
```

Ba lớp này phải sạch. Lớp thứ ba còn nhắc những ô chờ nhóm tác giả điền; xem
`SugarTech/SugarTech_submission_checklist.md` để biết danh sách đầy đủ.
"""


def main():
    counts = {}
    counts["sugartech"] = _group(SUGARTECH, PAPER / "SugarTech")
    counts["rae"] = _group(RAE_FILES, PAPER / "RAE")

    figdir = SRC / "springer_figures"
    n_fig = 0
    if figdir.is_dir():
        for f in sorted(figdir.glob("Fig*")):
            n_fig += _copy(f, PAPER / "SugarTech" / "figures" / f.name)
    counts["figures"] = n_fig

    n_rev = 0
    for name in REVIEW:
        n_rev += _copy(RAE / name, PAPER / "review" / name)
    # cộng cả những tài liệu viết tay đã nằm sẵn trong paper/review/
    n_rev += len([f for f in (PAPER / "review").rglob("*.md")
                  if f.name not in REVIEW])
    counts["review"] = n_rev

    (PAPER / "README.md").write_text(readme(counts), encoding="utf-8")

    # bộ nộp: tên theo vai trò
    n_sub = 0
    for src_name, dst_name in SUBMIT_MAP.items():
        n_sub += _copy(SRC / src_name, SUBMIT / dst_name)
    for f in sorted(figdir.glob("Fig*.tif")) if figdir.is_dir() else []:
        n_sub += _copy(f, SUBMIT / "figures" / f.name)
    # Springer đặt tên tệp SI theo thứ tự: ESM_1 là tệp .docx phụ lục,
    # ESM_2 là gói mã nguồn và dữ liệu này.
    n_sub += _copy(RAE / "ESM_2.zip", SUBMIT / "ESM_2.zip")

    log(f"paper/SugarTech : {counts['sugartech']} tệp + {n_fig} hình")
    log(f"paper/RAE       : {counts['rae']} tệp")
    log(f"paper/review    : {counts['review']} tệp")
    log(f"submission_SugarTech: {n_sub} tệp")
    log(f"→ {PAPER}  và  {SUBMIT}")


if __name__ == "__main__":
    main()
