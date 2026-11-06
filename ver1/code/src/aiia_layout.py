#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lớp trình bày của Artificial Intelligence in Agriculture (KeAi/Elsevier).

Hướng dẫn của tạp chí đòi bốn thứ mà bản thảo dựng cho Sugar Tech không có:

    1. mục được đánh số rõ ràng ("1.", "1.1", "1.1.1")
    2. chú thích bảng và hình theo kiểu Elsevier ("Table 1." / "Fig. 1.")
    3. danh mục tài liệu kiểu author-date của Elsevier (xem journal_style.py)
    4. các khai báo mang đúng tên mục của Elsevier, trong đó có khai báo về
       công cụ trí tuệ nhân tạo kèm tên, phiên bản và nhà sản xuất

Ba việc 1, 2, 4 là trình bày thuần: chúng không đổi một con số nào, nên làm
thành một lượt xử lý trên văn bản đã dựng thay vì rẽ nhánh khắp bộ dựng bài.
Mỗi phép thay đều được kiểm là đã khớp đúng số lần mong đợi; nếu bộ dựng đổi
cách viết mà lớp này không theo kịp thì nó dừng lại chứ không âm thầm bỏ qua.

Tên tệp phụ lục cũng đổi: "Online Resource" là cách gọi của Springer, Elsevier
gọi là "Appendix A"/"Appendix B".
"""

from __future__ import annotations

import re

# ── 1. mục được đánh số ─────────────────────────────────────────────────────
# (tiêu đề trong bản dựng, tiêu đề của AIIA, có đánh số hay không)
SECTION_MAP = [
    ("Introduction", "Introduction", True),
    ("Materials and Methods", "Materials and methods", True),
    ("Results", "Results", True),
    ("Discussion", "Discussion", True),
    ("Conclusion", "Conclusions", True),
]
# Mục nào được phép nâng câu dẫn in đậm thành tiểu mục có số. Giới hạn đúng ba
# mục này để các khai báo ở cuối bài không bị nâng theo.
SUBSECTIONED = {"Materials and methods", "Results", "Discussion"}

# ── 4. tên mục khai báo theo Elsevier ───────────────────────────────────────
DECL_MAP = [
    ("**Competing Interests.**", "### Declaration of competing interest\n"),
    ("**Funding.**", "### Funding\n"),
    ("**Author Contributions.**", "### CRediT authorship contribution statement\n"),
    ("**Data Availability.**", "### Data availability\n"),
    ("**Code Availability.**", "### Code availability\n"),
    ("**Supplementary Information.**", "### Appendix A and Appendix B. Supplementary data\n"),
    ("**Ethics Approval.**", "### Ethics approval\n"),
]


def _promote_leads(body: str, sec_no: int) -> str:
    """Nâng câu dẫn in đậm đầu đoạn thành tiểu mục "N.M." và bỏ câu dẫn khỏi đoạn.

    Câu dẫn của bản dựng đã là một đầu đề đúng nghĩa ("**Study area and image
    dataset.** Ảnh được chụp …"), nên phép nâng chỉ là đổi cách trình bày.
    """
    out, n = [], 0
    for line in body.split("\n"):
        m = re.match(r"^\*\*([^*]{3,90}?)\.\*\*\s+(.+)$", line)
        if m:
            n += 1
            out.append(f"### {sec_no}.{n}. {m.group(1)}")
            out.append("")
            out.append(m.group(2))
        else:
            out.append(line)
    return "\n".join(out)


def number_sections(md: str) -> str:
    """Đánh số các mục chính và nâng câu dẫn thành tiểu mục trong ba mục nội dung."""
    heads = {old: (new, numbered) for old, new, numbered in SECTION_MAP}
    # [phần trước mục đầu, "## Tiêu đề", thân, "## Tiêu đề", thân, …]
    parts = re.split(r"(?m)^(## .+)$", md)
    out = [parts[0]]
    sec_no = 0
    seen = set()
    for i in range(1, len(parts), 2):
        head = parts[i][3:].strip()
        body = parts[i + 1]
        if head in heads:
            new_head, numbered = heads[head]
            seen.add(head)
            if numbered:
                sec_no += 1
                out.append(f"## {sec_no}. {new_head}")
            else:
                out.append(f"## {new_head}")
            out.append(_promote_leads(body, sec_no)
                       if new_head in SUBSECTIONED else body)
        else:
            out.append(parts[i])
            out.append(body)
    missing = [o for o, _n, _b in SECTION_MAP if o not in seen]
    if missing:
        raise SystemExit(f"lớp AIIA: không thấy mục {missing} trong bản dựng")
    return "".join(out)


def captions(md: str) -> tuple[str, int, int]:
    """Chú thích kiểu Elsevier: "Table 1." và "Fig. 1." (có dấu chấm sau số)."""
    md, n_t = re.subn(r"(?m)^\*\*Table (\d+)\*\* ", r"**Table \1.** ", md)
    md, n_f = re.subn(r"(?m)^\*\*Fig\. (\d+)\*\* ", r"**Fig. \1.** ", md)
    return md, n_t, n_f


def appendices(md: str) -> str:
    """"Online Resource" của Springer thành "Appendix" của Elsevier."""
    md = md.replace("Online Resource 1", "Appendix A")
    md = md.replace("Online Resource 2", "Appendix B")
    return md


def acknowledgements(md: str) -> str:
    """Chuyển Lời cảm ơn xuống cuối bài, ngay trước danh mục tài liệu.

    Guide for Authors của AIIA nói rõ: "Collate acknowledgements in a separate
    section at the end of the article before the references and do not,
    therefore, include them on the title page, as a footnote to the title or
    otherwise." Bộ dựng chung đặt đoạn này ngay dưới từ khoá theo kiểu Springer,
    nên lớp AIIA phải nhấc nó xuống.
    """
    m = re.search(r"(?m)^\*\*Acknowledgements\*\* (.+?)\n", md)
    if not m:
        raise SystemExit("lớp AIIA: không thấy đoạn Lời cảm ơn để chuyển xuống cuối")
    body = m.group(1).strip()
    md = md[:m.start()] + md[m.end():]
    md = re.sub(r"\n{3,}", "\n\n", md, count=0)
    refs = "## References\n"
    if refs not in md:
        raise SystemExit("lớp AIIA: không thấy mục References để chèn Lời cảm ơn")
    return md.replace(refs, "### Acknowledgements\n\n" + body + "\n\n" + refs, 1)


def declarations(md: str, ai_declaration: str) -> str:
    """Đổi tên các mục khai báo và chèn khai báo về công cụ trí tuệ nhân tạo."""
    md = md.replace("## Statements and Declarations\n",
                    "## Declarations\n")
    for old, new in DECL_MAP:
        if old not in md:
            raise SystemExit(f"lớp AIIA: không thấy khai báo {old!r} trong bản dựng")
        md = md.replace(old, new, 1)
    anchor = "### Ethics approval\n"
    if anchor not in md:
        raise SystemExit("lớp AIIA: không thấy mục Ethics approval để chèn khai báo AI")
    md = md.replace(anchor, ai_declaration.rstrip() + "\n\n" + anchor, 1)
    return md
