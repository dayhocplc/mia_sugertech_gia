#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra tính nhất quán của bản thảo sau khi sửa.

    python3 src/check_consistency.py --docx ~/Downloads/Full_Manuscript_RAE_revised.docx

Bốn lỗi tìm thấy trong vòng rà soát vừa rồi đều cùng một kiểu: sửa bài từng
phần qua nhiều vòng, đoạn mới mâu thuẫn đoạn cũ, hoặc hai chỗ đọc hai nguồn số
khác nhau. Script này bắt đúng lớp lỗi đó, tự động, trước khi gửi bản thảo đi.

Kiểm 6 nhóm:
  C1  mỗi đại lượng xuất hiện nhiều lần phải có cùng một giá trị
  C2  con số trong bài phải khớp results/*.json
  C3  "Table N" trong chữ phải khớp cấu trúc bảng N thật trong file
  C4  không còn ô giữ chỗ
  C5  không có cặp phát biểu mâu thuẫn đã biết
  C6  độ dài Abstract

Trả mã thoát 1 nếu có lỗi, để cắm được vào quy trình tự động.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from common import MODEL_LABELS, RESULTS, load_cfg, scope

FAIL, WARN = [], []


def ok(msg):    print(f"  ✓ {msg}")
def bad(msg):   FAIL.append(msg); print(f"  ✗ {msg}")
def warn(msg):  WARN.append(msg); print(f"  ⚠ {msg}")


def accepted_text(docx: pathlib.Path, skill_dir: pathlib.Path | None) -> tuple[str, bytes]:
    """Văn bản sau khi Accept mọi Track Changes, cùng XML thô."""
    tmp = pathlib.Path(tempfile.mkdtemp()) / "a.docx"
    script = (skill_dir / "scripts/accept_changes.py") if skill_dir else None
    if script and script.exists():
        subprocess.run(["python3", str(script), str(docx), str(tmp)],
                       capture_output=True, check=False)
    src = tmp if tmp.exists() else docx
    xml = zipfile.ZipFile(src).read("word/document.xml")
    import html
    t = html.unescape(re.sub(r"<[^>]+>", "", xml.decode().replace("</w:p>", "\n")))
    return t, xml


def seed_mean(seedvar: dict, model: str, metric: str) -> float | None:
    v = [e[metric] for e in seedvar.get(model, {}).get("runs", [])
         if e.get("status") == "ok" and metric in e]
    return float(np.mean(v)) if v else None


# Luật hình thức khác nhau giữa hai tạp chí; đặt chung một chỗ để không phải
# sửa rải rác khi thêm nơi nộp thứ ba.
JOURNAL_RULES = {
    "rae": {
        "name": "Research in Agricultural Engineering",
        # Mẫu khớp văn bản ĐÃ TRÍCH từ .docx: dấu ** của markdown không còn,
        # chú thích nằm đầu dòng, còn viện dẫn thì nằm giữa câu.
        "table_cap": r"(?m)^Table (\d+)\. ",
        "fig_cap": r"(?m)^Figure (\d+)\. ",
        "fig_ref": "Figure",
        # một mục trong danh mục: "Surname A.B., ... (2017): ..."
        "ref_entry": r"{sur}\b[^\n]*?\b{yr}\b",
        "ref_line": r"^([A-ZÀ-Ýa-z][\w'’\-]+)[^\n]*?\((\d{4}[ab]?)\)",
        "title_max": 100, "title_no_colon": True,
        "abstract": (None, 200), "keywords": (5, 6), "keywords_lower": True,
        "refs_max": 25,
        # cột mà chữ trong bài hứa bảng N sẽ có
        "claims": {"Table 2": ["AP", "AP50", "AP75"],
                   "Table 3": ["Precision", "Recall", "F1"]},
        "pdf": "manuscript/RAE_manuscript.pdf", "pages_max": 15, "pages_hard": 24,
        "spacing": "Times 12, giãn 1,5, lề 2,5 cm",
    },
    "sugartech": {
        "name": "Sugar Tech",
        "table_cap": r"(?m)^Table (\d+) [A-Z]",
        "fig_cap": r"(?m)^Fig\. (\d+) [A-Z]",
        "fig_ref": "Fig.",
        # Springer: "Chen, Jiasheng, Hongwei Li, ... 2026. Tiêu đề."
        "ref_entry": r"{sur}\b[^\n]*?\b{yr}\b",
        "ref_line": r"^([A-ZÀ-Ýa-z][\w'’\-]+)[,.][^\n]*?\b(\d{4}[ab]?)\.",
        "title_max": None, "title_no_colon": False,
        "abstract": (150, 250), "keywords": (4, 6), "keywords_lower": False,
        "refs_max": None,
        "claims": {"Table 1": ["AP", "AP50", "AP75"],
                   "Table 2": ["Precision", "Recall", "F1"],
                   "Table 3": ["Parameters", "GMACs", "FPS"]},
        "pdf": "manuscript/SugarTech_manuscript.pdf", "pages_max": 25, "pages_hard": 25,
        "spacing": "giãn đôi, lề 2,5 cm",
    },
    "aiia": {
        "name": "Artificial Intelligence in Agriculture",
        # Mục được đánh số theo yêu cầu "clearly defined and numbered sections",
        # nên chú thích bảng/hình cũng theo kiểu Elsevier.
        "table_cap": r"(?m)^Table (\d+)\. ",
        "fig_cap": r"(?m)^Fig\. (\d+)\. ",
        "fig_ref": "Fig.",
        # Elsevier author-date: "Mini, G.A., Sales, D.O., Luppe, M., 2020. Tiêu đề."
        "ref_entry": r"{sur}\b[^\n]*?\b{yr}\b",
        "ref_line": r"^([A-ZÀ-Ýa-z][\w\'’\-]+),[^\n]*?,\s*(\d{4}[ab]?)\.",
        # Guide for Authors không nêu giới hạn tiêu đề, abstract, từ khoá hay số
        # trang nào; để None thay vì bịa ra một con số.
        "title_max": None, "title_no_colon": False,
        "abstract": (None, None), "keywords": (None, None), "keywords_lower": False,
        # Guide for Authors chỉ đòi abstract "concise and factual", không nêu số
        # từ — khác với trường hợp chưa tra được giới hạn.
        "abstract_no_limit": True,
        "refs_max": None,
        "claims": {"Table 1": ["AP", "AP50", "AP75"],
                   "Table 2": ["Precision", "Recall", "F1"],
                   "Table 3": ["Parameters", "GMACs", "FPS"]},
        "pdf": "manuscript/AIIA_manuscript.pdf", "pages_max": None, "pages_hard": None,
        "spacing": "một cột, không giới hạn số trang",
    },
}
RULES = JOURNAL_RULES["rae"]
# tiền tố tên tệp của từng tạp chí, dùng cho bản phụ lục đi kèm
_PREFIX = {"rae": "RAE", "sugartech": "SugarTech", "aiia": "AIIA"}


def c1_c2_numbers(text: str, seedvar: dict, cfg: dict) -> None:
    """Mọi chỉ số gắn với tên mô hình phải cùng một giá trị, và khớp số đo.

    Bản đầu chỉ kiểm AP50 nên bỏ lọt weed recall xuất hiện hai giá trị khác nhau
    (0,471 ở một đoạn, 0,483 ở đoạn khác). Nay kiểm cả nhóm chỉ số tỷ lệ.
    """
    primary, _ = scope(cfg)
    print("\nC1+C2 — AP50 trong bài so với số đo")
    _c1_ap50(text, seedvar, primary)
    _c1_rates(text, seedvar, primary)


def _known_values(k: str) -> set[float]:
    """Mọi giá trị dạng tỷ lệ đã TÍNH ĐƯỢC cho một mô hình, từ mọi tệp kết quả.

    Phép kiểm cũ giả định mỗi mô hình chỉ được xuất hiện với MỘT giá trị tỷ lệ.
    Giả định đó đúng khi bài chỉ báo cáo recall dạng full, nhưng sai từ khi bài
    cố ý báo cáo cả hai biến thể ma trận, recall theo ngưỡng IoU, theo diện
    tích, theo tứ phân vị kích thước và theo buổi chụp. Thay vào đó kiểm điều
    mạnh hơn và vẫn đúng: mỗi số gán cho một mô hình phải TỒN TẠI trong kết quả
    đã tính, tức không có số nào được bịa ra hoặc gõ sai.
    """
    vals: set[float] = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            vals.add(round(float(node), 3))

    for name in ("seed_variance.json", "review_analysis.json", "bootstrap.json",
                 "leakage_analysis.json", f"{k}.json"):
        f = RESULTS / name
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        walk(d.get(k, d) if name != f"{k}.json" else d)
    # cho phép sai số làm tròn ở chữ số thứ ba
    out = set()
    for v in vals:
        for delta in (-0.001, 0.0, 0.001):
            out.add(round(v + delta, 3))
    return out


def _c1_rates(text: str, seedvar: dict, primary: list[str]) -> None:
    """Mỗi số tỷ lệ gán cho một mô hình phải có trong kết quả đã tính."""
    print("\nC1b — số tỷ lệ gán cho mô hình so với kết quả đã tính")
    for k in primary:
        known = _known_values(k)
        if not known:
            warn(f"{MODEL_LABELS[k]}: không nạp được kết quả để đối chiếu")
            continue
        label = re.escape(MODEL_LABELS[k])
        found, unknown = set(), set()
        for pat in (rf"(0\.[0-9]{{2,3}})\s*±\s*0\.[0-9]+\s+for\s+{label}\b",
                    rf"(?<!%)(?<!% )(0\.[0-9]{{2,3}})\s+for\s+{label}\b"):
            for m in re.finditer(pat, text):
                if text[max(0, m.start() - 2):m.start()].strip().endswith("%"):
                    continue
                v = round(float(m.group(1)), 3)
                found.add(v)
                # giá trị in 2 chữ số là bản làm tròn của một giá trị 3 chữ số
                if not (v in known or any(abs(v - w) < 0.005 for w in known)):
                    unknown.add(v)
        if unknown:
            bad(f"{MODEL_LABELS[k]}: {sorted(unknown)} không khớp giá trị nào đã tính")
        elif found:
            ok(f"{MODEL_LABELS[k]}: {len(found)} giá trị tỷ lệ, tất cả khớp kết quả đã tính")
        else:
            warn(f"{MODEL_LABELS[k]}: không tìm thấy giá trị tỷ lệ nào gắn tên mô hình")


def _c1_ap50(text: str, seedvar: dict, primary: list[str]) -> None:
    """AP50 chính của mỗi mô hình phải có mặt, và mọi số cỡ phần trăm gắn tên mô
    hình phải tồn tại trong kết quả đã tính.

    Bản trước đòi mỗi mô hình chỉ xuất hiện với MỘT giá trị cỡ phần trăm. Từ khi
    bài báo cáo cả AP, AP50 và AP75 cạnh nhau, đòi hỏi đó sai: ba giá trị khác
    nhau cho cùng một mô hình là đúng, miễn mỗi giá trị có nguồn.
    """
    for k in primary:
        want = seed_mean(seedvar, k, "ap50")
        if want is None:
            continue
        known = _known_values(k)
        label = re.escape(MODEL_LABELS[k])
        found = set()
        for pat in (rf"([0-9]+\.[0-9]+)%?\s*(?:±[^)]*)?\s*for\s+{label}",
                    rf"{label}\s*\(([0-9]+\.[0-9]+)",
                    rf"{label}\s+([0-9]+\.[0-9]+)%"):
            for m in re.finditer(pat, text):
                v = float(m.group(1))
                if 10 < v < 100:
                    found.add(round(v, 2))
        if not found:
            warn(f"{MODEL_LABELS[k]}: không tìm thấy số cỡ phần trăm nào trong bài")
            continue
        unknown = sorted(v for v in found
                         if not any(abs(v - w) < 0.06 for w in known))
        if unknown:
            bad(f"{MODEL_LABELS[k]}: {unknown} không khớp giá trị nào đã tính")
        elif not any(abs(v - want) < 0.15 for v in found):
            bad(f"{MODEL_LABELS[k]}: AP50 đo được {want:.2f} không xuất hiện trong bài "
                f"(các số tìm thấy: {sorted(found)})")
        else:
            ok(f"{MODEL_LABELS[k]}: AP50 {want:.2f} có trong bài; "
               f"{len(found)} số cỡ phần trăm đều khớp kết quả đã tính")


def c3_tables(text: str, xml: bytes) -> None:
    """Chữ hứa Bảng N có gì thì bảng N phải có đúng thứ đó."""
    import html as _h
    print("\nC3 — tham chiếu 'Table N' so với cấu trúc bảng thật")
    x = xml.decode()
    tables = []
    for t in re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S):
        rows = re.findall(r"<w:tr\b.*?</w:tr>", t, re.S)
        if not rows:
            continue
        hdr = [_h.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c)))
               for c in re.findall(r"<w:tc>.*?</w:tc>", rows[0], re.S)]
        tables.append(hdr)
    # bảng bố cục ảnh không có tiêu đề kiểu bảng số liệu — bỏ qua
    # Nhận bảng số liệu theo "ô tiêu đề đầu tiên có chữ" thay vì theo danh sách
    # từ khoá cứng — bảng theo buổi chụp mở đầu bằng "Capture session" nên danh
    # sách cũ bỏ sót nó và báo thiếu một bảng.
    data_tables = [h for h in tables if h and h[0].strip()]
    ok(f"tìm thấy {len(data_tables)} bảng số liệu")

    # Mọi "Table N" được nhắc trong chữ phải có bảng tương ứng. Bản đầu không
    # kiểm điều này nên bỏ lọt việc bài viện dẫn Table 4 mà bảng đó không tồn tại.
    refs = sorted({int(m.group(1)) for m in re.finditer(r"Table (\d)", text)})
    caps = sorted({int(m.group(1)) for m in re.finditer(RULES["table_cap"], text)})
    if refs:
        ok(f"chữ viện dẫn Table {refs}")
    missing_tbl = [n for n in refs if n > len(data_tables)]
    if missing_tbl:
        bad(f"viện dẫn Table {missing_tbl} nhưng chỉ có {len(data_tables)} bảng số liệu")
    else:
        ok(f"mọi Table được viện dẫn đều tồn tại ({len(data_tables)} bảng)")
    missing_cap = [n for n in refs if n not in caps]
    if missing_cap:
        bad(f"Table {missing_cap} được viện dẫn nhưng không có chú thích")
    else:
        ok("mọi Table được viện dẫn đều có chú thích")

    claims = RULES["claims"]
    for name, need in claims.items():
        if name not in text:
            continue
        idx = int(name.split()[1]) - 1
        if idx >= len(data_tables):
            bad(f"{name} được nhắc trong chữ nhưng không có bảng tương ứng")
            continue
        hdr = data_tables[idx]
        missing = [c for c in need if not any(c.lower() in h.lower() for h in hdr)]
        if missing:
            bad(f"{name}: chữ hứa {need} nhưng bảng thiếu cột {missing} — có {hdr}")
        else:
            ok(f"{name}: cột khớp lời hứa trong chữ ({len(hdr)} cột)")


def c7_caption_scope(text: str, xml: bytes) -> None:
    """Chú thích bảng không được ngụ ý nhiều mô hình hơn số hàng thật."""
    import html as _h
    print("\nC7 — chú thích bảng so với số hàng")
    x = xml.decode()
    tbls = []
    for t in re.findall(r"<w:tbl>.*?</w:tbl>", x, re.S):
        rows = re.findall(r"<w:tr\b.*?</w:tr>", t, re.S)
        if not rows:
            continue
        hdr = [_h.unescape("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c)))
               for c in re.findall(r"<w:tc>.*?</w:tc>", rows[0], re.S)]
        if hdr and hdr[0].strip():
            tbls.append(len(rows) - 1)
    for m in re.finditer(r"(Table \d)\. ([^.]{0,110})", text):
        name, cap = m.group(1), m.group(2)
        n = int(name.split()[1])
        if n - 1 >= len(tbls):
            continue
        n_rows = tbls[n - 1]
        vague = any(w in cap.lower() for w in ("the evaluated models", "all models",
                                              "five models"))
        if vague and n_rows < 5:
            bad(f"{name}: chú thích nói {cap.strip()!r} nhưng bảng chỉ có {n_rows} hàng")
        else:
            ok(f"{name}: chú thích khớp {n_rows} hàng")


def c8_figures(text: str) -> None:
    """Mọi hình phải có chú thích VÀ được viện dẫn trong thân bài.

    Tạp chí trả lại bản thảo có hình không được nhắc tới, và người đọc không
    biết nhìn hình lúc nào. Kiểm cả hai chiều: chú thích không có viện dẫn, và
    viện dẫn không có chú thích.
    """
    print("\nC8 — hình: chú thích và viện dẫn")
    # Chú thích nhận theo VỊ TRÍ đầu dòng, không theo dấu chấm: một viện dẫn
    # cuối câu ("shown in Figure 3.") cũng có dấu chấm ngay sau, nên mẫu cũ
    # loại nhầm nó và báo hình không được viện dẫn.
    cap_spans, caps = [], set()
    for m in re.finditer(RULES["fig_cap"], text, re.M):
        caps.add(int(m.group(1))); cap_spans.append(m.span())
    refs = set()
    for m in re.finditer(re.escape(RULES["fig_ref"]) + r" ?(\d)", text):
        if not any(a <= m.start() < b for a, b in cap_spans):
            refs.add(int(m.group(1)))
    if not caps and not refs:
        warn("bài không có hình nào"); return
    no_ref = sorted(caps - refs)
    no_cap = sorted(refs - caps)
    if no_ref:
        bad(f"Figure {no_ref} có chú thích nhưng không được viện dẫn trong thân bài")
    if no_cap:
        bad(f"Figure {no_cap} được viện dẫn nhưng không có chú thích")
    if not no_ref and not no_cap:
        ok(f"{len(caps)} hình, đều có chú thích và được viện dẫn")


def c9_images_embedded(docx: pathlib.Path, text: str) -> None:
    """Hình phải NHÚNG thật trong file, không chỉ có chú thích.

    C8 kiểm chú thích và viện dẫn, và nó báo xanh trong khi file .docx không có
    một ảnh nào — LibreOffice liên kết ảnh thay vì nhúng. Chỉ nhìn PDF mới thấy.
    """
    print("\nC9 — ảnh có thật trong file")
    try:
        media = [n for n in zipfile.ZipFile(docx).namelist() if n.startswith("word/media/")]
    except Exception as e:
        warn(f"không đọc được media: {e}"); return
    n_cap = len(re.findall(r"(?m)^Figure \d\. ", text))
    if n_cap and not media:
        bad(f"{n_cap} chú thích hình nhưng KHÔNG có ảnh nào nhúng trong file")
    elif len(media) < n_cap:
        bad(f"{n_cap} chú thích hình nhưng chỉ {len(media)} ảnh nhúng")
    else:
        ok(f"{len(media)} ảnh nhúng cho {n_cap} hình")


def c10_figure_order(text: str) -> None:
    """Số hình phải tăng dần theo thứ tự xuất hiện trong bài."""
    print("\nC10 — thứ tự đánh số hình")
    nums = [int(m.group(1)) for m in re.finditer(RULES["fig_cap"], text, re.M)]
    if nums != sorted(nums):
        bad(f"hình xuất hiện theo thứ tự {nums} — phải tăng dần")
    elif nums != list(range(1, len(nums) + 1)):
        bad(f"số hình không liên tục: {nums}")
    else:
        ok(f"{len(nums)} hình, đánh số liên tục theo thứ tự xuất hiện")


def c4_placeholders(text: str) -> None:
    print("\nC4 — ô giữ chỗ còn sót")
    # Mẫu phải bắt ô giữ chỗ dù cụm "to be confirmed" nằm ở GIỮA dấu ngoặc —
    # bản đầu chỉ bắt khi nó đứng ngay sau '[', nên bỏ lọt chỗ đã đánh dấu cố ý.
    pats = [r"\[[^\]]*to be confirmed[^\]]*\]", r"\[[^\]]*insert[^\]]*\]",
            r"\[[^\]]*TO BE COMPLETED[^\]]*\]",
            r"\[model\(s\)[^\]]*\]", r"\[add [^\]]*\]",
            r"\bXXX\b", r"\bTODO\b", r"\[[^\]]*TBD[^\]]*\]"]
    hits = [m.group(0) for p in pats for m in re.finditer(p, text, re.I)]
    allowed = [h for h in hits if "scientific name" in h.lower()]
    # Hai tuyên bố bắt buộc của tạp chí (đóng góp tác giả, kinh phí) không suy
    # ra được từ dữ liệu. Chúng là việc của nhóm tác giả, nên báo riêng thành
    # CẢNH BÁO thay vì LỖI — nhưng vẫn phải hiện ở mỗi lần kiểm.
    author = [h for h in hits
              if "to be completed by the authors" in h.lower()
              or "to be confirmed or amended by the authors" in h.lower()]
    real = [h for h in hits if h not in allowed and h not in author]
    if real:
        bad(f"còn {len(real)} ô giữ chỗ bỏ quên: {real[:3]}")
    else:
        ok("không còn ô giữ chỗ bỏ quên"
           + (f" ({len(allowed)} chỗ cố ý đánh dấu)" if allowed else ""))
    if author:
        warn(f"{len(author)} tuyên bố chờ nhóm tác giả điền hoặc xác nhận "
             "(ORCID / lời cảm ơn / kinh phí / đóng góp tác giả / khai dùng AI) — "
             "không nộp bài khi chúng còn đây")


def _entry_surname(line: str) -> str:
    """Họ của tác giả đầu trong một mục tài liệu, kể cả họ nhiều chữ.

    Hai kiểu trình bày phải cùng cho ra một kết quả:
        RAE:      "de Souza M.F., do Amaral L.R., … (2020): …"
        Springer: "de Souza, M.F., L.R. do Amaral, … 2020. …"
    Lấy các từ đầu cho tới khi gặp chữ viết tắt ("M.F.") hoặc dấu phẩy.
    """
    out = []
    for tok in line.replace(",", " , ").split():
        if tok == ",":
            break
        if re.fullmatch(r"[A-ZÀ-Ý]\.(?:-?[A-ZÀ-Ý]\.)*", tok):
            break
        out.append(tok)
        if len(out) >= 4:
            break
    return " ".join(out) or line[:20]


def c13_supplementary(text: str, sup_path) -> None:
    """Mọi bảng và hình phụ lục phải được dẫn trong bài, và ngược lại.

    Hướng dẫn của Sugar Tech: "the text must make specific mention of the
    material as a citation". Một bảng phụ lục không ai dẫn tới là một bảng
    không ai đọc.
    """
    print("\nC13 — phụ lục so với phần dẫn trong bài")
    import pathlib as _p
    f = _p.Path(sup_path)
    if not f.exists():
        warn(f"không thấy {f.name} — bỏ qua"); return
    sup = f.read_text(encoding="utf-8")
    have_t = {int(m) for m in re.findall(r"\*\*Supplementary Table S(\d+)\.", sup)}
    have_f = {int(m) for m in re.findall(r"Supplementary Figure S(\d+)\.", sup)}
    # Phần dẫn có thể viết số nhiều: "Supplementary Tables S4 and S5"
    cited_t, cited_f = set(), set()
    for m in re.finditer(r"Supplementary Tables? (S\d+(?:[,\s]+(?:and\s+)?S\d+)*)", text):
        cited_t |= {int(x) for x in re.findall(r"S(\d+)", m.group(1))}
    for m in re.finditer(r"Supplementary Figures? (S\d+(?:[,\s]+(?:and\s+)?S\d+)*)", text):
        cited_f |= {int(x) for x in re.findall(r"S(\d+)", m.group(1))}
    for kind, have, cited in (("bảng", have_t, cited_t), ("hình", have_f, cited_f)):
        if not have:
            continue
        orphan = sorted(have - cited)
        ghost = sorted(cited - have)
        if orphan:
            bad(f"{kind} phụ lục có nhưng không được dẫn: {['S%d' % i for i in orphan]}")
        if ghost:
            bad(f"{kind} phụ lục được dẫn nhưng không tồn tại: {['S%d' % i for i in ghost]}")
        if not orphan and not ghost:
            ok(f"{len(have)} {kind} phụ lục đều khớp hai chiều")


def c11_citations(text: str) -> None:
    """Mọi trích dẫn trong chữ phải có mục tương ứng ở danh mục, và ngược lại."""
    print("\nC11 — trích dẫn so với danh mục tài liệu")
    mark = next((m for m in ("## REFERENCES", "## References", "REFERENCES",
                             "References") if m in text), None)
    if not mark:
        warn("không thấy mục REFERENCES"); return
    body, refs = text.rsplit(mark, 1)
    cited = set()
    # Họ có thể nhiều chữ ("de Souza", "dos Santos Ferreira") và một ngoặc đơn
    # có thể chứa nhiều trích dẫn ngăn bằng dấu chấm phẩy. Bản trước chỉ lấy từ
    # đầu của họ và chỉ lấy trích dẫn đầu trong ngoặc, nên bỏ sót cả hai dạng.
    part_ = r"(?:de|del|dos|das|da|du|van|von|der|den|la|le|el|bin|ter)"
    sur = rf"(?:{part_} )?[A-ZÀ-Ý][\w'’\-]+(?:(?: {part_})? [A-ZÀ-Ý][\w'’\-]+)?"
    one = rf"({sur})(?: et al\.| and (?:{sur}))?[, ]+(\d{{4}}[ab]?)"
    for grp in re.finditer(r"\(([^()]{4,400}?)\)", body):
        for part in grp.group(1).split(";"):
            m = re.fullmatch(rf"\s*{one}\s*", part)
            if m:
                cited.add((m.group(1), m.group(2)))
    for m in re.finditer(rf"\b({sur})(?: et al\.| and (?:{sur}))? \((\d{{4}}[ab]?)\)", body):
        cited.add((m.group(1), m.group(2)))
    pat = RULES["ref_entry"]
    missing = [f"{a} {b}" for a, b in sorted(cited)
               if not re.search(pat.format(sur=re.escape(a), yr=b), refs)]
    if missing:
        bad(f"{len(missing)} trích dẫn không có trong danh mục: {missing[:4]}")
    else:
        ok(f"{len(cited)} trích dẫn đều có mục tương ứng")
    # chiều ngược lại: mục không được viện dẫn
    unused = []
    for line in refs.split("\n"):
        m = re.match(RULES["ref_line"], line.strip())
        if not m:
            continue
        full = _entry_surname(line.strip())
        if (m.group(1), m.group(2)) in cited or (full, m.group(2)) in cited:
            continue
        unused.append(f"{full} {m.group(2)}")
    if unused:
        warn(f"{len(unused)} mục không được viện dẫn: {unused[:4]}")
    else:
        ok("mọi mục trong danh mục đều được viện dẫn")


def c12_journal_rules(text: str) -> None:
    """Quy định hình thức của tạp chí đang nhắm tới (xem JOURNAL_RULES)."""
    R = RULES
    print(f"\nC12 — quy định hình thức của {R['name']}")
    lines = text.split("\n")
    title = next((l.lstrip("# ").strip() for l in lines if l.strip()), "")
    if R["title_max"] and len(title) > R["title_max"]:
        bad(f"tiêu đề {len(title)} ký tự (giới hạn {R['title_max']})")
    elif R["title_no_colon"] and ":" in title:
        bad("tiêu đề có phụ đề sau dấu hai chấm — tạp chí không cho phép")
    else:
        ok(f"tiêu đề {len(title)} ký tự")

    m = re.search(r"\*\*Abstract:?\*\*\s*(.+)|\bAbstract:?\s+(.+)", text)
    if m:
        n = len((m.group(1) or m.group(2)).split())
        lo, hi = R["abstract"]
        rng = ("không giới hạn" if lo is None and hi is None
               else f"yêu cầu {lo or 0}-{hi}" if hi else f"yêu cầu từ {lo}")
        if (lo and n < lo) or (hi and n > hi):
            bad(f"abstract {n} từ ({rng})")
        else:
            ok(f"abstract {n} từ ({rng})")

    m = re.search(r"\*\*Keywords:?\*\*\s*(.+)|\bKeywords:?\s+(.+)", text)
    if m:
        kw = (m.group(1) or m.group(2)).strip().rstrip(".")
        parts = [k.strip() for k in kw.split(";") if k.strip()]
        lo, hi = R["keywords"]
        if lo is not None and hi is not None and not lo <= len(parts) <= hi:
            bad(f"{len(parts)} từ khoá (yêu cầu {lo}-{hi})")
        elif R["keywords_lower"] and kw != kw.lower():
            bad("từ khoá phải viết thường")
        else:
            ok(f"{len(parts)} từ khoá")

    mark = next((x for x in ("## References", "## REFERENCES", "REFERENCES", "References")
                 if x in text), None)
    if mark and R["refs_max"]:
        refs = text.rsplit(mark, 1)[1]
        n = len([l for l in refs.split("\n")
                 if re.match(r"^[A-ZÀ-Ýa-z][\w'’\-]+ [A-Z]", l.strip())])
        (ok if n <= R["refs_max"] else bad)(
            f"{n} tài liệu tham khảo (giới hạn {R['refs_max']})")
        warn("tối đa 2 tự trích dẫn — chỉ nhóm tác giả xác nhận được")

    if R["pages_hard"] is None:
        ok(f"không giới hạn số trang ({R['spacing']})"); return
    pdf = pathlib.Path(R["pdf"])
    if not pdf.exists():
        warn(f"chưa render {pdf} nên không đếm được số trang"); return
    try:
        import subprocess
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))
    except Exception as e:
        warn(f"không đếm được số trang ({type(e).__name__})"); return
    if pages > R["pages_hard"]:
        bad(f"{pages} trang ({R['spacing']}) — vượt mức {R['pages_hard']} trang")
    elif pages > R["pages_max"]:
        warn(f"{pages} trang ({R['spacing']}) — trên giới hạn {R['pages_max']}; "
             "thư gửi Editor phải giải trình phần vượt")
    else:
        ok(f"{pages} trang ({R['spacing']}), trong giới hạn {R['pages_max']}")


def c5_contradictions(text: str) -> None:
    """Các cặp phát biểu không thể cùng đúng."""
    print("\nC5 — cặp phát biểu mâu thuẫn")
    pairs = [
        ("trained once, from a single random seed", "from different random seeds",
         "số lần huấn luyện"),
        ("accurately and effectively detect weeds", "went undetected",
         "khẳng định độ chính xác so với tỷ lệ bỏ sót"),
        ("Basis of comparison below", "Scope of the comparison",
         "tham chiếu sai hướng"),
    ]
    bad_found = False
    for a, b, what in pairs:
        if a in text and b in text:
            bad(f"mâu thuẫn ({what}): cùng có '{a[:40]}...' và '{b[:40]}...'")
            bad_found = True
    if not bad_found:
        ok("không thấy cặp mâu thuẫn nào trong danh sách kiểm")


def c6_abstract(text: str, limit: int | None, no_limit: bool = False) -> None:
    print("\nC6 — độ dài Abstract")
    m1 = re.search(r"\bAbstract:?\s", text)
    m2 = re.search(r"\bKeywords:?\s", text)
    i, j = (m1.end() - 1 if m1 else -1), (m2.start() if m2 else -1)
    if i < 0 or j < 0:
        warn("không tách được Abstract"); return
    n = len(text[i:j].split())
    if limit and n > limit:
        bad(f"Abstract {n} từ, vượt giới hạn {limit}")
    elif limit:
        ok(f"Abstract {n} từ, trong giới hạn {limit}")
    elif no_limit:
        ok(f"Abstract {n} từ — tạp chí không đặt giới hạn số từ")
    else:
        warn(f"Abstract {n} từ — chưa biết giới hạn của tạp chí, nhóm tác giả cần kiểm")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", required=True)
    ap.add_argument("--skill-dir", default=None, help="thư mục skill docx (có accept_changes.py)")
    # Giới hạn 200 từ lấy từ "Instructions for authors" của tạp chí, mục ii.
    ap.add_argument("--abstract-limit", type=int, default=None,
                    help="ghi đè giới hạn abstract; mặc định lấy theo --journal")
    ap.add_argument("--journal", choices=sorted(JOURNAL_RULES), default="rae",
                    help="bộ luật hình thức áp dụng (mặc định rae)")
    a = ap.parse_args()

    global RULES
    RULES = JOURNAL_RULES[a.journal]
    print(f"Bộ luật hình thức: {RULES['name']}")

    cfg = load_cfg()
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    docx_path = pathlib.Path(a.docx)
    text, xml = accepted_text(docx_path,
                              pathlib.Path(a.skill_dir) if a.skill_dir else None)

    print(f"Kiểm tra: {a.docx}")
    c1_c2_numbers(text, seedvar, cfg)
    c3_tables(text, xml)
    c7_caption_scope(text, xml)
    c8_figures(text)
    c9_images_embedded(docx_path, text)
    c10_figure_order(text)
    c4_placeholders(text)
    c5_contradictions(text)
    c6_abstract(text, a.abstract_limit or RULES["abstract"][1],
                no_limit=bool(RULES.get("abstract_no_limit")))
    c11_citations(text)
    c12_journal_rules(text)
    c13_supplementary(text, docx_path.parent / f"{_PREFIX[a.journal]}_supplementary.md")

    print(f"\n{'='*60}")
    print(f"LỖI: {len(FAIL)}   CẢNH BÁO: {len(WARN)}")
    for f in FAIL:
        print(f"  ✗ {f}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
