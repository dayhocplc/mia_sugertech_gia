#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuyển danh mục tài liệu giữa hai kiểu trình bày của hai tạp chí.

Research in Agricultural Engineering dùng:
    Ren S., He K., Girshick R., Sun J. (2017): Faster R-CNN: Towards ...
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 39: 1137–1149.

Sugar Tech (Springer) in ra dạng khác — kiểm trên một bài đã đăng của chính
tạp chí (doi 10.1007/s12355-025-01605-2):
    Esteban, D.A.A., Z.M. de Souza, C.A. Tormena, ... 2019. Soil compaction, ...
    Geoderma 347: 1-10.

Khác biệt: tác giả đầu đảo "Họ, Chữ-viết-tắt"; các tác giả sau để chữ viết tắt
trước họ; trước tác giả cuối thêm "and"; năm đứng sau danh sách tác giả, không
ngoặc đơn; sau năm là dấu chấm; tên tạp chí in nghiêng và KHÔNG có dấu phẩy
trước số tập.

Hàm ở đây làm việc chuyển đổi đó một cách máy móc và báo lại những mục không
khớp mẫu, để không có mục nào bị đổi sai mà im lặng.
"""

from __future__ import annotations

import re

# "Ren S., He K., Girshick R., Sun J. (2017): phần còn lại"
_RE_ENTRY = re.compile(r"^(?P<authors>.+?)\s*\((?P<year>\d{4}[ab]?)\):\s*(?P<rest>.+)$",
                       re.S)
# một tác giả: "Ren S." / "de Souza M.F." / "Lin T.-Y." / "NVIDIA Corporation"
_RE_AUTHOR = re.compile(r"^(?P<sur>.+?)\s+(?P<ini>(?:[A-ZÀ-Ý]\.(?:-[A-ZÀ-Ý]\.)*)+)$")


def _springer_authors(block: str) -> str | None:
    """Đổi danh sách tác giả sang thứ tự Springer, hoặc None nếu không nhận ra."""
    parts = [p.strip() for p in block.split(",") if p.strip()]
    out = []
    for i, p in enumerate(parts):
        m = _RE_AUTHOR.match(p)
        if not m:
            # tổ chức làm tác giả (ví dụ "NVIDIA Corporation") — giữ nguyên
            if i == 0 and len(parts) == 1:
                return p
            return None
        sur, ini = m.group("sur"), m.group("ini")
        out.append(f"{sur}, {ini}" if i == 0 else f"{ini} {sur}")
    if len(out) == 1:
        return out[0]
    if len(out) == 2:
        return f"{out[0]}, and {out[1]}"
    return ", ".join(out[:-1]) + f", and {out[-1]}"


def _springer_rest(rest: str) -> str:
    """Chuẩn hoá phần sau năm.

    Hai việc: bỏ dấu phẩy trước số tập ('Journal, 39: 1137-1149.' → 'Journal 39:
    …'), và in NGHIÊNG tên tạp chí như hướng dẫn yêu cầu. Tên tạp chí là đoạn
    nằm giữa dấu chấm kết thúc nhan đề bài và số tập; nếu không nhận được dạng
    đó thì để nguyên chứ không đoán.
    """
    rest = re.sub(r",\s*(\d+(?:\(\d+\))?):", r" \1:", rest)
    m = re.search(r"\.\s+([A-Z][^.]{2,80}?)\s+(\d+(?:\(\d+\))?):", rest)
    if m:
        return rest[:m.start(1)] + f"*{m.group(1)}*" + rest[m.end(1):]
    # Kỷ yếu hội nghị: hướng dẫn yêu cầu in nghiêng cả tên sách, mà mục kỷ yếu
    # không có số tập nên mẫu trên bỏ qua. Bắt riêng dạng "Proceedings of ...".
    m = re.search(r"(In: )?(Proceedings of [^,]+?)(,|\.)", rest)
    if m:
        return rest[:m.start(2)] + f"*{m.group(2)}*" + rest[m.end(2):]
    # Chương sách: "In: Biên-tập-viên (eds): Tên sách. Nhà xuất bản, Nơi."
    m = re.search(r"\(eds?\):?\s+([^.]{4,100}?)\.", rest)
    if m:
        return rest[:m.start(1)] + f"*{m.group(1)}*" + rest[m.end(1):]
    return rest


def to_springer(refs: list[str]) -> tuple[list[str], list[str]]:
    """Trả (danh mục đã chuyển, danh sách mục không chuyển được).

    Mục không khớp mẫu được giữ NGUYÊN VĂN và báo ra, chứ không đoán bừa.
    """
    done, failed = [], []
    for r in refs:
        one = " ".join(r.split())
        m = _RE_ENTRY.match(one)
        if not m:
            done.append(one); failed.append(one[:70]); continue
        auth = _springer_authors(m.group("authors"))
        if auth is None:
            done.append(one); failed.append(one[:70]); continue
        done.append(f"{auth}. {m.group('year')}. {_springer_rest(m.group('rest'))}")
    return sorted(done, key=str.lower), failed



# ---------------------------------------------------------------- Elsevier
# Artificial Intelligence in Agriculture (KeAi/Elsevier) dùng kiểu author-date
# của Elsevier. Mẫu trong Guide for Authors:
#     Van der Geer, J., Hanraads, J.A.J., Lupton, R.A., 2010. The art of writing
#     a scientific article. J. Sci. Commun. 163, 51-59.
# Khác Springer ở bốn chỗ: MỌI tác giả đều "Họ, Chữ-viết-tắt"; không có "and"
# trước tác giả cuối; năm là một trường riêng sau danh sách tác giả; tên tạp chí
# viết tắt theo LTWA, không in nghiêng, không có dấu phẩy trước số tập mà có dấu
# phẩy giữa số tập và số trang.

# Viết tắt theo LTWA cho đúng những tạp chí mà bài này trích. Chỉ liệt kê tên
# nào tra được chắc chắn; tên không có trong bảng được giữ nguyên và báo ra, vì
# một chữ viết tắt bịa ra còn tệ hơn một tên đầy đủ.
_ABBREV = {
    "Computers and Electronics in Agriculture": "Comput. Electron. Agric.",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence":
        "IEEE Trans. Pattern Anal. Mach. Intell.",
    "Biosystems Engineering": "Biosyst. Eng.",
    "Crop Protection": "Crop Prot.",
    "Frontiers in Plant Science": "Front. Plant Sci.",
    "Pakistan Journal of Agricultural Sciences": "Pak. J. Agric. Sci.",
    "Agricultural Reviews": "Agric. Rev.",
    "Advances in Neural Information Processing Systems":
        "Adv. Neural Inf. Process. Syst.",
    # Những tên này LTWA không rút ngắn (một từ, hoặc đã là dạng ngắn)
    "IEEE Access": "IEEE Access",
    "Agriculture": "Agriculture",
    "Sugar Tech": "Sugar Tech",
}


def _elsevier_authors(block: str) -> str | None:
    """Mọi tác giả thành "Họ, I.N."; trả None nếu không nhận ra dạng."""
    parts = [p.strip() for p in block.split(",") if p.strip()]
    out = []
    for i, p in enumerate(parts):
        m = _RE_AUTHOR.match(p)
        if not m:
            if i == 0 and len(parts) == 1:
                return p                      # tổ chức làm tác giả
            return None
        out.append(f"{m.group('sur')}, {m.group('ini')}")
    return ", ".join(out)


def _abbrev_journal(rest: str, unknown: list[str]) -> str:
    """Viết tắt tên tạp chí nằm giữa dấu chấm kết nhan đề và số tập."""
    m = re.search(r"\.\s+([A-Z][^.]{2,80}?)\s+(\d+(?:\(\d+\))?),", rest)
    if not m:
        return rest
    name = re.sub(r"^In:\s*", "", m.group(1))
    if name.startswith(("Proceedings of", "Advances in Neural")):
        return rest                  # kỷ yếu, không phải tên tạp chí
    short = _ABBREV.get(name)
    if short is None:
        unknown.append(name)
        return rest
    return rest[:m.start(1)] + short + rest[m.end(1):]


def _elsevier_rest(rest: str, unknown: list[str]) -> str:
    """Chuẩn hoá phần sau năm sang kiểu Elsevier.

    Thứ tự các phép thay rất quan trọng: phải xử lý ngày hội nghị TRƯỚC, vì quy
    tắc "số tập: số trang" của tạp chí sẽ ăn mất cụm "…, 2020: 558-573" của một
    mục kỷ yếu và biến năm hội nghị thành số tập.
    """
    # 1. kỷ yếu có ngày: "…, Glasgow, Aug 23-28, 2020: 558-573." → "…, Glasgow, pp. 558-573."
    rest = re.sub(r",\s*(?:Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sep|Oct|Nov|Dec)"
                  r"[^,:]*,\s*\d{4}:\s*([\d\u2013\u2014-]+)\.",
                  r", pp. \1.", rest)
    # 2. kỷ yếu không có số trang: "…, Toulon, Apr 24-26, 2017." → "…, Toulon."
    rest = re.sub(r",\s*(?:Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sep|Oct|Nov|Dec)"
                  r"[^,:]*,\s*\d{4}\.\s*$", ".", rest)
    # 3. kỷ yếu không có ngày: "(ECCV): 740-755." → "(ECCV), pp. 740-755."
    rest = re.sub(r"\):\s*([\d\u2013\u2014-]+)\.", r"), pp. \1.", rest)
    rest = re.sub(r"([A-Za-z\)]),?\s*:\s*(\d+[\u2013\u2014-]\d+)\.",
                  r"\1, pp. \2.", rest)
    # 4. tạp chí: ", 39: 1137-1149." → " 39, 1137-1149." rồi viết tắt tên tạp chí
    rest = re.sub(r",\s*(\d+(?:\(\d+\))?):\s*", r" \1, ", rest)
    rest = _abbrev_journal(rest, unknown)
    # 5. "In:" của kiểu RAE thành "in:" của Elsevier; mục kỷ yếu nào thiếu hẳn
    #    tiền tố thì thêm, vì Elsevier yêu cầu "in: Proceedings of …"
    rest = re.sub(r"\bIn:\s+", "in: ", rest)
    rest = re.sub(r"(?<!in: )\b(Proceedings of )", r"in: \1", rest, count=1)
    rest = re.sub(r"in: in: ", "in: ", rest)
    # 6. biên tập viên: "(eds): " → "(Eds.), " và dấu phẩy sau họ của họ
    m = re.search(r"in:\s+(.+?)\s*\(eds?\):?", rest)
    if m:
        eds = _elsevier_authors(m.group(1).replace(".,", ".,")) or m.group(1)
        rest = rest[:m.start(1)] + eds + " (Eds.)," + rest[m.end():]
    rest = re.sub(r"\(eds?\):?", "(Eds.),", rest)
    # 7. ngày truy cập theo kiểu Elsevier
    rest = re.sub(r"\(accessed (\w+) (\d+), (\d{4})\)",
                  lambda m: f"(accessed {m.group(2)} {_MONTH.get(m.group(1), m.group(1))} "
                            f"{m.group(3)})", rest)
    rest = rest.replace("Available at ", "")
    return " ".join(rest.split())


_MONTH = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
          "May": "May", "Jun": "June", "June": "June", "Jul": "July",
          "July": "July", "Aug": "August", "Sep": "September", "Oct": "October",
          "Nov": "November", "Dec": "December"}


def to_elsevier(refs: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Trả (danh mục kiểu Elsevier, mục không chuyển được, tên tạp chí chưa có viết tắt).

    Mục không khớp mẫu được giữ NGUYÊN VĂN và báo ra. Thứ tự: theo chữ cái rồi
    theo năm, như hướng dẫn của Elsevier yêu cầu.
    """
    done, failed, unknown = [], [], []
    for r in refs:
        one = " ".join(r.split())
        m = _RE_ENTRY.match(one)
        if not m:
            done.append(one); failed.append(one[:70]); continue
        auth = _elsevier_authors(m.group("authors"))
        if auth is None:
            done.append(one); failed.append(one[:70]); continue
        done.append(f"{auth}, {m.group('year')}. "
                    f"{_elsevier_rest(m.group('rest'), unknown)}")
    return (sorted(done, key=_sort_key), failed, sorted(set(unknown)))


def _sort_key(entry: str) -> tuple:
    """Theo chữ cái của họ tác giả đầu, rồi theo năm."""
    m = re.search(r",\s*(\d{4})([ab]?)\.", entry)
    year = int(m.group(1)) if m else 0
    suffix = m.group(2) if m else ""
    return (entry.split(",")[0].lower(), year, suffix, entry.lower())

if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from build_manuscript import REFERENCES
    style = sys.argv[1] if len(sys.argv) > 1 else "springer"
    if style == "elsevier":
        out, bad, unk = to_elsevier(REFERENCES)
    else:
        out, bad = to_springer(REFERENCES)
        unk = []
    for i, r in enumerate(out, 1):
        print(f"{i:2d}. {r}")
    print(f"\n{len(out)} mục; {len(bad)} mục giữ nguyên văn vì không khớp mẫu:")
    for b in bad:
        print("   !", b)
    if unk:
        print(f"{len(unk)} tên tạp chí chưa có viết tắt trong bảng (giữ tên đầy đủ):")
        for u in unk:
            print("   ?", u)
