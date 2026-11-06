#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Máy sửa file .docx bằng Track Changes, không phụ thuộc python-docx.

Chỉ đụng vào `word/document.xml`. Mọi thay đổi văn bản đều được bọc trong
`<w:del>` + `<w:ins>` để đồng tác giả thấy đúng chỗ nào bị sửa và có thể
Accept/Reject từng cái trong Word.

Dùng qua ba hàm:
    doc = Document(path)
    doc.replace_run_text(old, new, ...)      # sửa một đoạn chữ
    doc.set_table_cell(t, r, c, new)         # sửa một ô bảng
    doc.insert_paragraph_after(anchor, text) # chèn đoạn mới
    doc.save(out_path)
"""

from __future__ import annotations

import datetime
import html
import pathlib
import re
import shutil
import subprocess
import tempfile

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TR_RE = re.compile(r"<w:tr\b.*?</w:tr>", re.S)
TC_RE = re.compile(r"<w:tc>.*?</w:tc>", re.S)
TBL_RE = re.compile(r"<w:tbl>.*?</w:tbl>", re.S)
TEXT_RE = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")


def esc(s: str) -> str:
    return html.escape(s, quote=False)


class Document:
    def __init__(self, path: str | pathlib.Path, author: str = "Claude (repro pipeline)",
                 merge_runs_script: str | pathlib.Path | None = None):
        self.src = pathlib.Path(path)
        self.author = author
        self.date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="docx_edit_"))
        self.unpacked = self.tmp / "unpacked"
        subprocess.run(["unzip", "-q", str(self.src), "-d", str(self.unpacked)], check=True)
        for p in self.unpacked.rglob("*"):
            if p.is_symlink():
                p.unlink()
        if merge_runs_script:
            subprocess.run(["python3", str(merge_runs_script), str(self.unpacked)],
                           check=True, capture_output=True)
        self.doc_path = self.unpacked / "word/document.xml"
        self.xml = self.doc_path.read_text(encoding="utf-8")
        self._next_id = 9000
        self.log: list[str] = []

    # ---------- tiện ích ----------

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _tracked(self, old_run_xml: str, old_text: str, new_text: str) -> str:
        """Dựng cặp <w:del>/<w:ins> thay cho một run. Chuỗi rỗng = xoá hẳn."""
        rpr = re.search(r"<w:rPr>.*?</w:rPr>", old_run_xml, re.S)
        rpr = rpr.group(0) if rpr else ""
        out = ""
        if old_text:
            out += (f'<w:del w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"><w:r>{rpr}'
                    f'<w:delText xml:space="preserve">{esc(old_text)}</w:delText>'
                    f"</w:r></w:del>")
        if new_text:
            out += (f'<w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"><w:r>{rpr}'
                    f'<w:t xml:space="preserve">{esc(new_text)}</w:t>'
                    f"</w:r></w:ins>")
        return out


    def _find_paragraph(self, anchor_text: str) -> str:
        """Tìm đúng MỘT đoạn chứa `anchor_text`.

        Báo lỗi nếu khớp nhiều đoạn — mốc mơ hồ từng khiến đoạn Abstract bị xoá
        nhầm thay cho đoạn Conclusion, nên ở đây phải hỏng ồn ào chứ không im lặng
        chọn đoạn đầu tiên.
        """
        paras = re.findall(r"<w:p\b(?:(?!</w:p>).)*?</w:p>", self.xml, re.S)
        hits = [p for p in paras
                if anchor_text in html.unescape("".join(TEXT_RE.findall(p)))]
        if not hits:
            raise SystemExit(f"Không thấy đoạn chứa: {anchor_text!r}")
        if len(hits) > 1:
            previews = "\n  ".join(
                html.unescape("".join(TEXT_RE.findall(h)))[:90] for h in hits)
            raise SystemExit(
                f"Mốc mơ hồ — {len(hits)} đoạn cùng chứa {anchor_text!r}:\n  {previews}\n"
                "Hãy dùng một chuỗi dài hơn, chỉ có ở đúng đoạn cần sửa.")
        return hits[0]

    # ---------- sửa chữ ----------

    def replace_text(self, old: str, new: str, count: int = 1, required: bool = True) -> int:
        """Thay `old` bằng `new` ở cấp run, có track change.

        `old` phải nằm gọn trong một `<w:t>` (chạy merge_runs.py trước để chắc chắn).
        Nếu `new` giống hệt `old` thì không sinh track change nào.
        """
        if old == new:
            return 0
        pattern = re.compile(
            r"(<w:r\b(?:(?!</w:r>).)*?<w:t[^>]*>)([^<]*?)(" + re.escape(esc(old)) + r")([^<]*?)(</w:t>)"
            r"((?:(?!</w:r>).)*?</w:r>)", re.S)
        done = 0

        def sub(m):
            nonlocal done
            if done >= count:
                return m.group(0)
            done += 1
            before, _, after = m.group(2), m.group(3), m.group(4)
            run = m.group(0)
            rpr = re.search(r"<w:rPr>.*?</w:rPr>", run, re.S)
            rpr = rpr.group(0) if rpr else ""
            out = ""
            if before:
                out += f'<w:r>{rpr}<w:t xml:space="preserve">{before}</w:t></w:r>'
            out += self._tracked(run, old, new)
            if after:
                out += f'<w:r>{rpr}<w:t xml:space="preserve">{after}</w:t></w:r>'
            return out

        self.xml = pattern.sub(sub, self.xml)
        if done:
            self.log.append(f"chữ: {old[:60]!r} → {new[:60]!r}")
        elif required:
            raise SystemExit(f"Không tìm thấy trong document.xml: {old!r}")
        return done

    # ---------- sửa bảng ----------

    def table_cells(self, table_index: int) -> list[list[str]]:
        t = TBL_RE.findall(self.xml)[table_index]
        return [[html.unescape("".join(TEXT_RE.findall(c))) for c in TC_RE.findall(r)]
                for r in TR_RE.findall(t)]

    def set_cell(self, table_index: int, row: int, col: int, new_text: str) -> None:
        tables = TBL_RE.findall(self.xml)
        t_old = tables[table_index]
        rows = TR_RE.findall(t_old)
        r_old = rows[row]
        cells = TC_RE.findall(r_old)
        c_old = cells[col]
        old_text = html.unescape("".join(TEXT_RE.findall(c_old)))
        if old_text == new_text:
            return

        runs = re.findall(r"<w:r\b(?:(?!</w:r>).)*?</w:r>", c_old, re.S)
        if not runs:
            raise SystemExit(f"Ô bảng {table_index}[{row}][{col}] không có run nào để sửa")
        c_new = c_old.replace(runs[0], self._tracked(runs[0], old_text, new_text), 1)
        for extra in runs[1:]:                    # gộp phần chữ còn lại vào cùng một lần xoá
            c_new = c_new.replace(extra, "", 1)

        r_new = r_old.replace(c_old, c_new, 1)
        t_new = t_old.replace(r_old, r_new, 1)
        self.xml = self.xml.replace(t_old, t_new, 1)
        self.log.append(f"bảng {table_index}[{row}][{col}]: {old_text!r} → {new_text!r}")


    # ---------- thay cả cấu trúc bảng ----------

    def replace_table(self, table_index: int, header: list[str],
                      rows: list[list[str]]) -> None:
        """Thay TOÀN BỘ một bảng — kể cả số cột — dưới dạng Track Changes.

        `set_cell` chỉ đổi được chữ trong ô có sẵn, nên không dùng được khi bảng
        mới có bố cục cột khác. Ở đây các hàng cũ được đánh dấu xoá
        (`<w:trPr><w:del/></w:trPr>` cộng `<w:delText>` cho mọi run) và các hàng
        mới được đánh dấu chèn (`<w:trPr><w:ins/></w:trPr>`), nên Word hiển thị
        đúng kiểu redline và Accept/Reject hoạt động bình thường.

        `<w:tblGrid>` được dựng lại theo số cột mới, chia đều tổng bề rộng cũ —
        thiếu bước này thì Word vẽ bảng lệch khi số cột thay đổi.
        """
        tables = TBL_RE.findall(self.xml)
        if table_index >= len(tables):
            raise SystemExit(f"Không có bảng thứ {table_index} (tổng {len(tables)})")
        t_old = tables[table_index]
        old_rows = TR_RE.findall(t_old)
        if not old_rows:
            raise SystemExit(f"Bảng {table_index} không có hàng nào")

        # bề rộng tổng, lấy từ tblW nếu có, không thì cộng gridCol cũ
        m_w = re.search(r'<w:tblW w:w="(\d+)"', t_old)
        if m_w and int(m_w.group(1)) > 0:
            total = int(m_w.group(1))
        else:
            total = sum(int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"', t_old)) or 8750

        n_col = len(header)
        col_w = total // n_col
        grid = ("<w:tblGrid>"
                + "".join(f'<w:gridCol w:w="{col_w}"/>' for _ in range(n_col))
                + "</w:tblGrid>")

        # định dạng ô lấy từ ô đầu tiên của bảng cũ, để giữ lề và viền
        first_tc = TC_RE.search(old_rows[0])
        m_mar = re.search(r"<w:tcMar>.*?</w:tcMar>", first_tc.group(0), re.S) if first_tc else None
        tc_mar = m_mar.group(0) if m_mar else ""
        m_b = re.search(r"<w:rPr>.*?<w:b/>.*?</w:rPr>", old_rows[0], re.S)
        bold_rpr = m_b.group(0) if m_b else "<w:rPr><w:b/></w:rPr>"

        def cell(text: str, mark: str, rpr: str = "") -> str:
            body = (f'<w:{mark} w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"><w:r>{rpr}')
            body += (f'<w:delText xml:space="preserve">{esc(text)}</w:delText>'
                     if mark == "del" else
                     f'<w:t xml:space="preserve">{esc(text)}</w:t>')
            body += f"</w:r></w:{mark}>"
            return (f'<w:tc><w:tcPr><w:tcW w:w="{col_w}" w:type="dxa"/>{tc_mar}</w:tcPr>'
                    f"<w:p>{body}</w:p></w:tc>")

        # hàng cũ: đánh dấu xoá cả dấu kết hàng lẫn mọi run bên trong
        deleted = ""
        for r in old_rows:
            body = r
            for run in re.findall(r"<w:r\b(?:(?!</w:r>).)*?</w:r>", r, re.S):
                txt = html.unescape("".join(TEXT_RE.findall(run)))
                body = body.replace(run, self._tracked(run, txt, ""), 1)
            del_tr = (f'<w:trPr><w:del w:id="{self._id()}" w:author="{esc(self.author)}" '
                      f'w:date="{self.date}"/></w:trPr>')
            m_trpr = re.search(r"<w:trPr>.*?</w:trPr>", body, re.S)
            if m_trpr:
                body = body.replace(m_trpr.group(0), del_tr, 1)
            else:
                # Thứ tự trong <w:tr> do schema quy định: tblPrEx rồi mới tới trPr.
                # Chèn trPr lên trước tblPrEx làm file sai schema và Word từ chối mở.
                m_ex = re.search(r"<w:tblPrEx>.*?</w:tblPrEx>", body, re.S)
                if m_ex:
                    body = body.replace(m_ex.group(0), m_ex.group(0) + del_tr, 1)
                else:
                    body = re.sub(r"(<w:tr\b[^>]*>)", r"\1" + del_tr, body, count=1)
            deleted += body

        # hàng mới: đánh dấu chèn
        ins_tr = lambda: (f'<w:trPr><w:ins w:id="{self._id()}" '
                          f'w:author="{esc(self.author)}" w:date="{self.date}"/></w:trPr>')
        inserted = f"<w:tr>{ins_tr()}" + "".join(
            cell(h, "ins", bold_rpr) for h in header) + "</w:tr>"
        for row in rows:
            cells = [str(c) for c in row] + [""] * (n_col - len(row))
            inserted += (f"<w:tr>{ins_tr()}"
                         + "".join(cell(c, "ins") for c in cells[:n_col]) + "</w:tr>")

        t_new = t_old
        m_grid = re.search(r"<w:tblGrid>.*?</w:tblGrid>", t_old, re.S)
        if m_grid:
            t_new = t_new.replace(m_grid.group(0), grid, 1)
        # bỏ hết hàng cũ khỏi vị trí cũ rồi ghép lại theo thứ tự xoá → chèn
        for r in old_rows:
            t_new = t_new.replace(r, "", 1)
        t_new = t_new.replace("</w:tbl>", deleted + inserted + "</w:tbl>", 1)

        self.xml = self.xml.replace(t_old, t_new, 1)
        self.log.append(f"thay cấu trúc bảng {table_index}: "
                        f"{len(old_rows)} hàng cũ → {len(rows) + 1} hàng mới, "
                        f"{n_col} cột")


    def insert_table_after(self, anchor_text: str, header: list[str],
                           rows: list[list[str]], caption: str,
                           model_table_index: int = 0) -> None:
        """Chèn MỘT BẢNG MỚI (kèm chú thích) ngay sau đoạn chứa `anchor_text`.

        Cần cho trường hợp bài viện dẫn "Table 4" mà bảng đó chưa từng tồn tại.
        Bảng mới sao định dạng (viền, lề ô) từ bảng số liệu có sẵn ở
        `model_table_index` để trông đồng bộ với phần còn lại của bài.
        Toàn bộ hàng và đoạn chú thích đều đánh dấu chèn (Track Changes).
        """
        target = self._find_paragraph(anchor_text)
        tables = TBL_RE.findall(self.xml)
        if model_table_index >= len(tables):
            raise SystemExit(f"Không có bảng mẫu thứ {model_table_index}")
        model = tables[model_table_index]

        m_pr = re.search(r"<w:tblPr>.*?</w:tblPr>", model, re.S)
        tbl_pr = m_pr.group(0) if m_pr else '<w:tblPr><w:tblW w:w="8750" w:type="dxa"/></w:tblPr>'
        m_w = re.search(r'<w:tblW w:w="(\d+)"', tbl_pr)
        total = int(m_w.group(1)) if m_w and int(m_w.group(1)) > 0 else 8750
        m_tc = TC_RE.search(model)
        m_mar = re.search(r"<w:tcMar>.*?</w:tcMar>", m_tc.group(0), re.S) if m_tc else None
        tc_mar = m_mar.group(0) if m_mar else ""

        n_col = len(header)
        col_w = total // n_col
        grid = ("<w:tblGrid>" + "".join(f'<w:gridCol w:w="{col_w}"/>' for _ in range(n_col))
                + "</w:tblGrid>")

        def cell(text: str, bold: bool) -> str:
            rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
            return (f'<w:tc><w:tcPr><w:tcW w:w="{col_w}" w:type="dxa"/>{tc_mar}</w:tcPr>'
                    f'<w:p><w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"><w:r>{rpr}'
                    f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:ins></w:p></w:tc>')

        def ins_tr():
            return (f'<w:trPr><w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"/></w:trPr>')

        body = f"<w:tr>{ins_tr()}" + "".join(cell(h, True) for h in header) + "</w:tr>"
        for row in rows:
            cells = [str(c) for c in row] + [""] * (n_col - len(row))
            body += f"<w:tr>{ins_tr()}" + "".join(
                cell(c, False) for c in cells[:n_col]) + "</w:tr>"

        mark = (f'<w:rPr><w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                f'w:date="{self.date}"/></w:rPr>')
        cap = (f"<w:p><w:pPr>{mark}</w:pPr>"
               f'<w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
               f'w:date="{self.date}"><w:r>'
               f'<w:t xml:space="preserve">{esc(caption)}</w:t></w:r></w:ins></w:p>')

        block = cap + f"<w:tbl>{tbl_pr}{grid}{body}</w:tbl>" + f"<w:p><w:pPr>{mark}</w:pPr></w:p>"
        self.xml = self.xml.replace(target, target + block, 1)
        self.log.append(f"chèn bảng mới ({n_col} cột, {len(rows)} hàng) sau "
                        f"{anchor_text[:40]!r}")

    # ---------- chèn đoạn ----------

    def insert_paragraph_after(self, anchor_text: str, paragraphs: list[str],
                               style_from_anchor: bool = True) -> None:
        """Chèn một hoặc nhiều đoạn mới ngay sau đoạn chứa `anchor_text`."""
        target = self._find_paragraph(anchor_text)

        ppr = re.search(r"<w:pPr>.*?</w:pPr>", target, re.S)
        ppr_inner = ppr.group(0)[len("<w:pPr>"):-len("</w:pPr>")] if (ppr and style_from_anchor) else ""
        ppr_inner = re.sub(r"<w:rPr>.*?</w:rPr>", "", ppr_inner, flags=re.S)

        blocks = ""
        for text in paragraphs:
            mark = (f'<w:rPr><w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"/></w:rPr>')
            blocks += (f"<w:p><w:pPr>{ppr_inner}{mark}</w:pPr>"
                       f'<w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                       f'w:date="{self.date}"><w:r>'
                       f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:ins></w:p>')
        self.xml = self.xml.replace(target, target + blocks, 1)
        self.log.append(f"chèn {len(paragraphs)} đoạn sau {anchor_text[:50]!r}")


    @staticmethod
    def _split_lead(text: str) -> tuple[str, str]:
        """Tách phần tiêu đề con dẫn đầu đoạn (vd 'Overall model comparison. ').

        Bản thảo in đậm cụm này; nếu ghi đè bằng một run trơn thì mất định dạng.
        Chỉ coi là tiêu đề khi cụm đầu ngắn và không chứa chữ số — để không cắt
        nhầm một câu bình thường kết thúc bằng dấu chấm.
        """
        i = text.find(". ")
        if 0 < i <= 60 and not any(ch.isdigit() for ch in text[:i]):
            return text[:i + 1], text[i + 1:]
        return "", text

    def replace_paragraph(self, anchor_text: str, new_paragraphs: list[str]) -> None:
        """Thay trọn một đoạn (tìm theo `anchor_text`) bằng một hoặc nhiều đoạn mới.

        Đoạn cũ bị bọc <w:del> nên vẫn đọc được ở chế độ Track Changes; đoạn mới
        nằm trong <w:ins>. Giữ nguyên định dạng đoạn (pPr) của đoạn cũ.
        """
        target = self._find_paragraph(anchor_text)

        ppr_m = re.search(r"<w:pPr>.*?</w:pPr>", target, re.S)
        ppr_inner = ppr_m.group(0)[len("<w:pPr>"):-len("</w:pPr>")] if ppr_m else ""
        ppr_inner = re.sub(r"<w:rPr>.*?</w:rPr>", "", ppr_inner, flags=re.S)

        # đoạn cũ: mọi run thành <w:del>, dấu kết đoạn cũng đánh dấu xoá
        # (xoá dấu kết đoạn = nhập đoạn này vào đoạn ngay sau nó khi Accept)
        old_body = target
        runs = re.findall(r"<w:r\b(?:(?!</w:r>).)*?</w:r>", target, re.S)
        for run in runs:
            txt = html.unescape("".join(TEXT_RE.findall(run)))
            old_body = old_body.replace(run, self._tracked(run, txt, ""), 1)
        del_mark = (f'<w:del w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"/>')
        if ppr_m:
            old_body = old_body.replace(ppr_m.group(0),
                                        f"<w:pPr>{ppr_inner}<w:rPr>{del_mark}</w:rPr></w:pPr>", 1)
        else:
            # Đoạn không có <w:pPr> thì phải TẠO mới, nếu không dấu kết đoạn không
            # được đánh dấu xoá và Word để lại một đoạn rỗng sau khi Accept.
            old_body = re.sub(r"(<w:p\b[^>]*>)",
                              r"\1" + f"<w:pPr><w:rPr>{del_mark}</w:rPr></w:pPr>",
                              old_body, count=1)

        # định dạng của run đầu đoạn cũ — thường là tiêu đề con in đậm
        first_run = runs[0] if runs else ""
        m_rpr = re.search(r"<w:rPr>.*?</w:rPr>", first_run, re.S)
        lead_rpr = m_rpr.group(0) if m_rpr else ""

        blocks = ""
        for text in new_paragraphs:
            mark = (f'<w:rPr><w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                    f'w:date="{self.date}"/></w:rPr>')
            lead, rest = self._split_lead(text)
            body = ""
            if lead:
                body += (f"<w:r>{lead_rpr}"
                         f'<w:t xml:space="preserve">{esc(lead)}</w:t></w:r>')
            if rest:
                body += f'<w:r><w:t xml:space="preserve">{esc(rest)}</w:t></w:r>'
            blocks += (f"<w:p><w:pPr>{ppr_inner}{mark}</w:pPr>"
                       f'<w:ins w:id="{self._id()}" w:author="{esc(self.author)}" '
                       f'w:date="{self.date}">{body}</w:ins></w:p>')

        self.xml = self.xml.replace(target, old_body + blocks, 1)
        self.log.append(f"thay đoạn {anchor_text[:50]!r} bằng {len(new_paragraphs)} đoạn mới")

    # ---------- ghi ra ----------

    def save(self, out_path: str | pathlib.Path) -> pathlib.Path:
        out = pathlib.Path(out_path).resolve()
        self.doc_path.write_text(self.xml, encoding="utf-8")
        out.unlink(missing_ok=True)
        subprocess.run(["zip", "-Xrq", str(out), "."], cwd=self.unpacked, check=True)
        return out

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
