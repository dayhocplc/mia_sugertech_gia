#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xuất hình theo đúng quy cách tệp của Springer.

    python3 src/export_springer_figures.py

Hướng dẫn của Sugar Tech:
  * đặt tên tệp bằng "Fig" kèm số hình, ví dụ Fig1.tif;
  * halftone (ảnh chụp) tối thiểu 300 dpi; combination art tối thiểu 600 dpi;
  * bề ngang vừa cột: 84 mm (hai cột) hoặc 174 mm (một cột), cao không quá
    234 mm;
  * ảnh màu nộp ở RGB 8 bit.

Hình 1 (sơ đồ), Hình 2 (ma trận nhầm lẫn) và Hình 3 (đường cong) là nét vẽ ghép
chữ; Hình 4 là ảnh đồng ruộng có hộp và chữ chồng lên. Cả bốn đều là combination
art → 600 dpi, xuất ra TIFF (định dạng Springer ưu tiên) và PNG dự phòng.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import FIGURES, RAE, log

OUT = RAE / "manuscript" / "springer_figures"
MM_PER_INCH = 25.4
# (tệp nguồn, số hình, bề ngang mm, dpi)
SPEC = [("fig1_workflow", 1, 174, 600),
        ("fig6_confusion", 2, 174, 600),
        ("fig8_curves", 3, 174, 600),
        ("fig9_gsd", 4, 174, 600),
        ("fig5_qualitative", 5, 174, 600)]
MAX_HEIGHT_MM = 234


def main():
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    for stem, num, width_mm, dpi in SPEC:
        # ưu tiên bản dpi cao nếu đã vẽ; nếu chưa thì báo cách vẽ
        hi = FIGURES / f"dpi{dpi}" / f"{stem}.png"
        src = hi if hi.exists() else FIGURES / f"{stem}.png"
        if not src.exists():
            log(f"thiếu {src} — chạy:  RAE_FIG_DPI={dpi} RAE_FIG_LANG=en python3 src/make_figures.py --all"); continue
        im = Image.open(src).convert("RGB")       # Springer: RGB 8 bit/kênh

        # Kích thước thật của hình ở dpi đích. Vẽ ở dpi cao rồi chỉ đóng dấu
        # dpi là đủ; chỉ THU NHỎ khi vượt khổ, và không bao giờ phóng to.
        w_mm = im.width / dpi * MM_PER_INCH
        h_mm = im.height / dpi * MM_PER_INCH
        scale = min(1.0, width_mm / w_mm, MAX_HEIGHT_MM / h_mm)
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
            log(f"Fig{num}: thu {scale:.2f}× cho vừa khổ {width_mm}×{MAX_HEIGHT_MM} mm")
        elif w_mm < width_mm * 0.9:
            log(f"Fig{num}: chỉ rộng {w_mm:.0f} mm trên khổ cho phép {width_mm} mm — "
                f"vẽ lại với RAE_FIG_DPI thấp hơn nếu muốn lấp hết bề ngang")

        # LZW không mất dữ liệu nhưng ảnh chụp ở 600 dpi cho tệp ~50 MB, quá
        # nặng để nộp. Với hình có nội dung ảnh chụp, dùng JPEG TRONG TIFF: vẫn
        # là TIFF như hướng dẫn yêu cầu, vẫn 600 dpi, còn khoảng một phần mười.
        photographic = im.width * im.height > 12_000_000
        tif_kw = ({"compression": "jpeg", "quality": 92} if photographic
                  else {"compression": "tiff_lzw"})
        for ext, kw in (("tif", tif_kw), ("png", {})):
            p = OUT / f"Fig{num}.{ext}"
            im.save(p, dpi=(dpi, dpi), **kw)
            log(f"Fig{num}.{ext}  {im.width}×{im.height} px  {dpi} dpi  "
                f"{im.width / dpi * MM_PER_INCH:.0f}×{im.height / dpi * MM_PER_INCH:.0f} mm"
                f"  {p.stat().st_size / 1024:.0f} KB")
    log(f"→ {OUT}")


if __name__ == "__main__":
    main()
