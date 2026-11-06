#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đóng gói bản nộp kèm bài báo thành một file .zip.

    python3 src/make_package.py                       # gói chính, không kèm ảnh/trọng số
    python3 src/make_package.py --with-images         # kèm cả ảnh (~2,5 GB)
    python3 src/make_package.py --with-weights        # kèm trọng số tốt nhất (~1,5 GB)

Gói chính chứa mọi thứ cần để kiểm tra lại bài báo mà không cần tải ảnh:
mã nguồn, nhãn COCO, kết quả, bảng, hình, log huấn luyện, và MANIFEST.sha256
liệt kê checksum của TỪNG file ảnh — nhờ đó người phản biện xác minh được bộ
dữ liệu là đúng bộ đã dùng, kể cả khi ảnh được gửi riêng.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import RAE, ROOT, RESULTS, load_cfg, log

SPLITS = ("train", "valid", "test")

# Bản thảo dạng .md đi theo gói để check_consistency chạy được từ trong gói.
PAPER_MD = {
    "rae": ("RAE_manuscript.md", "RAE_supplementary.md"),
    "sugartech": ("SugarTech_manuscript.md", "SugarTech_supplementary.md"),
    "aiia": ("AIIA_manuscript.md", "AIIA_supplementary.md"),
}

# Tên tạp chí đích và tên gói phụ lục mà tạp chí ấy dùng.
JOURNAL_NAME = {
    "rae": "Research in Agricultural Engineering",
    "sugartech": "Sugar Tech",
    "aiia": "Artificial Intelligence in Agriculture",
}


def sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def copy_tree(src: pathlib.Path, dst: pathlib.Path, patterns: list[str]) -> int:
    n = 0
    for pat in patterns:
        for p in sorted(src.glob(pat)):
            if not p.is_file():
                continue
            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    return n


def build(stage: pathlib.Path, cfg: dict, with_images: bool, with_weights: bool,
          journal: str = "rae") -> dict:
    counts = {}

    # --- mã nguồn ---
    code = stage / "code"
    counts["code"] = copy_tree(RAE, code, ["src/*.py", "configs/*.yaml",
                                           "run_all.sh", "setup_env.sh", "README.md"])

    # --- nhãn COCO: đây là "dữ liệu" theo nghĩa tái lập được ---
    ann = stage / "data/annotations"
    n_ann = 0
    for kind, key in (("bbox", "bbox_root"), ("polygon", "polygon_root")):
        root = ROOT / cfg["dataset"][key]
        for split in SPLITS:
            src = root / split / "_annotations.coco.json"
            if src.exists():
                # giữ nguyên bố cục gốc để code chạy thẳng, không cần sửa cấu hình
                out = ann / kind / split / "_annotations.coco.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
                n_ann += 1
        for readme in ("README.roboflow.txt", "README.dataset.txt"):
            if (root / readme).exists():
                shutil.copy2(root / readme, ann / kind / readme)
    counts["annotations"] = n_ann
    if (RESULTS / "dataset_stats.json").exists():
        shutil.copy2(RESULTS / "dataset_stats.json", stage / "data/dataset_stats.json")

    # --- kết quả, bảng, hình ---
    counts["results"] = copy_tree(RESULTS, stage / "results", ["*.json", "common/*.json"])
    counts["tables"] = copy_tree(RAE / "tables", stage / "tables", ["*.csv", "*.md"])
    # copy_tree dùng glob (không đệ quy) nên figures/dpi600 — bản 600 dpi 62 MB
    # chỉ dùng để nộp bài — tự động nằm ngoài gói.
    counts["figures"] = copy_tree(RAE / "figures", stage / "figures", ["*.pdf", "*.png"])
    # Hai báo cáo tiếng Việt này nhắc tới lần nộp RAE, nên chỉ đi theo gói nội
    # bộ: hồ sơ của một lần nộp khác không phải việc của phản biện.
    if journal == "rae":
        for f in ("BAO_CAO_KET_QUA.md", "KNOWLEDGE_GRAPH.md"):
            if (RAE / f).exists():
                shutil.copy2(RAE / f, stage / f)
    # Bản thảo dạng .md đi kèm để audit.py A10 (số trong bài so với results/)
    # chạy được từ trong gói; trước đây A10 luôn báo "chưa có bản thảo".
    (stage / "manuscript").mkdir(parents=True, exist_ok=True)
    # Hình 600 dpi của Springer (TIFF + PNG, ~50 MB) đi kèm hồ sơ nộp bài, không
    # nằm trong gói mã và dữ liệu — nhét vào đây làm gói phình gấp đôi mà không
    # phục vụ việc tái lập, vì figures/ đã có bản PDF và PNG dựng lại được.
    # Gói đi kèm một lần nộp chỉ mang bản thảo của lần nộp ấy; gói nội bộ
    # ("rae") mang tất cả để đối chiếu.
    names = PAPER_MD.get(journal, PAPER_MD["aiia"])
    if journal == "rae":
        names = tuple(dict.fromkeys(sum(PAPER_MD.values(), ()))) + (
            "RAE_title_page.md", "RAE_cover_letter.md",
            "RAE_response_to_reviewers.md", "RAE_withdrawal_letter.md",
            "SugarTech_submission_checklist.md", "SugarTech_cover_letter.md")
    for name in names:
        for nm in (name, name.replace(".md", ".docx")):
            f = RAE / "manuscript" / nm
            if f.exists():
                shutil.copy2(f, stage / "manuscript" / nm)

    # --- log: đủ để dựng lại đường học, không kèm trọng số ---
    logs = stage / "logs"
    counts["logs"] = copy_tree(RAE / "logs", logs, ["*.log"])
    n_run = 0
    for model in cfg["models"]:
        run = RAE / "runs" / model
        if not run.is_dir():
            continue
        dst = logs / "runs" / model
        n_run += copy_tree(run, dst, ["log.txt", "metrics.json", "config.yaml",
                                      "results.csv", "args.yaml", "*.png", "*.jpg",
                                      "weights/*.yaml"])
    counts["run_logs"] = n_run

    # --- checksum của MỌI file, kể cả ảnh không đóng gói ---
    lines = [f"# MANIFEST — SHA-256, sinh {datetime.date.today().isoformat()}", ""]
    lines.append("## Các file có trong gói này")
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            lines.append(f"{sha256(p)}  {p.relative_to(stage)}")

    lines += ["", "## Ảnh của bộ dữ liệu"]
    if with_images:
        lines.append("# (ảnh nằm trong gói, tại data/images/)")
    else:
        lines.append("# (ảnh KHÔNG nằm trong gói — checksum dưới đây để xác minh bộ ảnh "
                     "nhận được là đúng bộ đã dùng)")
    n_img = 0
    for kind, key in (("bbox", "bbox_root"), ("polygon", "polygon_root")):
        root = ROOT / cfg["dataset"][key]
        for split in SPLITS:
            for img in sorted((root / split).glob("*.jpg")):
                lines.append(f"{sha256(img)}  data/images/{kind}/{split}/{img.name}")
                n_img += 1
                if with_images:
                    out = stage / "data/images" / kind / split / img.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, out)
    counts["images_checksummed"] = n_img
    counts["images_included"] = n_img if with_images else 0
    (stage / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- trọng số: chỉ checkpoint cuối, bỏ các checkpoint giữa chừng ---
    if with_weights:
        n_w = 0
        for model, spec in cfg["models"].items():
            run = RAE / "runs" / model
            src = (run / "model_final.pth" if spec["framework"] == "detectron2"
                   else run / "weights/best.pt")
            if src.exists():
                out = stage / "weights" / model / src.name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
                n_w += 1
        counts["weights"] = n_w

    return counts


def _si_header(journal: str) -> str:
    """Tiêu đề Springer bắt buộc có trong TỪNG tệp Supplementary Information.

    Hướng dẫn của Sugar Tech: "Please include in each file the following
    information: article title, journal name, author names; affiliation and
    e-mail address of the corresponding author." Tiêu đề này do đó phải nằm
    trong gói, không chỉ ở hồ sơ nộp.
    """
    import re
    if journal in ("sugartech", "aiia"):
        import build_manuscript_sugartech as M
    else:
        import build_manuscript as M
    name = JOURNAL_NAME[journal]
    def strip(t: str) -> str:
        t = re.sub(r"<sup>[^<]*</sup>", "", t)      # chỉ số chú thích
        return re.sub(r"<[^>]+>", "", t).replace("*", "").strip()
    return (f"**Article title:** {strip(M.TITLE)}\n\n"
            f"**Journal:** {name}\n\n"
            f"**Authors:** {strip(M.AUTHORS)}\n\n"
            f"**Affiliation:** {strip(M.AFFIL)}\n\n"
            f"**{strip(M.CORRESP)}**\n")


def _readme_en(stage, cfg, counts, with_images, with_weights, stats, comb,
               journal: str = "sugartech") -> None:
    """README của gói nộp quốc tế, bằng tiếng Anh.

    Bản tiếng Việt ở `write_readme` phục vụ nhóm tác giả; bản này đi ra ngoài
    cho phản biện quốc tế, nên nói rõ từng thư mục là gì, chạy lại thế nào, và
    thành thật rằng chú thích trong mã nguồn viết bằng tiếng Việt.
    """
    env = cfg.get("environment", {})
    # Springer gọi tệp phụ lục là Online Resource; Elsevier/KeAi gọi là Appendix.
    label = "Appendix B" if journal == "aiia" else "Online Resource 2"
    txt = f"""# {label} — code, data, logs and results

{_si_header(journal)}
Packaged {datetime.date.today().isoformat()}.

This archive contains everything needed to reproduce every number, table and
figure in the article without access to a GPU and without downloading the
images. The image files themselves are supplied by the corresponding author on
request; `MANIFEST.sha256` lists the SHA-256 checksum of every one of the
{counts['images_checksummed']} of them, so a recipient can verify that the image
set received is the set used here.

## Contents

```
code/
  src/                 the full pipeline, {counts['code']} files: training,
                       evaluation, bootstrap, significance testing, figure and
                       table generation, and the three verification layers
  configs/paper.yaml   every hyper-parameter and evaluation setting; no value
                       used in the article is hard-coded anywhere else
  run_all.sh           end-to-end driver
  setup_env.sh         environment construction
data/
  annotations/         COCO-format annotations for both Roboflow exports,
                       train/valid/test, as used ({counts['annotations']} files)
  dataset_stats.json   image and instance counts behind Table 1
{"  images/              the image files" if with_images else "  (images are not in this archive — see MANIFEST.sha256)"}
results/               raw measurements, one JSON per model plus the pooled
                       analyses ({counts['results']} files). Every number printed
                       in the article is read from this directory
tables/                Tables 1-5 as CSV and Markdown
figures/               Figures 1-4 as PDF and PNG
logs/                  training and evaluation logs ({counts['logs'] + counts['run_logs']} files),
                       including per-seed console logs and the per-run
                       directories from which the training curves are drawn
manuscript/            the manuscript and the Supplementary Information as
                       Markdown, so that the automatic checks below can verify
                       the text against results/
{"weights/               final checkpoint of each model" if with_weights else ""}
MANIFEST.sha256        SHA-256 of every file in this archive and of every image
```

## Dataset

Images were captured at Tan Hung farm, Chau Thanh district, Tay Ninh province,
Vietnam. Two Roboflow exports of the same capture campaign were annotated at
two levels of granularity:

| Export | Annotation | Train | Valid | Test | Source images | Instances |
|---|---|---|---|---|---|---|
| bbox | bounding box | {stats['bbox']['train']['images']} | {stats['bbox']['valid']['images']} | {stats['bbox']['test']['images']} | {stats['bbox']['_total']['source_images']} | {stats['bbox']['_total']['instances_total']} |
| polygon | polygon | {stats['polygon']['train']['images']} | {stats['polygon']['valid']['images']} | {stats['polygon']['test']['images']} | {stats['polygon']['_total']['source_images']} | {stats['polygon']['_total']['instances_total']} |

Union of the two exports: {comb['source_images_union']} distinct source images, of which
{comb['source_images_shared']} carry annotations in both formats.

**Important.** The two exports are annotated at different granularity
({stats['polygon']['test']['instances_per_image']} against {stats['bbox']['test']['instances_per_image']} instances per image in the test
partition), so the bounding-box and polygon models cannot be ranked against
each other. The article reports them on separate test partitions for this
reason, and the point is set out under "Basis of comparison" in the Methods.

## Reproducing the article

Without a GPU and without the images, the computational chain can be checked
in full:

```bash
python3 code/src/selftest.py          # recomputes metrics from stored predictions
python3 code/src/audit.py             # 11 consistency checks over results/
python3 code/src/check_consistency.py --journal {journal} \\
       --docx manuscript/{PAPER_MD[journal][0].replace(".md", ".docx")}
```

With a GPU and the image set in place:

```bash
bash code/setup_env.sh                # torch cu118 + Detectron2 + Ultralytics
bash code/run_all.sh full             # all models, all seeds
```

No number in `tables/` or `figures/` is typed by hand; all are generated from
`results/*.json`, and the third check above verifies that every number printed
in the manuscript still matches the one in `results/`.

## Verification

```bash
sha256sum -c MANIFEST.sha256 --ignore-missing
```

## Note on language

The article, this README and all generated tables and figures are in English.
Comments inside the Python source are in Vietnamese, the working language of
the authors. Function and variable names, log output and all file formats are
in English, so the code can be followed without Vietnamese.

## Environment used

{env.get('gpu', 'NVIDIA GeForce RTX 3080 Ti (12 GB)')} - CUDA {env.get('cuda', '11.8')} - PyTorch {env.get('torch', '2.1.2')} -
Detectron2 {env.get('detectron2', '0.6')} - Ultralytics {env.get('ultralytics', '8.3.253')} - Python {env.get('python', '3.10')}
"""
    (stage / "README.md").write_text(txt, encoding="utf-8")
    (stage / "README.txt").write_text(
        txt.replace("**", "").replace("# ", "").lstrip(), encoding="utf-8")


def write_readme(stage: pathlib.Path, cfg: dict, counts: dict,
                 with_images: bool, with_weights: bool,
                 journal: str = "rae") -> None:
    stats = json.loads((stage / "data/dataset_stats.json").read_text())
    comb = stats["_combined"]
    if journal in ("sugartech", "aiia"):
        return _readme_en(stage, cfg, counts, with_images, with_weights, stats,
                          comb, journal)
    txt = f"""# Supplementary material

{_si_header(journal)}
Đóng gói ngày {datetime.date.today().isoformat()}.

## Nội dung

```
code/           mã nguồn đầy đủ của pipeline ({counts['code']} file)
data/
  annotations/  nhãn COCO của cả hai bản xuất Roboflow ({counts['annotations']} file)
  dataset_stats.json
{"  images/       ảnh của bộ dữ liệu" if with_images else "  (ảnh gửi riêng — xem MANIFEST.sha256)"}
results/        số đo thô, dạng JSON ({counts['results']} file)
tables/         Bảng 1, 2, 3 dạng CSV và Markdown
figures/        các hình của bài báo
logs/           log huấn luyện và đánh giá ({counts['logs'] + counts['run_logs']} file)
{"weights/        trọng số tốt nhất của từng mô hình" if with_weights else ""}
BAO_CAO_KET_QUA.md   báo cáo kết quả chi tiết (tiếng Việt)
MANIFEST.sha256      checksum của mọi file, kể cả {counts['images_checksummed']} ảnh
```

## Bộ dữ liệu

Ảnh chụp tại nông trại Tân Hưng, huyện Châu Thành, tỉnh Tây Ninh, Việt Nam.
Hai bản xuất Roboflow của cùng đợt chụp, gán nhãn ở hai định dạng:

| Bản xuất | Nhãn | Train | Valid | Test | Ảnh gốc | Đối tượng |
|---|---|---|---|---|---|---|
| bbox | hộp | {stats['bbox']['train']['images']} | {stats['bbox']['valid']['images']} | {stats['bbox']['test']['images']} | {stats['bbox']['_total']['source_images']} | {stats['bbox']['_total']['instances_total']} |
| polygon | đa giác | {stats['polygon']['train']['images']} | {stats['polygon']['valid']['images']} | {stats['polygon']['test']['images']} | {stats['polygon']['_total']['source_images']} | {stats['polygon']['_total']['instances_total']} |

Hợp hai bản: {comb['source_images_union']} ảnh gốc riêng biệt, {comb['source_images_shared']} ảnh
có nhãn ở cả hai định dạng.

**Lưu ý quan trọng:** hai bản xuất gán nhãn ở độ mịn khác nhau
({stats['polygon']['test']['instances_per_image']} so với {stats['bbox']['test']['instances_per_image']} đối tượng/ảnh ở tập test),
nên năm mô hình không xếp hạng chung được. Chi tiết ở mục 3 của BAO_CAO_KET_QUA.md
và ở đoạn "Basis of comparison" trong bài báo.

## Chạy lại

```bash
bash code/setup_env.sh        # torch cu118 + Detectron2 + Ultralytics
bash code/run_all.sh full     # 5 mô hình × 50 epoch, ~6 giờ trên RTX 3080 Ti
```

Kiểm tra phần tính toán mà không cần GPU và không cần ảnh:

```bash
python3 code/src/selftest.py
```

Mọi tham số nằm ở `code/configs/paper.yaml`. Không con số nào trong `tables/`
hay `figures/` được gõ tay — tất cả sinh từ `results/*.json`.

## Môi trường đã dùng

NVIDIA GeForce RTX 3080 Ti (12 GB) · CUDA 11.8 · PyTorch 2.1.2 ·
Detectron2 0.6 · Ultralytics 8.3.253 · Python 3.10

## Xác minh

```bash
sha256sum -c MANIFEST.sha256 --ignore-missing
```
"""
    (stage / "README.md").write_text(txt, encoding="utf-8")
    # bản thuần văn bản để đọc được mà không cần giải nén trình kết xuất
    (stage / "README.txt").write_text(
        txt.replace("**", "").replace("#", "").lstrip(), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--journal", choices=("rae", "sugartech", "aiia"), default="rae",
                    help="sugartech: ESM_2.zip kiểu Springer; aiia: Appendix_B.zip "
                         "kiểu Elsevier/KeAi")
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--with-weights", action="store_true")
    a = ap.parse_args()

    cfg = load_cfg()
    # Springer: "Name the files consecutively, e.g. ESM_3.mpg, ESM_4.pdf".
    # ESM_1 là tệp Supplementary Information dạng .docx; gói này là ESM_2.
    name = {"sugartech": "ESM_2_code_data_logs_results",
            "aiia": "Appendix_B_code_data_logs_results",
            }.get(a.journal, "RAE_weed_sugarcane_supplementary")
    if a.with_images:
        name += "_with_images"
    if a.with_weights:
        name += "_with_weights"
    zipname = {"sugartech": "ESM_2.zip", "aiia": "Appendix_B.zip"}.get(
        a.journal, f"{name}.zip")
    out = pathlib.Path(a.out) if a.out else RAE / zipname

    stage = RAE / "_package" / name
    shutil.rmtree(stage.parent, ignore_errors=True)
    stage.mkdir(parents=True)

    log("đang gom file…")
    counts = build(stage, cfg, a.with_images, a.with_weights, a.journal)
    write_readme(stage, cfg, counts, a.with_images, a.with_weights, a.journal)

    log("đang nén…")
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, pathlib.Path(name) / p.relative_to(stage))

    shutil.rmtree(stage.parent, ignore_errors=True)
    size = out.stat().st_size / 1048576
    log(f"{out}  ({size:.1f} MB)")
    for k, v in counts.items():
        print(f"    {k:22s} {v}")


if __name__ == "__main__":
    main()
