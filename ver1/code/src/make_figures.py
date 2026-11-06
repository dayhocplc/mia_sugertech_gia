#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh các hình của bản thảo từ results/*.json và trọng số đã huấn luyện.

    python3 src/make_figures.py --workflow --bars --confusion   # không cần GPU
    python3 src/make_figures.py --qualitative --n 3             # cần trọng số

  fig1_workflow      Hình 1 — sơ đồ quy trình
  fig4_training      Hình 4 — đường học (loss / mAP) của năm mô hình
  fig5_qualitative   Hình 5 — kết quả phát hiện của năm mô hình trên cùng ảnh test
  fig6_confusion     ma trận nhầm lẫn 3×3 của năm mô hình
  fig7_metrics       biểu đồ cột Precision/Recall/F1/Accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from common import (FIGURES, RAE, MODEL_LABELS, MODEL_ORDER, RESULTS, coco_json,
                    dataset_root, load_cfg, log, remap_categories)

# Bản thảo nộp là tiếng Anh, nên hình cũng phải tiếng Anh. Đặt LANG=vi để lấy
# lại bản tiếng Việt dùng cho báo cáo nội bộ.
LANG = os.environ.get("RAE_FIG_LANG", "en")

TXT = {
    "en": {
        "cm_excl": "Weed F1, background-excluded:",
        "cm_full": "Weed F1, full:",
        "cm_note": "Dashed: the background row and column that the background-excluded convention discards",

        "c_recall": "Weed recall (full variant)",
        "c_prec": "Weed precision (full variant)",
        "c_fppi": "False weed detections per image",
        "c_fpr": "Spurious weed detections retained",
        "c_tpr": "Correct weed detections retained",
'flow': {'head': [('Field image acquisition',
                    'Tan Hung farm, Chau Thanh, Tay Ninh · iPhone 12 · nadir view, 140 cm\n'
                    'single day, three sessions on three plots · 4080 × 3072 px'),
                   ('Annotation (Roboflow)',
                    'both classes, sugarcane and weed, annotated in both conventions\n'
                    'export rescaled to 1280 × 1280 px')],
          'colhead': ['Bounding-box convention', 'Polygon convention'],
          'cols': [['1 211 source images\nsplit 80:10:10 by source image',
                    'augmentation, training partition only\nthree versions per source image',
                    'Faster R-CNN · RetinaNet · YOLOv10\n1024 / 1024 / 640 px, 50 epochs, 5 / 4 / 5 seeds',
                    'common test partition\n124 images'],
                   ['504 source images\nsplit 80:10:10 by source image',
                    'same augmentation recipe',
                    'Mask R-CNN · YOLOv8-seg\none run each',
                    'own test partition\n50 images']],
          'augnote': 'flips · 90° rotations · random crop 0–20% · brightness ±15% · exposure '
                     '±10% · noise 1.01%',
          'tail': ('Evaluation — identical protocol, separate partitions',
                   'COCO AP / AP50 / AP75 · confusion matrix in both variants · IoU 0.50 · '
                   'confidence 0.30\n'
                   'two-level bootstrap over images and seeds · the two branches are never '
                   'ranked against each other')},
 'pred': 'predicted',
 'true': 'ground truth',
 'value': 'value',
 'qual': 'Green boxes: sugarcane · red boxes: weed. Top row of each pair: ground-truth '
         'annotation; bottom row: prediction with confidence score.',
 'qual_ok': 'success case',
 'qual_missed': 'annotated weed not detected',
 'gsd_x': 'effective GSD (× acquisition)',
 'gsd_y': 'weed-class recall',
 'gsd_h': 'equivalent camera height (cm)',
 'gsd_a': 'a  recall',
 'gsd_b': 'c  by weed size',
 'gsd_q': 'area quartile',
 'gsd_p': 'weed-class precision',
 'gsd_c': 'b  precision',
 'gsd_deg': 'detail removed (simulated higher camera)',
 'gsd_til': 'tiled on native frames (seed 0)',
 'qual_bad': 'failure case',
 'qual_gt': 'ground truth',
 'qual_pred': 'prediction',
 'qual_recall': 'weed recall'},
    "vi": {
        "cm_excl": "F1 cỏ, loại nền:",
        "cm_full": "F1 cỏ, giữ nền:",
        "cm_note": "Nét đứt: hàng và cột nền mà quy ước loại nền vứt bỏ",

        "c_recall": "Recall cỏ (biến thể giữ nền)",
        "c_prec": "Precision cỏ (biến thể giữ nền)",
        "c_fppi": "Số báo nhầm cỏ mỗi ảnh",
        "c_fpr": "Tỷ lệ dự đoán cỏ thừa còn giữ lại",
        "c_tpr": "Tỷ lệ dự đoán cỏ đúng còn giữ lại",
'flow': {'head': [('Thu thập ảnh đồng ruộng',
                    'Tân Hưng, Châu Thành, Tây Ninh · iPhone 12 · nadir 140 cm\n'
                    'một ngày, ba buổi trên ba lô · 4080 × 3072 px'),
                   ('Gán nhãn (Roboflow)',
                    'cả hai lớp mía và cỏ, ở cả hai quy ước\nxuất về 1280 × 1280 px')],
          'colhead': ['Quy ước hộp bao', 'Quy ước đa giác'],
          'cols': [['1 211 ảnh nguồn\nchia 80:10:10 theo ảnh nguồn',
                    'tăng cường, chỉ trên phần train\nba bản mỗi ảnh nguồn',
                    'Faster R-CNN · RetinaNet · YOLOv10\n1024 / 1024 / 640 px, 50 epoch, 5 / 4 / 5 seed',
                    'tập test dùng chung\n124 ảnh'],
                   ['504 ảnh nguồn\nchia 80:10:10 theo ảnh nguồn',
                    'cùng công thức tăng cường',
                    'Mask R-CNN · YOLOv8-seg\nmỗi mô hình một lượt',
                    'tập test riêng\n50 ảnh']],
          'augnote': 'lật · xoay 90° · cắt ngẫu nhiên 0–20% · sáng ±15% · phơi sáng ±10% · '
                     'nhiễu 1,01%',
          'tail': ('Đánh giá — cùng giao thức, khác tập test',
                   'COCO AP / AP50 / AP75 · ma trận nhầm lẫn ở cả hai biến thể · IoU 0,50 · '
                   'tin cậy 0,30\n'
                   'bootstrap hai tầng theo ảnh và seed · hai nhánh không xếp hạng chung')},
 'pred': 'dự đoán',
 'true': 'nhãn thật',
 'value': 'giá trị',
 'qual': 'Hộp xanh: mía · hộp đỏ: cỏ. Hàng trên mỗi cặp: nhãn thật; hàng dưới: dự đoán kèm '
         'điểm tin cậy.',
 'qual_ok': 'ca thành công',
 'qual_bad': 'ca thất bại',
 'qual_missed': 'cỏ có nhãn nhưng không phát hiện',
 'gsd_x': 'GSD hiệu dụng (× lúc chụp)',
 'gsd_y': 'recall lớp cỏ',
 'gsd_h': 'độ cao camera tương đương (cm)',
 'gsd_a': 'a  recall',
 'gsd_b': 'c  theo kích thước cỏ',
 'gsd_q': 'phân vị diện tích',
 'gsd_p': 'precision lớp cỏ',
 'gsd_c': 'b  precision',
 'gsd_deg': 'làm mất chi tiết (camera cao hơn)',
 'gsd_til': 'cắt ô trên ảnh gốc (seed 0)',
 'qual_gt': 'nhãn thật',
 'qual_pred': 'dự đoán',
 'qual_recall': 'recall cỏ'},
}[LANG]

plt.rcParams.update({
    # Chữ trong hình dùng CÙNG phông với thân bài để bản thảo đồng nhất.
    # Hướng dẫn của Springer khuyến nghị (chứ không bắt buộc) Helvetica/Arial:
    # "To add lettering, it is best to use Helvetica or Arial". Đặt
    # RAE_FIG_FONT=sans để quay lại sans-serif nếu toà soạn yêu cầu.
    "font.family": ("sans-serif" if os.environ.get("RAE_FIG_FONT") == "sans"
                    else "serif"),
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    # RAE_FIG_DPI=600 để vẽ lại ở độ phân giải Springer yêu cầu cho combination
    # art. Vẽ lại ở dpi cao khác hẳn phóng to ảnh đã vẽ: nét và chữ thật sự sắc
    # hơn, không phải nội suy thêm điểm ảnh rỗng.
    "savefig.bbox": "tight",
    "savefig.dpi": int(os.environ.get("RAE_FIG_DPI", "300")),
})
CLASS_COLOR = {"sugarcane": "#2e7d32", "weed": "#c62828"}   # Hình 5: mía xanh, cỏ đỏ


# Bản 600 dpi cho Springer ghi sang thư mục riêng: nếu ghi đè figures/ thì
# fig5_qualitative.pdf phình từ 2 MB lên 32 MB và gói mã-dữ liệu phình theo.
DPI = int(os.environ.get("RAE_FIG_DPI", "300"))
OUTDIR = FIGURES if DPI <= 300 else FIGURES / f"dpi{DPI}"


def save(fig, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}")
    plt.close(fig)
    log(f"{stem} → {OUTDIR.relative_to(RAE)}/{stem}.pdf, .png")


def fig_workflow() -> None:
    """Hình 1 — quy trình, hai nhánh nhãn chạy song song từ đầu đến cuối.

    Bản trước vẽ một luồng thẳng trong khi chú thích nói hai nhánh được giữ
    riêng — phản biện chỉ ra mâu thuẫn đó. Hai cột song song làm rõ ngay vì sao
    năm mô hình không xếp hạng chung được: chúng không dùng chung tập test nào.
    """
    f = TXT["flow"]
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 13.2); ax.axis("off")
    EDGE = "#37474f"

    def box(x, y, w, h, title, body, shade=False, tfs=9, bfs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.07,rounding_size=0.10",
                                    linewidth=1.0, edgecolor=EDGE,
                                    facecolor="#eceff1" if shade else "#ffffff"))
        if title:
            ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                    fontweight="bold", fontsize=tfs)
            ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                    fontsize=bfs, color=EDGE)
        else:
            ax.text(x + w / 2, y + h / 2, body, ha="center", va="center",
                    fontsize=bfs, color=EDGE)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.0, color=EDGE))

    y = 12.1
    for i, (t, b) in enumerate(f["head"]):
        box(1.2, y, 9.6, 1.05, t, b, shade=(i == 0))
        arrow(6.0, y, 6.0, y - 0.38)
        y -= 1.43

    # hai cột song song
    XL, XR, W = 0.7, 6.5, 4.8
    for cx, head in zip((XL, XR), f["colhead"]):
        ax.text(cx + W / 2, y + 0.46, head, ha="center", va="bottom",
                fontweight="bold", fontsize=9)
    arrow(6.0, y + 0.42, XL + W / 2, y + 0.06)
    arrow(6.0, y + 0.42, XR + W / 2, y + 0.06)
    y -= 0.25

    rows = max(len(f["cols"][0]), len(f["cols"][1]))
    for r in range(rows):
        for cx, col in zip((XL, XR), f["cols"]):
            if r < len(col):
                box(cx, y - 0.86, W, 0.86, None, col[r], shade=(r % 2 == 0))
        if r < rows - 1:
            for cx in (XL, XR):
                arrow(cx + W / 2, y - 0.86, cx + W / 2, y - 1.12)
        y -= 1.12

    # Công thức tăng cường nằm đầy đủ trong Methods; nhắc lại ở đây chỉ làm chữ
    # đè lên hai mũi tên hội tụ.
    for cx in (XL, XR):
        arrow(cx + W / 2, y, 6.0, y - 0.34)
    box(1.2, y - 1.44, 9.6, 1.10, *f["tail"])
    y -= 1.44

    ax.set_ylim(y - 0.2, 13.3)
    fig.tight_layout()
    save(fig, "fig1_workflow")


def _detectron2_curve(model_key: str):
    """Đọc runs/<model>/metrics.json của Detectron2 → (iter, total_loss, val_loss)."""
    from common import RUNS
    p = RUNS / model_key / "metrics.json"
    if not p.exists():
        return None
    it, tr, va = [], [], []
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "total_loss" not in d:
            continue
        it.append(d["iteration"])
        tr.append(d["total_loss"])
        va.append(d.get("total_val_loss"))
    if not it:
        return None
    return np.array(it), np.array(tr, float), np.array(
        [np.nan if v is None else v for v in va], float)


def _ultralytics_curve(model_key: str):
    """Đọc runs/<model>/results.csv của Ultralytics → (epoch, train loss, mAP50)."""
    import csv as _csv
    from common import RUNS
    p = RUNS / model_key / "results.csv"
    if not p.exists():
        return None
    rows = list(_csv.DictReader(p.open()))
    if not rows:
        return None
    cols = {k.strip(): k for k in rows[0]}

    def col(*cands):
        for c in cands:
            if c in cols:
                return [float(r[cols[c]]) for r in rows]
        return None

    ep = col("epoch")
    box = col("train/box_loss")
    cls = col("train/cls_loss")
    m50 = col("metrics/mAP50(B)", "metrics/mAP50(M)")
    if ep is None or box is None:
        return None
    loss = np.array(box) + (np.array(cls) if cls else 0)
    return np.array(ep), loss, (np.array(m50) if m50 else None)


def fig_training(cfg: dict) -> None:
    """Đường học của các mô hình trong phạm vi bài, TRỤC HOÀNH LÀ EPOCH.

    Bản trước để Detectron2 theo iteration và Ultralytics theo epoch, rồi chú
    thích rằng hai trục không quy đổi được. Phản biện chỉ ra điều đó không đúng:
    số bước mỗi epoch là số ảnh train chia batch hiệu dụng, ở đây 2 892 / 16 ≈
    181, nên 50 epoch ≈ 9 050 iteration, khớp đúng độ dài trục đã vẽ. Quy đổi
    rồi thì ba bảng con nằm trên cùng một thang và đọc so sánh được.
    """
    from common import scope
    primary, _ = scope(cfg)
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    n_train = stats["bbox"]["train"]["images"]
    eff_batch = cfg["train"]["batch_size"]
    iters_per_epoch = max(1, round(n_train / eff_batch))
    panels = []
    for k in primary:
        if cfg["models"][k]["framework"] == "detectron2":
            c = _detectron2_curve(k)
            if c:
                ep = np.asarray(c[0], float) / iters_per_epoch
                panels.append((k, ep, {"train loss": c[1], "val loss": c[2]}, None))
        else:
            c = _ultralytics_curve(k)
            if c:
                extra = {"mAP@50": c[2]} if c[2] is not None else None
                panels.append((k, np.asarray(c[0], float), {"train loss": c[1]}, extra))
    if not panels:
        log("chưa có log huấn luyện trong runs/ — bỏ qua Hình 4")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.8), squeeze=False)
    for ax, (k, x, losses, extra) in zip(axes[0], panels):
        for name, y in losses.items():
            if y is None or np.all(np.isnan(y)):
                continue
            ax.plot(x, y, linewidth=1.2, label=name)
        ax.set_title(MODEL_LABELS[k], fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_xlim(0, cfg["train"]["epochs"])
        ax.set_ylabel("loss", fontsize=8)
        ax.grid(alpha=0.25)
        if extra:
            ax2 = ax.twinx()
            for name, y in extra.items():
                ax2.plot(x, y, linewidth=1.2, linestyle="--", color="#c62828", label=name)
            ax2.set_ylabel("mAP@50", fontsize=8, color="#c62828")
            ax2.set_ylim(0, 1)
            ax2.spines["right"].set_visible(True)
        ax.legend(frameon=False, fontsize=7)
    save(fig, "fig4_training")


def fig_gsd() -> None:
    """Hình tỷ lệ — recall và precision theo độ phân giải mặt đất, hai chiều.

    Panel a và b đặt cạnh nhau hai thí nghiệm trên CÙNG một trục tỷ lệ: làm mất
    chi tiết (chiều thô, gsd_analysis) và cắt ô trên ảnh gốc (chiều mịn,
    tiling_analysis). Điểm 1,00× của hai thí nghiệm gần trùng nhau, nên người đọc
    tự thấy mức khớp giữa chúng. Panel c cho thấy cú sụt theo chiều thô tập trung
    ở đối tượng nhỏ.

    Vì sao phải có panel precision: thí nghiệm cắt ô nâng recall rất ít nhưng phá
    precision, nên một hình chỉ vẽ recall sẽ nói ngược kết luận.
    """
    f = RESULTS / "gsd_analysis.json"
    if not f.exists():
        log("thiếu results/gsd_analysis.json — chạy src/gsd_analysis.py trước"); return
    g = json.loads(f.read_text())
    ks = [f"{x:g}" for x in g["scales"]]
    mods = [k for k in MODEL_ORDER if k in g["models"]] or list(g["models"])
    x = [g["models"][mods[0]][k]["effective_gsd_multiplier"] for k in ks]

    tf = RESULTS / "tiling_analysis.json"
    tl = json.loads(tf.read_text()) if tf.exists() else None
    tks = [str(k) for k in tl["tiles"]] if tl else []
    # bỏ ô 1×1 khỏi đường cắt ô: nó trùng điểm 1,00× của thí nghiệm kia
    tx = [tl["models"][mods[0]][k]["effective_gsd_multiplier"] for k in tks] if tl else []

    # chỉ gán nhãn một tập mốc thưa; vẽ hết mọi mốc thì nhãn chồng nhau
    TICKS = [0.33, 0.5, 1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 70 / 25.4))

    def series(ax, metric):
        for i, k in enumerate(mods):
            y = [g["models"][k][s][metric]["mean"] for s in ks]
            e = [g["models"][k][s][metric]["sd"] or 0 for s in ks]
            ax.errorbar(x, y, yerr=e, marker="o", ms=3.4, lw=1.3, capsize=2,
                        color=f"C{i}", label=MODEL_LABELS[k])
            if tl and k in tl["models"]:
                ty = [tl["models"][k][s][metric]["mean"] for s in tks]
                ax.plot(tx, ty, marker="^", ms=3.6, lw=1.1, ls="--", mfc="none",
                        color=f"C{i}")
        ax.set_xscale("log")
        ax.set_xticks(TICKS)
        ax.set_xticklabels([f"{v:g}" for v in TICKS], fontsize=6)
        ax.minorticks_off(); ax.set_ylim(0, None)
        ax.set_xlabel(TXT["gsd_x"], fontsize=7)

    ax = axes[0]
    series(ax, "weed_recall")
    ax.set_ylabel(TXT["gsd_y"])
    ax.set_title(TXT["gsd_a"], fontsize=8, loc="left")
    h = [plt.Line2D([], [], color=f"C{i}", marker="o", ms=3.4, lw=1.3)
         for i in range(len(mods))]
    if tl:
        h += [plt.Line2D([], [], color="0.35", marker="o", ms=3.4, lw=1.3),
              plt.Line2D([], [], color="0.35", marker="^", ms=3.6, lw=1.1, ls="--",
                         mfc="none")]
    lab = [MODEL_LABELS[k] for k in mods] + ([TXT["gsd_deg"], TXT["gsd_til"]] if tl else [])
    ax.legend(h, lab, fontsize=6, frameon=False)

    ax = axes[1]
    series(ax, "weed_precision")
    ax.set_ylabel(TXT["gsd_p"])
    ax.set_title(TXT["gsd_c"], fontsize=8, loc="left")

    ax = axes[2]
    k0 = mods[0]
    for q in range(4):
        y = [g["models"][k0][s]["recall_by_quartile"][q] for s in ks]
        ax.plot(x, y, marker="s", ms=3.2, lw=1.3, label=f"Q{q + 1}")
    ax.set_xlabel(TXT["gsd_x"], fontsize=7); ax.set_ylabel(TXT["gsd_y"])
    ax.set_xscale("log"); ax.set_xticks([v for v in TICKS if v >= 1])
    ax.set_xticklabels([f"{v:g}" for v in TICKS if v >= 1], fontsize=6)
    ax.minorticks_off(); ax.set_ylim(0, None)
    ax.legend(fontsize=6, frameon=False, title=TXT["gsd_q"], title_fontsize=6)
    ax.set_title(f"{TXT['gsd_b']} ({MODEL_LABELS[k0]})", fontsize=8, loc="left")

    # trục phụ: độ cao camera tương đương, vì đó là đại lượng người đọc chỉnh được
    # GSD hiệu dụng tỷ lệ thuận với độ cao: 1,00× ứng với 140 cm, 0,50× với 70 cm
    height = lambda v: g["camera_height_cm"] * v
    for ax_, xs in zip(axes, (TICKS, TICKS, [v for v in TICKS if v >= 1])):
        top = ax_.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
        top.set_xscale("log")
        top.set_xticks(xs)
        top.set_xticklabels([f"{height(v):.0f}" for v in xs], fontsize=6)
        # trục log tự thêm nhãn phụ dạng "3 × 10⁰" đè lên nhãn thật
        top.minorticks_off()
        top.set_xticks([], minor=True)
        top.set_xlabel(TXT["gsd_h"], fontsize=7)
    fig.tight_layout()
    save(fig, "fig9_gsd")


def fig_curves(cfg: dict) -> None:
    """Hình 3 — ba đường cong minh chứng cho kết quả.

    (a) precision-recall của lớp cỏ ở biến thể giữ nền;
    (b) FROC: recall theo số báo nhầm trung bình mỗi ảnh. Đây là dạng dùng được
        của "ROC" cho bài toán phát hiện: trục hoành là số dương tính giả tuyệt
        đối, không phải tỷ lệ — vì số vị trí "không vật thể" trong ảnh là vô
        hạn nên tỷ lệ dương tính giả không định nghĩa được;
    (c) ROC thật, cho câu hỏi CÓ tập âm tính xác định: điểm tin cậy tách được
        dự đoán cỏ đúng khỏi dự đoán cỏ thừa đến đâu.
    """
    import json

    from common import scope
    primary, _ = scope(cfg)
    f = RESULTS / "review_analysis.json"
    if not f.exists():
        log("chưa có review_analysis.json — chạy src/review_analysis.py trước"); return
    cur = json.loads(f.read_text()).get("curves")
    if not cur:
        log("review_analysis.json chưa có khoá 'curves' — chạy lại script đó"); return

    ev = cfg["eval"]
    colour = {k: c for k, c in zip(primary, ["#1565c0", "#c62828", "#2e7d32"])}
    fig, ax = plt.subplots(1, 3, figsize=(174 / 25.4, 62 / 25.4))

    for k in primary:
        pr = cur["pr"][k]
        ax[0].plot([d["recall"] for d in pr], [d["precision"] for d in pr],
                   color=colour[k], linewidth=1.3, label=MODEL_LABELS[k])
        at = min(pr, key=lambda d: abs(d["conf"] - ev["conf_threshold"]))
        ax[0].plot(at["recall"], at["precision"], "o", color=colour[k], ms=4.5,
                   markeredgecolor="white", markeredgewidth=0.6, zorder=5)

        fr = cur["froc"][k]
        ax[1].plot([d["fp_per_image"] for d in fr], [d["recall"] for d in fr],
                   color=colour[k], linewidth=1.3)
        at = min(fr, key=lambda d: abs(d["conf"] - ev["conf_threshold"]))
        ax[1].plot(at["fp_per_image"], at["recall"], "o", color=colour[k], ms=4.5,
                   markeredgecolor="white", markeredgewidth=0.6, zorder=5)

        ro = cur["roc"][k]
        ax[2].plot([d["fpr"] for d in ro["points"]], [d["tpr"] for d in ro["points"]],
                   color=colour[k], linewidth=1.3,
                   label=f"{MODEL_LABELS[k]} (AUC {ro['auc']:.2f})")

    ax[0].set_xlabel(TXT["c_recall"]); ax[0].set_ylabel(TXT["c_prec"])
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_title("(a)", fontsize=9, loc="left")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")

    ax[1].set_xlabel(TXT["c_fppi"]); ax[1].set_ylabel(TXT["c_recall"])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b)", fontsize=9, loc="left")

    ax[2].plot([0, 1], [0, 1], linestyle=":", linewidth=0.9, color="#90a4ae")
    ax[2].set_xlabel(TXT["c_fpr"]); ax[2].set_ylabel(TXT["c_tpr"])
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)
    ax[2].set_title("(c)", fontsize=9, loc="left")
    ax[2].legend(frameon=False, fontsize=6.5, loc="lower right")

    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig8_curves")


def _results(only: list[str] | None = None) -> dict:
    out = {}
    for k in (only or MODEL_ORDER):
        p = RESULTS / f"{k}.json"
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def fig_confusion(cfg: dict) -> None:
    """Ma trận nhầm lẫn, ĐÁNH DẤU hàng và cột mà quy ước loại nền xoá đi.

    Đây là hình trung tâm của bài: toàn bộ lập luận nằm ở chỗ xoá hàng và cột
    "background". Một ma trận trơn chỉ cho người đọc các con số; đánh dấu hai
    dải bị xoá và in kèm hệ quả bên dưới thì người đọc thấy ngay vì sao cùng
    một dự đoán lại cho F1 0,98 hoặc 0,48 tuỳ quy ước.
    """
    from matplotlib.patches import Rectangle as _Rect

    from common import scope
    primary, _ = scope(cfg)
    res = _results(primary)
    if not res:
        log("chưa có results/*.json — bỏ qua ma trận nhầm lẫn")
        return
    labels = cfg["dataset"]["classes"] + ["background"]
    bg = len(labels) - 1
    fig, axes = plt.subplots(1, len(res), figsize=(174 / 25.4, 78 / 25.4))
    axes = np.atleast_1d(axes)
    for ax, (k, r) in zip(axes, res.items()):
        m = np.array(r["confusion_matrix"], float)
        norm = m / np.clip(m.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#263238")
        for xy, w, h in (((-0.5, bg - 0.5), len(labels), 1),
                         ((bg - 0.5, -0.5), 1, len(labels))):
            ax.add_patch(_Rect(xy, w, h, fill=False, edgecolor="#c62828",
                               linewidth=1.6, linestyle=(0, (3, 2)), zorder=4,
                               clip_on=False))
        tp = m[1, 1]
        f1 = lambda fp, fn: (2 * tp / (2 * tp + fp + fn)) if tp else 0.0
        ax.set_xlabel(
            f"{TXT['cm_excl']} {f1(m[0, 1], m[1, 0]):.2f}\n"
            f"{TXT['cm_full']} {f1(m[0, 1] + m[bg, 1], m[1, 0] + m[1, bg]):.2f}",
            fontsize=7.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7)
        ax.set_title(MODEL_LABELS[k], fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel(TXT["true"], fontsize=8)
        for s in ax.spines.values():
            s.set_visible(True)
    fig.text(0.5, 0.012, TXT["cm_note"], ha="center", fontsize=7, color="#c62828")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig6_confusion")


def fig_bars(cfg: dict) -> None:
    res = _results()
    if not res:
        log("chưa có results/*.json — bỏ qua biểu đồ cột")
        return
    metrics = ["precision", "recall", "f1", "accuracy"]
    names = ["Precision", "Recall", "F1-score", "Accuracy"]
    keys = list(res)
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (m, nm) in enumerate(zip(metrics, names)):
        vals = [res[k]["metrics"]["filtered"]["overall_accuracy"] if m == "accuracy"
                else res[k]["metrics"]["filtered"]["macro"][m] for k in keys]
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=nm)
        ax.bar_label(b, fmt="%.3f", fontsize=6, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[k] for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(TXT["value"])
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig7_metrics")


def _missed_weed(gt_boxes, gt_cls, det_boxes, det_cls, det_scores, iou_thr):
    """Các hộp cỏ thật không được dự đoán nào mang nhãn cỏ ghép vào."""
    import numpy as np
    from metrics import box_iou_batch
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls, int)
    db = np.asarray(det_boxes, float).reshape(-1, 4)
    dc = np.asarray(det_cls, int)
    w_gt = np.nonzero(gt_cls == 1)[0]
    if not len(w_gt):
        return []
    if not len(db):
        return [gt_boxes[i] for i in w_gt]
    iou = np.asarray(box_iou_batch(gt_boxes[w_gt], db))
    out = []
    for r, i in enumerate(w_gt):
        hit = np.nonzero(iou[r] > iou_thr)[0]
        if not any(dc[j] == 1 for j in hit):
            out.append(gt_boxes[i])
    return out


def fig_qualitative(cfg: dict, n_images: int) -> None:
    """Hình 4 — mỗi mô hình một ca thành công và một ca thất bại, kèm nhãn thật.

    Bản trước chỉ vẽ ba ảnh dễ, không có nhãn thật, không có ca thất bại và
    không nêu tiêu chí chọn ảnh — phản biện chỉ ra rằng đó đúng là thứ cần minh
    hoạ mà hình lại thiếu. Bản này chọn ảnh theo recall lớp cỏ tính riêng từng
    ảnh: ca thành công là ảnh ở phân vị 50, ca thất bại ở phân vị 10, nên không
    phải ảnh thuận lợi nhất cũng không phải tệ nhất.
    """
    import cv2
    import numpy as np

    from common import scope
    from evaluate import load_test_set
    from metrics import confusion_matrix_image
    from predict import build_predictor

    ev = cfg["eval"]
    names = cfg["dataset"]["classes"]
    primary, _ = scope(cfg)
    items, _ = load_test_set(cfg, "bbox", None)

    predictors = {}
    for k in primary:
        try:
            predictors[k] = build_predictor(k, cfg, conf=ev["conf_threshold"])
        except SystemExit as e:
            log(f"bỏ qua {k}: {e}")
    if not predictors:
        return

    # recall lớp cỏ của từng ảnh, cho từng mô hình
    cache, per = {}, {k: [] for k in predictors}
    for idx, (meta, path, gb, gc) in enumerate(items):
        if (gc == 1).sum() == 0:            # ảnh không có cỏ thì recall vô định
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        cache[idx] = (meta, path, gb, gc)
        for k, run in predictors.items():
            pred = run(img)
            m = confusion_matrix_image(gb, gc, pred.boxes, pred.classes, pred.scores,
                                       num_classes=len(names),
                                       conf_threshold=ev["conf_threshold"],
                                       iou_threshold=ev["iou_threshold"])
            tp = m[1, 1]; fn = m[1, :].sum() - tp
            per[k].append((float(tp / (tp + fn)) if tp + fn else 0.0, idx))

    def pick(k, pct):
        v = sorted(per[k])
        return v[max(0, min(len(v) - 1, int(round(pct / 100 * (len(v) - 1)))))][1]

    def draw(ax, img, boxes, classes, scores=None, title=None):
        h, w = img.shape[:2]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i, (b, cl) in enumerate(zip(boxes, classes)):
            col = CLASS_COLOR[names[int(cl)]]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, edgecolor=col, linewidth=1.1))
            label = names[int(cl)] if scores is None else f"{names[int(cl)]} {scores[i]:.2f}"
            right = b[2] > w * 0.55
            tx = min(b[2], w - 2) if right else max(b[0], 2)
            ty = b[1] - 4 if b[1] > 12 else min(b[3] + 9, h - 2)
            ax.text(tx, ty, label, fontsize=5.0, color="white",
                    ha="right" if right else "left", clip_on=True,
                    bbox=dict(facecolor=col, pad=0.6, edgecolor="none"))
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=7.5)

    picked = {}
    cols = len(predictors)
    # Springer: rộng tối đa 174 mm, cao tối đa 234 mm. Bốn hàng ảnh vuông nên
    # chiều cao là ràng buộc; chọn khổ vừa cả hai rồi để dpi quyết định số điểm.
    w_in, h_in = 174 / 25.4, 232 / 25.4
    fig, axes = plt.subplots(4, cols, figsize=(w_in, h_in), squeeze=False)
    for c, (k, run) in enumerate(predictors.items()):
        for r_pair, (pct, tag) in enumerate(((50, TXT["qual_ok"]), (10, TXT["qual_bad"]))):
            idx = pick(k, pct)
            meta, path, gb, gc = cache[idx]
            img = cv2.imread(str(path))
            rec = dict((i, v) for v, i in per[k])[idx]
            # Hướng dẫn của Springer: "Do not include titles or captions within
            # your illustrations"; phần hình chỉ được đánh dấu bằng chữ thường.
            # Mọi mô tả chuyển hết sang chú thích trong bản thảo.
            n_panel = 2 * r_pair * cols + c
            draw(axes[2 * r_pair][c], img, gb, gc, None,
                 f"({chr(97 + n_panel)})")
            pred = run(img)
            keep = pred.scores > ev["conf_threshold"]
            # Phản biện nói đúng: hàng dự đoán không cho thấy LỖI ở đâu, người
            # đọc phải tự đối chiếu hai hàng. Vẽ thêm lên hàng dự đoán các cỏ
            # bị bỏ sót, bằng nét đứt, để lỗi hiện ra ngay trong khung.
            ax = axes[2 * r_pair + 1][c]
            draw(ax, img, pred.boxes[keep], pred.classes[keep], pred.scores[keep],
                 f"({chr(97 + n_panel + cols)})")
            miss = _missed_weed(gb, gc, pred.boxes[keep], pred.classes[keep],
                                pred.scores[keep], ev["iou_threshold"])
            for b in miss:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, edgecolor="#ffd600", linewidth=1.4,
                                       linestyle=(0, (4, 2)), zorder=5))
            picked.setdefault(idx, []).append(k)
    # Chú thích đầy đủ nằm trong bản thảo; ở đây chỉ cần chú giải màu một dòng,
    # và bỏ suptitle đặt ở y rất nhỏ vì nó để lại một dải trắng lớn dưới hình.
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(facecolor=CLASS_COLOR[n_], label=n_) for n_ in names]
               + [Line2D([0], [0], color="#ffd600", lw=1.6, linestyle=(0, (4, 2)),
                         label=TXT["qual_missed"])],
               loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "fig5_qualitative")
    # Hai mô hình có thể rơi vào CÙNG một ảnh ở phân vị 10 — phản biện bắt đúng
    # hiện tượng này ở ô (g) và (h). Ghi lại để chú thích nói rõ thay vì để
    # người đọc nghi ngờ hình bị lặp.
    import json as _json
    shared = {str(i): [MODEL_LABELS[k] for k in v]
              for i, v in picked.items() if len(v) > 1}
    (RESULTS / "qualitative_picks.json").write_text(
        _json.dumps({"shared_images": shared,
                     "n_panels_per_model": 2}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--confusion", action="store_true")
    ap.add_argument("--bars", action="store_true")
    ap.add_argument("--qualitative", action="store_true")
    ap.add_argument("--gsd", action="store_true",
                    help="Hình recall theo độ phân giải mặt đất")
    ap.add_argument("--curves", action="store_true",
                    help="Hình PR / FROC / ROC")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="số ảnh cho Hình 5")
    a = ap.parse_args()
    if not any([a.workflow, a.training, a.confusion, a.bars, a.qualitative,
                a.curves, a.gsd, a.all]):
        a.all = True

    cfg = load_cfg()
    if a.workflow or a.all:
        fig_workflow()
    if a.training or a.all:
        fig_training(cfg)
    if a.confusion or a.all:
        fig_confusion(cfg)
    if a.bars or a.all:
        fig_bars(cfg)
    if a.curves or a.all:
        fig_curves(cfg)
    if a.gsd or a.all: fig_gsd()
    if a.qualitative or a.all:
        fig_qualitative(cfg, a.n)


if __name__ == "__main__":
    main()
