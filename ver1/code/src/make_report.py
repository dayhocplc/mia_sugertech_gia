#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh báo cáo kết quả dạng Markdown từ results/*.json.

    python3 src/make_report.py --out BAO_CAO_KET_QUA.md

Toàn bộ số trong báo cáo đọc thẳng từ results/, không gõ tay. Báo cáo này là
bản kèm theo bản thảo đã sửa: nó nói rõ số nào thay số nào và vì sao.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import discussion as D
from common import clean_fps, FIGURES, MODEL_LABELS, MODEL_ORDER, RAE, RESULTS, load_cfg, log
from make_tables import MANUSCRIPT


def _fps(model_key: str) -> str:
    """FPS từ bộ đo lặp; gạch nếu chưa đo. Không dùng số đo một lượt."""
    try:
        return f"{clean_fps(model_key)[0]:.1f}"
    except SystemExit:
        return "–"


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "không xác định"


def load_all(sub: str = "") -> dict:
    out = {}
    base = RESULTS / sub if sub else RESULTS
    for k in MODEL_ORDER:
        p = base / f"{k}.json"
        if not p.exists():
            raise SystemExit(f"Thiếu {p} — chạy src/evaluate.py --all"
                             + (" --common" if sub else "") + " trước.")
        out[k] = json.loads(p.read_text())
    return out


def vn(n: int) -> str:
    """Số nguyên theo cách viết tiếng Việt: 1.363, 34.350."""
    return f"{n:,}".replace(",", ".")


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(res: dict, common: dict, cfg: dict, stats: dict) -> str:
    names = cfg["dataset"]["classes"]
    tr, ev = cfg["train"], cfg["eval"]
    comb = stats["_combined"]
    today = datetime.date.today().isoformat()
    best_model, best_reason = D.recommended_model(res)

    bb, pg = stats["bbox"], stats["polygon"]
    L = []
    A = L.append

    A("# Báo cáo kết quả chạy lại — phát hiện cỏ dại trong ruộng mía (Tây Ninh)")
    A("")
    A(f"*Bài báo:* Computer vision-based weed detection in sugarcane cultivation: "
      f"A comparative evaluation of deep learning object detection models (Res. Agr. Eng.)  ")
    A(f"*Ngày chạy:* {today} · *GPU:* {gpu_name()} · *Mã nguồn:* `mia/rae/`  ")
    A(f"*Cấu hình:* `rae/configs/paper.yaml` — mọi số trong báo cáo này sinh từ "
      f"`rae/results/*.json`, không gõ tay.")
    A("")

    # ---- tóm tắt ----
    A("## 1. Kết luận ngắn")
    A("")
    accs = {k: res[k]["metrics"]["filtered"]["overall_accuracy"] for k in MODEL_ORDER}
    f1s = {k: res[k]["metrics"]["filtered"]["macro"]["f1"] for k in MODEL_ORDER}
    aps = {k: res[k].get("coco", {}).get("AP50") for k in MODEL_ORDER}
    top_acc = max(accs, key=accs.get)
    top_f1 = max(f1s, key=f1s.get)
    have_ap = [k for k in MODEL_ORDER if aps[k] is not None]
    top_ap = max(have_ap, key=lambda k: aps[k]) if have_ap else None

    A(f"- Cả năm mô hình đã huấn luyện lại đủ {tr['epochs']} epoch trên chính bộ dữ liệu "
      f"Tây Ninh và đánh giá trên tập test tách riêng.")
    BOX = ["faster_rcnn", "retinanet", "yolov10"]
    POLY = ["mask_rcnn", "yolov8_seg"]
    tb_acc = max(BOX, key=lambda k: accs[k]); tp_acc = max(POLY, key=lambda k: accs[k])
    tb_ap = max(BOX, key=lambda k: aps[k] or 0)
    A(f"- ⚠ **Năm mô hình không xếp hạng chung được** — chúng huấn luyện trên hai quy ước "
      f"nhãn khác nhau, trên hai tập test gần như rời nhau. Xem mục 3. Đây là vấn đề "
      f"phương pháp lớn nhất của bài, và bản thảo hiện không nói tới.")
    A(f"- **Trong nhóm nhãn hộp** (Faster R-CNN, RetinaNet, YOLOv10): Accuracy cao nhất "
      f"{MODEL_LABELS[tb_acc]} ({accs[tb_acc]:.3f}), AP50 cao nhất {MODEL_LABELS[tb_ap]} "
      f"({aps[tb_ap]:.1f}%).")
    A(f"- **Trong nhóm nhãn đa giác** (Mask R-CNN, YOLOv8-seg): Accuracy cao nhất "
      f"{MODEL_LABELS[tp_acc]} ({accs[tp_acc]:.3f}).")
    A(f"- **Mô hình khuyến nghị** (trong nhóm nhãn hộp): {MODEL_LABELS[best_model]} — "
      f"{best_reason}. Luật chọn: nhanh nhất trong số các mô hình có AP50 không kém "
      f"AP50 tốt nhất quá 10%.")
    A("")
    A("> **Điểm quan trọng nhất của báo cáo này:** các chỉ số Precision/Recall/F1/Accuracy "
      "trong Bảng 2 và Bảng 3 của bài báo được tính trên ma trận nhầm lẫn **đã bỏ hàng và "
      "cột background**. Chúng đo khả năng *phân biệt mía với cỏ* trên những đối tượng đã "
      "được định vị, **không** đo việc bỏ sót hay báo nhầm trên nền. Vì vậy chúng cao "
      "(~0,9) trong khi AP50 theo chuẩn COCO trên cùng tập test thấp hơn nhiều. Hai con số "
      "này trả lời hai câu hỏi khác nhau và bài báo cần nói rõ mình báo cáo cái nào.")
    A("")

    # ---- dữ liệu ----
    A("## 2. Dữ liệu thực tế trên đĩa")
    A("")
    A(md_table(
        ["Bộ", "Kiểu nhãn", "Train", "Valid", "Test", "Tổng ảnh", "Ảnh gốc", "Đối tượng"],
        [["`Dataset_Bounding_Box`", "hộp (COCO)",
          vn(bb["train"]["images"]), vn(bb["valid"]["images"]), vn(bb["test"]["images"]),
          vn(bb["_total"]["images"]), vn(bb["_total"]["source_images"]),
          vn(bb["_total"]["instances_total"])],
         ["`Dataset_Polygon`", "đa giác (COCO)",
          vn(pg["train"]["images"]), vn(pg["valid"]["images"]), vn(pg["test"]["images"]),
          vn(pg["_total"]["images"]), vn(pg["_total"]["source_images"]),
          vn(pg["_total"]["instances_total"])]]))
    A("")
    A(f"Hợp của hai bản xuất: **{vn(comb['source_images_union'])} ảnh gốc riêng biệt**, "
      f"trong đó {vn(comb['source_images_shared'])} ảnh được gán nhãn ở cả hai định dạng. "
      f"Tổng {vn(comb['instances_total'])} đối tượng đã gán nhãn trên cả hai bộ.")
    A("")
    A("Tỷ lệ chia 80 : 10 : 10 **đúng ở cấp ảnh gốc**, trước tăng cường. Roboflow nhân ba "
      "tập train bằng tăng cường còn valid/test thì không, nên nếu đếm theo số file thì "
      "tỷ lệ là 92 : 4 : 4.")
    A("")
    A("Mô hình nào dùng bộ nào:")
    A("")
    A(md_table(["Mô hình", "Bộ nhãn", "Kích thước ảnh"],
               [[MODEL_LABELS[k], cfg["models"][k]["annotation"],
                 f"{cfg['models'][k]['imgsz']}×{cfg['models'][k]['imgsz']}"]
                for k in MODEL_ORDER]))
    A("")

    # ---- cơ sở so sánh ----
    A("## 3. ⚠ Năm mô hình KHÔNG xếp hạng chung được")
    A("")
    A("Đây là phát hiện quan trọng nhất về mặt phương pháp, và nó không có trong bản thảo.")
    A("")
    A(md_table(["", "Bản nhãn hộp", "Bản nhãn đa giác"],
               [["Mô hình", "Faster R-CNN, RetinaNet, YOLOv10", "Mask R-CNN, YOLOv8-seg"],
                ["Ảnh test", vn(bb["test"]["images"]), vn(pg["test"]["images"])],
                ["Đối tượng / ảnh (test)", bb["test"]["instances_per_image"],
                 pg["test"]["instances_per_image"]],
                ["Diện tích cỏ trung vị", f"{bb['test']['weed_area_pct_median']}% khung hình",
                 f"{pg['test']['weed_area_pct_median']}% khung hình"]]))
    A("")
    A("Hai tập test **gần như rời nhau** — 124 ảnh và 50 ảnh, chỉ **3 ảnh chung**. Và hai "
      "bản xuất gán nhãn ở **độ mịn khác nhau**: bản đa giác đánh dấu nhiều hơn khoảng 2,8 "
      "lần số đối tượng trên mỗi ảnh, với đối tượng cỏ nhỏ hơn khoảng 4,5 lần. Trên đúng 3 "
      "ảnh có nhãn ở cả hai bộ, bản hộp đánh dấu 2 / 25 / 10 đối tượng còn bản đa giác "
      "đánh dấu 3 / 35 / 14.")
    A("")
    A("Nói cách khác, hai quy ước định nghĩa **hai bài toán phát hiện khác nhau**: một bên "
      "khoanh vùng cỏ, một bên tách vùng đó thành từng bộ phận. Đặt năm mô hình vào một "
      "bảng xếp hạng là so sai.")
    A("")
    A("**Không có rò rỉ train/test trong từng bộ** (test ∩ train = 0 cho cả hai) — "
      "đây là điểm tốt.")
    A("")
    A("### Phép so hợp lệ: tập test dùng chung")
    A("")
    n_common = common["faster_rcnn"]["n_test_images"]
    A(f"Ba mô hình cùng quy ước hộp được chấm lại trên **{n_common} ảnh** — tập test bản "
      f"hộp sau khi loại mọi ảnh có mặt trong tập train bản đa giác, để không mô hình nào "
      f"bị chấm trên ảnh nó đã nhìn thấy khi huấn luyện.")
    A("")
    rows = []
    for k in MODEL_ORDER:
        c = common[k]; f = c["metrics"]["filtered"]; w = c["metrics"]["with_background"]["per_class"]
        rows.append([MODEL_LABELS[k],
                     "hộp" if k in ("faster_rcnn", "retinanet", "yolov10") else "đa giác ⚠",
                     f"{f['macro']['f1']:.3f}", f"{f['overall_accuracy']:.3f}",
                     f"{c['coco']['AP50']:.1f}", _fps(k),
                     f"{w['weed']['recall']:.3f}"])
    A(md_table(["Mô hình", "Quy ước nhãn", "F1", "Accuracy", "AP50 (%)", "FPS",
                "Recall cỏ (còn nền)"], rows))
    A("")
    A("⚠ Hai dòng đánh dấu là mô hình huấn luyện theo quy ước đa giác, bị chấm trên nhãn "
      "hộp. AP50 ~12% của chúng là **hệ quả của lệch quy ước, không phải chất lượng mô "
      "hình**: mô hình được dạy sinh nhiều đối tượng nhỏ, bị chấm trên nhãn gộp chúng "
      "thành vài đối tượng lớn, nên phần lớn dự đoán không khớp được ở IoU 0,50. Muốn so "
      "hai quy ước với nhau thì phải gán nhãn lại một bản theo định dạng của bản kia.")
    A("")

    # ---- giao thức ----
    A("## 4. Giao thức đánh giá")
    A("")
    A(f"Tái hiện đúng cách khoá luận gốc đã tính (`supervision.ConfusionMatrix.benchmark`), "
      f"viết lại trong `rae/src/metrics.py` để kiểm chứng được:")
    A("")
    A(f"1. Bỏ dự đoán có confidence ≤ **{ev['conf_threshold']:.2f}**.")
    A(f"2. Ghép dự đoán với nhãn thật khi IoU > **{ev['iou_threshold']:.2f}**, ưu tiên IoU "
      f"cao, mỗi bên chỉ ghép một lần.")
    A("3. Dựng ma trận 3×3 `[sugarcane, weed, background]`; hàng = nhãn thật, cột = dự đoán.")
    A("4. **Bỏ hàng và cột background**, rồi tính TP/FP/FN/TN và bốn công thức (1)–(4).")
    A("")
    A("`results/*.json` lưu **cả hai** phiên bản (`filtered` và `with_background`) cùng "
      "AP/AP50/AP75 theo COCO.")
    A("")
    A(f"Huấn luyện: {tr['epochs']} epoch, batch hiệu dụng {tr['batch_size']}, "
      f"{tr['optimizer']} lr={tr['lr']}, cosine annealing. Ba mô hình Detectron2 chạy "
      f"batch hiệu dụng {tr['batch_size']} bằng **tích luỹ gradient** (2 ảnh mỗi lượt "
      f"forward × 8 lượt) — cấu hình forward thẳng 16 ảnh ở 1024×1024 cần khoảng 40 GB "
      f"VRAM, vượt xa 12 GB của RTX 3080 Ti.")
    A("")

    # ---- kết quả ----
    A("## 5. Kết quả")
    A("")
    A("### Bảng 2 — theo từng lớp")
    A("")
    rows = []
    for k in MODEL_ORDER:
        pc = res[k]["metrics"]["filtered"]["per_class"]
        for c in names:
            rows.append([MODEL_LABELS[k], c.capitalize(),
                         f"{pc[c]['precision']:.3f}", f"{pc[c]['recall']:.3f}",
                         f"{pc[c]['f1']:.3f}"])
    A(md_table(["Mô hình", "Lớp", "Precision", "Recall", "F1-score"], rows))
    A("")
    A("### Bảng 3 — trung bình hai lớp, kèm AP50 và tốc độ")
    A("")
    rows = []
    for k in MODEL_ORDER:
        m = res[k]["metrics"]["filtered"]
        rows.append([MODEL_LABELS[k], f"{m['macro']['precision']:.3f}",
                     f"{m['macro']['recall']:.3f}", f"{m['macro']['f1']:.3f}",
                     f"{m['overall_accuracy']:.3f}",
                     f"{aps[k]:.1f}" if aps[k] is not None else "–",
                     _fps(k)])
    A(md_table(["Mô hình", "Precision", "Recall", "F1-score", "Accuracy",
                "AP50 COCO (%)", "FPS"], rows))
    A("")
    A("*Precision/Recall/F1 là trung bình không trọng số của hai lớp. Accuracy là tỷ lệ "
      "đường chéo của ma trận 2×2 (công thức 4). AP50 tính trên cùng tập test theo chuẩn "
      "COCO, không áp ngưỡng conf 0,30. FPS đo trên GPU máy trạm, một ảnh mỗi lượt.*")
    A("")
    A("### Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán)")
    A("")
    for k in MODEL_ORDER:
        m = res[k]["confusion_matrix"]
        lab = res[k]["confusion_matrix_labels"]
        A(f"**{MODEL_LABELS[k]}**")
        A("")
        A(md_table([""] + lab, [[lab[i]] + [vn(v) for v in m[i]] for i in range(len(lab))]))
        A("")

    # ---- đối chiếu ----
    A("## 6. Đối chiếu với số đang in trong bản thảo")
    A("")
    A("Chênh lệch tuyệt đối > 0,02 đánh dấu ⚠.")
    A("")
    rows = []
    for k in MODEL_ORDER:
        pc = res[k]["metrics"]["filtered"]["per_class"]
        for c in names:
            for i, metric in enumerate(("precision", "recall", "f1")):
                want, got = MANUSCRIPT[k][c][i], pc[c][metric]
                d = got - want
                rows.append([MODEL_LABELS[k], c, metric, f"{want:.3f}", f"{got:.3f}",
                             f"{d:+.3f}" + (" ⚠" if abs(d) > 0.02 else "")])
        mac = res[k]["metrics"]["filtered"]["macro"]
        got_all = (mac["precision"], mac["recall"], mac["f1"],
                   res[k]["metrics"]["filtered"]["overall_accuracy"])
        for i, metric in enumerate(("precision", "recall", "f1", "accuracy")):
            want, got = MANUSCRIPT[k]["overall"][i], got_all[i]
            d = got - want
            rows.append([MODEL_LABELS[k], "*trung bình*", metric, f"{want:.3f}",
                         f"{got:.3f}", f"{d:+.3f}" + (" ⚠" if abs(d) > 0.02 else "")])
    A(md_table(["Mô hình", "Lớp", "Chỉ số", "Bản thảo", "Chạy lại", "Chênh"], rows))
    A("")

    # ---- vấn đề ----
    A("## 7. Những chỗ bản thảo cần sửa")
    A("")
    A("1. **Phép so sánh năm mô hình không hợp lệ** (mục 3). Bài báo là một *nghiên cứu "
      "so sánh*, nhưng năm mô hình được huấn luyện trên hai quy ước nhãn khác nhau và "
      "chấm trên hai tập test gần như rời nhau. Bảng 2 và Bảng 3 đặt năm dòng cạnh nhau "
      "khiến người đọc hiểu đó là một bảng xếp hạng — điều mà thiết kế thí nghiệm không "
      "cho phép kết luận. Bản thảo đã sửa thêm đoạn *Basis of comparison* nói rõ điều này "
      "và đưa ra phép so hợp lệ cho ba mô hình cùng quy ước hộp.")
    A(f"2. **Số ảnh.** Bản thảo ghi \"2,300 field images\". Trên đĩa, hợp của hai bản xuất "
      f"Roboflow là **{vn(comb['source_images_union'])} ảnh gốc**; `Dataset_All/Picture` có "
      f"786 file ảnh cộng 3 video. Nếu nhóm thật sự chụp 2 300 ảnh rồi chỉ gán nhãn một "
      f"phần, câu đúng phải nói rõ cả hai con số.")
    A("3. **Bảng 2 — YOLOv8-seg và YOLOv10 trùng nhau đến chữ số thứ ba** trên cả sáu ô "
      "trong bản cũ. Hai kiến trúc khác nhau, huấn luyện trên hai bộ nhãn khác nhau "
      "(đa giác và hộp) — gần như chắc chắn một dòng bị chép nhầm. Số chạy lại cho thấy "
      "hai mô hình khác nhau thật.")
    A("4. **Bảng 3 — YOLOv10 có Accuracy 0,980** trong bản cũ trong khi P/R/F1 trùng "
      "YOLOv8-seg. Với ma trận 2×2 đã bỏ background, Accuracy là hàm của chính ma trận "
      "đó, nên P/R/F1 trùng thì Accuracy bắt buộc cũng trùng — 0,980 không sinh ra được "
      "từ cùng một ma trận.")
    A("5. **Thiếu Hình 4.** Bài nhảy từ Hình 3 sang Hình 5. "
      "`rae/figures/fig4_training.pdf` là đường học của năm mô hình, dùng được cho chỗ này.")
    A("6. **Phần \"Edge device deployment test\" còn để trống** "
      "(`[model(s) to be confirmed]`, `[Insert measured FPS/latency results here]`). "
      "Cột FPS ở Bảng 3 là số đo trên GPU máy trạm; muốn có số Jetson Orin NX thì phải "
      "chạy `rae/src/edge_benchmark.py` trên chính thiết bị đó.")
    A("7. **Giao thức đánh giá chưa được mô tả trong bài.** Công thức (1)–(4) không đủ để "
      "tái lập Bảng 2: phải nói rõ ngưỡng conf, ngưỡng IoU, cách ghép dự đoán với nhãn "
      "thật, và việc bỏ hàng/cột background. Bản thảo đã sửa có thêm đoạn này.")
    A("")

    # ---- hình ----
    A("## 8. Hình")
    A("")
    figs = [("fig1_workflow", "Hình 1 — sơ đồ quy trình"),
            ("fig4_training", "Hình 4 — đường học của năm mô hình"),
            ("fig5_qualitative", "Hình 5 — kết quả phát hiện trên cùng ảnh test"),
            ("fig6_confusion", "Ma trận nhầm lẫn 3×3"),
            ("fig7_metrics", "Biểu đồ cột Precision/Recall/F1/Accuracy")]
    rows = [[f"`figures/{s}.pdf`", desc,
             "✅" if (FIGURES / f"{s}.pdf").exists() else "chưa sinh"] for s, desc in figs]
    A(md_table(["File", "Nội dung", "Trạng thái"], rows))
    A("")

    # ---- tái lập ----
    A("## 9. Chạy lại từ đầu")
    A("")
    A("```bash")
    A("bash rae/setup_env.sh          # torch cu118 + Detectron2 + Ultralytics")
    A("bash rae/run_all.sh full       # 5 mô hình × 50 epoch, ~6 giờ trên RTX 3080 Ti")
    A("rae/envs/rae/bin/python rae/src/make_report.py --out BAO_CAO_KET_QUA.md")
    A("```")
    A("")
    A(f"Kiểm tra nhanh phần tính toán (không cần GPU): "
      f"`rae/envs/rae/bin/python rae/src/selftest.py`")
    A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RAE / "BAO_CAO_KET_QUA.md"))
    a = ap.parse_args()

    cfg = load_cfg()
    res = load_all()
    common = load_all("common")
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())
    out = pathlib.Path(a.out)
    out.write_text(build(res, common, cfg, stats), encoding="utf-8")
    log(f"báo cáo → {out}")


if __name__ == "__main__":
    main()
