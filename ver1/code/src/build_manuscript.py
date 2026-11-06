#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh TRỌN bản thảo tiếng Anh từ dữ liệu, thay vì vá bản cũ từng chỗ.

    python3 src/build_manuscript.py

Vì sao viết lại thay vì vá tiếp: bản thảo đã qua nhiều vòng sửa cục bộ, và mỗi
vòng lại sinh ra một mâu thuẫn mới giữa đoạn cũ và đoạn mới (số liệu hai nguồn,
chú thích bảng không khớp nội dung, viện dẫn bảng không tồn tại). Sinh trọn bài
từ một nguồn số liệu duy nhất thì những lỗi đó không thể xảy ra về mặt cấu trúc.

Đầu ra:
  manuscript/RAE_manuscript.md    bản đọc được, dùng để rà soát
  manuscript/RAE_manuscript.html  bản trung gian
  manuscript/RAE_manuscript.docx  bản nộp (chuyển bằng LibreOffice)

MỌI con số đi qua discussion.seed_mean / clean_fps — không đọc thẳng file kết quả.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import discussion as D
from common import (MODEL_LABELS, RAE, RESULTS, clean_fps, load_cfg, log, scope,
                    throughput_requirement)

OUT = RAE / "manuscript"
OUT.mkdir(exist_ok=True)

# Mỗi hình đặt ngay sau đoạn viện dẫn nó. `anchor` là phần đầu đoạn đó; nếu
# không tìm thấy thì báo lỗi chứ không im lặng bỏ qua — một phép ghép chuỗi
# không khớp đã từng làm rơi mất nguyên một bảng.
FIGURES = [
    # Ma trận nhầm lẫn là hình trung tâm của bài: chính nó cho thấy dải mà quy
    # ước bỏ background xoá đi. Đưa nó xuống Phụ lục để tiết kiệm trang là sai
    # chỗ, nên nó nằm ở bài chính. Chỉ đường học còn ở Phụ lục (Figure S1), vì
    # đó là tài liệu hỗ trợ.
    ("fig1_workflow", "**Study area and image dataset.**",
     "Figure 1. Workflow of the study, from image acquisition and annotation through "
     "model training to evaluation. The two annotation conventions run in parallel to "
     "separate test partitions and are never ranked against each other."),
    ("fig6_confusion", "**The deleted band.**",
     "Figure 2. Confusion matrices for the full variant on the common test partition, "
     "first training run (seed 0); rows are ground truth and columns prediction. The "
     "dashed outline marks the background row and column, which the background-excluded "
     "convention deletes before computing its metrics: the background column counts "
     "annotated instances that produced no detection, and the background row counts "
     "detections matching no annotation. Weed-class F1-score under both conventions is "
     "printed beneath each matrix. Counts are from a single run and so differ slightly "
     "from the seed-averaged values in Table 3."),
    ("fig8_curves", "**The whole operating curve, not one point.**",
     "Figure 3. Operating characteristics of the three detectors for the weed class in "
     "the full variant, first training run (seed 0). (a) precision–recall curve, the "
     "filled circle marking the working threshold of 0.30. (b) free-response operating "
     "characteristic: recall against the mean number of false weed detections per image, "
     "which is the form of operating characteristic defined for a detection task, since "
     "the unbounded set of object-free image locations gives no usable false-positive "
     "rate. (c) receiver operating characteristic for the confidence score as a "
     "discriminator between weed detections that match an annotated weed and those that "
     "do not, with the area under each curve; the dotted diagonal is chance. All three "
     "curves use the same confidence grid, 0.05 to 0.95 in steps of 0.01, and all "
     "detections were retained down to a score of 0.05 at inference, so the curves end "
     "at different points because the models reach different recall, not because they "
     "were sampled differently."),
    ("fig9_gsd", "**Does the number of pixels on a weed actually govern recall?**",
     "Figure 4. Weed-class recall when image detail is reduced to simulate a camera "
     "mounted further from the canopy. Each test image is downsampled and resampled "
     "back to its original size, so the network input, the annotations and the detector "
     "are unchanged and only the detail carried by each object is lost; the lower axis "
     "gives the resulting ground sample distance as a multiple of that of the "
     "acquisition and the upper axis the camera height that would produce it, assuming "
     "ground sample distance scales linearly with height from the 140 cm used here. "
     "(a) all three detectors, error bars one standard deviation over the same "
     "converged training runs as the rest of the paper. (b) the same for Faster R-CNN "
     "split by quartile of annotated weed area, Q1 smallest. The experiment runs only "
     "towards coarser acquisition, because the source images were exported at "
     "1280 × 1280 and no finer version exists."),
    ("fig5_qualitative", "**Qualitative results.**",
     "Figure 5. Detection results on the test set, first training run (seed 0). For each "
     "model a success case and a failure case are shown as a pair: ground-truth "
     "annotation above, prediction with confidence score below. Green boxes: sugarcane; "
     "red boxes: weed; a yellow dashed box on a prediction panel marks an annotated "
     "weed the model did not detect, so the misses can be read without comparing "
     "panels. The success case is the test image at the median of per-image "
     "weed-class Recall for that model and the failure case the image at the tenth "
     "percentile, so neither is the most favourable nor the most unfavourable example. "
     "Where the selection rule returned the same test image for two models, their "
     "panels show one image scored twice rather than two images; this is stated with "
     "the figure. No scale bar is given because the ground footprint of the frame was "
     "not measured."),
]


def _or1(journal: str) -> str:
    """Sugar Tech yêu cầu dẫn TỆP phụ lục bằng tên "Online Resource N".

    Hướng dẫn: 'Refer to the supplementary files as "Online Resource"'. Đánh số
    bảng/hình bên trong tệp vẫn là S1, S2…; chỉ lần dẫn ĐẦU TIÊN cần nói rõ
    chúng nằm trong tệp nào.
    """
    return " of Online Resource 1" if journal in ("sugartech", "aiia") else ""


def _detcap_clause() -> str:
    """Câu tường thuật thí nghiệm nâng trần số phát hiện, sinh từ số đo thật."""
    f = RESULTS / "detcap_analysis.json"
    if not f.exists():
        return "was not performed."
    d = json.loads(f.read_text())
    lo, hi = (d["by_cap"][str(c)] for c in d["caps"])
    return (f"added {hi['n_raw_detections']['mean'] - lo['n_raw_detections']['mean']:.0f} "
            "detections across the partition and changed AP50 by "
            f"{d['delta_ap50']:+.2f} points, weed-class recall by "
            f"{d['delta_recall']:+.3f} and weed-class F1-score by {d['delta_f1']:+.3f} "
            f"(Supplementary Table S15). The cap is reached only in the low-confidence "
            "tail; at the working threshold the most this model returns on any test "
            f"image is {hi['max_detections_per_image']['mean']:.0f} detections, so none "
            "of the figures reported here is bounded by it.")


def _cfg_name(cfg: dict, key: str) -> str:
    """Tên cấu hình như người đọc cần: R50-FPN 3x, yolov10s, … chứ không phải đường dẫn."""
    c = str(cfg["models"][key].get("config", ""))
    if c.endswith(".yaml"):
        stem = c.rsplit("/", 1)[-1][:-5]
        return stem.replace("_", "-").replace("faster-rcnn-", "").replace("retinanet-", "") \
                   .replace("mask-rcnn-", "").replace("R-50-FPN", "ResNet-50-FPN")
    return c[:-3] if c.endswith(".pt") else c


def figure_block(stem: str, caption: str) -> str:
    """Khối hình: ảnh nhúng + chú thích ngay dưới."""
    src = RAE / "figures" / f"{stem}.png"
    if not src.exists():
        raise SystemExit(f"Thiếu file hình: {src}")
    return f"![{stem}]({src.resolve()})\n\n{caption}\n"


def _place(blocks: list[str], which: list[tuple]) -> list[str]:
    out, placed = [], set()
    for b in blocks:
        out.append(b)
        for stem, anchor, cap in which:
            if stem not in placed and b.lstrip().startswith(anchor):
                out.append(figure_block(stem, cap))
                placed.add(stem)
    missing = [s for s, _a, _c in which if s not in placed]
    if missing:
        raise SystemExit(f"Không đặt được hình {missing} — mốc đoạn văn không khớp")
    return out


def place_figures(blocks: list[str]) -> list[str]:
    """Hình 1–2 nằm trong phần Methods."""
    return _place(blocks, FIGURES[:1])


def place_figures_results(blocks: list[str]) -> list[str]:
    """Hình 3–4 nằm trong phần Results."""
    return _place(blocks, FIGURES[1:])

# Tiêu đề: tạp chí giới hạn 100 ký tự và cấm phụ đề. Bản cũ dài 142 ký tự và
# có phụ đề sau dấu hai chấm.
TITLE = "Detection completeness limits deep learning weed detection in field-grown sugarcane"

# Tạp chí yêu cầu bản thảo ĐÃ LÀM MÙ: tên tác giả, cơ quan và địa chỉ liên hệ
# nằm ở Title Page nộp riêng, không nằm trong tệp bản thảo. Hai hằng số này
# dùng để sinh tệp title page, không ghép vào bản thảo nữa.
AUTHORS = ("Nguyen Huynh Truong Gia<sup>1</sup>, Nguyen Duc Khuyen<sup>1</sup>, "
           "Vo Van Qui<sup>1</sup>, Tran Quoc Khanh<sup>1</sup>, "
           "Tran Thi Kim Nga<sup>1,*</sup>")
AFFIL = ("<sup>1</sup>Faculty of Mechanical Engineering – Technology, "
         "Nong Lam University, Ho Chi Minh City, Vietnam")
CORRESP = "*Corresponding author*: ttknga@hcmuaf.edu.vn"

KEYWORDS = ("convolutional neural network; deep learning; computer vision; "
            "weed detection; sugarcane; detection recall")

# Khai báo công cụ trí tuệ nhân tạo. Hai tạp chí đòi khác nhau:
#   Sugar Tech  — "use of an LLM should be properly documented in the Methods"
#   AIIA        — phải nêu "the name of the model or tool, version and extension
#                 numbers, and manufacturer"
# Hai công cụ đã tham gia quá trình làm bài này, nên khai cả hai; phiên bản
# chính xác thì chỉ nhóm tác giả biết, và để trống có chủ ý thay vì bịa ra một
# con số — check_consistency.py sẽ chặn bản thảo cho tới khi được điền.
AI_TOOL_SENTENCE = {
    "sugartech":
        "A large language model (Claude, Anthropic) was used as a programming and "
        "drafting assistant in preparing this work — writing and revising the "
        "analysis scripts released in Online Resource 2, running those analyses, "
        "and drafting and revising manuscript text.",
    "aiia":
        "Two large language models were used as programming and drafting assistants "
        "in preparing this work: Qwen2.5-Coder-32B-Instruct (Alibaba Cloud) and "
        "Claude Opus 5 (model identifier claude-opus-5, Anthropic), the latter from "
        "October to November 2026. Their use covered writing and revising the "
        "analysis scripts released in Online Resource 2, running those analyses, and "
        "drafting and revising manuscript text.",
}

REFERENCES = [
    # Lưu ở dạng chuẩn "Tác-giả (năm): …" như mọi mục khác; journal_style.py
    # chuyển sang kiểu Springer hoặc Elsevier tuỳ tạp chí đích. Trước đây mục này
    # được nhập sẵn kiểu Springer nên bộ chuyển không nhận ra và bỏ qua.
    "Azghadi M.R., Olsen A., Wood J., Saleh A., Calvert B., Granshaw T., Fillols E., "
    "Philippa B. (2025): Precision robotic spot-spraying: Reducing herbicide use and "
    "enhancing environmental outcomes in sugarcane. Computers and Electronics in "
    "Agriculture, 235: 110365.",
    "Bolya D., Foley S., Hays J., Hoffman J. (2020): TIDE: A general toolbox for identifying object detection errors. In: Proceedings of the 16th European Conference on Computer Vision (ECCV), Glasgow, Aug 23–28, 2020: 558–573.",
    "He K., Zhang X., Ren S., Sun J. (2016): Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, June 27–30, 2016: 770–778.",
    "Jocher G., Chaurasia A., Qiu J. (2023): Ultralytics YOLO (Version 8.3.253). Computer software. Available at https://github.com/ultralytics/ultralytics (accessed Mar 10, 2025).",
    "Lin T.-Y., Dollár P., Girshick R., He K., Hariharan B., Belongie S. (2017a): Feature pyramid networks for object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, July 21–26, 2017: 2117–2125.",
    "Loshchilov I., Hutter F. (2017): SGDR: Stochastic gradient descent with warm restarts. In: Proceedings of the 5th International Conference on Learning Representations (ICLR), Toulon, Apr 24–26, 2017.",
    "Loshchilov I., Hutter F. (2019): Decoupled weight decay regularization. In: Proceedings of the 7th International Conference on Learning Representations (ICLR), New Orleans, May 6–9, 2019.",
    "NVIDIA Corporation (2023): Jetson Orin NX Series Module Data Sheet, DS-10712-001_v1.3. Santa Clara, NVIDIA Corporation.",
    "Oksuz K., Cam B.C., Akbas E., Kalkan S. (2018): Localization recall precision (LRP): A new performance metric for object detection. In: Proceedings of the 15th European Conference on Computer Vision (ECCV), Munich, Sep 8–14, 2018: 504–519.",
    "Partel V., Kakarla S.C., Ampatzidis Y. (2019): Development and evaluation of a low-cost and smart technology for precision weed management utilizing artificial intelligence. Computers and Electronics in Agriculture, 157: 339–350.",
    "Paszke A., Gross S., Massa F., Lerer A., Bradbury J., Chanan G., Killeen T., Lin Z., Gimelshein N., Antiga L., Desmaison A., Köpf A., Yang E., DeVito Z., Raison M., Tejani A., Chilamkurthy S., Steiner B., Fang L., Bai J., Chintala S. (2019): PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems 32, Vancouver, Dec 8–14, 2019: 8024–8035.",
    "Wu Y., Kirillov A., Massa F., Lo W.-Y., Girshick R. (2019): Detectron2. Computer software. Available at https://github.com/facebookresearch/detectron2 (accessed Mar 10, 2025).",
    "de Souza M.F., do Amaral L.R., de Medeiros Oliveira S.R., Coutinho M.A.N., Netto C.F. "
    "(2020): Spectral differentiation of sugarcane from weeds. Biosystems Engineering, 190: 41–46.",
    "Donga T.K., Eklo O.M. (2018): Environmental load of pesticides used in conventional "
    "sugarcane production in Malawi. Crop Protection, 108: 71–77.",
    "dos Santos Ferreira A., Freitas D.M., Da Silva G.G., Pistori H., Folhes M.T. (2017): "
    "Weed detection in soybean crops using ConvNets. Computers and Electronics in "
    "Agriculture, 143: 314–324.",
    "He K., Gkioxari G., Dollár P., Girshick R. (2017): Mask R-CNN. Proceedings of the 2017 "
    "IEEE International Conference on Computer Vision (ICCV), Venice: 2980–2988.",
    "Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. (2017b): Focal loss for dense object "
    "detection. Proceedings of the 2017 IEEE International Conference on Computer Vision "
    "(ICCV), Venice: 2999–3007.",
    "Lin T.-Y., Maire M., Belongie S., Hays J., Perona P., Ramanan D., Dollár P., "
    "Zitnick C.L. (2014): Microsoft COCO: Common objects in context. Proceedings of the "
    "European Conference on Computer Vision (ECCV): 740–755.",
    "Mini G.A., Sales D.O., Luppe M. (2020): Weed segmentation in sugarcane crops using "
    "Mask R-CNN through aerial images. Proceedings of the 2020 International Conference on "
    "Computational Science and Computational Intelligence (CSCI): 485–491.",
    "Mirajkar S.J., Devarumath R.M., Nikam A.A., Sushir K.V., Babu H., Suprasanna P. (2019): "
    "Sugarcane (Saccharum spp.): Breeding and genomics. In: Al-Khayri J., Jain S., "
    "Johnson D. (eds): Advances in Plant Breeding Strategies: Industrial and Food Crops. "
    "Springer, Cham.",
    "Moazzam S.I., Khan U.S., Qureshi W.S., Tiwana M.I., Rashid N., Alasmary W.S., Hamza A. "
    "(2021): A patch-image based classification approach for detection of weeds in sugar "
    "beet crop. IEEE Access, 9: 121698–121715.",
    "Ren S., He K., Girshick R., Sun J. (2017): Faster R-CNN: Towards real-time object "
    "detection with region proposal networks. IEEE Transactions on Pattern Analysis and "
    "Machine Intelligence, 39: 1137–1149.",
    "Singh V., Singh D., Kumar H. (2024): Efficient application of deep neural networks for "
    "identifying small and multiple weed patches using drone images. IEEE Access, 12: "
    "71982–71996.",
    "Sohail R., Nawaz Q., Hamid I., Gilani S.M.M., Mumtaz I., Mateen A., Chauhdary J.N. "
    "(2021): An analysis on machine vision and image processing techniques for weed "
    "detection in agricultural crops. Pakistan Journal of Agricultural Sciences, 58: 187–204.",
    "Suganthi M., Muthukrishnan P., Chinnusamy C. (2019): Crop weed competition in "
    "sugarcane – A review. Agricultural Reviews, 40: 239.",
    "Sun C., Zhang M., Zhou M., Zhou X. (2024): An improved transformer network with "
    "multi-scale convolution for weed identification in sugarcane field. IEEE Access, 12: "
    "31168–31181.",
    "Wang A., Chen H., Liu L., Chen K., Lin Z., Han J., Ding G. (2024): YOLOv10: Real-time "
    "end-to-end object detection. Advances in Neural Information Processing Systems, 37: "
    "107984–108011.",
    "Yu J., Schumann A.W., Cao Z., Sharpe S.M., Boyd N.S. (2019): Weed detection in "
    "perennial ryegrass with deep learning convolutional neural network. Frontiers in Plant "
    "Science, 10: 1422.",
    "Zhang X., Cui J., Liu H., Han Y., Ai H., Dong C., Zhang J., Chu Y. (2023): Weed "
    "identification in soybean seedling stage based on optimized Faster R-CNN algorithm. "
    "Agriculture, 13: 175.",
]


# ---------------------------------------------------------------- tiện ích

def md_table(header: list[str], rows: list[list], caption: str) -> str:
    out = [caption, ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def ci(boot: dict, model: str, metric: str, fmt: str) -> str:
    m = (boot.get("models", {}) or {}).get(model, {})
    if metric not in m:
        return ""
    lo, hi = m[metric]["ci95"]
    return f" [{fmt.format(lo)}, {fmt.format(hi)}]"


# ---------------------------------------------------------------- Methods

def load_review() -> tuple[dict, dict]:
    """Phân tích bổ sung cho vòng phản biện: review_analysis + leakage_analysis.

    Hai tệp này do src/review_analysis.py và src/leakage_analysis.py sinh ra.
    Thiếu một trong hai thì bản thảo không dựng được, vì các con số mới (chỉ số
    dạng full, phân tầng kích thước, Holm, bootstrap hai tầng, ngày chụp, recipe
    augmentation) đều nằm trong đó.
    """
    ra = RESULTS / "review_analysis.json"
    la = RESULTS / "leakage_analysis.json"
    for f, script in ((ra, "review_analysis.py"), (la, "leakage_analysis.py")):
        if not f.exists():
            raise SystemExit(f"Thiếu {f.name}. Chạy trước:  python3 src/{script}")
    rev = json.loads(ra.read_text())
    lk = json.loads(la.read_text())
    lk = dict(lk, **lk.get("augmentation", {}))
    return rev, lk


def methods(cfg: dict, stats: dict, res: dict, seedvar: dict, boot: dict,
            journal: str = "rae") -> list[str]:
    """Mục Phương pháp, dùng chung cho cả hai tạp chí.

    journal="sugartech": bản Sugar Tech có mục Discussion riêng mang phần diễn
    giải hai biến thể ma trận nhầm lẫn, nên ở đây chỉ giữ cơ chế tính và bỏ
    phần bình luận — nếu không, cùng một lập luận xuất hiện hai lần trong bài.
    """
    tr, ev = cfg["train"], cfg["eval"]
    t = throughput_requirement(cfg)
    bb, pg = stats["bbox"], stats["polygon"]
    comb = stats["_combined"]
    primary, secondary = scope(cfg)
    n_seeds = len([e for e in seedvar["faster_rcnn"]["runs"] if e.get("status") == "ok"])
    rev, lk = load_review()
    # Springer và Elsevier viết tắt "Fig."; RAE viết đủ "Figure".
    fig_label = "Fig." if journal in ("sugartech", "aiia") else "Figure"
    n_conv = ", ".join(
        str(len([e for e in seedvar[k]["runs"] if e.get("status") == "ok"]))
        for k in primary)

    P = []
    P.append(
        "**Study area and image dataset.** The workflow from acquisition to evaluation "
        f"is summarised in {fig_label} 1. Field images of sugarcane (cultivar KK3) and "
        "weeds "
        "were collected at Tan Hung farm, Chau Thanh district, Tay Ninh province, "
        "Vietnam, with a handheld iPhone 12 held about 140 cm above the ground, optical "
        f"axis vertical. Acquisition took place in a single day, on {lk['capture_date']}, "
        f"in three sessions ({lk['sessions_desc']}) on three separate plots, two with a "
        "recorded crop age of 40 and 60 days after planting and the third a mixed "
        "stand; the images are therefore correlated in illumination and in plot, and "
        "the dataset does not sample growth stage or season as a designed factor. The "
        "weed flora comprised purple nutsedge (*Cyperus rotundus* L.), Bermuda grass "
        "(*Cynodon dactylon* (L.) Pers.), crowfoot grass (*Dactyloctenium aegyptium* "
        "(L.) Willd.) and a fourth species known locally as ginger-leaf grass. In total "
        f"{comb['source_images_union']:,} source images were annotated. The camera wrote "
        "frames of 4080 × 3072 pixels, three in five of them in portrait orientation, "
        "rescaled on export to a square 1280 × 1280 pixels "
        "without preserving the aspect ratio.")

    P.append(
        "**Image annotation and data augmentation.** All images were annotated on the "
        "Roboflow platform (Roboflow Inc., Des Moines, IA, USA) in two conventions, a "
        "bounding-box set and a polygon set, both classes in both. The bounding-box "
        f"set draws on {bb['_total']['source_images']:,} source images and the polygon "
        f"set on {pg['_total']['source_images']:,}, {comb['source_images_shared']:,} "
        "appearing in both. Each was divided 80:10:10 at the level of source images "
        "before augmentation, so no augmented copy of a training image can appear in a "
        "validation or test partition. Augmentation was applied to the training "
        "partition only, each source image yielding three versions generated by "
        "independently sampling a horizontal and a vertical flip each with probability "
        "0.5, one of four 90° rotations with equal probability, a random crop of 0 to "
        "20% of the image, brightness within ±15%, exposure within ±10%, and "
        "salt-and-pepper noise on 1.01% of pixels. Two consequences are not apparent "
        "from the image counts: the crop is a scale augmentation rather than a rigid "
        "transformation, so the three versions are not related by any exact "
        f"axis-aligned mapping and in {lk['n_sources_count_differs']} of the "
        f"{bb['train']['source_images']:,} training source images "
        f"({lk['pct_sources_count_differs']:.0f}%) they carry different instance counts "
        "because boxes cropped past the frame boundary are dropped on export; and all "
        "three are random draws, so the unmodified original is not itself retained. "
        f"The bounding-box set therefore holds {bb['train']['images']:,} training, "
        f"{bb['valid']['images']:,} validation and {bb['test']['images']} test images "
        f"carrying {bb['_total']['instances_total']:,} annotated instances, and the "
        f"polygon set {pg['train']['images']:,}, {pg['valid']['images']} and "
        f"{pg['test']['images']} carrying {pg['_total']['instances_total']:,}. "
        "Augmentation internal to the training pipelines was disabled apart from "
        "horizontal flipping, so every model saw the same data.")

    # Chất lượng nhãn: không có lần gán nhãn độc lập thứ hai nên không có hệ số
    # đồng thuận. Phản biện vòng hai nêu đúng rằng mọi con số recall và precision
    # của bài đứng trên bộ nhãn này, nên phải nói rõ quy trình và đo được gì thì
    # đo. R18 đo độ hạt trên chính nhãn; phần quy trình là thông tin của tác giả.
    ag = rev["annotation_granularity"]
    P.append(
        "**Annotation procedure and its granularity.** The images were annotated by three "
        "people on the Roboflow platform, after a common set of conventions had been "
        "agreed between them. All annotations were then reviewed and inconsistencies "
        "corrected, any disagreement remaining at that point being settled by the "
        "reviewer. Because the review was applied to the same set of labels rather than "
        "as an independent second annotation, no inter-annotator agreement coefficient "
        "can be given, and every "
        "recall and precision figure below is conditional on this single set of labels. "
        "What can be measured is the granularity of the convention that was used, and it "
        "bears on how the results should be read. On the test partition the median "
        f"sugarcane box covers {ag['sugarcane_box_frame_fraction']['median'] * 100:.0f}% of "
        "the frame and "
        f"{ag['sugarcane_box_frame_fraction']['frac_above_half_frame'] * 100:.0f}% of them "
        "cover more than half of it, the largest the whole frame; "
        f"{ag['sugarcane_box_pairs_overlapping_above_half']['fraction'] * 100:.0f}% of "
        "pairs of sugarcane boxes overlap by more than half the smaller box; and "
        f"{ag['weed_boxes_inside_a_sugarcane_box']['n']} of the "
        f"{ag['weed_boxes_inside_a_sugarcane_box']['of']} annotated weed boxes lie "
        "entirely inside a sugarcane box. The crop class is therefore annotated as "
        "overlapping foliage rather than as separated plants, and a weed is frequently "
        "annotated within the canopy of a crop box rather than against bare soil. A weed "
        "patch is delineated as one box where a second annotator might have drawn several "
        "or none, which is a source of uncertainty that this study cannot bound.")

    P.append(
        "**Scope of the comparison.** The two annotation sets are not interchangeable, "
        "and this governs what can be compared. Their test partitions share only three "
        "images, and they annotate the same scenes at different granularity: the "
        f"polygon set marks {pg['test']['instances_per_image']:.1f} instances per test image "
        f"against {bb['test']['instances_per_image']:.1f} for the bounding-box set, with a "
        f"median weed instance occupying {pg['test']['weed_area_pct_median']:.2f}% of the "
        f"frame against {bb['test']['weed_area_pct_median']:.2f}%. One convention delineates "
        "weed patches, the other resolves them into constituent parts, so a metric "
        "computed under one is not comparable with the same metric under the other. "
        "The comparison reported here is accordingly restricted to the three models "
        "trained on the bounding-box convention, which share a single test partition of "
        f"{bb['test']['images']} images and an identical protocol. The two "
        "polygon-trained models are reported separately and are not ranked against "
        "them.")

    P.append(
        "**Object detection models.** Five models were trained, spanning the three "
        "detector families named in the Introduction. Faster R-CNN (Ren et al. 2017) "
        "generates candidate regions with a Region Proposal Network, then classifies "
        "and refines them; Mask R-CNN (He et al. 2017) extends it to instance "
        "segmentation. RetinaNet (Lin et al. 2017b) predicts densely in one stage and "
        "addresses foreground–background imbalance with Focal Loss. YOLOv8-seg (Jocher "
        "et al. 2023) is anchor-free and single-stage, and YOLOv10 (Wang et al. 2024) "
        "additionally removes the need for non-maximum suppression at inference. The "
        "three Detectron2 models used a ResNet-50 backbone (He et al. 2016) with a "
        "Feature Pyramid Network (Lin et al. 2017a); all five were initialised from "
        "COCO-pretrained weights (Lin et al. 2014). The exact configurations were "
        + D._and_list([f"{MODEL_LABELS[k]} {_cfg_name(cfg, k)} at "
                       f"{cfg['models'][k]['imgsz']} px"
                       for k in list(cfg["models"])])
        + ". The two input sizes are each framework's own default for the chosen "
          "configuration and were not harmonised, because changing either would have "
          "meant departing from the setting that framework is tuned for; the "
          "consequence is examined in the Results.")

    P.append(
        "**Model training configuration.** All models were trained on one workstation "
        "with an NVIDIA GeForce RTX 3080 Ti (12 GB VRAM). Faster R-CNN, Mask R-CNN and "
        "RetinaNet used Detectron2 v0.6 (Wu et al. 2019) with PyTorch 2.1.2 (Paszke et "
        "al. 2019); YOLOv8-seg and YOLOv10 used Ultralytics 8.3.253 (Jocher et al. "
        "2023). "
        f"Training ran for {tr['epochs']} epochs with {tr['optimizer']} (Loshchilov and "
        f"Hutter 2019), an initial learning rate of {tr['lr']}, cosine annealing "
        f"(Loshchilov and Hutter 2017), weight decay {tr.get('weight_decay', 0.0001)}, a "
        f"three-epoch warm-up and an effective batch of {tr['batch_size']} "
        "realised by gradient accumulation. The learning rate was not tuned per "
        "architecture, so that the architectures rather than the search effort spent on "
        "each are what differ. No early stopping and no checkpoint selection were "
        "applied: every model was trained for the full schedule and the final-epoch "
        "weights evaluated, because the two frameworks default differently and taking "
        "each default would have given one model a selection advantage. Inference "
        "settings are in Supplementary Table S1" + _or1(journal) + ". One of them, the "
        "cap of 100 detections per image that the Detectron2 framework applies, is "
        "reached by RetinaNet on 14 to 18 of the 124 test images depending on the seed, "
        "against medians of 14 and 9 detections per image for the other two at that same "
        "retention floor of 0.05. Whether "
        "that cap bounds what is reported was measured rather than assumed: repeating "
        "inference with the cap raised to 300 "
        + _detcap_clause()
        + " Training curves are shown in Supplementary Figure S1.")

    P.append(
        "**Evaluation metrics.** Two families of metric are reported, because they "
        "answer different questions and the conclusions depend on both. The first is "
        "average precision under the COCO protocol (Lin et al. 2014): AP averaged over "
        "intersection-over-union thresholds from 0.50 to 0.95, with AP50 and AP75. "
        "These penalise missed objects and spurious detections together. The second is "
        "derived from a confusion matrix over matched detections, the family applied "
        "weed-detection studies commonly report, and is reported here in two variants "
        "that differ only in whether the background row and column are kept. "
        f"Predictions scoring at most {ev['conf_threshold']:.2f} are discarded; each "
        "remaining prediction is matched to at most one ground-truth instance with an "
        f"IoU above {ev['iou_threshold']:.2f}, matches assigned in order of decreasing "
        "IoU; unmatched ground truth and unmatched predictions go to a background row "
        "and column. In the *background-excluded* variant those are deleted before the "
        "metrics are computed, so the only errors counted are instances given the wrong "
        "class: the variant answers whether a localised object is labelled correctly "
        "and cannot answer how many objects were found. In the *full* variant both are "
        "retained, so an undetected instance is a false negative and a detection with "
        "no matching ground truth a false positive. Both variants are reported for "
        "Precision, Recall and F1-score and for both classes. "
        + ("" if journal in ("sugartech", "aiia") else
           "This is done because they differ by about a factor of two on identical "
           "predictions, and quoting Recall from one alongside Precision from the "
           "other would describe no single operating point. ")
        + "Since both are defined at one confidence and one IoU threshold, and "
        "confidence scores are not calibrated to a common scale across frameworks, the "
        "sensitivity to both is reported in the Results. Precision, Recall and F1-score "
        "are computed in the usual way from the counts of true positives, false "
        "positives and false negatives of each class. Accuracy is quoted only for the "
        "background-excluded variant, where every item is a matched object and the "
        "quantity is the share of them given the right class; it is undefined once the "
        "background row and column are retained, because a detection task has no "
        "enumerable true negatives, and it is therefore absent from the full-variant "
        "columns.")

    P.append(
        f"**Statistical design.** Two sources of variability are quantified. The finite "
        f"test set is addressed by bootstrap: metrics are recomputed on {boot['n_boot']} "
        f"resamples of the {boot['n_images']} test images drawn with replacement, with "
        "differences between models taken within an iteration so that the correlation "
        "between models scored on identical images is preserved. Training variability "
        "is addressed by repetition: each bounding-box model was trained repeatedly "
        "from different random seeds with data, schedule and hyperparameters held "
        f"fixed, yielding {n_conv} converged runs respectively, and seed-level values "
        "are compared with Welch's t-test, the three pairwise P-values adjusted by the "
        "Holm–Bonferroni procedure and effect sizes given as Hedges' g with a 95% "
        "interval. Each of these holds one source fixed, so they are combined in a "
        "two-level bootstrap that resamples test images and training runs together and "
        "takes each difference within an iteration; that interval is quoted wherever "
        "this paper states that two architectures do or do not differ. A variance-sum "
        "approximation was considered and rejected, because it assumes the two sources "
        "are independent and additive on the metric scale, which does not hold for a "
        "ratio such as Recall near the middle of its range. The two polygon-trained "
        "models were trained once each, so no comparison involving them is tested.")

    P.append(
        "**Inference-speed measurement and deployment screening.** A detector for an "
        "autonomous sprayer must supply detections at a rate set by the forward speed "
        f"of the implement. At {t['sprayer_speed_kmh']:.0f} km/h with successive frames "
        f"advancing no more than {t['max_frame_advance_m']:.2f} m, the deployment "
        f"hardware must deliver at least {t['fps_required_on_device']:.1f} frames per "
        "second. The frame advance is an assumption about the implement, not a "
        "measurement: it is the distance the boom may travel between two frames before "
        "a weed could pass through the treated band unobserved, and it depends on "
        "nozzle spacing, band overlap and how far ahead of the boom the camera looks. "
        "The ground footprint of the frame was not measured here, so the figure is "
        "stated as a parameter and its consequences are mapped rather than asserted: "
        "Supplementary Table S13 shows which architectures qualify as it is varied, and "
        "the conclusion below holds only in the stricter part of that range. An "
        "embedded accelerator of the class intended here "
        f"delivers roughly one {D._ordinal(t['embedded_slowdown'][1])} to one "
        f"{D._ordinal(t['embedded_slowdown'][0])} of a desktop GPU's throughput "
        "(NVIDIA 2023), a ratio taken from vendor specifications and not measured here, "
        f"so a model must reach about {t['fps_required_on_workstation']:.0f} frames per "
        "second on the workstation to remain plausible under the more conservative "
        "assumption. Neither ratio was measured on embedded hardware, and the two "
        "frameworks were timed as their authors ship them — Detectron2 in eager "
        "PyTorch and Ultralytics through its own inference path — so the comparison "
        "reflects the frameworks as deployed by a practitioner rather than the "
        "architectures under equal engineering effort; an export to a common optimised "
        "runtime would narrow it by an unknown amount. Throughput was measured over "
        "ten repeated passes of a 40-image "
        "subset of the test set, discarding the first ten inferences, timing the "
        "complete per-image path at batch size 1 in single precision with the GPU "
        "synchronised after every image. Compute is quoted in GMACs because the two "
        "frameworks report it under different conventions and mixing them understates "
        "the difference between architectures twofold. The screen is a necessary and "
        "not a sufficient condition — confirming feasibility needs the target hardware, "
        "which was unavailable — and it addresses throughput alone, where a working "
        "implement is also constrained by end-to-end latency and nozzle actuation "
        "delay, neither measured here.")

    # Sugar Tech: "Use of an LLM should be properly documented in the Methods
    # section"; chỉ "AI assisted copy editing" mới không phải khai. Việc dùng
    # mô hình ngôn ngữ ở đây vượt xa sửa câu chữ, nên mục này phải có mặt và
    # phải do nhóm tác giả xác nhận chứ không ai khác.
    if journal in ("sugartech", "aiia"):
        P.append(
            "**Use of artificial-intelligence tools.** " + AI_TOOL_SENTENCE[journal]
            + " The study design, the interpretation of the results and the "
            "conclusions are the authors\' own; every reported number is produced by the "
            "released code from the released data and was checked by the authors. No AI "
            "tool is listed as an author, and the authors take full responsibility for "
            "the content of this article.")
    return P


# ---------------------------------------------------------------- Results

def results_section(cfg, stats, res, seedvar, boot, comp) -> list[str]:
    """Trả về DÃY KHỐI theo đúng thứ tự xuất hiện trong bài.

    Bản đầu trả riêng đoạn văn và bảng rồi ghép lại bằng `str.replace` theo mốc
    chuỗi. Một mốc viết sai (thừa dấu **) khiến Bảng 4 biến mất mà không có lỗi
    nào báo — `str.replace` không khớp thì im lặng. Trả về dãy khối thì vị trí
    bảng là do cấu trúc quyết định, không thể lệch.
    """
    primary, secondary = scope(cfg)
    names = cfg["dataset"]["classes"]
    t = throughput_requirement(cfg)
    rev, lk = load_review()
    fm = rev["full_metrics"]
    B = []          # dãy khối: đoạn văn và bảng, theo đúng thứ tự
    order = sorted(primary, key=lambda k: -D.seed_mean(k, "ap50"))

    # --- Bảng 2: average precision, với SD theo seed cho cả ba AP và KTC bootstrap ---
    rows = []
    for k in order:
        cells = [MODEL_LABELS[k]]
        for metric, dec in (("ap", 1), ("ap50", 2), ("ap75", 1)):
            m_, sd_ = D.seed_mean(k, metric), D.seed_sd(k, metric)
            cells.append(f"{m_:.{dec}f} ± {sd_:.{dec}f}" if sd_ is not None
                         else f"{m_:.{dec}f}")
        cells.append(ci(boot, k, "ap50", "{:.1f}"))
        cells.append(str(len([e for e in seedvar[k]["runs"] if e.get("status") == "ok"])))
        rows.append(cells)
    tbl2 = md_table(
        ["Model", "AP (mean ± SD)", "AP50 (mean ± SD)", "AP75 (mean ± SD)",
         "AP50 95% CI (test set)", "Runs"], rows,
        "**Table 2. Average precision of the three bounding-box models on the common test "
        f"partition ({stats['bbox']['test']['images']} images), under the COCO protocol. "
        "Means and standard deviations are over each model's converged training runs, the "
        "number of which is given in the last column; the confidence interval is the "
        "separate 95% percentile interval from resampling test images, computed on the "
        "first run of each model. The two quantities measure different things and are not "
        "to be combined by eye — the Statistics subsection reports the comparison that "
        "uses both.**")

    B.append(
        "**Detection quality.** Table 2 reports average precision on the common test "
        "partition. Ranked by AP50 the order was "
        + D._and_list([f"{MODEL_LABELS[k]} ({D.seed_mean(k, 'ap50'):.2f} ± "
                       f"{D.seed_sd(k, 'ap50'):.2f})" for k in order])
        + ". That ordering is not reproduced by the stricter criteria, nor by most of "
          "the other measures collected in this study; Table 5 sets them all out "
          "together. The subsections that follow separate the two properties AP50 "
          "conflates: whether an object is found at all, and whether a found object is "
          "given the right class.")
    B.append(tbl2)

    # --- Bảng 3 (mới): hai biến thể ma trận nhầm lẫn, cạnh nhau ---
    rows = []
    for k in order:
        w, c = fm[k]["weed"], fm[k]["sugarcane"]
        rows.append([
            MODEL_LABELS[k],
            f"{D.seed_mean(k, 'weed_precision'):.3f}",
            f"{D.seed_mean(k, 'weed_recall'):.3f}",
            f"{w['precision']['mean']:.3f} ± {w['precision']['sd']:.3f}",
            f"{w['recall']['mean']:.3f} ± {w['recall']['sd']:.3f}",
            f"{w['f1']['mean']:.3f} ± {w['f1']['sd']:.3f}",
            f"{c['recall']['mean']:.3f} ± {c['recall']['sd']:.3f}",
        ])
    tbl3 = md_table(
        ["Model", "Precision (weed, bg-excl.)", "Recall (weed, bg-excl.)",
         "Precision (weed, full)", "Recall (weed, full)", "F1 (weed, full)",
         "Recall (sugarcane, full)"], rows,
        "**Table 3. The same predictions scored under both confusion-matrix variants, at a "
        f"confidence threshold of {cfg['eval']['conf_threshold']:.2f} and an IoU threshold "
        f"of {cfg['eval']['iou_threshold']:.2f}. The background-excluded columns count only "
        "instances given the wrong class, and so describe class discrimination among "
        "localised objects; the full columns retain the background row and column, so an "
        "undetected instance counts as a false negative and a spurious detection as a false "
        "positive. All values are means over each model's converged runs, with the standard "
        "deviation across runs where more than one value enters. The two variants differ by "
        "roughly a factor of two on identical predictions.**")

    cm = {k: res[k]["confusion_matrix"] for k in primary}
    f1s = {k: D.seed_mean(k, "f1") for k in primary}
    accs = {k: D.seed_mean(k, "accuracy") for k in primary}
    fw = {k: fm[k]["weed"] for k in primary}
    B.append(
        "**Class discrimination against detection completeness.** The same predictions "
        "give very different pictures under the two variants of Table 3. With the "
        "background row and column removed, all three separated sugarcane from weeds "
        "reliably once an object had been localised: the two-class mean F1-score "
        f"ranged from {min(f1s.values()):.3f} to {max(f1s.values()):.3f}. Retaining "
        "them on identical predictions gives, for the weed class and for "
        + D._and_list([MODEL_LABELS[k] for k in order])
        + " respectively, Precision of "
        + D._and_list([f"{fw[k]['precision']['mean']:.3f}" for k in order])
        + ", Recall of "
        + D._and_list([f"{fw[k]['recall']['mean']:.3f}" for k in order])
        + " and F1-score of "
        + D._and_list([f"{fw[k]['f1']['mean']:.3f}" for k in order])
        + ". That gap is the central methodological observation of this work: a model "
          "can assign the correct class to almost every object it finds while finding "
          "only half of those present and raising almost as many spurious weed "
          "detections as correct ones — which for a sprayer is herbicide applied where "
          "none was needed.")
    B.append(tbl3)

    k0 = order[0]
    tp0, fpc0, fnc0 = cm[k0][1][1], cm[k0][0][1], cm[k0][1][0]
    fpb0, fnb0 = cm[k0][2][1], cm[k0][1][2]
    _f1 = lambda fp, fn: 2 * tp0 / (2 * tp0 + fp + fn)
    B.append(
        "**The deleted band.** Figure 2 shows what the two variants do to the same "
        f"predictions. Taking {MODEL_LABELS[k0]} as the worked example: of the "
        f"{tp0 + fnc0 + fnb0} annotated weed instances in the test partition, {tp0} were "
        f"detected and classified as weed, {fnc0} were detected but called sugarcane, and "
        f"{fnb0} produced no detection at all; in the other direction the model placed "
        f"{fpb0} weed boxes where nothing was annotated and {fpc0} on sugarcane. Scored "
        f"over the whole matrix the weed class gives an F1-score of "
        f"{_f1(fpc0 + fpb0, fnc0 + fnb0):.2f}. Deleting the background row and column "
        f"first removes {fnb0} false negatives and {fpb0} false positives from the "
        f"arithmetic — the two largest off-diagonal entries — after which the identical "
        f"predictions score {_f1(fpc0, fnc0):.2f}. The deleted column is the weed a "
        "sprayer would drive past untreated and the deleted row is the ground it would "
        "spray for nothing, so between them they are the whole of the field performance; "
        "what survives deletion is whether an object already found carries the right "
        "label, which these models do almost perfectly. The first two are the closer "
        "proxies for what the implement does; no field trial was run here, so they "
        "remain proxies rather than measurements of field performance.")

    B.append(
        "**Direction of the class errors.** Where the two classes were confused, the "
        "direction matters agronomically and is read directly from the confusion "
        "matrices of Figure 2. Sugarcane instances reported as weed numbered "
        + D._and_list([f"{cm[k][0][1]} for {MODEL_LABELS[k]}" for k in order])
        + ", against weed instances reported as sugarcane of "
        + D._and_list([f"{cm[k][1][0]}" for k in order])
        + ". Both are small beside the undetected and spurious counts, so neither "
          "direction is a material error mode here. Of the two, sugarcane reported as "
          "weed is the more harmful: that instance is sprayed and the herbicide reaches "
          "the crop.")

    B.extend(D.statistics_section(seedvar, boot, cfg, rev))
    cur = rev.get("curves")
    if cur:
        at30 = {k: min(cur["froc"][k], key=lambda d: abs(d["conf"] - 0.30)) for k in order}
        mx = {k: max(cur["froc"][k], key=lambda d: d["recall"]) for k in order}
        B.append(
            "**The whole operating curve, not one point.** Every figure above is taken at "
            f"a single confidence threshold of {cfg['eval']['conf_threshold']:.2f}; Figure "
            "3 shows the complete trade-off. In the precision–recall plane (a) the three "
            "architectures trace closely spaced curves, differing in precision by a median "
            "of 0.075 over the recall range they share, so they differ mainly in where on "
            "a common trade-off each sits rather than in how favourable that trade-off is. "
            "Panel (b) restates this in the units an operator would use. At the working "
            "threshold the models produce "
            + D._and_list([f"{at30[k]['fp_per_image']:.1f} for {MODEL_LABELS[k]}"
                           for k in order])
            + " spurious weed detections per image, and further recall is expensive: "
            + D._and_list([f"{MODEL_LABELS[k]} reaches {mx[k]['recall']:.2f} only at "
                           f"{mx[k]['fp_per_image']:.1f} false detections per image"
                           for k in order])
            + ". Panel (c) asks whether the confidence score can separate correct weed "
              "detections from spurious ones — the one place in a detection problem where "
              "a receiver operating characteristic is well defined, because the negatives "
              "are a finite identifiable set rather than the unbounded set of object-free "
              "locations. The areas under the curve are "
            + D._and_list([f"{cur['roc'][k]['auc']:.2f} for {MODEL_LABELS[k]}"
                           for k in order])
            + ": separation is moderate, which is why spurious detections cannot be "
              "thresholded away and why the F1-score is nearly flat across the threshold "
              "range for two of the three models.")

    B.extend(D.what_limits_detection(rev, lk, cfg))
    B.extend(D.resolution_control(rev, cfg, fig="Figure 4"))

    # Bảng theo buổi chụp chuyển sang Phụ lục S9 để giữ bài trong giới hạn
    # trang của tạp chí; con số then chốt vẫn nằm trong chữ ở mục trên.

    B.extend(D.literature_comparison(res, rev))
    B.extend(D.qualitative_results(res, rev, fig="Figure 5"))

    # --- Bảng 4: chi phí suy luận ---
    rows = []
    for k in order:
        c = comp.get(k, {})
        fps = clean_fps(k)[0]
        rows.append([MODEL_LABELS[k], f"{c.get('params', 0) / 1e6:.1f}",
                     f"{c.get('gmacs_median', float('nan')):.1f}",
                     f"{c.get('peak_vram_mb', float('nan')):.0f}", f"{fps:.1f}",
                     "yes" if fps >= t["fps_required_on_workstation"] else "no"])
    tbl4 = md_table(
        ["Model", "Params (M)", "GMACs", "Peak VRAM (MB)", "FPS", "Screen passed"], rows,
        f"**Table 4. Inference cost and deployment screening for the three bounding-box "
        f"models. Throughput is the mean over ten repeated passes on the workstation GPU; "
        f"the screen is the {t['fps_required_on_workstation']:.0f} frames per second "
        f"derived in the Methods. GFLOPs are computed at each model's own input "
        f"resolution (Table 1) and are therefore not directly comparable between rows.**")
    B.extend(D.inference_cost(res, cfg, comp, rev))
    B.append(tbl4)

    # --- Bảng 5: mọi thang đo cạnh nhau, đặt cuối vì là phần tổng hợp ---
    mr = rev.get("metric_ranking")
    if mr:
        rows, prev = [], None
        for m in mr["metrics"]:
            grp = m["group"] if m["group"] != prev else ""
            prev = m["group"]
            cells = [grp, m["label"] + ("" if m["better"] == "hi" else " \u2193")]
            for k in order:
                v = m["values"][k]
                txt = (f"{v:,.0f}" if abs(v) >= 100 else
                       f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")
                cells.append(f"**{txt}**" if m["rank"][k] == 1 else txt)
            rows.append(cells)
        wins = mr["wins"]
        top = sorted(order, key=lambda k: -wins[k])
        B.append(
            "**The ranking is a property of the yardstick.** Table 5 places every measure "
            f"used in this study side by side. Across the {mr['n_metrics']} measures, each "
            "of the three architectures ranks first on at least one, and the two that lead "
            "most often do so on "
            + D._and_list([f"{wins[k]} measure" + ("s" if wins[k] != 1 else "")
                           + f" for {MODEL_LABELS[k]}" for k in top])
            + ". Counting wins across measures would overstate how independent they "
              "are — three of them are forms of COCO average precision and two are "
              "forms of computational cost — so the same table is also read by family: "
            + D._and_list([f"{mr['family_wins'][k]} of the {mr['n_families']} metric "
                           f"families for {MODEL_LABELS[k]}" for k in
                           sorted(order, key=lambda x: -mr['family_wins'][x])])
            + (f", with {mr['n_families_split']} family split between architectures"
               if mr["n_families_split"] else "")
            + ". A reader who takes AP50 as the criterion concludes that "
            f"{MODEL_LABELS[top[0]]} is the better detector; one who takes AP75 or "
            "weed-class precision concludes the opposite; one who takes weed-class "
            "F1-score in the full variant concludes that neither leads. The split is "
            "mechanical rather than mysterious, and follows from how many detections each "
            "model returns — "
            + D.det_count_phrase(rev, cfg, order)
            + ". A detector that proposes "
            "freely finds more and is wrong more often, so it leads on recall, average "
            "recall and the area-weighted measure; one that proposes sparingly is right "
            "more often and misses more, so it leads on precision, on the "
            "background-excluded F1-score and at the stricter overlap threshold. Neither "
            "behaviour is better in the abstract: which is better depends on whether weed "
            "left untreated or herbicide applied to bare ground is the more expensive "
            "error, which is an agronomic question and not one a single number can "
            "settle.")
        B.append(md_table(
            ["Metric family", "Measure"] + [MODEL_LABELS[k] for k in order], rows,
            "**Table 5. Every measure reported in this study, for the three bounding-box "
            "models. Bold marks the best value in each row; \u2193 marks a measure for "
            "which lower is better. Values are means over each model's converged training "
            "runs, except throughput and compute, which are architectural properties.**"))

    # Bảng hai mô hình polygon chuyển sang Phụ lục S10: chúng chấm trên tập
    # test khác nên không so được với ba mô hình chính, và giữ chúng trong bài
    # tốn một trang mà không phục vụ lập luận nào.
    B.append(
        "**Models trained on the polygon convention.** For completeness, "
        + " and ".join(MODEL_LABELS[k] for k in secondary)
        + " were evaluated on the polygon test partition of "
        + f"{stats['polygon']['test']['images']} images under the same protocol, "
          "reaching AP50 of "
        + D._and_list([f"{res[k]['coco']['AP50']:.1f}" for k in secondary])
        + " respectively (Supplementary Table S10). Those values are computed against "
          "ground truth that resolves weed patches into constituent parts and describe "
          "a different task; each model was trained once on 50 test images, so every "
          "value is correspondingly uncertain.")

    return B


# ---------------------------------------------------------------- ghép bài

STATEMENTS = [
    # Đóng góp tác giả và kinh phí nhận diện tác giả, nên theo yêu cầu làm mù
    # của tạp chí chúng nằm ở Title Page nộp riêng, không ở đây.
    "## DATA AND CODE AVAILABILITY",
    "The annotation files in COCO format, the evaluation and analysis code, the per-image "
    "results, the per-seed training logs and the scripts that regenerate every table and "
    "figure are provided as supplementary material, with SHA-256 checksums of all image "
    "files so that the image set, supplied separately on request because of its size, can "
    "be verified as the one used here. Trained weights are available on request.",
    "## CONFLICT OF INTEREST",
    "The authors declare no conflict of interest.",
]


def build(cfg, stats, res, seedvar, boot, comp) -> str:
    D.use_seed_variance(seedvar)          # nguồn số duy nhất, đặt trước mọi hàm sinh chữ
    tr = cfg["train"]
    primary, _ = scope(cfg)

    # Bản thảo làm mù: không có tên tác giả, cơ quan hay địa chỉ liên hệ.
    L = [f"# {TITLE}", "",
         "**Abstract:** " + D.abstract_v2(res, seedvar, stats, cfg, load_review()[0]), "",
         f"**Keywords:** {KEYWORDS}", "", "## INTRODUCTION", ""]
    L += [p + "\n" for p in D.introduction()]

    L += ["## MATERIAL AND METHODS", ""]
    meth = methods(cfg, stats, res, seedvar, boot)
    # Bảng 1 đi ngay sau đoạn cấu hình huấn luyện
    fw = {"detectron2": "Detectron2 / PyTorch / CUDA 11.8", "ultralytics": "Ultralytics"}
    rows1 = [["Framework"] + [fw[cfg["models"][k]["framework"]] for k in primary],
             ["Backbone"] + ["ResNet-50 + FPN" if cfg["models"][k]["framework"] == "detectron2"
                             else cfg["models"][k]["config"].replace(".pt", "")
                             for k in primary],
             ["Annotation format"] + [cfg["models"][k]["annotation"] for k in primary],
             ["Input size (px)"] + [f"{cfg['models'][k]['imgsz']} × {cfg['models'][k]['imgsz']}"
                                    for k in primary],
             ["Effective batch size"] + [tr["batch_size"]] * len(primary),
             ["Epochs"] + [tr["epochs"]] * len(primary),
             ["Optimiser"] + [f"{tr['optimizer']}, lr {tr['lr']}, cosine annealing"] * len(primary),
             ["Training runs (seeds)"] + [len([e for e in seedvar[k]["runs"]
                                               if e.get("status") == "ok"]) for k in primary]]
    tbl1 = md_table(["Setting"] + [MODEL_LABELS[k] for k in primary], rows1,
                    "**Table 1. Training configuration of the three bounding-box models. "
                    "The two polygon-trained models used the same optimiser, learning rate "
                    "and schedule, at 1024 × 1024 (Mask R-CNN) and 640 × 640 "
                    "(YOLOv8-seg).**")
    meth_blocks = []
    for p in meth:
        meth_blocks.append(p)
        if p.startswith("**Model training configuration."):
            meth_blocks.append(tbl1)
    L += [b + "\n" for b in place_figures(meth_blocks)[:0]] or []
    meth_placed = place_figures(meth_blocks)
    L += [b + "\n" for b in meth_placed]

    L += ["## RESULTS AND DISCUSSION", ""]
    for block in place_figures_results(results_section(cfg, stats, res, seedvar, boot, comp)):
        L.append(block + "\n")
    out = "\n".join(L)

    # Kiểm ngay tại chỗ: mọi bảng tạo ra phải có mặt, đánh số liên tục.
    # Ba lần trong quá trình làm, một phép ghép chuỗi không khớp đã làm rơi
    # nguyên một bảng mà không báo lỗi — kiểm ở đây thì hỏng là hỏng ồn ào.
    caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d)\.", out, re.M)]
    n_tbl = out.count("\n|---")
    if caps != list(range(1, len(caps) + 1)):
        raise SystemExit(f"Đánh số bảng không liên tục: {caps}")
    if n_tbl != len(caps):
        raise SystemExit(f"{len(caps)} chú thích bảng nhưng {n_tbl} bảng thật — "
                         "có bảng bị rơi khi ghép")
    log(f"kiểm tra bảng: {len(caps)} bảng, đánh số liên tục 1–{len(caps)}")

    # Tạp chí quy định đúng bốn mục và không có mục Limitations, nên phần hạn
    # chế là khối CUỐI của Results and Discussion, có nhãn in đậm rõ ràng để
    # không bị đọc nhầm thành một kết quả — đó mới là điều phản biện phản đối.
    rev, lk = load_review()
    lim = D.limitations_v2(res, seedvar, stats, cfg, rev, lk)
    out += "\n" + "\n\n".join(lim) + "\n"
    concl = D.conclusion_v2(res, seedvar, stats, cfg, rev)
    out += "\n## CONCLUSION\n\n" + "\n\n".join(concl) + "\n"
    out += "\n" + "\n\n".join(STATEMENTS) + "\n"
    # Sắp theo chữ cái tại thời điểm dựng, để thêm trích dẫn mới không phải
    # tự tìm đúng vị trí trong danh sách.
    out += "\n## REFERENCES\n\n" + "\n\n".join(sorted(REFERENCES, key=str.lower)) + "\n"
    return out


def _normalise_fonts(docx_path: pathlib.Path, font: str, body_pt: int) -> None:
    """Ép mọi phông trong .docx về một phông, và chú thích về đúng cỡ thân bài.

    LibreOffice sinh ra một mớ style với phông khác nhau — Noto Sans cho
    Heading1, Liberation Serif cho Heading2, Liberation Sans cho Heading — nên
    bản thảo in ra không đồng nhất. Tạp chí yêu cầu một phông cho toàn bài, và
    chú thích hình phải cùng cỡ với thân bài.
    """
    import re
    import shutil
    import zipfile

    half = str(body_pt * 2)
    tmp = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx_path) as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in ("word/styles.xml", "word/document.xml"):
                x = data.decode()
                # mọi thuộc tính tên phông → một phông duy nhất
                x = re.sub(r'(w:(?:ascii|hAnsi|cs|eastAsia))="[^"]*"',
                           lambda m: f'{m.group(1)}="{font}"', x)
                x = re.sub(r'(w:(?:ascii|hAnsi|cs|eastAsia)Theme)="[^"]*"', "", x)
                if item.filename == "word/styles.xml":
                    # cỡ mặc định và cỡ của style Caption về đúng cỡ thân bài
                    x = re.sub(r"(<w:docDefaults>.*?)<w:sz w:val=\"\d+\"/>",
                               lambda m: m.group(1) + f'<w:sz w:val="{half}"/>',
                               x, flags=re.S)
                    x = re.sub(
                        r'(<w:style [^>]*w:styleId="Caption".*?)<w:sz w:val="\d+"/>',
                        lambda m: m.group(1) + f'<w:sz w:val="{half}"/>',
                        x, flags=re.S)
                data = x.encode()
            zout.writestr(item, data)
    shutil.move(str(tmp), str(docx_path))


def _add_page_numbers(docx_path: pathlib.Path) -> None:
    """Chèn chân trang có trường PAGE vào .docx.

    Tạp chí yêu cầu "use the automatic page numbering function". LibreOffice
    chuyển từ HTML không sinh chân trang, nên thêm trực tiếp vào gói OOXML:
    một part footer1.xml, một quan hệ trỏ tới nó, khai báo content-type, và
    một <w:footerReference> trong <w:sectPr> của document.xml.
    """
    import shutil
    import zipfile

    NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')
    footer = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:ftr {NS}><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        '<w:r><w:t>1</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

    tmp = docx_path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx_path) as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        if "word/footer1.xml" in names:          # đã có chân trang, không đụng
            tmp.unlink(missing_ok=True)
            return
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/_rels/document.xml.rels":
                x = data.decode()
                x = x.replace("</Relationships>",
                              '<Relationship Id="rIdFtr1" Type="http://schemas.openxml'
                              'formats.org/officeDocument/2006/relationships/footer" '
                              'Target="footer1.xml"/></Relationships>')
                data = x.encode()
            elif item.filename == "[Content_Types].xml":
                x = data.decode()
                x = x.replace("</Types>",
                              '<Override PartName="/word/footer1.xml" ContentType='
                              '"application/vnd.openxmlformats-officedocument.'
                              'wordprocessingml.footer+xml"/></Types>')
                data = x.encode()
            elif item.filename == "word/document.xml":
                x = data.decode()
                if "<w:sectPr" in x:
                    i = x.index("<w:sectPr")
                    j = x.index(">", i) + 1
                    x = x[:j] + '<w:footerReference w:type="default" r:id="rIdFtr1"/>' + x[j:]
                data = x.encode()
            zout.writestr(item, data)
        zout.writestr("word/footer1.xml", footer)
    shutil.move(str(tmp), str(docx_path))


def to_docx(md_path: pathlib.Path, double_spaced: bool = False,
            font_pt: int = 12, page_numbers: bool = False) -> pathlib.Path | None:
    """Markdown → HTML → DOCX bằng LibreOffice (máy này không có pandoc)."""
    import html as _h
    import re as _re
    text = md_path.read_text(encoding="utf-8")
    body = []
    in_tbl = False
    for line in text.split("\n"):
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(_re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            if not in_tbl:
                body.append('<table border="1" cellspacing="0" cellpadding="4">')
                in_tbl = True
            tag = "th" if len(body) and body[-1].startswith("<table") else "td"
            body.append("<tr>" + "".join(f"<{tag}>{md_inline(c)}</{tag}>"
                                         for c in cells) + "</tr>")
            continue
        if in_tbl:
            body.append("</table>"); in_tbl = False
        m_img = _re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m_img:
            # LibreOffice LIÊN KẾT ảnh khi src là đường dẫn file, nên .docx xuất ra
            # không có ảnh nào. Nhúng bằng data-URI base64 thì ảnh nằm hẳn trong file.
            import base64
            import io
            src = pathlib.Path(m_img.group(2))
            # Ảnh hiển thị ở 640 px trong tài liệu, nên nhúng nguyên bản 2 400 px
            # chỉ làm tệp phình (hình định tính một mình đã 11 MB). Thu về 1 600 px
            # — vẫn dư cho bản in — và giữ bản đầy đủ trong figures/ để nộp riêng
            # như hướng dẫn tác giả yêu cầu.
            data = src.read_bytes()
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(data))
                if im.width > 1600:
                    h = round(im.height * 1600 / im.width)
                    im = im.convert("RGB").resize((1600, h), Image.LANCZOS)
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "PNG", optimize=True)
                data, mime = buf.getvalue(), "png"
                # Sơ đồ và biểu đồ là nét vẽ, PNG nén tốt. Hình định tính là ảnh
                # chụp đồng ruộng: PNG để nó ở 6 MB một mình, JPEG xuống còn vài
                # trăm KB mà mắt không phân biệt được ở khổ in 640 px.
                if len(data) > 500 * 1024:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=88, optimize=True,
                                           progressive=True)
                    data, mime = buf.getvalue(), "jpeg"
            except Exception as e:
                log(f"không thu nhỏ được {src.name} ({type(e).__name__}) — nhúng nguyên bản")
                mime = "png"
            b64 = base64.b64encode(data).decode()
            # Hình cao hơn rộng chiếm gần trọn một trang ở bề ngang 640 px. Thu
            # bề ngang lại cho hình dọc để giữ bài trong giới hạn trang.
            w = 640
            try:
                if im.height > im.width * 1.1:
                    w = 500
            except Exception:
                pass
            body.append(f'<p><img src="data:image/{mime};base64,{b64}" width="{w}"/></p>')
            continue
        if s.startswith("# "):
            body.append(f"<h1>{md_inline(s[2:])}</h1>")
        elif s.startswith("## "):
            body.append(f"<h2>{md_inline(s[3:])}</h2>")
        elif s:
            body.append(f"<p>{md_inline(s)}</p>")
    if in_tbl:
        body.append("</table>")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(
        "<html><head><meta charset='utf-8'><style>"
        # Đúng định dạng tạp chí yêu cầu: Times New Roman 12, giãn dòng 1,5, lề
        # 2,5 cm mỗi cạnh. Nhờ vậy số trang của tệp .docx là số trang mà toà
        # soạn sẽ đếm, không phải số trang của một định dạng tuỳ tiện.
        "@page{size:A4;margin:2.5cm}"
        # RAE: Times 12, giãn 1,5. Sugar Tech: bài nguyên tác tính theo trang
        # GIÃN ĐÔI, nên số trang chỉ đúng khi render đúng giãn cách của họ.
        # Chỉ MỘT tên phông. Để nguyên "'Times New Roman',serif" thì LibreOffice
        # ghi thẳng chuỗi "Times New Roman;serif" làm tên phông trong styles.xml,
        # Word không tìm thấy phông đó và tự thay bằng phông khác.
        f"body{{font-family:'Times New Roman';font-size:{font_pt}pt;"
        f"line-height:{2.0 if double_spaced else 1.5}}}"
        f"table{{border-collapse:collapse;font-size:{max(8, font_pt - 2)}pt;margin:10px 0}}"
        "th{background:#eee;text-align:left}h1{font-size:16pt}h2{font-size:13pt}"
        "</style></head><body>" + "\n".join(body) + "</body></html>", encoding="utf-8")
    # LibreOffice cần nêu rõ bộ lọc khi nguồn là HTML; "--convert-to docx" trơn
    # bị từ chối với "no export filter found".
    r = subprocess.run(["soffice", "--headless",
                        "--convert-to", "docx:MS Word 2007 XML",
                        "--outdir", str(md_path.parent), str(html_path)],
                       capture_output=True, text=True, timeout=300)
    out = html_path.with_suffix(".docx")
    if out.exists():
        try:
            _normalise_fonts(out, "Times New Roman", font_pt)
        except Exception as e:
            log(f"không chuẩn hoá được phông cho {out.name} ({type(e).__name__})")
    if out.exists() and page_numbers:
        try:
            _add_page_numbers(out)
        except Exception as e:
            log(f"không chèn được số trang vào {out.name} ({type(e).__name__})")
    return out if out.exists() else None


def md_inline(s: str) -> str:
    import html as _h
    import re as _re
    s = _h.escape(s, quote=False)
    s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = _re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = s.replace("&amp;nbsp;", "&nbsp;").replace("&lt;sup&gt;", "<sup>") \
         .replace("&lt;/sup&gt;", "</sup>")
    return s


def main():
    cfg = load_cfg()
    res = {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in MODEL_LABELS}
    seedvar = json.loads((RESULTS / "seed_variance.json").read_text())
    boot = json.loads((RESULTS / "bootstrap.json").read_text())
    comp = json.loads((RESULTS / "complexity.json").read_text())
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())

    md = build(cfg, stats, res, seedvar, boot, comp)
    md_path = OUT / "RAE_manuscript.md"
    md_path.write_text(md, encoding="utf-8")
    log(f"{len(md.split())} từ → {md_path}")
    d = to_docx(md_path)
    if d:
        final = OUT / "RAE_manuscript.docx"
        if d != final:
            d.replace(final)
        log(f"→ {final}")
    else:
        log("chuyển sang .docx thất bại — kiểm tra soffice")


if __name__ == "__main__":
    main()
