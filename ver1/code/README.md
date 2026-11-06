# Code cho bài báo — phát hiện cỏ dại trong ruộng mía

Mã nguồn tái lập cho bản thảo *"Detection Completeness and Onboard Throughput of
Three Deep Learning Detectors for Site-Specific Weed Control in Sugarcane"*, trên
bộ ảnh Tây Ninh của nhóm.

**Bản thảo và mọi tài liệu của bài nằm ở [`paper/`](paper/)**, bộ tệp đem nộp ở
[`submission_SugarTech/`](submission_SugarTech/). Xem [`paper/README.md`](paper/README.md)
để biết tệp nào là gì và dựng lại bằng lệnh nào. Có hai phiên bản cùng số liệu:
bản Sugar Tech (Springer, SCIE) đang nhắm nộp, và bản Research in Agricultural
Engineering đã qua phản biện vòng một.

Mọi tham số của bài báo nằm ở **[`configs/paper.yaml`](configs/paper.yaml)** —
sửa ở đó, không sửa trong code. Không con số nào trong `tables/` hay `figures/`
được gõ tay; tất cả sinh từ `results/*.json`.

## Chạy

```bash
bash rae/setup_env.sh          # env riêng: torch cu118 + Detectron2 + Ultralytics
bash rae/run_all.sh smoke      # 2 epoch mỗi mô hình, kiểm tra đường ống
bash rae/run_all.sh full       # 50 epoch như Bảng 1
bash rae/run_all.sh tables     # chỉ dựng lại bảng/hình từ results/ đã có
```

Từng bước riêng lẻ:

```bash
P=rae/envs/rae/bin/python
$P rae/src/data_prep.py --stats --to-yolo
$P rae/src/train_detectron2.py --model faster_rcnn --micro-batch 2
$P rae/src/train_ultralytics.py --model yolov10
$P rae/src/evaluate.py --all
$P rae/src/make_tables.py
$P rae/src/make_figures.py --all
$P rae/src/edge_benchmark.py --model yolov10 --export engine --half   # trên Jetson
$P rae/src/selftest.py
$P rae/src/update_manuscript.py --docx ~/Downloads/Full_Manuscript_RAE.docx \
     --out ~/Downloads/Full_Manuscript_RAE_revised.docx
```

## Dữ liệu

Hai bản xuất Roboflow của cùng đợt chụp tại nông trại Tân Hưng, Châu Thành,
Tây Ninh, nằm trong `TayNinh/Picture and Video/`:

| Bộ | Kiểu nhãn | Ảnh (sau tăng cường) | Ảnh gốc | Đối tượng | Mô hình dùng |
|---|---|---|---|---|---|
| `Dataset_Bounding_Box` | hộp, COCO | 3 139 (train 2 892 / valid 123 / test 124) | 1 211 | 34 350 | Faster R-CNN, RetinaNet, YOLOv10 |
| `Dataset_Polygon` | đa giác, COCO | 1 312 (train 1 212 / valid 50 / test 50) | 504 | 33 380 | Mask R-CNN, YOLOv8-seg |

Roboflow đánh id 0 cho lớp gộp giữ chỗ, 1 = sugarcane, 2 = weed; toàn bộ
pipeline ánh xạ lại về **0 = sugarcane, 1 = weed** và bỏ lớp giữ chỗ.
Tỷ lệ chia 80 : 10 : 10 là **theo ảnh gốc**; tập train được Roboflow nhân ba
bằng tăng cường, valid và test thì không — nên tỷ lệ theo số file là 92 : 4 : 4.

## Giao thức đánh giá

Sao đúng cách khoá luận gốc đã tính (`supervision.ConfusionMatrix.benchmark`),
nhưng viết lại trong [`src/metrics.py`](src/metrics.py) để kiểm chứng được:

1. bỏ dự đoán có confidence ≤ **0,30**;
2. ghép dự đoán với nhãn thật khi IoU > **0,50**, ưu tiên IoU cao, mỗi bên ghép một lần;
3. dựng ma trận 3×3 `[sugarcane, weed, background]`, hàng = nhãn thật, cột = dự đoán;
4. **bỏ hàng và cột background**, rồi tính TP/FP/FN/TN và bốn công thức (1)–(4).

Bước 4 là lý do các con số trong Bảng 2/3 cao (~0,95) trong khi AP50 theo chuẩn
COCO thấp hơn nhiều: sau khi bỏ background, FP không còn đếm báo nhầm trên nền
và FN không còn đếm bỏ sót — chỉ còn nhầm lẫn mía ↔ cỏ. `results/*.json` lưu
**cả hai** phiên bản (`filtered` và `with_background`) cùng AP/AP50/AP75 theo COCO,
để bản thảo ghi rõ mình báo cáo cái nào.

## Batch 16 trên GPU 12 GB

Bảng 1 ghi batch 16 ở 1024×1024 với ResNet-50 + FPN; cấu hình đó cần khoảng
40 GB VRAM, còn máy trong bài là RTX 3080 Ti 12 GB. `train_detectron2.py` giữ
nguyên batch **hiệu dụng** 16 bằng tích luỹ gradient: `--micro-batch 2` forward
2 ảnh mỗi lượt, cộng gradient 8 lượt rồi mới cập nhật trọng số. Về mặt toán học
đây là batch 16. Nếu chạy trên máy đủ VRAM thì `--micro-batch 16` cho hành vi
y hệt một lần forward 16 ảnh.

## Cấu trúc

```
configs/paper.yaml          tham số Bảng 1 + giao thức đánh giá
src/common.py               đường dẫn, đọc config, ánh xạ category id
src/metrics.py              ma trận nhầm lẫn + công thức (1)–(4)
src/data_prep.py            thống kê bộ dữ liệu, COCO → YOLO
src/train_detectron2.py     Faster R-CNN, Mask R-CNN, RetinaNet (+ AdamW, tích luỹ gradient)
src/train_ultralytics.py    YOLOv8-seg, YOLOv10
src/predict.py              một giao diện suy luận chung cho cả năm mô hình
src/evaluate.py             chạy tập test → results/<model>.json
src/make_tables.py          Bảng 1, 2, 3 + đối chiếu với số in trong bản thảo
src/make_figures.py         Hình 1 (quy trình), ma trận nhầm lẫn, biểu đồ cột, Hình 5
src/edge_benchmark.py       đo FPS trên Jetson Orin NX (TensorRT/ONNX)
src/docx_edit.py            máy sửa .docx bằng Track Changes
src/update_manuscript.py    ghi số chạy lại vào bản thảo, dạng Track Changes
src/selftest.py             kiểm tra metrics và chuyển đổi nhãn (không cần GPU)
results/ tables/ figures/   đầu ra, sinh lại được hoàn toàn
runs/                       checkpoint và log huấn luyện
```

## Ba chỗ trong bản thảo cần đối chiếu lại

Chạy `src/make_tables.py` sẽ sinh `tables/check_vs_manuscript.md` so từng ô với
số đang in. Ba điểm đã thấy ngay từ dữ liệu, trước khi huấn luyện:

1. **Số ảnh.** Bản thảo ghi "2 300 field images". Trên đĩa, hợp của hai bản xuất
   Roboflow là **1 363 ảnh gốc** (1 211 bộ hộp + 504 bộ đa giác, chung 352),
   và `Dataset_All/Picture` có 786 file ảnh cộng 3 video. Cần xác định 2 300 đếm
   theo đơn vị nào.
2. **Bảng 2 — YOLOv8-seg và YOLOv10 trùng nhau đến chữ số thứ ba** trên cả sáu ô.
   Hai kiến trúc khác nhau, huấn luyện trên hai bộ nhãn khác nhau (đa giác và hộp),
   trùng khớp như vậy gần như chắc là một dòng bị chép nhầm.
3. **Bảng 3 — YOLOv10 có Accuracy 0,980** trong khi P/R/F1 trùng YOLOv8-seg.
   Với ma trận 2×2 đã bỏ background, Accuracy là hàm của chính ma trận đó, nên
   P/R/F1 trùng thì Accuracy bắt buộc cũng trùng. 0,980 không sinh ra được từ
   cùng một ma trận.

Ngoài ra, phần "Edge device deployment test" trong bản thảo còn để trống
(`[model(s) to be confirmed]`, `[Insert measured FPS/latency results here]`);
`src/edge_benchmark.py` sinh đúng các con số đó khi chạy trên Jetson Orin NX.
