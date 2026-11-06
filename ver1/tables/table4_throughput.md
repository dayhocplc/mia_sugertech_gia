| Model | Params (M) | GMACs | FPS (RTX 3080 Ti) | Vượt ngưỡng sàng lọc 56 FPS? |
|---|---|---|---|---|
| Faster R-CNN | 41.3 | 206.2 | 23.0 | không đạt |
| RetinaNet | 36.3 | 204.4 | 23.4 | không đạt |
| YOLOv10 | 8.1 | 12.4 | 216.3 | đạt |

Ngưỡng suy từ yêu cầu nông học đặt trước: máy phun 6 km/h, hai khung hình liên tiếp lệch không quá 0.30 m ⇒ cần 5.6 FPS TRÊN THIẾT BỊ; nhân hệ số quy đổi 10× (giả định thiết bị biên chậm hơn GPU máy trạm) ⇒ 56 FPS trên máy trạm. GMACs (nhân-cộng, không phải FLOPs=2×MAC) đo ở kích thước đầu vào riêng của từng mô hình, không quy về cùng một cỡ — xem chú thích trong bài.
