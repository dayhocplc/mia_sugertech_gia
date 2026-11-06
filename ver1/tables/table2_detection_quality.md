| Model | AP | AP50 [95% CI] | AP75 | Weed recall | Sugarcane recall |
|---|---|---|---|---|---|
| Faster R-CNN | 26.8 | 53.1 [49.9, 56.8] | 24.9 | 0.471 | 0.729 |
| RetinaNet | 29.0 | 56.9 [53.8, 60.1] | 26.4 | 0.492 | 0.715 |
| YOLOv10 | 27.3 | 50.1 [46.8, 54.0] | 26.5 | 0.398 | 0.668 |

AP theo chuẩn COCO trên tập test chung của ba mô hình. Recall tính trên ma trận nhầm lẫn CÒN hàng background, nên có tính cả đối tượng bỏ sót. Khoảng tin cậy 95% từ bootstrap lấy mẫu lại ảnh test.
