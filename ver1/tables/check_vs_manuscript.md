# Đối chiếu số chạy lại với số in trong bản thảo

Cột *bản thảo* lấy từ Bảng 2 và Bảng 3 của Full_Manuscript_RAE.docx.
Chênh lệch > 0,02 được đánh dấu ⚠ và cần xem lại trước khi nộp.

| Mô hình | Lớp | Chỉ số | Bản thảo | Chạy lại | Chênh |
|---|---|---|---|---|---|
| Faster R-CNN | sugarcane | precision | 0.959 | 0.975 | +0.016 |
| Faster R-CNN | sugarcane | recall | 0.948 | 0.963 | +0.015 |
| Faster R-CNN | sugarcane | f1 | 0.953 | 0.969 | +0.016 |
| Faster R-CNN | weed | precision | 0.964 | 0.972 | +0.008 |
| Faster R-CNN | weed | recall | 0.972 | 0.981 | +0.009 |
| Faster R-CNN | weed | f1 | 0.968 | 0.977 | +0.009 |
| Faster R-CNN | *trung bình* | precision | 0.962 | 0.974 | +0.012 |
| Faster R-CNN | *trung bình* | recall | 0.960 | 0.972 | +0.012 |
| Faster R-CNN | *trung bình* | f1 | 0.960 | 0.973 | +0.013 |
| Faster R-CNN | *trung bình* | accuracy | 0.962 | 0.973 | +0.011 |
| Mask R-CNN | sugarcane | precision | 0.829 | 0.917 | +0.088 ⚠ |
| Mask R-CNN | sugarcane | recall | 0.928 | 0.894 | -0.034 ⚠ |
| Mask R-CNN | sugarcane | f1 | 0.867 | 0.905 | +0.038 ⚠ |
| Mask R-CNN | weed | precision | 0.959 | 0.949 | -0.010 |
| Mask R-CNN | weed | recall | 0.897 | 0.961 | +0.064 ⚠ |
| Mask R-CNN | weed | f1 | 0.927 | 0.955 | +0.028 ⚠ |
| Mask R-CNN | *trung bình* | precision | 0.894 | 0.933 | +0.039 ⚠ |
| Mask R-CNN | *trung bình* | recall | 0.912 | 0.927 | +0.015 |
| Mask R-CNN | *trung bình* | f1 | 0.900 | 0.930 | +0.030 ⚠ |
| Mask R-CNN | *trung bình* | accuracy | 0.908 | 0.939 | +0.031 ⚠ |
| RetinaNet | sugarcane | precision | 0.901 | 0.974 | +0.073 ⚠ |
| RetinaNet | sugarcane | recall | 0.885 | 0.944 | +0.059 ⚠ |
| RetinaNet | sugarcane | f1 | 0.893 | 0.959 | +0.066 ⚠ |
| RetinaNet | weed | precision | 0.928 | 0.961 | +0.033 ⚠ |
| RetinaNet | weed | recall | 0.938 | 0.982 | +0.044 ⚠ |
| RetinaNet | weed | f1 | 0.933 | 0.971 | +0.038 ⚠ |
| RetinaNet | *trung bình* | precision | 0.914 | 0.968 | +0.054 ⚠ |
| RetinaNet | *trung bình* | recall | 0.911 | 0.963 | +0.052 ⚠ |
| RetinaNet | *trung bình* | f1 | 0.913 | 0.965 | +0.052 ⚠ |
| RetinaNet | *trung bình* | accuracy | 0.917 | 0.966 | +0.049 ⚠ |
| YOLOv8-seg | sugarcane | precision | 0.950 | 0.933 | -0.017 |
| YOLOv8-seg | sugarcane | recall | 0.946 | 0.915 | -0.031 ⚠ |
| YOLOv8-seg | sugarcane | f1 | 0.948 | 0.924 | -0.024 ⚠ |
| YOLOv8-seg | weed | precision | 0.969 | 0.960 | -0.009 |
| YOLOv8-seg | weed | recall | 0.967 | 0.969 | +0.002 |
| YOLOv8-seg | weed | f1 | 0.968 | 0.964 | -0.004 |
| YOLOv8-seg | *trung bình* | precision | 0.960 | 0.947 | -0.013 |
| YOLOv8-seg | *trung bình* | recall | 0.957 | 0.942 | -0.015 |
| YOLOv8-seg | *trung bình* | f1 | 0.958 | 0.944 | -0.014 |
| YOLOv8-seg | *trung bình* | accuracy | 0.960 | 0.952 | -0.008 |
| YOLOv10 | sugarcane | precision | 0.950 | 0.976 | +0.026 ⚠ |
| YOLOv10 | sugarcane | recall | 0.946 | 0.963 | +0.017 |
| YOLOv10 | sugarcane | f1 | 0.948 | 0.969 | +0.021 ⚠ |
| YOLOv10 | weed | precision | 0.969 | 0.970 | +0.001 |
| YOLOv10 | weed | recall | 0.967 | 0.981 | +0.014 |
| YOLOv10 | weed | f1 | 0.968 | 0.975 | +0.007 |
| YOLOv10 | *trung bình* | precision | 0.960 | 0.973 | +0.013 |
| YOLOv10 | *trung bình* | recall | 0.957 | 0.972 | +0.015 |
| YOLOv10 | *trung bình* | f1 | 0.958 | 0.972 | +0.014 |
| YOLOv10 | *trung bình* | accuracy | 0.980 | 0.973 | -0.007 |

## Hai điểm phải xử lý trong bản thảo

1. Bảng 2 đang cho YOLOv8-seg và YOLOv10 **đúng cùng bộ ba số** (0,950/0,946/0,948 và 0,969/0,967/0,968). Hai mô hình khác nhau, chạy trên hai kiểu nhãn khác nhau, gần như không thể trùng khớp đến chữ số thứ ba — nhiều khả năng một dòng bị chép nhầm.
2. Bảng 3 cho YOLOv10 Accuracy 0,980 trong khi Precision/Recall/F1 trùng YOLOv8-seg. Với ma trận 2×2 đã bỏ background, Accuracy là hàm của chính ma trận đó, nên P/R/F1 trùng nhau thì Accuracy bắt buộc cũng trùng. Con số 0,980 không thể sinh ra từ cùng một ma trận.
