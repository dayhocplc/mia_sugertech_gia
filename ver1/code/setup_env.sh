#!/usr/bin/env bash
# Dựng môi trường cho pipeline RAE: torch cu118 + Detectron2 + Ultralytics.
# Máy đích: RTX 3080 Ti, CUDA 11.8 — đúng cấu hình ghi trong bản thảo.
#
#   bash rae/setup_env.sh
#
# Env nằm ở rae/envs/rae, không đụng tới .venv hay env mmdet đang có.
set -euo pipefail

CONDA=${CONDA:-/home/thanhtran/anaconda3/bin/conda}
RAE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="$RAE/envs/rae"
P="$ENV/bin/python"

echo "### 1/5 tạo env python 3.10 tại $ENV"
[ -x "$P" ] || "$CONDA" create -y -p "$ENV" python=3.10 -q

echo "### 2/5 torch 2.1.2 + cu118"
"$P" -m pip install -q torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "### 3/5 phụ thuộc chung"
# numpy<2 vì Detectron2 chưa hỗ trợ numpy 2.
# setuptools<81 vì torch 2.1 (cpp_extension) còn import pkg_resources, thứ đã bị
# gỡ khỏi setuptools 81 — thiếu chốt này thì bước dịch Detectron2 hỏng ngay.
"$P" -m pip install -q "numpy<2" "setuptools<81" wheel "opencv-python-headless<5" \
    pycocotools pyyaml matplotlib pandas tqdm

echo "### 4/5 Detectron2 (biên dịch từ nguồn, cần nvcc — có sẵn CUDA 11.8)"
"$P" -c "import detectron2" 2>/dev/null || \
  TORCH_CUDA_ARCH_LIST="8.6" "$P" -m pip install -q --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git"

echo "### 5/5 Ultralytics (>=8.3 mới có YOLOv10)"
"$P" -m pip install -q "ultralytics>=8.3.0,<8.4"
# Ultralytics kéo numpy 2 lên, mà torch 2.1.2 dựng với numpy 1 — phải hạ lại,
# không thì torch báo "Failed to initialize NumPy: _ARRAY_API not found".
# Ultralytics cũng kéo opencv-python 5 (yêu cầu numpy>=2) — hạ luôn.
"$P" -m pip install -q "numpy<2" "opencv-python<5" "opencv-python-headless<5"

echo "### kiểm tra"
"$P" - <<'PY'
import torch, detectron2, ultralytics, cv2, numpy
print("torch       ", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
print("gpu         ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("detectron2  ", detectron2.__version__)
print("ultralytics ", ultralytics.__version__)
print("opencv      ", cv2.__version__, "| numpy", numpy.__version__)
PY
echo "### XONG — dùng: $P"
