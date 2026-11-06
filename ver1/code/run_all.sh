#!/usr/bin/env bash
# Chạy trọn pipeline của bài báo: chuẩn bị dữ liệu → huấn luyện 5 mô hình →
# đánh giá → sinh bảng và hình.
#
#   bash rae/run_all.sh smoke      # 2 epoch mỗi mô hình, kiểm tra đường ống (~30 phút)
#   bash rae/run_all.sh full       # 50 epoch như bản thảo (nhiều giờ)
#   bash rae/run_all.sh tables     # chỉ sinh lại bảng và hình từ results/*.json
set -euo pipefail

RAE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="${RAE_PYTHON:-$RAE/envs/rae/bin/python}"
[ -x "$P" ] || { echo "Chưa có env. Chạy trước: bash rae/setup_env.sh"; exit 1; }
export PYTHONPATH="$RAE/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
cd "$RAE"

MODE=${1:-smoke}
SMOKE=""; [ "$MODE" = "smoke" ] && SMOKE="--smoke"
MICRO=${MICRO_BATCH:-2}

log() { echo -e "\n=== $* ===" ; }

if [ "$MODE" != "tables" ]; then
  log "1/4 chuẩn bị dữ liệu"
  "$P" src/data_prep.py --stats --to-yolo

  log "2/4 huấn luyện"
  for m in faster_rcnn mask_rcnn retinanet; do
    log "  $m (Detectron2, micro-batch $MICRO)"
    "$P" src/train_detectron2.py --model "$m" --micro-batch "$MICRO" $SMOKE \
      2>&1 | tee "logs/train_${m}.log"
  done
  for m in yolov8_seg yolov10; do
    log "  $m (Ultralytics)"
    "$P" src/train_ultralytics.py --model "$m" $SMOKE 2>&1 | tee "logs/train_${m}.log"
  done

  log "3/4 đánh giá trên tập test"
  "$P" src/evaluate.py --all 2>&1 | tee logs/evaluate.log
fi

log "4/4 bảng và hình"
"$P" src/make_tables.py
"$P" src/make_figures.py --all --n 3
"$P" src/make_report.py --out BAO_CAO_KET_QUA.md

echo
echo "Bảng : $RAE/tables/"
echo "Hình : $RAE/figures/"
echo "Số   : $RAE/results/"
echo "Báo cáo: $RAE/BAO_CAO_KET_QUA.md"
echo "Đối chiếu với bản thảo: $RAE/tables/check_vs_manuscript.md"
