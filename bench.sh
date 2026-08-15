#!/usr/bin/env bash
# Benchmark each production backend separately. Models and optional dependencies
# must already be available; this script never installs or downloads them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if (($# != 1)); then
  echo "Usage: ./bench.sh /path/to/consented-test-image.jpg" >&2
  exit 2
fi

INPUT="$1"
if [[ "$INPUT" != /* ]]; then
  INPUT="$PWD/$INPUT"
fi
OUT_DIR="$SCRIPT_DIR/bench_out"
MODEL_CACHE="${MODEL_CACHE:-$SCRIPT_DIR/.cache}"
if [[ "$MODEL_CACHE" != /* ]]; then
  MODEL_CACHE="$PWD/$MODEL_CACHE"
fi
if [[ -d "$MODEL_CACHE" ]]; then
  MODEL_CACHE="$(cd "$MODEL_CACHE" && pwd -P)"
fi

YUNET_MODEL="${YUNET_MODEL:-$MODEL_CACHE/yunet/face_detection_yunet_2023mar.onnx}"
SAM_VIT_B="${SAM_VIT_B:-$MODEL_CACHE/sam/sam_vit_b_01ec64.pth}"
SAM_VIT_L="${SAM_VIT_L:-$MODEL_CACHE/sam/sam_vit_l_0b3195.pth}"
SAM_VIT_H="${SAM_VIT_H:-$MODEL_CACHE/sam/sam_vit_h_4b8939.pth}"
BEN2_MODEL="${BEN2_MODEL:-$MODEL_CACHE/ben2/BEN2_Base.onnx}"
BIREFNET_MODEL="${BIREFNET_MODEL:-$MODEL_CACHE/birefnet/BiRefNet_dynamic}"
FILTER="timing|face_detection|background_removal|Found|Selected|Background|Processed|failed|unavailable|skipped|Error"
BENCH_START_SECONDS=$SECONDS

status() {
  local elapsed=$((SECONDS - BENCH_START_SECONDS))
  printf '[bench +%02dm%02ds] %s\n' "$((elapsed / 60))" "$((elapsed % 60))" "$*"
}

run_uv() {
  PASSPORT_PHOTOS_SEGMENTATION_CACHE_DIR="$OUT_DIR/segmentation_cache" \
    uv run --project "$SCRIPT_DIR" --no-sync \
      --extra deepface --extra rembg --extra sam --extra ben2 --extra birefnet "$@"
}

status "Starting benchmark"
status "Input: $INPUT"
status "Model cache: $MODEL_CACHE"

required_files=(
  "$INPUT"
  "$YUNET_MODEL"
  "$SAM_VIT_B"
  "$SAM_VIT_L"
  "$SAM_VIT_H"
  "$MODEL_CACHE/deepface/.deepface/weights/retinaface.h5"
  "$MODEL_CACHE/rembg/birefnet-portrait.onnx"
  "$MODEL_CACHE/rembg/bria-rmbg.onnx"
  "$MODEL_CACHE/rembg/birefnet-general.onnx"
  "$MODEL_CACHE/rembg/isnet-general-use.onnx"
  "$MODEL_CACHE/rembg/u2net_human_seg.onnx"
  "$BEN2_MODEL"
  "$BIREFNET_MODEL/config.json"
  "$BIREFNET_MODEL/BiRefNet_config.py"
  "$BIREFNET_MODEL/birefnet.py"
  "$BIREFNET_MODEL/model.safetensors"
)
status "Checking ${#required_files[@]} required input/model files..."
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required benchmark file: $required_file" >&2
    exit 2
  fi
done
status "Required files found"

INPUT="$(cd "$(dirname "$INPUT")" && pwd -P)/$(basename "$INPUT")"

# Resolve final symlink targets for every required file before deleting old output.
status "Checking resolved paths before replacing previous output..."
run_uv python -c '
from pathlib import Path
import sys

output = Path(sys.argv[1]).resolve()
for raw_path in sys.argv[2:]:
    path = Path(raw_path).resolve(strict=True)
    if path == output or output in path.parents:
        raise SystemExit(f"Refusing to delete benchmark output containing required path: {path}")
' "$OUT_DIR" "${required_files[@]}"
status "Resolved path safety check passed"

# Warm the exact lazy imports used by the backends before measurements begin. A fresh
# environment may spend tens of seconds generating .pyc/Numba caches on first import.
# Keep rembg/pymatting and PyTorch in separate interpreters because their wheels can load
# different OpenMP runtimes on macOS.
warm_import() {
  local label="$1"
  local statement="$2"
  status "Warm-up import: $label"
  run_uv python -c '
import sys

exec(sys.argv[1])
' "$statement" >/dev/null
  status "Warm-up import succeeded: $label"
}

status "Warming optional imports and bytecode (first run can take tens of seconds)..."
warm_import "DeepFace/TensorFlow" "from deepface import DeepFace"
warm_import "ONNX Runtime" "import onnxruntime"
warm_import "rembg/pymatting" "from rembg import new_session, remove"
warm_import "Segment Anything" "from segment_anything import SamPredictor, sam_model_registry"
warm_import "PyTorch" "import torch"
warm_import "Transformers/BiRefNet loader" "from transformers import AutoModelForImageSegmentation"

# Decode the image before replacing any previous benchmark output.
status "Decoding benchmark image with OpenCV..."
run_uv python -c '
import sys
import cv2

if cv2.imread(sys.argv[1], cv2.IMREAD_COLOR) is None:
    raise SystemExit(f"Could not decode benchmark image: {sys.argv[1]}")
' "$INPUT" >/dev/null
status "Image decode succeeded"

status "Preflight passed; resetting output directory: $OUT_DIR"
rm -rf -- "$OUT_DIR"
mkdir -p "$OUT_DIR"

run_pp() {
  run_uv python -u "$SCRIPT_DIR/process_image.py" "$@"
}

summarize() {
  grep -E "$FILTER" || true
}

status "Face detection backends separately (original background, untiled)"
for backend in haar yunet deepface retinaface mtcnn auto; do
  status "Face backend: $backend (pipeline)"
  run_pp "$INPUT" "$OUT_DIR/face_${backend}.png" --keep-background --face-backend "$backend" --yunet-model "$YUNET_MODEL" --overwrite -v 2>&1 | tee "$OUT_DIR/face_${backend}.log" | summarize

  status "Face backend: $backend (list-faces only)"
  run_pp "$INPUT" --list-faces --face-backend "$backend" --yunet-model "$YUNET_MODEL" -v 2>&1 | tee "$OUT_DIR/face_${backend}_list.log" | summarize
done

echo ""
status "Background backends separately (fixed face=haar, untiled)"
status "Background backend: none with face Haar"
run_pp "$INPUT" "$OUT_DIR/bg_haar_none.png" --face-backend haar --background-backend none --overwrite -v 2>&1 | tee "$OUT_DIR/bg_haar_none.log" | summarize

for model in bria-rmbg birefnet-general birefnet-portrait; do
  status "Background backend: rembg model=$model with face Haar"
  run_pp "$INPUT" "$OUT_DIR/bg_haar_rembg_${model}.png" --face-backend haar --background-backend rembg --rembg-model "$model" --overwrite -v 2>&1 | tee "$OUT_DIR/bg_haar_rembg_${model}.log" | summarize
done

status "Background backend: BiRefNet_dynamic with face Haar"
run_pp "$INPUT" "$OUT_DIR/bg_haar_birefnet_dynamic.png" --face-backend haar --background-backend birefnet --birefnet-model "$BIREFNET_MODEL" --overwrite -v 2>&1 | tee "$OUT_DIR/bg_haar_birefnet_dynamic.log" | summarize

for model in u2net_human_seg isnet-general-use; do
  status "Background backend: rembg model=$model with face Haar"
  run_pp "$INPUT" "$OUT_DIR/bg_haar_rembg_${model}.png" --face-backend haar --background-backend rembg --rembg-model "$model" --overwrite -v 2>&1 | tee "$OUT_DIR/bg_haar_rembg_${model}.log" | summarize
done

status "Background backend: BEN2 with face Haar"
run_pp "$INPUT" "$OUT_DIR/bg_haar_ben2.png" --face-backend haar --background-backend ben2 --ben2-model "$BEN2_MODEL" --overwrite -v 2>&1 | tee "$OUT_DIR/bg_haar_ben2.log" | summarize

status "Background backend: SAM vit_l with face Haar (large model)"
run_pp "$INPUT" "$OUT_DIR/bg_haar_sam_vit_l.png" --face-backend haar --background-backend sam --sam-checkpoint "$SAM_VIT_L" --sam-model-type vit_l --overwrite --resegment -v 2>&1 | tee "$OUT_DIR/bg_haar_sam_vit_l.log" | summarize

status "Background backend: SAM vit_h with face Haar (largest model)"
run_pp "$INPUT" "$OUT_DIR/bg_haar_sam_vit_h.png" --face-backend haar --background-backend sam --sam-checkpoint "$SAM_VIT_H" --sam-model-type vit_h --overwrite --resegment -v 2>&1 | tee "$OUT_DIR/bg_haar_sam_vit_h.log" | summarize

status "Background backend: SAM vit_b with face Haar"
run_pp "$INPUT" "$OUT_DIR/bg_haar_sam_vit_b.png" --face-backend haar --background-backend sam --sam-checkpoint "$SAM_VIT_B" --sam-model-type vit_b --overwrite --resegment -v 2>&1 | tee "$OUT_DIR/bg_haar_sam_vit_b.log" | summarize

status "Background backend: auto with face Haar (fixed quality-ranked chain)"
run_pp "$INPUT" "$OUT_DIR/bg_haar_auto.png" --face-backend haar --background-backend auto --sam-checkpoint "$SAM_VIT_L" --sam-model-type vit_l --overwrite --resegment -v 2>&1 | tee "$OUT_DIR/bg_haar_auto.log" | summarize

echo ""
status "Bench done. Outputs and full logs: $OUT_DIR/"
