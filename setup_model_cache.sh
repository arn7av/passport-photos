#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_SOURCE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/passport-photos/models"
SOURCE_DIR="${PASSPORT_PHOTOS_MODEL_DIR:-$DEFAULT_SOURCE_DIR}"
CACHE_DIR="$SCRIPT_DIR/.cache"
DOWNLOAD=0
PRINT_CURL=0
SOURCE_SET=0

usage() {
  cat <<'EOF'
Usage: ./setup_model_cache.sh [OPTIONS] [MODEL_DIR]

Link the complete supported model set from MODEL_DIR into the project .cache.
The script never accesses the network unless --download is supplied.

Options:
  --source-dir DIR  Read/download models in DIR. The positional MODEL_DIR is an alias.
  --download        Download missing models with curl, verify SHA-256, then link them.
  --print-curl      Print verified curl commands for every model and exit unchanged.
  -h, --help        Show this help.

Default model directory:
  $PASSPORT_PHOTOS_MODEL_DIR, when set; otherwise
  $XDG_DATA_HOME/passport-photos/models, or
  ~/.local/share/passport-photos/models

Warning: the complete set is several GiB. BRIA RMBG 2.0 has non-commercial model
terms. BiRefNet_dynamic includes executable custom Python model code; review it
before use. Validation requires sha256sum or shasum.
EOF
}

while (($#)); do
  case "$1" in
    --source-dir)
      if (($# < 2)); then
        echo "Error: --source-dir requires a directory." >&2
        exit 2
      fi
      SOURCE_DIR="$2"
      SOURCE_SET=1
      shift 2
      ;;
    --download)
      DOWNLOAD=1
      shift
      ;;
    --print-curl)
      PRINT_CURL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if (($# > 1)); then
        echo "Error: specify only one model directory." >&2
        exit 2
      fi
      if (($# == 1)); then
        if ((SOURCE_SET)); then
          echo "Error: specify only one model directory." >&2
          exit 2
        fi
        SOURCE_DIR="$1"
        SOURCE_SET=1
        shift
      fi
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if ((SOURCE_SET)); then
        echo "Error: specify only one model directory." >&2
        exit 2
      fi
      SOURCE_DIR="$1"
      SOURCE_SET=1
      shift
      ;;
  esac
done

if (($#)); then
  echo "Error: unexpected argument: $1" >&2
  exit 2
fi

case "$SOURCE_DIR" in
  "~") SOURCE_DIR="$HOME" ;;
  "~/"*) SOURCE_DIR="$HOME/${SOURCE_DIR#\~/}" ;;
esac
if [[ -z "$SOURCE_DIR" || "$SOURCE_DIR" == "/" ]]; then
  echo "Error: model directory must not be empty or the filesystem root." >&2
  exit 2
fi

model_paths=(
  "face_detection_yunet_2023mar.onnx"
  "retinaface.h5"
  "sam_vit_b_01ec64.pth"
  "sam_vit_l_0b3195.pth"
  "sam_vit_h_4b8939.pth"
  "BiRefNet-portrait-epoch_150.onnx"
  "bria-rmbg-2.0.onnx"
  "BiRefNet-general-epoch_244.onnx"
  "isnet-general-use.onnx"
  "u2net_human_seg.onnx"
  "BEN2_Base.onnx"
  "BiRefNet_dynamic/config.json"
  "BiRefNet_dynamic/BiRefNet_config.py"
  "BiRefNet_dynamic/birefnet.py"
  "BiRefNet_dynamic/model.safetensors"
)


model_urls=(
  "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx"
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx"
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
  "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
  "https://huggingface.co/PramaLLC/BEN2/resolve/e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f/BEN2_Base.onnx"
  "https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/280306042f57b7a33854319da62fd86aaa89ec4c/config.json"
  "https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/280306042f57b7a33854319da62fd86aaa89ec4c/BiRefNet_config.py"
  "https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/280306042f57b7a33854319da62fd86aaa89ec4c/birefnet.py"
  "https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/280306042f57b7a33854319da62fd86aaa89ec4c/model.safetensors"
)

model_sha256=(
  "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
  "ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf"
  "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
  "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
  "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
  "1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99"
  "5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958"
  "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c"
  "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
  "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c"
  "22cea62108ff53b7ccc20f7a008bf30494228d84b1687f29ecbe76936a998101"
  "557f461de22ede5b0a2ff60967e579180194ded47587f6e30a3a09ad08ae248b"
  "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52"
  "2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831"
  "e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d"
)

sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$file" | awk '{print $1}'
  else
    echo "Error: sha256sum or shasum is required." >&2
    return 2
  fi
}

verify_sha256() {
  local expected="$1"
  local file="$2"
  local actual
  [[ -f "$file" ]] || return 1
  actual="$(sha256_file "$file")" || return
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch: $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
}

print_curl_commands() {
  local index target partial
  cat <<'EOF'
verify_sha256() {
  expected="$1"
  file="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum -- "$file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 -- "$file" | awk '{print $1}')"
  else
    echo "sha256sum or shasum is required" >&2
    return 2
  fi
  [[ "$actual" == "$expected" ]] || {
    echo "SHA-256 mismatch: $file" >&2
    return 1
  }
}
EOF
  printf 'mkdir -p -- %q\n' "$SOURCE_DIR/BiRefNet_dynamic"
  for ((index = 0; index < ${#model_paths[@]}; index++)); do
    target="$SOURCE_DIR/${model_paths[$index]}"
    partial="${target}.part"
    printf 'if [[ -f %q ]] && verify_sha256 %q %q; then echo %q; else curl --fail --location --retry 3 --continue-at - %q --output %q && { verify_sha256 %q %q || { rm -f -- %q; false; }; } && mv -- %q %q; fi\n' \
      "$target" "${model_sha256[$index]}" "$target" "Already verified: $target" \
      "${model_urls[$index]}" "$partial" "${model_sha256[$index]}" "$partial" \
      "$partial" "$partial" "$target"
  done
}

if ((PRINT_CURL)); then
  print_curl_commands
  exit 0
fi

if ((DOWNLOAD)); then
  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required with --download." >&2
    exit 2
  fi
  if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "Error: sha256sum or shasum is required with --download." >&2
    exit 2
  fi
  mkdir -p -- "$SOURCE_DIR/BiRefNet_dynamic"
  SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd -P)"
  case "$SOURCE_DIR/" in
    "$CACHE_DIR/"*)
      echo "Error: model source directory must be outside the project cache: $CACHE_DIR" >&2
      exit 2
      ;;
  esac
  echo "Downloading the complete model set into $SOURCE_DIR"
  echo "Notice: BRIA RMBG 2.0 is non-commercial; BiRefNet_dynamic contains custom Python code." >&2
  for ((index = 0; index < ${#model_paths[@]}; index++)); do
    target="$SOURCE_DIR/${model_paths[$index]}"
    if verify_sha256 "${model_sha256[$index]}" "$target" 2>/dev/null; then
      echo "Already verified: $target"
      continue
    elif [[ -f "$target" ]]; then
      echo "Replacing model with invalid SHA-256: $target" >&2
    fi
    partial="${target}.part"
    mkdir -p -- "$(dirname "$target")"
    echo "Downloading: ${model_paths[$index]}"
    curl --fail --location --retry 3 --continue-at - \
      "${model_urls[$index]}" \
      --output "$partial"
    if ! verify_sha256 "${model_sha256[$index]}" "$partial"; then
      rm -f -- "$partial"
      echo "Error: downloaded ${model_paths[$index]} failed SHA-256 verification." >&2
      exit 2
    fi
    mv -- "$partial" "$target"
  done
elif [[ -d "$SOURCE_DIR" ]]; then
  SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd -P)"
else
  echo "Error: model directory does not exist: $SOURCE_DIR" >&2
  echo "Use --download to fetch the complete set, or --print-curl to review the commands." >&2
  exit 2
fi

case "$SOURCE_DIR/" in
  "$CACHE_DIR/"*)
    echo "Error: model source directory must be outside the project cache: $CACHE_DIR" >&2
    exit 2
    ;;
esac

missing=0
for ((index = 0; index < ${#model_paths[@]}; index++)); do
  source_file="$SOURCE_DIR/${model_paths[$index]}"
  if [[ ! -f "$source_file" ]]; then
    echo "Missing model: $source_file" >&2
    missing=$((missing + 1))
    continue
  fi
  if ! verify_sha256 "${model_sha256[$index]}" "$source_file"; then
    missing=$((missing + 1))
  fi
done
if ((missing > 0)); then
  echo "Cache was not changed because $missing model file(s) are missing or invalid." >&2
  echo "Run with --download, or inspect all commands with --print-curl." >&2
  exit 2
fi

cache_dirs=(
  "$CACHE_DIR"
  "$CACHE_DIR/yunet"
  "$CACHE_DIR/sam"
  "$CACHE_DIR/rembg"
  "$CACHE_DIR/ben2"
  "$CACHE_DIR/birefnet"
  "$CACHE_DIR/birefnet/BiRefNet_dynamic"
  "$CACHE_DIR/deepface"
  "$CACHE_DIR/deepface/.deepface"
  "$CACHE_DIR/deepface/.deepface/weights"
)
for cache_path in "${cache_dirs[@]}"; do
  if [[ -L "$cache_path" ]]; then
    echo "Cannot create cache through a directory symlink: $cache_path" >&2
    exit 2
  fi
  if [[ -e "$cache_path" && ! -d "$cache_path" ]]; then
    echo "Cannot create cache: $cache_path exists and is not a directory." >&2
    exit 2
  fi
done
mkdir -p "${cache_dirs[@]}"

link_model() {
  local source_file="$1"
  local cache_file="$2"
  if [[ "$source_file" == "$cache_file" ]]; then
    echo "Refusing to replace a model with a self-referential cache link: $source_file" >&2
    exit 2
  fi
  if [[ -e "$cache_file" && ! -L "$cache_file" ]]; then
    echo "Refusing to replace a non-symlink cache path: $cache_file" >&2
    exit 2
  fi
  rm -f -- "$cache_file"
  ln -s "$source_file" "$cache_file"
  printf 'Linked %-42s -> %s\n' "${cache_file#"$SCRIPT_DIR/"}" "$source_file"
}

link_model \
  "$SOURCE_DIR/face_detection_yunet_2023mar.onnx" \
  "$CACHE_DIR/yunet/face_detection_yunet_2023mar.onnx"

link_model \
  "$SOURCE_DIR/retinaface.h5" \
  "$CACHE_DIR/deepface/.deepface/weights/retinaface.h5"

link_model "$SOURCE_DIR/sam_vit_b_01ec64.pth" "$CACHE_DIR/sam/sam_vit_b_01ec64.pth"
link_model "$SOURCE_DIR/sam_vit_l_0b3195.pth" "$CACHE_DIR/sam/sam_vit_l_0b3195.pth"
link_model "$SOURCE_DIR/sam_vit_h_4b8939.pth" "$CACHE_DIR/sam/sam_vit_h_4b8939.pth"

# rembg requires canonical filenames that differ from several release assets.
link_model \
  "$SOURCE_DIR/BiRefNet-portrait-epoch_150.onnx" \
  "$CACHE_DIR/rembg/birefnet-portrait.onnx"
link_model \
  "$SOURCE_DIR/bria-rmbg-2.0.onnx" \
  "$CACHE_DIR/rembg/bria-rmbg.onnx"
link_model \
  "$SOURCE_DIR/BiRefNet-general-epoch_244.onnx" \
  "$CACHE_DIR/rembg/birefnet-general.onnx"
link_model \
  "$SOURCE_DIR/isnet-general-use.onnx" \
  "$CACHE_DIR/rembg/isnet-general-use.onnx"
link_model \
  "$SOURCE_DIR/u2net_human_seg.onnx" \
  "$CACHE_DIR/rembg/u2net_human_seg.onnx"

link_model "$SOURCE_DIR/BEN2_Base.onnx" "$CACHE_DIR/ben2/BEN2_Base.onnx"

for model_file in config.json BiRefNet_config.py birefnet.py model.safetensors; do
  link_model \
    "$SOURCE_DIR/BiRefNet_dynamic/$model_file" \
    "$CACHE_DIR/birefnet/BiRefNet_dynamic/$model_file"
done

echo "Model cache ready: $CACHE_DIR"
