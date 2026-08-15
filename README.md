# Passport Photos

Creates passport-style crops from images by detecting and ranking faces, applying consistent framing, optionally replacing the background, and writing either one photo or a tiled print sheet.

Safe defaults:

- One 2×2-inch image at 400 DPI (`800×800` pixels)
- Original background preserved
- Local YuNet face detection when its model is available, then Haar, then DeepFace/OpenCV when installed
- No runtime model downloads
- Existing output files are not overwritten

## Requirements and installation

- Python `>=3.11,<3.14`
- [uv](https://docs.astral.sh/uv/)

```sh
git clone https://github.com/arn7av/passport-photos.git
cd passport-photos
uv sync
```

The core installation includes OpenCV, Pillow, and NumPy. Install only the optional backends you need:

```sh
uv sync --extra deepface  # DeepFace/OpenCV, RetinaFace, and MTCNN
uv sync --extra rembg     # rembg + ONNX Runtime
uv sync --extra sam       # Segment Anything + PyTorch
uv sync --extra ben2      # BEN2 ONNX + ONNX Runtime
uv sync --extra birefnet  # BiRefNet_dynamic + PyTorch/Transformers
uv sync --extra all       # every optional backend
```

## Quickstart

The default preserves the source background:

```sh
uv run passport-photos input.jpg output.png
```

Use the Haar detector explicitly; it needs no optional dependency:

```sh
uv run passport-photos input.jpg output.png --face-backend haar
```

Create a tiled sheet with exact-size photos centered on the configured paper:

```sh
uv run passport-photos input.jpg sheet.png --tiled
```

Inspect and select face candidates:

```sh
uv run passport-photos input.jpg --list-faces
uv run passport-photos input.jpg output.png --export-faces preview/
uv run passport-photos input.jpg output.png --face-index 1
```

Preview files are protected by the same no-overwrite default as final outputs. In batch mode, each source gets a collision-resistant preview subdirectory based on its complete filename.

Process a directory into another directory:

```sh
uv run passport-photos input_directory --output-dir output_directory
```

## Local models

The application does not download model weights. An explicit backend fails with an actionable error when its required dependency or model is unavailable. `--background-backend auto` tries only models already present locally and preserves the original background if all of them fail.

| Backend/model | Expected project cache location | Notes |
|---|---|---|
| YuNet | `.cache/yunet/face_detection_yunet_2023mar.onnx` | No optional Python dependency |
| RetinaFace | `.cache/deepface/.deepface/weights/retinaface.h5` | Requires the `deepface` extra |
| MTCNN | Packaged with MTCNN | Requires the `deepface` extra |
| rembg BiRefNet Portrait | `.cache/rembg/birefnet-portrait.onnx` | Release asset uses a different filename |
| rembg BRIA RMBG 2.0 | `.cache/rembg/bria-rmbg.onnx` | Model terms are non-commercial |
| rembg BiRefNet General | `.cache/rembg/birefnet-general.onnx` | Release asset uses a different filename |
| rembg ISNet General | `.cache/rembg/isnet-general-use.onnx` | Local ONNX file |
| rembg U²-Net Human | `.cache/rembg/u2net_human_seg.onnx` | Local ONNX file |
| BEN2 Base | `.cache/ben2/BEN2_Base.onnx` | Requires the `ben2` extra |
| BiRefNet Dynamic | `.cache/birefnet/BiRefNet_dynamic/` | Complete pinned snapshot with custom Python code |
| SAM ViT-B/L/H | `.cache/sam/sam_vit_{b,l,h}_*.pth` | Checkpoint architecture must match `--sam-model-type` |

### Complete optional model cache

`setup_model_cache.sh` is available in a repository checkout or source distribution and covers every model in the table. With no flags it only validates and links files already present in the external model directory; it does not access the network.

```sh
# Print complete, resumable curl + SHA-256 commands without changing any files
./setup_model_cache.sh --print-curl

# Explicitly opt in to downloading the complete several-GiB model set, then link it
./setup_model_cache.sh --download

# Or link a complete model set that already exists elsewhere
./setup_model_cache.sh /path/to/models
```

The external model directory defaults to `$PASSPORT_PHOTOS_MODEL_DIR`, then `$XDG_DATA_HOME/passport-photos/models`, then `~/.local/share/passport-photos/models`. Relative directories are resolved to absolute paths before symlinks are created.

The all-model download is intentionally opt-in because it is large and includes models with additional terms. In particular, BRIA RMBG 2.0 is non-commercial, and the pinned BiRefNet_dynamic snapshot contains executable custom Python model code. The helper verifies every existing or downloaded artifact against a committed SHA-256 value before linking it. Review upstream terms and the downloaded Python files before use.

If you need only one backend, download only its model and pass its path explicitly (`--yunet-model`, `--sam-checkpoint`, `--ben2-model`, or `--birefnet-model`), or place the file at the cache location in the table. For rembg, use a canonical model name with a canonical file in `.cache/rembg/` or `$U2NET_HOME`.

## Face detection

`--face-backend` accepts:

- `auto`: local YuNet, then Haar, then DeepFace's OpenCV adapter when installed
- `yunet`: require a local YuNet ONNX model
- `haar`: OpenCV's bundled Haar cascade
- `deepface`: DeepFace with its OpenCV detector adapter
- `retinaface`: DeepFace with manually supplied RetinaFace weights
- `mtcnn`: DeepFace with MTCNN's packaged weights

Explicit backends do not silently switch to another detector. Face candidates are filtered, deduplicated, and ranked by confidence and area. Confidence values are backend-specific and should not be compared across detector implementations.

## Background removal

`--background-backend` accepts:

- `none`: preserve the original background; this is the default and is equivalent to `--keep-background`
- `sam`: require a local SAM checkpoint and matching architecture
- `rembg`: require the selected canonical local ONNX model
- `ben2`: require a local `BEN2_Base.onnx`
- `birefnet`: require a complete local `BiRefNet_dynamic` snapshot; loading is local-only
- `auto`: try available local models in this order, then preserve the original background:
  1. rembg BRIA RMBG 2.0
  2. rembg BiRefNet General
  3. rembg BiRefNet Portrait
  4. BiRefNet Dynamic
  5. rembg U²-Net Human
  6. rembg ISNet General
  7. BEN2
  8. SAM ViT-L
  9. SAM ViT-H
  10. SAM ViT-B

`auto` is opt-in because its first available model may have additional usage terms. Background removal runs after the aspect-ratio-preserving source crop and before final resizing. SAM masks are stored as non-pickle `.npz` files under `.cache/segmentation/`; use `--resegment` to ignore a matching cached mask. `PASSPORT_PHOTOS_SEGMENTATION_CACHE_DIR` can move this image-derived cache elsewhere.

## Output and CLI UX

Run `uv run passport-photos --help` for the complete option list. Common controls include:

| Option | Purpose |
|---|---|
| `--tiled` | Create a print sheet instead of one photo |
| `--photo-width`, `--photo-height` | Photo dimensions in inches |
| `--paper-width`, `--paper-height` | Tiled paper dimensions in inches |
| `--dpi` | Pixel density and embedded print metadata |
| `--format {auto,png,jpeg}` | Select output format; `auto` uses the extension |
| `--jpeg-quality 1-100` | JPEG encoding quality |
| `--output-dir DIR` | Explicit batch destination |
| `--overwrite` | Replace existing outputs and face-preview files |
| `--dry-run` | Decode inputs and show planned outputs without detection, model execution, or writes |
| `-v`, `-vv` | Show status/timing or debug details |
| `-q` | Reduce logger output |

A tiled sheet never scales a completed grid. Each photo is first cropped to the configured aspect ratio and resized to its exact dimensions at the configured DPI, whole photos are placed on an exact paper-size white canvas, and any remaining space becomes centered margins. The command rejects dimensions that round to zero pixels and paper too small for one photo.

## Configuration

Use `[passport-photos]` in `passport-photos.toml` or `config.toml`, or `[tool.passport-photos]` in `pyproject.toml`:

```toml
[passport-photos]
face_backend = "auto"
background_backend = "none"

yunet_model = "/path/to/face_detection_yunet_2023mar.onnx"
sam_checkpoint = "/path/to/sam_vit_l_0b3195.pth"
sam_model_type = "vit_l"
rembg_model = "birefnet-portrait"
ben2_model = "/path/to/BEN2_Base.onnx"
birefnet_model = "/path/to/BiRefNet_dynamic"

photo_width = 2
photo_height = 2
paper_width = 10
paper_height = 8
dpi = 400
output_format = "auto"
jpeg_quality = 95
```

Precedence, highest first:

1. Explicit CLI values
2. `--config PATH`
3. `PASSPORT_PHOTOS_CONFIG` or `PASSPORT_PHOTOS_TOML`
4. `./config.toml`
5. `./passport-photos.toml`
6. Project/current-directory `pyproject.toml`
7. `~/.config/passport-photos/config.toml`
8. Built-in defaults

Relative model paths in a config file are resolved from that file's directory. Invalid backend, format, dimension, DPI, and quality values fail immediately instead of silently changing behavior.

`U2NET_HOME` and `DEEPFACE_HOME` can override dependency-specific model directories. The application otherwise checks the project `.cache/` tree and documented portable fallback locations.

## Passport crop semantics

The source crop uses these composition rules:

1. Expand the detected face box by 10% on each side.
2. Use the average eye height as the vertical landmark.
3. Treat the padded face height as `1.25` framing units.
4. Request a crop height of `2 ×` that resolution and derive its width from the configured photo aspect ratio.
5. Center the face horizontally and place the eyes `0.95 ×` the resolution below the top.
6. Clip every requested edge to the image and crop inward to preserve the requested aspect ratio.

The source is never stretched and white side strips are not added when the desired crop reaches an image boundary.

## Benchmark

`bench.sh` is available in a repository checkout or source distribution. It requires an explicit local image so private fixtures cannot be included accidentally. Use only an image you have permission to process:

```sh
./bench.sh /path/to/consented-test-image.jpg
```

Before replacing `bench_out/`, the script verifies required-file presence, resolves symlink targets, decodes the image, and warms the exact lazy imports used by every optional backend in isolated interpreters so incompatible native runtimes are not combined. A fresh environment can spend tens of seconds generating Python bytecode and Numba caches during this explicitly untimed warm-up; moving that work ahead of the benchmark keeps it out of individual backend results. It prints elapsed-time status for every preflight step and backend, and runs Python unbuffered so progress is visible while output is piped into logs. Model contents are then exercised by their individual benchmark runs. Backend timings still include per-process model/session construction and inference, so normal OS file-cache and system-load variation remains. It uses `uv run --no-sync`, so it never installs or updates dependencies. Images, logs, and image-derived SAM mask caches all stay in the project-local ignored `bench_out/` directory. Override `MODEL_CACHE`, `YUNET_MODEL`, `SAM_VIT_B`, `SAM_VIT_L`, `SAM_VIT_H`, `BEN2_MODEL`, or `BIREFNET_MODEL` when needed.
