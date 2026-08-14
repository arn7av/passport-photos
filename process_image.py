"""Passport photo pipeline: face detection, aspect crop, background, and tiling.

All model weights are user-supplied. The default face chain is local YuNet, Haar, then
DeepFace/OpenCV. Background ``auto`` follows a fixed quality-ranked local model chain
and finally preserves the original background.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import logging
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

__version__ = "0.1.0"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG = logging.getLogger("passport_photos")


def _setup_logging(verbosity: int = 0, quiet: int = 0) -> None:
    level = logging.WARNING
    if quiet:
        level = logging.ERROR - quiet * 10
        level = max(logging.ERROR, level)
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    # only configure if not already configured
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOG.addHandler(handler)
    LOG.setLevel(level)
    # also ensure root doesn't duplicate
    LOG.propagate = False


# --------------------------------------------------------------------------- #
# Constants & helpers
# --------------------------------------------------------------------------- #

YUNET_SCORE_THRESH = 0.6
YUNET_NMS_THRESH = 0.3
BIREFNET_MAX_SIDE = 1024
BIREFNET_SIZE_MULTIPLE = 32
MAX_OUTPUT_PIXELS = 100_000_000
MODEL_CACHE_DIR = Path(__file__).parent / ".cache"
SEGMENTATION_CACHE_DIR = Path(
    os.environ.get("PASSPORT_PHOTOS_SEGMENTATION_CACHE_DIR", MODEL_CACHE_DIR / "segmentation")
).expanduser()
SAM_CHECKPOINT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

# Module-level caches (avoid re-probing filesystem / re-creating detectors)

_HAAR_CASCADE: cv2.CascadeClassifier | None = None  # type: ignore[type-arg]
_YUNET_DETECTOR_CACHE: dict[Any, Any] = {}
_SAM_CACHE: dict[str, Any] = {}
_REMBG_SESSION_CACHE: dict[tuple[str, str], Any] = {}
_ONNX_SESSION_CACHE: dict[str, Any] = {}
_BIREFNET_CACHE: dict[str, Any] = {}
_TOML_CACHE: dict[str, dict] = {}  # type: ignore[type-arg]
_FILE_CONFIG_CACHE: dict[str | None, dict] = {}  # type: ignore[type-arg]


@functools.lru_cache(maxsize=1)
def _yunet_cache_path() -> Path:
    """Return the conventional YuNet cache path without creating it."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "passport-photos" / "yunet.onnx"


def _get_haar_cascade() -> cv2.CascadeClassifier:
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None and not _HAAR_CASCADE.empty():
        return _HAAR_CASCADE
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade at {cascade_path}")
    _HAAR_CASCADE = cascade
    return cascade


def _get_yunet_detector(model_path: Path, width: int, height: int):
    """Cache YuNet detector per model — setInputSize cheaper than recreate."""
    model_key = str(model_path)
    if model_key in _YUNET_DETECTOR_CACHE:
        det = _YUNET_DETECTOR_CACHE[model_key]
        try:
            det.setInputSize((width, height))
            return det
        except Exception:
            pass
    # create new
    det = cv2.FaceDetectorYN_create(  # type: ignore[attr-defined]
        str(model_path), "", (width, height), YUNET_SCORE_THRESH, YUNET_NMS_THRESH, 5000
    )
    _YUNET_DETECTOR_CACHE[model_key] = det
    return det


# --------------------------------------------------------------------------- #
# Config / SAM checkpoint resolution
# --------------------------------------------------------------------------- #
# Model/backend precedence is CLI > TOML > portable local fallback.
#
# Config file example (passport-photos.toml):
#   [passport-photos]
#   sam_checkpoint = "./sam_vit_l_0b3195.pth"  # or ~/.cache/passport-photos/sam_vit_l_0b3195.pth
#   face_backend = "auto"        # auto | yunet | haar | deepface | retinaface | mtcnn
#   background_backend = "auto"  # auto | sam | rembg | ben2 | birefnet | none


def _load_toml(path: Path, *, strict: bool = False) -> dict:
    key = str(path)
    if key in _TOML_CACHE and not strict:
        return _TOML_CACHE[key]
    try:
        import tomllib  # py3.11+

        with open(path, "rb") as f:
            data = tomllib.load(f)
            _TOML_CACHE[key] = data
            return data
    except Exception as e:
        if strict:
            raise ValueError(f"Could not read TOML config {path}: {e}") from e
        LOG.warning("Ignoring unreadable TOML config %s: %s", path, e)
        _TOML_CACHE[key] = {}
        return {}


def _collect_file_config(cli_config: str | None = None) -> dict:
    # simple per-process cache keyed by cli_config; env assumed stable for process
    cache_key = cli_config or "__default__"
    if cache_key in _FILE_CONFIG_CACHE:
        return dict(_FILE_CONFIG_CACHE[cache_key])
    cfg: dict = {}
    # Priority: low → high (later overrides earlier)
    # 1. global / project defaults, 2. local passport-photos.toml, 3. env-specified, 4. CLI --config
    candidates: list[Path] = []
    candidates.append(Path.home() / ".config" / "passport-photos" / "config.toml")
    candidates.extend(
        [
            Path(__file__).parent / "pyproject.toml",
            Path.cwd() / "pyproject.toml",
        ]
    )
    candidates.extend(
        [
            Path(__file__).parent / "passport-photos.toml",
            Path.cwd() / "passport-photos.toml",
            Path.cwd() / "config.toml",
        ]
    )
    env_cfg = os.environ.get("PASSPORT_PHOTOS_CONFIG") or os.environ.get("PASSPORT_PHOTOS_TOML")
    if env_cfg:
        candidates.append(Path(env_cfg))
    explicit_config = Path(cli_config).expanduser() if cli_config else None
    if explicit_config:
        if not explicit_config.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit_config}")
        candidates.append(explicit_config)
    for p in candidates:
        if not p.is_file():
            continue
        data = _load_toml(p, strict=(explicit_config is not None and p == explicit_config))
        section = None
        if "passport-photos" in data:
            section = data["passport-photos"]
        elif (
            "tool" in data and isinstance(data["tool"], dict) and "passport-photos" in data["tool"]
        ):
            section = data["tool"]["passport-photos"]
        elif "tool" in data and "passport_photos" in data["tool"]:
            section = data["tool"]["passport_photos"]
        if section and isinstance(section, dict):
            resolved_section = dict(section)
            for model_key in (
                "sam_checkpoint",
                "sam-checkpoint",
                "yunet_model",
                "yunet-model",
                "yunet_checkpoint",
                "yunet-checkpoint",
                "rembg_model",
                "rembg-model",
                "ben2_model",
                "ben2-model",
                "birefnet_model",
                "birefnet-model",
            ):
                value = resolved_section.get(model_key)
                if isinstance(value, str):
                    model_path = Path(value).expanduser()
                    if (
                        model_path.suffix.lower() in {".pth", ".onnx", ".safetensors"}
                        or model_key in {"birefnet_model", "birefnet-model"}
                    ) and not model_path.is_absolute():
                        resolved_section[model_key] = str((p.parent / model_path).resolve())
            cfg.update(resolved_section)
        if p.name == "passport-photos.toml" and not section:
            for k, v in data.items():
                if k in (
                    "sam_checkpoint",
                    "sam-checkpoint",
                    "yunet_model",
                    "yunet-model",
                    "yunet_checkpoint",
                    "yunet-checkpoint",
                    "rembg_model",
                    "rembg-model",
                    "ben2_model",
                    "ben2-model",
                    "birefnet_model",
                    "birefnet-model",
                    "face_backend",
                    "background_backend",
                ):
                    normalized_key = k.replace("-", "_")
                    resolved_value = v
                    if isinstance(v, str):
                        model_path = Path(v).expanduser()
                        if (
                            model_path.suffix.lower() in {".pth", ".onnx", ".safetensors"}
                            or normalized_key == "birefnet_model"
                        ) and not model_path.is_absolute():
                            resolved_value = str((p.parent / model_path).resolve())
                    cfg[normalized_key] = resolved_value
    _FILE_CONFIG_CACHE[cache_key] = dict(cfg)
    return cfg


def _sam_checkpoint_candidates(model_type: str) -> list[Path]:
    """Return local-only checkpoint candidates for one SAM architecture."""
    filename = SAM_CHECKPOINT_FILENAMES[model_type]
    return [
        MODEL_CACHE_DIR / "sam" / filename,
        MODEL_CACHE_DIR / filename,
        Path.cwd() / filename,
        Path(__file__).parent / filename,
        Path.home() / ".cache" / "passport-photos" / filename,
    ]


def resolve_sam_checkpoint_for_type(
    model_type: str,
    preferred_checkpoint: Path | None = None,
    preferred_model_type: str | None = None,
) -> Path | None:
    """Resolve one SAM architecture for the automatic multi-checkpoint chain."""
    if model_type not in SAM_CHECKPOINT_FILENAMES:
        return None
    if (
        preferred_checkpoint is not None
        and preferred_checkpoint.is_file()
        and preferred_model_type == model_type
    ):
        return preferred_checkpoint.resolve()
    for candidate in _sam_checkpoint_candidates(model_type):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_sam_checkpoint(
    cli_value: str | None = None, config: dict | None = None, warn: bool = True
) -> Path | None:
    # Precedence: CLI > TOML > local fallback.
    if config is None:
        config = _collect_file_config()
    if cli_value:
        p = Path(cli_value).expanduser()
        if p.is_file():
            return p.resolve()
        if warn:
            print(f"Warning: --sam-checkpoint {p} not found.", file=sys.stderr)
        return None
    for key in ("sam_checkpoint", "sam-checkpoint", "samCheckpoint"):
        if key in config:
            p = Path(str(config[key])).expanduser()
            if p.is_file():
                return p.resolve()
            # config miss is not a warning — it may be a default placeholder (e.g. ./sam_vit_l_0b3195.pth)
            if warn:
                LOG.warning("Configured SAM checkpoint not found: %s", p)
            return None
    for model_type in ("vit_l", "vit_h", "vit_b"):
        checkpoint = resolve_sam_checkpoint_for_type(model_type)
        if checkpoint is not None:
            return checkpoint
    return None


def resolve_sam_model_type(
    cli_value: str | None = None,
    config: dict | None = None,
    checkpoint: Path | None = None,
) -> str:
    """Resolve SAM architecture from an explicit value, checkpoint name, or config."""
    if config is None:
        config = _collect_file_config()
    allowed = ["vit_b", "vit_l", "vit_h"]
    if cli_value and cli_value.lower() in allowed:
        return cli_value.lower()

    resolved_checkpoint = checkpoint or resolve_sam_checkpoint(config=config, warn=False)
    if resolved_checkpoint:
        name = resolved_checkpoint.name.lower()
        for model_type in ("vit_h", "vit_l", "vit_b"):
            if model_type in name:
                return model_type

    for key in ("sam_model_type", "sam-model-type", "sam_model", "sam-model"):
        if key in config and str(config[key]).lower() in allowed:
            return str(config[key]).lower()
    return "vit_l"


def resolve_ben2_model(
    cli_value: str | None = None,
    config: dict | None = None,
    warn: bool = False,
) -> Path | None:
    """Resolve a local BEN2 ONNX model without allowing runtime downloads."""
    if config is None:
        config = _collect_file_config()
    values: list[Path] = []
    if cli_value:
        values.append(Path(cli_value).expanduser())
    else:
        for key in ("ben2_model", "ben2-model", "ben2Model"):
            if key in config:
                values.append(Path(str(config[key])).expanduser())
                break
        values.extend(
            [
                MODEL_CACHE_DIR / "ben2" / "BEN2_Base.onnx",
                MODEL_CACHE_DIR / "BEN2_Base.onnx",
                Path.cwd() / "BEN2_Base.onnx",
                Path(__file__).parent / "BEN2_Base.onnx",
                Path.home() / ".cache" / "passport-photos" / "BEN2_Base.onnx",
            ]
        )
    for candidate in values:
        if candidate.is_file():
            return candidate.resolve()
    if warn and values:
        LOG.warning("BEN2 model not found: %s", values[0])
    return None


def _valid_birefnet_model_dir(path: Path) -> bool:
    required = ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors")
    return path.is_dir() and all((path / name).is_file() for name in required)


def resolve_birefnet_model(
    cli_value: str | None = None,
    config: dict | None = None,
    warn: bool = False,
) -> Path | None:
    """Resolve a complete local BiRefNet_dynamic Transformers snapshot."""
    if config is None:
        config = _collect_file_config()
    values: list[Path] = []
    if cli_value:
        values.append(Path(cli_value).expanduser())
    else:
        for key in ("birefnet_model", "birefnet-model", "birefnetModel"):
            if key in config:
                values.append(Path(str(config[key])).expanduser())
                break
        values.extend(
            [
                MODEL_CACHE_DIR / "birefnet" / "BiRefNet_dynamic",
                MODEL_CACHE_DIR / "BiRefNet_dynamic",
                Path.cwd() / "BiRefNet_dynamic",
                Path(__file__).parent / "BiRefNet_dynamic",
                Path.home() / ".cache" / "passport-photos" / "BiRefNet_dynamic",
            ]
        )
    for candidate in values:
        model_dir = candidate.parent if candidate.is_file() else candidate
        if _valid_birefnet_model_dir(model_dir):
            return model_dir.resolve()
    if warn and values:
        LOG.warning(
            "BiRefNet_dynamic snapshot is incomplete or missing: %s "
            "(need config.json, BiRefNet_config.py, birefnet.py, and model.safetensors)",
            values[0],
        )
    return None


def resolve_yunet_model(
    cli_value: str | None = None, config: dict | None = None, warn: bool = False
) -> Path | None:
    """Resolve a user-supplied YuNet ONNX model from CLI, TOML, or local paths."""
    if config is None:
        config = _collect_file_config()
    if cli_value:
        p = Path(cli_value).expanduser()
        if p.is_file():
            return p.resolve()
        if warn:
            print(f"Warning: --yunet-model {p} not found.", file=sys.stderr)
        return None

    for key in (
        "yunet_model",
        "yunet-model",
        "yunet_checkpoint",
        "yunet-checkpoint",
        "yunet_onnx",
        "yunet-onnx",
    ):
        if key in config:
            p = Path(str(config[key])).expanduser()
            if p.is_file():
                return p.resolve()
            if warn:
                LOG.warning("Configured YuNet model not found: %s", p)
            return None
    # Fallbacks are local-only; model downloads are always user-managed.
    candidates = []
    with contextlib.suppress(Exception):
        candidates.append(_yunet_cache_path())
    candidates.extend(
        [
            MODEL_CACHE_DIR / "yunet" / "face_detection_yunet_2023mar.onnx",
            MODEL_CACHE_DIR / "yunet" / "yunet.onnx",
            MODEL_CACHE_DIR / "face_detection_yunet_2023mar.onnx",
            MODEL_CACHE_DIR / "yunet.onnx",
            Path.cwd() / "face_detection_yunet_2023mar.onnx",
            Path.cwd() / "yunet.onnx",
            Path(__file__).parent / "face_detection_yunet_2023mar.onnx",
            Path(__file__).parent / "yunet.onnx",
            Path.home() / ".cache" / "passport-photos" / "yunet.onnx",
            Path.home() / ".cache" / "passport-photos" / "face_detection_yunet_2023mar.onnx",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


# Backwards compat alias: old name used 'checkpoint' for YuNet; now 'model' is correct term.
resolve_yunet_checkpoint = resolve_yunet_model  # type: ignore


REMBG_RANKED_MODELS = [
    "bria-rmbg",
    "birefnet-general",
    "birefnet-portrait",
    "u2net_human_seg",
    "isnet-general-use",
]
REMBG_SUPPORTED_MODELS = set(REMBG_RANKED_MODELS) | {"u2net", "u2net_cloth_seg"}
REMBG_MODEL_ALIASES: dict[str, str] = {
    "birefnet": "birefnet-general",
    "birefnet-general-use": "birefnet-general",
    "birefnet_general": "birefnet-general",
    "general": "birefnet-general",
    "birefnet-portrait-use": "birefnet-portrait",
    "birefnet_portrait": "birefnet-portrait",
    "portrait": "birefnet-portrait",
    "birefnet_portrait-soft": "birefnet-portrait",
    "birefnet-portrait-soft": "birefnet-portrait",
    "birefnet-portrait-epoch_150": "birefnet-portrait",
    "birefnet-general-epoch_244": "birefnet-general",
    "isnet": "isnet-general-use",
    "isnet-general": "isnet-general-use",
    "isnet_general": "isnet-general-use",
    "u2net_human": "u2net_human_seg",
    "u2net-human-seg": "u2net_human_seg",
    "u2net_cloth": "u2net_cloth_seg",
    "bria": "bria-rmbg",
    "bria_rmbg": "bria-rmbg",
    "bria-rmbg-2.0": "bria-rmbg",
    "bria-rmbg-2_0": "bria-rmbg",
    "bria_rmbg-2.0": "bria-rmbg",
}


def _normalize_rembg_stem(stem: str) -> str:
    """Normalize a filename stem to canonical rembg model name (handles BiRefNet-epoch_*.onnx etc)."""
    s = stem.strip().lower()
    # direct alias
    if s in REMBG_MODEL_ALIASES:
        return REMBG_MODEL_ALIASES[s]
    # epoch files: BiRefNet-portrait-epoch_150 -> birefnet-portrait
    if s.startswith("birefnet-portrait"):
        return "birefnet-portrait"
    if s.startswith("birefnet-general"):
        return "birefnet-general"
    if s.startswith("bria"):
        return "bria-rmbg"
    if s == "isnet-general-use":
        return "isnet-general-use"
    return s


def _validate_rembg_model_name(value: str) -> str:
    normalized = _normalize_rembg_stem(REMBG_MODEL_ALIASES.get(value.lower(), value.lower()))
    if normalized not in REMBG_SUPPORTED_MODELS:
        expected = ", ".join(sorted(REMBG_SUPPORTED_MODELS))
        raise ValueError(f"Unsupported rembg model {value!r}; expected one of: {expected}")
    return normalized


def resolve_rembg_model(
    cli_value: str | None = None, config: dict | None = None, warn: bool = False
) -> str | None:
    """Resolve a canonical, user-supplied rembg model name or ONNX path."""
    if config is None:
        config = _collect_file_config()

    def _set_u2net_home_for_path(p: Path, stem: str) -> str:
        """Use a caller-supplied canonical rembg model without copying it."""
        canon = _validate_rembg_model_name(stem)
        expected_name = f"{canon}.onnx"
        if p.name.lower() != expected_name.lower():
            raise ValueError(
                f"rembg expects {expected_name}; rename {p.name} or place a canonical copy "
                "in U2NET_HOME"
            )
        if not p.is_file():
            raise FileNotFoundError(f"rembg model not found: {p}")
        os.environ["U2NET_HOME"] = str(p.parent.resolve())
        return canon

    if cli_value:
        value = cli_value.strip()
        path = Path(value).expanduser()
        if path.suffix.lower() == ".onnx":
            return _set_u2net_home_for_path(path, path.stem)
        return _validate_rembg_model_name(value)

    for key in ("rembg_model", "rembg-model", "rembgModel"):
        if key in config:
            value = str(config[key]).strip()
            path = Path(value).expanduser()
            if path.suffix.lower() == ".onnx":
                return _set_u2net_home_for_path(path, path.stem)
            return _validate_rembg_model_name(value)
    # Explicit rembg runs default to the portrait-specific model; auto has its own fixed chain.
    return "birefnet-portrait"


def _rembg_model_path(model_name: str) -> Path:
    """Return the exact local file rembg expects for a model name."""
    default_home = Path(os.environ.get("XDG_DATA_HOME", "~")).expanduser() / ".u2net"
    model_home = Path(os.environ.get("U2NET_HOME", default_home)).expanduser()
    return model_home / f"{model_name}.onnx"


def resolve_backend(
    cli_value: str | None,
    config_keys: list[str],
    config: dict | None,
    default: str,
    allowed: list[str],
) -> str:
    # Precedence: CLI > TOML > default.
    if config is None:
        config = _collect_file_config()
    if cli_value:
        v = cli_value.lower()
        if v in allowed:
            return v
        print(f"Warning: backend '{cli_value}' not in {allowed}, using {default}", file=sys.stderr)
    for ck in config_keys:
        if ck in config:
            v = str(config[ck]).lower()
            if v in allowed:
                return v
            raise ValueError(
                f"Invalid config value for {ck}: {config[ck]!r}; expected one of {allowed}"
            )
    return default


# --------------------------------------------------------------------------- #
# Debug / I/O
# --------------------------------------------------------------------------- #
def _save_debug(path: str | Path, image_bgr: np.ndarray) -> None:
    """Save debug/preview image — fast path via cv2.imwrite (no double cvtColor)."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite handles BGR/BGRA directly; avoid PIL double conversion
        # Ensure uint8
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)
        # Small optimization: if image is RGBA as BGR->RGBA mistake, just write
        ok = cv2.imwrite(str(path), image_bgr)
        if not ok:
            # Fallback via PIL for exotic formats
            if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(path)
            elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
                rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
                Image.fromarray(rgba).save(path)
            else:
                Image.fromarray(image_bgr).save(path)
    except Exception as e:
        print(f"Warning: could not save debug image {path}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Geometry helpers — NMS / filtering / ordering
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _nms(boxes: list[dict], iou_thresh: float = 0.3) -> list[dict]:
    """Greedy NMS — expects boxes sorted desc by confidence."""
    keep: list[dict] = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if _iou(best["bbox"], b["bbox"]) < iou_thresh]
    return keep


def _filter_and_rank(
    raw: list[dict], img_shape: tuple[int, int], conf_thresh: float = 0.0
) -> list[dict]:
    """Filter ghosts & rank by (confidence, area) descending."""
    h_img, w_img = img_shape[:2]
    img_area = h_img * w_img
    filtered: list[dict] = []
    for det in raw:
        x, y, w, h = det["bbox"]
        x1 = max(0, min(x, w_img))
        y1 = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))
        w_clipped = x2 - x1
        h_clipped = y2 - y1
        if w_clipped <= 0 or h_clipped <= 0:
            continue
        area = w_clipped * h_clipped
        if area < 0.001 * img_area or min(w_clipped, h_clipped) < 40:
            continue
        ar = w_clipped / h_clipped
        if not 0.6 <= ar <= 1.5 or det["confidence"] < conf_thresh:
            continue
        det["bbox"] = (x1, y1, w_clipped, h_clipped)
        det["area"] = area
        facial_area = det.get("facial_area")
        if isinstance(facial_area, dict):
            facial_area.update({"x": x1, "y": y1, "w": w_clipped, "h": h_clipped})
        filtered.append(det)

    filtered.sort(key=lambda d: (d["confidence"], d["area"]), reverse=True)
    ranked = _nms(
        sorted(filtered, key=lambda d: d["confidence"], reverse=True),
        iou_thresh=YUNET_NMS_THRESH,
    )
    ranked.sort(key=lambda d: (d["confidence"], d["area"]), reverse=True)
    return ranked


def _to_face_dict(bbox, confidence: float, img_shape) -> dict:
    """Canonical face dict used throughout the pipeline."""
    x, y, w, h = map(int, bbox)
    left_eye = (int(x + 0.30 * w), int(y + 0.38 * h))
    right_eye = (int(x + 0.70 * w), int(y + 0.38 * h))
    return {
        "bbox": (x, y, w, h),
        "area": w * h,
        "confidence": float(confidence),
        "facial_area": {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "left_eye": left_eye,
            "right_eye": right_eye,
        },
    }


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _detect_faces_yunet(
    image: np.ndarray, yunet_model: str | Path | None = None
) -> list[dict] | None:
    """Run YuNet with a resolved local ONNX model, or return ``None`` if unavailable."""
    if not hasattr(cv2, "FaceDetectorYN_create"):
        return None

    model_path = None
    if yunet_model is not None:
        p = Path(yunet_model).expanduser()
        if p.is_file():
            model_path = p.resolve()
        else:
            return None
    if model_path is None:
        model_path = resolve_yunet_model(warn=False)
    if model_path is None or not Path(model_path).is_file():
        return None
    if model_path.stat().st_size < 100_000:
        print(f"YuNet model {model_path} is too small to be valid.", file=sys.stderr)
        return None
    try:
        h, w = image.shape[:2]
        # YuNet works best at ~640-1280 max dimension; large 4000x3000 causes eye-only false positive (score 0.6, bbox quarter).
        # Downscale for detection, then scale bbox/eyes back to original. This fixes quarter-face bug.
        max_dim = max(h, w)
        target_max = 1000  # 1000 gives 1232x1776 detection with 0.93 score vs 0.60 at 4000
        scale = 1.0
        detect_img = image
        if max_dim > target_max:
            scale = target_max / max_dim
            nh, nw = int(h * scale), int(w * scale)
            detect_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            h_det, w_det = detect_img.shape[:2]
        else:
            h_det, w_det = h, w
            scale = 1.0
        detector = _get_yunet_detector(model_path, w_det, h_det)
        _, faces = detector.detect(detect_img)
        if faces is None:
            return []
        raw: list[dict] = []
        for det in faces:
            # det coords are in detect_img space; scale back to original
            x, y, w_, h_ = (det[0:4] / scale).astype(int) if scale != 1.0 else det[0:4].astype(int)
            score = float(det[14]) if det.shape[0] > 14 else float(det[4])
            x_re, y_re = (
                int(det[4] / scale) if scale != 1.0 else int(det[4]),
                int(det[5] / scale) if scale != 1.0 else int(det[5]),
            )
            x_le, y_le = (
                int(det[6] / scale) if scale != 1.0 else int(det[6]),
                int(det[7] / scale) if scale != 1.0 else int(det[7]),
            )
            fd = _to_face_dict((x, y, w_, h_), score, image.shape)
            if 0 <= x_re < image.shape[1] and 0 <= y_re < image.shape[0]:
                fd["facial_area"]["right_eye"] = (x_re, y_re)
                fd["facial_area"]["left_eye"] = (x_le, y_le)
            raw.append(fd)
        return _filter_and_rank(raw, image.shape, conf_thresh=YUNET_SCORE_THRESH)
    except Exception as e:
        print(f"YuNet detection failed ({e}).", file=sys.stderr)
        return None


def _detect_faces_haar(image: np.ndarray) -> list[dict]:
    """Haar-cascade fallback with histogram equalization & NMS — cached cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_cascade = _get_haar_cascade()

    candidates = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(candidates) == 0:
        candidates = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
    if len(candidates) == 0:
        return []

    h_img, w_img = image.shape[:2]
    raw: list[dict] = []
    for x, y, w, h in candidates:
        area = w * h
        cx, cy = x + w / 2, y + h / 2
        dist = abs(cx - w_img / 2) / w_img + abs(cy - h_img / 2) / h_img
        area_norm = min(1.0, area / (0.15 * h_img * w_img))
        confidence = 0.7 * area_norm + 0.3 * (1 - min(1, dist))
        raw.append(_to_face_dict((x, y, w, h), confidence, image.shape))

    return _filter_and_rank(raw, image.shape, conf_thresh=0.15)


def _install_mtcnn_weight_loader_workaround() -> None:
    """Make MTCNN close bundled LZ4 weight streams in the correct order."""
    try:
        from importlib import resources

        import joblib  # type: ignore
        import lz4.frame  # type: ignore
        from mtcnn.stages import stage_onet, stage_pnet, stage_rnet  # type: ignore
    except ImportError:
        return

    def load_weights(weights_name: str):
        candidates = [
            Path(weights_name).resolve(),
            resources.files("mtcnn.assets.weights") / weights_name,
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if str(candidate).lower().endswith(".lz4"):
                with (
                    resources.as_file(candidate) as weight_path,
                    lz4.frame.open(weight_path, mode="rb") as stream,
                ):
                    return joblib.load(stream)
            return joblib.load(candidate)
        raise FileNotFoundError(f"MTCNN weights file '{weights_name}' was not found.")

    # MTCNN 1.0 imports this function into each stage module. Its upstream
    # joblib.load(path) path leaves an LZ4 reader alive after the raw file closes,
    # which Python 3.13 reports as an ignored "I/O operation on closed file".
    for stage_module in (stage_pnet, stage_rnet, stage_onet):
        stage_module.__dict__["load_weights"] = load_weights


def _detect_faces_deepface(input_path: str, detector_backend: str = "opencv") -> list[dict] | None:
    """Run one DeepFace detector adapter, importing the optional dependency lazily."""
    if detector_backend == "retinaface":
        configured_home = os.environ.get("DEEPFACE_HOME")
        structured_home = MODEL_CACHE_DIR / "deepface"
        structured_weights = structured_home / ".deepface" / "weights" / "retinaface.h5"
        legacy_weights = MODEL_CACHE_DIR / ".deepface" / "weights" / "retinaface.h5"
        if configured_home:
            deepface_home = Path(configured_home).expanduser()
        elif structured_weights.is_file():
            deepface_home = structured_home
            os.environ["DEEPFACE_HOME"] = str(deepface_home.resolve())
        elif legacy_weights.is_file():
            deepface_home = MODEL_CACHE_DIR
            os.environ["DEEPFACE_HOME"] = str(deepface_home.resolve())
        else:
            deepface_home = Path.home()
        weights = deepface_home / ".deepface" / "weights" / "retinaface.h5"
        if not weights.is_file():
            print(
                f"RetinaFace weights not found at {weights}. Download them manually; "
                "the application never downloads models.",
                file=sys.stderr,
            )
            return None

    try:
        from deepface import DeepFace  # type: ignore
    except ImportError:
        return None

    if detector_backend == "mtcnn":
        _install_mtcnn_weight_loader_workaround()

    try:
        faces = DeepFace.extract_faces(
            img_path=input_path,
            detector_backend=detector_backend,
            enforce_detection=True,
        )
        raw: list[dict] = []
        for f in faces:
            fa = f["facial_area"]
            if "left_eye" not in fa or "right_eye" not in fa:
                x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                fa["left_eye"] = (int(x + 0.3 * w), int(y + 0.35 * h))
                fa["right_eye"] = (int(x + 0.7 * w), int(y + 0.35 * h))
            conf = float(f.get("confidence", 0.6))
            raw.append(
                {
                    "bbox": (int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])),
                    "area": int(fa["w"]) * int(fa["h"]),
                    "confidence": conf,
                    "facial_area": fa,
                }
            )
        return raw
    except Exception as e:
        print(f"DeepFace {detector_backend} detection failed ({e}).", file=sys.stderr)
        return None


def detect_faces_all(
    image: np.ndarray, input_path: str, method: str = "auto", yunet_model: str | Path | None = None
) -> list[dict]:
    """Run preferred detector(s), return ranked list (best first). Never crashes."""
    method = method.lower()
    all_faces: list[dict] | None = None

    if method in ("deepface", "retinaface", "mtcnn"):
        detector_backend = "opencv" if method == "deepface" else method
        all_faces = _detect_faces_deepface(input_path, detector_backend=detector_backend)
        if all_faces is not None:
            return _filter_and_rank(all_faces, image.shape)
        print(
            f"DeepFace {detector_backend} backend requested but unavailable or failed.",
            file=sys.stderr,
        )
        return []

    if method == "haar":
        haar = _detect_faces_haar(image)
        LOG.info("Haar cascade found %d face(s) after filtering/NMS.", len(haar))
        return haar

    if method == "yunet":
        yunet = _detect_faces_yunet(image, yunet_model=yunet_model)
        if yunet is None:
            print("YuNet requested but no valid local model is available.", file=sys.stderr)
            return []
        LOG.info("YuNet found %d face(s).", len(yunet))
        return yunet

    if method == "auto":
        yunet = _detect_faces_yunet(image, yunet_model=yunet_model)
        if yunet:
            LOG.info("Auto selected YuNet (%d face(s)).", len(yunet))
            return yunet

        haar = _detect_faces_haar(image)
        if haar:
            LOG.info("Auto selected Haar (%d face(s)).", len(haar))
            return haar

        df = _detect_faces_deepface(input_path)
        if df:
            ranked = _filter_and_rank(df, image.shape)
            if ranked:
                LOG.info("Auto selected DeepFace/OpenCV (%d face(s)).", len(ranked))
                return ranked
        return []

    print(f"Unknown face backend '{method}', using auto.", file=sys.stderr)
    return detect_faces_all(image, input_path, method="auto")


def detect_faces(
    image: np.ndarray,
    input_path: str,
    select_face: bool = False,
    face_index: int | None = None,
    export_faces_dir: str | Path | None = None,
    face_backend: str = "auto",
    yunet_model: str | Path | None = None,
    crop_aspect: float = 1.0,
    overwrite_exports: bool = False,
    protected_export_paths: tuple[str | Path, ...] = (),
) -> dict:
    """Detect and pick a single face. Handles empty/ghost cases gracefully."""
    faces = detect_faces_all(image, input_path, method=face_backend, yunet_model=yunet_model)

    if export_faces_dir is not None:
        export_all_faces(
            image,
            faces,
            export_faces_dir,
            crop_aspect=crop_aspect,
            overwrite=overwrite_exports,
            protected_paths=protected_export_paths,
        )

    if len(faces) == 0:
        raise ValueError(
            f"No faces found in {input_path} (backend={face_backend}). Try --face-backend haar/yunet/retinaface/mtcnn or "
            f"check lighting, or use --export-faces to inspect detections."
        )

    print(
        f"\nFound {len(faces)} candidate face(s) (backend={face_backend}, sorted by decreasing confidence → size):"
    )
    for i, f in enumerate(faces):
        fa = f["facial_area"]
        print(
            f"  [{i}] conf={f['confidence']:.3f}  bbox=({fa['x']},{fa['y']},{fa['w']},{fa['h']})  "
            f"area={fa['w'] * fa['h']}  eyes={fa['left_eye']}/{fa['right_eye']}"
        )
    if export_faces_dir is not None:
        print(
            f"  Previews written to {Path(export_faces_dir).resolve()}  (face_00.png is top-ranked)"
        )

    if face_index is not None:
        if not 0 <= face_index < len(faces):
            raise ValueError(
                f"--face-index {face_index} out of range [0,{len(faces) - 1}]. "
                f"Use --export-faces to inspect candidates."
            )
        print(f"Selected face [{face_index}] via --face-index.")
        return faces[face_index]

    if select_face:
        # Guard: interactive selection requires a TTY and is not valid for batch
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--select-face requires an interactive terminal (TTY). Use --face-index N for non-interactive/CLI selection, or --export-faces for batch."
            )
        while True:
            choice = (
                input(f"Select face index [0-{len(faces) - 1}] (Enter=best, 'l'=list): ")
                .strip()
                .lower()
            )
            if choice == "":
                print("Using best-ranked face [0].")
                return faces[0]
            if choice == "l":
                for i, f in enumerate(faces):
                    fa = f["facial_area"]
                    print(f"  [{i}] conf={f['confidence']:.3f} bbox={fa}")
                continue
            try:
                idx = int(choice)
                if 0 <= idx < len(faces):
                    return faces[idx]
                print(f"Out of range [0,{len(faces) - 1}].")
            except ValueError:
                print(f"Invalid choice '{choice}'. Enter a number or press Enter.")

    print(f"Selected best-ranked face [0] (conf={faces[0]['confidence']:.3f}).")
    return faces[0]


def _path_collision_key(path: str | Path) -> str:
    """Return a conservative path key that also catches case-insensitive collisions."""
    resolved = str(Path(path).expanduser().resolve())
    return unicodedata.normalize("NFC", resolved).casefold()


def _path_is_at_or_below(path: str | Path, parent: str | Path) -> bool:
    path_key = _path_collision_key(path)
    parent_key = _path_collision_key(parent).rstrip(os.sep)
    return path_key == parent_key or path_key.startswith(parent_key + os.sep)


def _preview_dir_for_source(base_dir: str | Path, source: Path, *, batch: bool) -> Path:
    if not batch:
        return Path(base_dir)
    digest = hashlib.sha256(source.name.encode("utf-8")).hexdigest()[:12]
    return Path(base_dir) / f"{source.name}_{digest}"


def export_all_faces(
    image: np.ndarray,
    faces: list[dict],
    output_dir: str | Path,
    *,
    crop_aspect: float = 1.0,
    overwrite: bool = False,
    protected_paths: tuple[str | Path, ...] = (),
) -> list[Path]:
    """Export candidate crops without silently replacing inputs or prior previews."""
    import json

    out = Path(output_dir)
    metadata_path = out / "faces.json"
    preview_paths = [out / f"face_{idx:02d}.png" for idx in range(len(faces))]
    if out.exists() and not out.is_dir():
        raise FileExistsError(f"Face preview path exists and is not a directory: {out}")
    owned_existing: list[Path] = []
    if out.is_dir():
        for path in out.iterdir():
            folded_name = path.name.casefold()
            face_number = folded_name.removeprefix("face_").removesuffix(".png")
            if folded_name == "faces.json" or (
                folded_name.startswith("face_")
                and folded_name.endswith(".png")
                and face_number.isdigit()
            ):
                owned_existing.append(path)

    protected_keys = {_path_collision_key(path) for path in protected_paths}
    for protected_path in protected_paths:
        if _path_is_at_or_below(out, protected_path):
            raise ValueError(f"Face preview directory overlaps a protected file: {out}")
    for candidate in set(preview_paths + owned_existing + [metadata_path]):
        if _path_collision_key(candidate) in protected_keys:
            raise ValueError(f"Face preview would overwrite a protected file: {candidate}")
    if owned_existing and not overwrite:
        raise FileExistsError(
            f"Face preview files already exist in {out}; use --overwrite to replace them"
        )

    out.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_path in owned_existing:
            stale_path.unlink(missing_ok=True)

    written: list[Path] = []
    meta = []
    for idx, face in enumerate(faces):
        try:
            crop = crop_image(image, face, idx=None, output_aspect=crop_aspect)
        except Exception as e:
            print(f"Warning: could not crop face [{idx}] ({e}) — skipping export.", file=sys.stderr)
            continue
        filename = preview_paths[idx]
        _save_output_image(crop, filename)
        written.append(filename)
        area = face["facial_area"]
        meta.append(
            {
                "index": idx,
                "confidence": face["confidence"],
                "bbox": {"x": area["x"], "y": area["y"], "w": area["w"], "h": area["h"]},
                "eyes": {"left": area["left_eye"], "right": area["right_eye"]},
                "file": filename.name,
            }
        )
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Exported {len(written)} face crop(s) to {out.resolve()}")
    return written


def crop_image(
    image: np.ndarray,
    face: dict,
    idx: int | None = None,
    *,
    output_aspect: float = 1.0,
) -> np.ndarray:
    """Create an in-bounds passport crop at the requested width/height ratio."""
    if face is None or "facial_area" not in face:
        raise ValueError("Invalid face dict — missing facial_area")
    fa = face["facial_area"]
    try:
        fx, fy, fw, fh = int(fa["x"]), int(fa["y"]), int(fa["w"]), int(fa["h"])
        left_eye, right_eye = fa["left_eye"], fa["right_eye"]
    except Exception as e:
        raise ValueError(f"Malformed facial_area {fa}: {e}") from e

    if fw <= 0 or fh <= 0:
        raise ValueError(f"Face bbox has non-positive size fw={fw} fh={fh}")

    ypad = int(0.1 * fh)
    xpad = int(0.1 * fw)
    fy = fy - ypad
    fx = fx - xpad
    fh = int(fh + 2 * ypad)
    fw = int(fw + 2 * xpad)

    try:
        eye_height = (int(left_eye[1]) + int(right_eye[1])) // 2
        face_center_x = fx + fw // 2
    except Exception:
        eye_height = fy + int(0.38 * fh)
        face_center_x = fx + fw // 2

    if not math.isfinite(output_aspect) or output_aspect <= 0:
        raise ValueError(f"output aspect ratio must be positive and finite (got {output_aspect!r})")

    resolution = max(1, fh / 1.25)
    desired_height = max(1, int(round(2 * resolution)))
    desired_width = max(1, int(round(desired_height * output_aspect)))
    desired_left = face_center_x - desired_width // 2
    desired_top = eye_height - int(0.95 * resolution)

    # Clip the originally requested rectangle at every edge. If clipping changes the
    # aspect ratio, crop inward rather than translating the frame or stretching pixels.
    image_height, image_width = image.shape[:2]
    crop_top = max(0, desired_top)
    crop_left = max(0, desired_left)
    crop_bottom = min(image_height, desired_top + desired_height)
    crop_right = min(image_width, desired_left + desired_width)
    available_height = crop_bottom - crop_top
    available_width = crop_right - crop_left
    if available_width > 0 and available_height > 0:
        if available_width / available_height > output_aspect:
            crop_width = max(1, int(math.floor(available_height * output_aspect)))
            crop_left = min(
                max(crop_left, face_center_x - crop_width // 2), crop_right - crop_width
            )
            crop_right = crop_left + crop_width
        else:
            crop_height = max(1, int(math.floor(available_width / output_aspect)))
            crop_bottom = crop_top + crop_height

    if crop_bottom <= crop_top or crop_right <= crop_left:
        raise ValueError(
            f"Crop would be empty: top={crop_top} bottom={crop_bottom} "
            f"left={crop_left} right={crop_right} image={image.shape}"
        )

    cropped = image[crop_top:crop_bottom, crop_left:crop_right].copy()
    if cropped.size == 0:
        raise ValueError("Crop resulted in empty image — face near border?")

    if idx is not None:
        with contextlib.suppress(Exception):
            _save_debug(f"face{idx}.png", cropped)
        print("Image cropped to face")
    return cropped


def _sam_checkpoint_identity(checkpoint: Path, model_type: str) -> str:
    resolved = checkpoint.resolve()
    checkpoint_stat = resolved.stat()
    return (
        f"{resolved}:{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}:{model_type.lower()}"
    )


def _get_sam_model(checkpoint: Path, model_type: str | None = None):
    """Cache SAM model per checkpoint path (cpu). Heavy ~400 MB, avoid reload on batch.

    model_type: vit_b | vit_l | vit_h — inferred from checkpoint name if not given.
    """
    if model_type is None:
        # infer from checkpoint filename
        n = checkpoint.name.lower()
        if "vit_l" in n:
            model_type = "vit_l"
        elif "vit_h" in n:
            model_type = "vit_h"
        else:
            model_type = "vit_b"
    model_type = model_type.lower()
    from segment_anything import sam_model_registry  # type: ignore

    if model_type not in sam_model_registry:
        print(
            f"Warning: SAM model_type {model_type} not in registry, falling back to vit_l",
            file=sys.stderr,
        )
        model_type = "vit_l"
    key = _sam_checkpoint_identity(checkpoint, model_type)
    if key in _SAM_CACHE:
        return _SAM_CACHE[key]
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device="cpu")
    _SAM_CACHE[key] = sam
    return sam


def _seg_cache_path_for_image(image: np.ndarray, suffix: str | None = None) -> Path:
    """Return an ignored, per-image SAM mask-cache path under the model cache."""
    try:
        h = hashlib.sha1(image.tobytes()).hexdigest()[:12]
    except Exception:
        # fallback: hash shape+sum if tobytes fails
        h = hashlib.sha1(f"{image.shape}{image.sum()}".encode()).hexdigest()[:12]
    if suffix:
        # sanitize suffix for filename: keep alnum, -, _
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in suffix)[:32]
        return SEGMENTATION_CACHE_DIR / f"segmentation_{h}_{safe}.npz"
    return SEGMENTATION_CACHE_DIR / f"segmentation_{h}.npz"


def _remove_background_sam(
    image, sam_checkpoint: Path | None, resegment=False, sam_model_type: str | None = None
):
    """Remove the background with SamPredictor and a matching local checkpoint."""
    try:
        from segment_anything import SamPredictor  # type: ignore
    except ImportError:
        return None, False

    if sam_checkpoint is None or not Path(sam_checkpoint).is_file():
        print(
            f"SAM checkpoint {sam_checkpoint} not found, skipping SAM background removal.",
            file=sys.stderr,
        )
        return None, False

    sam_checkpoint = Path(sam_checkpoint).resolve()
    checkpoint_identity = _sam_checkpoint_identity(
        sam_checkpoint,
        sam_model_type or resolve_sam_model_type(checkpoint=sam_checkpoint),
    )
    checkpoint_digest = hashlib.sha256(checkpoint_identity.encode()).hexdigest()[:16]
    cache_suffix = f"recall-v3_{sam_model_type or 'auto'}_{checkpoint_digest}"
    seg_path = _seg_cache_path_for_image(image, suffix=cache_suffix)
    # A cache is valid only for this image hash, checkpoint, and model type.
    cache_path: Path | None = seg_path if seg_path.exists() and not resegment else None
    cached_mask = None
    used_save = cache_path is not None
    if used_save and cache_path is not None and not resegment:
        try:
            with np.load(cache_path, allow_pickle=False) as cached_data:
                seg = cached_data["segmentation"]
            if seg.ndim != 2 or seg.shape != image.shape[:2]:
                print(
                    f"Cached {cache_path} shape {seg.shape} != image {image.shape[:2]}, regenerating.",
                    file=sys.stderr,
                )
                cached_mask = None
                used_save = False
            else:
                cached_mask = {"segmentation": seg.astype(bool, copy=False)}
                print(f"Using cached {cache_path} (shape={seg.shape})")
        except Exception as e:
            print(f"Could not load {cache_path} ({e}), regenerating.", file=sys.stderr)
            cached_mask = None
            used_save = False

    if cached_mask is not None and not resegment and used_save:
        max_mask = cached_mask
        # cached mask was already selected with improved logic; no need to re-filter
    else:
        sam = _get_sam_model(sam_checkpoint, model_type=sam_model_type)
        print(
            f"Segmentation model loaded ({sam_model_type or 'auto-inferred'} from {sam_checkpoint})"
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_mask = None
        try:
            from segment_anything import SamPredictor  # type: ignore

            predictor = SamPredictor(sam)
            predictor.set_image(image_rgb)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            chest = (w // 2, min(h - 1, int(h * 0.78)))
            # Bottom corners often contain shoulders in a passport crop, so they must
            # not be negative prompts. Positive-only prompts favor foreground recall.
            prompt_sets = [
                (np.array([center]), np.array([1])),
                (np.array([center, chest]), np.array([1, 1])),
            ]
            all_cands = []  # (raw_score, area, segmentation, prompt_idx, mask_idx)
            for pi, (pcs, lbs) in enumerate(prompt_sets):
                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=pcs,
                        point_labels=lbs,
                        multimask_output=True,
                    )
                    for mi, seg in enumerate(masks):
                        area = float(seg.mean())
                        if not (0.12 <= area <= 0.80):
                            continue
                        if not seg[center[1], center[0]] or not seg[chest[1], chest[0]]:
                            continue
                        patch_h = max(1, h // 20)
                        patch_w = max(1, w // 20)
                        top_corner_coverage = max(
                            float(seg[:patch_h, :patch_w].mean()),
                            float(seg[:patch_h, -patch_w:].mean()),
                        )
                        if top_corner_coverage > 0.50:
                            continue
                        all_cands.append((float(scores[mi]), area, seg, pi, mi))
                except Exception as e:
                    print(f"SAM predictor prompt {pi} failed {e}", file=sys.stderr)

            if all_cands:
                best_score = max(candidate[0] for candidate in all_cands)
                near_best = [
                    candidate for candidate in all_cands if candidate[0] >= best_score - 0.05
                ]
                anchor = max(near_best, key=lambda candidate: (candidate[1], candidate[0]))

                def mask_iou(candidate) -> float:
                    intersection = np.logical_and(anchor[2], candidate[2]).sum()
                    union = np.logical_or(anchor[2], candidate[2]).sum()
                    return float(intersection / max(union, 1))

                compatible = [candidate for candidate in near_best if mask_iou(candidate) >= 0.65]
                segmentation = np.logical_or.reduce([candidate[2] for candidate in compatible])
                union_area = float(segmentation.mean())
                if union_area > 0.80:
                    segmentation = anchor[2]
                    union_area = anchor[1]
                    compatible = [anchor]
                max_mask = {
                    "segmentation": segmentation,
                    "area": int(segmentation.sum()),
                    "bbox": [0, 0, w, h],
                    "predicted_iou": float(best_score),
                    "stability_score": 0.95,
                }
                print(
                    f"SAM Predictor: merged {len(compatible)}/{len(all_cands)} plausible masks: "
                    f"area={union_area:.1%} best_score={best_score:.3f} "
                    f"(largest prompt {anchor[3]} mask {anchor[4]}) - recall-first"
                )
            else:
                print("SAM Predictor produced no valid full-person mask.", file=sys.stderr)
                max_mask = None
        except Exception as e:
            print(f"SAM Predictor failed ({e}).", file=sys.stderr)
            max_mask = None
        if max_mask is None:
            return None, False
        # Cache the selected predictor mask.
        try:
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                seg_path,
                segmentation=np.asarray(max_mask["segmentation"], dtype=np.bool_),
            )
            print(f"Cached segmentation to {seg_path}")
        except Exception as e:
            print(f"Warning: could not cache {seg_path} ({e})", file=sys.stderr)

    if max_mask is None or "segmentation" not in max_mask:
        print("No valid SAM mask found, skipping.", file=sys.stderr)
        return None, False

    seg = max_mask["segmentation"]
    if seg.shape[:2] != image.shape[:2]:
        print(
            f"SAM mask shape {seg.shape[:2]} != image {image.shape[:2]}, skipping SAM.",
            file=sys.stderr,
        )
        return None, False

    # Use mask directly as foreground (center-containing). No bbox inversion heuristic (fragile).
    segmentation = seg.astype(bool)
    if not segmentation[seg.shape[0] // 2, seg.shape[1] // 2]:
        print("SAM mask center is background; inverting mask.", file=sys.stderr)
        segmentation = np.logical_not(segmentation)
    # Fill small holes without opening/eroding hair or clothing; then expand by one
    # pixel because retaining a little background is safer than clipping the person.
    try:
        seg_u8 = segmentation.astype(np.uint8) * 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        seg_u8 = cv2.dilate(seg_u8, dilation_kernel, iterations=1)
        refined = seg_u8 > 0
        if (
            refined[seg.shape[0] // 2, seg.shape[1] // 2]
            and abs(refined.mean() - segmentation.mean()) < 0.15
        ):
            segmentation = refined
        else:
            print(
                "Conservative SAM refinement failed validation; keeping raw mask", file=sys.stderr
            )
    except Exception as e:
        print(f"Mask refinement skipped {e}", file=sys.stderr)

    alpha = cv2.GaussianBlur(segmentation.astype(np.float32), (5, 5), 0)
    result = (
        image.astype(np.float32) * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
    ).astype(np.uint8)

    print(f"Background removed (SAM, checkpoint={sam_checkpoint})")
    return result, True


def _remove_background_rembg(image, rembg_model: str | None = None):
    """Remove the background with a preflighted local rembg ONNX model."""
    try:
        from rembg import new_session, remove  # type: ignore
    except ImportError:
        return None, False

    # Explicit rembg selection; the multi-backend automatic ranking is handled by remove_background.
    model_name = rembg_model or resolve_rembg_model(warn=False) or "birefnet-portrait"
    try:
        model_name = _validate_rembg_model_name(model_name.strip())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return None, False
    # Ensure aliases covered via REMBG_MODEL_ALIASES / _normalize_rembg_stem, keep old fallback for compat
    if model_name in ("birefnet", "birefnet-general-use", "birefnet_general", "general"):
        model_name = "birefnet-general"
    elif model_name in (
        "birefnet-portrait-use",
        "birefnet_portrait",
        "portrait",
        "birefnet_portrait-soft",
        "birefnet-portrait-epoch_150",
    ):
        model_name = "birefnet-portrait"
    elif model_name in ("isnet", "isnet-general", "isnet_general"):
        model_name = "isnet-general-use"
    elif model_name in ("u2net_human", "u2net-human-seg"):
        model_name = "u2net_human_seg"
    elif model_name == "u2net_cloth":
        model_name = "u2net_cloth_seg"
    elif model_name in ("bria", "bria_rmbg", "bria-rmbg-2.0", "bria-rmbg-2_0"):
        model_name = "bria-rmbg"
    configured_model = _rembg_model_path(model_name)
    local_models = [
        MODEL_CACHE_DIR / "rembg" / f"{model_name}.onnx",
        MODEL_CACHE_DIR / f"{model_name}.onnx",
    ]
    if not configured_model.is_file():
        configured_model = next(
            (candidate for candidate in local_models if candidate.is_file()),
            configured_model,
        )
    if not configured_model.is_file():
        print(
            f"rembg model {model_name} not found at {configured_model}; "
            "download it manually before running this backend.",
            file=sys.stderr,
        )
        return None, False

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rembg normally calls pooch.retrieve while creating a session. The model was
        # preflighted above, so disable checksum retrieval for this call to guarantee
        # that rembg cannot replace or download it; ONNX Runtime still validates it.
        session_key = (model_name, str(configured_model.resolve()))
        session = _REMBG_SESSION_CACHE.get(session_key)
        if session is None:
            previous_checksum_setting = os.environ.get("MODEL_CHECKSUM_DISABLED")
            previous_model_home = os.environ.get("U2NET_HOME")
            os.environ["MODEL_CHECKSUM_DISABLED"] = "1"
            os.environ["U2NET_HOME"] = str(configured_model.parent.resolve())
            try:
                session = new_session(model_name)  # type: ignore
                _REMBG_SESSION_CACHE[session_key] = session
            finally:
                if previous_checksum_setting is None:
                    os.environ.pop("MODEL_CHECKSUM_DISABLED", None)
                else:
                    os.environ["MODEL_CHECKSUM_DISABLED"] = previous_checksum_setting
                if previous_model_home is None:
                    os.environ.pop("U2NET_HOME", None)
                else:
                    os.environ["U2NET_HOME"] = previous_model_home
        out = remove(rgb, session=session)  # type: ignore
        if isinstance(out, bytes):
            from io import BytesIO

            pil_img = Image.open(BytesIO(out)).convert("RGBA")
        elif isinstance(out, Image.Image):
            pil_img = out.convert("RGBA")
        else:
            pil_img = Image.fromarray(out)

        white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        white_bg.paste(pil_img, mask=pil_img.split()[3])
        rgb_white = cv2.cvtColor(np.array(white_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        print(f"Background removed (rembg, model={model_name})")
        return rgb_white, True
    except Exception as e:
        print(f"rembg background removal failed: {e}", file=sys.stderr)
        return None, False


def _composite_on_white(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite a BGR image over white using a soft foreground mask."""
    if alpha.shape[:2] != image.shape[:2]:
        alpha = cv2.resize(
            alpha.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    return (
        image.astype(np.float32) * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
    ).astype(np.uint8)


def _remove_background_ben2(image: np.ndarray, model_path: Path | None):
    """Run the official BEN2 Base ONNX export with a user-supplied local model."""
    if model_path is None or not model_path.is_file():
        print(
            f"BEN2 model {model_path} not found; download BEN2_Base.onnx manually.",
            file=sys.stderr,
        )
        return None, False
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return None, False

    try:
        model_key = str(model_path.resolve())
        session = _ONNX_SESSION_CACHE.get(model_key)
        if session is None:
            session = ort.InferenceSession(
                model_key,
                providers=["CPUExecutionProvider"],
            )
            _ONNX_SESSION_CACHE[model_key] = session

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        input_tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        output = session.run(None, {session.get_inputs()[0].name: input_tensor})[0]
        if not isinstance(output, np.ndarray):
            raise TypeError(f"Unexpected BEN2 output type: {type(output).__name__}")
        alpha = np.squeeze(output).astype(np.float32)
        minimum = float(alpha.min())
        maximum = float(alpha.max())
        if maximum - minimum > 1e-6:
            alpha = (alpha - minimum) / (maximum - minimum)
        else:
            alpha = np.clip(alpha, 0.0, 1.0)
        result = _composite_on_white(image, alpha)
        print(f"Background removed (BEN2, model={model_path})")
        return result, True
    except Exception as e:
        print(f"BEN2 background removal failed: {e}", file=sys.stderr)
        return None, False


def _get_birefnet_model(model_dir: Path):
    """Load and cache a local-only BiRefNet_dynamic Transformers snapshot on CPU."""
    model_key = str(model_dir.resolve())
    if model_key in _BIREFNET_CACHE:
        return _BIREFNET_CACHE[model_key]
    from transformers import AutoModelForImageSegmentation  # type: ignore

    model = AutoModelForImageSegmentation.from_pretrained(
        model_key,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device="cpu").float().eval()
    _BIREFNET_CACHE[model_key] = model
    return model


def _remove_background_birefnet(image: np.ndarray, model_dir: Path | None):
    """Run BiRefNet_dynamic from a complete, pinned local Transformers snapshot."""
    if model_dir is None or not _valid_birefnet_model_dir(model_dir):
        print(
            f"BiRefNet_dynamic model snapshot {model_dir} is missing or incomplete.",
            file=sys.stderr,
        )
        return None, False
    try:
        import torch
    except ImportError:
        return None, False

    try:
        model = _get_birefnet_model(model_dir)
        original_h, original_w = image.shape[:2]
        scale = min(1.0, BIREFNET_MAX_SIDE / max(original_h, original_w))
        inference_h = max(1, int(round(original_h * scale)))
        inference_w = max(1, int(round(original_w * scale)))

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (inference_h, inference_w) != (original_h, original_w):
            rgb = cv2.resize(
                rgb,
                (inference_w, inference_h),
                interpolation=cv2.INTER_AREA,
            )

        padded_h = (
            (inference_h + BIREFNET_SIZE_MULTIPLE - 1)
            // BIREFNET_SIZE_MULTIPLE
            * BIREFNET_SIZE_MULTIPLE
        )
        padded_w = (
            (inference_w + BIREFNET_SIZE_MULTIPLE - 1)
            // BIREFNET_SIZE_MULTIPLE
            * BIREFNET_SIZE_MULTIPLE
        )
        if (padded_h, padded_w) != (inference_h, inference_w):
            border_mode = (
                cv2.BORDER_REFLECT_101
                if inference_h > 1 and inference_w > 1
                else cv2.BORDER_REPLICATE
            )
            rgb = cv2.copyMakeBorder(
                rgb,
                0,
                padded_h - inference_h,
                0,
                padded_w - inference_w,
                border_mode,
            )

        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        with torch.inference_mode():
            prediction = model(input_tensor)[-1].sigmoid()
        alpha = prediction[0].squeeze().float().cpu().numpy()
        alpha = alpha[:inference_h, :inference_w]
        result = _composite_on_white(image, alpha)
        print(
            f"Background removed (BiRefNet_dynamic, model={model_dir}, input={padded_w}x{padded_h})"
        )
        return result, True
    except Exception as e:
        print(f"BiRefNet_dynamic background removal failed: {e}", file=sys.stderr)
        return None, False


def remove_background(
    image,
    background_backend: str = "auto",
    sam_checkpoint: Path | None = None,
    rembg_model: str | None = None,
    ben2_model: Path | None = None,
    birefnet_model: Path | None = None,
    resegment=False,
    sam_model_type: str | None = None,
):
    """Apply an explicit backend or the fixed quality-ranked local automatic chain."""
    background_backend = background_backend.lower()
    if background_backend == "none":
        LOG.info("Background removal skipped (--background-backend none / --keep-background).")
        return image
    if background_backend == "sam":
        result, ok = _remove_background_sam(
            image, sam_checkpoint=sam_checkpoint, resegment=resegment, sam_model_type=sam_model_type
        )
        if ok:
            return result
        raise RuntimeError(
            f"SAM backend requested but failed (checkpoint={sam_checkpoint} "
            f"model_type={sam_model_type}). Install the SAM extra and provide a matching local checkpoint."
        )
    if background_backend == "rembg":
        result, ok = _remove_background_rembg(image, rembg_model=rembg_model)
        if ok:
            return result
        raise RuntimeError(
            "rembg backend requested but unavailable or failed. Install the rembg extra "
            "and provide a canonical local ONNX file through --rembg-model or U2NET_HOME."
        )
    if background_backend == "ben2":
        result, ok = _remove_background_ben2(image, model_path=ben2_model)
        if ok:
            return result
        raise RuntimeError(
            "BEN2 backend requested but unavailable or failed. Install the ben2 extra "
            "and provide a local BEN2_Base.onnx through --ben2-model."
        )
    if background_backend == "birefnet":
        result, ok = _remove_background_birefnet(image, model_dir=birefnet_model)
        if ok:
            return result
        raise RuntimeError(
            "BiRefNet backend requested but unavailable or failed. Install the birefnet "
            "extra and provide a complete local BiRefNet_dynamic snapshot through "
            "--birefnet-model."
        )

    # Quality-ranked automatic chain requested for passport photos:
    # BRIA > BiRefNet General > BiRefNet Portrait > BiRefNet Dynamic >
    # U2Net Human > ISNet General > BEN2 > SAM ViT-L > ViT-H > ViT-B.
    for model_name in REMBG_RANKED_MODELS[:3]:
        result, ok = _remove_background_rembg(image, rembg_model=model_name)
        if ok:
            return result

    if birefnet_model is not None and _valid_birefnet_model_dir(birefnet_model):
        result, ok = _remove_background_birefnet(image, model_dir=birefnet_model)
        if ok:
            return result

    for model_name in REMBG_RANKED_MODELS[3:]:
        result, ok = _remove_background_rembg(image, rembg_model=model_name)
        if ok:
            return result

    if ben2_model is not None and ben2_model.is_file():
        result, ok = _remove_background_ben2(image, model_path=ben2_model)
        if ok:
            return result

    for model_type in ("vit_l", "vit_h", "vit_b"):
        checkpoint = resolve_sam_checkpoint_for_type(
            model_type,
            preferred_checkpoint=sam_checkpoint,
            preferred_model_type=sam_model_type,
        )
        if checkpoint is None:
            continue
        result, ok = _remove_background_sam(
            image,
            sam_checkpoint=checkpoint,
            resegment=resegment,
            sam_model_type=model_type,
        )
        if ok:
            return result

    print("Warning: no configured local background remover succeeded.", file=sys.stderr)
    print("Install a background extra and provide the corresponding local model.", file=sys.stderr)
    print(
        "Keeping original background (use --keep-background / --background-backend none to silence).",
        file=sys.stderr,
    )
    return image


def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    LOG.info("Image resized to %dx%d", width, height)
    return resized_image


def _output_path_for_format(output_path: str | Path, fmt: str | None) -> Path:
    p = Path(output_path)
    if fmt and fmt != "auto":
        fmt_norm = fmt.lower().lstrip(".")
        ext = ".jpg" if fmt_norm in ("jpeg", "jpg") else f".{fmt_norm}"
        p = p.with_suffix(ext)
    return p


def _save_output_image(
    image_bgr: np.ndarray,
    output_path: str | Path,
    dpi: int = 400,
    quality: int = 95,
    fmt: str | None = None,
) -> None:
    """Save final image with DPI metadata and format handling."""
    p = _output_path_for_format(output_path, fmt)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert BGR->RGBA for Pillow save to preserve channels correctly
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        # Write via Pillow to embed dpi
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # Pillow dpi param expects tuple
        save_kwargs: dict[str, Any] = {"dpi": (dpi, dpi)}
        if p.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = quality
            save_kwargs["subsampling"] = 0
            pil = pil.convert("RGB")
        pil.save(p, **save_kwargs)
    else:
        rgba = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA) if image_bgr.shape[2] == 4 else image_bgr
        )
        Image.fromarray(rgba).save(p, dpi=(dpi, dpi))


def process_image(
    input_path: str,
    output_path: str,
    *,
    width: float = 2,
    height: float = 2,
    tiled_width: float = 10,
    tiled_height: float = 8,
    dpi: int = 400,
    select_face: bool = False,
    face_index: int | None = None,
    export_faces_dir: str | Path | None = None,
    face_backend: str = "auto",
    background_backend: str = "none",
    sam_checkpoint: str | Path | None = None,
    sam_model_type: str | None = None,
    yunet_model: str | Path | None = None,
    rembg_model: str | None = None,
    ben2_model: str | Path | None = None,
    birefnet_model: str | Path | None = None,
    keep_untiled: bool = True,
    keep_background: bool = False,
    resegment: bool = False,
    output_format: str | None = None,
    jpeg_quality: int = 95,
    overwrite: bool = False,
    dry_run: bool = False,
):
    if output_format and output_format.lower().lstrip(".") not in {
        "auto",
        "png",
        "jpg",
        "jpeg",
    }:
        raise ValueError(
            f"Unsupported output format {output_format!r}; expected auto, png, or jpeg"
        )
    dimensions = {
        "photo width": width,
        "photo height": height,
        "paper width": tiled_width,
        "paper height": tiled_height,
    }
    for name, value in dimensions.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number (got {value!r})")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer (got {dpi!r})")
    photo_width_px = int(round(width * dpi))
    photo_height_px = int(round(height * dpi))
    paper_width_px = int(round(tiled_width * dpi))
    paper_height_px = int(round(tiled_height * dpi))
    pixel_dimensions = {
        "photo width": photo_width_px,
        "photo height": photo_height_px,
    }
    if not keep_untiled:
        pixel_dimensions.update(
            {
                "paper width": paper_width_px,
                "paper height": paper_height_px,
            }
        )
    for name, value in pixel_dimensions.items():
        if value < 1:
            raise ValueError(f"{name} rounds to zero pixels at {dpi} DPI")
    output_width = photo_width_px if keep_untiled else paper_width_px
    output_height = photo_height_px if keep_untiled else paper_height_px
    output_pixels = output_width * output_height
    if output_pixels > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"Output would contain {output_pixels:,} pixels; maximum is "
            f"{MAX_OUTPUT_PIXELS:,}. Reduce dimensions or DPI."
        )
    out_p = _output_path_for_format(output_path, output_format)
    if out_p.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output {out_p} exists; pass overwrite=True to replace it")
    try:
        img_cv = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise FileNotFoundError(f"Could not read image at {input_path}")

        effective_bg_backend = "none" if keep_background else background_backend
        if dry_run:
            target_width = int(round((width if keep_untiled else tiled_width) * dpi))
            target_height = int(round((height if keep_untiled else tiled_height) * dpi))
            print(
                f"[dry-run] input={input_path} output={out_p} size={target_width}x{target_height} "
                f"face_backend={face_backend} background_backend={effective_bg_backend}"
            )
            return None
        # Only resolve SAM checkpoint if background removal may need it (avoid noisy fallback warnings)
        resolved_sam: Path | None = None
        if effective_bg_backend in ("sam", "auto"):
            if isinstance(sam_checkpoint, Path):
                resolved_sam = (
                    sam_checkpoint
                    if sam_checkpoint.is_file()
                    else resolve_sam_checkpoint(cli_value=str(sam_checkpoint), warn=False)
                )
            elif isinstance(sam_checkpoint, str) and sam_checkpoint:
                p = Path(sam_checkpoint).expanduser()
                resolved_sam = p.resolve() if p.is_file() else None
            else:
                resolved_sam = resolve_sam_checkpoint(warn=False)
            if sam_checkpoint and resolved_sam is None:
                raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")

        # Resolve YuNet model early (for logging) — terminology: YuNet uses 'model' (.onnx), SAM uses 'checkpoint' (.pth)
        resolved_yunet: Path | None = None
        if face_backend in ("yunet", "auto"):
            if isinstance(yunet_model, Path):
                resolved_yunet = (
                    yunet_model
                    if yunet_model.is_file()
                    else resolve_yunet_model(cli_value=str(yunet_model), warn=False)
                )
            elif isinstance(yunet_model, str) and yunet_model:
                p = Path(yunet_model).expanduser()
                resolved_yunet = p.resolve() if p.is_file() else None
            else:
                resolved_yunet = resolve_yunet_model(warn=False)
            if yunet_model and resolved_yunet is None:
                raise FileNotFoundError(f"YuNet model not found: {yunet_model}")
            if resolved_yunet:
                LOG.debug("Resolved YuNet model: %s", resolved_yunet)

        resolved_rembg_model = None
        resolved_ben2_model = None
        resolved_birefnet_model = None
        resolved_sam_type = sam_model_type
        if effective_bg_backend in ("rembg", "auto"):
            resolved_rembg_model = resolve_rembg_model(
                cli_value=rembg_model,
                warn=False,
            )
        if effective_bg_backend in ("ben2", "auto"):
            resolved_ben2_model = resolve_ben2_model(
                cli_value=str(ben2_model) if ben2_model else None,
                warn=False,
            )
        if effective_bg_backend in ("birefnet", "auto"):
            resolved_birefnet_model = resolve_birefnet_model(
                cli_value=str(birefnet_model) if birefnet_model else None,
                warn=False,
            )
        if effective_bg_backend == "ben2" and resolved_ben2_model is None:
            raise FileNotFoundError(f"BEN2 model not found: {ben2_model or 'local cache'}")
        if effective_bg_backend == "birefnet" and resolved_birefnet_model is None:
            raise FileNotFoundError(
                f"BiRefNet_dynamic snapshot not found or incomplete: "
                f"{birefnet_model or 'local cache'}"
            )
        if effective_bg_backend in ("sam", "auto"):
            resolved_sam_type = resolve_sam_model_type(
                cli_value=sam_model_type,
                checkpoint=resolved_sam,
            )

        import time as _time

        _t0_total = _time.perf_counter()
        _t_face0 = _time.perf_counter()
        face = detect_faces(
            img_cv,
            input_path,
            select_face=select_face,
            face_index=face_index,
            export_faces_dir=export_faces_dir,
            face_backend=face_backend,
            yunet_model=resolved_yunet,
            crop_aspect=width / height,
            overwrite_exports=overwrite,
            protected_export_paths=(input_path, out_p),
        )
        _t_face1 = _time.perf_counter()
        LOG.info(
            "[timing] face_detection backend=%s time=%.3fs",
            face_backend,
            _t_face1 - _t_face0,
        )
        img_cv_cropped = crop_image(img_cv, face, output_aspect=width / height)
        if effective_bg_backend != "none":
            _t_bg0 = _time.perf_counter()
            img_cv_cropped = remove_background(
                img_cv_cropped,
                background_backend=effective_bg_backend,
                sam_checkpoint=resolved_sam,
                rembg_model=resolved_rembg_model
                if isinstance(resolved_rembg_model, str)
                else rembg_model,
                ben2_model=resolved_ben2_model,
                birefnet_model=resolved_birefnet_model,
                resegment=resegment,
                sam_model_type=resolved_sam_type,
            )
            _t_bg1 = _time.perf_counter()
            LOG.info(
                "[timing] background_removal backend=%s time=%.3fs",
                effective_bg_backend,
                _t_bg1 - _t_bg0,
            )
        else:
            LOG.info("Background removal skipped.")

        # Resize one photo to its exact print dimensions, then center as many whole
        # copies as fit on an exact-size white paper canvas. Never scale the sheet:
        # doing so would change the physical dimensions of every passport photo.
        if not keep_untiled:
            single_w = int(round(width * dpi))
            single_h = int(round(height * dpi))
            target_w = int(round(tiled_width * dpi))
            target_h = int(round(tiled_height * dpi))
            tiles_x = target_w // single_w
            tiles_y = target_h // single_h
            if tiles_x < 1 or tiles_y < 1:
                raise ValueError(
                    "Configured paper is too small for one photo: "
                    f"photo={width}x{height}in paper={tiled_width}x{tiled_height}in"
                )
            img_single = resize_image(img_cv_cropped, single_w, single_h)
            img_cv_resized = np.full(
                (target_h, target_w, img_single.shape[2]),
                255,
                dtype=img_single.dtype,
            )
            grid_w = tiles_x * single_w
            grid_h = tiles_y * single_h
            offset_x = (target_w - grid_w) // 2
            offset_y = (target_h - grid_h) // 2
            for row in range(tiles_y):
                top = offset_y + row * single_h
                for column in range(tiles_x):
                    left = offset_x + column * single_w
                    img_cv_resized[top : top + single_h, left : left + single_w] = img_single
            LOG.info(
                "Image tiled %d times horizontally and %d times vertically",
                tiles_x,
                tiles_y,
            )
        else:
            img_cv_resized = resize_image(
                img_cv_cropped,
                int(round(width * dpi)),
                int(round(height * dpi)),
            )

        _save_output_image(img_cv_resized, out_p, dpi=dpi, quality=jpeg_quality)
        _t_total = _time.perf_counter() - _t0_total
        LOG.info(
            "[timing] total pipeline time=%.3fs face_backend=%s bg_backend=%s",
            _t_total,
            face_backend,
            effective_bg_backend,
        )
        print(f"Processed image saved as {out_p}  dpi={dpi}")
        return face
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        raise


# --------------------------------------------------------------------------- #
# CLI — unified, backwards-compatible
# --------------------------------------------------------------------------- #
def _build_parser(file_config: dict) -> argparse.ArgumentParser:
    # Help must not probe the filesystem or expose configured local model paths.
    default_rembg = "birefnet-portrait"
    default_sam_type = "inferred"
    default_face_backend = resolve_backend(
        None,
        ["face_backend"],
        file_config,
        "auto",
        ["auto", "yunet", "haar", "deepface", "retinaface", "mtcnn"],
    )
    default_bg_backend = resolve_backend(
        None,
        ["background_backend", "background-backend"],
        file_config,
        "none",
        ["auto", "sam", "rembg", "ben2", "birefnet", "none"],
    )

    parser = argparse.ArgumentParser(
        description="Passport photo — detect largest/central face, crop to 2×2in, optionally tile to paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog="passport-photos",
    )
    # Input / output — keep positional for compat; also support batch via directory
    parser.add_argument(
        "input_path",
        nargs="?",
        type=str,
        help="Path to an input image or a directory of images for batch processing.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=str,
        help="Path to output image or directory (batch). For batch input (directory), output must be a directory — prefer --output-dir DIR for clarity. Ignored with --list-faces if no output needed.",
    )

    # General
    general = parser.add_argument_group("General")
    general.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to passport-photos.toml config file (default: ./passport-photos.toml, pyproject.toml [tool.passport-photos], env PASSPORT_PHOTOS_CONFIG)",
    )
    general.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    general.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-v info, -vv debug)"
    )
    general.add_argument(
        "-q", "--quiet", action="count", default=0, help="Decrease verbosity (quiet warnings)"
    )
    general.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode inputs and show planned outputs without detection, model execution, or writes",
    )
    general.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and face-preview files instead of failing or skipping",
    )

    # Face detection
    face_grp = parser.add_argument_group("Face detection")
    face_grp.add_argument(
        "--face-backend",
        type=str,
        default=None,
        choices=["auto", "yunet", "haar", "deepface", "retinaface", "mtcnn"],
        help=(
            f"Face detection backend (default from config: {default_face_backend}). "
            "deepface uses its OpenCV adapter; RetinaFace requires manually supplied weights; "
            "MTCNN weights are bundled with the deepface extra. No backend downloads models"
        ),
    )
    face_grp.add_argument(
        "--yunet-model",
        type=str,
        default=None,
        dest="yunet_model",
        metavar="PATH",
        help="Path to a manually supplied YuNet ONNX model; otherwise search local cache paths",
    )
    face_grp.add_argument(
        "--yunet-checkpoint", type=str, default=None, dest="yunet_model", help=argparse.SUPPRESS
    )  # deprecated alias for --yunet-model
    face_grp.add_argument(
        "--yunet-onnx", type=str, default=None, dest="yunet_model", help=argparse.SUPPRESS
    )  # another alias
    face_grp.add_argument(
        "--select-face",
        action="store_true",
        help="Interactively pick a ranked face (single-image TTY only; --face-index overrides)",
    )
    face_grp.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Non-interactive pick: use ranked face N (0=best). Overrides --select-face",
    )
    face_grp.add_argument(
        "--export-faces",
        type=str,
        default=None,
        metavar="DIR",
        help="Export ALL detected face crops (face_00.png best) + faces.json to DIR for manual picking",
    )
    face_grp.add_argument(
        "--list-faces",
        action="store_true",
        help="Only detect & list faces (sorted by confidence→size); do not write output image",
    )

    # Background removal
    bg_grp = parser.add_argument_group("Background removal")
    bg_grp.add_argument(
        "--background-backend",
        type=str,
        default=None,
        choices=["auto", "sam", "rembg", "ben2", "birefnet", "none"],
        help=(
            f"Background backend (default: {default_bg_backend}). auto follows the fixed "
            "quality order documented in README.md and preserves the background only if "
            f"every local model fails. Default explicit rembg model: {default_rembg}."
        ),
    )
    bg_grp.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a manually supplied SAM checkpoint; otherwise search local cache paths",
    )
    bg_grp.add_argument(
        "--sam-model-type",
        type=str,
        default=None,
        choices=["vit_b", "vit_l", "vit_h"],
        dest="sam_model_type",
        help=f"SAM architecture; must match the checkpoint (config default: {default_sam_type})",
    )
    bg_grp.add_argument(
        "--sam-model", type=str, default=None, dest="sam_model_type", help=argparse.SUPPRESS
    )  # alias
    bg_grp.add_argument(
        "--rembg-model",
        type=str,
        default=None,
        dest="rembg_model",
        metavar="NAME|PATH",
        help=f"Canonical local rembg model name or ONNX path (default: {default_rembg}).",
    )
    bg_grp.add_argument(
        "--ben2-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a local BEN2_Base.onnx; otherwise search local cache paths",
    )
    bg_grp.add_argument(
        "--birefnet-model",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing a complete local BiRefNet_dynamic Transformers snapshot; "
            "otherwise search local cache paths"
        ),
    )
    bg_grp.add_argument(
        "--keep-background",
        action="store_true",
        help="Skip background removal (shortcut for --background-backend none)",
    )
    bg_grp.add_argument(
        "--resegment",
        action="store_true",
        help="Force re-segmentation (ignore segmentation_*.npz cache)",
    )

    out_grp = parser.add_argument_group("Output / tiling")
    out_grp.add_argument(
        "--tiled",
        action="store_true",
        help="Create a centered print sheet; the default is one untiled photo",
    )
    out_grp.add_argument(
        "--keep-untiled",
        dest="tiled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    out_grp.add_argument(
        "--no-tiling",
        dest="tiled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    out_grp.add_argument(
        "--photo-width",
        "--width",
        type=float,
        default=None,
        dest="width",
        metavar="INCHES",
        help="Single-photo width in inches (config: photo_width)",
    )
    out_grp.add_argument(
        "--photo-height",
        "--height",
        type=float,
        default=None,
        dest="height",
        metavar="INCHES",
        help="Single-photo height in inches (config: photo_height)",
    )
    out_grp.add_argument(
        "--paper-width",
        "--tiled-width",
        type=float,
        default=None,
        dest="tiled_width",
        metavar="INCHES",
        help="Tiled paper width in inches (config: paper_width)",
    )
    out_grp.add_argument(
        "--paper-height",
        "--tiled-height",
        type=float,
        default=None,
        dest="tiled_height",
        metavar="INCHES",
        help="Tiled paper height in inches (config: paper_height)",
    )
    out_grp.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Output print resolution in dots per inch (config: dpi)",
    )
    out_grp.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["auto", "png", "jpeg", "jpg"],
        default=None,
        help="Output format; auto uses the output filename extension",
    )
    out_grp.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        metavar="1-100",
        help="JPEG quality (config: jpeg_quality)",
    )
    out_grp.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Batch output directory, or parent for a bare single-output filename",
    )
    parser.set_defaults(tiled=False)

    return parser


def _validate_file_config(file_config: dict) -> None:
    enum_values = {
        "face_backend": {"auto", "yunet", "haar", "deepface", "retinaface", "mtcnn"},
        "background_backend": {"auto", "sam", "rembg", "ben2", "birefnet", "none"},
        "background-backend": {"auto", "sam", "rembg", "ben2", "birefnet", "none"},
        "sam_model_type": {"vit_b", "vit_l", "vit_h"},
        "sam-model-type": {"vit_b", "vit_l", "vit_h"},
        "output_format": {"auto", "png", "jpeg", "jpg"},
        "format": {"auto", "png", "jpeg", "jpg"},
    }
    for key, allowed in enum_values.items():
        if key not in file_config:
            continue
        value = str(file_config[key]).lower()
        if value not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ValueError(
                f"Invalid config value for {key}: {file_config[key]!r}; expected {expected}"
            )


def _resolve_dimensions(args, file_config: dict):
    def _cfg_number(keys, default, converter):
        for key in keys:
            if key not in file_config:
                continue
            try:
                return converter(file_config[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid config value for {key}: {file_config[key]!r}; "
                    f"expected {converter.__name__}"
                ) from exc
        return default

    width = (
        args.width
        if args.width is not None
        else _cfg_number(["photo_width", "photo-width", "width"], 2.0, float)
    )
    height = (
        args.height
        if args.height is not None
        else _cfg_number(["photo_height", "photo-height", "height"], 2.0, float)
    )
    tiled_width = (
        args.tiled_width
        if args.tiled_width is not None
        else _cfg_number(["paper_width", "paper-width", "tiled_width", "tiled-width"], 10.0, float)
    )
    tiled_height = (
        args.tiled_height
        if args.tiled_height is not None
        else _cfg_number(
            ["paper_height", "paper-height", "tiled_height", "tiled-height"], 8.0, float
        )
    )
    dpi = args.dpi if args.dpi is not None else _cfg_number(["dpi"], 400, int)
    fmt = str(
        args.output_format
        or file_config.get("output_format")
        or file_config.get("format")
        or "auto"
    ).lower()
    quality = args.jpeg_quality
    if quality is None:
        quality = _cfg_number(["jpeg_quality", "quality"], 95, int)

    for name, value in [
        ("photo-width", width),
        ("photo-height", height),
        ("paper-width", tiled_width),
        ("paper-height", tiled_height),
    ]:
        if not math.isfinite(value) or value <= 0 or value > 100:
            raise ValueError(f"--{name} must be finite, >0, and <=100 inches (got {value})")
    if dpi <= 0 or dpi > 2400:
        raise ValueError(f"--dpi must be 1..2400 (got {dpi})")
    if not 1 <= quality <= 100:
        raise ValueError(f"--jpeg-quality must be 1..100 (got {quality})")
    if fmt not in {"auto", "png", "jpeg", "jpg"}:
        raise ValueError(f"output format must be auto, png, or jpeg (got {fmt!r})")
    photo_width_px = int(round(width * dpi))
    photo_height_px = int(round(height * dpi))
    paper_width_px = int(round(tiled_width * dpi))
    paper_height_px = int(round(tiled_height * dpi))
    pixel_dimensions = [
        ("photo-width", photo_width_px),
        ("photo-height", photo_height_px),
    ]
    if args.tiled:
        pixel_dimensions.extend(
            [
                ("paper-width", paper_width_px),
                ("paper-height", paper_height_px),
            ]
        )
    for name, value in pixel_dimensions:
        if value < 1:
            raise ValueError(f"--{name} rounds to zero pixels at {dpi} DPI")
    output_width = paper_width_px if args.tiled else photo_width_px
    output_height = paper_height_px if args.tiled else photo_height_px
    output_pixels = output_width * output_height
    if output_pixels > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"output would contain {output_pixels:,} pixels; maximum is "
            f"{MAX_OUTPUT_PIXELS:,}. Reduce dimensions or DPI"
        )
    return width, height, tiled_width, tiled_height, dpi, fmt, quality


def main_cli(argv: list[str] | None = None):
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if any(flag in effective_argv for flag in ("-h", "--help", "--version")):
        # Help/version should always work, even when a discovered config is invalid.
        _build_parser({}).parse_args(effective_argv)
        return

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(effective_argv)
    try:
        file_config = _collect_file_config(cli_config=config_args.config)
        _validate_file_config(file_config)
        parser = _build_parser(file_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"passport-photos: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    args = parser.parse_args(effective_argv)

    _setup_logging(args.verbose, args.quiet)
    LOG.debug("Loaded config: %s", file_config)

    # Validate required positionals unless --list-faces or --version/help already handled
    # If no input provided and not in help/version path, show error
    if not args.input_path:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: input_path is required", file=sys.stderr)
        raise SystemExit(2)
    # output_path may be omitted for --list-faces --export-faces only, or batch with --output-dir
    has_batch_output_dir = (
        bool(getattr(args, "output_dir", None))
        and args.input_path
        and Path(args.input_path).is_dir()
    )
    if not args.output_path and not args.list_faces and not has_batch_output_dir:
        parser.print_usage(sys.stderr)
        print(
            f"{parser.prog}: error: output_path is required (or use --output-dir DIR for batch, or --list-faces)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # For batch with --output-dir only, synthesize output_path as the dir so later checks pass (actual dir resolved later)
    if not args.output_path and has_batch_output_dir:
        args.output_path = args.output_dir

    face_backend = resolve_backend(
        args.face_backend,
        ["face_backend"],
        file_config,
        "auto",
        ["auto", "yunet", "haar", "deepface", "retinaface", "mtcnn"],
    )
    bg_backend = resolve_backend(
        args.background_backend,
        ["background_backend", "background-backend"],
        file_config,
        "none",
        ["auto", "sam", "rembg", "ben2", "birefnet", "none"],
    )
    if args.keep_background:
        bg_backend = "none"

    needs_background = not args.list_faces and bg_backend != "none"

    sam_ckpt = None
    sam_model_type = getattr(args, "sam_model_type", None)
    if needs_background and bg_backend in ("sam", "auto"):
        sam_ckpt = resolve_sam_checkpoint(
            cli_value=args.sam_checkpoint,
            config=file_config,
        )
        if args.sam_checkpoint and sam_ckpt is None:
            parser.error(f"SAM checkpoint not found: {args.sam_checkpoint}")
        sam_model_type = resolve_sam_model_type(
            cli_value=sam_model_type,
            config=file_config,
            checkpoint=sam_ckpt,
        )
        LOG.debug("SAM resolved: checkpoint=%s model_type=%s", sam_ckpt, sam_model_type)

    yunet_cli = getattr(args, "yunet_model", None)
    yunet_ckpt = None
    if face_backend in ("yunet", "auto"):
        yunet_ckpt = resolve_yunet_model(
            cli_value=yunet_cli,
            config=file_config,
            warn=False,
        )
        if yunet_cli and yunet_ckpt is None:
            parser.error(f"YuNet model not found: {yunet_cli}")
        if yunet_ckpt:
            LOG.info("YuNet model resolved: %s", yunet_ckpt)

    rembg_model_name = None
    if needs_background and bg_backend in ("rembg", "auto"):
        try:
            rembg_model_name = resolve_rembg_model(
                cli_value=getattr(args, "rembg_model", None),
                config=file_config,
                warn=False,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        LOG.debug("rembg model resolved: %s", rembg_model_name)

    ben2_cli = getattr(args, "ben2_model", None)
    ben2_model_path = None
    if needs_background and bg_backend in ("ben2", "auto"):
        ben2_model_path = resolve_ben2_model(
            cli_value=ben2_cli,
            config=file_config,
            warn=False,
        )
        if ben2_cli and ben2_model_path is None:
            parser.error(f"BEN2 model not found: {ben2_cli}")

    birefnet_cli = getattr(args, "birefnet_model", None)
    birefnet_model_path = None
    if needs_background and bg_backend in ("birefnet", "auto"):
        birefnet_model_path = resolve_birefnet_model(
            cli_value=birefnet_cli,
            config=file_config,
            warn=False,
        )
        if birefnet_cli and birefnet_model_path is None:
            parser.error(f"BiRefNet_dynamic snapshot missing or incomplete: {birefnet_cli}")

    if needs_background and bg_backend == "sam" and sam_ckpt is None:
        parser.error("--background-backend sam requires --sam-checkpoint or config sam_checkpoint")
    if needs_background and bg_backend == "ben2" and ben2_model_path is None:
        parser.error("--background-backend ben2 requires --ben2-model or config ben2_model")
    if needs_background and bg_backend == "birefnet" and birefnet_model_path is None:
        parser.error(
            "--background-backend birefnet requires --birefnet-model or config birefnet_model"
        )

    try:
        width, height, tiled_width, tiled_height, dpi, fmt, quality = _resolve_dimensions(
            args, file_config
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Guard: --select-face requires TTY; --face-index is the CLI way to pick
    if args.select_face and args.face_index is None and not sys.stdin.isatty():
        print(
            "Error: --select-face requires an interactive terminal (TTY). Use --face-index N (e.g. --face-index 0) for non-interactive selection, or --export-faces for batch review.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.select_face and args.face_index is not None:
        LOG.info(
            "--face-index %s overrides --select-face, ignoring interactive prompt.", args.face_index
        )

    if args.list_faces and args.dry_run:
        parser.error("--dry-run cannot be combined with --list-faces; listing faces runs detection")

    # --list-faces early path (supports directory batch) — with timing
    if args.list_faces:
        import time as _time

        inp = Path(args.input_path)
        targets: list[Path]
        if inp.is_dir():
            targets = [
                p
                for p in inp.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
            ]
            if not targets:
                print(f"No images found in directory {inp}", file=sys.stderr)
                raise SystemExit(1)
        else:
            targets = [inp]
        exit_code = 0
        last_faces: list[dict] = []
        for t in targets:
            img = cv2.imread(str(t), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Could not read {t}", file=sys.stderr)
                exit_code = 1
                continue
            _t0 = _time.perf_counter()
            faces = detect_faces_all(img, str(t), method=face_backend, yunet_model=yunet_ckpt)
            _dt = _time.perf_counter() - _t0
            LOG.info(
                "[timing] face_detection backend=%s file=%s time=%.3fs faces=%d",
                face_backend,
                t.name,
                _dt,
                len(faces),
            )
            last_faces = faces
            export_dir: Path | None = None
            if args.export_faces:
                export_dir = _preview_dir_for_source(
                    args.export_faces,
                    t,
                    batch=len(targets) > 1,
                )
                export_all_faces(
                    img,
                    faces,
                    export_dir,
                    crop_aspect=width / height,
                    overwrite=args.overwrite,
                    protected_paths=(t,),
                )
            if not faces:
                print(f"{t}: No faces found.")
                exit_code = 1
                continue
            print(
                f"\n{t}: Found {len(faces)} candidate face(s) (backend={face_backend}, sorted by decreasing confidence → size):"
            )
            for i, f in enumerate(faces):
                fa = f["facial_area"]
                print(
                    f"  [{i}] conf={f['confidence']:.3f}  bbox=({fa['x']},{fa['y']},{fa['w']},{fa['h']})  area={fa['w'] * fa['h']}"
                )
            if export_dir is not None:
                print(f"  Previews in {export_dir.resolve()}  (face_00.png is best-ranked)")
        if len(targets) == 1:
            print(
                f"\nListed {len(last_faces)} face(s). Use --face-index N or --select-face to pick one."
            )
        raise SystemExit(exit_code)

    # Batch mode: if input is directory, process each image to output directory (explicit)
    inp = Path(args.input_path)
    # Resolve output for batch vs single — --output-dir explicit when input is directory
    explicit_output_dir = getattr(args, "output_dir", None)
    if inp.is_dir():
        if explicit_output_dir:
            outp = Path(explicit_output_dir)
        elif args.output_path:
            outp = Path(args.output_path)
            if outp.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
                print(
                    f"Error: input is a directory but output {outp} looks like a file (has image extension). For batch, provide a directory — use --output-dir DIR or a directory path without extension.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
        else:
            print(
                "Error: batch mode (input is directory) requires an output directory. Provide positional output_path as directory or --output-dir DIR.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if outp.exists() and not outp.is_dir():
            print(
                f"Error: input is directory but output {outp} is not a directory.", file=sys.stderr
            )
            raise SystemExit(2)
        if args.select_face and args.face_index is None:
            print(
                "Error: --select-face is not supported in batch mode (input is directory). Use --face-index N or --export-faces for per-image review.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        images = sorted(
            p
            for p in inp.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        )
        if not images:
            print(f"No images found in {inp}", file=sys.stderr)
            raise SystemExit(1)

        input_dir = inp.resolve()
        output_dir = outp.resolve()
        if output_dir == input_dir or output_dir.is_relative_to(input_dir):
            parser.error("batch output directory must be outside the input directory")

        source_paths = {_path_collision_key(source) for source in images}
        planned_outputs: dict[str, Path] = {}
        plan: list[tuple[Path, Path]] = []
        for src in images:
            if fmt and fmt != "auto":
                fmt_norm = fmt.lower().lstrip(".")
                ext = ".jpg" if fmt_norm in ("jpeg", "jpg") else f".{fmt_norm}"
            else:
                ext = src.suffix
            dst = outp / f"{src.stem}{ext}"
            destination_key = _path_collision_key(dst)
            if destination_key in source_paths:
                parser.error(f"batch output would overwrite an input image: {dst}")
            if destination_key in planned_outputs:
                parser.error(
                    f"batch inputs {planned_outputs[destination_key].name} and {src.name} "
                    f"map to the same output {dst.name}; rename one input or use auto format"
                )
            planned_outputs[destination_key] = src
            plan.append((src, dst))

        if args.export_faces:
            preview_root = Path(args.export_faces).resolve()
            if preview_root == input_dir or preview_root.is_relative_to(input_dir):
                parser.error("batch face-preview directory must be outside the input directory")
            for _, destination in plan:
                if _path_is_at_or_below(preview_root, destination):
                    parser.error(
                        f"batch face-preview directory overlaps planned output file: {destination}"
                    )
            preview_keys: dict[str, Path] = {}
            for src in images:
                preview_dir = _preview_dir_for_source(args.export_faces, src, batch=True)
                preview_key = _path_collision_key(preview_dir)
                for _, destination in plan:
                    if _path_is_at_or_below(destination, preview_dir):
                        parser.error(
                            f"batch output {destination} overlaps face-preview directory {preview_dir}"
                        )
                if preview_key in preview_keys:
                    parser.error(
                        f"batch inputs {preview_keys[preview_key].name} and {src.name} "
                        "map to the same face-preview directory"
                    )
                preview_keys[preview_key] = src

        if not args.dry_run:
            outp.mkdir(parents=True, exist_ok=True)
        processed = 0
        planned = 0
        skipped = 0
        failed = 0
        for src, dst in plan:
            if dst.exists() and not args.overwrite and not args.dry_run:
                print(f"Skipping {src.name} → {dst.name} exists (use --overwrite)", file=sys.stderr)
                skipped += 1
                continue
            try:
                process_image(
                    str(src),
                    str(dst),
                    width=width,
                    height=height,
                    tiled_width=tiled_width,
                    tiled_height=tiled_height,
                    dpi=dpi,
                    select_face=args.select_face,
                    face_index=args.face_index,
                    export_faces_dir=_preview_dir_for_source(args.export_faces, src, batch=True)
                    if args.export_faces
                    else None,
                    face_backend=face_backend,
                    background_backend=bg_backend,
                    sam_checkpoint=sam_ckpt,
                    sam_model_type=sam_model_type,
                    yunet_model=yunet_ckpt,
                    rembg_model=rembg_model_name,
                    ben2_model=ben2_model_path,
                    birefnet_model=birefnet_model_path,
                    keep_untiled=not args.tiled,
                    keep_background=(bg_backend == "none"),
                    resegment=args.resegment,
                    output_format=fmt,
                    jpeg_quality=quality,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    planned += 1
                else:
                    processed += 1
            except SystemExit:
                raise
            except Exception as e:
                print(f"Failed {src}: {e}", file=sys.stderr)
                LOG.debug("Traceback", exc_info=True)
                failed += 1
        if failed:
            print(f"Batch completed with {failed} failure(s).", file=sys.stderr)
            raise SystemExit(1)
        if args.dry_run:
            print(f"Batch dry run: {planned} planned, {failed} failed in {outp.resolve()}")
        else:
            print(
                f"Batch done: {processed} processed, {skipped} skipped, {failed} failed "
                f"in {outp.resolve()}"
            )
        return

    # Single file mode — resolve output (supports --output-dir)
    explicit_output_dir = getattr(args, "output_dir", None)
    if explicit_output_dir and not args.output_path:
        print(
            "Error: --output-dir provided but no output filename given. Provide output_path filename (e.g. --output-dir ./out output.png or use positional output).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if explicit_output_dir and args.output_path:
        cand = Path(args.output_path)
        # If output_path is bare filename or relative, place inside --output-dir
        if cand.parent == Path(".") or str(cand.parent) == "":
            outp = Path(explicit_output_dir) / cand
        else:
            # An output path with its own directory takes precedence.
            outp = cand
    else:
        if not args.output_path:
            parser.print_usage(sys.stderr)
            print(f"{parser.prog}: error: output_path is required", file=sys.stderr)
            raise SystemExit(2)
        outp = Path(args.output_path)
    outp = _output_path_for_format(outp, fmt)
    if _path_collision_key(outp) == _path_collision_key(args.input_path):
        parser.error("output path must not overwrite the input image")
    # Guard overwrite
    if outp.exists() and not args.overwrite and not args.dry_run:
        print(
            f"Error: output {outp} exists (use --overwrite to overwrite, --dry-run to preview).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    process_image(
        args.input_path,
        str(outp),
        width=width,
        height=height,
        tiled_width=tiled_width,
        tiled_height=tiled_height,
        dpi=dpi,
        select_face=args.select_face,
        face_index=args.face_index,
        export_faces_dir=args.export_faces,
        face_backend=face_backend,
        background_backend=bg_backend,
        sam_checkpoint=sam_ckpt,
        sam_model_type=sam_model_type,
        yunet_model=yunet_ckpt,
        rembg_model=rembg_model_name,
        ben2_model=ben2_model_path,
        birefnet_model=birefnet_model_path,
        keep_untiled=not args.tiled,
        keep_background=(bg_backend == "none"),
        resegment=args.resegment,
        output_format=fmt,
        jpeg_quality=quality,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main_cli()
