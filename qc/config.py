"""Central configuration.

All parameters live in dataclasses and can be overridden via a JSON file
(``config.json``). Fields that are not specified keep their default value.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    # "sim", "pi", "usb:0", "folder:/path/to/images"
    source: str = "sim"
    width: int = 1280
    height: int = 960
    fps: int = 20
    # A fixed exposure is essential for reproducible images (see "controlled lighting").
    # 0 = automatic (only useful for first tests).
    exposure_us: int = 4000
    analogue_gain: float = 1.0
    lock_white_balance: bool = True
    colour_gains: tuple[float, float] = (1.6, 1.6)
    # Simulator
    sim_lighting: str = "front"        # "front" (front light) or "back" (backlight)
    sim_part_type: str = "A"           # "A" 2 holes, "B" hole + slot, "C" triangle, "D" square
    sim_any_angle: bool = False        # parts arrive in any rotation (0–360°)
    sim_defect_rate: float = 0.3
    sim_speed_px: float = 14.0         # conveyor speed per frame


@dataclass
class LocalizationConfig:
    # "auto" detects whether the part is brighter or darker than the background
    polarity: str = "auto"             # "auto" | "bright" | "dark"
    min_part_area_ratio: float = 0.02  # minimum part area relative to the image
    border_margin_px: int = 4          # part must not touch the image border
    canvas_margin_ratio: float = 0.12  # margin around the aligned part
    use_ecc_refine: bool = True        # sub-pixel fine alignment via ECC
    # Accept parts lying upside down (mirror image)? Sensible for flat parts with
    # through holes; keep False if the mirrored part counts as a different part.
    allow_mirror: bool = False
    work_width: int = 640              # internal working resolution (width)


@dataclass
class TriggerConfig:
    # Capture as soon as the part is fully visible and inside the centre band
    center_band_ratio: float = 0.12
    # Conveyor direction: "x" (left↔right) or "y" (top↔bottom)
    axis: str = "x"
    detect_width: int = 320            # resolution for the fast part detection
    rearm_frames: int = 3              # frames without a centred part before re-arming


@dataclass
class MethodConfig:
    # Enabled inspection methods (combined with OR → one method NOK = part NOK)
    methods: list[str] = field(default_factory=lambda: ["geometry", "diff", "pca"])
    # Image representation per method (see qc/representations.py)
    diff_representation: str = "norm"
    pca_representation: str = "edges"

    # --- Geometry: holes + outline ---
    hole_min_area_px: int = 40
    pos_tol_px: float = 6.0            # minimum position tolerance
    dia_tol_ratio: float = 0.10        # minimum diameter tolerance (relative)
    tol_sigma: float = 4.0             # tolerance = max(minimum tolerance, k·σ from learning)
    rim_edge_threshold: float = 0.25   # edge ratio on the hole rim → "not drilled through"
    shape_tol_ratio: float = 0.25      # aspect-ratio tolerance (slot/cut-out)
    outline_min_area_px: int = 80      # minimum area of an outline deviation

    # --- Reference image comparison (difference map) ---
    diff_z_threshold: float = 6.0
    diff_sigma_floor: float = 0.04
    diff_blur: int = 5
    diff_min_area_px: int = 30

    # --- PCA anomaly detection (ML) ---
    pca_size: int = 96                 # width of the downscaled image
    pca_variance: float = 0.95
    pca_margin: float = 1.5            # threshold = margin · max(leave-one-out score)
    pca_residual_threshold: float = 0.25


@dataclass
class StorageConfig:
    models_dir: str = "data/models"
    results_dir: str = "data/results"
    archive_dir: str = "data/archive"      # ZIP archives of finished batches
    save_ok_images: bool = False
    save_nok_images: bool = True
    history_size: int = 50


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    target_reference_count: int = 15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return _merge(cls(), data)

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None or not Path(path).exists():
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge(obj: Any, data: dict[str, Any]) -> Any:
    """Recursively overwrite the fields of a dataclass with values from ``data``."""
    for f in fields(obj):
        if f.name not in data:
            continue
        current = getattr(obj, f.name)
        value = data[f.name]
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, f.name, tuple(value))
        else:
            setattr(obj, f.name, value)
    return obj


def method_config_from_dict(data: dict[str, Any]) -> MethodConfig:
    return _merge(MethodConfig(), data)
