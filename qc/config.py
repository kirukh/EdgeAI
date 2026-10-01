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
    # Focus (only cameras with autofocus, e.g. Camera Module 3):
    # None = focus ONCE at start-up on what is under the camera, then keep the lens fixed.
    # A number = fixed lens position in dioptres (1 / distance in m, e.g. 4.0 = 25 cm).
    # Continuous autofocus is never used – it would refocus between parts.
    lens_position: float | None = None
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
    # What the fine alignment uses as datum: "outline" (outer contour, like datum edges on a
    # drawing – displaced holes cannot hide their offset), "all" (outline + holes) or
    # "auto" (outline, except for round parts, whose outline carries no rotation information)
    ecc_datum: str = "auto"
    # Rotation search: "full" (0–360°, any shape/orientation), "flip" (0°/180° only – parts
    # guided mechanically, faster) or "off" (parts always arrive in the same orientation)
    rotation_search: str = "full"
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
    # Absolute tolerances from the drawing (used instead of the learned ones once the
    # camera is calibrated in mm). None = use the learned tolerances.
    pos_tol_mm: float | None = None    # max. hole position deviation
    dia_tol_mm: float | None = None    # max. hole diameter deviation (±)

    # --- Decision: how the methods are combined ---
    # "fusion": a method decides on its own only above its "standalone" level; below that
    #           (score 1.0 … standalone) at least two methods must agree.
    # "or":     any method with score ≥ 1 → NOK (most sensitive, most false alarms)
    decision: str = "fusion"
    geometry_standalone: float = 1.0   # geometry measurements are reliable → decide alone
    diff_standalone: float = 1.0
    pca_standalone: float = 1.5        # ML alone only if clearly above its threshold

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
class LightingConfig:
    # "single": one lighting (as before). "dual": for every part one image with backlight
    # (holes/outline, exact silhouette) and one with front light (surface, blind holes).
    mode: str = "single"
    idle_light: str = "front"          # light that is on between parts (used by the trigger)
    settle_frames: int = 2             # frames to skip after switching the light (camera pipeline delay)
    back_methods: list[str] = field(default_factory=lambda: ["geometry"])
    front_methods: list[str] | None = None   # None = the methods chosen in the UI / method.methods


@dataclass
class InspectionConfig:
    # Several images of the same part while it passes the camera; majority vote.
    # 1 = off. 3 recommended against random false alarms (dust, noise, reflections).
    shots_per_part: int = 1
    tie_is_nok: bool = True            # with an even number of shots: tie → NOK (safe side)


@dataclass
class SelfLearningConfig:
    # Collect clearly good parts during inspection and train them in automatically in the
    # background (with safety checks, see recipe.learn_collected). The supervisor does nothing.
    enabled: bool = True
    auto: bool = True                  # False = only via the button in the setup area
    auto_every: int = 30               # train in automatically once this many good parts were collected
    margin: float = 0.5                # collect only if EVERY method score < margin (half of its threshold)
    max_pool: int = 100                # collected parts per model (reservoir sampling over the whole run)
    max_references: int = 80           # anchor + learned references after learning (more hardly helps)
    growth_limit: float = 0.3          # reject if a statistical threshold loosens by more than 30 %


@dataclass
class IOConfig:
    enabled: bool = False              # True = use the Raspberry Pi GPIO pins below
    active_high: bool = True
    reject_pin: int | None = None      # BCM pin for the ejector (via driver/relay!)
    reject_delay_ms: int = 500         # travel time camera → ejector
    reject_pulse_ms: int = 150
    ok_lamp_pin: int | None = None
    nok_lamp_pin: int | None = None
    lamp_ms: int = 800
    front_light_pin: int | None = None
    back_light_pin: int | None = None


@dataclass
class DriftConfig:
    window: int = 20                   # parts in the rolling window
    contrast_tol: float = 0.15         # ±15 % contrast part ↔ belt (lighting ageing, exposure)
    background_tol: float = 20.0       # grey levels (ambient light, dirty belt/backlight)
    area_tol: float = 0.03             # ±3 % part area (camera distance/zoom changed)
    sharpness_tol: float = 0.25        # −25 % edge steepness (focus, motion blur)


@dataclass
class CalibrationConfig:
    file: str = "data/calibration.json"
    board_cols: int = 9                # inner corners of the checkerboard
    board_rows: int = 6
    square_mm: float = 10.0


@dataclass
class UIConfig:
    # PIN for the setup area (methods, sensitivity, calibration, lighting, GPIO, models).
    # The supervisor screen needs no PIN. "" = setup area without PIN. CHANGE THE DEFAULT.
    setup_pin: str = "1234"
    setup_timeout_min: int = 30        # setup area locks itself after this idle time


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    inspection: InspectionConfig = field(default_factory=InspectionConfig)
    self_learning: SelfLearningConfig = field(default_factory=SelfLearningConfig)
    io: IOConfig = field(default_factory=IOConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
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
