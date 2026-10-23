"""Inspection methods.

Three methods that can be used individually or combined:

``geometry`` – classical image processing
    Holes/cut-outs are segmented as background pixels inside the part
    outline and matched against the learned nominal geometry (Hungarian
    matching). Detects: missing holes, holes not drilled through (edge ring
    without see-through), mispositioned, too large/small, wrongly shaped and
    extra holes, as well as outline deviations.

``diff`` – reference image comparison / difference map
    Per-pixel mean and spread image from the good parts; a new part is
    compared as a z-score map. Finds surface defects (scratches,
    discoloration) and anything else that deviates from the reference.

``pca`` – machine learning (unsupervised anomaly detection)
    PCA subspace of the good parts; the reconstruction error of a new part is
    the anomaly score. The threshold is calibrated via leave-one-out on the
    good parts – no defect examples are required.

All methods operate on the aligned part (``AlignedPart``) and return
``Defect`` objects with a location (canvas coordinates) for visualisation.
"""

from __future__ import annotations

from ..config import MethodConfig
from .base import DEFECT_LABELS, METHOD_LABELS, Defect, Method, MethodResult, robust_outliers
from .diff import DiffMethod
from .geometry import GeometryMethod, Hole, find_holes
from .pca import PCAMethod

__all__ = [
    "DEFECT_LABELS",
    "METHODS",
    "METHOD_LABELS",
    "Defect",
    "DiffMethod",
    "GeometryMethod",
    "Hole",
    "Method",
    "MethodResult",
    "PCAMethod",
    "create",
    "find_holes",
    "robust_outliers",
]

METHODS: dict[str, type[Method]] = {"geometry": GeometryMethod, "diff": DiffMethod, "pca": PCAMethod}


def create(name: str, cfg: MethodConfig) -> Method:
    """Inspection method by name (``geometry``, ``diff`` or ``pca``)."""
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {', '.join(METHODS)}")
    return METHODS[name](cfg)
