"""Pinhole-camera geometry, flame size, and (replaceable) volume estimation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Tuple


class DistanceMode(Enum):
    METRIC = "metric"
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"

    @property
    def in_metres(self) -> bool:
        return self is not DistanceMode.UNCALIBRATED

    @property
    def unit(self) -> str:
        return "m" if self.in_metres else "u"      # "u" = arbitrary uncalibrated units


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    approximate: bool
    source: str

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Rescale to another capture resolution (assumes the same field of view, i.e. no crop)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                                width, height, self.approximate, self.source + " (rescaled)")


def load_intrinsics(cfg, frame_w: int, frame_h: int) -> CameraIntrinsics:
    cw, ch = cfg.CAMERA_CALIB_RESOLUTION
    if cfg.CAMERA_INTRINSICS_FILE:
        data = json.loads(Path(cfg.CAMERA_INTRINSICS_FILE).read_text())
        base = CameraIntrinsics(float(data["fx"]), float(data["fy"]), float(data["cx"]),
                                float(data["cy"]), int(data.get("width", cw)),
                                int(data.get("height", ch)), False,
                                f"file {cfg.CAMERA_INTRINSICS_FILE}")
    elif cfg.CAMERA_FX is not None:
        fx = float(cfg.CAMERA_FX)
        fy = float(cfg.CAMERA_FY) if cfg.CAMERA_FY is not None else fx
        cx = float(cfg.CAMERA_CX) if cfg.CAMERA_CX is not None else cw / 2.0
        cy = float(cfg.CAMERA_CY) if cfg.CAMERA_CY is not None else ch / 2.0
        base = CameraIntrinsics(fx, fy, cx, cy, cw, ch, False, "config.py")
    else:
        fx = (frame_w / 2.0) / math.tan(math.radians(cfg.CAMERA_HFOV_DEG) / 2.0)
        base = CameraIntrinsics(fx, fx, frame_w / 2.0, frame_h / 2.0, frame_w, frame_h, True,
                                f"approximate, assumed {cfg.CAMERA_HFOV_DEG:.0f}° horizontal FOV")
    return base.scaled_to(frame_w, frame_h)


@dataclass(frozen=True)
class VolumeEstimate:
    volume: float
    semi_axes: Tuple[float, float, float]   # (a: half width, b: half height, c: half depth)
    method: str


class VolumeEstimator(Protocol):
    """Swap this for a segmentation / point-cloud based estimator later."""
    def estimate(self, width: float, height: float) -> VolumeEstimate: ...


class EllipsoidVolumeEstimator:
    """APPROXIMATE GEOMETRIC VOLUME of an ellipsoid fitted to the apparent bounding box:
    V = 4/3·π·a·b·c with a = W/2, b = H/2, c = a·depth_ratio. The depth extent c is an
    assumption, not a measurement. This is not the true combustion volume."""
    method = "ellipsoid, c = a * depth_ratio"

    def __init__(self, depth_ratio: float = 1.0):
        self.depth_ratio = float(depth_ratio)

    def estimate(self, width: float, height: float) -> VolumeEstimate:
        a, b = width / 2.0, height / 2.0
        c = a * self.depth_ratio
        return VolumeEstimate((4.0 / 3.0) * math.pi * a * b * c, (a, b, c), self.method)


@dataclass(frozen=True)
class FlameGeometry:
    u: float
    v: float
    X: float            # horizontal displacement (+ right)
    Y: float            # vertical displacement, image convention (+ DOWN)
    Z: float            # distance along the optical axis
    range_: float       # straight-line camera-to-flame distance
    width: float
    height: float
    volume: VolumeEstimate


def compute_flame_geometry(bbox_xyxy, Z: float, intr: CameraIntrinsics,
                           estimator: VolumeEstimator) -> FlameGeometry:
    """Assumes the whole flame lies at a single depth Z and that its visible bounding box
    represents its apparent size."""
    x1, y1, x2, y2 = bbox_xyxy
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    X = (u - intr.cx) * Z / intr.fx
    Y = (v - intr.cy) * Z / intr.fy
    W = (x2 - x1) * Z / intr.fx
    H = (y2 - y1) * Z / intr.fy
    return FlameGeometry(u, v, X, Y, Z, math.sqrt(X * X + Y * Y + Z * Z), W, H,
                         estimator.estimate(W, H))