"""Maps relative Depth Anything output to approximate metres using known distances."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_EPS = 1e-6


@dataclass
class CalibratedDistance:
    distance_m: float
    extrapolated: bool


class DistanceCalibrator:
    """Depth Anything's relative output is affine-invariant *disparity* (larger = nearer), so
    the mapping is modelled as  1/Z = alpha * d + beta   (>= 2 points, least squares)
    or                           Z = k / d              (1 point, assumes no offset).
    Z is the distance along the camera's optical axis. This is an approximation: the model's
    scale/shift can drift between frames and scenes, so recalibrate when the setup changes."""

    def __init__(self, path=None, initial_points=None):
        self.path = Path(path) if path else None
        self.points: List[Tuple[float, float]] = []
        self._model: Optional[str] = None
        self._k = self._alpha = self._beta = self._rms = 0.0
        self._lo = self._hi = 0.0
        if not self._load() and initial_points:
            self.points = [(float(d), float(z)) for d, z in initial_points]
        self._fit()

    @property
    def is_calibrated(self) -> bool:
        return self._model is not None

    def add_point(self, relative_depth: float, distance_m: float):
        self.points.append((float(relative_depth), float(distance_m)))
        self._fit()

    def clear(self):
        self.points = []
        self._fit()

    def save(self):
        if self.path:
            self.path.write_text(json.dumps({"points": self.points}, indent=2))

    def _load(self) -> bool:
        if not self.path or not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text())
            self.points = [(float(d), float(z)) for d, z in data.get("points", [])]
            return True
        except Exception as exc:
            print(f"[calibration] could not read {self.path}: {exc}")
            return False

    def _fit(self):
        self._model = None
        pts = [(d, z) for d, z in self.points if d > _EPS and z > _EPS]
        if not pts:
            return
        d = np.array([p[0] for p in pts])
        z = np.array([p[1] for p in pts])
        if len(pts) >= 2 and np.ptp(d) > 1e-6:
            alpha, beta = np.polyfit(d, 1.0 / z, 1)
            if alpha > 0 and np.all(alpha * d + beta > _EPS):
                self._model, self._alpha, self._beta = "inverse-affine", float(alpha), float(beta)
        if self._model is None:
            self._model, self._k = "inverse-scale", float(np.median(z * d))
        pred = np.array([self._predict(x) or math.nan for x in d])
        self._rms = float(np.sqrt(np.nanmean((pred - z) ** 2)))
        self._lo, self._hi = float(d.min()), float(d.max())

    def _predict(self, d: float) -> Optional[float]:
        if self._model == "inverse-affine":
            denom = self._alpha * d + self._beta
            return None if denom <= _EPS else 1.0 / denom
        if self._model == "inverse-scale":
            return None if d <= _EPS else self._k / d
        return None

    def distance(self, relative_depth: float) -> Optional[CalibratedDistance]:
        if not self.is_calibrated:
            return None
        z = self._predict(float(relative_depth))
        if z is None or not math.isfinite(z) or z <= 0:
            return None
        margin = 0.1 * (self._hi - self._lo) if self._hi > self._lo else 0.1 * abs(self._hi)
        extrap = relative_depth < self._lo - margin or relative_depth > self._hi + margin
        return CalibratedDistance(z, extrap)

    def describe(self) -> str:
        if not self.is_calibrated:
            return "No calibration points."
        pts = ", ".join(f"d={d:.3f}→{z:.2f} m" for d, z in self.points)
        return f"{len(self.points)} pt(s), {self._model}, RMS {self._rms * 100:.1f} cm\n{pts}"