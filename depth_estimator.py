"""Depth Anything V2 wrapper, depth extraction, trigger policy, background worker."""
from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch

from fire_detector import Detection

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _auto_device() -> str:
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")


def _import_depth_anything_v2(repo_dir, metric: bool):
    """Import DepthAnythingV2 from the cloned repo (or from an installed package)."""
    if repo_dir:
        base = Path(repo_dir).expanduser().resolve()
        search = base / "metric_depth" if metric else base
        if search.is_dir() and str(search) not in sys.path:
            sys.path.insert(0, str(search))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import 'depth_anything_v2'. Clone the official repo into the project "
            "folder (git clone https://github.com/DepthAnything/Depth-Anything-V2) and set "
            "DEPTH_REPO_DIR in config.py.") from exc
    return DepthAnythingV2


class DepthAnythingPredictor:
    """Your existing predictor: same model construction, checkpoint loading, infer_image, colorize."""

    def __init__(self, encoder="vits", device=None, checkpoint=None, metric=False,
                 max_depth=20.0, repo_dir=None, input_size=518):
        # NOTE: the official DepthAnythingV2.infer_image() moves its input to the
        # auto-detected device (CUDA -> MPS -> CPU) internally, so the model must live
        # there too. To force CPU, hide the GPU instead: CUDA_VISIBLE_DEVICES="".
        auto = _auto_device()
        requested = device if device and device != "auto" else auto
        if requested != auto:
            print(f"[depth] WARNING: requested device '{requested}' differs from the device "
                  f"Depth Anything's infer_image() uses ('{auto}'); using '{auto}'.")
        self.device = auto
        self.metric = bool(metric)
        self.input_size = int(input_size)

        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Invalid encoder: {encoder}")

        DepthAnythingV2 = _import_depth_anything_v2(repo_dir, self.metric)
        ckpt = Path(checkpoint) if checkpoint else Path(f"depth_anything_v2_{encoder}.pth")
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint not found: {ckpt}\n"
                f"'{encoder}' needs the matching checkpoint (vits = Small, vitb = Base, vitl = Large).")

        # Load model
        if self.metric:
            self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder], max_depth=float(max_depth))
        else:
            self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        self.model = self.model.to(self.device).eval()

        # Default colormap
        self.cmap = matplotlib.colormaps["turbo"]

    def infer_image(self, image):
        """Run depth estimation on a single BGR image. Returns an HxW float32 map."""
        return self.model.infer_image(image, self.input_size)

    def colorize(self, depth):
        """Convert depth map to turbo colormap (BGR)."""
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def warmup(self, shape=(480, 640, 3)) -> None:
        self.infer_image(np.zeros(shape, dtype=np.uint8))


# --------------------------------------------------------------------------
# Depth extraction from the flame bounding box
# --------------------------------------------------------------------------
@dataclass
class RegionDepth:
    value: float
    n_pixels: int
    bbox_d: Tuple[int, int, int, int]   # bbox mapped into depth-map coordinates
    roi_d: Tuple[int, int, int, int]    # sampled ROI in depth-map coordinates


def extract_region_depth(depth_map: np.ndarray, bbox_xyxy, frame_shape, near_is_high: bool,
                         shrink: float = 0.6, keep_fraction: float = 0.6,
                         min_pixels: int = 25) -> Optional[RegionDepth]:
    """Robust depth of the flame region.

    Relative Depth Anything output is disparity-like (LARGER = NEARER), metric output is
    metres (SMALLER = NEARER). Background pixels inside the box are the *far* ones, so we
    keep only the nearest `keep_fraction` of valid ROI pixels and take their median.
    """
    fh, fw = frame_shape[:2]
    dh, dw = depth_map.shape[:2]
    sx, sy = dw / float(fw), dh / float(fh)          # handles depth maps of another resolution

    x1, y1, x2, y2 = bbox_xyxy
    bx1, by1 = int(round(x1 * sx)), int(round(y1 * sy))
    bx2, by2 = int(round(x2 * sx)), int(round(y2 * sy))
    bx1, bx2 = max(0, bx1), min(dw, max(bx1 + 1, bx2))
    by1, by2 = max(0, by1), min(dh, max(by1 + 1, by2))

    cxd, cyd = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    half_w = max(1.5, (bx2 - bx1) * shrink / 2.0)
    half_h = max(1.5, (by2 - by1) * shrink / 2.0)
    rx1, rx2 = max(0, int(math.floor(cxd - half_w))), min(dw, int(math.ceil(cxd + half_w)))
    ry1, ry2 = max(0, int(math.floor(cyd - half_h))), min(dh, int(math.ceil(cyd + half_h)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    vals = depth_map[ry1:ry2, rx1:rx2].astype(np.float64).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < min_pixels:
        return None

    if near_is_high:
        thr = np.quantile(vals, 1.0 - keep_fraction)
        vals = vals[vals >= thr]
    else:
        thr = np.quantile(vals, keep_fraction)
        vals = vals[vals <= thr]
    if vals.size < min_pixels:
        return None

    return RegionDepth(float(np.median(vals)), int(vals.size),
                       (bx1, by1, bx2, by2), (rx1, ry1, rx2, ry2))


# --------------------------------------------------------------------------
# Jobs / results
# --------------------------------------------------------------------------
@dataclass
class DepthJob:
    frame_id: int
    frame: np.ndarray            # the exact frame YOLO ran on
    detection: Detection         # the exact detection from that frame
    capture_time: float          # time.monotonic()


@dataclass
class DepthResult:
    frame_id: int
    capture_time: float
    detection: Detection
    frame_shape: Tuple[int, ...]
    relative_depth: Optional[float]   # model units (or metres if is_metric)
    n_pixels: int
    inference_ms: float
    is_metric: bool
    depth_preview: Optional[np.ndarray] = None
    error: Optional[str] = None
    message: Optional[str] = None


class DepthWorker:
    """Single background thread, one job at a time. submit() returns False while busy
    (the job is dropped, never queued), so requests cannot accumulate."""

    def __init__(self, predictor: DepthAnythingPredictor, on_result: Callable[[DepthResult], None],
                 shrink=0.6, keep_fraction=0.6, min_pixels=25, preview_width=260):
        self._predictor = predictor
        self._on_result = on_result
        self._kw = dict(shrink=shrink, keep_fraction=keep_fraction, min_pixels=min_pixels)
        self._preview_width = preview_width
        self._cv = threading.Condition()
        self._job: Optional[DepthJob] = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="DepthWorker", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def is_busy(self) -> bool:
        with self._cv:
            return self._busy

    def submit(self, job: DepthJob) -> bool:
        with self._cv:
            if self._busy or self._stop:
                return False
            self._job = job
            self._busy = True
            self._cv.notify()
            return True

    def _run(self):
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._job = self._job, None
            try:
                result = self._process(job)
                self._on_result(result)
            except Exception:
                traceback.print_exc()
            finally:
                with self._cv:
                    self._busy = False

    def _process(self, job: DepthJob) -> DepthResult:
        det = job.detection
        try:
            t0 = time.perf_counter()
            depth = self._predictor.infer_image(job.frame)
            ms = (time.perf_counter() - t0) * 1000.0
            region = extract_region_depth(depth, det.bbox, job.frame.shape,
                                          near_is_high=not self._predictor.metric, **self._kw)
            preview = self._make_preview(depth, region)
            return DepthResult(
                frame_id=job.frame_id, capture_time=job.capture_time, detection=det,
                frame_shape=job.frame.shape,
                relative_depth=None if region is None else region.value,
                n_pixels=0 if region is None else region.n_pixels,
                inference_ms=ms, is_metric=self._predictor.metric, depth_preview=preview,
                message=None if region is not None else "No valid depth pixels in flame region")
        except Exception as exc:
            traceback.print_exc()
            return DepthResult(job.frame_id, job.capture_time, det, job.frame.shape,
                               None, 0, 0.0, self._predictor.metric, error=str(exc))

    def _make_preview(self, depth, region: Optional[RegionDepth]):
        color = self._predictor.colorize(depth)
        if region is not None:
            bx1, by1, bx2, by2 = region.bbox_d
            rx1, ry1, rx2, ry2 = region.roi_d
            cv2.rectangle(color, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv2.rectangle(color, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        h, w = color.shape[:2]
        scale = self._preview_width / float(w)
        return cv2.resize(color, (self._preview_width, max(1, int(h * scale))))


# --------------------------------------------------------------------------
# Trigger policy
# --------------------------------------------------------------------------
class DepthTrigger:
    """Decides when a new depth estimate is worthwhile (called from the YOLO thread)."""

    def __init__(self, cooldown_s, refresh_s, move_px, size_ratio, absent_frames):
        self.cooldown_s = cooldown_s
        self.refresh_s = refresh_s
        self.move_px = move_px
        self.size_ratio = size_ratio
        self.absent_frames = absent_frames
        self._last_finish = -math.inf
        self._ref: Optional[Detection] = None
        self._ref_time = -math.inf
        self._absent = 0
        self._need_new = True

    def note_absent(self):
        self._absent += 1
        if self._absent >= self.absent_frames:
            self._need_new = True

    def should_trigger(self, det: Detection, now: float, busy: bool):
        self._absent = 0
        if busy:
            return False, "busy"
        if now - self._last_finish < self.cooldown_s:
            return False, "cooldown"
        if self._need_new or self._ref is None:
            return True, "new flame"
        ref = self._ref
        if math.hypot(det.cx - ref.cx, det.cy - ref.cy) > self.move_px:
            return True, "moved"
        dw = abs(det.width - ref.width) / max(1.0, ref.width)
        dh = abs(det.height - ref.height) / max(1.0, ref.height)
        if max(dw, dh) > self.size_ratio:
            return True, "size changed"
        if now - self._ref_time > self.refresh_s:
            return True, "refresh"
        return False, "up to date"

    def mark_submitted(self, det: Detection, now: float):
        self._ref, self._ref_time, self._need_new = det, now, False

    def mark_finished(self, now: float):
        self._last_finish = now