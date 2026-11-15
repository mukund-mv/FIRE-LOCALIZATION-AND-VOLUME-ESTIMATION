#!/usr/bin/env python3
"""Real-time 3D fire detection, localisation and volume estimation.
YOLO (continuous) + Depth Anything V2 (triggered, background thread) + live Qt/OpenGL dashboard."""
from __future__ import annotations
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import torch

from PyQt5 import QtCore, QtWidgets

import config as cfg
from calibration import DistanceCalibrator
from depth_estimator import (
    DepthAnythingPredictor, DepthJob, DepthResult,
    DepthTrigger, DepthWorker
)
from fire_detector import Detection, FireDetector
from geometry import (
    CameraIntrinsics, DistanceMode, EllipsoidVolumeEstimator,
    FlameGeometry, compute_flame_geometry, load_intrinsics
)
from visualisation import (
    DARK_STYLE, InfoPanel, Scene3D, VideoLabel, annotate_frame
)

GREEN, ORANGE, RED, GREY = "#5ee08a", "#ffb347", "#ff5c5c", "#9a9ab8"


# --------------------------------------------------------------------------
# Threading helpers
# --------------------------------------------------------------------------
class Bridge(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal()
    depth_ready = QtCore.pyqtSignal(object)
    fatal = QtCore.pyqtSignal(str)


class FrameMailbox:
    """Single-slot mailbox: the GUI always gets the newest frame; stale ones are overwritten."""
    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._pending = False

    def put(self, item) -> bool:
        with self._lock:
            self._item = item
            was_pending, self._pending = self._pending, True
            return not was_pending          # True => caller should emit a signal

    def take(self):
        with self._lock:
            item, self._item, self._pending = self._item, None, False
            return item


def open_camera():
    backends = {"auto": cv2.CAP_ANY, "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
                "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
                "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
                "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)}
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX, backends.get(cfg.CAMERA_BACKEND, cv2.CAP_ANY))
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {cfg.CAMERA_INDEX}. Check `ls /dev/video*`, and that your "
            "user is in the 'video' group (sudo usermod -aG video $USER, then log out/in).")
    if getattr(cfg, "CAMERA_FOURCC", None):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Camera opened but returned no frame.")
    print(f"      camera delivers {frame.shape[1]}x{frame.shape[0]} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    return cap, frame.shape


class CaptureDetectLoop(threading.Thread):
    """Continuously: grab frame -> YOLO -> maybe submit (frame, bbox) to the depth worker."""

    def __init__(self, cap, detector, trigger, depth_worker, mailbox, bridge):
        super().__init__(name="CaptureDetect", daemon=True)
        self.cap, self.detector, self.trigger = cap, detector, trigger
        self.depth_worker, self.mailbox, self.bridge = depth_worker, mailbox, bridge
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        failures, frame_id, fps = 0, 0, None
        try:
            while not self._stop_evt.is_set():
                t_start = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 60:
                        self.bridge.fatal.emit("Camera stopped delivering frames.")
                        break
                    time.sleep(0.02)
                    continue
                failures = 0
                frame_id += 1

                dets = self.detector.detect(frame)
                primary = dets[0] if dets else None
                now = time.monotonic()
                if primary is None:
                    self.trigger.note_absent()
                else:
                    go, _reason = self.trigger.should_trigger(primary, now, self.depth_worker.is_busy())
                    if go:
                        # Frame and bbox come from the same iteration => always consistent.
                        job = DepthJob(frame_id, frame.copy(), primary, now)
                        if self.depth_worker.submit(job):
                            self.trigger.mark_submitted(primary, now)

                annotated = annotate_frame(frame, dets, fps or 0.0, self.depth_worker.is_busy())
                if self.mailbox.put((annotated, primary, len(dets), fps or 0.0)):
                    self.bridge.frame_ready.emit()

                dt = time.perf_counter() - t_start
                inst = 1.0 / dt if dt > 0 else 0.0
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
        except Exception as exc:
            traceback.print_exc()
            self.bridge.fatal.emit(f"Detection loop crashed: {exc}")
        finally:
            self.cap.release()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
@dataclass
class Estimate:
    geom: FlameGeometry
    mode: DistanceMode
    extrapolated: bool
    result: DepthResult


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge, mailbox, loop, depth_worker, intrinsics: CameraIntrinsics,
                 calibrator: DistanceCalibrator, volume_estimator):
        super().__init__()
        self.setWindowTitle(cfg.WINDOW_TITLE)
        self.resize(1400, 900)
        self.mailbox, self.loop, self.depth_worker = mailbox, loop, depth_worker
        self.intrinsics, self.calibrator, self.volume_estimator = intrinsics, calibrator, volume_estimator

        self.last_depth: Optional[DepthResult] = None
        self._estimate: Optional[Estimate] = None
        self._last_fire_seen = -1e9
        self._scene_state = ""

        # ---- layout

        # ---- layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # Keep Scene3D internally so existing geometry/update calls
        # continue working, but do not display it.
        self.scene = Scene3D(
            cfg.SCENE_MIN_EXTENT,
            cfg.FLAME_VISUAL_SCALE,
            cfg.FLAME_FLICKER
        )

        # Camera and depth preview side by side
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(12)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()

        t1 = QtWidgets.QLabel("LIVE CAMERA · YOLO FLAME DETECTION")
        t2 = QtWidgets.QLabel("DEPTH MAP · DEPTH ANYTHING V2")

        t1.setObjectName("title")
        t2.setObjectName("title")

        self.video = VideoLabel("Starting camera…", (480, 360))

        self.depth_preview = VideoLabel(
            "Depth map\n(after first estimate)",
            (480, 360)
        )
        self.depth_preview.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.depth_preview.setVisible(cfg.SHOW_DEPTH_PREVIEW)

        left.addWidget(t1)
        left.addWidget(self.video, 1)

        right.addWidget(t2)
        right.addWidget(self.depth_preview, 1)

        top.addLayout(left, 1)
        top.addLayout(right, 1)

        root.addLayout(top, 1)

        # Status banner
        self.banner = QtWidgets.QLabel("")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setWordWrap(True)
        root.addWidget(self.banner)

        # Specifications and calibration underneath
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(14)

        self.info = InfoPanel()
        self.info.setMinimumWidth(650)

        bottom.addWidget(self.info, 1)
        bottom.addWidget(self._build_calibration_box())

        root.addLayout(bottom)

        self.statusBar().showMessage(f"Camera intrinsics: {intrinsics.source} "
                                     f"(fx={intrinsics.fx:.0f}, fy={intrinsics.fy:.0f})")

        bridge.frame_ready.connect(self.on_frame)
        bridge.depth_ready.connect(self.on_depth)
        bridge.fatal.connect(self.on_fatal)

        self._clear_estimate_fields()
        self._update_banner()
        self._update_calib_label()

    def _build_calibration_box(self):
        box = QtWidgets.QGroupBox("Distance calibration")
        box.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(box)
        help_ = QtWidgets.QLabel("Hold the flame at a KNOWN distance (measured along the camera "
                                 "axis), wait for a depth reading, then add a point. Use 2+ distances.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color:#9a9ab8; font-size:11px;")
        self.calib_label = QtWidgets.QLabel("")
        self.calib_label.setWordWrap(True)
        b_add = QtWidgets.QPushButton("Add calibration point…")
        b_clear = QtWidgets.QPushButton("Clear calibration")
        b_add.clicked.connect(self.on_add_calibration)
        b_clear.clicked.connect(self.on_clear_calibration)
        for w in (help_, self.calib_label, b_add, b_clear):
            lay.addWidget(w)
        if cfg.DEPTH_MODE == "metric":
            b_add.setEnabled(False); b_clear.setEnabled(False)
        return box

    # ---- helpers ---------------------------------------------------------------
    def _fire_visible(self, now):
        return (now - self._last_fire_seen) <= cfg.FIRE_LOST_TIMEOUT_S

    def _update_banner(self, estimate: Optional[Estimate] = None):
        if cfg.DEPTH_MODE == "metric":
            text, style = "METRIC depth checkpoint — distances in metres (still approximate)", GREEN
        elif estimate is not None and estimate.mode is DistanceMode.CALIBRATED:
            if estimate.extrapolated:
                text, style = ("CALIBRATED distance — EXTRAPOLATED outside the calibrated range: "
                               "treat as low confidence"), ORANGE
            else:
                text, style = "CALIBRATED distance (approximate, from your calibration points)", GREEN
        elif self.calibrator.is_calibrated and estimate is None:
            text, style = "Distance mode: CALIBRATED (waiting for estimate)", GREEN
        else:
            text, style = ("UNCALIBRATED — relative depth only. Values are in arbitrary units (u), "
                           "NOT metres. Add calibration points to get metres."), RED
        if self.intrinsics.approximate:
            text += f"   |   Intrinsics APPROXIMATE ({self.intrinsics.source})"
        self.banner.setText(text)
        self.banner.setStyleSheet(f"color:{style}; font-weight:bold; padding:4px; "
                                  f"border:1px solid {style}; border-radius:4px;")

    def _update_calib_label(self):
        self.calib_label.setText("Metric checkpoint: calibration not needed." if cfg.DEPTH_MODE == "metric"
                                 else self.calibrator.describe())

    def _clear_estimate_fields(self):
        for k in ("Camera distance", "X coordinate", "Y coordinate (down +)", "Z coordinate",
                  "Flame width", "Flame height", "Volume (APPROX. geometric)", "Distance mode"):
            self.info.set(k, "—", GREY)

    # ---- slots -----------------------------------------------------------------
    def on_frame(self):
        item = self.mailbox.take()
        if item is None:
            return
        annotated, primary, n, fps = item
        now = time.monotonic()
        self.video.set_image(annotated)
        self.info.set("YOLO inference FPS", f"{fps:0.1f}")
        if primary is not None:
            self._last_fire_seen = now
            self.info.set("Fire detected", "YES", RED)
            self.info.set("Class", primary.class_name)
            self.info.set("Confidence", f"{primary.confidence * 100:.1f} %")
            self.info.set("2D centre (px)", f"({primary.cx:.0f}, {primary.cy:.0f})")
        else:
            self.info.set("Fire detected", "NO", GREEN)
            for k in ("Class", "Confidence", "2D centre (px)"):
                self.info.set(k, "—", GREY)
        self._refresh_dynamic(now)

    def _refresh_dynamic(self, now):
        if self.last_depth is not None:
            self.info.set("Depth age", f"{now - self.last_depth.capture_time:.1f} s")
        visible = self._fire_visible(now)

        if self._estimate is not None:
            if (now - self._last_fire_seen) <= cfg.SNAPSHOT_HOLD_S:
                if now - getattr(self, "_last_status_update", 0.0) > 0.5:
                    self._last_status_update = now
                    self._set_snapshot_status(now, visible)
                return                                   # keep the snapshot on screen
            self._estimate = None                        # held long enough: clear it
            self._scene_state = ""
            self._clear_estimate_fields()
            self._update_banner()

        if visible and self._scene_state != "waiting":
            self.scene.clear_flame("Fire detected — waiting for depth estimate…")
            self._scene_state = "waiting"
        elif not visible and self._scene_state != "empty":
            self.scene.clear_flame("NO FIRE DETECTED — empty coordinate space")
            self._scene_state = "empty"
            self._clear_estimate_fields()

    def on_depth(self, result: DepthResult):
        self.last_depth = result
        if result.error:
            self.statusBar().showMessage(f"Depth error: {result.error}")
            return
        self.info.set("Depth inference time", f"{result.inference_ms:.0f} ms")
        if result.depth_preview is not None and cfg.SHOW_DEPTH_PREVIEW:
            self.depth_preview.set_image(result.depth_preview)
        if result.relative_depth is None:
            self.statusBar().showMessage(result.message or "No valid depth for flame region")
            return
        rel_txt = (f"{result.relative_depth:.3f} m (metric)" if result.is_metric
                   else f"{result.relative_depth:.4f} (relative, unitless)")
        self.info.set("Relative depth (model)", rel_txt)
        try:
            self._recompute()
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"Display error: {exc}")

    def on_fatal(self, message):
        QtWidgets.QMessageBox.critical(self, "Error", message)
        self.close()

    def _recompute(self):
        r = self.last_depth
        if r is None or r.error or r.relative_depth is None:
            return
        rel, extrap = r.relative_depth, False
        if cfg.DEPTH_MODE == "metric":
            Z, mode = rel, DistanceMode.METRIC
        else:
            conv = self.calibrator.distance(rel)
            if conv is not None:
                Z, mode, extrap = conv.distance_m, DistanceMode.CALIBRATED, conv.extrapolated
            else:
                Z, mode = cfg.UNCALIBRATED_SCALE / max(rel, 1e-3), DistanceMode.UNCALIBRATED
        h, w = r.frame_shape[:2]
        intr = self.intrinsics.scaled_to(w, h)
        d = r.detection
        geom = compute_flame_geometry(d.bbox, Z, intr, self.volume_estimator)
        self._estimate = Estimate(geom, mode, extrap, r)
        self._show_estimate()

    def _set_snapshot_status(self, now, visible):
        est = self._estimate
        age = now - est.result.capture_time
        tail = "" if visible else "  ·  flame no longer detected (snapshot held)"
        self.scene.set_status(f"Flame snapshot — {self._snap_label}  ·  taken {age:.0f} s ago{tail}",
                              self._snap_colour)

    def _show_estimate(self):
        est = self._estimate
        g, mode = est.geom, est.mode
        u = mode.unit
        colour = {DistanceMode.METRIC: GREEN, DistanceMode.CALIBRATED: ORANGE if est.extrapolated else GREEN,
                  DistanceMode.UNCALIBRATED: RED}[mode]
        tag = "" if mode.in_metres else "  (arbitrary units, NOT metres)"
        self.info.set("Camera distance", f"{g.range_:.3f} {u}{tag}", colour)
        self.info.set("X coordinate", f"{g.X:+.3f} {u}", colour)
        self.info.set("Y coordinate (down +)", f"{g.Y:+.3f} {u}", colour)
        self.info.set("Z coordinate", f"{g.Z:.3f} {u}", colour)
        self.info.set("Flame width", f"{g.width:.3f} {u}", colour)
        self.info.set("Flame height", f"{g.height:.3f} {u}", colour)
        vol = g.volume.volume
        vtxt = (f"{vol:.3e} m³  ({vol * 1e6:.2f} cm³)" if mode.in_metres else f"{vol:.3e} u³")
        self.info.set("Volume (APPROX. geometric)", vtxt, colour)
        label = {DistanceMode.METRIC: "METRIC", DistanceMode.CALIBRATED:
                 "CALIBRATED (extrapolated)" if est.extrapolated else "CALIBRATED",
                 DistanceMode.UNCALIBRATED: "UNCALIBRATED (relative, arbitrary units)"}[mode]
        self.info.set("Distance mode", label, colour)

        a, b, c = g.volume.semi_axes
        self.scene.update_flame(g.X, g.Y, g.Z, a, b, c, u, f"Z = {g.Z:.2f} {u}")
        self._scene_state = "shown"
        self._snap_label, self._snap_colour = label, colour
        now = time.monotonic()
        self._last_status_update = now
        self._set_snapshot_status(now, self._fire_visible(now))
        self._update_banner(est)

    # ---- calibration -------------------------------------------------------------
    def on_add_calibration(self):
        r = self.last_depth
        if r is None or r.relative_depth is None or r.error:
            QtWidgets.QMessageBox.information(self, "Calibration",
                                              "No depth reading yet. Show a flame to the camera first.")
            return
        age = time.monotonic() - r.capture_time
        if age > 12:
            QtWidgets.QMessageBox.information(
                self, "Calibration",
                f"The last depth reading is {age:.0f} s old. Hold the flame at the known distance "
                "until a fresh reading appears, then try again.")
            return
        dist, ok = QtWidgets.QInputDialog.getDouble(
            self, "Add calibration point",
            f"Relative depth of the last reading: {r.relative_depth:.4f}\n"
            "Known flame distance along the camera axis (metres):", 0.5, 0.05, 100.0, 3)
        if ok:
            self.calibrator.add_point(r.relative_depth, dist)
            self.calibrator.save()
            self._update_calib_label()
            self._recompute()
            self._update_banner(self._estimate)

    def on_clear_calibration(self):
        if QtWidgets.QMessageBox.question(self, "Clear calibration",
                                          "Remove all calibration points?") == QtWidgets.QMessageBox.Yes:
            self.calibrator.clear()
            self.calibrator.save()
            self._update_calib_label()
            self._recompute()
            self._update_banner(self._estimate)

    # ---- shutdown ----------------------------------------------------------------
    def closeEvent(self, event):
        self.loop.stop()
        self.depth_worker.stop()
        self.loop.join(timeout=3.0)
        super().closeEvent(event)


# --------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    try:
        print("[1/4] Loading YOLO model…")
        detector = FireDetector(cfg.YOLO_MODEL_PATH, cfg.YOLO_CONF_THRESHOLD,
                                cfg.YOLO_FIRE_CLASSES, cfg.YOLO_DEVICE)
        print(f"[2/4] Loading Depth Anything V2 ({cfg.DEPTH_ENCODER}, mode={cfg.DEPTH_MODE})…")
        predictor = DepthAnythingPredictor(
            encoder=cfg.DEPTH_ENCODER, device=cfg.DEPTH_DEVICE, checkpoint=cfg.DEPTH_CHECKPOINT,
            metric=(cfg.DEPTH_MODE == "metric"), max_depth=cfg.DEPTH_METRIC_MAX,
            repo_dir=cfg.DEPTH_REPO_DIR, input_size=cfg.DEPTH_INPUT_SIZE)
        print(f"      depth device: {predictor.device}")
        print("[3/4] Opening camera…")
        cap, shape = open_camera()
        print("[4/4] Warming up models…")
        detector.warmup(shape)
        predictor.warmup(shape)
    except Exception as exc:
        traceback.print_exc()
        print(f"\nStartup failed: {exc}")
        sys.exit(1)

    intrinsics = load_intrinsics(cfg, shape[1], shape[0])
    calibrator = DistanceCalibrator(cfg.CALIBRATION_FILE, cfg.CALIBRATION_POINTS)
    volume_estimator = EllipsoidVolumeEstimator(cfg.FLAME_DEPTH_RATIO)
    trigger = DepthTrigger(cfg.DEPTH_COOLDOWN_S, cfg.DEPTH_REFRESH_S, cfg.DEPTH_MOVE_THRESHOLD_PX,
                           cfg.DEPTH_SIZE_CHANGE_RATIO, cfg.DEPTH_NEW_FLAME_AFTER_ABSENT_FRAMES)
    bridge, mailbox = Bridge(), FrameMailbox()

    def on_depth_result(result: DepthResult):
        trigger.mark_finished(time.monotonic())
        bridge.depth_ready.emit(result)

    depth_worker = DepthWorker(predictor, on_depth_result, cfg.DEPTH_ROI_SHRINK,
                               cfg.DEPTH_NEAR_KEEP_FRACTION, cfg.DEPTH_MIN_VALID_PIXELS)
    loop = CaptureDetectLoop(cap, detector, trigger, depth_worker, mailbox, bridge)

    window = MainWindow(bridge, mailbox, loop, depth_worker, intrinsics, calibrator, volume_estimator)
    window.show()
    depth_worker.start()
    loop.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()