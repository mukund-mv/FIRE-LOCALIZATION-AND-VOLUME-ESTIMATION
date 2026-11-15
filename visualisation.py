
"""Qt widgets: video label, info panel, and lightweight 2D flame localisation."""
from __future__ import annotations

import math

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

FONT = cv2.FONT_HERSHEY_SIMPLEX

DARK_STYLE = """
QWidget { background: #12121c; color: #e6e6f0; font-size: 13px; }
QLabel#title { color: #ff9a3c; font-weight: bold; font-size: 14px; padding: 2px; }
QGroupBox { border: 1px solid #33334a; border-radius: 6px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #ff9a3c; }
QPushButton { background: #26263a; border: 1px solid #44445e; border-radius: 4px; padding: 6px; }
QPushButton:hover { background: #33334d; }
"""


def annotate_frame(frame, detections, fps, depth_busy):
    """Draw YOLO boxes, labels, 2D centre and status on a COPY of the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for i, d in enumerate(detections):
        primary = i == 0
        color = (0, 80, 255) if primary else (0, 170, 255)

        cv2.rectangle(
            out, (d.x1, d.y1), (d.x2, d.y2),
            color, 3 if primary else 2
        )

        label = f"{d.class_name}: {int(d.confidence * 100)}%"
        (tw, th), base = cv2.getTextSize(label, FONT, 0.55, 1)
        ly = max(d.y1, th + 10)

        cv2.rectangle(
            out, (d.x1, ly - th - 10), (d.x1 + tw, ly + base - 10),
            color, cv2.FILLED
        )
        cv2.putText(
            out, label, (d.x1, ly - 7), FONT, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA
        )

        if primary:
            cx, cy = int(round(d.cx)), int(round(d.cy))
            cv2.drawMarker(
                out, (cx, cy), (255, 255, 255),
                cv2.MARKER_CROSS, 14, 2
            )
            cv2.putText(
                out, f"centre ({cx}, {cy}) px",
                (max(4, d.x1), min(h - 8, d.y2 + 18)),
                FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

    cv2.putText(
        out, f"YOLO FPS: {fps:0.1f}", (10, 22),
        FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA
    )

    if detections:
        cv2.putText(
            out, f"FIRE DETECTED ({len(detections)})", (10, 46),
            FONT, 0.65, (0, 60, 255), 2, cv2.LINE_AA
        )
    else:
        cv2.putText(
            out, "No fire detected", (10, 46),
            FONT, 0.65, (0, 220, 0), 2, cv2.LINE_AA
        )

    if depth_busy:
        cv2.putText(
            out, "Depth: computing...", (10, 70),
            FONT, 0.6, (255, 200, 0), 2, cv2.LINE_AA
        )

    return out


def bgr_to_pixmap(img) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QtGui.QImage(
        rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
    )
    return QtGui.QPixmap.fromImage(qimg)


class VideoLabel(QtWidgets.QLabel):
    def __init__(self, placeholder="", min_size=(320, 240)):
        super().__init__(placeholder)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(*min_size)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored
        )
        self.setStyleSheet(
            "background:#05050a; border:1px solid #33334a;"
        )
        self._pix = None

    def set_image(self, bgr):
        self._pix = bgr_to_pixmap(bgr)
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(
                self._pix.scaled(
                    self.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.FastTransformation
                )
            )


class InfoPanel(QtWidgets.QWidget):
    ROWS = [
        "Fire detected", "Class", "Confidence", "2D centre (px)",
        "Camera distance", "X coordinate", "Y coordinate (down +)",
        "Z coordinate", "Flame width", "Flame height",
        "Volume (APPROX. geometric)", "Relative depth (model)",
        "Distance mode", "Depth inference time", "Depth age",
        "YOLO inference FPS"
    ]

    def __init__(self, rows_per_column=8):
        super().__init__()
        grid = QtWidgets.QGridLayout(self)
        grid.setVerticalSpacing(3)
        self._values = {}

        mono = QtGui.QFont("Consolas")
        mono.setStyleHint(QtGui.QFont.Monospace)

        for i, key in enumerate(self.ROWS):
            row = i % rows_per_column
            col = (i // rows_per_column) * 2

            name = QtWidgets.QLabel(key)
            name.setStyleSheet("color:#9a9ab8;")

            val = QtWidgets.QLabel("—")
            val.setFont(mono)
            val.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )

            grid.addWidget(name, row, col)
            grid.addWidget(val, row, col + 1)
            self._values[key] = val

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

    def set(self, key, text, color=None):
        lbl = self._values[key]
        lbl.setText(text)
        lbl.setStyleSheet(f"color:{color};" if color else "")


def _nice_ceil(x: float) -> float:
    if x <= 0:
        return 1.0

    base = 10 ** math.floor(math.log10(x))

    for m in (1, 2, 5, 10):
        if x <= m * base * (1 + 1e-9):
            return m * base

    return 10 * base


class _SceneCanvas(QtWidgets.QWidget):
    """Paints two lightweight orthographic projections using QPainter."""

    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.setMinimumSize(300, 260)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, event):
        try:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

            painter.fillRect(self.rect(), QtGui.QColor("#080811"))

            width = self.width()
            height = self.height()
            gap = 12
            panel_width = (width - gap) / 2

            left = QtCore.QRectF(5, 5, panel_width - 5, height - 10)
            right = QtCore.QRectF(
                panel_width + gap, 5, panel_width - 5, height - 10
            )

            self._draw_projection(
                painter, left, "TOP VIEW", "X position", 0
            )
            self._draw_projection(
                painter, right, "SIDE VIEW", "Y position (down +)", 1
            )

            painter.end()
        finally:
            painter.end()
    def _draw_projection(self, painter, panel, title, axis_label, mode):
        scene = self.scene
        E = max(scene._extent or scene._min_extent, 0.01)
        unit = scene._unit

        # Panel background
        painter.setPen(QtGui.QPen(QtGui.QColor("#303047"), 1))
        painter.setBrush(QtGui.QColor("#10101b"))
        painter.drawRoundedRect(panel, 7, 7)

        # Plot area
        plot = panel.adjusted(38, 38, -12, -35)
        if plot.width() < 30 or plot.height() < 30:
            return

        # Scale
        scale = min(
            plot.width() / (2 * E),
            plot.height() / (2 * E)
        )

        centre_x = plot.center().x()
        bottom_y = plot.bottom()

        def point(horizontal, depth):
            return QtCore.QPointF(
                centre_x + horizontal * scale,
                bottom_y - depth * scale
            )

        # Camera is at the origin of the coordinate system.
        camera = point(0, 0)

        # Title
        painter.setPen(QtGui.QColor("#ff9a3c"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)

        painter.drawText(
            QtCore.QRectF(
                panel.left() + 10,
                panel.top() + 8,
                panel.width() - 20,
                22
            ),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            title
        )

        # Camera label
        painter.drawText(
            QtCore.QRectF(
                camera.x() + 8,
                camera.y() - 19,
                90,
                18
            ),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            "CAMERA"
        )

        # Grid
        step = _nice_ceil(E / 5.0)
        grid_pen = QtGui.QPen(QtGui.QColor(65, 65, 90, 100), 1)
        painter.setPen(grid_pen)

        x = -E
        while x <= E + step * 0.01:
            painter.drawLine(point(x, 0), point(x, 2 * E))
            x += step

        z = 0.0
        while z <= 2 * E + step * 0.01:
            painter.drawLine(point(-E, z), point(E, z))
            z += step

        # Horizontal and depth axes
        painter.setPen(QtGui.QPen(QtGui.QColor("#e06b6b"), 2))
        painter.drawLine(point(-E, 0), point(E, 0))

        painter.setPen(QtGui.QPen(QtGui.QColor("#5ee08a"), 2))
        painter.drawLine(point(0, 0), point(0, 2 * E))

        # Axis labels
        painter.setPen(QtGui.QColor("#8585a3"))
        font = painter.font()
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)

        painter.drawText(
            QtCore.QRectF(
                plot.left(), plot.bottom() + 5,
                plot.width(), 17
            ),
            QtCore.Qt.AlignCenter,
            f"{axis_label} ({unit})"
        )

        painter.drawText(
            QtCore.QRectF(
                panel.left() + 5, plot.top(),
                30, plot.height()
            ),
            QtCore.Qt.AlignCenter,
            "DEPTH"
        )

        # Camera marker
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#54c8ff"))
        painter.drawEllipse(camera, 5, 5)

        # Flame
        if not scene._has_flame:
            return

        if mode == 0:
            horizontal = scene._X
            radius = scene._a
        else:
            horizontal = scene._Y
            radius = scene._b

        depth = scene._Z

        flame_center = point(horizontal, depth)
        flame_radius_x = max(4, abs(radius * scale))
        flame_radius_y = max(4, abs(scene._c * scale))

        # Flame glow
        glow = QtGui.QRadialGradient(flame_center, max(
            flame_radius_x, flame_radius_y
        ) * 2.5)
        glow.setColorAt(0.0, QtGui.QColor(255, 130, 35, 180))
        glow.setColorAt(0.5, QtGui.QColor(255, 70, 15, 65))
        glow.setColorAt(1.0, QtGui.QColor(255, 50, 0, 0))

        painter.setBrush(glow)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(
            flame_center,
            flame_radius_x * 2.5,
            flame_radius_y * 2.5
        )

        # Flame body
        painter.setBrush(QtGui.QColor("#ff9a3c"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffe0a0"), 1.5))
        painter.drawEllipse(
            flame_center,
            flame_radius_x,
            flame_radius_y
        )

        # Flame centre
        painter.setBrush(QtGui.QColor("#fff3c4"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(flame_center, 3, 3)

        # Coordinates
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(
            QtCore.QRectF(
                flame_center.x() + 8,
                flame_center.y() - 22,
                120,
                20
            ),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"Z: {depth:.2f} {unit}"
        )


        # Plot rectangle.
        plot = panel.adjusted(38, 38, -12, -35)

        if plot.width() < 30 or plot.height() < 30:
            return

        # The same scale is used on both axes to preserve proportions.
        scale = min(
            plot.width() / (2 * E),
            plot.height() / (2 * E)
        )

        centre_x = plot.center().x()
        bottom_y = plot.bottom()

        def point(horizontal, depth):
            return QtCore.QPointF(
                centre_x + horizontal * scale,
                bottom_y - depth * scale
            )

        # Grid.
        step = _nice_ceil(E / 5.0)
        grid_pen = QtGui.QPen(QtGui.QColor(65, 65, 90, 100), 1)
        painter.setPen(grid_pen)

        x = -E
        while x <= E + step * 0.01:
            p1 = point(x, 0)
            p2 = point(x, 2 * E)
            painter.drawLine(p1, p2)
            x += step

        z = 0.0
        while z <= 2 * E + step * 0.01:
            p1 = point(-E, z)
            p2 = point(E, z)
            painter.drawLine(p1, p2)
            z += step

        # Coordinate axes.
        painter.setPen(QtGui.QPen(QtGui.QColor("#e06b6b"), 2))
        painter.drawLine(point(-E, 0), point(E, 0))

        painter.setPen(QtGui.QPen(QtGui.QColor("#5ee08a"), 2))
        painter.drawLine(point(0, 0), point(0, 2 * E))

        # Tick labels.
        painter.setPen(QtGui.QColor("#8585a3"))
        font = painter.font()
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        camera = QtCore.QPointF(
            rect.left() + 20,
            rect.center().y()
        )
        painter.drawText(
            QtCore.QRectF(
                plot.left(), plot.bottom() + 5,
                plot.width(), 17
            ),
            QtCore.Qt.AlignCenter,
            f"{axis_label} ({unit})"
        )

        painter.drawText(
            QtCore.QRectF(
                plot.left() - 34, plot.top(),
                30, 20
            ),
            QtCore.Qt.AlignRight,
            f"{2 * E:g}"
        )
        painter.drawText(
            QtCore.QRectF(
                plot.left() - 34, plot.bottom() - 16,
                30, 20
            ),
            QtCore.Qt.AlignRight,
            "0"
        )

        # Camera at the origin.
        camera = point(0, 0)
        painter.setPen(QtGui.QPen(QtGui.QColor("#5ee8ff"), 2))
        painter.setBrush(QtGui.QColor("#5ee8ff"))
        painter.drawEllipse(camera, 5, 5)

        painter.setPen(QtGui.QColor("#9deeff"))
        painter.drawText(
            QtCore.QRectF(
                camera.x() + 8, camera.y() - 19, 90, 18
            ),
            "CAMERA"
        )

        flame = scene._flame
        if flame is None:
            painter.setPen(QtGui.QColor("#77778f"))
            painter.drawText(
                plot, QtCore.Qt.AlignCenter,
                "Waiting for flame localisation..."
            )
            return

        X, Y, Z, a, b, c = flame

        if mode == 0:
            horizontal = X
            horizontal_radius = a
        else:
            horizontal = Y
            horizontal_radius = b

        flame_centre = point(horizontal, Z)
        flame_radius_x = max(horizontal_radius * scale, 3)
        flame_radius_y = max(c * scale, 3)

        # Camera-to-flame line.
        ray_pen = QtGui.QPen(
            QtGui.QColor(220, 220, 235, 150), 1.5,
            QtCore.Qt.DashLine
        )
        painter.setPen(ray_pen)
        painter.drawLine(camera, flame_centre)

        # Dashed reference line to the central distance axis.
        drop_pen = QtGui.QPen(
            QtGui.QColor(255, 160, 75, 150), 1,
            QtCore.Qt.DotLine
        )
        painter.setPen(drop_pen)
        painter.drawLine(
            flame_centre, point(0, Z)
        )

        # Flame glow. Flicker changes its apparent size slightly.
        phase = scene._phase
        flicker_h = (
            1.0 + 0.06 * math.sin(phase)
            + 0.03 * math.sin(2.7 * phase)
            if scene._flicker else 1.0
        )
        flicker_w = (
            1.0 + 0.03 * math.sin(1.3 * phase + 1.0)
            if scene._flicker else 1.0
        )

        rx = flame_radius_x * scene._visual_scale * flicker_w
        ry = flame_radius_y * scene._visual_scale * flicker_h

        flame_rect = QtCore.QRectF(
            flame_centre.x() - rx,
            flame_centre.y() - ry,
            2 * rx,
            2 * ry
        )

        glow_radius = max(rx, ry, 8) * 2.0
        gradient = QtGui.QRadialGradient(
            flame_centre, glow_radius
        )
        gradient.setColorAt(0.0, QtGui.QColor(255, 245, 130, 230))
        gradient.setColorAt(0.25, QtGui.QColor(255, 125, 25, 190))
        gradient.setColorAt(0.6, QtGui.QColor(255, 65, 10, 65))
        gradient.setColorAt(1.0, QtGui.QColor(255, 40, 0, 0))

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(
            QtCore.QRectF(
                flame_centre.x() - glow_radius,
                flame_centre.y() - glow_radius,
                2 * glow_radius,
                2 * glow_radius
            )
        )

        painter.setPen(
            QtGui.QPen(QtGui.QColor("#ffb347"), 1.5)
        )
        painter.setBrush(QtGui.QColor(255, 110, 15, 100))
        painter.drawEllipse(flame_rect)

        # Flame centre.
        painter.setPen(QtGui.QPen(QtGui.QColor("#fff2a6"), 1))
        painter.setBrush(QtGui.QColor("#fff2a6"))
        painter.drawEllipse(flame_centre, 3, 3)

        # Flame label.
        painter.setPen(QtGui.QColor("#ffd08a"))
        label = scene._label or f"Z = {Z:.2f} {unit}"
        painter.drawText(
            QtCore.QRectF(
                flame_centre.x() + 9,
                flame_centre.y() - ry - 20,
                max(100, plot.width() / 2),
                18
            ),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            label
        )


class Scene3D(QtWidgets.QWidget):
    """
    Lightweight replacement for the old OpenGL scene.

    Keeps the original Scene3D class name and public API so main.py
    does not need to be rewritten.

    Top view:  X against Z distance.
    Side view: Y against Z distance.
    """

    def __init__(self, min_extent=0.5, visual_scale=1.0, flicker=True):
        super().__init__()
        self._has_flame = False
        self._min_extent = float(min_extent)
        self._visual_scale = float(visual_scale)
        self._flicker = bool(flicker)

        self._extent = None
        self._unit = "m"
        self._flame = None
        self._label = ""
        self._phase = 0.0

        self.status = QtWidgets.QLabel("")
        self.status.setAlignment(QtCore.Qt.AlignCenter)

        self.canvas = _SceneCanvas(self)

        self.hint = QtWidgets.QLabel(
            "Top view: X vs distance   ·   Side view: Y vs distance"
        )
        self.hint.setAlignment(QtCore.Qt.AlignCenter)
        self.hint.setStyleSheet(
            "color:#7a7a98; font-size:11px; padding:3px;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.hint)

        self._set_extent(
            _nice_ceil(self._min_extent)
        )
        self.clear_flame(
            "NO FIRE DETECTED — empty coordinate space"
        )

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)

    def _set_extent(self, extent):
        self._extent = float(extent)
        step = _nice_ceil(self._extent / 5.0)
        self.hint.setText(
            f"Grid step: {step:g} {self._unit}   ·   "
            "Top: X position   ·   Side: Y position (image-down +)   ·   "
            "Vertical axis: Z distance"
        )
        self.canvas.update()

    def reset_view(self):
        """Kept for compatibility with the existing reset button."""
        self.canvas.update()

    def set_status(self, text, color="#e6e6f0"):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color:{color}; font-weight:bold; padding:2px;"
        )

    def clear_flame(self, message):
        self._flame = None
        self._label = ""
        self.set_status(message, "#9a9ab8")
        self.canvas.update()

    def update_flame(self, X, Y, Z, a, b, c, unit, label=""):
        """
        Camera-frame values:
          X = horizontal image position
          Y = vertical image position (down is positive)
          Z = distance from camera
          a, b, c = ellipsoid semi-axes along X, Y and Z
        """
        values = (X, Y, Z, a, b, c)
        if not all(math.isfinite(float(v)) for v in values):
            return

        X, Y, Z, a, b, c = map(float, values)

        a = max(a, 1e-4)
        b = max(b, 1e-4)
        c = max(c, 1e-4)

        self._unit = str(unit)
        self._label = str(label)

        # Expand the displayed coordinate space to fit the estimate.
        need = max(
            abs(X) + a,
            abs(Y) + b,
            (abs(Z) + c) / 1.8
        ) * 1.25

        extent = _nice_ceil(
            max(need, self._min_extent)
        )

        if extent != self._extent:
            self._set_extent(extent)
        else:
            self.canvas.update()

        self._flame = (X, Y, Z, a, b, c)

    def _tick(self):
        if self._flame is not None and self._flicker:
            self._phase += 0.35
            self.canvas.update()
