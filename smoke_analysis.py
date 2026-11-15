
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from collections import deque
from datetime import datetime
import csv
import os
import time
import threading


# ============================================================
# FIREWATCH - OPTICAL SMOKE DETECTION
# Tkinter GUI + OpenCV Headless
# ============================================================

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480

HISTORY_LENGTH = 180

BRIGHTNESS_THRESHOLD = 200

BRIGHTNESS_Z_THRESHOLD = 3.0
FLOW_Z_THRESHOLD = 2.5
STRONG_ANOMALY_THRESHOLD = 6.0

PERSISTENCE_FRAMES = 8
CLEAR_FRAMES = 30

CSV_FILE = "firewatch_smoke_data.csv"


# ============================================================
# ENGINE
# ============================================================

class SmokeEngine:

    def __init__(self):

        self.background = self.create_background()

        self.previous_gray = None

        self.brightness_history = deque(maxlen=HISTORY_LENGTH)
        self.flow_history = deque(maxlen=HISTORY_LENGTH)
        self.foreground_history = deque(maxlen=HISTORY_LENGTH)

        self.graph_brightness = deque(maxlen=HISTORY_LENGTH)
        self.graph_flow = deque(maxlen=HISTORY_LENGTH)
        self.graph_anomaly = deque(maxlen=HISTORY_LENGTH)

        self.baseline = None

        self.persistence = 0
        self.clear_persistence = 0
        self.alert_active = False

        self.frame_count = 0

        self.latest_data = {}

        self.csv_rows = []

        self.lock = threading.Lock()

    @staticmethod
    def create_background():

        return cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=30,
            detectShadows=False
        )

    def reset(self):

        with self.lock:

            self.background = self.create_background()

            self.previous_gray = None

            self.brightness_history.clear()
            self.flow_history.clear()
            self.foreground_history.clear()

            self.graph_brightness.clear()
            self.graph_flow.clear()
            self.graph_anomaly.clear()

            self.baseline = None

            self.persistence = 0
            self.clear_persistence = 0
            self.alert_active = False

            self.frame_count = 0

            self.latest_data = {}
            self.csv_rows = []

    def calibrate(self):

        with self.lock:

            if len(self.brightness_history) < 30:
                return False

            self.baseline = {

                "brightness": float(
                    np.mean(self.brightness_history)
                ),

                "brightness_std": max(
                    float(np.std(self.brightness_history)),
                    1.0
                ),

                "flow": float(
                    np.mean(self.flow_history)
                ),

                "flow_std": max(
                    float(np.std(self.flow_history)),
                    0.02
                ),

                "foreground": float(
                    np.mean(self.foreground_history)
                ),

                "foreground_std": max(
                    float(np.std(self.foreground_history)),
                    0.005
                )
            }

            self.persistence = 0
            self.clear_persistence = 0
            self.alert_active = False

            return True

    @staticmethod
    def safe_z(value, mean, std, minimum):

        return abs(value - mean) / max(std, minimum)

    def process(self, frame):

        with self.lock:

            frame = cv2.resize(
                frame,
                (WIDTH, HEIGHT)
            )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ----------------------------------------
            # BRIGHTNESS
            # ----------------------------------------

            brightness = float(np.mean(gray))

            brightness_std = float(np.std(gray))

            _, bright_mask = cv2.threshold(
                gray,
                BRIGHTNESS_THRESHOLD,
                255,
                cv2.THRESH_BINARY
            )

            bright_coverage = float(
                np.count_nonzero(bright_mask)
                / bright_mask.size
            )

            # ----------------------------------------
            # FOREGROUND
            # ----------------------------------------

            foreground = self.background.apply(gray)

            _, foreground = cv2.threshold(
                foreground,
                200,
                255,
                cv2.THRESH_BINARY
            )

            foreground_coverage = float(
                np.count_nonzero(foreground)
                / foreground.size
            )

            # ----------------------------------------
            # OPTICAL FLOW
            # ----------------------------------------

            flow_magnitude = 0.0
            flow_std = 0.0
            flow_u = 0.0
            flow_v = 0.0
            flow_direction = 0.0

            flow_visual = np.zeros_like(frame)

            if self.previous_gray is not None:

                flow = cv2.calcOpticalFlowFarneback(
                    self.previous_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    21,
                    3,
                    5,
                    1.2,
                    0
                )

                magnitude, angle = cv2.cartToPolar(
                    flow[..., 0],
                    flow[..., 1]
                )

                flow_magnitude = float(
                    np.mean(magnitude)
                )

                flow_std = float(
                    np.std(magnitude)
                )

                flow_u = float(
                    np.mean(flow[..., 0])
                )

                flow_v = float(
                    np.mean(flow[..., 1])
                )

                flow_direction = float(
                    np.degrees(
                        np.arctan2(flow_v, flow_u)
                    )
                )

                hsv = np.zeros_like(frame)

                hsv[..., 0] = (
                    angle * 180 / np.pi / 2
                ).astype(np.uint8)

                hsv[..., 1] = 255

                hsv[..., 2] = cv2.normalize(
                    magnitude,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                flow_visual = cv2.cvtColor(
                    hsv,
                    cv2.COLOR_HSV2BGR
                )

            self.previous_gray = gray.copy()

            # ----------------------------------------
            # HISTORY
            # ----------------------------------------

            self.brightness_history.append(brightness)
            self.flow_history.append(flow_magnitude)
            self.foreground_history.append(foreground_coverage)

            self.graph_brightness.append(brightness)
            self.graph_flow.append(flow_magnitude)

            self.frame_count += 1

            # ----------------------------------------
            # ANOMALY DETECTION
            # ----------------------------------------

            brightness_z = 0.0
            flow_z = 0.0
            foreground_z = 0.0

            anomaly_score = 0.0
            suspicious = False

            if self.baseline is not None:

                brightness_z = self.safe_z(
                    brightness,
                    self.baseline["brightness"],
                    self.baseline["brightness_std"],
                    1.0
                )

                flow_z = self.safe_z(
                    flow_magnitude,
                    self.baseline["flow"],
                    self.baseline["flow_std"],
                    0.02
                )

                foreground_z = self.safe_z(
                    foreground_coverage,
                    self.baseline["foreground"],
                    self.baseline["foreground_std"],
                    0.005
                )

                brightness_score = min(
                    100,
                    brightness_z * 20
                )

                flow_score = min(
                    100,
                    flow_z * 20
                )

                foreground_score = min(
                    100,
                    foreground_z * 20
                )

                anomaly_score = (
                    brightness_score * 0.4
                    + flow_score * 0.4
                    + foreground_score * 0.2
                )

                suspicious = (
                    (
                        brightness_z > BRIGHTNESS_Z_THRESHOLD
                        and flow_z > FLOW_Z_THRESHOLD
                    )
                    or brightness_z > STRONG_ANOMALY_THRESHOLD
                    or flow_z > STRONG_ANOMALY_THRESHOLD
                    or (
                        foreground_z > 5
                        and brightness_z > 2
                    )
                )

                if suspicious:

                    self.persistence += 1
                    self.clear_persistence = 0

                else:

                    self.persistence = max(
                        0,
                        self.persistence - 1
                    )

                    self.clear_persistence += 1

                if self.persistence >= PERSISTENCE_FRAMES:
                    self.alert_active = True

                if self.clear_persistence >= CLEAR_FRAMES:
                    self.alert_active = False
                    self.persistence = 0

            self.graph_anomaly.append(anomaly_score)

            # ----------------------------------------
            # STATUS
            # ----------------------------------------

            if self.baseline is None:

                status = "CALIBRATION REQUIRED"

            elif self.frame_count < 30:

                status = "ANALYSING"

            elif self.alert_active:

                status = "SMOKE INDICATOR DETECTED"

            elif suspicious:

                status = "POSSIBLE SMOKE INDICATOR"

            else:

                status = "NORMAL"

            data = {

                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "brightness": brightness,
                "brightness_std": brightness_std,
                "bright_coverage": bright_coverage,
                "foreground_coverage": foreground_coverage,

                "flow_magnitude": flow_magnitude,
                "flow_std": flow_std,
                "flow_u": flow_u,
                "flow_v": flow_v,
                "flow_direction": flow_direction,

                "brightness_z": brightness_z,
                "flow_z": flow_z,
                "foreground_z": foreground_z,

                "anomaly_score": anomaly_score,

                "status": status,
                "event": int(self.alert_active)
            }

            self.latest_data = data

            self.csv_rows.append(data)

            return {
                "frame": frame.copy(),
                "gray": gray.copy(),
                "bright_mask": bright_mask.copy(),
                "foreground": foreground.copy(),
                "flow_visual": flow_visual.copy(),
                "data": data.copy()
            }


# ============================================================
# GUI
# ============================================================

class FirewatchApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "FIREWATCH | Optical Smoke Detection"
        )

        self.root.geometry("1450x920")

        self.root.configure(
            bg="#080e19"
        )

        self.engine = SmokeEngine()

        self.camera = None

        self.running = False
        self.paused = False

        self.latest_result = None

        self.last_export_count = 0

        self.setup_ui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    def setup_ui(self):

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "TButton",
            background="#243650",
            foreground="white",
            padding=8
        )

        # HEADER
        header = tk.Frame(
            self.root,
            bg="#080e19"
        )

        header.pack(
            fill="x",
            padx=25,
            pady=15
        )

        tk.Label(
            header,
            text="🔥 FIREWATCH",
            font=("Arial", 25, "bold"),
            fg="white",
            bg="#080e19"
        ).pack(side="left")

        tk.Label(
            header,
            text="OPTICAL SMOKE MONITORING SYSTEM",
            font=("Arial", 10),
            fg="#8ca4c1",
            bg="#080e19"
        ).pack(
            side="left",
            padx=20,
            pady=10
        )

        # CONTROLS
        controls = tk.Frame(
            self.root,
            bg="#101b2c"
        )

        controls.pack(
            fill="x",
            padx=25,
            pady=5
        )

        tk.Button(
            controls,
            text="START CAMERA",
            command=self.start_camera,
            bg="#174b42",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        ).pack(side="left", padx=6, pady=10)

        tk.Button(
            controls,
            text="STOP CAMERA",
            command=self.stop_camera,
            bg="#57272c",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        ).pack(side="left", padx=6)

        tk.Button(
            controls,
            text="CAPTURE CLEAN-AIR BASELINE",
            command=self.calibrate,
            bg="#24466d",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        ).pack(side="left", padx=6)

        tk.Button(
            controls,
            text="RESET",
            command=self.reset,
            bg="#424b5b",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        ).pack(side="left", padx=6)

        tk.Button(
            controls,
            text="EXPORT CSV",
            command=self.export_csv,
            bg="#704b24",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        ).pack(side="left", padx=6)

        # STATUS
        self.status_frame = tk.Frame(
            self.root,
            bg="#263447",
            height=65
        )

        self.status_frame.pack(
            fill="x",
            padx=25,
            pady=12
        )

        self.status_label = tk.Label(
            self.status_frame,
            text="CAMERA OFFLINE",
            font=("Arial", 19, "bold"),
            fg="white",
            bg="#263447"
        )

        self.status_label.pack(
            pady=15
        )

        # MAIN CONTENT
        main = tk.Frame(
            self.root,
            bg="#080e19"
        )

        main.pack(
            fill="both",
            expand=True,
            padx=25
        )

        left = tk.Frame(
            main,
            bg="#080e19"
        )

        left.pack(
            side="left",
            fill="both",
            expand=True
        )

        right = tk.Frame(
            main,
            bg="#101b2c",
            width=350
        )

        right.pack(
            side="right",
            fill="y",
            padx=(15, 0)
        )

        # CAMERA
        tk.Label(
            left,
            text="LIVE CAMERA FEED",
            fg="#9bb1cc",
            bg="#080e19",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.camera_label = tk.Label(
            left,
            bg="#111d2c",
            text="WAITING FOR CAMERA",
            fg="white",
            font=("Arial", 16)
        )

        self.camera_label.pack(
            fill="both",
            expand=True,
            pady=8
        )

        # IMAGE PROCESSING
        images = tk.Frame(
            left,
            bg="#080e19"
        )

        images.pack(fill="x")

        self.image_labels = []

        for title in [
            "DENSE OPTICAL FLOW",
            "BRIGHT REGIONS",
            "FOREGROUND MOTION"
        ]:

            panel = tk.Frame(
                images,
                bg="#101b2c"
            )

            panel.pack(
                side="left",
                fill="both",
                expand=True,
                padx=4
            )

            tk.Label(
                panel,
                text=title,
                fg="#9bb1cc",
                bg="#101b2c",
                font=("Arial", 8, "bold")
            ).pack(pady=5)

            label = tk.Label(
                panel,
                bg="#111d2c"
            )

            label.pack(
                fill="both",
                expand=True
            )

            self.image_labels.append(label)

        # METRICS
        tk.Label(
            right,
            text="SYSTEM METRICS",
            fg="white",
            bg="#101b2c",
            font=("Arial", 14, "bold")
        ).pack(
            pady=15
        )

        self.metric_labels = {}

        metric_names = [
            "Brightness",
            "Brightness variation",
            "Bright region coverage",
            "Foreground coverage",
            "Optical flow",
            "Flow direction",
            "Anomaly score",
            "Alert persistence"
        ]

        for name in metric_names:

            box = tk.Frame(
                right,
                bg="#19283c"
            )

            box.pack(
                fill="x",
                padx=12,
                pady=5
            )

            tk.Label(
                box,
                text=name.upper(),
                fg="#8ca4c1",
                bg="#19283c",
                font=("Arial", 8)
            ).pack(
                anchor="w",
                padx=10,
                pady=(6, 0)
            )

            value = tk.Label(
                box,
                text="--",
                fg="white",
                bg="#19283c",
                font=("Arial", 15, "bold")
            )

            value.pack(
                anchor="w",
                padx=10,
                pady=(2, 7)
            )

            self.metric_labels[name] = value

        self.graph_label = tk.Label(
            right,
            bg="#101b2c"
        )

        self.graph_label.pack(
            fill="x",
            padx=5,
            pady=10
        )

        self.footer = tk.Label(
            self.root,
            text="FIREWATCH | Experimental optical anomaly detection",
            bg="#080e19",
            fg="#647d9c",
            font=("Arial", 9)
        )

        self.footer.pack(
            pady=10
        )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    def start_camera(self):

        if self.running:
            return

        self.camera = cv2.VideoCapture(
            0,
            cv2.CAP_V4L2
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            HEIGHT
        )

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            self.status_label.config(
                text="CAMERA CONNECTION FAILED",
                bg="#57272c"
            )

            return

        self.running = True

        self.status_label.config(
            text="ANALYSING CAMERA FEED",
            bg="#24466d"
        )

        self.update_frame()

    def stop_camera(self):

        self.running = False

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.status_label.config(
            text="CAMERA STOPPED",
            bg="#263447"
        )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    def calibrate(self):

        if self.engine.calibrate():

            self.status_label.config(
                text="CLEAN-AIR BASELINE CAPTURED",
                bg="#174b42"
            )

        else:

            self.status_label.config(
                text="NEED AT LEAST 30 FRAMES FIRST",
                bg="#704b24"
            )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.engine.reset()

        self.status_label.config(
            text="ANALYSIS RESET - RECAPTURE BASELINE",
            bg="#263447"
        )

    # --------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------

    def update_frame(self):

        if not self.running or self.camera is None:
            return

        ret, frame = self.camera.read()

        if not ret:

            self.status_label.config(
                text="CAMERA FRAME ERROR",
                bg="#57272c"
            )

            self.stop_camera()

            return

        result = self.engine.process(frame)

        self.latest_result = result

        self.update_display(result)

        self.root.after(
            20,
            self.update_frame
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    @staticmethod
    def to_photo(image, size):

        image = cv2.resize(
            image,
            size
        )

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        return ImageTk.PhotoImage(
            Image.fromarray(image)
        )

    def update_display(self, result):

        data = result["data"]

        # Main camera
        photo = self.to_photo(
            result["frame"],
            (700, 480)
        )

        self.camera_label.configure(
            image=photo,
            text=""
        )

        self.camera_label.image = photo

        # Analysis images
        image_sources = [
            result["flow_visual"],
            result["bright_mask"],
            result["foreground"]
        ]

        for label, source in zip(
            self.image_labels,
            image_sources
        ):

            photo = self.to_photo(
                source,
                (210, 135)
            )

            label.configure(image=photo)
            label.image = photo

        # Status
        status = data["status"]

        if status == "SMOKE INDICATOR DETECTED":

            color = "#762329"

        elif status == "POSSIBLE SMOKE INDICATOR":

            color = "#704b24"

        elif status == "NORMAL":

            color = "#174b42"

        else:

            color = "#263447"

        self.status_frame.configure(bg=color)

        self.status_label.configure(
            text=status,
            bg=color
        )

        # Metrics
        values = {

            "Brightness": f'{data["brightness"]:.2f}',

            "Brightness variation":
                f'{data["brightness_std"]:.2f}',

            "Bright region coverage":
                f'{data["bright_coverage"] * 100:.2f}%',

            "Foreground coverage":
                f'{data["foreground_coverage"] * 100:.2f}%',

            "Optical flow":
                f'{data["flow_magnitude"]:.3f} px/frame',

            "Flow direction":
                f'{data["flow_direction"]:.1f} deg',

            "Anomaly score":
                f'{data["anomaly_score"]:.1f}/100',

            "Alert persistence":
                f'{self.engine.persistence}/{PERSISTENCE_FRAMES}'
        }

        for name, value in values.items():

            self.metric_labels[name].configure(
                text=value
            )

        # Graph
        graph = self.create_graph()

        photo = self.to_photo(
            graph,
            (330, 250)
        )

        self.graph_label.configure(
            image=photo
        )

        self.graph_label.image = photo

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    def create_graph(self):

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(5, 4),
            dpi=75
        )

        fig.patch.set_facecolor("#101b2c")

        datasets = [

            (
                self.engine.graph_brightness,
                "Brightness",
                "#ffb347"
            ),

            (
                self.engine.graph_flow,
                "Optical flow",
                "#00c8ff"
            ),

            (
                self.engine.graph_anomaly,
                "Anomaly score",
                "#ff554c"
            )
        ]

        for ax, (data, title, color) in zip(
            axes,
            datasets
        ):

            ax.set_facecolor("#19283c")

            ax.plot(
                list(data),
                color=color,
                linewidth=1.5
            )

            ax.set_title(
                title,
                color="white",
                fontsize=8
            )

            ax.tick_params(
                colors="white",
                labelsize=6
            )

            ax.grid(alpha=0.15)

            for spine in ax.spines.values():
                spine.set_color("#34465f")

        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasAgg(fig)

        canvas.draw()

        graph = np.asarray(
            canvas.buffer_rgba()
        )[:, :, :3].copy()

        plt.close(fig)

        return cv2.cvtColor(
            graph,
            cv2.COLOR_RGB2BGR
        )

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    def export_csv(self):

        rows = self.engine.csv_rows

        if not rows:

            self.status_label.configure(
                text="NO MEASUREMENTS TO EXPORT"
            )

            return

        filename = (
            "firewatch_smoke_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        with open(
            filename,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

        self.status_label.configure(
            text=f"EXPORTED {len(rows)} MEASUREMENTS: {filename}"
        )

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def close(self):

        self.running = False

        if self.camera is not None:
            self.camera.release()

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = FirewatchApp(root)

    root.mainloop()
