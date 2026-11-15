"""All user-tunable settings. Edit this file, not the inference logic.

Drop-in layout: this script, my_model.pt, depth_anything_v2_vits.pth and the
depth_anything_v2/ package all live in the same folder."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _first_existing(candidates, default):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(default)


# ----------------------------------------------------------------- YOLO ----
# Same as: --model my_model.pt --thresh 0.1
YOLO_MODEL_PATH = _first_existing([BASE_DIR / "my_model.pt", Path.cwd() / "my_model.pt",
                                   BASE_DIR / "models" / "my_model.pt"],
                                  BASE_DIR / "my_model.pt")
YOLO_CONF_THRESHOLD = 0.3
# Class names (from model.names) that count as flame. None = every class.
# If your model also has a "smoke" class, set this, e.g. ["fire"].
YOLO_FIRE_CLASSES = None
YOLO_DEVICE = None            # None = Ultralytics default; or "cpu", "0", "mps"

# --------------------------------------------------------------- Camera ----
# Same as: --source usb0 --resolution 1280x720
CAMERA_INDEX = 0
CAMERA_BACKEND = "v4l2"
CAMERA_FOURCC = "MJPG"        # None = camera default format
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Camera intrinsics (pixels). Leave FX = None to use the APPROXIMATE
# horizontal-field-of-view estimate below. For real values, calibrate your
# webcam with OpenCV (checkerboard) and enter them here, or put them in a JSON
# file {"fx":..,"fy":..,"cx":..,"cy":..,"width":..,"height":..}.
CAMERA_FX = None
CAMERA_FY = None              # None -> same as FX
CAMERA_CX = None              # None -> centre of CAMERA_CALIB_RESOLUTION
CAMERA_CY = None
CAMERA_CALIB_RESOLUTION = (1280, 720)  # resolution the intrinsics refer to
CAMERA_INTRINSICS_FILE = None          # optional JSON path
CAMERA_HFOV_DEG = 65.0                 # used only for the approximate fallback

# ---------------------------------------------------------- Depth model ----
# "relative": standard Depth Anything V2 checkpoint (needs calibration below)
# "metric"  : a Depth Anything V2 *metric* checkpoint (output already in metres)
DEPTH_MODE = "relative"
DEPTH_ENCODER = "vits"
DEPTH_CHECKPOINT = _first_existing([BASE_DIR / "depth_anything_v2_vits.pth",
                                    Path.cwd() / "depth_anything_v2_vits.pth",
                                    BASE_DIR / "models" / "depth_anything_v2_vits.pth",
                                    BASE_DIR / "Depth-Anything-V2" / "depth_anything_v2_vits.pth"],
                                   BASE_DIR / "depth_anything_v2_vits.pth")
# Folder that contains the depth_anything_v2/ package (this folder, or a cloned subfolder)
DEPTH_REPO_DIR = _first_existing(
    [p for p in (BASE_DIR, BASE_DIR / "Depth-Anything-V2") if (p / "depth_anything_v2").is_dir()],
    BASE_DIR)
DEPTH_DEVICE = "auto"         # "auto" = CUDA -> MPS -> CPU (see note in depth_estimator.py)
DEPTH_INPUT_SIZE = 518
DEPTH_METRIC_MAX = 20.0       # max_depth the metric checkpoint was trained with (20 indoor, 80 outdoor)

# Depth extraction from the flame bounding box
DEPTH_ROI_SHRINK = 0.6            # use the central 60% of the box (corners are mostly background)
DEPTH_NEAR_KEEP_FRACTION = 0.6    # keep the nearest 60% of ROI pixels, then take the median
DEPTH_MIN_VALID_PIXELS = 25

# --------------------------------------------------- Depth trigger logic ----
DEPTH_COOLDOWN_S = 5.0
DEPTH_REFRESH_S = 5.0
DEPTH_MOVE_THRESHOLD_PX = float("inf")   # never re-measure just because the flame moved
DEPTH_SIZE_CHANGE_RATIO = float("inf")
DEPTH_NEW_FLAME_AFTER_ABSENT_FRAMES = 15   # absence this long => next detection is a "new flame"

# ---------------------------------------------------------- Calibration ----
CALIBRATION_FILE = BASE_DIR / "calibration.json"
# Used only if CALIBRATION_FILE doesn't exist yet: [(relative_depth, metres), ...]
CALIBRATION_POINTS = []
# Uncalibrated display only: Z_arbitrary = UNCALIBRATED_SCALE / relative_depth.
# These are arbitrary units "u", NOT metres.
UNCALIBRATED_SCALE = 1.0

# ---------------------------------------------------------------- Flame ----
FLAME_DEPTH_RATIO = 1.0           # ellipsoid: c = a * FLAME_DEPTH_RATIO

# -------------------------------------------------------------- Display ----
WINDOW_TITLE = "Real-time 3D fire localisation"
FIRE_LOST_TIMEOUT_S = 1.0         # no detection for this long => empty 3D scene
SCENE_MIN_EXTENT = 0.5            # smallest half-width of the 3D grid (m or u)
FLAME_VISUAL_SCALE = 1.0          # 1.0 = drawn at true estimated size
FLAME_FLICKER = True
SHOW_DEPTH_PREVIEW = True
SNAPSHOT_HOLD_S = 2.0