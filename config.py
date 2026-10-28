"""
config.py
===========================================================================
Central configuration module for the AI-Powered Blind Navigation Assistant.

All tunable parameters live here so the rest of the application stays clean,
modular and easy to customise.
===========================================================================
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Model weights location (configurable via env for cloud deployments)
import os
_MODELS_FALLBACK = BASE_DIR / "models"
_MODELS_EXTERNAL = Path("D:/program/models")
# Allow override via environment variable
_env_models = os.environ.get("MODELS_DIR") or os.environ.get("MODEL_DIR")
if _env_models:
    MODELS_DIR = Path(_env_models)
elif _MODELS_EXTERNAL.exists():
    MODELS_DIR = _MODELS_EXTERNAL
else:
    MODELS_DIR = _MODELS_FALLBACK
INSIGHTFACE_MODELS_ROOT = MODELS_DIR / "insightface"
MODELS_ROOT = MODELS_DIR
# "n" (nano) is ~10x faster than "l" (large) on CPU with minor accuracy loss.
_ONNX_PATH = MODELS_DIR / "yolo11n.onnx"
MODEL_PATH = _ONNX_PATH if _ONNX_PATH.exists() else MODELS_DIR / "yolo11n.pt"
MODEL_DOWNLOAD_NAME = "yolo11n.pt"

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
CAMERA_SET_RESOLUTION = True
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# Requested capture rate. A 60 FPS webcam will run at 60; ordinary laptop
# webcams report 60 but still deliver ~30 FPS (a hardware limit). The web
# stream is no longer capped below the camera rate, so the pipeline runs at
# whatever the camera actually provides.
CAMERA_FPS = 60
# Request the MJPG codec: most webcams only deliver 30 FPS at VGA when MJPG
# is enabled; without it Windows may fall back to a slow uncompressed mode.
CAMERA_FORCE_MJPG = True
MAX_CONSECUTIVE_EMPTY_FRAMES = 30

# ---------------------------------------------------------------------------
# Detection settings (YOLO inference)
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
# Must match the exported ONNX input size (320). The .pt fallback accepts any.
IMG_SIZE = 320

# Run YOLO every Nth frame and reuse the last detections in between.
# 1 = detect on every frame; 2-3 is a good CPU trade-off.
DETECT_EVERY_N_FRAMES = 2

# All 80 COCO classes. A few are aliased for cleaner labels (see CLASS_ALIASES
# in detector.py). Names here must match the POST-ALIAS form.
TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "plant", "bed", "table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair dryer", "toothbrush",
]

# ---------------------------------------------------------------------------
# Distance estimation (monocular, approximate)
# ---------------------------------------------------------------------------

# Camera focal length in pixels.  This value depends on the specific webcam
# and resolution.  Typical webcams at 640x480 have a focal length around 500-600.
# Calibrate by measuring a known-width object at a known distance:
#   focal_length = (object_width_px * known_distance_m) / real_width_m
FOCAL_LENGTH = 500.0

# Real-world reference widths (metres) for distance estimation.
# When the actual width of an object is known, the system estimates:
#   distance = (real_width * focal_length) / bbox_width_px
# For unknown objects the average_reference_width is used as fallback.
AVERAGE_REFERENCE_WIDTH = 0.5

REFERENCE_OBJECT_WIDTHS = {
    "person":         0.50,
    "bicycle":        0.60,
    "car":            1.80,
    "motorcycle":     0.80,
    "bus":            2.50,
    "truck":          2.50,
    "chair":          0.45,
    "couch":          2.00,
    "table":          1.20,
    "bed":            1.50,
    "dining table":   1.20,
    "laptop":         0.35,
    "cell phone":     0.07,
    "tv":             1.00,
    "bottle":         0.08,
    "cup":            0.08,
    "dog":            0.40,
    "cat":            0.30,
    "horse":          0.60,
    "cow":            0.80,
    "elephant":       1.20,
    "bear":           0.80,
    "zebra":          0.60,
    "giraffe":        0.70,
    "refrigerator":   0.80,
    "oven":           0.60,
    "microwave":      0.50,
    "sink":           0.50,
    "toilet":         0.40,
    "skateboard":     0.30,
    "surfboard":      0.60,
    "snowboard":      0.30,
    "frisbee":        0.25,
    "tennis racket":  0.30,
    "baseball bat":   0.07,
    "umbrella":       1.00,
    "handbag":        0.35,
    "suitcase":       0.55,
    "backpack":       0.45,
}

# ---------------------------------------------------------------------------
# Position estimation thresholds (fraction of frame width)
# ---------------------------------------------------------------------------

LEFT_THRESHOLD = 0.33
RIGHT_THRESHOLD = 0.66

# Position smoothing: a detection must remain in the same zone for this many
# consecutive frames before the announced position changes.
POSITION_SMOOTH_FRAMES = 3

# ---------------------------------------------------------------------------
# Obstacle priority thresholds (distance in metres)
# ---------------------------------------------------------------------------

CRITICAL_DISTANCE = 0.8
HIGH_DISTANCE = 1.5
MEDIUM_DISTANCE = 3.0

# Priority weights for different object types.
# Higher weight = more dangerous / important.
OBJECT_TYPE_WEIGHTS = {
    "person":         1.0,
    "car":            0.9,
    "motorcycle":     0.9,
    "bicycle":        0.8,
    "bus":            0.9,
    "truck":          0.9,
    "dog":            0.7,
    "chair":          0.5,
    "couch":          0.5,
    "table":          0.5,
    "bench":          0.5,
    "fire hydrant":   0.4,
    "stop sign":      0.4,
    "parking meter":  0.3,
}
DEFAULT_OBJECT_WEIGHT = 0.3

# Minimum confidence for a detection to be considered for announcements.
ANNOUNCEMENT_CONFIDENCE_MIN = 0.40

# ---------------------------------------------------------------------------
# Navigation logic
# ---------------------------------------------------------------------------

# Minimum time between "path ahead is clear" announcements.
CLEAR_PATH_INTERVAL = 10.0

# When multiple obstacles exist, only the top-N most urgent are announced.
MAX_ANNOUNCED_OBJECTS = 3

# Minimum frames a detection must be seen before it is announced (temporal
# stability filter).
DETECTION_STABILITY_FRAMES = 2

# ---------------------------------------------------------------------------
# Family member recognition (embedding-based face recognition)
#
# This is an OPTIONAL INFORMATION LAYER implemented by family_recognition.py.
# It NEVER replaces YOLO object detection / obstacle avoidance; it only adds
# an identity to the existing "person" announcements when it can do so with
# high confidence. If disabled or unavailable, the project behaves exactly as
# before (plain "person" announcements).
# ---------------------------------------------------------------------------

# Global on/off switch for the whole feature.
FAMILY_RECOGNITION_ENABLED = True

# Directory where family photos + embeddings + metadata are stored locally.
FAMILY_DATA_DIR = "family_data"

# Directory (relative to models/) where the InsightFace models live.
INSIGHTFACE_ROOT = "insightface"

# Minimum cosine similarity for a face to be confidently identified.
# This embedding extractor (ArcFace @ w600k_r50) produces highly separable
# embeddings: genuine matches are typically > 0.55 while strangers are around
# 0.2-0.35. We require a conservative 0.50 to strongly favour "unknown person"
# over a wrong identity. RAISE this (e.g. 0.58) to be even stricter; LOWER it
# (e.g. 0.42) to recognise people more eagerly at the risk of false claims.
# Never use a value that would guess an identity on a borderline match.
FACE_RECOGNITION_THRESHOLD = 0.50

# How often (in pipeline iterations) face recognition runs per PERSON.
# The YOLO loop runs every DETECT_EVERY_N_FRAMES; face recognition is
# additionally subsampled so it runs far less often (cheap on CPU).
# 6 = run every 6 YOLO detections (approx a few FPS for a single person).
FACE_RECOGNITION_INTERVAL = 6

# Identity must be seen consistently for this many recognitions before it
# becomes confident (temporal stability / debouncing).
FACE_RECOGNITION_STABILITY = 2

# Minimum face bounding box side (px) accepted for recognition. Faces smaller
# than this are treated as "too far / too small" and fall back to unknown.
FACE_MIN_SIZE = 24

# Maximum photos a single family member may register.
FAMILY_MAX_PHOTOS = 6

# How the assistant labels a recognised person in the guidance text:
#   "relationship"  -> "Your father is 2 meters ahead."
#   "name"          -> "Rajesh is 2 meters ahead."
#   "name+relation" -> "Your father Rajesh is 2 meters ahead."
# Default is "name" so a recognised member is always shown by name; change
# it on the Settings page (or here) to prefer relationship wording.
FAMILY_ANNOUNCE_MODE = "name"

# Minimum time before the same recognised person is re-announced in the
# guidance text (avoids flicker when identity/distance are unchanged).
FAMILY_ANNOUNCE_COOLDOWN_SECONDS = 4.0

# Standard relationship choices offered in the registration UI.
FAMILY_RELATIONSHIPS = [
    "Father", "Mother", "Brother", "Sister", "Grandfather", "Grandmother",
    "Son", "Daughter", "Husband", "Wife", "Friend",
]

# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------

WINDOW_NAME = "AI Blind Navigation Assistant"

# ---------------------------------------------------------------------------
# Vehicle awareness (provided by vehicle_awareness.py)
# ---------------------------------------------------------------------------

VEHICLE_WARNING_ENABLED = True
VEHICLE_CLASSES = ["car", "motorcycle", "bicycle", "bus", "truck"]
# Distance (m) change across tracked samples that implies meaningful movement.
VEHICLE_APPROACH_DISTANCE_DELTA = 0.6
# Frames of temporal evidence required before claiming a vehicle is moving.
VEHICLE_MOTION_MIN_SAMPLES = 6
# Beyond this distance vehicles are reported but never called "approaching".
VEHICLE_APPROACH_MAX_DISTANCE = 8.0
# Standstill: below this distance a stationary vehicle is a hard warning anyway.
VEHICLE_VERY_CLOSE_DISTANCE = 2.0

# ---------------------------------------------------------------------------
# Low-light mode (provided by utils.low_light.py helpers)
# ---------------------------------------------------------------------------

LOW_LIGHT_MODE = "auto"              # "auto" | "on" | "off"
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 70  # mean gray below this => low light
LOW_LIGHT_ENHANCE_GAMMA = 0.6
LOW_LIGHT_ENHANCE_CLAHE = True       # contrast-limited adaptive histogram equalisation
LOW_LIGHT_REPORT_INTERVAL_SECONDS = 10.0

# ---------------------------------------------------------------------------
# Free-space / walkable-area detection (provided by free_space.py)
# ---------------------------------------------------------------------------

FREE_SPACE_ENABLED = True
FREE_SPACE_COLUMNS = ["LEFT", "CENTER", "RIGHT"]
# These labels, if present near the column base, block that column.
FREE_SPACE_BLOCK_LABELS = [
    "car", "bus", "truck", "motorcycle", "bicycle", "person",
    "chair", "couch", "table", "bed", "refrigerator", "oven",
]
FREE_SPACE_HORIZON_FRACTION = 0.6   # how far down the frame we consider
FREE_SPACE_MIN_OBSTACLE_HEIGHT = 0.25  # min box-height fraction to block

# ---------------------------------------------------------------------------
# Safe-path guidance (provided by path_planner.py)
# ---------------------------------------------------------------------------

SAFE_PATH_ENABLED = True
SAFE_PATH_INTERVAL_SECONDS = 8.0     # re-announce guidance only this often

# ---------------------------------------------------------------------------
# Depth provider abstraction (provided by depth_provider.py)
# ---------------------------------------------------------------------------

DEPTH_ENABLED = True
DEPTH_PROVIDER = "monocular"         # "monocular" | "rgbd"
# Distances from a real depth sensor are preferred over monocular estimates.
DEPTH_FALLBACK_TO_MONOCULAR = True

# ---------------------------------------------------------------------------
# Custom navigation model (provided by navigation_detector.py)
# ---------------------------------------------------------------------------

NAVIGATION_MODEL_ENABLED = True
NAVIGATION_MODEL_PATH = MODELS_DIR / "navigation_model.pt"   # may not exist
NAVIGATION_MODEL_CONFIDENCE = 0.45
NAVIGATION_MODEL_CLASSES = [
    "door", "stairs_up", "stairs_down", "curb", "pothole",
    "crosswalk", "traffic light", "zebra crossing",
]

# ---------------------------------------------------------------------------
# Scene summary (provided by scene_summary.py)
# ---------------------------------------------------------------------------

SCENE_SUMMARY_ENABLED = True
SCENE_SUMMARY_INTERVAL_SECONDS = 45.0   # refresh cycle
SCENE_SUMMARY_MAX_OBJECTS = 5

# ---------------------------------------------------------------------------
# Local environment memory (provided by environment_memory.py)
# ---------------------------------------------------------------------------

ENVIRONMENT_MEMORY_ENABLED = True
ENVIRONMENT_MEMORY_MAX_ITEMS = 50
ENVIRONMENT_MEMORY_TTL_SECONDS = 300.0

# ---------------------------------------------------------------------------
# Emergency SOS (provided by sos.py)
# ---------------------------------------------------------------------------

SOS_ENABLED = True
SOS_CONTACTS_FILE = "sos_contacts.json"
SOS_CONFIRM_TIMEOUT_SECONDS = 15.0
# Delivery methods that require extra hardware/apps report "not configured"
# instead of pretending a message was sent:
#   "log"        - local log file (default, works everywhere)
#   "phone_tel"  - tel: link saved in web settings (manual caller action)
SOS_DELIVERY_METHODS = ["log", "phone_tel"]

# ---------------------------------------------------------------------------
# Event history (provided by history_store.py)
# ---------------------------------------------------------------------------

HISTORY_ENABLED = True
HISTORY_MAX_ENTRIES = 300
HISTORY_MERGE_SECONDS = 1.5           # collapse duplicate events within this time

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

EXIT_KEY_CODE = ord("q")
