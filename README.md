# ============================================================================
# BlindNav AI - Intelligent Navigation Assistant for Visually Impaired
# ============================================================================

An assistive application that opens the webcam, detects objects in real time
with a pretrained **YOLO11n** model, estimates their distance, and shows clear
on-screen navigation guidance. It also recognises registered family members,
tracks approaching vehicles, estimates walkable free space / safe paths,
enhances low-light frames, supports a custom navigation model, builds scene
summaries, remembers recently-seen places, provides an explicit SOS flow and
an event history.

Everything runs **offline on this device**. Every optional feature reports
its real state ("MODEL NOT INSTALLED", honest delivery results) instead of
fabricating output, and a critical obstacle warning always has priority over
safe-path guidance. Guidance is visual only - shown on the live camera feed
and the dashboard.

## Features

- Real-time webcam detection (YOLO11n ONNX / PyTorch, COCO classes).
- Throttled, deduplicated guidance with a priority ladder
  (CRITICAL > HIGH > MEDIUM > LOW). Critical warnings always surface first.
- Family recognition: register members from photos, identify by
  relationship/name, never blocks obstacle warnings.
- **Safe path** - 3-column walkable-area estimate and guidance.
- **Vehicle awareness** - temporal approach detection ("vehicle approaching").
- **Low light** - brightness measurement + gamma/CLAHE enhancement.
- **Depth** - monocular focal-length estimation, RGB-D ready abstraction.
- **Scene summaries** and environment memory of recently-seen places.
- **Emergency SOS** - explicit arm->confirm with location, local log + `tel:`
  shortcut, honest "delivered" reporting.
- **History** - ring-buffer of navigation/path/family/SOS events.

## Project structure

```
BlindNavigator/
├── server.py             # Web server + streaming pipeline (hub)
├── main.py               # Desktop OpenCV-window entry point
├── detector.py           # YOLO loading/inference/filtering
├── obstacle_manager.py   # Tracking, distance, priority
├── navigation.py         # Guidance decision engine
├── family_recognition.py # Face registration + recognition (insightface)
├── vehicle_awareness.py  # Approaching-vehicle detection
├── free_space.py         # Walkable column analysis
├── path_planner.py       # Safe-path guidance
├── depth_provider.py     # Monocular / (future) RGB-D abstraction
├── navigation_detector.py# Optional custom navigation model (graceful)
├── scene_summary.py      # Scene descriptions
├── environment_memory.py # Recently-seen place memory
├── sos.py                # Confirmed local SOS delivery
├── history_store.py      # Event ring buffer
├── config.py             # All tunable settings
├── utils.py              # Drawing, FPS, brightness helpers
├── static/               # Web UI (HTML/JS/CSS)
├── models/               # Local model weights (yolo11n, insightface)
├── requirements.txt      # Core + optional dependencies
├── run_web.bat           # One-click start (sets PYTHONPATH, opens browser)
└── README.md
```

Model weights live **outside the project** at `D:\program\models` by default
(see `config.MODELS_DIR`, falls back to `<project>/models`), keeping the
project small. Insightface models load from `config.INSIGHTFACE_MODELS_ROOT`.

## Requirements

- Windows 10/11
- Python **3.10+** (3.10.11 used in development)
- A working webcam

## Setup

```powershell
cd BlindNavigator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keep the project small (recommended): install into a shared folder with
`--target`, then run with `PYTHONPATH`:

```powershell
python -m pip install --target D:\program\pylibs -r requirements.txt
$env:PYTHONPATH = "D:\program\pylibs"
python server.py
# or just double-click run_web.bat  ->  http://127.0.0.1:8000
```

## Web UI

Pages: **Dashboard**, **Family**, **History**, **Settings**, **Emergency**,
**About**.

API (REST): `/api/status`, `/api/control`, `/api/settings`, `/api/cameras`,
`/api/history`, `/api/navigation/path`, `/api/depth/status`,
`/api/scene/summary`, `/api/low-light`, `/api/family/*`, `/api/sos/*`.
Live frame stream: `/ws/stream`.

## Optional dependencies

| Feature               | Backend                          | Without it                          |
|-----------------------|----------------------------------|-------------------------------------|
| Custom nav model      | `.pt` file in `models/`          | "MODEL NOT INSTALLED"               |

Reportable honesty: nothing is ever "sent" unless a delivery method really
completed, and absent optional components are reported rather than faked.

## Configuration

All parameters live in `config.py`, organised by area (detection, family,
vehicle awareness, low light, free space, safe path, depth, navigation model,
scene summary, environment memory, SOS, history).

## Troubleshooting

- **"Could not open camera"** - webcam is in use elsewhere; close Teams/Zoom
  or change the camera index on the Settings page (or `CAMERA_INDEX`).
- **Slow FPS** - CPU-only torch build; lower capture size in `config.py`.
- **Model download fails** - rerun after restoring internet; weights land in
  `models\` (or `D:\program\models`) after the first success.

## Roadmap

- Custom fine-tuned navigation model (`door`, stairs, etc.) drop-in.
- RGB-D depth provider over the existing abstraction.
- Automatic SMS/email alert delivery in the SOS flow (PAWS/OTP providers).
