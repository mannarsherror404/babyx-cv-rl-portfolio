# BabyX — computer vision and feedback prototype

This folder contains one **local macOS/Python 3.11 app**. The CustomTkinter GUI (`baby_gui.py`) combines camera-based face, hand, and object detection with chat, taught object names, and feedback.

The refreshed interface opens before camera/model initialization finishes. Its left panel shows BabyX's current focus and teaching action; the conversation and feedback controls sit on the right. If camera access fails, the app stays open and offers **Reconnect camera**.

The feedback policy stores scores in `rl_policy.json`, but the GUI's reply generator currently uses fixed reply rules; feedback does **not** yet change which reply is chosen. No model training is required to run this app.

## Run the main GUI

Use a Mac with Python 3.11, a camera, and Camera permission for the app launching the script. Double-click `run.command`. On first launch, it runs `setup.command` automatically if needed; dependency installation can download over 1 GB and take several minutes. The environment lives in `~/Library/Application Support/BabyX-CV-RL/venv`, outside this project folder. Alternatively, from Terminal:

```sh
cd "/path/to/4 | CV + RL"
./run.command
```

The existing `yolov8s-world.pt` is used when present. If it is absent, Ultralytics may need internet access to obtain the weight on first use. The app will report a camera error if macOS has not granted access.

## Project map

```text
4 | CV + RL/
├── baby_gui.py            Main desktop GUI
├── baby_gui-1.py          Previous GUI, retained as a local fallback
├── baby_vision.py         Camera and detections for the GUI
├── baby_brain.py          Rule-based chat replies and feedback context
├── memory.py              Taught object names (JSON)
├── object_store.py        Local visual features and crops
├── rl_agent.py            Feedback score storage and action selector
├── assets/                Character sprites and local profile picture
├── tests/                 Hardware-free core tests
├── requirements.txt       Main GUI dependencies
├── setup.command          Main GUI setup
└── run.command            Main GUI launcher
```

## Check without camera or microphone

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

This checks memory and feedback logic only. A real launch of the GUI, camera detection, and model inference still needs to be checked manually on a machine with Camera permission.

## GitHub and privacy

The `.gitignore` keeps local virtual environments, caches, model weights, taught memories, feedback, captured object crops, and `assets/me_dp.png` out of a future repository. The files remain on this Mac. Review the remaining sprites and any personal content before publishing. The Finder `Icon` file has not been removed.

The separate legacy `baby_x` experiment was moved out of this active project to a local archive. The generated project `.venv` was removed during cleanup; `run.command` recreates the required environment outside the project. The root `yolov8s-world.pt` was kept. Model weights are excluded from Git by default, so a fresh clone will need first-run download or a separately supplied weight. Do not load `.pkl` memory files from untrusted sources.

No Git repository, remote, or upload is created by these scripts.
