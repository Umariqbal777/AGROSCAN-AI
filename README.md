# AgroScan AI (VS Code / Local Edition)

This is your original single-cell Colab script, split into a normal Flask
project layout so it runs cleanly in VS Code instead of a notebook.

## What changed vs. your original script

- **Split into files.** Python logic → `app.py`. Each `..._HTML` string is now
  its own file in `templates/`, exactly as it was written to disk at runtime
  in your script.
- **Removed Colab-only bits.** `pyngrok`, the `!pip install` cell, and the
  `cloudflared` tunnel subprocess are gone — VS Code just runs the Flask dev
  server directly on `http://127.0.0.1:5000`. (You can still deploy behind a
  tunnel/reverse proxy later if you want a public URL; the app itself doesn't
  care.)
- **Model path.** `MODEL_PATH` now reads from an env var
  (`MODEL_PATH=/path/to/best.pt`) and falls back to `best.pt` in the project
  root — same default as before.
- Everything else (routes, DB schema, gatekeeper logic, PDF export JS,
  translations, templates) is unchanged.

## Project structure

```
agroscan/
├── app.py                 # Flask app, routes, model loading
├── requirements.txt
├── database.db             # created automatically on first run
├── best.pt                 # ← YOU add your trained YOLO11 weights here
├── templates/
│   ├── layout.html
│   ├── login.html
│   ├── register.html
│   ├── home_logged_in.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── metrics.html
│   └── team.html
└── static/
    ├── uploads/             # uploaded + sample leaf images
    └── umar.png, shruti.png, sudhanshu.png, vaishnavi.png   # team photos (placeholders)
```

## 1. Prerequisites

- Python 3.10 or 3.11 (TensorFlow + Ultralytics are picky about very new
  Python versions — 3.10/3.11 is the safest bet).
- VS Code with the **Python** extension installed.
- Your trained model weights file, `best.pt` (YOLO11 classification model
  trained on PlantVillage).

## 2. Open the project in VS Code

1. Unzip/copy the `agroscan/` folder onto your machine.
2. In VS Code: **File → Open Folder…** → select `agroscan/`.

## 3. Create a virtual environment

Open a terminal in VS Code (`` Ctrl+` ``) and run:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

VS Code should pop up a prompt asking "Select this environment for the
workspace?" — click **Yes**. Otherwise, select it manually: `Ctrl+Shift+P` →
**Python: Select Interpreter** → pick the `venv` one.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, Flask-Login, TensorFlow, Pillow, deep-translator,
Ultralytics and NumPy. First install can take a few minutes (TensorFlow is
large).

## 5. Add your model weights

Copy your trained `best.pt` file into the project root, next to `app.py`:

```
agroscan/best.pt
```

If you keep it somewhere else, set an environment variable instead:

```bash
# macOS/Linux
export MODEL_PATH=/full/path/to/best.pt
# Windows PowerShell
$env:MODEL_PATH="C:\full\path\to\best.pt"
```

Without this file, the app still starts, but the `/predict` page will show
"Model failed to load" (same behavior as your original script).

## 6. Run it

From the VS Code terminal (with `venv` active):

```bash
python app.py
```

You should see something like:

```
Loading Gatekeeper (MobileNetV2)...
Gatekeeper ready.
AgroScan AI v4.1 starting on http://127.0.0.1:5000 ...
Loading YOLO11 model from .../best.pt...
YOLO11 loaded successfully.
```

Then open **http://127.0.0.1:5000** in your browser. Register an account,
log in, and try **Analyze → Try with sample leaf** or upload your own image.

Alternatively, press **F5** in VS Code (Run → Start Debugging) after adding
a small `launch.json` (VS Code will offer to create one for "Python File"
automatically the first time you hit F5 with `app.py` open).

## 7. Stopping the server

`Ctrl+C` in the terminal.

## Notes / gotchas

- **`database.db`** is created automatically on first run in the project
  root — no setup needed.
- **Team photos** (`static/umar.png`, etc.) are empty placeholder files right
  now — drop real images in with the same filenames, or edit the `team()`
  route in `app.py` if you want different names.
- **Translation** uses `deep-translator`'s Google backend, which calls out to
  the internet — if you're offline, translated strings will silently fall
  back to English.
- If TensorFlow install fails on your Python version, install Python 3.10/3.11
  in a separate environment (e.g. via `pyenv` or the official installer) and
  redo step 3 with that interpreter.
- Debug mode (`debug=True`) is on for convenience (auto-reloads on code
  changes, shows tracebacks in-browser). Turn it off for anything
  resembling production use.
