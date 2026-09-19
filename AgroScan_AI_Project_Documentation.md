# AgroScan AI — Complete Project Documentation
### (Interview Revision Guide)

---

## 1. One-Line Pitch

> "AgroScan AI is a full-stack plant disease detection system: a YOLO11
> image classification model, fine-tuned on the PlantVillage dataset (38
> crop/disease classes), served through a Flask web app with user accounts,
> a 'gatekeeper' model that rejects non-plant images, scan history, an
> analytics dashboard, and multi-language support."

Two halves to know cold:
1. **The training pipeline** (Colab/Kaggle script) — how the model was built.
2. **The web application** (Flask) — how the model is served to users.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING (offline, once)                 │
│  Kaggle dataset → train/val split → YOLO11n-cls fine-tune    │
│  → best.pt weights → confusion matrix / classification report│
└─────────────────────────────────────────────────────────────┘
                              │
                              │  best.pt (~10MB)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE (Flask web app)                   │
│                                                                │
│  User uploads leaf image                                     │
│        │                                                      │
│        ▼                                                      │
│  Gatekeeper (MobileNetV2, ImageNet weights)                  │
│  "Is this even a plant?" → reject non-plant images            │
│        │ yes                                                  │
│        ▼                                                      │
│  YOLO11n-cls (your fine-tuned model)                          │
│  → predicted class + confidence                               │
│        │                                                      │
│        ▼                                                      │
│  SQLite (users, predictions) ← Flask-Login session            │
│        │                                                      │
│        ▼                                                      │
│  Dashboard (Chart.js) + PDF export (jsPDF) + i18n translation │
└─────────────────────────────────────────────────────────────┘
```

**Why two models instead of one?** This is a good interview talking point.
A single 38-class classifier trained only on plant leaves has no concept of
"not a plant" — if you feed it a photo of a car, it will still confidently
pick one of its 38 known classes because softmax always sums to 1. The
**gatekeeper** (a general-purpose ImageNet model) acts as an out-of-
distribution filter: it checks whether the image's top predicted ImageNet
labels contain plant-related keywords (leaf, tree, vegetable, fruit, etc.)
before ever calling the specialist model. This is a practical, low-cost way
to add basic OOD (out-of-distribution) rejection without training a
dedicated anomaly detector.

---

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Classification model | YOLO11n-cls (Ultralytics) | Nano variant = fast, small, good for a CPU-friendly demo; "-cls" = classification head, not object detection |
| Gatekeeper model | MobileNetV2 (Keras, ImageNet weights) | Lightweight, pretrained, no training needed — used purely for its general-purpose feature/label space |
| Training dataset | PlantVillage (Kaggle) | Standard benchmark dataset for plant disease classification, ~38 classes across 14 crop species |
| Backend | Flask + Flask-Login | Lightweight, simple session-based auth |
| Database | SQLite | Zero-config, file-based — fine for a single-instance demo |
| Frontend | Jinja2 templates + Tailwind (CDN) + Chart.js | Server-rendered HTML, no separate frontend build step |
| PDF export | jsPDF (client-side, pure JS) | Draws the report programmatically from data — no screenshot/html2canvas hack, works in restricted/tunnel environments |
| i18n | deep-translator (Google Translate backend) | On-the-fly translation instead of maintaining static locale files |
| Training environment | Google Colab + Kaggle API | Free GPU (T4), dataset pulled directly via Kaggle API |
| Model export | ONNX | Framework-agnostic format for potential deployment outside PyTorch/Ultralytics |
| Production serving | Gunicorn | WSGI server — Flask's built-in dev server isn't meant for real traffic |

---

## 4. The Training Pipeline — Step by Step

This is the script you'd walk an interviewer through if asked "how did you
train the model?"

### 4.1 Environment setup
- Installs `ultralytics`, `kaggle`, `seaborn`, `matplotlib`, `scikit-learn`.
- Checks for a CUDA GPU (`torch.cuda.is_available()`) and **hard-stops** if
  none is found — training a CNN/classifier on CPU in Colab would take
  hours instead of minutes.
- Pulls Kaggle credentials (`KAGGLE_USERNAME` / `KAGGLE_KEY`) at runtime via
  `input()`/`getpass()` rather than hardcoding them — a basic but important
  credential-hygiene habit worth mentioning.

### 4.2 Data acquisition & preparation
- Downloads the PlantVillage dataset directly from Kaggle
  (`abdallahalidev/plantvillage-dataset`) using the Kaggle CLI.
- The raw dataset ships pre-organized by class folder (e.g.
  `Tomato___Early_blight/`, `Potato___healthy/`), but with **no train/val
  split** — the script creates one itself:
  - For each class folder, shuffles the images and does an **80/20 split**
    (`split = int(len(imgs) * 0.8)`).
  - Moves (not copies) files into `plantvillage_data/train/<class>/` and
    `plantvillage_data/val/<class>/`.
  - This directory layout (`train/<class_name>/*.jpg`, `val/<class_name>/*.jpg`)
    is exactly the format Ultralytics' classification trainer expects —
    no manual dataset YAML/annotation needed, unlike object detection.
- Cleans up the raw zip/intermediate folder afterward to save disk space.

**Interview note:** a random 80/20 split per class (rather than a single
global shuffle) keeps the split **stratified** — every class is
represented proportionally in both train and val, which matters a lot when
class sizes are imbalanced (some PlantVillage classes have far more images
than others).

### 4.3 Training
```python
model = YOLO('yolo11n-cls.pt')
results = model.train(
    data='/content/plantvillage_data',
    epochs=15,
    imgsz=224,
    batch=64,
    workers=4,
    cache=True,
    amp=True,
    device=0
)
```
Know what each argument means:
- **`yolo11n-cls.pt`** — starts from Ultralytics' pretrained YOLO11 "nano"
  classification checkpoint (already trained on ImageNet). This is
  **transfer learning**, not training from scratch — the model already
  knows general visual features (edges, textures, shapes) and only needs
  to adapt those features to distinguish plant diseases.
- **`epochs=15`** — full passes over the training set. Modest number,
  reasonable for fine-tuning (vs. training from scratch, which needs far more).
- **`imgsz=224`** — standard input resolution for lightweight classifiers
  (matches typical ImageNet-pretrained backbones like MobileNet).
- **`batch=64`** — number of images per gradient step; larger batches use
  more GPU memory but make training more stable/faster per epoch on a GPU.
- **`workers=4`** — CPU threads for data loading, so the GPU doesn't sit
  idle waiting for images to be read/decoded/augmented.
- **`cache=True`** — caches decoded images in RAM after the first epoch, so
  subsequent epochs don't re-read from disk — big speed win when disk I/O
  is the bottleneck (common in Colab).
- **`amp=True`** — **Automatic Mixed Precision**: does math in FP16 where
  safe instead of FP32, roughly halving memory use and speeding up
  training on modern GPUs, with negligible accuracy loss.
- **`device=0`** — use GPU index 0 (the T4 Colab assigns).

### 4.4 Evaluation & research artifacts
After training, the script does a full **manual validation pass** (not just
reading Ultralytics' built-in metrics) to generate publication-style
outputs:
- **Training curves**: top-1 / top-5 accuracy over epochs, and train vs.
  validation loss — read from `results.csv`, plotted with Matplotlib.
- **Confusion matrix** (38×38): built by re-running inference on every
  validation image in batches of 64 and comparing predicted vs. true class,
  using `sklearn.metrics.confusion_matrix`, visualized as a Seaborn heatmap.
- **Classification report**: precision/recall/F1-score **per class**, via
  `sklearn.metrics.classification_report` — this matters more than plain
  accuracy for an imbalanced dataset, because it shows whether the model is
  failing silently on rare classes.
- **ONNX export**: `model.export(format='onnx')` — converts the trained
  PyTorch model into a portable format that can run outside PyTorch (e.g.
  in a mobile app, browser via ONNX.js, or a different serving stack).
- Everything gets zipped into `AgroScan_Research_Package.zip` and
  downloaded — this is your "evidence" folder for a research write-up.

### 4.5 Key metrics vocabulary (be ready to define these)
- **Top-1 accuracy**: the model's single best guess is correct.
- **Top-5 accuracy**: the correct class appears anywhere in the model's top
  5 guesses — more forgiving, useful when classes are visually similar.
- **Precision**: of everything the model labeled as class X, what fraction
  actually was X (false-positive control).
- **Recall**: of everything that actually was class X, what fraction did
  the model catch (false-negative control).
- **F1-score**: harmonic mean of precision and recall — a single number
  balancing both.
- **Confusion matrix**: a grid showing exactly which classes get mixed up
  with which — e.g. you'd expect "Tomato Early Blight" and "Tomato Late
  Blight" to be more confusable with each other than with "Apple Scab".

---

## 5. The Web Application — Component by Component

### 5.1 Authentication
- `flask-login` handles session state (`current_user`, `@login_required`).
- Passwords are never stored in plaintext — `werkzeug.security.
  generate_password_hash` / `check_password_hash` (uses salted hashing
  under the hood).
- Users table: `id, username (unique), password_hash`.

### 5.2 Prediction flow (`/predict`)
1. User picks an image source: fresh upload, "reuse last image", or a
   built-in sample leaf.
2. **Gatekeeper check** (`is_valid_plant`): resizes to 224×224, runs
   MobileNetV2, decodes top-5 ImageNet predictions, checks if any decoded
   label contains a plant-related keyword (leaf, tree, fruit, crop, etc.).
   If not → rejected with a friendly error before ever touching the
   specialist model (saves compute and avoids nonsense predictions).
3. **YOLO11 inference**: runs the fine-tuned classifier, takes `probs.top1`
   (predicted class index) and `probs.top1conf` (confidence).
4. **Confidence gating**: predictions under 35% confidence are treated as
   "unsure" and shown as a warning rather than a definitive result —
   guards against confidently-wrong outputs on ambiguous images.
5. Result is saved to the `predictions` table (`user_id, image_path,
   prediction, confidence, time, model_used`) and rendered with a
   description + remedies pulled from a hand-curated `disease_info` dict.

### 5.3 Dashboard & analytics
- Aggregates the logged-in user's own prediction history via SQL
  (`GROUP BY prediction`, `GROUP BY strftime('%Y-%m', time)`) to build:
  - Disease-distribution donut chart
  - Monthly scan-activity bar chart
  - Overall "health score" (% of scans that came back healthy)
  - Recent-scans list
- **PDF export** is drawn **programmatically with jsPDF** (rectangles,
  text, computed bar heights) rather than screenshotting the DOM — more
  reliable across environments and gives full control over layout/branding
  in the exported report.

### 5.4 Internationalization
- `deep-translator`'s `GoogleTranslator` translates UI strings on the fly
  based on the session's chosen language, with an in-memory cache
  (`translation_cache`) so the same string isn't re-translated every
  request.
- Supports 11 languages relevant to Indian agriculture (Hindi, Urdu,
  Bengali, Telugu, Tamil, Marathi, Gujarati, Punjabi, Malayalam, Kannada,
  English) — a deliberate accessibility choice for the target user base
  (farmers who may not be comfortable in English).

### 5.5 Data model (SQLite)
```sql
users (id, username UNIQUE, password_hash)
predictions (id, user_id → users.id, image_path, model_used,
             prediction, confidence, time)
```
Simple, normalized enough for the app's needs — one-to-many between users
and their prediction history.

---

## 6. Design Decisions Worth Defending in an Interview

| Decision | Why it's defensible |
|---|---|
| YOLO11 for classification (not detection) | Task is "what disease is on this leaf", not "where is the disease" — a classification head is simpler, faster, and matches the labeled data (whole-image labels, no bounding boxes) |
| Nano model size | Runs on CPU in production without a GPU — appropriate tradeoff for a demo/small-scale deployment vs. a larger, more accurate but heavier model |
| Separate gatekeeper model instead of a 39th "not a plant" class | Training a robust "other" class needs a huge, diverse negative dataset; reusing an existing general-purpose model is far cheaper and still catches the obvious cases |
| SQLite instead of Postgres/MySQL | Zero ops overhead for a single-instance app; explicitly called out as a limitation for scaling (see below) |
| Client-side PDF generation | No server-side headless-browser/screenshot dependency, works even behind tunnels/restricted network policies |
| On-the-fly translation vs. static locale files | Faster to support 11 languages without manually translating every string; tradeoff is a runtime network dependency and no human-reviewed translation quality |

---

## 7. Known Limitations (good to raise proactively — shows self-awareness)

- **SQLite doesn't handle concurrent writes well** — fine for a demo, would
  need Postgres/MySQL for multiple simultaneous users at scale.
- **Confidence threshold (35%) and gatekeeper keyword list are heuristics**,
  not learned — could misclassify unusual-but-valid plant photos as
  invalid, or vice versa.
- **No adversarial/robustness testing** — the model hasn't been stress
  tested against blurry photos, unusual lighting, or partial leaves beyond
  whatever variation exists naturally in PlantVillage.
- **PlantVillage images are mostly lab-controlled (plain backgrounds)** —
  a well-known dataset criticism is that models trained on it may not
  generalize well to messy real-world field photos. Worth mentioning if
  asked about generalization.
- **Translation quality is unreviewed** (auto-translated, not
  human-verified) — a risk for a farmer-facing tool where mistranslated
  remedy instructions could cause harm.
- **No rate limiting / abuse protection** on uploads or login attempts.

## 8. Possible Improvements (shows forward thinking)

- Fine-tune or ensemble with a model trained on real-world field images
  (e.g. PlantDoc dataset) to close the lab-vs-field generalization gap.
- Replace the keyword-based gatekeeper with a proper OOD-detection method
  (e.g. energy-based scoring, or a small binary "is-a-leaf" classifier
  trained specifically for this).
- Move to Postgres + object storage (S3-style) for real multi-user scale.
- Add human-reviewed translations for the disease/remedy text specifically
  (highest-stakes content), keep auto-translate for UI chrome only.
- Add explainability (e.g. Grad-CAM heatmap overlay) so users can see
  *which part* of the leaf drove the prediction — builds trust.

---

## 9. Likely Interview Questions & How to Answer

**Q: Walk me through what happens when a user uploads an image.**
A: Two-stage inference — gatekeeper first checks it's plant-like, then the
fine-tuned YOLO11 classifier predicts the specific disease class, with a
confidence threshold gating low-certainty results before they're shown or
saved.

**Q: Why YOLO for classification instead of a standard CNN like ResNet?**
A: Ultralytics packages YOLO11 with a ready classification head/training
API that's very fast to fine-tune and export (including ONNX), with
sensible training defaults (AMP, caching, augmentation) out of the box —
practically, it was less setup than writing a custom PyTorch training loop
for a fairly standard task.

**Q: How did you validate the model wasn't overfitting?**
A: Tracked train loss vs. validation loss curves over epochs, and looked at
per-class precision/recall in the classification report rather than only
overall accuracy — overfitting would show as diverging train/val loss or
strong performance on common classes but poor recall on rare ones.

**Q: What would you change for a production deployment vs. this demo?**
A: Swap SQLite for a real database, add proper rate limiting/input
validation, serve behind Gunicorn instead of Flask's dev server, put
uploaded images behind proper storage (not local disk) sized for horizontal
scaling, and replace the heuristic gatekeeper with a trained OOD detector.

**Q: How do you handle a case where the model is unsure?**
A: A confidence threshold (35%) below which the result is treated as
"unsure" rather than shown as fact — better to say "I don't know" than to
confidently mislead a farmer into the wrong treatment.

**Q: What's the accuracy of your model?**
A: *(Fill this in from your actual `classification_report_full.csv` /
`raw_training_metrics.csv` — pull the final epoch's top-1/top-5 accuracy
and overall F1 before the interview so you have real numbers, not
placeholders.)*

---

## 10. Quick-Reference Cheat Sheet

- **Model**: YOLO11n-cls (Ultralytics), transfer-learned from ImageNet weights
- **Dataset**: PlantVillage, 38 classes, 14 crop species
- **Split**: 80/20 train/val, stratified per class
- **Training**: 15 epochs, 224×224 input, batch 64, AMP enabled, Colab T4 GPU
- **Gatekeeper**: MobileNetV2 (ImageNet), keyword-based plant/non-plant filter
- **Confidence floor**: 35% (below this, result shown as "unsure")
- **Backend**: Flask + Flask-Login + SQLite
- **Frontend**: Jinja2 + Tailwind (CDN) + Chart.js
- **Export formats**: `.pt` (PyTorch/Ultralytics), `.onnx`
- **Languages supported**: 11 (English + 10 Indian regional languages)
- **Deployment**: Gunicorn, persistent disk for SQLite + uploads, ~2GB RAM minimum

---

*Fill in your actual final accuracy/F1 numbers from the training run before
the interview — reviewers will often ask for a specific number, and "around
X%, let me check the exact figure" is a much weaker answer than having it
memorized.*
