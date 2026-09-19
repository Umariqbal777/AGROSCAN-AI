# =========================================================================
#  AgroScan AI - PRODUCTION APP (v4.1 - PDF Export Edition)
#  Model: YOLO11 (Ultralytics) | Dataset: PlantVillage (38 Classes)
#  Feature: "Gatekeeper" (MobileNetV2) rejects non-plants
#  UI: Organic dark-luxury aesthetic + Dashboard PDF Export
#
#  Local / VS Code edition — templates live in templates/, no Colab/
#  ngrok/cloudflare tunnel code. Run with: python app.py
# =========================================================================

import os
import time
import sqlite3
from datetime import datetime, timedelta

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageDraw
from deep_translator import GoogleTranslator
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess

# -------------------
# Config
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "database.db")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# Path to your trained YOLO11 classification weights.
# Put your best.pt file in the project root (same folder as app.py),
# or change this path to point at wherever you keep it.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "best.pt"))

MODEL_LOADING = True
SAMPLE_FILENAME = "sample_leaf_ai.png"
YOLO_MODEL = None
GATEKEEPER_MODEL = None

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def format_class_name(name):
    return name.replace('___', ': ').replace('__', ' ').replace('_', ' ').replace('(', '').replace(')', '')


LANGUAGES = {
    'en': "English", 'hi': "हिन्दी", 'ur': "اردو", 'bn': "বাংলা",
    'te': "తెలుగు", 'ta': "தமிழ்", 'mr': "मराठी", 'gu': "ગુજરાતી",
    'pa': "ਪੰਜਾਬੀ", 'ml': "മലയാളം", 'kn': "ಕನ್ನಡ"
}

disease_info = {
    'default': {'description': 'Detailed information pending for this specific variety.', 'remedies': ['Consult local agricultural extension', 'Isolate the plant', 'Ensure proper drainage']},
    'Potato___Early_blight': {'description': 'Fungal disease causing concentric rings ("target spots") on older leaves.', 'remedies': ['Apply mancozeb or chlorothalonil', 'Rotate crops', 'Burn infected debris']},
    'Potato___Late_blight': {'description': 'Serious water-mold disease causing dark lesions and white mold.', 'remedies': ['Apply copper fungicides proactively', 'Destroy infected plants immediately', 'Reduce humidity']},
    'Potato___healthy': {'description': 'Plant is healthy.', 'remedies': []},
    'Tomato___Bacterial_spot': {'description': 'Bacterial infection causing small water-soaked spots.', 'remedies': ['Copper sprays', 'Avoid overhead watering']},
    'Tomato___Early_blight': {'description': 'Common fungus causing bullseye patterns on lower leaves.', 'remedies': ['Mulch soil', 'Prune lower leaves', 'Fungicide application']},
    'Tomato___Late_blight': {'description': 'Aggressive disease causing large grey/brown spots.', 'remedies': ['Remove plant immediately', 'Preventative fungicide']},
    'Tomato___Leaf_Mold': {'description': 'Fungus appearing as olive-green mold on leaf undersides.', 'remedies': ['Increase air circulation', 'Reduce humidity']},
    'Tomato___Septoria_leaf_spot': {'description': 'Small circular spots with dark borders.', 'remedies': ['Remove infected leaves', 'Keep leaves dry']},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'description': 'Tiny pests causing yellow stippling and webbing.', 'remedies': ['Neem oil', 'Insecticidal soap', 'Spray with water']},
    'Tomato___Target_Spot': {'description': 'Fungal disease causing necrotic lesions.', 'remedies': ['Apply fungicides', 'Remove infected debris']},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'description': 'Viral disease spread by whiteflies causing curling.', 'remedies': ['Control whiteflies', 'Use resistant varieties', 'Remove plant']},
    'Tomato___Tomato_mosaic_virus': {'description': 'Virus causing mottling and stunting.', 'remedies': ['Disinfect tools', 'Remove plant', 'Wash hands']},
    'Tomato___healthy': {'description': 'Plant is healthy.', 'remedies': []},
    'Pepper,_bell___Bacterial_spot': {'description': 'Bacterial spots on leaves and fruit.', 'remedies': ['Copper sprays', 'Use disease-free seeds']},
    'Pepper,_bell___healthy': {'description': 'Plant is healthy.', 'remedies': []},
    'Corn_(maize)___Common_rust_': {'description': 'Fungal rust pustules on leaves.', 'remedies': ['Fungicides if severe', 'Plant resistant hybrids']},
    'Corn_(maize)___Northern_Leaf_Blight': {'description': 'Long cigar-shaped grey lesions.', 'remedies': ['Crop rotation', 'Fungicides']},
    'Apple___Apple_scab': {'description': 'Fungal scabs on leaves and fruit.', 'remedies': ['Fungicides', 'Remove fallen leaves in autumn']},
    'Apple___Black_rot': {'description': 'Fungal rot affecting fruit and leaves.', 'remedies': ['Prune infected wood', 'Remove mummified fruit']},
    'Apple___Cedar_apple_rust': {'description': 'Bright orange spots, requires cedar trees nearby.', 'remedies': ['Remove nearby galls', 'Fungicides']},
}

VALID_PLANT_KEYWORDS = [
    'leaf', 'plant', 'vegetable', 'fruit', 'flower', 'tree', 'grass',
    'agriculture', 'pot', 'greenhouse', 'garden', 'cabbage', 'broccoli',
    'cauliflower', 'zucchini', 'cucumber', 'squash', 'pumpkin', 'corn',
    'ear', 'pepper', 'bell_pepper', 'potato', 'tomato', 'hay', 'crop',
    'lettuce', 'spinach', 'produce', 'apple', 'grape', 'orange', 'strawberry'
]

translation_cache = {}


def get_translation(text, target_language):
    if not text or target_language == 'en':
        return text
    key = (text, target_language)
    if key in translation_cache:
        return translation_cache[key]
    try:
        out = GoogleTranslator(source='auto', target=target_language).translate(text)
        translation_cache[key] = out
        return out
    except Exception:
        return text


def tr_route(text):
    lang = session.get('language', 'en')
    return get_translation(text, lang)


# -------------------
# DB helpers
# -------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     password_hash TEXT NOT NULL)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id INTEGER NOT NULL,
                     image_path TEXT NOT NULL,
                     model_used TEXT NOT NULL,
                     prediction TEXT NOT NULL,
                     confidence REAL NOT NULL,
                     time TEXT NOT NULL,
                     FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()


def create_sample_image():
    sample_img = os.path.join(UPLOAD_DIR, SAMPLE_FILENAME)
    if not os.path.exists(sample_img):
        img = Image.new("RGB", (400, 260), (30, 60, 30))
        d = ImageDraw.Draw(img)
        d.text((30, 30), "AgroScan Sample Leaf", fill=(100, 220, 100))
        img.save(sample_img)


# -------------------
# Flask app setup
# -------------------
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET", "agroscan-stable-secret-key-v4")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False


@app.before_request
def make_session_permanent():
    session.permanent = True


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    pass


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    r = cur.fetchone()
    conn.close()
    if r:
        u = User()
        u.id = r[0]
        u.username = r[1]
        return u
    return None


@app.context_processor
def inject_globals():
    def tr(text):
        return tr_route(text)
    return dict(_=tr, languages=LANGUAGES, ACTIVE_MODEL_KEY="YOLO11")


# -------------------
# Models
# -------------------
def load_gatekeeper():
    global GATEKEEPER_MODEL
    print("Loading Gatekeeper (MobileNetV2)...")
    try:
        GATEKEEPER_MODEL = MobileNetV2(weights='imagenet')
        print("Gatekeeper ready.")
    except Exception as e:
        print(f"Gatekeeper failed to load: {e}")


def is_valid_plant(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype("float32")
        arr = np.expand_dims(arr, 0)
        arr = mobile_preprocess(arr)
        preds = GATEKEEPER_MODEL.predict(arr, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]
        for _, label, prob in decoded:
            label = label.lower()
            if any(k in label for k in VALID_PLANT_KEYWORDS):
                return True, label
        return False, decoded[0][1].replace('_', ' ')
    except Exception as e:
        print(f"Gatekeeper warning: {e}")
        return True, "bypass"


def load_yolo_model():
    global YOLO_MODEL, MODEL_LOADING
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        MODEL_LOADING = False
        return
    print(f"Loading YOLO11 model from {MODEL_PATH}...")
    try:
        YOLO_MODEL = YOLO(MODEL_PATH)
        MODEL_LOADING = False
        print("YOLO11 loaded successfully.")
    except Exception as e:
        print(f"Failed to load YOLO: {e}")
        MODEL_LOADING = False


# -------------------
# Routes
# -------------------
@app.route('/language/<lang>', methods=['GET', 'POST'])
def set_language(lang):
    if lang in LANGUAGES:
        session['language'] = lang
        session.modified = True
        print(f"[LANG] Set to: {lang}")
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'ok': True, 'lang': lang, 'label': LANGUAGES.get(lang, lang)})
    return redirect(request.referrer or url_for('home'))


@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template("home_logged_in.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row[2], password):
            u = User()
            u.id = row[0]
            u.username = row[1]
            login_user(u)
            return redirect(url_for("home"))
        flash(tr_route("Invalid username or password."))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password_hash) VALUES (?,?)",
                        (username, generate_password_hash(password)))
            conn.commit()
            conn.close()
            flash(tr_route("Registration successful! Please login."))
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            flash(tr_route("Username already taken."))
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT prediction, COUNT(*) FROM predictions WHERE user_id=? GROUP BY prediction", (current_user.id,))
    disease_rows = cur.fetchall()
    disease_labels = [str(r[0]) for r in disease_rows]
    disease_values = [int(r[1]) for r in disease_rows]

    cur.execute("SELECT strftime('%Y-%m', time) as month, COUNT(*) as count FROM predictions WHERE user_id = ? GROUP BY month ORDER BY month", (current_user.id,))
    monthly_rows = cur.fetchall()
    monthly_labels = [datetime.strptime(row[0], '%Y-%m').strftime('%b %Y') for row in monthly_rows]
    monthly_values = [int(row[1]) for row in monthly_rows]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (current_user.id,))
    total_scans = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND prediction NOT LIKE '%healthy%'", (current_user.id,))
    diseases_found = int(cur.fetchone()[0])

    cur.execute("SELECT AVG(confidence) FROM predictions WHERE user_id=?", (current_user.id,))
    avg_conf_raw = cur.fetchone()[0]
    avg_conf = round(float(avg_conf_raw), 1) if avg_conf_raw else 0.0

    cur.execute("SELECT prediction, confidence, time FROM predictions WHERE user_id=? ORDER BY time DESC LIMIT 5", (current_user.id,))
    recent_rows = cur.fetchall()
    conn.close()

    recent_scans = [{"prediction": str(r[0]), "confidence": int(round(r[1])), "time": str(r[2])} for r in recent_rows]
    healthy_count = total_scans - diseases_found
    has_data = len(disease_labels) > 0

    return render_template(
        "dashboard.html",
        disease_labels=disease_labels,
        disease_values=disease_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        total_scans=total_scans,
        diseases_found=diseases_found,
        avg_conf=avg_conf,
        recent_scans=recent_scans,
        healthy_count=healthy_count,
        has_data=has_data,
        active_model="YOLO11")


@app.route('/switch_model', methods=['POST'])
@login_required
def switch_model():
    flash(tr_route("Only YOLO11 is available in this version."))
    return redirect(url_for('dashboard'))


@app.route("/history")
@login_required
def history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT image_path, prediction, confidence, time, model_used FROM predictions WHERE user_id=? ORDER BY time DESC", (current_user.id,))
    rows = cur.fetchall()
    conn.close()
    hist = []
    ist_offset = timedelta(hours=5, minutes=30)
    for img_url, pred, conf, ts, model_used in rows:
        try:
            ts_dt = datetime.fromisoformat(ts)
        except Exception:
            ts_dt = datetime.strptime(ts.split('.')[0], "%Y-%m-%d %H:%M:%S")
        ist = ts_dt + ist_offset
        is_healthy = 'healthy' in pred.lower()
        hist.append({
            "image_path": img_url,
            "prediction": pred,
            "confidence": int(round(conf)),
            "timestamp": ist.strftime("%b %d, %Y %I:%M %p"),
            "model_used": model_used,
            "is_healthy": is_healthy
        })
    return render_template("history.html", history=hist)


@app.route("/metrics")
@login_required
def metrics():
    return render_template('metrics.html', training_results=None, confusion_exists=False,
                            report_exists=False, report_text="Metrics available in Research Package.")


@app.route("/team")
def team():
    team_members = [
        {"name": "Umar Iqbal", "role": "Project Leader", "contributions": tr_route("Lead Developer"), "img": url_for('static', filename='umar.png'), "emoji": "👨‍💻"},
        {"name": "Shruti Bajpai", "role": "Research Lead", "contributions": tr_route("Documentation"), "img": url_for('static', filename='shruti.png'), "emoji": "👩‍🔬"},
        {"name": "Sudhanshu Tiwari", "role": "Data Lead", "contributions": tr_route("Preprocessing"), "img": url_for('static', filename='sudhanshu.png'), "emoji": "👨‍🔬"},
        {"name": "Vaishnavi Singh", "role": "Support", "contributions": tr_route("Presentation"), "img": url_for('static', filename='vaishnavi.png'), "emoji": "👩‍🎨"}
    ]
    return render_template('team.html', team_members=team_members,
                            project_summary="AgroScan AI: YOLO11 Edition — 38 Crop Classes", title='Our Team')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_page():
    global MODEL_LOADING, YOLO_MODEL
    last_uploaded_image_url = session.get('last_uploaded_image_url')
    if YOLO_MODEL is None:
        if MODEL_LOADING:
            return render_template("predict.html", model_loading=True,
                                    last_uploaded_image_url=last_uploaded_image_url,
                                    sample_filename=SAMPLE_FILENAME)
        else:
            flash(tr_route("Model failed to load. Check server logs."))
            return render_template("predict.html", model_loading=False,
                                    last_uploaded_image_url=last_uploaded_image_url,
                                    sample_filename=SAMPLE_FILENAME)

    if request.method == "POST":
        file_to_analyze = None
        current_image_url = None
        source_type = request.form.get('source', 'upload')

        if source_type == 'reuse' and last_uploaded_image_url:
            filename = os.path.basename(last_uploaded_image_url).split('?')[0]
            file_to_analyze = os.path.join(UPLOAD_DIR, filename)
            current_image_url = last_uploaded_image_url
            if not os.path.exists(file_to_analyze):
                flash(tr_route("File not found."))
                return redirect(url_for('predict_page'))
        elif source_type == 'sample':
            file_to_analyze = os.path.join(UPLOAD_DIR, SAMPLE_FILENAME)
            current_image_url = url_for('static', filename=f'uploads/{SAMPLE_FILENAME}')
        elif source_type == 'upload':
            if "file" not in request.files or request.files["file"].filename == "":
                flash(tr_route("No image uploaded."))
                return redirect(url_for('predict_page'))
            file = request.files["file"]
            if not allowed_file(file.filename):
                flash(tr_route("Invalid file type."))
                return redirect(url_for('predict_page'))
            fname = secure_filename(file.filename)
            unique_fname = f"{int(time.time())}_{fname}"
            file_to_analyze = os.path.join(UPLOAD_DIR, unique_fname)
            file.save(file_to_analyze)
            current_image_url = url_for('static', filename=f'uploads/{unique_fname}')

        try:
            is_plant, detected_object = is_valid_plant(file_to_analyze)
            if not is_plant:
                flash(tr_route(f"⚠️ Invalid: Looks like '{detected_object}', not a crop."))
                return render_template("predict.html", model_loading=False,
                                        last_uploaded_image_url=current_image_url,
                                        sample_filename=SAMPLE_FILENAME)

            results = YOLO_MODEL(file_to_analyze)
            top1_idx = results[0].probs.top1
            conf = results[0].probs.top1conf.item() * 100
            raw_name = results[0].names[top1_idx]
            pred_name = format_class_name(raw_name)

            if conf < 35:
                flash(tr_route(f"⚠️ Low Confidence ({int(conf)}%): Unsure about result."))
                return render_template("predict.html", model_loading=False,
                                        last_uploaded_image_url=current_image_url,
                                        sample_filename=SAMPLE_FILENAME)

            session['last_uploaded_image_url'] = current_image_url
            timestamp = datetime.utcnow().isoformat()
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions (user_id, image_path, prediction, confidence, time, model_used) VALUES (?,?,?,?,?,?)",
                (current_user.id, current_image_url, pred_name, conf, timestamp, "YOLO11"))
            conn.commit()
            conn.close()

            info = disease_info.get(raw_name, disease_info['default'])
            lang = session.get('language', 'en')
            desc = get_translation(info.get('description', ''), lang)
            remedies = [get_translation(r, lang) for r in info.get('remedies', [])]
            conf_level = 'High' if conf >= 85 else 'Medium' if conf >= 60 else 'Low'

            return render_template("result.html", prediction=pred_name, confidence=round(conf, 2),
                                    image_path=current_image_url, description=desc, remedies=remedies,
                                    confidence_level=conf_level)
        except Exception as e:
            flash(tr_route(f"Error: {e}"))
            return render_template("predict.html", model_loading=False,
                                    last_uploaded_image_url=last_uploaded_image_url,
                                    sample_filename=SAMPLE_FILENAME)

    return render_template("predict.html", model_loading=False,
                            last_uploaded_image_url=last_uploaded_image_url,
                            sample_filename=SAMPLE_FILENAME)


# -------------------
# Startup
# -------------------
def bootstrap():
    """Run once at process start: DB, sample image, and background model loads."""
    init_db()
    create_sample_image()
    load_gatekeeper()


if __name__ == "__main__":
    bootstrap()

    import threading
    threading.Thread(target=load_yolo_model, daemon=True).start()

    print("AgroScan AI v4.1 starting on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
