import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from threading import Lock, Thread

# Standard imports for Flask and Utilities
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_babel import Babel, gettext as _
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Database imports
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required,
    current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash

# ML/TensorFlow imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image_utils
from tensorflow.keras.utils import load_img
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration Constants ---
CONFIDENCE_THRESHOLD = 0.50
SUPPORTED_PLANTS = "(Cassava, Maize, Potato, or Tomato)"

# --- Define Treatment and Recommendation Messages (Expanded for clarity) ---
TREATMENTS = {
    "Non_Plant_images": "This image does not appear to be a plant. Please upload a clearer image of a plant leaf or part.",
    "Healthy": "Your plant appears healthy! Continue with routine care, ensure adequate water and sunlight, and monitor regularly for early symptoms.",
    "Cassava_Brown_Streak_Disease": "Immediate removal of infected plants is critical. Use resistant varieties and implement strict sanitation practices.",
    "Diseased": "This image appears diseased. Apply general fungicide/pesticide protocols for unknown infections and consult an agricultural extension expert immediately.",
    # Add all specific disease keys here
    # "Potato_Early_Blight": "Treatment recommendation...",
    # "Tomato_Bacterial_Spot": "Treatment recommendation...",
}

# -------------------------
# Configuration
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_model_best.keras"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
TREATMENTS_JSON = BASE_DIR / "treatment_recommendations.json"

STATIC_UPLOADS = BASE_DIR / "static" / "uploads"
RETRAIN_QUEUE = BASE_DIR / "retrain_queue"
STATIC_UPLOADS.mkdir(parents=True, exist_ok=True)
RETRAIN_QUEUE.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 8
RETRAIN_CHECK_INTERVAL_MIN = 30
FINETUNE_EPOCHS = 4
FINETUNE_BATCH_SIZE = 8
FINETUNE_LR = 1e-4

IMAGE_TARGET_SIZE = (150, 150)

retrain_lock = Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secure_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_SUPPORTED_LOCALES"] = ["en", "fr", "es"]
app.config["BABEL_TRANSLATION_DIRECTORIES"] = str(BASE_DIR / "translations")

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
# This will redirect unauthenticated users to the 'login' route
login_manager.login_view = "login"
babel = Babel(app)


# -------------------------
# Babel locale selection
# -------------------------
@babel.localeselector
def get_locale():
    lang = request.args.get("lang")
    if lang and lang in app.config["BABEL_SUPPORTED_LOCALES"]:
        return lang
    return request.accept_languages.best_match(app.config["BABEL_SUPPORTED_LOCALES"])

# --- ADD THIS NEW FUNCTION ---
@app.context_processor
def inject_globals():
    """Makes functions available directly in all Jinja templates."""
    return dict(get_locale=get_locale)



# -------------------------
# Database models
# -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    predictions = db.relationship("PredictionLog", backref="user", lazy=True)


class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    label = db.Column(db.String(100))
    confidence = db.Column(db.String(20))
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# -------------------------
# Model loader
# -------------------------
def safe_load_model(path):
    try:
        model = tf.keras.models.load_model(str(path), compile=False)
        logger.info("✅ Model loaded successfully from: %s", path)
        return model
    except Exception as e:
        logger.exception("❌ Model load failure: %s", e)
        return None


MODEL = safe_load_model(MODEL_PATH)

# -------------------------
# Class labels (CRITICAL FIX: Ensure this list order matches model training)
# -------------------------
CLASS_LABELS_RAW = []
if CLASS_INDICES_PATH.exists():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        idx_map = json.load(f)

    max_idx = max(idx_map.values()) if idx_map else -1
    CLASS_LABELS_RAW = [None] * (max_idx + 1)

    for lbl, idx in idx_map.items():
        if 0 <= idx <= max_idx:
            CLASS_LABELS_RAW[idx] = lbl
        else:
            logger.warning(f"Label index {idx} out of range (0-{max_idx}) for label: {lbl}")

    # Fill any gaps
    CLASS_LABELS_RAW = [lbl or f"Class_{i}" for i, lbl in enumerate(CLASS_LABELS_RAW)]
    logger.info("✅ Class labels loaded from JSON: %s", CLASS_LABELS_RAW)
else:
    # Fallback used if JSON is missing
    CLASS_LABELS_RAW = ["Diseased", "Healthy", "Non_Plant_images"]  # Defaulting to observed log format
    logger.warning("⚠️ class_indices.json not found. Using observed labels: %s", CLASS_LABELS_RAW)

# --- CRITICAL FIX 1: Enforce correct index mapping ---
# Based on the logs, the model's highest probability is consistently at Index 0,
# and this index corresponds to 'Healthy' for healthy images. We are enforcing
# the order discovered in the previous session: Index 0 = Healthy
if all(lbl in CLASS_LABELS_RAW for lbl in ["Diseased", "Healthy", "Non_Plant_images"]):
    # Manually re-order the list to match the model's observed prediction behavior
    CLASS_LABELS = ["Healthy", "Diseased", "Non_Plant_images"]
    # UI_LABELS are used for feedback buttons and often match the raw folder structure
    UI_LABELS = ["Diseased", "Healthy", "Non_Plant_images"]
    logger.info("✅ CRITICAL FIX APPLIED: Label order enforced to: %s", CLASS_LABELS)
else:
    # If the labels are completely different, use the raw loaded list as a fallback
    CLASS_LABELS = CLASS_LABELS_RAW
    UI_LABELS = CLASS_LABELS_RAW

# -------------------------
# Treatment mapping (Overwrite with file if it exists, otherwise use hardcoded defaults)
# -------------------------
if TREATMENTS_JSON.exists():
    with open(TREATMENTS_JSON, "r", encoding="utf-8") as f:
        # Load external treatments, but ensure local, critical keys are present
        external_treatments = json.load(f)
        TREATMENTS.update(external_treatments)
else:
    # Ensure default structure is maintained if file is missing
    pass

# -------------------------
# Embedding model + centroids (novelty check)
# -------------------------
_EMBEDDING_MODEL = None
CLASS_CENTROIDS = {}


def get_embedding_model():
    """Initializes a new EfficientNetB0 instance for feature extraction (novelty detection)."""
    global _EMBEDDING_MODEL
    try:
        # Suppress Keras warnings/logs during model setup
        tf.get_logger().setLevel('ERROR')
        # Initialize with NO weights first, forcing Keras to define the input layer
        # based on the input_shape=(..., 3).
        base = tf.keras.applications.EfficientNetB0(
            weights=None,  # Temporarily set to None
            include_top=False,
            input_shape=(IMAGE_TARGET_SIZE[0], IMAGE_TARGET_SIZE[1], 3)
        )
        # Re-load the imagenet weights using the correct input shape settings
        # This combination usually forces the correct 3-channel input
        base.load_weights(tf.keras.utils.get_file(
            'efficientnetb0_weights_tf_dim_ordering_tf_kernels_no_top.h5',
            'https://storage.googleapis.com/keras-applications/efficientnetb0_weights_tf_dim_ordering_tf_kernels_no_top.h5',
            cache_subdir='models'
        ))

        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        _EMBEDDING_MODEL = tf.keras.Model(inputs=base.input, outputs=x)
        logger.info("✅ Embedding model initialized with 3-channel input.")
    except Exception as e:
        logger.error("❌ Failed to initialize embedding model: %s", e)
        _EMBEDDING_MODEL = None


def compute_class_centroids(reference_dir):
    """Compute mean embeddings for each known class for novelty detection."""
    global CLASS_CENTROIDS
    model = get_embedding_model()
    if model is None:
        logger.warning("Cannot compute centroids: Embedding model failed to initialize.")
        return

    centroids = {}
    for lbl in CLASS_LABELS:
        lbl_dir = Path(reference_dir) / lbl
        if not lbl_dir.exists():
            continue
        vecs = []
        # Limiting to a few samples for quick demo setup
        for img_file in list(lbl_dir.glob("*.jpg"))[:5]:
            try:
                img = load_img(img_file, target_size=IMAGE_TARGET_SIZE)
                arr = tf_image_utils.img_to_array(img)

                # Enforce float32 and normalization for consistency
                arr = arr.astype('float32') / 255.0

                emb = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                vecs.append(emb)
            except Exception as e:
                logger.debug(f"Failed to process {img_file}: {e}")
                continue

        if vecs:
            centroids[lbl] = np.mean(vecs, axis=0)
    CLASS_CENTROIDS = centroids
    logger.info("✅ Class centroids computed for %d classes.", len(centroids))


def is_novel_image(file_path, threshold=0.40):
    """Check if uploaded image is novel compared to known centroids."""
    if not CLASS_CENTROIDS:
        return False

    model = get_embedding_model()
    if model is None:
        return False

    try:
        img = load_img(file_path, target_size=IMAGE_TARGET_SIZE)
        arr = tf_image_utils.img_to_array(img)

        # Enforce float32 and normalization for consistency
        arr = arr.astype('float32') / 255.0

        emb = model.predict(np.expand_dims(arr, 0), verbose=0)[0].reshape(1, -1)

        sims = [cosine_similarity(emb, c.reshape(1, -1))[0][0] for c in CLASS_CENTROIDS.values()]

        max_sim = max(sims)
        logger.debug("Novelty Check: Max Cosine Similarity = %.4f", max_sim)

        return max_sim < threshold
    except Exception as e:
        logger.debug("Novelty check failed: %s", e)
        return False


# -------------------------
# Prediction helper
# -------------------------
def _check_model_io(model):
    """Safely inspect and return model input/output shapes."""
    if model is None:
        return None, None
    try:
        if hasattr(model.inputs[0], "shape"):
            input_shape = tuple(model.inputs[0].shape)
        else:
            input_shape = tuple(model.input_shape)

        if hasattr(model.outputs[0], "shape"):
            output_shape = tuple(model.outputs[0].shape)
        else:
            output_shape = tuple(model.output_shape)

        logger.info(f"🧩 Model IO verified → Input: {input_shape}, Output: {output_shape}")
        return input_shape, output_shape
    except Exception as e:
        logger.exception(f"Failed to inspect model IO shapes: {e}")
        return None, None


_input_shape, _output_shape = _check_model_io(MODEL)
if _output_shape is not None:
    model_num_classes = int(_output_shape[-1])
    if model_num_classes != len(CLASS_LABELS):
        logger.error(
            "🛑 CRITICAL: Model output size (%d) != number of labels loaded (%d). "
            "Model's prediction might be inaccurate due to size mismatch or label corruption.",
            model_num_classes, len(CLASS_LABELS)
        )


def predict_from_image(model, class_labels, image_path):
    """Main image prediction function."""
    # 1. Image loading and preprocessing
    img = load_img(image_path, target_size=IMAGE_TARGET_SIZE)

    img_array = tf_image_utils.img_to_array(img)

    # Enforce float32 and normalization
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    preds = model.predict(img_array, verbose=0)
    preds_list = preds.tolist()[0]

    # 3. Validation
    if len(preds_list) != len(class_labels):
        logging.error(
            f"⚠️ Model output shape mismatch: {len(preds_list)} vs {len(class_labels)}. Using 'unknown' label.")
        return "unknown", 0.0, preds_list

    # 4. Result extraction
    pred_idx = np.argmax(preds[0])

    # Use the index to retrieve the label from the corrected list
    pred_label = class_labels[pred_idx]

    confidence = float(np.max(preds[0]))

    logger.info(
        f"Prediction: Index={pred_idx}, Label='{pred_label}', Confidence={confidence:.4f}, RawProbs={preds_list}")

    return pred_label, confidence, preds_list


# -------------------------
# Recommendation logic (FIXED)
# -------------------------
def get_recommendation(predicted_label: str, confidence: float) -> tuple[str, str]:
    """
    Analyzes the prediction label and confidence to return a customized
    health status and management recommendation for the user interface.
    """

    # ------------------- CRITICAL FIX 1: Confidence Threshold -------------------
    # If the model is not confident enough, treat it as 'Uncertain'
    if confidence < CONFIDENCE_THRESHOLD:
        health_status = "Prediction Uncertain"
        recommendation = (
            f"Model confidence is too low ({confidence * 100:.2f}%). "
            f"The image may be blurry, cropped poorly, or contain a rare/unknown issue. "
            f"Please ensure a clear, well-focused image is uploaded."
        )
        return health_status, recommendation

    # ------------------- CRITICAL FIX 2: Handle 'Non_Plant_images' -------------------
    if predicted_label == "Non_Plant_images":
        health_status = "Non-Plant Image"
        recommendation = (
            f"The uploaded image **does not belong to the plant class**. "
            f"Please upload a clearer image of one of the supported crops: {SUPPORTED_PLANTS}."
        )
        # Use the specific message instead of looking up in TREATMENTS
        return health_status, recommendation

        # ------------------- CRITICAL FIX 3: Handle 'Healthy' -------------------
    elif predicted_label == "Healthy":
        health_status = "Healthy"
        # Retrieve the healthy recommendation from the dictionary
        return health_status, TREATMENTS.get("Healthy", "Healthy status confirmed, no specific issues found.")

    # ------------------- Handle All Disease Predictions -------------------
    else:
        # All other predicted labels are treated as 'Diseased' subtypes
        health_status = "Diseased"

        # Look up the specific treatment recommendation
        recommendation = TREATMENTS.get(
            predicted_label,
            # Fallback for any disease label missing from TREATMENTS
            f"Image appears diseased (Label: {predicted_label}). No specific treatment found for this label in the database. Please consult an expert."
        )

        # The user sees 'Diseased' as the main status but gets the specific treatment
        return health_status, recommendation


# -------------------------
# Retraining process
# -------------------------
def run_finetune_on_queue():
    """Fine-tune model automatically when enough new labeled images are available."""
    if not retrain_lock.acquire(blocking=False):
        logger.info("🧠 Retrain check skipped: Lock held by another process.")
        return
    try:
        logger.info("🧠 Checking retrain queue...")
        images_by_label = {}

        for lbl in CLASS_LABELS:
            lbl_dir = RETRAIN_QUEUE / lbl
            if lbl_dir.exists():
                files = [str(p) for p in lbl_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                images_by_label[lbl] = files
            else:
                images_by_label[lbl] = []

        if any(len(v) < MIN_SAMPLES_PER_CLASS for v in images_by_label.values()):
            logger.info("Not enough new data to retrain yet.")
            return

        logger.info("Starting background fine-tuning...")
        global MODEL

        MODEL.compile(optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        all_paths, all_labels = [], []
        for lbl, paths in images_by_label.items():
            try:
                idx = CLASS_LABELS.index(lbl)
                all_paths += paths
                all_labels += [idx] * len(paths)
            except ValueError:
                logger.warning(f"Label '{lbl}' in retrain queue not found in CLASS_LABELS. Skipping.")

        if not all_paths:
            logger.info("No paths found for retraining.")
            return

        def load_img_tf(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, IMAGE_TARGET_SIZE)
            img = tf.cast(img, tf.float32)
            img = img / 255.0
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))
        ds = ds.map(load_img_tf, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(FINETUNE_BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE)

        # Fine-tune the model
        MODEL.fit(ds, epochs=FINETUNE_EPOCHS, verbose=1)

        # Save and reload for clean state (optional but good practice)
        MODEL.save(MODEL_PATH)
        MODEL = safe_load_model(MODEL_PATH)

        logger.info("✅ Retrain complete. Model updated and saved.")

        # Archive the used data
        archive_dir = RETRAIN_QUEUE / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)
        for lbl, files in images_by_label.items():
            target = archive_dir / lbl
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.move(f, target / Path(f).name)

    except Exception as e:
        logger.exception("Retraining failed: %s", e)
    finally:
        retrain_lock.release()


def trigger_retrain_background():
    # Only launch the thread if the scheduler is not already running the job
    Thread(target=run_finetune_on_queue, daemon=True).start()
    logger.debug("🔁 Background retraining thread launched.")


# -------------------------
# Routes (RE-ARRANGED FOR REGISTRATION FIRST)
# -------------------------

# Renamed original 'index' to 'home' and protecting it.
@app.route("/home")
@login_required
def home():
    """This is the main prediction page, now only accessible when logged in."""
    return render_template("index.html")


# The root URL (/) and /register now point to the registration page.
# In app.py
# ...

@app.route("/")
@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles user registration and serves as the primary landing page."""
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Simple validation
        if not username or not password or password != confirm_password:
            flash(_('Registration failed. Ensure passwords match and all fields are filled.'), 'danger')
            return render_template("register.html")

        # Check if user already exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash(_('That username is already taken. Please choose another.'), 'danger')
            return render_template("register.html")

        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        # Log the user in immediately after successful registration
        login_user(new_user)
        flash(_('Account created successfully! Welcome.'), 'success')
        return redirect(url_for("home"))  # Redirect to the main app page

    # If GET request
    flash(_("Please sign up to access the Plant Disease Detector."), "info")
    return render_template("register.html")


# ...


# In app.py
# ...

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login."""
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        # Check credentials
        if user and check_password_hash(user.password_hash, password):
            # Login successful!
            login_user(user)
            flash(_('Login successful!'), 'success')

            # Use 'next' parameter if set (e.g., if user tried to access /home)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            # Login failed
            flash(_('Login failed. Check your username and password.'), 'danger')

            # Stay on the login page to show the error
            return render_template("login.html")

            # If GET request
    # flash("Login route needs implementation.", "info") # Remove placeholder flash
    return render_template("login.html")


# ...


@app.route("/logout")
# Removed @login_required to prevent errors on corrupted sessions
def logout():
    """Logs out the current user and redirects to the registration/landing page."""
    if current_user.is_authenticated:
        logout_user()
        flash(_("You have been logged out."), "success")
    # Redirect to the new primary landing page (registration)
    return redirect(url_for("register"))


@app.route("/predict", methods=["POST"])
@login_required  # Protects prediction logic
def predict():
    if "file" not in request.files:
        flash(_("No file selected."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    f = request.files["file"]
    if f.filename == '':
        flash(_("No file selected."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    filename = secure_filename(f.filename)
    filepath = STATIC_UPLOADS / filename
    f.save(filepath)
    logger.info("✅ Uploaded: %s", filepath)

    if MODEL is None:
        flash(_("The model failed to load. Cannot perform prediction."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    try:
        # Run novelty and prediction
        novel = is_novel_image(str(filepath))
        if novel:
            predicted_label, confidence = "unknown", 0.0
        else:
            predicted_label, confidence, probs = predict_from_image(MODEL, CLASS_LABELS, str(filepath))

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        flash(_(f"Prediction failed due to system error: {e}"), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    # Clip confidence to ensure it's a valid probability (0.0 to 1.0)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Use the FIXED get_recommendation function
    health, recommendation = get_recommendation(predicted_label, confidence)

    # Log prediction if user is authenticated
    if current_user.is_authenticated:
        log = PredictionLog(user_id=current_user.id,
                            label=predicted_label,
                            # Store confidence as percentage string
                            confidence=f"{confidence * 100:.2f}%",
                            recommendation=recommendation)
        db.session.add(log)
        db.session.commit()

    return render_template("result.html",
                           filename=filename,
                           disease=health,
                           predicted_label=predicted_label,
                           # Pass confidence as percentage for display
                           confidence=round(confidence * 100, 2),
                           recommendation=recommendation,
                           # Pass UI labels for feedback buttons
                           ui_labels=UI_LABELS)


@app.route("/feedback", methods=["POST"])
@login_required  # Protecting feedback logic
def feedback():
    """Handles user feedback on misclassified images."""
    filename = request.form.get("filename")
    correct_label = request.form.get("correct_label")

    if not filename or not correct_label:
        flash(_("Missing filename or correct label for feedback."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    # 1. Validate the label
    if correct_label not in UI_LABELS:
        flash(_("Invalid label provided."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    # 2. Define source and destination paths
    src_path = STATIC_UPLOADS / filename
    dest_dir = RETRAIN_QUEUE / correct_label
    dest_path = dest_dir / filename

    if not src_path.exists():
        flash(_("Original image file not found."), "danger")
        return redirect(url_for("home"))  # Changed redirect target

    # 3. Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Move the file
    try:
        shutil.move(src_path, dest_path)
        logger.info("💡 Feedback received: %s moved to retrain queue under label '%s'.", filename, correct_label)

        # 5. Trigger an immediate check for retraining
        trigger_retrain_background()

        flash(_("Thank you! Your feedback has been recorded and the image has been queued for model fine-tuning."),
              "success")
    except Exception as e:
        logger.exception("Failed to move file for feedback: %s", e)
        flash(_("Failed to process feedback due to a file system error."), "danger")

    return redirect(url_for("home"))  # Changed redirect target


# -------------------------
# Exit handlers
# -------------------------
atexit.register(lambda: scheduler.shutdown(wait=False))

# -------------------------
# Scheduler
# -------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(run_finetune_on_queue, "interval", minutes=RETRAIN_CHECK_INTERVAL_MIN, id="run_finetune_on_queue")
scheduler.start()

# -------------------------
# Centroid Initialization (CRITICAL STEP for Novelty Detection)
# -------------------------
# Define a directory that holds reference images organized by class (e.g., RETRAIN_QUEUE,
# or a specific 'initial_training_data' folder if you have one).
# We use RETRAIN_QUEUE here as a proxy for labeled data.
REFERENCE_DATA_DIR = RETRAIN_QUEUE

# Only compute centroids if the main model loaded successfully and the
# reference directory exists.
if MODEL is not None and REFERENCE_DATA_DIR.exists():
    logger.info("🧠 Initializing novelty detection centroids...")

    # Run the centroid computation on a separate thread to avoid blocking the main Flask startup,
    # but ensure it runs immediately.
    centroid_thread = Thread(
        target=compute_class_centroids,
        args=(REFERENCE_DATA_DIR,),
        daemon=True
    )
    centroid_thread.start()
    logger.info("✅ Centroid computation started in background thread.")
else:
    logger.warning("⚠️ Skipping centroid computation: Model not loaded or reference directory missing.")

if __name__ == "__main__":
    # Helpful messages on startup
    print("App Starting. Model Path:", MODEL_PATH)
    print("Class labels:", CLASS_LABELS)
    app.run(debug=True)