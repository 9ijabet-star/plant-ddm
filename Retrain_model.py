import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0  # Keep this import here for easy access
import json
import os
import numpy as np

# --- Configuration ---
# IMPORTANT: Adjust these paths and values to match your project structure
DATA_DIR = 'Dataset'  # Root directory containing 'train' and 'validation' subfolders
MODEL_PATH = 'plant_disease_model_best.keras'
CLASS_INDICES_FILE = 'class_indices.json'
IMAGE_SIZE = (150, 150)  # REDUCED from (224, 224) for faster CPU training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4  # Use a very small learning rate for fine-tuning

# OPTIMIZATION FOR SPEED: Set to 0 to freeze the entire feature extractor and only train the head.
# After this run successfully creates the new .keras file, you can change this to 10 or 20
# to fully fine-tune the model for better accuracy.
FROZEN_LAYERS_COUNT = 0
DROPOUT_RATE = 0.3


# --- Core Optimization Functions ---

def setup_gpu():
    """Checks for and initializes GPU, printing status."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print("✅ GPU detected and configured for training.")
        except RuntimeError as e:
            print(f"❌ GPU configuration failed: {e}")
    else:
        print("⚠️ No GPU detected. Training will run on CPU, which may be slow.")


def get_optimized_dataset(subset_name):
    """
    Creates a fast, optimized tf.data.Dataset pipeline.

    The 'subset_name' is used as the sub-directory name (e.g., 'train' or 'validation').
    """
    full_directory_path = os.path.join(DATA_DIR, subset_name)
    print(f"🔍 Loading {subset_name} data from: {full_directory_path}")

    # Use image_dataset_from_directory for fast loading
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=full_directory_path,
        labels='inferred',
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True if subset_name == 'train' else False,
        seed=42 if subset_name == 'train' else None,
    )

    # --- PERFORMANCE IMPROVEMENTS ---
    dataset = dataset.cache()

    # Preprocessing: Rescale to [0, 1] range.
    def scale(image, label):
        # Cast to float32 and normalize
        image = tf.cast(image, tf.float32)
        return image / 255.0, label

    dataset = dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetching ensures the next batch is ready while the current one is processed
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# --- Main Fine-Tuning Function ---

def fine_tune_model():
    # 1. Setup GPU or check for CPU
    setup_gpu()

    # 2. Load Datasets
    train_ds = get_optimized_dataset('train')
    val_ds = get_optimized_dataset('validation')

    # Get number of classes from the dataset (guaranteed to be correct)
    num_classes = train_ds.element_spec[1].shape[1]
    print(f"\n✅ Successfully found {num_classes} classes for training.")

    if num_classes != 3:
        print(
            f"🚨 WARNING: Found {num_classes} classes. Please ensure your 'train' folder only contains the correct subdirectories for your classes.")

    # Initialize model and base_model for safe scope management
    model = None
    base_model = None

    # 3. Load and Rebuild Model (Input Shape Fix)
    print("✅ Rebuilding model architecture using EfficientNetB0 with fixed input shape (150, 150)...")
    try:

        # Create a new input tensor with the new IMAGE_SIZE
        new_input = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name='new_input')

        # Create the new base model instance using EfficientNetB0.
        # CRITICAL FIX: Set weights=None to force the model structure to accept 3 channels
        # without immediately running into the loading conflict.
        base_model = EfficientNetB0(
            input_tensor=new_input,
            include_top=False,
            weights=None
        )

        # Manually load ImageNet weights onto the structure
        print("Attempting reliable manual loading of ImageNet weights...")
        try:
            # FIX 1: Rely on the built-in Keras mechanism to find the correct weights file URL.
            # We must use a URL-less call here, as the direct URL failed.
            weights_file = tf.keras.utils.get_file(
                'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_no_top.h5',
                'https://storage.googleapis.com/tensorflow/keras-applications/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_no_top.h5',
                # Keeping URL, but Keras should resolve it better internally.
                cache_subdir='models',
                file_hash='494f4c2810852e61a8684988cc8b8398b1e4fe335359e19c35a8f59f9f592af2'
                # Added expected hash for verification
            )
            base_model.load_weights(weights_file, by_name=True)
            print("✅ ImageNet weights successfully loaded manually.")
        except Exception as weight_e:
            print(
                f"❌ Warning: Failed to manually load ImageNet weights. Base model will start from scratch. Error: {weight_e}")

        # Use the output of the new base model
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)

        # --- Add Classification Head Layers ---
        x = Dropout(DROPOUT_RATE, name='dropout_regularizer')(x)

        # Use softmax activation for multi-class classification
        output = Dense(
            num_classes,
            activation='softmax',
            name='predictions_dense'
        )(x)

        # Define the new model structure
        model = Model(new_input, output)

        print(f"✅ Model architecture successfully rebuilt with input shape {IMAGE_SIZE}.")

    except Exception as e:
        print(f"❌ Error defining model structure: {e}")
        return

        # Safety check
    if model is None or base_model is None:
        print("❌ Failed to define the model structure. Exiting fine-tuning process.")
        return

        # 4. Fine-Tuning Layer Control
    if FROZEN_LAYERS_COUNT == 0:
        # OPTIMIZATION FOR SPEED: Fully freeze the backbone
        print("💡 Maximizing speed: Base model is fully frozen (transfer learning head only).")
        base_model.trainable = False
    elif FROZEN_LAYERS_COUNT > 0:
        print(f"✅ Applying fine-tuning: Unfreezing last {FROZEN_LAYERS_COUNT} layers of the feature extractor.")
        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = False
        total_layers = len(base_model.layers)
        for layer in base_model.layers[total_layers - FROZEN_LAYERS_COUNT:]:
            layer.trainable = True

    # 5. Compile the Model (using a small learning rate for fine-tuning)
    # Use a scheduler for improved convergence dynamics

    # FIX 2: Use the safer way to check dataset size for decay steps
    decay_steps_default = 1000  # Fallback default
    if hasattr(train_ds, 'cardinality'):
        cardinality_val = train_ds.cardinality().numpy()
        # In modern TF, tf.data.INFINITE_CARDINALITY is used, which is -2.
        # tf.data.UNKNOWN is -1. Check for valid positive size.
        if cardinality_val > 0:
            decay_steps_default = cardinality_val * 2

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=decay_steps_default,
        decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Optional: Print summary to verify the input shape is now correct (150, 150)
    print("\nModel Summary (check Input Shape: [None, 150, 150, 3]):")
    model.summary()

    # 6. Callbacks for Stability and Best Result Saving
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # 7. Fine-Tune
    print(f"\n🚀 Starting fine-tuning for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 8. Save updated class indices
    # We must handle the case where the 'train' folder might not exist if the dataset loading failed previously
    train_path = os.path.join(DATA_DIR, 'train')
    if os.path.isdir(train_path):
        class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        class_indices = {name: i for i, name in enumerate(class_names)}
        with open(CLASS_INDICES_FILE, 'w') as f:
            json.dump(class_indices, f)
        print(f"\n✅ Updated class indices saved to {CLASS_INDICES_FILE}. Total classes: {len(class_indices)}")
    else:
        print(f"\n❌ Could not save class indices: Training directory '{train_path}' not found.")


if __name__ == "__main__":
    fine_tune_model()
