import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
DATASET_BASE_PATH = r"E:\skin cancer\refined dataset"
MODEL_SAVE_PATH = r"E:\skin cancer\models"
LOGS_BASE_PATH = r"E:\skin cancer\logs"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_BASE_PATH, exist_ok=True)

# Hardware Tuning for Quadro M2000M (2.8GB VRAM)
IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # Lowered to 8 to avoid OOM on 2GB-3GB cards
NUM_CLASSES = 3
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-6
PHASE1_EPOCHS = 20
PHASE2_END_EPOCH = 50 

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ==========================================
# 2. DATA PIPELINE (Fixed Shapes & Augmentation)
# ==========================================
def preprocess(x, y):
    # Standardizing pixel values for DenseNet121 architecture
    return tf.keras.applications.densenet.preprocess_input(x), y

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

print(f"\n📂 Loading dataset from: {DATASET_BASE_PATH}")

# FIX: label_mode='categorical' ensures (None, 3) shape to match metrics
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_BASE_PATH, 'train'),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_BASE_PATH, 'val'),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_BASE_PATH, 'test'),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# Apply Augmentation + Preprocessing to TRAIN
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# Apply ONLY Preprocessing to VAL and TEST
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. ARCHITECTURE
# ==========================================
base_model = tf.keras.applications.DenseNet121(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, name="Dropout_for_UQ"), # Active for MC Dropout later
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ==========================================
# 4. COMPILATION & CALLBACKS
# ==========================================
# Comprehensive metrics for skin lesion research paper
metrics = [
    'accuracy',
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

csv_log = callbacks.CSVLogger(os.path.join(LOGS_BASE_PATH, f"log_{TIMESTAMP}.csv"), append=True)

# Using .keras format (SavedModel equivalent in newer Keras)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_SAVE_PATH, f"best_model_{TIMESTAMP}.keras"),
    monitor='val_loss', save_best_only=True, mode='min', verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# ==========================================
# 5. PHASE 1: Head Warmup (Frozen Base)
# ==========================================
# FIX: Using 'categorical_crossentropy' to match categorical labels
model.compile(optimizer=optimizers.Adam(INITIAL_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"\n🚀 PHASE 1: Training Head for {PHASE1_EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 6. PHASE 2: Phase-Shifted Fine-Tuning
# ==========================================
print("\n🔓 PHASE 2: Unfreezing Last 3 Dense Blocks for Structural Fine-Tuning...")
base_model.trainable = True

# DenseNet121 logic: Unfreeze top layers only to preserve low-level edge detectors
for layer in base_model.layers[:377]:
    layer.trainable = False

# Re-compile with extremely low learning rate for stability
model.compile(optimizer=optimizers.Adam(FINE_TUNE_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"🚀 Fine-tuning from epoch 20 to {PHASE2_END_EPOCH}...")
model.fit(train_ds, validation_data=val_ds, initial_epoch=PHASE1_EPOCHS, 
          epochs=PHASE2_END_EPOCH, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 7. FINAL EXPORT
# ==========================================
final_model_name = os.path.join(MODEL_SAVE_PATH, f"densenet121_final_{TIMESTAMP}.keras")
model.save(final_model_name)

print(f"\n✅ Training complete.")
print(f"📊 Results logged to: {os.path.join(LOGS_BASE_PATH, f'log_{TIMESTAMP}.csv')}")
print(f"💾 Final model saved at: {final_model_name}")