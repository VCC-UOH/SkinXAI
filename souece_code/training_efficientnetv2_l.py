import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# ==========================================
# 1. CONFIGURATION & DIRECTORIES (IDENTICAL)
# ==========================================
DATASET_BASE_PATH = r"E:\skin cancer\refined dataset"
MODEL_SAVE_PATH = r"E:\skin cancer\models"
LOGS_BASE_PATH = r"E:\skin cancer\logs"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_BASE_PATH, exist_ok=True)

# Kept exactly the same as your DenseNet/Inception scripts
IMG_SIZE = (224, 224) 
BATCH_SIZE = 8 
NUM_CLASSES = 3
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-6
PHASE1_EPOCHS = 20
PHASE2_END_EPOCH = 50 

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ==========================================
# 2. DATA PIPELINE (IDENTICAL)
# ==========================================
def preprocess(x, y):
    # Standardizing for EfficientNetV2 architecture
    return tf.keras.applications.efficientnet_v2.preprocess_input(x), y

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_BASE_PATH, 'train'),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_BASE_PATH, 'val'),
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. ARCHITECTURE (MODEL SWAP ONLY)
# ==========================================
base_model = tf.keras.applications.EfficientNetV2L(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5), 
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ==========================================
# 4. MONITORING & METRICS (IDENTICAL)
# ==========================================
metrics = ['accuracy', tf.keras.metrics.AUC(name='auc'), 
           tf.keras.metrics.Precision(name='precision'), 
           tf.keras.metrics.Recall(name='recall')]

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_SAVE_PATH, f"best_effnetv2l_{TIMESTAMP}.keras"),
    monitor='val_loss', save_best_only=True, mode='min'
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
csv_log = callbacks.CSVLogger(os.path.join(LOGS_BASE_PATH, f"log_effnetv2l_{TIMESTAMP}.csv"))

# ==========================================
# 5. PHASE 1: Warmup (Frozen Base)
# ==========================================
model.compile(optimizer=optimizers.Adam(INITIAL_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"\n🚀 PHASE 1: Training Head for {PHASE1_EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 6. PHASE 2: Structural Fine-Tuning
# ==========================================
print("\n🔓 PHASE 2: Unfreezing Top Blocks for Structural Fine-Tuning...")
base_model.trainable = True

# For paper consistency, unfreezing the last 100 layers (standard deep-net fine-tune)
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(FINE_TUNE_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"🚀 Fine-tuning from epoch 20 to {PHASE2_END_EPOCH}...")
model.fit(train_ds, validation_data=val_ds, initial_epoch=PHASE1_EPOCHS, 
          epochs=PHASE2_END_EPOCH, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 7. FINAL EXPORT
# ==========================================
model.save(os.path.join(MODEL_SAVE_PATH, f"final_effnetv2l_{TIMESTAMP}.keras"))
print(f"✅ Training Complete. Model saved.")