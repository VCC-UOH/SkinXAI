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
IMG_SIZE = (299, 299) # InceptionV3 default input size
BATCH_SIZE = 8 
NUM_CLASSES = 3
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-6
PHASE1_EPOCHS = 20
PHASE2_END_EPOCH = 50 

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def preprocess(x, y):
    # Standardizing pixel values for InceptionV3 architecture
    return tf.keras.applications.inception_v3.preprocess_input(x), y

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

print(f"\n📂 Loading dataset from: {DATASET_BASE_PATH}")

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

# Apply Augmentation + Preprocessing
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. ARCHITECTURE
# ==========================================
base_model = tf.keras.applications.InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299, 299, 3)
)
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(299, 299, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, name="Dropout_for_UQ"),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ==========================================
# 4. COMPILATION & CALLBACKS
# ==========================================
metrics = [
    'accuracy',
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

csv_log = callbacks.CSVLogger(os.path.join(LOGS_BASE_PATH, f"log_inceptionv3_{TIMESTAMP}.csv"), append=True)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_SAVE_PATH, f"best_inceptionv3_{TIMESTAMP}.keras"),
    monitor='val_loss', save_best_only=True, mode='min', verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# ==========================================
# 5. PHASE 1: Head Warmup (Frozen Base)
# ==========================================
model.compile(optimizer=optimizers.Adam(INITIAL_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"\n🚀 PHASE 1: Training InceptionV3 Head for {PHASE1_EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 6. PHASE 2: Structural Fine-Tuning
# ==========================================
print("\n🔓 PHASE 2: Unfreezing Top Inception Blocks...")
base_model.trainable = True

# InceptionV3 logic: Unfreeze from the top mixed layers (standard is layer 249)
for layer in base_model.layers[:249]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(FINE_TUNE_LR), loss='categorical_crossentropy', metrics=metrics)

print(f"🚀 Fine-tuning from epoch 20 to {PHASE2_END_EPOCH}...")
model.fit(train_ds, validation_data=val_ds, initial_epoch=PHASE1_EPOCHS, 
          epochs=PHASE2_END_EPOCH, callbacks=[csv_log, checkpoint, early_stop])

# ==========================================
# 7. FINAL EXPORT
# ==========================================
final_model_name = os.path.join(MODEL_SAVE_PATH, f"inceptionv3_final_{TIMESTAMP}.keras")
model.save(final_model_name)

print(f"\n✅ InceptionV3 training complete.")
print(f"📊 Results logged to: {os.path.join(LOGS_BASE_PATH, f'log_inceptionv3_{TIMESTAMP}.csv')}")
print(f"💾 Final model saved at: {final_model_name}")