import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# --- CONFIGURATION ---
DATASET_BASE_PATH = r"E:\skin cancer\refined dataset"
MODEL_SAVE_PATH = r"E:\skin cancer\models"
LOGS_BASE_PATH = r"E:\skin cancer\logs"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_BASE_PATH, exist_ok=True)

IMG_SIZE = (224, 224) 
BATCH_SIZE = 8 # Optimized for your Quadro M2000M
NUM_CLASSES = 3
INITIAL_LR = 1e-4
PHASE1_EPOCHS = 20
PHASE2_END_EPOCH = 50 

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# --- DATA PIPELINE ---
def preprocess(x, y):
    return (x / 127.5) - 1.0, y

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1)
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

# --- ARCHITECTURE ---
inputs = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5, name="Dropout_for_UQ")(x) # Required for MC Dropout figures
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

# --- METRICS (FIXED) ---
metrics = [
    'accuracy',
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# --- TRAINING ---
model.compile(optimizer=optimizers.Adam(INITIAL_LR), loss='categorical_crossentropy', metrics=metrics)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_SAVE_PATH, f"best_custom_cnn_{TIMESTAMP}.keras"),
    monitor='val_loss', save_best_only=True, mode='min', verbose=1
)
csv_log = callbacks.CSVLogger(os.path.join(LOGS_BASE_PATH, f"log_custom_cnn_{TIMESTAMP}.csv"))

print(f"\n🚀 Starting Training with ALL metrics for {PHASE2_END_EPOCH} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_END_EPOCH, callbacks=[csv_log, checkpoint])

model.save(os.path.join(MODEL_SAVE_PATH, f"custom_cnn_final_{TIMESTAMP}.keras"))