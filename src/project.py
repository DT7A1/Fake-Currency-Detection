import os
import sys
import logging

# ---------------------------------------------------------
# 1. System Configuration (Strict Silent Mode)
# ---------------------------------------------------------
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import shutil
import json
import random
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix # type: ignore

import cv2 # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import splitfolders # type: ignore

# ✅ FIX: Fixed Random Seeds for reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------
# 2. Project Constants & Settings
# ---------------------------------------------------------
RAW_DATA_PATH    = "../data/dataset"
PROCESSED_PATH   = "../data/processed_data"
FINAL_SPLIT      = "../data/final_split"
OUTPUTS_DIR      = "../outputs"
MODEL_FILE       = "../models/currency_model.h5"
INDICES_FILE     = "../models/class_indices.json"
CM_PLOT_FILE     = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
LC_PLOT_FILE     = os.path.join(OUTPUTS_DIR, "learning_curve.png")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.0001

# 80% Training, 10% Validation, 10% Testing
SPLIT_RATIOS = (0.8, 0.1, 0.1)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)


# ---------------------------------------------------------
# 3. ✅ IMPROVED: Realistic Fake Image Generation
#    Simulates real-world counterfeiting defects:
#    - Low-quality ink (color shift)
#    - Poor printing (noise + blur)
#    - Missing security features (region blackout)
#    - Paper texture differences (pattern overlay)
#    - Misalignment (random rotation/shift)
# ---------------------------------------------------------
def generate_fake_image(img_array):
    """
    Applies realistic counterfeiting effects to a real banknote image.
    Returns a fake-looking version of the image.
    """
    fake = img_array.copy().astype(np.float32)

    # --- Effect 1: Color shift (cheap ink simulation) ---
    # Counterfeit notes often have slightly different color balance
    if random.random() > 0.3:
        channel = random.randint(0, 2)
        shift = random.uniform(-40, 40)
        fake[:, :, channel] = np.clip(fake[:, :, channel] + shift, 0, 255)

    # --- Effect 2: Low-quality printing noise ---
    # Adds random grain like a low-DPI printer
    if random.random() > 0.3:
        noise_level = random.uniform(10, 35)
        noise = np.random.normal(0, noise_level, fake.shape)
        fake = np.clip(fake + noise, 0, 255)

    # --- Effect 3: Blur (loss of fine detail) ---
    # Real notes have sharp micro-printing; fakes are often blurry
    if random.random() > 0.3:
        kernel_size = random.choice([3, 5, 7])
        fake = cv2.GaussianBlur(fake.astype(np.uint8),
                                (kernel_size, kernel_size), 0).astype(np.float32)

    # --- Effect 4: Missing security feature simulation ---
    # Blocks out a small random region (simulates missing watermark/thread)
    if random.random() > 0.4:
        h, w = fake.shape[:2]
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = x1 + random.randint(20, w // 4)
        y2 = y1 + random.randint(20, h // 4)
        fill_color = random.choice([
            [200, 200, 200],  # gray patch
            [255, 255, 255],  # white patch
            [180, 160, 130],  # paper-tone patch
        ])
        fake[y1:y2, x1:x2] = fill_color

    # --- Effect 5: Paper texture overlay ---
    # Counterfeit paper lacks the cotton-linen feel; simulated via pattern
    if random.random() > 0.5:
        texture = np.random.uniform(-15, 15, fake.shape)
        fake = np.clip(fake + texture, 0, 255)

    # --- Effect 6: Slight rotation / misalignment ---
    # Counterfeit printing is often slightly misaligned
    if random.random() > 0.5:
        h, w = fake.shape[:2]
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        fake = cv2.warpAffine(fake.astype(np.uint8), M, (w, h),
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    # --- Effect 7: Brightness inconsistency ---
    # Uneven lighting due to poor printing press
    if random.random() > 0.4:
        brightness = random.uniform(0.6, 1.3)
        fake = np.clip(fake * brightness, 0, 255)

    return fake.astype(np.uint8)


# ---------------------------------------------------------
# 4. Data Engineering Pipeline
# ---------------------------------------------------------
def prepare_and_split_data():
    """
    Handles data prep: Merging, Balancing with improved fake generation, and Splitting.
    """
    # --- Part A: Dataset Preparation ---
    real_dir = os.path.join(PROCESSED_PATH, 'Real')
    fake_dir = os.path.join(PROCESSED_PATH, 'Fake')

    if os.path.exists(PROCESSED_PATH) and \
       os.path.exists(real_dir) and len(os.listdir(real_dir)) > 0 and \
       os.path.exists(fake_dir) and len(os.listdir(fake_dir)) > 0:
        print("[INFO] Dataset already prepared. Skipping generation.")
    else:
        print("[INFO] Processing dataset (Merging & Generating Realistic Fakes)...")
        if os.path.exists(PROCESSED_PATH):
            shutil.rmtree(PROCESSED_PATH)

        os.makedirs(real_dir)
        os.makedirs(fake_dir)

        images = glob(RAW_DATA_PATH + "/**/*.jpg", recursive=True) + \
                 glob(RAW_DATA_PATH + "/**/*.png", recursive=True)

        if not images:
            print("[ERROR] No images found in source directory.")
            sys.exit(1)

        print(f"[INFO] Copying {len(images)} Real images...")
        for i, img_path in enumerate(images):
            shutil.copy(img_path, os.path.join(real_dir, f"real_{i}.jpg"))

        print(f"[INFO] Generating {len(images)} realistic fake samples...")
        real_imgs = glob(os.path.join(real_dir, "*.jpg"))
        count = 0

        while count < len(images):
            try:
                src_path = random.choice(real_imgs)
                img = load_img(src_path, target_size=IMG_SIZE)
                x = img_to_array(img)

                # ✅ Use improved realistic fake generation
                fake_x = generate_fake_image(x)

                save_path = os.path.join(fake_dir, f"fake_{count}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(fake_x, cv2.COLOR_RGB2BGR))
                count += 1
                sys.stdout.write(f"\r[INFO] Generating: {count}/{len(images)}")
                sys.stdout.flush()
            except Exception as e:
                continue

        print(f"\n[SUCCESS] Dataset preparation complete. Real: {len(images)}, Fake: {count}")

    # --- Part B: Data Splitting ---
    train_dir = os.path.join(FINAL_SPLIT, "train")
    val_dir   = os.path.join(FINAL_SPLIT, "val")
    test_dir  = os.path.join(FINAL_SPLIT, "test")

    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        print("[INFO] Data already split. Skipping.")
    else:
        if os.path.exists(FINAL_SPLIT):
            shutil.rmtree(FINAL_SPLIT)
        print(f"[INFO] Splitting: Train={SPLIT_RATIOS[0]*100:.0f}%,"
              f" Val={SPLIT_RATIOS[1]*100:.0f}%, Test={SPLIT_RATIOS[2]*100:.0f}%...")
        splitfolders.ratio(PROCESSED_PATH, output=FINAL_SPLIT,
                           seed=SEED, ratio=SPLIT_RATIOS, group_prefix=None)
        print("[SUCCESS] Splitting complete.")


# ---------------------------------------------------------
# 5. Evaluation Module
# ---------------------------------------------------------
def evaluate_pipeline(model=None):
    """
    Evaluates model on Train, Val, and Test sets.
    """
    print("\n" + "="*50)
    print(" STARTING EVALUATION PIPELINE")
    print("="*50)

    prepare_and_split_data()

    if model is None:
        if not os.path.exists(MODEL_FILE):
            print(f"[ERROR] Model not found at {MODEL_FILE}. Train first.")
            return
        print(f"[INFO] Loading model from: {MODEL_FILE}")
        model = load_model(MODEL_FILE)

    print("[INFO] Preparing data generators...")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen.flow_from_directory(
        f"{FINAL_SPLIT}/train", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    val_gen = datagen.flow_from_directory(
        f"{FINAL_SPLIT}/val", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    test_path = f"{FINAL_SPLIT}/test"
    if not os.path.exists(test_path):
        test_path = f"{FINAL_SPLIT}/val"
    test_gen = datagen.flow_from_directory(
        test_path, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    print("[INFO] Calculating metrics...")

    # ✅ FIX: Use steps to avoid incomplete batch issues
    _, train_acc = model.evaluate(train_gen, steps=len(train_gen), verbose=0)
    _, val_acc   = model.evaluate(val_gen,   steps=len(val_gen),   verbose=0)
    _, test_acc  = model.evaluate(test_gen,  steps=len(test_gen),  verbose=0)

    print("\n" + "="*50)
    print(" FINAL PERFORMANCE SUMMARY")
    print("="*50)
    print(f" [+] Training Accuracy   : {train_acc*100:.2f}%")
    print(f" [+] Validation Accuracy : {val_acc*100:.2f}%")
    print(f" [+] Testing Accuracy    : {test_acc*100:.2f}%")
    print("="*50)

    print("[INFO] Generating classification report for Test Set...")
    test_gen.reset()
    preds_probs = model.predict(test_gen, steps=len(test_gen), verbose=1)
    preds  = (preds_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes[:len(preds)]

    print("\n[INFO] Classification Report:")
    report = classification_report(y_true, preds, target_names=['Fake', 'Real'])
    print(report)

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CM_PLOT_FILE)
    plt.close()
    print(f"[SUCCESS] Confusion matrix saved: {CM_PLOT_FILE}")


# ---------------------------------------------------------
# 6. Training Module
# ---------------------------------------------------------
def train_pipeline():
    """
    Full training pipeline: Build -> Train -> Save -> Evaluate.
    """
    print("\n" + "="*50)
    print(" STARTING TRAINING PIPELINE")
    print("="*50)

    # Hardware Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            details = tf.config.experimental.get_device_details(gpus[0])
            print(f"[SYSTEM] GPU: {details.get('device_name', 'Unknown')}")
        except Exception:
            pass
    else:
        print("[WARNING] No GPU detected. Running on CPU.")

    prepare_and_split_data()

    print("[INFO] Configuring Data Generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        f"{FINAL_SPLIT}/train", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary')

    val_gen = val_datagen.flow_from_directory(
        f"{FINAL_SPLIT}/val", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode='binary')

    with open(INDICES_FILE, 'w') as f:
        json.dump(train_gen.class_indices, f)
    print(f"[INFO] Class indices: {train_gen.class_indices}")

    # Build Model
    print("[INFO] Building MobileNetV2 model...")
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    x   = GlobalAveragePooling2D()(base.output)
    x   = Dense(128, activation='relu')(x)
    x   = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    #model.summary()

    # Train
    print(f"[INFO] Training for up to {EPOCHS} epochs...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save Learning Curve
    print("[INFO] Saving learning curves...")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy'); plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss'); plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(LC_PLOT_FILE)
    plt.close()
    print(f"[SUCCESS] Learning curve saved: {LC_PLOT_FILE}")

    # Save Model
    model.save(MODEL_FILE)
    print(f"[SUCCESS] Model saved: {MODEL_FILE}")

    # Run Evaluation
    evaluate_pipeline(model)


# ---------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------
def main():
    while True:
        print("\n" + "*"*45)
        print(" Currency Detection System v0.5")
        print("*"*45)
        print(f" Split Ratios: Train={SPLIT_RATIOS[0]*100:.0f}%"
              f" | Val={SPLIT_RATIOS[1]*100:.0f}%"
              f" | Test={SPLIT_RATIOS[2]*100:.0f}%")
        print(" [1] Train New Model & Show Summary")
        print(" [2] Evaluate Saved Model")
        print(" [3] Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            train_pipeline()
            break
        elif choice == '2':
            evaluate_pipeline()
            break
        elif choice == '3':
            print("Exiting...")
            sys.exit(0)
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()