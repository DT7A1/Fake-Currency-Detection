import os
import sys

# ---------------------------------------------------------
# System Configuration (Strict Silent Mode)
# ---------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
warnings.filterwarnings('ignore')

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
# Important: Must use MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
MODEL_PATH = '../models/currency_model.h5'
RESULT_IMAGE = "../outputs/prediction_result.png"
IMG_SIZE = (224, 224)

def test_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at path: {image_path}")
        return

    print("[INFO] Loading model...", end='\r')
    try:
        model = load_model(MODEL_PATH)
        
        # 1. Load and Resize
        img_original = load_img(image_path, target_size=(600,600)) 
        img = load_img(image_path, target_size=IMG_SIZE)
        
        # 2. Convert to Array and Batch
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # 3. Preprocess (Normalization for MobileNetV2)
        x = preprocess_input(x)

        # 4. Inference
        print("[INFO] Analyzing image...", end='\r')
        prediction = float(model.predict(x, verbose=0).flatten()[0])

        # 5. Interpretation
        # 0 = Fake, 1 = Real
        if prediction > 0.5:
            label = "Real Currency ($)"
            confidence = prediction
            color = 'green'
            status = "AUTHENTIC"
        else:
            label = "Fake Currency (!)"
            confidence = 1 - prediction
            color = 'red'
            status = "COUNTERFEIT"

        # 6. Report Output
        print(" " * 40, end='\r') # Clean line
        print("-" * 50)
        print(f" CLASSIFICATION REPORT")
        print("-" * 50)
        print(f" Status     : {status}")
        print(f" Confidence : {confidence * 100:.2f}%")
        print("-" * 50)

        # 7. Visualization Output
        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        plt.axis('off')
        plt.title(f"{label}\nConfidence: {confidence*100:.2f}%", color=color, fontsize=14, fontweight='bold')
        plt.savefig(RESULT_IMAGE)
        print(f"[SUCCESS] Result image saved to: {RESULT_IMAGE}")

    except Exception as e:
        print(f"\n[ERROR] An exception occurred: {e}")

if __name__ == "__main__":
    # --- Set your test image path here ---
    TEST_IMAGE_PATH = "../samples/100$.jpg" 
    
    test_single_image(TEST_IMAGE_PATH)