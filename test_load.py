import os
import sys

# Change cwd to agridata-backend
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath('app'))

from app.lstm_forecaster import LSTMForecaster
import json

print("\n--- Starting debug test ---")

MODELS_DIR = "models"
model_key = "global_volume"
model_path_base = os.path.join(MODELS_DIR, model_key)

print(f"CWD: {os.getcwd()}")
print(f"Checking path: {model_path_base}_model.h5")
print(f"Path exists: {os.path.exists(f'{model_path_base}_model.h5')}")

if os.path.exists(f"{model_path_base}_model.h5"):
    try:
        print(f"📥 Loading global model from disk dynamically: {model_key}")
        forecaster = LSTMForecaster()
        forecaster.load_model(model_path_base)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Failed to load global model {model_key}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Model file not found on disk!")
