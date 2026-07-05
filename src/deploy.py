from pathlib import Path 
import shutil 

MODEL_PATH = Path("models/model.joblib")
DEPLOYMENT_PATH = Path("deployment/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run python src/train.py first.")

DEPLOYMENT_PATH.parent.mkdir(parents=True, exist_ok=True)

shutil.copy(MODEL_PATH, DEPLOYMENT_PATH)

print("Deployment step complete.")
print("Copied model to: ", DEPLOYMENT_PATH)