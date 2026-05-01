"""
model.py - Smart Agriculture AI
Handles training, saving, and loading of the RandomForest crop prediction model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "label_encoder.pkl")

# ── Feature order (must match form inputs) ───────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ─────────────────────────────────────────────────────────────────────────────
def train_model() -> dict:
    """
    Train a RandomForestClassifier on Crop_recommendation.csv.
    Saves model and label encoder to disk.
    Returns a dict with accuracy and classification report.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            "Download Crop_recommendation.csv and place it in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    ENC_PATH)

    print(f"[model.py] Training complete — Accuracy: {acc * 100:.2f}%")
    print(report)

    return {"accuracy": acc, "report": report}


# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load model + encoder from disk; train if not present."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
        print("[model.py] No saved model found — training now …")
        train_model()
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(ENC_PATH)
    return model, le


# ─────────────────────────────────────────────────────────────────────────────
def predict_crop(features: dict) -> dict:
    """
    Given a dict of feature values, return:
        - top_3   : [(crop_name, probability), …] sorted desc
        - all     : full probability distribution
    """
    model, le = load_model()

    X = np.array([[features[f] for f in FEATURES]])

    probas  = model.predict_proba(X)[0]          # shape: (n_classes,)
    classes = le.classes_                         # string labels

    prob_map = dict(zip(classes, probas))

    top_3 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "top_3": top_3,           # [(name, prob), ...]
        "all":   prob_map,
    }
