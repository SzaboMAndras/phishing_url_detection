import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

_MODEL_CACHE = {}

def _get_model_artifacts(model_type: str = "improved", models_dir: str = "../models"):
    model_type = model_type.lower()
    if model_type not in ["baseline", "improved"]:
        raise ValueError("model_type must be either 'baseline' or 'improved'")

    if model_type not in _MODEL_CACHE:
        filename = f"{model_type}_model.joblib"
        filepath = os.path.join(models_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file '{filepath}' not found. Ensure train.py has been executed."
            )

        _MODEL_CACHE[model_type] = joblib.load(filepath)

    return _MODEL_CACHE[model_type]


def validate_url(url: str) -> str:
    if url is None or not isinstance(url, str):
        raise ValueError("Invalid input: URL must be a non-empty string.")

    url_clean = url.strip()
    if not url_clean:
        raise ValueError("Invalid input: Provided URL string is empty.")

    if "." not in url_clean and not url_clean.lower().startswith(
        ("http://localhost", "https://localhost")
    ):
        raise ValueError(
            f"Malformed URL: '{url_clean}' does not have a valid domain structure."
        )

    return url_clean

def extract_url_features_single(url: str) -> pd.DataFrame:
    url_lower = url.lower()
    
    data = {
        # 1. URL Length
        "url_len": [len(url)],
        # 2. Total Count of Digits / Numbers
        "number_count": [sum(c.isdigit() for c in url)],
        # 3. Total Count of Special Characters (non-alphanumeric)
        "special_char_count": [sum(not c.isalnum() for c in url)],
    }
    
    keywords = ["login", "verify", "secure", "update", "pay", "account"]
    for kw in keywords:
        data[f"kw_{kw}"] = [1 if kw in url_lower else 0]

    return pd.DataFrame(data)

def predict(
    url: str, model_type: str = "improved", models_dir: str = "../models"
) -> dict:
    
    url_clean = validate_url(url)

    artifacts = _get_model_artifacts(model_type=model_type, models_dir=models_dir)
    model = artifacts["model"]
    threshold = artifacts.get("threshold", 0.5)

    domain_df = extract_url_features_single(url_clean)

    if model_type.lower() == "baseline":
        scaler = artifacts["scaler"]
        X_feats = scaler.transform(domain_df)
    else:
        tfidf = artifacts["tfidf"]
        tfidf_feats = tfidf.transform([url_clean])
        X_feats = hstack([tfidf_feats, domain_df.values]).tocsr()

    prob_float = float(model.predict_proba(X_feats)[0, 1])

    prob_float = float(np.clip(prob_float, 0.0, 1.0))

    label_str = "phishing" if prob_float >= threshold else "clean"

    return {"label": label_str, "probability": prob_float}