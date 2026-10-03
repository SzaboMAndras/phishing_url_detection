import os
import re
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import pandas as pd

def extract_url_features(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        urls = data["url"]
    else:
        urls = data

    features = pd.DataFrame(index=urls.index)

    # 1. URL Length
    features["url_len"] = urls.str.len()

    # 2. Total Count of Digits
    features["number_count"] = urls.str.count(r"\d")

    # 3. Total Count of Special Characters
    features["special_char_count"] = urls.str.count(r"[^a-zA-Z0-9]")

    # 4. Keyword Flags
    keywords = ["login", "verify", "secure", "update", "pay", "account"]
    for kw in keywords:
        features[f"kw_{kw}"] = urls.str.lower().str.contains(kw, regex=False).astype(int)

    return features

def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_probs)

    return {
        "Recall": recall,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "ROC-AUC": roc_auc,
    }

def main(
    data_path: str = "../data/train_cleaned.csv",
    output_dir: str = "../models",
    operating_threshold: float = 0.5,
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    X_raw = df["url"]
    y = df["label"].values

    print(f"Loaded {len(df)} samples from '{data_path}'.")
    print(f"Operating Threshold set to: {operating_threshold}\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline_cv_metrics = []
    improved_cv_metrics = []

    print("--- Starting 5-Fold Cross-Validation ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), start=1):
        X_tr_raw, X_va_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # Extract domain features
        X_tr_dom = extract_url_features(X_tr_raw)
        X_va_dom = extract_url_features(X_va_raw)

        # Scale numerical features for Logistic Regression stability
        scaler = StandardScaler()
        X_tr_dom_scaled = scaler.fit_transform(X_tr_dom)
        X_va_dom_scaled = scaler.transform(X_va_dom)

        # Baseline Model: Logistic Regression
        base_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        base_model.fit(X_tr_dom_scaled, y_tr)
        base_probs = base_model.predict_proba(X_va_dom_scaled)[:, 1]
        baseline_cv_metrics.append(
            calculate_metrics(y_va, base_probs, operating_threshold)
        )

        # 2. Improved Model Features: TF-IDF (3-5 n-grams)
        tfidf = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=10000
        )
        X_tr_tfidf = tfidf.fit_transform(X_tr_raw)
        X_va_tfidf = tfidf.transform(X_va_raw)

        # Improved Model: LightGBM
        X_tr_comb = hstack([X_tr_tfidf, X_tr_dom.values]).tocsr()
        X_va_comb = hstack([X_va_tfidf, X_va_dom.values]).tocsr()

        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(X_tr_comb, y_tr)
        imp_probs = lgb_model.predict_proba(X_va_comb)[:, 1]
        improved_cv_metrics.append(
            calculate_metrics(y_va, imp_probs, operating_threshold)
        )

    base_avg = {
        m: np.mean([f[m] for f in baseline_cv_metrics])
        for m in baseline_cv_metrics[0]
    }
    imp_avg = {
        m: np.mean([f[m] for f in improved_cv_metrics])
        for m in improved_cv_metrics[0]
    }

    print("\n" + "=" * 65)
    print(f"5-FOLD CROSS-VALIDATION RESULTS (Operating Threshold = {operating_threshold})")
    print("=" * 65)
    print(
        f"{'Metric':<12} | {'Baseline (LR)':<22} | {'Improved (LightGBM)':<25}"
    )
    print("-" * 65)
    for m in ["Recall", "FPR", "Precision", "F1", "ROC-AUC"]:
        print(f"{m:<12} | {base_avg[m]:<22.4f} | {imp_avg[m]:<25.4f}")
    print("=" * 65 + "\n")

    # Retrain Final Models on Entire Cleaned Training Dataset
    print("--- Retraining Final Production Models on 100% of Cleaned Training Data ---")

    full_domain_df = extract_url_features(X_raw)
    final_scaler = StandardScaler()
    full_domain_scaled = final_scaler.fit_transform(full_domain_df)

    final_baseline_model = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    )
    final_baseline_model.fit(full_domain_scaled, y)

    joblib.dump(
        {
            "model": final_baseline_model,
            "scaler": final_scaler,
            "threshold": operating_threshold,
        },
        os.path.join(output_dir, "baseline_model.joblib"),
    )
    print(f"Saved Baseline artifacts to '{output_dir}/baseline_model.joblib'")

    final_tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=10000
    )
    full_tfidf = final_tfidf.fit_transform(X_raw)
    full_comb = hstack([full_tfidf, full_domain_df.values]).tocsr()

    final_improved_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    final_improved_model.fit(full_comb, y)

    joblib.dump(
        {
            "model": final_improved_model,
            "tfidf": final_tfidf,
            "threshold": operating_threshold,
        },
        os.path.join(output_dir, "improved_model.joblib"),
    )
    print(f"Saved Improved artifacts to '{output_dir}/improved_model.joblib'\n")


if __name__ == "__main__":
    main()