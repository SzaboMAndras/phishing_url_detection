import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

def calculate_overlap(train_urls: pd.Series, eval_urls: pd.Series) -> dict:
    def normalize_url(url: str) -> str:
        if not isinstance(url, str):
            return ""
        url = url.lower()
        url = re.sub(r"^[a-z0-9+\-.]+://", "", url)
        url = re.sub(r"^www\.", "", url)
        return url.rstrip("/")

    def get_host(url: str) -> str:
        norm = normalize_url(url)
        return norm.split('/')[0].split('?')[0].split('#')[0].split(':')[0]

    def get_reg_domain(url: str) -> str:
        host = get_host(url)
        parts = host.split('.')
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    train_norm = set(train_urls.apply(normalize_url))
    eval_norm = eval_urls.apply(normalize_url)
    
    train_hosts = set(train_urls.apply(get_host))
    eval_hosts = eval_urls.apply(get_host)
    
    train_reg = set(train_urls.apply(get_reg_domain))
    eval_reg = eval_urls.apply(get_reg_domain)

    total = len(eval_urls)
    exact_count = sum(eval_norm.isin(train_norm))
    host_count = sum(eval_hosts.isin(train_hosts))
    reg_count = sum(eval_reg.isin(train_reg))

    return {
        "exact_count": exact_count,
        "exact_pct": (exact_count / total) * 100 if total > 0 else 0.0,
        "host_count": host_count,
        "host_pct": (host_count / total) * 100 if total > 0 else 0.0,
        "reg_count": reg_count,
        "reg_pct": (reg_count / total) * 100 if total > 0 else 0.0,
    }

def evaluate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    baseline_artifacts: dict,
    improved_artifacts: dict,
    train_urls: pd.Series = None,
    output_dir: str = "../predictions",
):
    os.makedirs(output_dir, exist_ok=True)
    urls = df["url"]
    y_true = df["label"].values if "label" in df.columns else None

    base_model = baseline_artifacts["model"]
    base_scaler = baseline_artifacts["scaler"]
    base_thresh = baseline_artifacts.get("threshold", 0.5)

    domain_features = extract_url_features(urls)
    domain_scaled = base_scaler.transform(domain_features)
    base_probs = base_model.predict_proba(domain_scaled)[:, 1]
    base_preds = (base_probs >= base_thresh).astype(int)

    imp_model = improved_artifacts["model"]
    imp_tfidf = improved_artifacts["tfidf"]
    imp_thresh = improved_artifacts.get("threshold", 0.5)

    tfidf_features = imp_tfidf.transform(urls)
    comb_features = hstack([tfidf_features, domain_features.values]).tocsr()
    imp_probs = imp_model.predict_proba(comb_features)[:, 1]
    imp_preds = (imp_probs >= imp_thresh).astype(int)

    output_df = df.copy()
    output_df["baseline_pred"] = base_preds
    output_df["baseline_prob"] = base_probs.round(4)
    output_df["improved_pred"] = imp_preds
    output_df["improved_prob"] = imp_probs.round(4)

    save_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    output_df.to_csv(save_path, index=False)
    print(f"Saved predictions to '{save_path}'.")

    if train_urls is not None:
        overlap = calculate_overlap(train_urls, urls)
        print("\n" + "=" * 65)
        print(f"TRAIN OVERLAP ANALYSIS: {dataset_name.upper()} (N = {len(df)})")
        print("=" * 65)
        print(f"  Exact URL Overlap:       {overlap['exact_count']:>5} / {len(df)} ({overlap['exact_pct']:.2f}%)")
        print(f"  Host Overlap:            {overlap['host_count']:>5} / {len(df)} ({overlap['host_pct']:.2f}%)")
        print(f"  Registered Domain Overlap: {overlap['reg_count']:>5} / {len(df)} ({overlap['reg_pct']:.2f}%)")

    if y_true is not None:
        base_metrics = calculate_metrics(y_true, base_probs, base_thresh)
        imp_metrics = calculate_metrics(y_true, imp_probs, imp_thresh)

        print("-" * 65)
        print(f"EVALUATION METRICS: {dataset_name.upper()}")
        print("-" * 65)
        print(
            f"{'Metric':<12} | {'Baseline (LR)':<22} | {'Improved (LightGBM)':<25}"
        )
        print("-" * 65)
        for m in ["Recall", "FPR", "Precision", "F1", "ROC-AUC"]:
            print(f"{m:<12} | {base_metrics[m]:<22.4f} | {imp_metrics[m]:<25.4f}")
        print("=" * 65 + "\n")

def main(
    models_dir: str = "../models",
    train_path: str = "../data/train.csv",
    ext_path: str = "../data/external_test.csv",
    test_path: str = "../data/test.csv",
):
    baseline_path = os.path.join(models_dir, "baseline_model.joblib")
    improved_path = os.path.join(models_dir, "improved_model.joblib")

    if not os.path.exists(baseline_path) or not os.path.exists(improved_path):
        raise FileNotFoundError(
            f"Model files not found in '{models_dir}'. Please run 'train.py' first."
        )

    baseline_artifacts = joblib.load(baseline_path)
    improved_artifacts = joblib.load(improved_path)
    print("Successfully loaded Baseline and Improved model artifacts.\n")

    train_urls = None
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        train_urls = train_df["url"]
    else:
        print(f"Warning: '{train_path}' not found. Overlap analysis will be skipped.")

    if os.path.exists(ext_path):
        print(f"Processing '{ext_path}'...")
        ext_df = pd.read_csv(ext_path)
        evaluate_dataset(
            ext_df, "external_test", baseline_artifacts, improved_artifacts, train_urls=train_urls
        )
    else:
        print(f"Warning: File '{ext_path}' not found. Skipping evaluation.")

    if os.path.exists(test_path):
        print(f"Processing '{test_path}'...")
        test_df = pd.read_csv(test_path)
        evaluate_dataset(
            test_df, "test", baseline_artifacts, improved_artifacts, train_urls=train_urls
        )
    else:
        print(f"Note: Target file '{test_path}' not found in current directory.")


if __name__ == "__main__":
    main()