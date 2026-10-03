import re
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_datasets(train_path: str, ext_path: str):
    train_df = pd.read_csv(train_path)
    ext_df = pd.read_csv(ext_path)
    return train_df, ext_df


def extract_url_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset_name

    # Basic length features
    df["url_len"] = df["url"].apply(len)

    # Special character and digit counts
    df["dot_count"] = df["url"].apply(lambda x: x.count("."))
    df["hyphen_count"] = df["url"].apply(lambda x: x.count("-"))
    df["slash_count"] = df["url"].apply(lambda x: x.count("/"))
    df["digit_count"] = df["url"].apply(lambda x: sum(c.isdigit() for c in x))
    df["special_char_count"] = df["url"].apply(
        lambda x: sum(not c.isalnum() for c in x)
    )

    # Protocol scheme extraction
    df["scheme"] = df["url"].apply(
        lambda x: urlparse(x).scheme if "://" in x else "none"
    )

    # Top-Level Domain (TLD) extraction
    def get_tld(url):
        try:
            netloc = urlparse(url).netloc or url.split("/")[0]
            netloc = netloc.split(":")[0].lower()
            parts = netloc.split(".")
            return parts[-1] if len(parts) > 1 else "none"
        except Exception:
            return "none"

    df["tld"] = df["url"].apply(get_tld)

    # Keyword Extraction
    keywords = [
        "login",
        "banking",
        "verify",
        "secure",
        "password",
        "username",
        "update",
        "pay",
        "account"
    ]
    for kw in keywords:
        df[f"kw_{kw}"] = df["url"].str.lower().str.contains(kw).astype(int)

    return df


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.lower()
    url = re.sub(r"^[a-z0-9+\-.]+://", "", url)  # Strip scheme
    url = re.sub(r"^www\.", "", url)  # Strip leading www.
    url = url.rstrip("/")  # Strip trailing slashes
    return url


def run_eda(train_path: str = "../data/train.csv"):
    train_raw = pd.read_csv(train_path)

    print("=" * 60)
    print("1. DATASET OVERVIEW & MISSING VALUES")
    print("=" * 60)
    print(
        f"Train shape: {train_raw.shape} | Nulls: {train_raw.isnull().sum().to_dict()}"
    )

    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)
    train_dist = train_raw["label"].value_counts(normalize=True).to_dict()
    print(
        f"Train class ratio (0/1): {train_dist.get(0, 0):.4f} / {train_dist.get(1, 0):.4f}"
    )

    # Extract features
    train_feat = extract_url_features(train_raw, "Train")

    print("\n" + "=" * 60)
    print("3. URL METRICS BY LABEL (TRAIN SET)")
    print("=" * 60)
    metrics = [
        "url_len",
        "dot_count",
        "hyphen_count",
        "digit_count",
        "special_char_count",
    ]
    summary = train_feat.groupby("label")[metrics].agg(["mean", "median", "std"])
    print(summary.round(2))

    print("\n" + "=" * 60)
    print("4. KEYWORD FREQUENCY BY LABEL (TRAIN SET)")
    print("=" * 60)
    keywords = [
        "login",
        "banking",
        "verify",
        "secure",
        "password",
        "username",
        "update",
        "pay",
        "account",
    ]
    kw_cols = [f"kw_{kw}" for kw in keywords]
    kw_summary = train_feat.groupby("label")[kw_cols].sum().T
    kw_summary.columns = ["Legitimate (0)", "Phishing (1)"]
    kw_summary.index = keywords
    print(kw_summary)

    print("\n" + "=" * 60)
    print("5. DUPLICATES AND CONFLICT AUDIT")
    print("=" * 60)
    train_feat["norm_url"] = train_feat["url"].apply(normalize_url)

    raw_dups_train = train_feat["url"].duplicated().sum()
    norm_dups_train = train_feat["norm_url"].duplicated().sum()

    grouped_train = train_feat.groupby("norm_url")["label"].nunique()
    conflicts_train = (grouped_train > 1).sum()

    print(
        f"Train -> Raw Duplicates: {raw_dups_train} | Normalized Duplicates: {norm_dups_train} | Label Conflicts: {conflicts_train}"
    )

    # Visualizations
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.kdeplot(
        data=train_feat,
        x="url_len",
        hue="label",
        common_norm=False,
        ax=axes[0, 0],
        fill=True,
    )
    axes[0, 0].set_title("URL Length Distribution by Class (Train)")
    axes[0, 0].set_xlim(0, 200)

    sns.boxplot(
        data=train_feat, x="label", y="digit_count", ax=axes[0, 1], palette="Set2"
    )
    axes[0, 1].set_title("Digit Count per URL by Class (Train)")
    axes[0, 1].set_ylim(0, 30)

    sns.countplot(
        data=train_feat, x="scheme", hue="label", ax=axes[1, 0], palette="viridis"
    )
    axes[1, 0].set_title("URL Scheme Count by Class (Train)")

    top_tlds = train_feat["tld"].value_counts().head(8).index
    sns.countplot(
        data=train_feat[train_feat["tld"].isin(top_tlds)],
        y="tld",
        hue="label",
        ax=axes[1, 1],
        palette="magma",
    )
    axes[1, 1].set_title("Top TLD Distribution by Class (Train)")

    plt.tight_layout()
    plt.savefig("../plots/eda_summary_plots.png", dpi=300)
    plt.close()

    # Plot 2: Dedicated Keyword Frequency Bar Plot
    plt.figure(figsize=(12, 6))
    kw_melted = kw_summary.reset_index().melt(
        id_vars="index", var_name="Class", value_name="Frequency"
    )
    kw_melted.rename(columns={"index": "Keyword"}, inplace=True)

    sns.barplot(
        data=kw_melted, x="Keyword", y="Frequency", hue="Class", palette="Set1"
    )
    plt.title(
        "Keyword Frequency in URLs by Class (Legitimate vs Phishing)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Keyword", fontsize=12)
    plt.ylabel("Occurrences Count", fontsize=12)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig("../plots/keyword_frequencies.png", dpi=300)
    plt.close()

    print(
        "\nSaved summary plots to 'plots/eda_summary_plots.png' and keyword counts plot to 'plots/keyword_frequencies.png'."
    )


if __name__ == "__main__":
    run_eda()