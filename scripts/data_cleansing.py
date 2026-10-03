import os
import re
import pandas as pd


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    # Lowercase
    url = url.lower()
    # Strip scheme
    url = re.sub(r"^[a-z0-9+\-.]+://", "", url)
    # Strip leading www.
    url = re.sub(r"^www\.", "", url)
    # Strip trailing slashes
    url = url.rstrip("/")
    return url


def clean_train_dataset(
    input_path: str = "../data/train.csv", output_path: str = "../data/train_cleaned.csv"
) -> pd.DataFrame:
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"[1/5] Loaded '{input_path}' with {initial_count} raw rows.")

    df["norm_url"] = df["url"].apply(normalize_url)

    label_counts = df.groupby("norm_url")["label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)

    df_no_conflicts = df[~df["norm_url"].isin(conflicting_urls)].copy()
    conflicts_removed = initial_count - len(df_no_conflicts)
    print(
        f"[2/5] Removed {conflicts_removed} rows corresponding to {len(conflicting_urls)} conflicting normalized URLs."
    )

    df_cleaned = df_no_conflicts.drop_duplicates(
        subset=["norm_url"], keep="first"
    ).copy()
    duplicates_removed = len(df_no_conflicts) - len(df_cleaned)
    print(f"[3/5] Removed {duplicates_removed} duplicate entries.")

    df_cleaned = df_cleaned.drop(columns=["norm_url"]).reset_index(drop=True)
    final_count = len(df_cleaned)

    print(
        f"[4/5] Cleaning complete. Retained {final_count} / {initial_count} rows ({final_count/initial_count:.1%})."
    )
    print("      Class Balance in Cleaned Data:")
    counts = df_cleaned["label"].value_counts().to_dict()
    print(
        f"      Legitimate (0): {counts.get(0, 0)} ({counts.get(0, 0)/final_count:.2%})"
    )
    print(
        f"      Phishing   (1): {counts.get(1, 0)} ({counts.get(1, 0)/final_count:.2%})"
    )
    
    df_cleaned.to_csv(output_path, index=False)
    print(f"[5/5] Saved cleaned dataset to '{output_path}'.")

    return df_cleaned


if __name__ == "__main__":
    clean_train_dataset()