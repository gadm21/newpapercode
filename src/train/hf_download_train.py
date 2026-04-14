#!/usr/bin/env python3
"""
Download the OfficeLocalization dataset from HuggingFace and train a
RandomForest classifier on rolling-variance features.

Dataset: https://huggingface.co/datasets/gadgadgad/OfficeLocalization

Usage:
    pip install huggingface_hub
    python hf_download_train.py
    python hf_download_train.py --cache-dir ./hf_data --window-len 1000 --var-window 20
"""

import os
import sys
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add train/ to path for local imports
_train_dir = os.path.dirname(os.path.abspath(__file__))
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from utils import (
    CSI_SUBCARRIER_MASK, CSI_Loader, FeatureSelector,
    TrainingDataset, set_global_seed, compute_all_metrics, print_metrics_summary,
)

# =========================================================================
# HuggingFace download
# =========================================================================
HF_REPO_ID = "gadgadgad/OfficeLocalization"

# Files in the repo and their labels + splits (based on the dataset metadata)
HF_FILES = [
    # (remote filename, label, split)
    ("empty_1.csv", "empty", "train"),
    ("empty_2.csv", "empty", "test"),
    ("five_1.csv",  "five",  "train"),
    ("five_2.csv",  "five",  "test"),
    ("one_1.csv",   "one",   "train"),
    ("one_2.csv",   "one",   "test"),
    ("two_1.csv",   "two",   "train"),
    ("two_2.csv",   "two",   "test"),
]


def download_dataset(cache_dir):
    """Download all CSV files from the HuggingFace repo.

    Returns
    -------
    list of (local_path, label, split)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required.  Install with:\n"
              "  pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)
    downloaded = []

    print(f"\n[download] Fetching {len(HF_FILES)} files from {HF_REPO_ID} ...")
    for fname, label, split in HF_FILES:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        print(f"  {fname:20s}  label={label:6s}  split={split:5s}  -> {local}")
        downloaded.append((local, label, split))

    print(f"[download] Done — {len(downloaded)} files cached in {cache_dir}\n")
    return downloaded


# =========================================================================
# Processing helpers (reuse from utils.py)
# =========================================================================
def load_and_process(filepath, guaranteed_sr=150, var_window=20,
                     window_len=500, stride=None, mode='flattened'):
    """Load a single CSV, apply pipeline, return windowed features (N, D)."""
    stride = stride or window_len

    loader = CSI_Loader(verbose=False)
    loader.guaranteed_sr = guaranteed_sr
    selector = FeatureSelector(mask=CSI_SUBCARRIER_MASK, verbose=False)

    data = loader.process(filepath)
    data = selector.process(data)
    mag = data['mag']  # (T, 52)

    # Rolling variance
    rv = TrainingDataset._rolling_variance(mag, var_window)

    # Window
    X = TrainingDataset._window_array_static(rv, window_len, stride, mode)
    return X


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download OfficeLocalization from HuggingFace and train RF")
    parser.add_argument("--cache-dir", type=str, default=os.path.join(_train_dir, "hf_data"),
                        help="Directory to cache downloaded files")
    parser.add_argument("--window-len", type=int, default=500,
                        help="Window length L (default: 500)")
    parser.add_argument("--var-window", type=int, default=20,
                        help="Rolling variance window W (default: 20)")
    parser.add_argument("--sr", type=int, default=150,
                        help="Guaranteed sampling rate (default: 150)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    # --- 1. Download ---
    files = download_dataset(args.cache_dir)

    # --- 2. Build label map ---
    labels_sorted = sorted(set(lbl for _, lbl, _ in files))
    label_map = {lbl: i for i, lbl in enumerate(labels_sorted)}
    print(f"[info] Labels: {label_map}")

    # --- 3. Load & process each file ---
    train_X_parts, train_y_parts = [], []
    test_X_parts, test_y_parts = [], []

    for fpath, label, split in files:
        print(f"  Processing {os.path.basename(fpath):20s}  label={label:6s}  split={split} ...", end="")
        X = load_and_process(
            fpath,
            guaranteed_sr=args.sr,
            var_window=args.var_window,
            window_len=args.window_len,
        )
        if X is None or len(X) == 0:
            print("  SKIP (no windows)")
            continue

        y = np.full(X.shape[0], label_map[label], dtype=np.int64)
        print(f"  windows={X.shape[0]}")

        if split == "train":
            train_X_parts.append(X)
            train_y_parts.append(y)
        else:
            test_X_parts.append(X)
            test_y_parts.append(y)

    X_train = np.concatenate(train_X_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    X_test = np.concatenate(test_X_parts, axis=0)
    y_test = np.concatenate(test_y_parts, axis=0)

    print(f"\n[info] Train: {X_train.shape}  ({len(np.unique(y_train))} classes)")
    print(f"[info] Test:  {X_test.shape}  ({len(np.unique(y_test))} classes)")

    # --- 4. Train RandomForest ---
    print("\n[train] Fitting RandomForest (200 trees, balanced) ...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=args.seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # --- 5. Evaluate ---
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

    metrics = compute_all_metrics(y_test, y_pred, y_prob=y_prob,
                                  n_classes=len(label_map))
    print_metrics_summary(metrics, title="RandomForest — Office Localization (HuggingFace)")

    print(f"\n[result] Accuracy: {metrics['accuracy']}")
    print(f"[result] Cohen's Kappa: {metrics['cohen_kappa']}")

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in range(len(label_map))]
    print("\n" + classification_report(y_test, y_pred, target_names=target_names))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


if __name__ == "__main__":
    main()
