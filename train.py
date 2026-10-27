"""
train.py
────────
One-shot script to train the churn model, compute economics, and persist
everything under artifacts/.  Run once before launching the dashboard.

Usage:
    python train.py [--data path/to/csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from data_pipeline import prepare_data, load_raw_data, engineer_features, encode_dataframe
from churn_model import train_model, evaluate_model, score_customers, save_model
from retention_economics import enrich_customer_data, build_segment_economics
from config import ARTIFACTS_DIR


def main(data_path=None):
    # Ensure Unicode prints don't crash on Windows cp1252 consoles
    import io, sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  Churn-Retention-Economics  -  Training Pipeline")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/5] Loading & engineering features ...")
    X_train, X_test, y_train, y_test, df_full, feature_names = prepare_data(data_path)
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(feature_names)}")

    # 2. Train
    print("\n[2/5] Training XGBoost model ...")
    model = train_model(X_train, y_train, X_test, y_test)

    # 3. Evaluate
    print("\n[3/5] Evaluating ...")
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"      {k:22s}  {v}")
    cm = metrics["confusion_matrix"]
    print(f"\n      Confusion Matrix:\n      {cm[0]}\n      {cm[1]}")

    # 4. Score full dataset (out-of-fold for train, held-out for test)
    print("\n[4/5] Computing CLV & margin-at-risk (OOF scoring) ...")

    # Out-of-fold predictions for training rows avoid data leakage:
    # each training customer is scored by a model that never saw it.
    oof_probs = cross_val_predict(
        model, X_train, y_train, cv=5, method="predict_proba",
    )[:, 1]

    # Held-out test predictions (already unseen by the final model)
    test_probs = score_customers(model, X_test)

    # Stitch together, aligned to df_full's original index
    churn_probs = np.empty(len(df_full), dtype=float)
    churn_probs[X_train.index] = oof_probs
    churn_probs[X_test.index]  = test_probs

    df_econ  = enrich_customer_data(df_full, churn_probs)
    seg_econ = build_segment_economics(df_econ)

    # 5. Persist
    print("\n[5/5] Saving artifacts ...")
    save_model(model, metrics, feature_names)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_econ.to_csv(ARTIFACTS_DIR / "customer_economics.csv", index=False)
    seg_econ.to_csv(ARTIFACTS_DIR / "segment_economics.csv", index=False)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(ARTIFACTS_DIR / "feature_importance.csv")

    print(f"\n[OK] All artifacts saved to {ARTIFACTS_DIR.resolve()}")
    print("\nTop-10 features by importance:")
    for feat, score in imp.head(10).items():
        print(f"      {feat:35s}  {score:.4f}")

    print("\nSegment economics:")
    print(seg_econ.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Done!  Launch the dashboard with:  streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the Telco churn CSV (default: data/...)")
    args = parser.parse_args()
    main(args.data)
