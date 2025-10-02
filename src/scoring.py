#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Scoring for EWS using absolute PD cutoffs
- Input: features + trained model + thresholds.json (absolute)
- Output: customer_id, prob_default_12m_calibrated, score_ews, tier, action
"""

import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------- Utils ----------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")

# ---------------- Tier & Action ----------------
def map_tier(p: float, thr: dict) -> str:
    if p >= thr["red"]:
        return "Red"
    if p >= thr["amber"]:
        return "Amber"
    return "Green"

def action_for(tier: str) -> str:
    return {
        "Green": "Theo dõi định kỳ; cập nhật BCTC đúng hạn.",
        "Amber": "Soát xét RM ≤10 ngày; yêu cầu management accounts; kiểm tra công nợ; hạn chế tăng hạn mức.",
        "Red":   "Họp KH ≤5 ngày; lập kế hoạch dòng tiền 13 tuần; xem xét covenant tightening/TSĐB; đưa watchlist."
    }[tier]

def load_thresholds(thr_path: Path) -> dict:
    meta = json.loads(thr_path.read_text(encoding="utf-8"))
    if "absolute" in meta and "thresholds" in meta["absolute"]:
        return {
            "red": float(meta["absolute"]["thresholds"]["red"]),
            "amber": float(meta["absolute"]["thresholds"]["amber"])
        }
    raise ValueError("thresholds.json không chứa absolute cutoffs hợp lệ")

# ---------------- Pipeline ----------------
def run_pipeline(features_path: Path, model_path: Path, thr_path: Path,
                 asof: str, outdir: Path):

    ensure_dir(outdir)
    # 1. Load features + model
    X = read_table(features_path)
    bundle = joblib.load(model_path)
    
    # Handle different model bundle structures
    if isinstance(bundle, dict):
        if "calibrated" in bundle:
            model = bundle["calibrated"]  # CalibratedClassifierCV from train_baseline.py
        elif "model" in bundle:
            model = bundle["model"]
        else:
            raise ValueError("Model bundle không có 'calibrated' hoặc 'model' key")
    else:
        model = bundle

    # Get feature columns from bundle if available
    if isinstance(bundle, dict) and "features" in bundle:
        feature_cols = bundle["features"]
        X_model = X[feature_cols]
    else:
        # Fallback: drop customer_id and use remaining columns
        X_model = X.drop(columns=["customer_id"], errors='ignore')

    # 2. Predict calibrated PD
    probs = model.predict_proba(X_model)[:, 1]

    # 3. Load absolute thresholds
    thr = load_thresholds(thr_path)

    # 4. Build dataframe
    df = pd.DataFrame({
        "customer_id": X["customer_id"],
        "prob_default_12m_calibrated": probs
    })
    df["score_ews"] = (100 * (1 - df["prob_default_12m_calibrated"])).round(2)
    df["tier"] = df["prob_default_12m_calibrated"].apply(lambda p: map_tier(p, thr))
    df["action"] = df["tier"].map(action_for)

    # 5. Save
    out_file = outdir / f"ews_scored_{asof}.csv"
    df.to_csv(out_file, index=False)

    thr_used = {"absolute": thr}
    (outdir / "thresholds_used.json").write_text(json.dumps(thr_used, indent=2), encoding="utf-8")

    return {"output_file": str(out_file), "thresholds": thr}

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="EWS Scoring (absolute cutoffs)")
    ap.add_argument("--features", required=True, help="Feature file (.parquet or .csv)")
    ap.add_argument("--model", required=True, help="Trained model.pkl")
    ap.add_argument("--thresholds", default="artifacts/calibration/thresholds.json",
                    help="thresholds.json từ calibrate.py (absolute)")
    ap.add_argument("--asof", required=True, help="As-of date (YYYY-MM-DD)")
    ap.add_argument("--outdir", default="artifacts/scoring")
    return ap.parse_args()

def main():
    args = parse_args()
    out = run_pipeline(Path(args.features), Path(args.model), Path(args.thresholds),
                       args.asof, Path(args.outdir))
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()


# python src/scoring.py --features data/processed/feature_ews.parquet --model artifacts/models/model_lgbm.pkl --thresholds artifacts/calibration/thresholds.json --asof 2025-06-30