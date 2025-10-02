#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Isotonic Calibration + Absolute Cutoffs for EWS
- Fit isotonic regression on raw model scores
- Output calibrated probabilities, EWS score (0–100), and tier (Green/Amber/Red)
- Thresholds defined in absolute PD terms (e.g., Red ≥ 20%, Amber ≥ 5%)
"""

import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ---------------- Utils ----------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")

# ---------------- Calibration ----------------
def isotonic_fit_predict(x: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    p = iso.predict(x)
    return iso, np.clip(p, 1e-6, 1 - 1e-6)

# ---------------- Pipeline ----------------
def run_pipeline(df: pd.DataFrame, y_col: str, score_col: str,
                 red_thr: float, amber_thr: float, outdir: Path):
    ensure_dir(outdir)
    df = df.dropna(subset=[y_col, score_col]).copy()
    y = df[y_col].astype(int).values
    s = df[score_col].astype(float).values

    positives = int(y.sum()) # type: ignore
    if positives < 100:
        print(f"[WARN] Positives quá ít ({positives}), isotonic có thể không ổn định.")

    # 1. Fit isotonic & predict calibrated probabilities
    iso, p_all = isotonic_fit_predict(s, y) # type: ignore

    # 2. Metrics đánh giá calibration
    metrics = {
        "AUC": float(roc_auc_score(y, p_all)), # type: ignore
        "PR_AUC": float(average_precision_score(y, p_all)), # type: ignore
        "Brier": float(brier_score_loss(y, p_all)) # type: ignore
    }

    # 3. Áp dụng absolute cutoff
    thresholds = {"red": red_thr, "amber": amber_thr}
    df["prob_calibrated"] = p_all
    df["score_ews"] = (100 * (1 - df["prob_calibrated"])).round(2)
    df["tier"] = np.where(df["prob_calibrated"] >= thresholds["red"], "Red",
                   np.where(df["prob_calibrated"] >= thresholds["amber"], "Amber", "Green"))

    # 4. Save outputs
    joblib.dump(iso, outdir / "calibrator.pkl")
    df.to_csv(outdir / "mapping.csv", index=False)
    (outdir / "thresholds.json").write_text(json.dumps({
        "absolute": {"thresholds": thresholds},
        "calibration": "isotonic_refit"
    }, indent=2), encoding="utf-8")

    return {"metrics": metrics, "positives": positives, "thresholds": thresholds}

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Isotonic calibration + absolute cutoff for EWS")
    ap.add_argument("--input", required=True, help="scores_raw.csv (chứa event_h12m & score_raw)")
    ap.add_argument("--y-col", default="event_h12m")
    ap.add_argument("--score-col", default="score_raw")
    ap.add_argument("--red-thr", type=float, default=0.20,
                    help="Ngưỡng PD cho Red (mặc định 0.20)")
    ap.add_argument("--amber-thr", type=float, default=0.05,
                    help="Ngưỡng PD cho Amber (mặc định 0.05)")
    ap.add_argument("--outdir", default="artifacts/calibration")
    return ap.parse_args()

def main():
    args = parse_args()
    df = read_table(Path(args.input))
    out = run_pipeline(df, args.y_col, args.score_col,
                       args.red_thr, args.amber_thr, Path(args.outdir))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()


# python src/calibrate.py --input data/processed/scores_raw.csv --y-col event_h12m --score-col score_raw --red-thr 0.20 --amber-thr 0.05 --outdir artifacts/calibration
