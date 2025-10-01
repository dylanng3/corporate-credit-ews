# src/calibrate.py
# Isotonic calibration + capacity cutoffs (no Platt, no HTML)
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import joblib, matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# -------- IO --------
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".parquet" or path.with_suffix(".parquet").exists():
        return pd.read_parquet(path if path.suffix.lower()==".parquet" else path.with_suffix(".parquet"))
    if path.suffix.lower()==".csv" or path.with_suffix(".csv").exists():
        return pd.read_csv(path if path.suffix.lower()==".csv" else path.with_suffix(".csv"))
    raise FileNotFoundError(f"Not found: {path}")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# -------- Metrics/plots --------
def ks_score(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))

def ece_score(y, p, n_bins=10):
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return float(sum(len(g)/len(df)*abs(g["y"].mean()-g["p"].mean()) for _, g in df.groupby("bin")))

def plot_curves(y, p, outdir: Path):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(mean, frac, "o-"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Reliability")
    plt.savefig(outdir/"calibration.png", bbox_inches="tight"); plt.close()
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.savefig(outdir/"pr_curve.png", bbox_inches="tight"); plt.close()

# -------- Main --------
def run_pipeline(df: pd.DataFrame, y_col: str, score_col: str,
                 red_pct: float, amber_pct: float, test_size: float, seed: int,
                 outdir: Path):
    ensure_dir(outdir)
    df = df.dropna(subset=[y_col, score_col]).copy()
    y = df[y_col].astype(int).values
    s = df[score_col].astype(float).values

    s_tr, s_te, y_tr, y_te = train_test_split(s, y, test_size=test_size, stratify=y, random_state=seed) # type: ignore

    # Isotonic calibration (only)
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_tr, y_tr)
    p_all = iso.predict(s) # type: ignore
    p_te  = iso.predict(s_te)

    metrics = {
        "AUC":   float(roc_auc_score(y_te, p_te)),
        "PR_AUC":float(average_precision_score(y_te, p_te)),
        "KS":    float(ks_score(y_te, p_te)),
        "Brier": float(brier_score_loss(y_te, p_te)),
        "ECE":   float(ece_score(y_te, p_te)),
    }

    # Capacity cutoffs
    thr = {
        "red":   float(np.quantile(p_all, 1 - red_pct)),
        "amber": float(np.quantile(p_all, 1 - red_pct - amber_pct)),
    }
    thr_json = {
        "capacity": {"thresholds": thr, "red_pct": red_pct, "amber_pct": amber_pct},
        "calibration": "isotonic"
    }

    # Mapping output
    df["prob_calibrated"] = p_all
    df["score_ews"] = (100*(1 - df["prob_calibrated"])).round(2)
    df["tier"] = np.where(df["prob_calibrated"]>=thr["red"], "RED",
                   np.where(df["prob_calibrated"]>=thr["amber"], "AMBER", "GREEN"))

    # Save
    joblib.dump(iso, outdir/"calibrator.pkl")
    df.to_csv(outdir/"mapping.csv", index=False)
    (outdir/"thresholds.json").write_text(json.dumps(thr_json, indent=2), encoding="utf-8")
    plot_curves(y_te, p_te, outdir)

    return {"metrics": metrics, "thresholds": thr_json}

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="Isotonic calibration + capacity cutoffs (no Platt, no HTML)")
    ap.add_argument("--input", required=True, help="scores_raw.csv (khuyến nghị)")
    ap.add_argument("--y-col", default="event_h12m")
    ap.add_argument("--score-col", default="score_raw", help="Điểm thô cần calibrate (không dùng prob_calibrated)")
    ap.add_argument("--red-pct", type=float, default=0.15)
    ap.add_argument("--amber-pct", type=float, default=0.10)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="artifacts/calibration")
    return ap.parse_args()

def main():
    args = parse_args()
    df = read_table(Path(args.input))
    out = run_pipeline(df, args.y_col, args.score_col, args.red_pct, args.amber_pct,
                       args.test_size, args.seed, Path(args.outdir))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

# python src/calibrate.py --input data/processed/scores_raw.csv --y-col event_h12m --score-col score_raw --red-pct 0.15 --amber-pct 0.10
