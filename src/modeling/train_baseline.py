# src/train_baseline.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)

# ---------- Utils ----------
def read_features(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet" or p.with_suffix(".parquet").exists():
        return pd.read_parquet(p if p.suffix==".parquet" else p.with_suffix(".parquet"))
    if p.suffix == ".csv" or p.with_suffix(".csv").exists():
        return pd.read_csv(p if p.suffix==".csv" else p.with_suffix(".csv"))
    raise FileNotFoundError(p)

def ensure_dir(d: Path): d.mkdir(parents=True, exist_ok=True)

def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_like = {"customer_id","sector_code","size_bucket",target_col}
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_like]
    zs_cols = [c for c in numeric if c.endswith("__zs_sector_size")]
    return zs_cols if len(zs_cols) >= 5 else numeric

def thresholds_by_quota(probs: np.ndarray, red_pct=0.05, amber_pct=0.10) -> Dict[str,float]:
    return {
        "red":   float(np.quantile(probs, 1 - red_pct)),
        "amber": float(np.quantile(probs, 1 - red_pct - amber_pct))
    }

def assign_tier(p: float, thr: Dict[str,float]) -> str:
    return "RED" if p >= thr["red"] else ("AMBER" if p >= thr["amber"] else "GREEN")

def plot_calibration_pr(y_true: np.ndarray, y_prob: np.ndarray, outdir: Path, name: str):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(mean_pred, frac_pos, "o-"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("Predicted prob"); plt.ylabel("Fraction of positives"); plt.title(f"Reliability — {name}")
    plt.savefig(outdir/f"calibration_{name}.png", bbox_inches="tight"); plt.close()
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name}")
    plt.savefig(outdir/f"pr_curve_{name}.png", bbox_inches="tight"); plt.close()

# ---------- Train + Calibrate + Save ----------
def train_and_calibrate(
    df: pd.DataFrame,
    target_col: str = "event_h12m",
    test_size: float = 0.2,
    seed: int = 42,
    red_pct: float = 0.05,
    amber_pct: float = 0.10,
    outdir: Path = Path("artifacts/models")
) -> Dict[str, Any]:

    ensure_dir(outdir)
    df = df.dropna(subset=[target_col]).copy()

    feats = select_feature_columns(df, target_col)
    X = df[feats].values
    y = df[target_col].astype(int).values

    # Split with indices tracking
    indices = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, indices, test_size=test_size, stratify=y, random_state=seed # type: ignore
    )

    # LightGBM (imbalanced-aware)
    pos = y_tr.mean()
    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, max_depth=6,
        min_child_samples=10, reg_lambda=0.1, random_state=seed,
        objective="binary", n_jobs=-1,
        scale_pos_weight=float((1 - pos) / max(pos, 1e-6)),
        verbose=-1
    )
    lgbm.fit(X_tr, y_tr)

    # Isotonic calibration (cv=5)
    calib = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv=5)  # type: ignore
    calib.fit(X_tr, y_tr)

    p_te  = calib.predict_proba(X_te)[:,1]
    p_all = calib.predict_proba(X)[:,1]

    # Metrics (holdout)
    metrics = {
        "AUC":   float(roc_auc_score(y_te, p_te)),
        "PR_AUC":float(average_precision_score(y_te, p_te)),
        "KS":    float(ks_score(y_te, p_te)),
        "Brier": float(brier_score_loss(y_te, p_te)),
    }

    # Thresholds & tiers (capacity)
    thr = thresholds_by_quota(p_all, red_pct=red_pct, amber_pct=amber_pct)
    tiers_all = [assign_tier(p, thr) for p in p_all]

    # Save model artifacts
    joblib.dump({"base": lgbm, "calibrated": calib, "features": feats}, outdir/"model_lgbm.pkl")
    (outdir/"thresholds.json").write_text(
        json.dumps({"percentiles":{"red_pct":red_pct,"amber_pct":amber_pct},"thresholds":thr}, indent=2),
        encoding="utf-8"
    )
    
    # Save baseline metrics for monitoring (IMPORTANT!)
    (outdir/"baseline_metrics.json").write_text(
        json.dumps({
            "auc": metrics["AUC"],
            "pr_auc": metrics["PR_AUC"],
            "ks": metrics["KS"],
            "brier": metrics["Brier"]
        }, indent=2),
        encoding="utf-8"
    )

    # Save scores for trace
    df_out = df.copy()
    df_out["prob_calibrated"] = p_all
    df_out["tier"] = tiers_all
    df_out["is_test"] = False  # Mark all as train initially
    df_out.loc[df.index[idx_te], "is_test"] = True  # Mark test rows
    keep_id = [c for c in ["customer_id","sector_code","size_bucket"] if c in df_out.columns]
    cols = keep_id + ["prob_calibrated","tier",target_col,"is_test"]
    df_out[cols].to_csv("data/processed/scores_calibrated.csv", index=False)
    
    # CRITICAL: Save train/test features separately for proper monitoring
    df_train = df.iloc[idx_tr].copy()
    df_test = df.iloc[idx_te].copy()
    
    # Save to data/processed/ for monitoring
    train_features_path = Path("data/processed/feature_ews_train.parquet")
    test_features_path = Path("data/processed/feature_ews_test.parquet")
    train_features_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(train_features_path, index=False)
    df_test.to_parquet(test_features_path, index=False)
    
    print(f"\nSaved for monitoring:")
    print(f"   Train set: {train_features_path} ({len(df_train)} rows, {y_tr.mean():.2%} default)")
    print(f"   Test set:  {test_features_path} ({len(df_test)} rows, {y_te.mean():.2%} default)")
    print(f"   Baseline metrics: {outdir}/baseline_metrics.json")
    print(f"\n To monitor on UNSEEN data (correct way):")
    print(f"   python run_monitoring.py \\")
    print(f"     --baseline-features {train_features_path} \\")
    print(f"     --current-features {test_features_path}")
    print(f"\n  IMPORTANT: Test set metrics should match baseline_metrics.json!")
    print(f"   Expected: AUC~{metrics['AUC']:.3f}, PR-AUC~{metrics['PR_AUC']:.3f}")
    print(f"   NOT: AUC~0.99, PR-AUC~0.88 (that's overfitting on training data!)")

    # Plots
    plot_calibration_pr(y_te, p_te, outdir, name="lgbm")

    # SHAP (global summary)
    try:
        explainer = shap.TreeExplainer(lgbm)
        shap_vals = explainer.shap_values(X)
        shap_vals = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals)==2 else shap_vals
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(outdir/"shap_summary.csv", index=False)
        shap.summary_plot(shap_vals, features=X, feature_names=feats, max_display=20, show=False)
        plt.tight_layout(); plt.savefig(outdir/"shap_summary.png", bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"SHAP failed: {e}", file=sys.stderr)

    return {"model":"lgbm","features_used":feats,"metrics":metrics,"thresholds":thr,"artifacts_dir":str(outdir.absolute())}

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Train baseline EWS (LightGBM) + isotonic calibration + SHAP")
    p.add_argument("--features", required=True, type=str, help="Path to features file (.parquet/.csv)")
    p.add_argument("--test-size", default=0.2, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--red-pct", default=0.05, type=float)
    p.add_argument("--amber-pct", default=0.10, type=float)
    p.add_argument("--outdir", default="artifacts/models", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    df = read_features(Path(args.features))
    out = train_and_calibrate(
        df=df, target_col="event_h12m", test_size=args.test_size, seed=args.seed,
        red_pct=args.red_pct, amber_pct=args.amber_pct, outdir=Path(args.outdir)
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

# python src/train_baseline.py --features data/processed/feature_ews.parquet --outdir artifacts/models
