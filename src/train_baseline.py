# src/train_baseline.py
# Baseline EWS (KHDN) — Logistic & LightGBM + Isotonic calibration + SHAP + HTML report
# - Đầu vào: features_ews.parquet|csv (tạo từ src/features.py)
# - Đầu ra: artifacts/ (model, calibration, metrics, shap, thresholds, report.html)
# - Split: Stratified 80/20 (vì features hiện tại là 1 mốc as-of); có thể đổi sang time-based nếu nhiều as-of.
# - Thresholds: mặc định Red=5% cao nhất, Amber=tiếp 10%, Green=còn lại (cấu hình được).

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# ---------------------------
# Utils
# ---------------------------
def read_features(path_no_ext: Path) -> pd.DataFrame:
    """Đọc features .parquet nếu có; fallback .csv."""
    p_parquet = path_no_ext if path_no_ext.suffix == ".parquet" else path_no_ext.with_suffix(".parquet")
    p_csv     = path_no_ext if path_no_ext.suffix == ".csv"     else path_no_ext.with_suffix(".csv")
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không thấy file: {p_parquet} hoặc {p_csv}")

def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Chọn cột numeric để train; bỏ id/categorical gốc. Ưu tiên các cột đã chuẩn hoá __zs_sector_size."""
    drop_like = {"customer_id", "sector_code", "size_bucket", target_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Nếu đã tạo z-score theo sector/size, ưu tiên dùng chúng:
    zs_cols = [c for c in numeric_cols if c.endswith("__zs_sector_size")]
    if len(zs_cols) >= 5:
        return zs_cols
    # nếu chưa có đủ z-score, dùng toàn bộ numeric (trừ target), để model tự xử lý
    return [c for c in numeric_cols if c != target_col]

def compute_thresholds_by_quota(scores: np.ndarray, red_pct=0.05, amber_pct=0.10) -> Dict[str, float]:
    """Ngưỡng theo công suất vận hành: top 5% -> Red, tiếp 10% -> Amber, còn lại -> Green."""
    q_red   = np.quantile(scores, 1 - red_pct)
    q_amber = np.quantile(scores, 1 - (red_pct + amber_pct))
    return {"red": float(q_red), "amber": float(q_amber)}

def tier_assign(prob: np.ndarray, thr: Dict[str, float]) -> List[str]:
    t = []
    for p in prob:
        if p >= thr["red"]:
            t.append("RED")
        elif p >= thr["amber"]:
            t.append("AMBER")
        else:
            t.append("GREEN")
    return t

def plot_and_save_calibration(y_true, y_prob, outdir: Path, name: str):
    """Lưu calibration curve & PR curve đơn giản."""
    # Reliability curve
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Calibrated")
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("Predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Reliability curve — {name}")
    plt.savefig(outdir / f"calibration_{name}.png", bbox_inches="tight")
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve — {name}")
    plt.savefig(outdir / f"pr_curve_{name}.png", bbox_inches="tight")
    plt.close()

# ---------------------------
# Training pipeline
# ---------------------------
def train_and_calibrate(
    df: pd.DataFrame,
    target_col: str = "event_h12m",
    model_type: str = "lightgbm",
    test_size: float = 0.2,
    random_state: int = 42,
    red_pct: float = 0.05,
    amber_pct: float = 0.10,
    outdir: Path = Path("artifacts")
) -> Dict[str, Any]:

    ensure_dir(outdir)
    # Drop NA target
    df = df.dropna(subset=[target_col]).copy()
    # Select features
    feats = select_feature_columns(df, target_col)
    X = df[feats].values
    y = df[target_col].astype(int).values

    # Stratified split (vì 1 mốc as-of; nếu nhiều mốc, nên split theo thời gian)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
    )

    # ----- Model
    if model_type.lower() == "logit":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        base = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )
        base.fit(X_train_s, y_train)

        # Isotonic calibration (fit trên train bằng cross-validation để tránh overfit)
        calib = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
        calib.fit(X_train_s, y_train)
        p_test = calib.predict_proba(X_test_s)[:,1]
        p_all  = calib.predict_proba(scaler.transform(X))[:,1]

        # Save artifacts
        import joblib
        joblib.dump({"scaler": scaler, "base": base, "calibrated": calib, "features": feats}, outdir/"model_logit.pkl")

        model_name = "logit"

    else:
        # LightGBM
        pos_rate = y_train.mean()
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

        lgbm = lgb.LGBMClassifier(
            n_estimators=300,           
            learning_rate=0.05,          
            num_leaves=15,              
            subsample=0.8,              
            colsample_bytree=0.8,       
            max_depth=6,              
            min_child_samples=10,       
            reg_lambda=0.1,           
            reg_alpha=0.0,       
            random_state=random_state,
            objective="binary", 
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbose=-1                  # Tắt warnings
        )
        lgbm.fit(X_train, y_train)

        # Calibrate bằng isotonic (cv=5)
        calib = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv=5)
        calib.fit(X_train, y_train)

        p_test = calib.predict_proba(X_test)[:,1]
        p_all  = calib.predict_proba(X)[:,1]

        import joblib
        joblib.dump({"base": lgbm, "calibrated": calib, "features": feats}, outdir/"model_lgbm.pkl")

        model_name = "lgbm"

    # ----- Metrics
    auc  = roc_auc_score(y_test, p_test)
    ap   = average_precision_score(y_test, p_test)
    brier= brier_score_loss(y_test, p_test)
    ks   = ks_score(y_test, p_test)

    # ----- Thresholds & tiers
    thr = compute_thresholds_by_quota(p_all, red_pct=red_pct, amber_pct=amber_pct)
    tiers_all = tier_assign(p_all, thr)

    # Save thresholds
    (outdir/"thresholds.json").write_text(json.dumps({"percentiles": {"red_pct": red_pct, "amber_pct": amber_pct}, "thresholds": thr}, indent=2), encoding="utf-8")

    # Save scores for all rows
    df_out = df.copy()
    df_out["prob_calibrated"] = p_all
    df_out["tier"] = tiers_all
    # giữ lại id nếu có
    cols_order = [c for c in ["customer_id","sector_code","size_bucket"] if c in df_out.columns] + ["prob_calibrated","tier",target_col]
    cols_order += [c for c in df_out.columns if c not in cols_order]
    df_out[cols_order].to_csv(outdir/"scores_all.csv", index=False)

    # ----- Plots
    plot_and_save_calibration(y_test, p_test, outdir, name=model_name)

    # ----- SHAP (cho LightGBM)
    shap_summary = None
    if model_type.lower() != "logit":
        try:
            explainer = shap.TreeExplainer(lgbm)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):   # binary -> list length 2
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            shap_summary = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_summary.to_csv(outdir/"shap_summary.csv", index=False)

            # SHAP summary plot (top 20)
            shap.summary_plot(shap_vals, features=X, feature_names=feats, max_display=20, show=False)
            plt.tight_layout()
            plt.savefig(outdir/"shap_summary.png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("[WARN] SHAP plotting failed:", e, file=sys.stderr)

    # ----- Report HTML
    report_html = f"""
    <html><head><meta charset="utf-8"><title>EWS Baseline Report</title></head>
    <body>
      <h1>EWS Baseline — {model_name.upper()}</h1>
      <h2>Holdout metrics</h2>
      <ul>
        <li>AUC: {auc:.4f}</li>
        <li>PR-AUC: {ap:.4f}</li>
        <li>KS: {ks:.4f}</li>
        <li>Brier: {brier:.4f}</li>
      </ul>
      <h2>Calibration & PR curves</h2>
      <img src="calibration_{model_name}.png" width="420" />
      <img src="pr_curve_{model_name}.png" width="420" />
      <h2>Thresholds</h2>
      <pre>{json.dumps(thr, indent=2)}</pre>
      <p>Red = top {int(red_pct*100)}% (prob ≥ red), Amber = tiếp {int(amber_pct*100)}%, Green = còn lại.</p>
      <h2>Outputs</h2>
      <ul>
        <li><code>scores_all.csv</code>: Xác suất đã hiệu chỉnh + Tier</li>
        <li><code>thresholds.json</code></li>
        {"<li><code>shap_summary.csv</code> + <code>shap_summary.png</code></li>" if shap_summary is not None else ""}
      </ul>
    </body></html>
    """
    (outdir/"report.html").write_text(report_html, encoding="utf-8")

    # Summary dict
    return {
        "model": model_name,
        "features_used": feats,
        "metrics": {"AUC": auc, "PR_AUC": ap, "KS": ks, "Brier": brier},
        "thresholds": thr,
        "artifacts_dir": str(outdir.absolute())
    }

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train baseline EWS model + calibration + SHAP")
    p.add_argument("--features", type=str, required=True, help="Path to features file (parquet or csv)")
    p.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm","logit"])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--red-pct", type=float, default=0.05)
    p.add_argument("--amber-pct", type=float, default=0.10)
    p.add_argument("--outdir", type=str, default="artifacts/models")
    return p.parse_args()

def main():
    args = parse_args()
    df = read_features(Path(args.features))
    out = train_and_calibrate(
        df=df,
        target_col="event_h12m",
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        red_pct=args.red_pct,
        amber_pct=args.amber_pct,
        outdir=Path(args.outdir)
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

# python src/train_baseline.py --features data/processed/feature_ews.parquet --model lightgbm