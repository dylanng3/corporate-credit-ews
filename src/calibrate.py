# src/calibrate.py
# Post-hoc calibration cho Early Warning System (KHDN)
# - Input: CSV/Parquet có cột nhãn (y) và cột dự báo thô (score_raw) — càng cao càng rủi ro
# - Calibration: Platt (logistic) hoặc Isotonic (sklearn)
# - Cutoff: theo Youden J hoặc theo công suất xử lý (capacity: top x% = RED, tiếp y% = AMBER)
# - Mapping: tạo score_ews = round(100*(1 - calibrated_prob), 2) và tier (GREEN/AMBER/RED)
# - Outputs: artifacts/ (calibrator.pkl, thresholds.json, mapping.csv, report.html, calibration.png, pr_curve.png)
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import joblib


# -----------------------------
# IO helpers
# -----------------------------
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # thử song song
    p_parq, p_csv = path.with_suffix(".parquet"), path.with_suffix(".csv")
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không tìm thấy file: {path}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Metrics & plotting
# -----------------------------
def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) với bin theo quantile."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    ece = 0.0
    for _, g in df.groupby("bin"):
        if len(g) == 0: 
            continue
        conf = g["p"].mean()
        acc  = g["y"].mean()
        w    = len(g) / len(df)
        ece += w * abs(acc - conf)
    return float(ece)

def plot_reliability_and_pr(y_true: np.ndarray, y_prob: np.ndarray, outdir: Path, tag: str):
    # Reliability
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0,1],[0,1], "--", label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Reliability curve — {tag}")
    plt.legend()
    plt.savefig(outdir / f"calibration_{tag}.png", bbox_inches="tight"); plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve — {tag}")
    plt.savefig(outdir / f"pr_curve_{tag}.png", bbox_inches="tight"); plt.close()


# -----------------------------
# Calibration models
# -----------------------------
class PlattCalibrator:
    """Platt scaling = logistic regression trên 1 feature (score_raw)."""
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, score_raw: np.ndarray, y: np.ndarray):
        X = score_raw.reshape(-1,1)
        self.model.fit(X, y)
        return self

    def predict_proba(self, score_raw: np.ndarray) -> np.ndarray:
        X = score_raw.reshape(-1,1)
        return self.model.predict_proba(X)[:,1]

class IsotonicCalibrator:
    """Isotonic regression: ánh xạ đơn điệu từ score_raw → prob."""
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, score_raw: np.ndarray, y: np.ndarray):
        self.model.fit(score_raw, y)
        return self

    def predict_proba(self, score_raw: np.ndarray) -> np.ndarray:
        return self.model.predict(score_raw)

def build_calibrator(method: str):
    if method.lower() in ["platt", "logistic", "platt_scaling"]:
        return PlattCalibrator()
    if method.lower() in ["isotonic", "iso"]:
        return IsotonicCalibrator()
    raise ValueError("method phải là 'platt' hoặc 'isotonic'")


# -----------------------------
# Thresholding
# -----------------------------
def thresholds_capacity(probs: np.ndarray, red_pct=0.10, amber_pct=0.10) -> Dict[str, float]:
    """RED = top red_pct, AMBER = tiếp amber_pct, GREEN = còn lại."""
    return {
        "red":   float(np.quantile(probs, 1 - red_pct)),
        "amber": float(np.quantile(probs, 1 - red_pct - amber_pct))
    }

def youden_cutoff(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    t_star = float(thr[idx])
    return t_star, {"youden": t_star, "tpr": float(tpr[idx]), "fpr": float(fpr[idx])}


def assign_tier(probs: np.ndarray, thr: Dict[str, float], mode: str = "capacity") -> np.ndarray:
    tiers = np.empty_like(probs, dtype=object)
    if mode == "capacity":
        tiers[probs >= thr["red"]]   = "RED"
        tiers[(probs < thr["red"]) & (probs >= thr["amber"])] = "AMBER"
        tiers[probs < thr["amber"]]  = "GREEN"
    elif mode == "youden":
        t = thr["youden"]
        tiers[probs >= t] = "RED"
        tiers[probs <  t] = "GREEN"
    else:
        raise ValueError("mode phải là 'capacity' hoặc 'youden'")
    return tiers


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    method: str,
    cut_mode: str,
    red_pct: float,
    amber_pct: float,
    test_size: float,
    seed: int,
    outdir: Path
):
    ensure_dir(outdir)

    # Chuẩn bị dữ liệu
    df = df.dropna(subset=[y_col, score_col]).copy()
    y = df[y_col].astype(int).values
    s = df[score_col].astype(float).values

    # Split (stratified)
    s_tr, s_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        s, y, np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed
    )

    # Fit calibrator trên train
    calib = build_calibrator(method)
    calib.fit(s_tr, y_tr)

    # Predict calibrated probs
    p_tr = calib.predict_proba(s_tr)
    p_te = calib.predict_proba(s_te)
    p_all = calib.predict_proba(s)

    # Metrics trên holdout
    auc  = roc_auc_score(y_te, p_te)
    ap   = average_precision_score(y_te, p_te)
    brier= brier_score_loss(y_te, p_te)
    ks   = ks_score(y_te, p_te)
    ece  = ece_score(y_te, p_te, n_bins=10)

    # Plots
    plot_reliability_and_pr(y_te, p_te, outdir, tag=method)

    # Cutoffs
    result_thresholds = {}
    if cut_mode == "capacity":
        thr = thresholds_capacity(p_all, red_pct=red_pct, amber_pct=amber_pct)
        result_thresholds["capacity"] = {"red_pct": red_pct, "amber_pct": amber_pct, "thresholds": thr}
        tiers = assign_tier(p_all, thr, mode="capacity")
    elif cut_mode == "youden":
        t_star, info = youden_cutoff(y_te, p_te)
        result_thresholds["youden"] = info
        tiers = assign_tier(p_all, {"youden": t_star}, mode="youden")
    else:
        raise ValueError("cut_mode phải là 'capacity' hoặc 'youden'")

    # Mapping output
    out = df.copy()
    out["prob_calibrated"] = p_all
    out["score_ews"] = np.round(100*(1 - out["prob_calibrated"]), 2)
    out["tier"] = tiers

    # Lưu artifacts
    joblib.dump(calib, outdir / "calibrator.pkl")
    out.to_csv(outdir / "mapping.csv", index=False)
    (outdir / "thresholds.json").write_text(json.dumps(result_thresholds, indent=2), encoding="utf-8")

    # HTML report
    report = f"""
    <html><head><meta charset="utf-8"><title>EWS Calibration Report</title></head>
    <body>
      <h1>EWS — Calibration ({method})</h1>
      <h2>Holdout metrics</h2>
      <ul>
        <li>AUC: {auc:.4f}</li>
        <li>PR-AUC: {ap:.4f}</li>
        <li>KS: {ks:.4f}</li>
        <li>Brier: {brier:.4f}</li>
        <li>ECE: {ece:.4f}</li>
      </ul>
      <h2>Calibration & PR</h2>
      <img src="calibration_{method}.png" width="420" />
      <img src="pr_curve_{method}.png" width="420" />
      <h2>Cutoff</h2>
      <pre>{json.dumps(result_thresholds, indent=2)}</pre>
      <h2>Mapping</h2>
      <p><code>mapping.csv</code> chứa các cột: <b>prob_calibrated</b>, <b>score_ews</b>, <b>tier</b>.</p>
      <p><b>score_ews = round(100 * (1 - prob_calibrated), 2)</b>.</p>
    </body></html>
    """
    (outdir / "report.html").write_text(report, encoding="utf-8")

    return {
        "metrics": {"AUC": auc, "PR_AUC": ap, "KS": ks, "Brier": brier, "ECE": ece},
        "cut_mode": cut_mode,
        "thresholds": result_thresholds,
        "outputs": {
            "calibrator": str((outdir / "calibrator.pkl").absolute()),
            "mapping": str((outdir / "mapping.csv").absolute()),
            "report": str((outdir / "report.html").absolute())
        }
    }


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Calibrate probabilities & map tiers for EWS.")
    p.add_argument("--input", type=str, required=True,
                   help="CSV/Parquet chứa cột nhãn và cột score_raw (càng cao càng rủi ro).")
    p.add_argument("--y-col", type=str, default="event_h12m", help="Tên cột nhãn (0/1).")
    p.add_argument("--score-col", type=str, default="score_raw",
                   help="Tên cột score đầu vào (continuous; càng cao càng rủi ro).")
    p.add_argument("--method", type=str, default="isotonic", choices=["isotonic","platt"],
                   help="Phương pháp calibration.")
    p.add_argument("--cut-mode", type=str, default="capacity", choices=["capacity","youden"],
                   help="Chọn cách đặt ngưỡng & tier.")
    p.add_argument("--red-pct", type=float, default=0.10, help="RED = top x (capacity mode).")
    p.add_argument("--amber-pct", type=float, default=0.10, help="AMBER = tiếp y (capacity mode).")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="artifacts_calibration")
    return p.parse_args()

def main():
    args = parse_args()
    df = read_table(Path(args.input))
    summary = run_pipeline(
        df=df,
        y_col=args.y_col,
        score_col=args.score_col,
        method=args.method,
        cut_mode=args.cut_mode,
        red_pct=args.red_pct,
        amber_pct=args.amber_pct,
        test_size=args.test_size,
        seed=args.seed,
        outdir=Path(args.outdir)
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

# python src/calibrate.py --input data/processed/scores_raw.csv --y-col event_h12m --score-col score_raw --method isotonic --cut-mode capacity --red-pct 0.15 --amber-pct 0.10 --outdir artifacts/calibration_iso