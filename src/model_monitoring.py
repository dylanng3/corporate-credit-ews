#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Monitoring for Corporate Credit EWS
Lightweight monitoring: Performance metrics + PSI + Alerts
"""

import argparse, json, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

# ============================================================================
# Utils
# ============================================================================

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def ks_score(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))

# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred)),
        "ks": float(ks_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }

def check_degradation(baseline: Dict, current: Dict) -> Dict[str, bool]:
    """Return alerts if metrics degraded"""
    return {
        "auc_alert": current["auc"] < 0.70 or (baseline["auc"] - current["auc"]) > 0.05,
        "brier_alert": current["brier"] > 0.15 or (current["brier"] - baseline["brier"]) > 0.05,
        "ks_alert": (baseline["ks"] - current["ks"]) / baseline["ks"] > 0.20
    }

# ============================================================================
# Population Stability Index (PSI)
# ============================================================================

def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    PSI = Σ(actual% - expected%) × ln(actual%/expected%)
    PSI < 0.10: Stable | 0.10-0.25: Moderate | ≥0.25: Severe drift
    """
    breakpoints = np.percentile(expected.dropna(), np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0
    
    exp_counts = pd.cut(expected, bins=list(breakpoints), include_lowest=True, duplicates='drop').value_counts()
    act_counts = pd.cut(actual, bins=list(breakpoints), include_lowest=True, duplicates='drop').value_counts()
    
    exp_pct = (exp_counts / len(expected)).replace(0, 0.0001)
    act_pct = (act_counts / len(actual)).replace(0, 0.0001)
    exp_pct, act_pct = exp_pct.align(act_pct, fill_value=0.0001)
    
    return float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())

def psi_for_features(baseline_df: pd.DataFrame, current_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    results = []
    for col in features:
        if col in baseline_df.columns and col in current_df.columns:
            psi = calculate_psi(baseline_df[col], current_df[col])
            status = "Stable" if psi < 0.10 else ("Moderate" if psi < 0.25 else "Severe")
            results.append({"feature": col, "psi": round(psi, 4), "status": status})
    return pd.DataFrame(results).sort_values("psi", ascending=False)

# ============================================================================
# Main Pipeline
# ============================================================================

def run_monitoring(
    baseline_features: Path,
    baseline_metrics: Dict,
    current_features: Path,
    model: Path,
    outdir: Path
):
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"MODEL MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load data
    baseline_df = read_table(baseline_features)
    current_df = read_table(current_features)
    model_artifact = joblib.load(model)
    features = model_artifact["features"]
    
    # 1. Performance Metrics (if labels available)
    alerts = {}
    current_metrics = {}
    if "event_h12m" in current_df.columns:
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 80)
        X = current_df[features]
        y = current_df["event_h12m"].astype(int).values
        y_pred = model_artifact["calibrated"].predict_proba(X)[:, 1]
        
        current_metrics = calculate_metrics(y, y_pred)
        alerts = check_degradation(baseline_metrics, current_metrics)
        
        for metric in ["auc", "pr_auc", "ks", "brier"]:
            base = baseline_metrics.get(metric, 0)
            curr = current_metrics[metric]
            status = "⚠️" if any(alerts.values()) else "✓"
            print(f"{metric.upper():<10} Baseline: {base:.4f}  Current: {curr:.4f}  {status}")
        
        if any(alerts.values()):
            print("\n⚠️  Performance degradation detected!")
    
    # 2. PSI
    print("\n📈 POPULATION STABILITY INDEX (PSI)")
    print("-" * 80)
    psi_df = psi_for_features(baseline_df, current_df, features)
    print("\nTop 10 Features with Highest PSI:")
    print(psi_df.head(10).to_string(index=False))
    
    severe = psi_df[psi_df["status"] == "Severe"]
    if len(severe) > 0:
        print(f"\n⚠️  {len(severe)} features have severe drift (PSI ≥ 0.25)")
    
    # 3. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    issues = []
    if any(alerts.values()):
        issues.append("Performance metrics degraded")
    if len(severe) > 5:
        issues.append(f"{len(severe)} features have severe drift")
    
    if issues:
        print("\n⚠️  ALERTS:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n💡 RECOMMENDATION: Consider retraining the model with recent data")
    else:
        print("\n✅ Model performance is stable. No critical issues detected.")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "alerts": alerts,
        "psi_severe_count": int(len(severe)),
        "issues": issues
    }
    
    report_file = outdir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(report, indent=2))
    psi_df.to_csv(outdir / "psi_details.csv", index=False)
    
    print(f"\n📄 Report: {report_file}")
    print("=" * 80)

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Monitor EWS model performance")
    parser.add_argument("--baseline-features", required=True, help="Baseline features (training data)")
    parser.add_argument("--baseline-metrics", required=True, help="Baseline metrics JSON")
    parser.add_argument("--current-features", required=True, help="Current features to monitor")
    parser.add_argument("--model", required=True, help="Trained model (model_lgbm.pkl)")
    parser.add_argument("--outdir", default="artifacts/monitoring", help="Output directory")
    args = parser.parse_args()
    
    baseline_metrics = json.loads(Path(args.baseline_metrics).read_text())
    
    run_monitoring(
        baseline_features=Path(args.baseline_features),
        baseline_metrics=baseline_metrics,
        current_features=Path(args.current_features),
        model=Path(args.model),
        outdir=Path(args.outdir)
    )

if __name__ == "__main__":
    main()
