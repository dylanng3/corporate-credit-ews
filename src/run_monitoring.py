#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os, socket
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.isotonic import IsotonicRegression

EPS = 1e-12

# -------- JSON Encoder for NumPy types --------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Check for NaN and convert to null
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# -------- IO --------
def read_table(p: Path) -> pd.DataFrame:
    s = p.suffix.lower()
    if s in {".parquet", ".pq"}: return pd.read_parquet(p)
    if s == ".feather": return pd.read_feather(p)
    return pd.read_csv(p, low_memory=False)

# -------- Metrics --------
def ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def calc_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str,float]:
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "ks": ks_score(y, p),
        "brier": brier_score_loss(y, p),
    } # type: ignore

def degrade_flags(base: Dict[str,float], cur: Dict[str,float]) -> Dict[str,bool]:
    ks_base = base.get("ks", 0.0)
    brier_base = base.get("brier", 0.0)
    pr_auc_base = base.get("pr_auc", 0.0)
    
    return {
        "auc":  (cur["auc"] < 0.70) or (base.get("auc",0) - cur["auc"] > 0.05),
        "brier": (cur["brier"] - brier_base > 0.01) or (brier_base > 0 and (cur["brier"] - brier_base) / brier_base > 0.20),
        "ks": (ks_base > 0) and ((ks_base - cur["ks"]) / max(ks_base, EPS) > 0.20),
        "pr_auc": (pr_auc_base > 0) and (pr_auc_base - cur.get("pr_auc", 0) > 0.05),
    }

# -------- PSI --------
def _psi_one(exp: pd.Series, act: pd.Series, bins: int=10) -> float:
    exp, act = exp.dropna().astype(float), act.dropna().astype(float)
    if len(exp)<max(50,bins*5) or len(act)<max(50,bins*5): return np.nan
    edges = np.unique(np.percentile(exp, np.linspace(0,100,bins+1)))
    if len(edges)<3: return np.nan
    ec = pd.cut(exp, edges, include_lowest=True, right=True, duplicates="drop").value_counts().sort_index() # type: ignore
    ac = pd.cut(act, edges, include_lowest=True, right=True, duplicates="drop").value_counts().sort_index() # type: ignore
    ep, ap = (ec/ec.sum()).replace(0,EPS), (ac/ac.sum()).replace(0,EPS)
    return float(((ap-ep)*np.log(ap/ep)).sum())

def psi_table(base_df: pd.DataFrame, cur_df: pd.DataFrame, feats: List[str], bins:int=10) -> pd.DataFrame:
    rows=[]
    for c in feats:
        if c not in base_df or c not in cur_df: continue
        # Only calculate PSI for numeric features
        if not pd.api.types.is_numeric_dtype(base_df[c]) or not pd.api.types.is_numeric_dtype(cur_df[c]):
            continue
        v=_psi_one(base_df[c], cur_df[c], bins)
        # Calculate missing rates
        missing_base = base_df[c].isna().mean()
        missing_cur = cur_df[c].isna().mean()
        missing_delta = missing_cur - missing_base
        
        # Severity classification
        if np.isnan(v):
            status = "insufficient"
            severity = "unknown"
        elif v < 0.10:
            status = "stable"
            severity = "ok"
        elif v < 0.25:
            status = "moderate"
            severity = "warn"
        else:
            status = "severe"
            severity = "severe"
        
        rows.append({
            "feature": c, 
            "psi": (None if np.isnan(v) else round(v,4)), 
            "status": status,
            "severity": severity,
            "missing_rate": round(missing_cur, 4),
            "missing_delta": round(missing_delta, 4)
        })
    return pd.DataFrame(rows).sort_values(["severity","psi"], ascending=[False,False], na_position="last")

# -------- Calibration Analysis --------
def calc_calibration_stats(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict:
    """Calculate calibration metrics and decile analysis"""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop") + 1
    
    deciles = []
    for dec in sorted(df["decile"].unique()):
        subset = df[df["decile"] == dec]
        deciles.append({
            "decile": int(dec),
            "pd_avg": round(subset["y_pred"].mean(), 5),
            "odr": round(subset["y_true"].mean(), 5),
            "count": len(subset)
        })
    
    # Calculate slope/intercept via simple linear regression on decile means
    if len(deciles) >= 3:
        pds = np.array([d["pd_avg"] for d in deciles])
        odrs = np.array([d["odr"] for d in deciles])
        slope = float(np.polyfit(pds, odrs, 1)[0])
        intercept = float(np.polyfit(pds, odrs, 1)[1])
    else:
        slope, intercept = 1.0, 0.0
    
    return {
        "method": "decile_linear",  # Changed from "isotonic" to reflect actual method
        "slope": round(slope, 4),
        "intercept": round(intercept, 5),
        "expected_odr": round(y_true.mean(), 5),
        "deciles": deciles
    }

# -------- Grade Mix & Approval Rate --------
def calc_grade_mix(y_pred: np.ndarray, cutoffs: Dict[str, float]) -> Tuple[Dict, float]:
    """Calculate grade distribution and approval rate"""
    # Define grade boundaries (example PD thresholds)
    boundaries = [
        (0.00, 0.005, "A"),
        (0.005, 0.01, "B"),
        (0.01, 0.02, "C"),
        (0.02, 0.05, "D"),
        (0.05, 0.10, "E"),
        (0.10, 0.20, "F"),
        (0.20, 1.00, "G")
    ]
    
    grade_counts = {g: 0 for _, _, g in boundaries}
    for pred in y_pred:
        for low, high, grade in boundaries:
            if low <= pred < high:
                grade_counts[grade] += 1
                break
    
    total = len(y_pred)
    grade_mix = {g: round(count / total, 4) for g, count in grade_counts.items()}
    
    # Approval rate: below amber threshold
    approval_rate = round((y_pred < cutoffs.get("amber_max_pd", 0.05)).mean(), 4)
    
    return grade_mix, approval_rate

# -------- Data Quality Checks --------
def calc_data_quality(base_df: pd.DataFrame, cur_df: pd.DataFrame, feats: List[str]) -> Dict:
    """Calculate overall data quality metrics"""
    # Overall missing rate
    missing_rate_overall = round(cur_df[feats].isna().mean().mean(), 4)
    
    # Top features with increased missing
    missing_increases = []
    for feat in feats:
        if feat not in base_df.columns or feat not in cur_df.columns:
            continue
        delta = cur_df[feat].isna().mean() - base_df[feat].isna().mean()
        if delta > 0.01:  # Only significant increases
            missing_increases.append({
                "feature": feat,
                "delta": round(delta, 4)
            })
    
    missing_increases = sorted(missing_increases, key=lambda x: x["delta"], reverse=True)[:10]
    
    # Schema changes
    base_cols = set(base_df.columns)
    cur_cols = set(cur_df.columns)
    schema_changes = {
        "added_columns": list(cur_cols - base_cols),
        "removed_columns": list(base_cols - cur_cols),
        "dtype_changes": []
    }
    
    for col in base_cols & cur_cols:
        if base_df[col].dtype != cur_df[col].dtype:
            schema_changes["dtype_changes"].append({
                "column": col,
                "old_dtype": str(base_df[col].dtype),
                "new_dtype": str(cur_df[col].dtype)
            })
    
    return {
        "missing_rate_overall": missing_rate_overall,
        "top_missing_increases": missing_increases,
        "schema_changes": schema_changes
    }

# -------- Metadata Collection --------
def collect_metadata(cur_df: pd.DataFrame, model_path: Path) -> Dict:
    """Collect run metadata"""
    timestamp = datetime.now()
    
    # Try to get git commit (if available)
    code_commit = os.environ.get("GIT_COMMIT", "unknown")
    if code_commit == "unknown":
        try:
            import subprocess
            result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], 
                                   capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                code_commit = result.stdout.strip()
        except:
            pass
    
    # Environment detection
    env = os.environ.get("ENVIRONMENT", "local")
    if "PROD" in env.upper() or "PRODUCTION" in env.upper():
        env = "prod"
    elif "STAGING" in env.upper() or "STG" in env.upper():
        env = "staging"
    else:
        env = "local"
    
    # Model version from filename or metadata
    model_version = "1.0.0"  # Default
    if "v" in model_path.stem:
        parts = model_path.stem.split("_v")
        if len(parts) > 1:
            model_version = parts[-1]
    
    return {
        "run_id": f"{timestamp.strftime('%Y-%m-%d_%H%MZ')}_v{model_version}",
        "date_run": timestamp.strftime("%Y-%m-%d"),
        "timestamp": timestamp.isoformat(),
        "env": env,
        "source": "batch",  # Could be parameterized
        "model_id": "pd_scorecard",
        "model_version": model_version,
        "code_commit": code_commit,
        "hostname": socket.gethostname()
    }

# -------- Long Format CSV Exports --------
def save_long_format_csvs(
    metadata: Dict,
    cur_metrics: Dict,
    psi_df: pd.DataFrame,
    calibration: Dict,
    grade_mix: Dict,
    approval_rate: float,
    psi_overall: float,
    outdir: Path
):
    """Save monitoring data in long format for PowerBI"""
    date_run = metadata["date_run"]
    version = metadata["model_version"]
    run_id = metadata["run_id"]
    env = metadata["env"]
    
    # 1. Metrics CSV
    metrics_rows = []
    for metric, value in cur_metrics.items():
        metrics_rows.append({
            "run_id": run_id,
            "date_run": date_run,
            "env": env,
            "model_version": version,
            "metric": metric.upper(),
            "value": round(value, 6)
        })
    metrics_rows.append({
        "run_id": run_id,
        "date_run": date_run,
        "env": env,
        "model_version": version,
        "metric": "PSI_OVERALL",
        "value": round(psi_overall, 6)
    })
    metrics_rows.append({
        "run_id": run_id,
        "date_run": date_run,
        "env": env,
        "model_version": version,
        "metric": "ApprovalRate",
        "value": round(approval_rate, 6)
    })
    for grade, pct in grade_mix.items():
        metrics_rows.append({
            "run_id": run_id,
            "date_run": date_run,
            "env": env,
            "model_version": version,
            "metric": f"Grade_{grade}",
            "value": round(pct, 6)
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    # Reorder columns for better readability
    cols = ["run_id", "date_run", "env", "model_version", "metric", "value"]
    metrics_df = metrics_df[cols]
    
    metrics_file = outdir / "monitoring_metrics.csv"
    if metrics_file.exists():
        existing = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    # De-duplicate: keep last occurrence for same date_run, model_version, metric
    metrics_df = metrics_df.drop_duplicates(subset=["date_run", "model_version", "metric"], keep="last")
    metrics_df.to_csv(metrics_file, index=False)
    
    # 2. PSI CSV
    psi_rows = []
    for _, row in psi_df.iterrows():
        psi_rows.append({
            "run_id": run_id,
            "date_run": date_run,
            "env": env,
            "model_version": version,
            "feature": row["feature"],
            "psi": row["psi"],
            "missing_rate": row["missing_rate"],
            "severity": row["severity"]
        })
    
    psi_csv_df = pd.DataFrame(psi_rows)
    # Reorder columns
    cols = ["run_id", "date_run", "env", "model_version", "feature", "psi", "missing_rate", "severity"]
    psi_csv_df = psi_csv_df[cols]
    
    psi_file = outdir / "monitoring_psi.csv"
    if psi_file.exists():
        existing = pd.read_csv(psi_file)
        psi_csv_df = pd.concat([existing, psi_csv_df], ignore_index=True)
    # De-duplicate: keep last occurrence for same date_run, model_version, feature
    psi_csv_df = psi_csv_df.drop_duplicates(subset=["date_run", "model_version", "feature"], keep="last")
    psi_csv_df.to_csv(psi_file, index=False)
    
    # 3. Calibration CSV
    calib_rows = []
    for decile_info in calibration["deciles"]:
        calib_rows.append({
            "run_id": run_id,
            "date_run": date_run,
            "env": env,
            "model_version": version,
            "decile": decile_info["decile"],
            "pd_avg": decile_info["pd_avg"],
            "odr": decile_info["odr"],
            "count": decile_info["count"]
        })
    
    calib_df = pd.DataFrame(calib_rows)
    # Reorder columns
    cols = ["run_id", "date_run", "env", "model_version", "decile", "pd_avg", "odr", "count"]
    calib_df = calib_df[cols]
    
    calib_file = outdir / "monitoring_calibration.csv"
    if calib_file.exists():
        existing = pd.read_csv(calib_file)
        calib_df = pd.concat([existing, calib_df], ignore_index=True)
    # De-duplicate: keep last occurrence for same date_run, model_version, decile
    calib_df = calib_df.drop_duplicates(subset=["date_run", "model_version", "decile"], keep="last")
    calib_df.to_csv(calib_file, index=False)

# -------- Core --------
def _run(bf: Path, bm: Dict[str,float], cf: Path, model_p: Path, outdir: Path, psi_bins:int=10, cutoffs_path: Optional[Path]=None):
    outdir.mkdir(parents=True, exist_ok=True)
    base_df, cur_df = read_table(bf), read_table(cf)
    art = joblib.load(model_p); feats: List[str] = art["features"]

    # Collect metadata
    metadata = collect_metadata(cur_df, model_p)
    
    # Load cutoffs
    if cutoffs_path and cutoffs_path.exists():
        cutoffs_data = json.loads(cutoffs_path.read_text())
        cutoffs = {
            "green_max_pd": cutoffs_data.get("absolute", {}).get("thresholds", {}).get("amber", 0.02),
            "amber_max_pd": cutoffs_data.get("absolute", {}).get("thresholds", {}).get("red", 0.05),
            "red_min_pd": cutoffs_data.get("absolute", {}).get("thresholds", {}).get("red", 0.05)
        }
    else:
        cutoffs = {"green_max_pd": 0.02, "amber_max_pd": 0.05, "red_min_pd": 0.05}
    
    # Alert thresholds
    alert_thresholds = {
        "auc_drop_pp": 5,
        "ks_min": 0.20,
        "brier_increase_abs": 0.01,
        "brier_increase_rel": 0.20,
        "pr_auc_drop_pp": 5,
        "psi_warn": 0.10,
        "psi_severe": 0.25
    }

    # Data window
    data_window = {
        "data_window_start": cur_df.index.min() if hasattr(cur_df.index, 'min') else "unknown",
        "data_window_end": cur_df.index.max() if hasattr(cur_df.index, 'max') else "unknown",
        "population_size": len(cur_df),
        "event_rate": round(cur_df["event_h12m"].mean(), 5) if "event_h12m" in cur_df.columns else None
    }
    
    # Convert dates to strings if they're datetime
    if hasattr(data_window["data_window_start"], "strftime"):
        data_window["data_window_start"] = data_window["data_window_start"].strftime("%Y-%m-%d")
    if hasattr(data_window["data_window_end"], "strftime"):
        data_window["data_window_end"] = data_window["data_window_end"].strftime("%Y-%m-%d")

    # Performance metrics (if labels available)
    cur_metrics, alerts, calibration, grade_mix, approval_rate = {}, {}, {}, {}, 0.0
    if "event_h12m" in cur_df.columns:
        X, y = cur_df[feats], cur_df["event_h12m"].astype(int).values
        y_prob = art["calibrated"].predict_proba(X)[:,1]
        cur_metrics = calc_metrics(y, y_prob) # type: ignore
        
        # Enhanced alerts with reasons
        degradation = degrade_flags(bm, cur_metrics)
        alerts = {
            "auc_alert": bool(degradation["auc"]),
            "brier_alert": bool(degradation["brier"]),
            "ks_alert": bool(degradation["ks"]),
            "pr_auc_alert": bool(degradation["pr_auc"]),
            "psi_alert": False  # Will update after PSI calculation
        }
        
        # Calibration analysis
        calibration = calc_calibration_stats(y, y_prob) # type: ignore
        
        # Grade mix and approval rate
        grade_mix, approval_rate = calc_grade_mix(y_prob, cutoffs)

    # PSI analysis
    psi = psi_table(base_df, cur_df, feats, bins=psi_bins)
    severe_cnt = int((psi["severity"]=="severe").sum())
    warn_cnt = int((psi["severity"]=="warn").sum())
    # Use median instead of mean to avoid outlier influence
    psi_overall = round(psi[psi["psi"].notna()]["psi"].median(), 4) if len(psi[psi["psi"].notna()]) > 0 else 0.0
    
    # Update PSI alert
    alerts["psi_alert"] = (psi_overall >= alert_thresholds["psi_warn"])
    
    # Data quality
    data_quality = calc_data_quality(base_df, cur_df, feats)
    
    # Enhance PSI table with feature list
    psi_by_feature = []
    for _, row in psi.iterrows():
        psi_by_feature.append({
            "feature": row["feature"],
            "psi": row["psi"] if pd.notna(row["psi"]) else None,
            "missing_rate": row["missing_rate"],
            "severity": row["severity"]
        })
    
    # Collect issues
    issues = []
    if alerts.get("auc_alert"):
        issues.append(f"AUC degradation detected: {round(cur_metrics.get('auc', 0), 4)} (baseline: {round(bm.get('auc', 0), 4)})")
    if alerts.get("ks_alert"):
        issues.append(f"KS degradation detected: {round(cur_metrics.get('ks', 0), 4)} (baseline: {round(bm.get('ks', 0), 4)})")
    if alerts.get("brier_alert"):
        issues.append(f"Brier score degradation detected: {round(cur_metrics.get('brier', 0), 4)} (baseline: {round(bm.get('brier', 0), 4)})")
    if alerts.get("pr_auc_alert"):
        issues.append(f"PR-AUC degradation detected: {round(cur_metrics.get('pr_auc', 0), 4)} (baseline: {round(bm.get('pr_auc', 0), 4)})")
    if severe_cnt > 0:
        severe_features = psi[psi["severity"]=="severe"]["feature"].tolist()
        issues.append(f"{severe_cnt} features with severe PSI: {', '.join(severe_features[:5])}")
    if warn_cnt > 0:
        issues.append(f"{warn_cnt} features with moderate PSI (warning level)")

    # Comprehensive report
    report = {
        **metadata,
        **data_window,
        "baseline_metrics": bm,
        "current_metrics": cur_metrics,
        "cutoffs": cutoffs,
        "approval_rate": approval_rate,
        "grade_mix": grade_mix,
        "calibration": calibration,
        "psi_overall": psi_overall,
        "psi_by_feature": psi_by_feature[:20],  # Top 20 for JSON
        **data_quality,
        "alert_thresholds": alert_thresholds,
        "alerts": alerts,
        "psi_severe_count": severe_cnt,
        "psi_warn_count": warn_cnt,
        "issues": issues
    }
    
    # Save JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (outdir/f"monitoring_{ts}.json").write_text(json.dumps(report, indent=2, cls=NumpyEncoder))
    
    # Save detailed PSI table
    psi.to_csv(outdir/"psi_details.csv", index=False)
    
    # Save long-format CSVs for PowerBI
    if cur_metrics:  # Only if we have performance data
        save_long_format_csvs(
            metadata, cur_metrics, psi, calibration, 
            grade_mix, approval_rate, psi_overall, outdir
        )

    # Enhanced console log
    print("\n" + "="*60)
    print("MODEL MONITORING REPORT")
    print("="*60)
    print(f"Run ID:       {metadata['run_id']}")
    print(f"Environment:  {metadata['env']}")
    print(f"Model:        {metadata['model_id']} v{metadata['model_version']}")
    print(f"Population:   {data_window['population_size']:,}")
    if data_window['event_rate']:
        print(f"Event Rate:   {data_window['event_rate']:.2%}")
    print("-"*60)
    
    if cur_metrics:
        print("PERFORMANCE METRICS:")
        for k, v in cur_metrics.items():
            baseline_val = bm.get(k, 0)
            delta = v - baseline_val
            delta_str = f"({delta:+.4f})" if delta != 0 else ""
            print(f"  {k.upper():12s}: {v:.4f} {delta_str}")
        print(f"  Approval Rate: {approval_rate:.2%}")
    
    print("-"*60)
    print("DATA DRIFT (PSI):")
    print(f"  Overall PSI:    {psi_overall:.4f}")
    print(f"  Severe shifts:  {severe_cnt}")
    print(f"  Warnings:       {warn_cnt}")
    
    print("-"*60)
    print("DATA QUALITY:")
    print(f"  Overall missing: {data_quality['missing_rate_overall']:.2%}")
    if data_quality['schema_changes']['added_columns']:
        print(f"  New columns:     {len(data_quality['schema_changes']['added_columns'])}")
    if data_quality['schema_changes']['removed_columns']:
        print(f"  Removed columns: {len(data_quality['schema_changes']['removed_columns'])}")
    
    if issues:
        print("-"*60)
        print("⚠️  ALERTS:")
        for issue in issues:
            print(f"  • {issue}")
    
    print("="*60)
    print(f"Report saved: {outdir/f'monitoring_{ts}.json'}")
    print(f"PowerBI CSVs: {outdir / 'monitoring_*.csv'}")
    print("="*60 + "\n")

# -------- Public API --------
def run_monitoring(
    baseline_features: Optional[Path]=None,
    baseline_metrics: Optional[Path]=None,
    current_features: Optional[Path]=None,
    model: Optional[Path]=None,
    outdir: Optional[Path]=None,
    cutoffs: Optional[Path]=None,
    psi_bins:int=10,
):
    base = Path(__file__).parent.parent
    bf = baseline_features or base/"data/processed/feature_ews.parquet"
    bm = baseline_metrics  or base/"artifacts/models/baseline_metrics.json"
    cf = current_features  or bf
    mp = model            or base/"artifacts/models/model_lgbm.pkl"
    od = outdir           or base/"artifacts/monitoring"
    ct = cutoffs          or base/"artifacts/calibration/thresholds.json"

    missing=[(k,p) for k,p in [("baseline_features",bf),("baseline_metrics",bm),("current_features",cf),("model",mp)] if not Path(p).exists()]
    if missing:
        for k,p in missing: print(f"[Missing] {k}: {p}")
        return {"status":"error","missing":[str(p) for _,p in missing]}

    bm_data = json.loads(Path(bm).read_text())
    _run(Path(bf), bm_data, Path(cf), Path(mp), Path(od), psi_bins=psi_bins, cutoffs_path=Path(ct) if Path(ct).exists() else None)
    return {"status":"success","outdir":str(od)}

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Model Monitoring - Production Grade")
    ap.add_argument("--baseline-features"); ap.add_argument("--baseline-metrics")
    ap.add_argument("--current-features");  ap.add_argument("--model")
    ap.add_argument("--outdir"); ap.add_argument("--cutoffs")
    ap.add_argument("--psi-bins", type=int, default=10)
    a = ap.parse_args()
    kw={}
    if a.baseline_features: kw["baseline_features"]=Path(a.baseline_features)
    if a.baseline_metrics:  kw["baseline_metrics"]=Path(a.baseline_metrics)
    if a.current_features:  kw["current_features"]=Path(a.current_features)
    if a.model:             kw["model"]=Path(a.model)
    if a.outdir:            kw["outdir"]=Path(a.outdir)
    if a.cutoffs:           kw["cutoffs"]=Path(a.cutoffs)
    kw["psi_bins"]=a.psi_bins
    res = run_monitoring(**kw)
    if res.get("status")!="success": raise SystemExit(1)

if __name__ == "__main__":
    main()

# python run_monitoring.py --baseline-features data/processed/feature_ews_train.parquet --current-features data/processed/feature_ews_test.parquet