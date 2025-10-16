#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

EPS = 1e-12

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
    return {
        "auc":  (cur["auc"]   < 0.70) or (base.get("auc",0)-cur["auc"]   > 0.05),
        "brier":(cur["brier"] > 0.15) or (cur["brier"]-base.get("brier",0)>0.05),
        "ks":   (ks_base>0) and ((ks_base-cur["ks"])/max(ks_base,EPS) > 0.20),
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
        v=_psi_one(base_df[c], cur_df[c], bins)
        status = "Insufficient" if np.isnan(v) else ("Stable" if v<0.10 else "Moderate" if v<0.25 else "Severe")
        rows.append({"feature": c, "psi": (None if np.isnan(v) else round(v,4)), "status": status})
    return pd.DataFrame(rows).sort_values(["status","psi"], ascending=[True,False], na_position="last")

# -------- Core --------
def _run(bf: Path, bm: Dict[str,float], cf: Path, model_p: Path, outdir: Path, psi_bins:int=10):
    outdir.mkdir(parents=True, exist_ok=True)
    base_df, cur_df = read_table(bf), read_table(cf)
    art = joblib.load(model_p); feats: List[str] = art["features"]

    # performance (nếu có nhãn)
    cur_metrics, alerts = {}, {}
    if "event_h12m" in cur_df.columns:
        X, y = cur_df[feats], cur_df["event_h12m"].astype(int).values
        y_prob = art["calibrated"].predict_proba(X)[:,1]
        cur_metrics = calc_metrics(y, y_prob) # type: ignore
        alerts = degrade_flags(bm, cur_metrics)

    # PSI
    psi = psi_table(base_df, cur_df, feats, bins=psi_bins)
    severe_cnt = int((psi["status"]=="Severe").sum())

    # summary + save
    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_metrics": bm, "current_metrics": cur_metrics,
        "alerts": alerts, "psi_severe_count": severe_cnt
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (outdir/f"monitoring_{ts}.json").write_text(json.dumps(report, indent=2))
    psi.to_csv(outdir/"psi_details.csv", index=False)

    # minimal console log
    print("=== Monitoring Summary ===")
    if cur_metrics: print("Current metrics:", {k: round(v,4) for k,v in cur_metrics.items()})
    print("Severe PSI features:", severe_cnt)
    if alerts and any(alerts.values()): print("Degradation alerts:", alerts)
    print("Report:", outdir/f"monitoring_{ts}.json")

# -------- Public API --------
def run_monitoring(
    baseline_features: Optional[Path]=None,
    baseline_metrics: Optional[Path]=None,
    current_features: Optional[Path]=None,
    model: Optional[Path]=None,
    outdir: Optional[Path]=None,
    psi_bins:int=10,
):
    base = Path(__file__).parent
    bf = baseline_features or base/"data/processed/feature_ews.parquet"
    bm = baseline_metrics  or base/"artifacts/models/baseline_metrics.json"
    cf = current_features  or bf
    mp = model            or base/"artifacts/models/model_lgbm.pkl"
    od = outdir           or base/"artifacts/monitoring"

    missing=[(k,p) for k,p in [("baseline_features",bf),("baseline_metrics",bm),("current_features",cf),("model",mp)] if not Path(p).exists()]
    if missing:
        for k,p in missing: print(f"[Missing] {k}: {p}")
        return {"status":"error","missing":[str(p) for _,p in missing]}

    bm_data = json.loads(Path(bm).read_text())
    _run(Path(bf), bm_data, Path(cf), Path(mp), Path(od), psi_bins=psi_bins)
    return {"status":"success","outdir":str(od)}

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Model Monitoring (minimal)")
    ap.add_argument("--baseline-features"); ap.add_argument("--baseline-metrics")
    ap.add_argument("--current-features");  ap.add_argument("--model")
    ap.add_argument("--outdir"); ap.add_argument("--psi-bins", type=int, default=10)
    a = ap.parse_args()
    kw={}
    if a.baseline_features: kw["baseline_features"]=Path(a.baseline_features)
    if a.baseline_metrics:  kw["baseline_metrics"]=Path(a.baseline_metrics)
    if a.current_features:  kw["current_features"]=Path(a.current_features)
    if a.model:             kw["model"]=Path(a.model)
    if a.outdir:            kw["outdir"]=Path(a.outdir)
    kw["psi_bins"]=a.psi_bins
    res = run_monitoring(**kw)
    if res.get("status")!="success": raise SystemExit(1)

if __name__ == "__main__":
    main()

# python run_monitoring.py --baseline-features data/processed/feature_ews_train.parquet --current-features data/processed/feature_ews_test.parquet