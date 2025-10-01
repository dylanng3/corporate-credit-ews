from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, shap, joblib, matplotlib.pyplot as plt

def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".parquet", ".csv"):
        return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
    if p.with_suffix(".parquet").exists(): return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():     return pd.read_csv(p.with_suffix(".csv"))
    raise FileNotFoundError(p)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def compute_shap(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.values)
    if isinstance(sv, list) and len(sv)==2: sv = sv[1]
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False).rename("mean_abs_shap")
    return sv, imp

def top_local(X: pd.DataFrame, sv: np.ndarray, k=5) -> pd.DataFrame:
    idx = np.argsort(-np.abs(sv), axis=1)[:, :k]
    rows = []
    for i in range(X.shape[0]):
        rec = {"row_id": i}
        for j, cidx in enumerate(idx[i], 1):
            col = X.columns[cidx]
            rec[f"feat{j}"] = col
            rec[f"shap{j}"] = float(sv[i, cidx]) # type: ignore
            rec[f"value{j}"] = float(X.iat[i, cidx]) if pd.api.types.is_numeric_dtype(X.dtypes[col]) else X.iat[i, cidx] # type: ignore
        rows.append(rec)
    return pd.DataFrame(rows)

def parse_args():
    ap = argparse.ArgumentParser(description="Explain LightGBM EWS bằng SHAP (compact).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", default="artifacts/shap")
    ap.add_argument("--max-display", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_dir(outdir)

    m = joblib.load(args.model)
    if "base" not in m or "features" not in m: raise ValueError("model_lgbm.pkl phải có {'base','features'}.")
    model, feats = m["base"], m["features"]

    df = read_table(Path(args.features))
    id_cols = [c for c in ("customer_id","sector_code","size_bucket") if c in df.columns]
    X = df[feats].copy()
    if args.sample and args.sample < len(X):
        X = X.sample(args.sample, random_state=42).reset_index(drop=True)
        ids = df.loc[X.index, id_cols].reset_index(drop=True) if id_cols else None
    else:
        ids = df[id_cols].reset_index(drop=True) if id_cols else None

    sv, imp = compute_shap(model, X)
    imp.to_csv(outdir/"feature_importance.csv", header=True)

    shap.summary_plot(sv, X.values, feature_names=X.columns, max_display=args.max_display, show=False)
    plt.tight_layout(); plt.savefig(outdir/"shap_summary.png", bbox_inches="tight"); plt.close()

    for f in ("icr_ttm","ccc","%util_mean_60d"):
        if f in X.columns:
            shap.dependence_plot(f, sv, X.values, feature_names=X.columns, show=False)
            plt.tight_layout(); plt.savefig(outdir/f"shap_dependence_{f.replace('%','pct').replace('/','_')}.png", bbox_inches="tight"); plt.close()

    tl = top_local(X, sv, k=5) # type: ignore
    if ids is not None: tl = pd.concat([ids, tl], axis=1)
    tl.to_csv(outdir/"top_drivers_per_customer.csv", index=False)

    summary = {
        "n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "top5_features": imp.head(5).index.tolist(),
        "outputs": {
            "feature_importance": str((outdir/"feature_importance.csv").resolve()),
            "top_drivers_per_customer": str((outdir/"top_drivers_per_customer.csv").resolve()),
            "shap_summary": str((outdir/"shap_summary.png").resolve())
        }
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

# python src/explain.py --model artifacts/models/model_lgbm.pkl --features data/processed/feature_ews.parquet --outdir artifacts/shap
