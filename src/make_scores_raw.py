from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import joblib, numpy as np, pandas as pd

ID_CANDIDATES = ["customer_id", "sector_code", "size_bucket"]

def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".parquet", ".csv"):
        return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
    if p.with_suffix(".parquet").exists(): return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():     return pd.read_csv(p.with_suffix(".csv"))
    raise FileNotFoundError(f"Không tìm thấy {p} (.parquet/.csv)")

def select_id_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in ID_CANDIDATES if c in df.columns]

def load_lgbm_model(pkl: Path):
    obj = joblib.load(pkl)
    if "base" not in obj or "features" not in obj:
        raise ValueError("Model pickle phải có keys {'base','features'} (train_baseline.py).")
    return obj["base"], obj["features"]

def main():
    ap = argparse.ArgumentParser(description="Xuất scores_raw (probability thô) từ features + LightGBM model.")
    ap.add_argument("--features", required=True, help="Path .parquet/.csv features")
    ap.add_argument("--model",    required=True, help="Path model_lgbm.pkl (từ train_baseline.py)")
    ap.add_argument("--y-col",    default="event_h12m", help="Tên cột nhãn 0/1 (để giữ lại trong output)")
    ap.add_argument("--out",      default="data/processed/scores_raw.csv", help="CSV đầu ra")
    args = ap.parse_args()

    df  = read_table(Path(args.features))
    base, feats = load_lgbm_model(Path(args.model))
    if args.y_col not in df.columns:
        raise ValueError(f"Không thấy cột nhãn '{args.y_col}' trong features.")

    X = df[feats].values
    prob = base.predict_proba(X)[:, 1]

    out_df = df[select_id_columns(df) + [args.y_col]].copy()
    out_df["score_raw"] = prob

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(json.dumps({
        "features_in": str(Path(args.features).resolve()),
        "model_in":    str(Path(args.model).resolve()),
        "rows":        int(len(out_df)),
        "output":      str(out_path.resolve())
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

# python src/make_scores_raw.py --features data/processed/feature_ews.parquet --model artifacts/models/model_lgbm.pkl