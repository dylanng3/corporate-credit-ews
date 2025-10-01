# src/make_scores_raw.py
# Tạo scores_raw.csv từ features + model đã huấn luyện (chưa hiệu chỉnh)
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib

def read_table(path_no_ext: Path) -> pd.DataFrame:
    """Đọc .parquet nếu có, fallback .csv; nếu đã có đuôi thì đọc trực tiếp."""
    if path_no_ext.suffix:
        if path_no_ext.suffix.lower() == ".parquet":
            return pd.read_parquet(path_no_ext)
        elif path_no_ext.suffix.lower() == ".csv":
            return pd.read_csv(path_no_ext)
    p_parq = path_no_ext.with_suffix(".parquet")
    p_csv  = path_no_ext.with_suffix(".csv")
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không tìm thấy {p_parq} hoặc {p_csv}")

def select_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["customer_id", "sector_code", "size_bucket"]
    return [c for c in candidates if c in df.columns]

def load_model(pkl_path: Path) -> Dict[str, Any]:
    obj = joblib.load(pkl_path)
    # Hỗ trợ 2 dạng pkl từ train_baseline.py:
    # - LightGBM: {"base": lgbm, "calibrated": calib, "features": feats}
    # - Logistic: {"scaler": scaler, "base": logit, "calibrated": calib, "features": feats}
    return obj

def compute_raw_scores(df: pd.DataFrame, model_obj: Dict[str, Any], raw_kind: str = "prob") -> np.ndarray:
    feats = model_obj.get("features", None)
    if feats is None:
        raise ValueError("Model pickle không chứa danh sách features. Hãy train lại bằng train_baseline.py (phiên bản đã lưu 'features').")
    X = df[feats].values

    # Logistic?
    if "scaler" in model_obj:
        scaler = model_obj["scaler"]
        base = model_obj["base"]
        Xs = scaler.transform(X)
        if raw_kind == "prob":
            return base.predict_proba(Xs)[:, 1]
        elif raw_kind == "margin":
            # decision_function của Logistic là logit margin
            return base.decision_function(Xs)
        else:
            raise ValueError("raw_kind phải là 'prob' hoặc 'margin'.")

    # LightGBM?
    if "base" in model_obj:
        base = model_obj["base"]
        if raw_kind == "prob":
            return base.predict_proba(X)[:, 1]
        elif raw_kind == "margin":
            # raw_score=True trả về margin
            return base.predict(X, raw_score=True)
        else:
            raise ValueError("raw_kind phải là 'prob' hoặc 'margin'.")

    raise ValueError("Không nhận diện được cấu trúc model pickle.")

def main():
    ap = argparse.ArgumentParser(description="Tạo scores_raw.csv để calibrate EWS.")
    ap.add_argument("--features", required=True, type=str, help="Đường dẫn features (.parquet/.csv).")
    ap.add_argument("--model", required=True, type=str, help="Đường dẫn model pickle (từ train_baseline.py).")
    ap.add_argument("--y-col", default="event_h12m", type=str, help="Tên cột nhãn 0/1.")
    ap.add_argument("--raw-kind", default="prob", choices=["prob","margin"], help="Kiểu score thô: prob (mặc định) hoặc margin.")
    ap.add_argument("--out", default="data/processed/scores_raw.csv", type=str, help="Đường dẫn xuất CSV.")
    args = ap.parse_args()

    df = read_table(Path(args.features))
    model_obj = load_model(Path(args.model))

    if args.y_col not in df.columns:
        raise ValueError(f"Không thấy cột nhãn '{args.y_col}' trong features.")

    # Lấy id cols (nếu có) để giữ lại
    id_cols = select_id_columns(df)

    # Tính score thô
    score_raw = compute_raw_scores(df, model_obj, raw_kind=args.raw_kind)

    out_df = df[id_cols + [args.y_col]].copy()
    out_df["score_raw"] = score_raw

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(json.dumps({
        "features_in": str(Path(args.features).absolute()),
        "model_in": str(Path(args.model).absolute()),
        "rows": int(len(out_df)),
        "raw_kind": args.raw_kind,
        "output": str(out_path.absolute())
    }, indent=2))

if __name__ == "__main__":
    main()

# python src/make_scores_raw.py --features data/processed/feature_ews.parquet --model artifacts/model_lgbm.pkl --y-col event_h12m --raw-kind prob --out data/processed/scores_raw.csv