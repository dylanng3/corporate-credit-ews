# src/explain.py
# Explainability cho mô hình LightGBM trong EWS (KHDN)
# - Inputs:
#   --model    : đường dẫn tới model_lgbm.pkl (tạo bởi train_baseline.py)
#   --features : file features (.parquet hoặc .csv) dùng để suy luận
#   --outdir   : thư mục xuất kết quả (CSV/PNG/JSON)
# - Outputs:
#   feature_importance.csv            (global: mean |SHAP|)
#   top_drivers_per_customer.csv      (local top-5 driver/khách hàng)
#   shap_summary.png
#   shap_dependence_icr_ttm.png
#   shap_dependence_ccc.png
#   shap_dependence_pctutil_mean_60d.png  (nếu các cột có mặt)
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt


# --------------------------
# IO helpers
# --------------------------
def read_table(path: Path) -> pd.DataFrame:
    """Đọc Parquet nếu có, fallback CSV. Nếu path đã có đuôi thì đọc trực tiếp."""
    if path.suffix:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    p_parq = path.with_suffix(".parquet")
    p_csv  = path.with_suffix(".csv")
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không tìm thấy {p_parq} hoặc {p_csv}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --------------------------
# SHAP core
# --------------------------
def build_explainer(model):
    """SHAP explainer cho mô hình dạng cây (LightGBM)."""
    return shap.TreeExplainer(model)

def compute_shap(model, X_df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """Tính SHAP values và global importance (mean |SHAP|)."""
    explainer = build_explainer(model)
    shap_values = explainer.shap_values(X_df.values)
    # Binary classifier: shap trả list [class0, class1] -> lấy class1
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_arr = shap_values[1]
    else:
        shap_arr = shap_values
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    imp = pd.Series(mean_abs, index=X_df.columns).sort_values(ascending=False).rename("mean_abs_shap")
    return shap_arr, imp

def top_drivers_local(X_df: pd.DataFrame, shap_arr: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Với từng hàng, lấy top-k đặc trưng có |SHAP| lớn nhất."""
    abs_vals = np.abs(shap_arr)
    top_idx = np.argsort(-abs_vals, axis=1)[:, :k]
    records = []
    for i in range(X_df.shape[0]):
        rec = {"row_id": i}
        for j, col_idx in enumerate(top_idx[i], start=1):
            col = X_df.columns[col_idx]
            rec[f"feat{j}"]  = col
            rec[f"shap{j}"]  = float(shap_arr[i, col_idx])
            rec[f"value{j}"] = float(X_df.iloc[i, col_idx]) if np.issubdtype(X_df.dtypes[col], np.number) else X_df.iloc[i, col_idx]
        records.append(rec)
    return pd.DataFrame(records)


# --------------------------
# Main
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Explain LightGBM EWS model bằng SHAP.")
    ap.add_argument("--model", required=True, type=str, help="Đường dẫn model_lgbm.pkl")
    ap.add_argument("--features", required=True, type=str, help="Đường dẫn features (.parquet/.csv)")
    ap.add_argument("--outdir", default="explain_outputs", type=str, help="Thư mục xuất kết quả")
    ap.add_argument("--max-display", default=20, type=int, help="Số feature hiển thị tối đa trong SHAP summary plot")
    ap.add_argument("--sample", type=int, default=0, help="(Tuỳ chọn) Lấy mẫu N dòng cho SHAP để chạy nhanh")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load model (định dạng do train_baseline.py lưu)
    model_obj = joblib.load(args.model)
    if "base" not in model_obj:
        raise ValueError("Script này kỳ vọng LightGBM pickle có key 'base'.")
    model = model_obj["base"]
    feats = model_obj.get("features")
    if feats is None:
        raise ValueError("Model pickle thiếu danh sách 'features'. Hãy train bằng phiên bản train_baseline.py đã lưu 'features'.")

    # Load data và chọn đúng cột features
    df = read_table(Path(args.features))
    id_cols = [c for c in ["customer_id", "sector_code", "size_bucket"] if c in df.columns]
    X_df = df[feats].copy()

    # (Tuỳ chọn) Lấy mẫu để SHAP nhanh hơn
    if args.sample and args.sample < len(X_df):
        X_df = X_df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        if id_cols:
            df_ids = df.loc[X_df.index, id_cols].reset_index(drop=True)
        else:
            df_ids = None
    else:
        df_ids = df[id_cols].reset_index(drop=True) if id_cols else None

    # Tính SHAP
    shap_arr, imp = compute_shap(model, X_df)

    # Lưu global importance
    imp.to_csv(outdir / "feature_importance.csv", header=True, index=True)

    # SHAP summary plot
    shap.summary_plot(shap_arr, X_df.values, feature_names=X_df.columns, max_display=args.max_display, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    # Dependence plots cho 3 feature mẫu (nếu có)
    for f in ["icr_ttm", "ccc", "%util_mean_60d"]:
        if f in X_df.columns:
            shap.dependence_plot(f, shap_arr, X_df.values, feature_names=X_df.columns, show=False)
            plt.tight_layout()
            safe = f.replace("%", "pct").replace("/", "_")
            plt.savefig(outdir / f"shap_dependence_{safe}.png", bbox_inches="tight")
            plt.close()

    # Local top-5 drivers / khách hàng
    top_local = top_drivers_local(X_df, shap_arr, k=5)
    if df_ids is not None:
        top_local = pd.concat([df_ids, top_local], axis=1)
    top_local.to_csv(outdir / "top_drivers_per_customer.csv", index=False)

    # Summary JSON
    summary = {
        "n_rows": int(X_df.shape[0]),
        "n_features": int(X_df.shape[1]),
        "top5_features": imp.head(5).index.tolist(),
        "outputs": {
            "feature_importance": str((outdir / "feature_importance.csv").absolute()),
            "top_drivers_per_customer": str((outdir / "top_drivers_per_customer.csv").absolute()),
            "shap_summary": str((outdir / "shap_summary.png").absolute()),
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

# python src/explain.py --model artifacts/models/model_lgbm.pkl --features data/processed/feature_ews.parquet --outdir artifacts/shap