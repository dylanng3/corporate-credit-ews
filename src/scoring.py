# src/scoring.py
# Batch scoring EWS (KHDN) tại một mốc asof_date
# - Nhận features + model (đã train) + (tuỳ chọn) calibrator + thresholds
# - Trả ra: customer_id, prob_default_12m_calibrated, score_ews(0-100), tier, action
# - Xuất: reports/ews_scored_{asof}.csv
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

# Import calibrator class to fix pickle loading
try:
    from calibrate import IsotonicCalibrator
except ImportError:
    # If import fails, define a dummy class
    class IsotonicCalibrator:
        pass

# ---------------------------
# IO helpers
# ---------------------------
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    p_parq, p_csv = path.with_suffix(".parquet"), path.with_suffix(".csv")
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Không tìm thấy {path} (.parquet hoặc .csv)")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Model & scoring
# ---------------------------
def load_model(pkl_path: Path) -> Dict[str, Any]:
    obj = joblib.load(pkl_path)
    # Kỳ vọng cấu trúc từ train_baseline.py:
    # - LightGBM: {"base": lgbm, "calibrated": CalibratedClassifierCV, "features": feats}
    # - Logistic : {"scaler": scaler, "base": logit, "calibrated": CalibratedClassifierCV, "features": feats}
    return obj

def compute_raw(df: pd.DataFrame, model_obj: Dict[str, Any], raw_kind: str = "prob") -> np.ndarray:
    feats = model_obj.get("features")
    if feats is None:
        raise ValueError("Model pickle không có 'features'. Hãy train bằng train_baseline.py (bản lưu features).")
    X = df[feats].values

    # Logistic?
    if "scaler" in model_obj:
        Xs = model_obj["scaler"].transform(X)
        base = model_obj["base"]
        if raw_kind == "prob":
            return base.predict_proba(Xs)[:, 1]
        elif raw_kind == "margin":
            return base.decision_function(Xs)
        else:
            raise ValueError("raw_kind phải là 'prob' hoặc 'margin'.")

    # LightGBM?
    if "base" in model_obj:
        base = model_obj["base"]
        if raw_kind == "prob":
            return base.predict_proba(X)[:, 1]
        elif raw_kind == "margin":
            return base.predict(X, raw_score=True)
        else:
            raise ValueError("raw_kind phải là 'prob' hoặc 'margin'.")

    raise ValueError("Không nhận diện được cấu trúc model pickle.")

# ---------------------------
# Calibrator & thresholds
# ---------------------------
def apply_calibrator(score_raw: np.ndarray, calibrator_pkl: Path | None) -> np.ndarray:
    """Nếu có calibrator.pkl (từ src/calibrate.py), áp dụng; ngược lại coi score_raw là xác suất đã calibrated."""
    if calibrator_pkl is None:
        return score_raw
    calib = joblib.load(calibrator_pkl)
    # Calibrator trong calibrate.py có API predict_proba(score_raw)
    try:
        return calib.predict_proba(score_raw)
    except Exception:
        # Trường hợp calibrator là sklearn CalibratedClassifierCV (hiếm khi dùng ở bước này)
        # thì score_raw có thể phải reshape(-1,1)
        return calib.predict_proba(score_raw.reshape(-1, 1))[:, 1]

def load_thresholds(thr_path: Path | None, probs_all: np.ndarray, red_pct=0.15, amber_pct=0.10) -> Dict[str, float]:
    """Ưu tiên thresholds.json; nếu không có, đặt theo capacity (top x%, y%)."""
    if thr_path is not None and thr_path.exists():
        meta = json.loads(thr_path.read_text(encoding="utf-8"))
        # Ưu tiên 'capacity' nếu có
        if "capacity" in meta and "thresholds" in meta["capacity"]:
            return {
                "red": float(meta["capacity"]["thresholds"]["red"]),
                "amber": float(meta["capacity"]["thresholds"]["amber"]),
            }
        # Nếu chỉ có 'youden' → map 2-tiers (RED/GREEN). Ta suy ra amber theo median (tuỳ chọn).
        if "youden" in meta and "youden" in meta["youden"]:
            ythr = float(meta["youden"]["youden"])
            # tạo amber ở giữa để vẫn có 3 tier (tuỳ chọn: median)
            amber = float(np.median(probs_all))
            return {"red": ythr, "amber": amber}
    # Fall back: capacity theo phần trăm
    return {
        "red":   float(np.quantile(probs_all, 1 - red_pct)),
        "amber": float(np.quantile(probs_all, 1 - red_pct - amber_pct)),
    }

def map_tier(prob: float, thr: Dict[str, float]) -> str:
    if prob >= thr["red"]:
        return "Red"
    if prob >= thr["amber"]:
        return "Amber"
    return "Green"

def action_for(tier: str) -> str:
    return {
        "Green": "Theo dõi định kỳ; cập nhật BCTC đúng hạn.",
        "Amber": "Soát xét RM ≤10 ngày; yêu cầu management accounts; kiểm tra công nợ; hạn chế tăng hạn mức.",
        "Red":   "Họp KH ≤5 ngày; lập kế hoạch dòng tiền 13 tuần; xem xét covenant tightening/TSĐB; đưa watchlist."
    }[tier]

# ---------------------------
# Main pipeline
# ---------------------------
def run_scoring(
    features_path: Path,
    model_path: Path,
    asof: str,
    outdir: Path,
    calibrator_path: Path | None = None,
    thresholds_path: Path | None = None,
    raw_kind: str = "prob",
    id_cols_hint: List[str] | None = None,
) -> Path:
    ensure_dir(outdir)

    # Load
    df = read_table(features_path)
    model_obj = load_model(model_path)

    # Chọn id columns
    if id_cols_hint is None:
        id_cols_hint = ["customer_id", "sector_code", "size_bucket"]
    id_cols = [c for c in id_cols_hint if c in df.columns]

    # Raw scores (uncalibrated prob / margin)
    score_raw = compute_raw(df, model_obj, raw_kind=raw_kind)

    # Calibrated probability
    prob_cal = apply_calibrator(score_raw, calibrator_path)

    # Thresholds
    thr = load_thresholds(thresholds_path, probs_all=prob_cal)

    # Build output
    out = pd.DataFrame()
    if id_cols:
        out[id_cols] = df[id_cols]
    else:
        out["row_id"] = np.arange(len(df))
    out["prob_default_12m_calibrated"] = prob_cal
    out["score_ews"] = np.round(100 * (1 - out["prob_default_12m_calibrated"]), 2)
    out["tier"] = [map_tier(p, thr) for p in out["prob_default_12m_calibrated"]]
    out["action"] = out["tier"].map(action_for)

    # Save
    out_path = outdir / f"ews_scored_{asof}.csv"
    out.to_csv(out_path, index=False)

    # Lưu thresholds sử dụng thực tế để trace
    (outdir / "thresholds_used.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")

    return out_path

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch scoring + Action map cho EWS (KHDN).")
    p.add_argument("--features", required=True, type=str, help="Đường dẫn features (.parquet/.csv).")
    p.add_argument("--model", required=True, type=str, help="Đường dẫn model pickle (từ train_baseline.py).")
    p.add_argument("--asof", required=True, type=str, help="Asof date để chèn vào tên file output (YYYY-MM-DD).")
    p.add_argument("--outdir", default=".", type=str, help="Thư mục gốc để xuất reports/")
    p.add_argument("--calibrator", type=str, default=None, help="(Tuỳ chọn) calibrator.pkl (từ calibrate.py).")
    p.add_argument("--thresholds", type=str, default=None, help="(Tuỳ chọn) thresholds.json (từ calibrate.py).")
    p.add_argument("--raw-kind", default="prob", choices=["prob","margin"], help="Kiểu score_raw đưa vào calibrator (mặc định prob).")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = run_scoring(
        features_path=Path(args.features),
        model_path=Path(args.model),
        asof=args.asof,
        outdir=Path(args.outdir),
        calibrator_path=Path(args.calibrator) if args.calibrator else None,
        thresholds_path=Path(args.thresholds) if args.thresholds else None,
        raw_kind=args.raw_kind,
    )
    print(f"Done. Wrote: {out_path}")

if __name__ == "__main__":
    main()

# python src/scoring.py --features data/processed/feature_ews.parquet --model artifacts/models/model_lgbm.pkl --calibrator artifacts/calibration_iso/calibrator.pkl --thresholds artifacts/calibration_iso/thresholds.json --raw-kind prob --asof 2025-06-30 --outdir artifacts/scoring
