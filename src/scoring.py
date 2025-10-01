# src/scoring.py  — compact, LightGBM-only, isotonic calibrator support
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

# ---------- IO ----------
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" or path.with_suffix(".parquet").exists():
        return pd.read_parquet(path if path.suffix.lower()==".parquet" else path.with_suffix(".parquet"))
    if path.suffix.lower() == ".csv" or path.with_suffix(".csv").exists():
        return pd.read_csv(path if path.suffix.lower()==".csv" else path.with_suffix(".csv"))
    raise FileNotFoundError(f"Không tìm thấy {path} (.parquet|.csv)")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ---------- Model ----------
def load_model(pkl_path: Path) -> Dict[str, Any]:
    obj = joblib.load(pkl_path)
    # Kỳ vọng từ train_baseline.py: {"base": lgbm, "features": feats, ...}
    if "base" not in obj or "features" not in obj:
        raise ValueError("Model pickle phải có {'base','features'} (từ train_baseline.py).")
    return obj

def predict_raw_prob(df: pd.DataFrame, model_obj: Dict[str, Any]) -> np.ndarray:
    feats = model_obj["features"]
    X = df[feats].values
    # LightGBM predict_proba → cột 1 là xác suất class=1
    return model_obj["base"].predict_proba(X)[:, 1]

# ---------- Calibrator & thresholds ----------
def apply_isotonic(score_raw: np.ndarray, calibrator_pkl: Path | None) -> np.ndarray:
    """Nếu có calibrator.pkl (IsotonicRegression), dùng .predict; nếu không, coi score_raw là prob đã calibrated."""
    if calibrator_pkl is None:
        return score_raw
    calib = joblib.load(calibrator_pkl)  # IsotonicRegression
    return calib.predict(score_raw)      # trả về (n,)

def load_thresholds(thr_path: Path | None, probs_all: np.ndarray,
                    red_pct=0.15, amber_pct=0.10) -> Dict[str, float]:
    if thr_path and thr_path.exists():
        meta = json.loads(thr_path.read_text(encoding="utf-8"))
        if "capacity" in meta and "thresholds" in meta["capacity"]:
            t = meta["capacity"]["thresholds"]
            return {"red": float(t["red"]), "amber": float(t["amber"])}
    # fallback: capacity theo phần trăm
    return {
        "red":   float(np.quantile(probs_all, 1 - red_pct)),
        "amber": float(np.quantile(probs_all, 1 - red_pct - amber_pct)),
    }

def map_tier(p: float, thr: Dict[str, float]) -> str:
    return "Red" if p >= thr["red"] else ("Amber" if p >= thr["amber"] else "Green")

def action_for(tier: str) -> str:
    return {
        "Green": "Theo dõi định kỳ; cập nhật BCTC đúng hạn.",
        "Amber": "Soát xét RM ≤10 ngày; yêu cầu management accounts; kiểm tra công nợ; hạn chế tăng hạn mức.",
        "Red":   "Họp KH ≤5 ngày; lập kế hoạch dòng tiền 13 tuần; xem xét covenant tightening/TSĐB; đưa watchlist."
    }[tier]

# ---------- Main ----------
def run_scoring(
    features_path: Path,
    model_path: Path,
    asof: str,
    outdir: Path,
    calibrator_path: Path | None = None,
    thresholds_path: Path | None = None,
    id_cols_hint: List[str] | None = None,
) -> Path:
    ensure_dir(outdir)

    df = read_table(features_path)
    model_obj = load_model(model_path)

    id_cols = [c for c in (id_cols_hint or ["customer_id","sector_code","size_bucket"]) if c in df.columns]

    # 1) raw prob (uncalibrated)
    prob_raw = predict_raw_prob(df, model_obj)

    # 2) calibrated prob (isotonic nếu có)
    prob_cal = apply_isotonic(prob_raw, calibrator_path)

    # 3) thresholds
    thr = load_thresholds(thresholds_path, prob_cal)

    # 4) build output
    out = pd.DataFrame({"prob_default_12m_calibrated": prob_cal})
    out["score_ews"] = np.round(100 * (1 - out["prob_default_12m_calibrated"]), 2)
    out["tier"] = [map_tier(p, thr) for p in out["prob_default_12m_calibrated"]]
    out["action"] = out["tier"].map(action_for)
    if id_cols:  # chèn ID lên đầu
        out = pd.concat([df[id_cols].reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    # 5) save
    out_path = outdir / f"ews_scored_{asof}.csv"
    out.to_csv(out_path, index=False)
    (outdir / "thresholds_used.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
    return out_path

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Batch scoring + Action map cho EWS (KHDN) — compact")
    p.add_argument("--features", required=True, help="Path features (.parquet/.csv)")
    p.add_argument("--model", required=True, help="Path model_lgbm.pkl (train_baseline.py)")
    p.add_argument("--asof", required=True, help="YYYY-MM-DD để đặt tên file output")
    p.add_argument("--outdir", default="artifacts/scoring", help="Thư mục xuất file")
    p.add_argument("--calibrator", default="artifacts/calibration/calibrator.pkl", help="(tuỳ chọn) artifacts/calibration/calibrator.pkl")
    p.add_argument("--thresholds", default="artifacts/calibration/thresholds.json", help="(tuỳ chọn) artifacts/calibration/thresholds.json")
    return p.parse_args()

def main():
    args = parse_args()
    out = run_scoring(
        features_path=Path(args.features),
        model_path=Path(args.model),
        asof=args.asof,
        outdir=Path(args.outdir),
        calibrator_path=Path(args.calibrator) if args.calibrator else None,
        thresholds_path=Path(args.thresholds) if args.thresholds else None,
    )
    print(f"Done. Wrote: {out}")

if __name__ == "__main__":
    main()

# python src/scoring.py --features data/processed/feature_ews.parquet --model artifacts/models/model_lgbm.pkl --asof 2025-06-30