
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtesting EWS monthly cohorts with rolling metrics, threshold sweep, calibration,
segment view, and append-safe outputs (de-duplicated by keys).

Usage (example):
python src/backtest_monthly.py --data data/processed/backtest_cohorts.parquet --as-of-col as_of_date --pd-col pd_12m --y-col y_event_12m --start 2024-01 --end 2025-06 --outdir artifacts/backtest/

Input schema (minimal per row = 1 customer-month):
    customer_id : id
    as_of_date  : date-like (any day in month is OK)
    pd_12m      : calibrated PD in [0,1]
    y_event_6m  : binary target within forward window (e.g., default within 6m)
    [optional] segment columns (e.g., sector/product/channel/region)
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


def _to_month_period(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M")


def ks_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute KS = max|CDF1 - CDF0| with robust ties handling."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(y_score)
    y_true = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    cdf_pos = np.cumsum(y_true) / n_pos
    cdf_neg = np.cumsum(1 - y_true) / n_neg
    ks = np.max(np.abs(cdf_pos - cdf_neg))
    return float(ks)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return decile calibration table: decile, count, pd_avg, odr, abs_err (bp)."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    try:
        bins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        y_prob = y_prob + np.random.default_rng(42).normal(0, 1e-12, size=len(y_prob))
        bins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")

    df = pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
    out = df.groupby("bin").agg(
        count=("y", "size"),
        odr=("y", "mean"),
        pd_avg=("p", "mean"),
    ).reset_index().sort_values("bin")
    out["decile"] = np.arange(1, len(out) + 1)
    out["abs_err_bp"] = (out["pd_avg"] - out["odr"]).abs() * 1e4
    cols = ["decile", "count", "pd_avg", "odr", "abs_err_bp"]
    return out[cols]


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.1) -> float:
    """Compute lift at top frac (e.g., 0.1 for top 10%)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    n = max(1, int(len(p) * frac))
    idx = np.argsort(-p)[:n]
    base = y.mean() + 1e-12
    return float(y[idx].mean() / base)


def sweep_thresholds(y_true: np.ndarray, y_score: np.ndarray, cuts: np.ndarray) -> pd.DataFrame:
    rows = []
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    events = y.sum()
    for c in cuts:
        mask = p >= c
        alerts = int(mask.sum())
        rate = float(mask.mean())
        prec = float(y[mask].mean()) if alerts > 0 else np.nan
        rec = float(y[mask].sum() / events) if events > 0 else np.nan
        rows.append({"cut": float(c), "alert_rate": rate, "precision": prec, "recall": rec, "alerts": alerts})
    return pd.DataFrame(rows)


def metrics_for_cohort(
    df: pd.DataFrame,
    pd_col: str,
    y_col: str,
    cut_amber: float,
    cut_red: float,
    min_records: int = 200,
) -> Optional[dict]:
    df = df.dropna(subset=[pd_col, y_col])
    if len(df) < min_records:
        return None
    y = df[y_col].astype(int).to_numpy()
    p = df[pd_col].clip(1e-6, 1 - 1e-6).astype(float).to_numpy()

    # Metrics
    auc = roc_auc_score(y, p) if y.sum() > 0 else np.nan
    ks = ks_score(y, p)
    brier = float(np.mean((y - p) ** 2))
    pr_auc = average_precision_score(y, p) if y.sum() > 0 else np.nan
    lift10 = lift_at_k(y, p, 0.10)
    lift20 = lift_at_k(y, p, 0.20)

    def _eval(cut: float):
        m = p >= cut
        alerts = int(m.sum())
        rate = float(m.mean())
        prec = float(y[m].mean()) if alerts > 0 else np.nan
        rec = float(y[m].sum() / y.sum()) if y.sum() > 0 else np.nan
        return rate, prec, rec, alerts

    amber_rate, amber_prec, amber_rec, amber_cnt = _eval(cut_amber)
    red_rate, red_prec, red_rec, red_cnt = _eval(cut_red)

    calib = calibration_table(y, p, n_bins=10)
    return {
        "auc": float(auc),
        "ks": float(ks),
        "brier": float(brier),
        "pr_auc": float(pr_auc),
        "lift_10pct": float(lift10),
        "lift_20pct": float(lift20),
        "amber_alert_rate": amber_rate,
        "amber_precision": amber_prec,
        "amber_recall": amber_rec,
        "amber_alerts": amber_cnt,
        "red_alert_rate": red_rate,
        "red_precision": red_prec,
        "red_recall": red_rec,
        "red_alerts": red_cnt,
        "calibration": calib,
    }


def segment_metrics(
    df: pd.DataFrame,
    seg_col: str,
    pd_col: str,
    y_col: str,
    cut_amber: float,
    cut_red: float,
    min_records: int = 200,
) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(seg_col):
        m = metrics_for_cohort(g, pd_col, y_col, cut_amber, cut_red, min_records=min_records)
        if m is None:
            continue
        rows.append({"segment": seg, **{k: v for k, v in m.items() if k != "calibration"}})
    return pd.DataFrame(rows)


def append_dedup_csv(path: Path, df_new: pd.DataFrame, subset: list) -> None:
    if path.exists():
        parse_cols = [c for c in subset if ("date" in c) or ("month" in c)]
        old = pd.read_csv(path, parse_dates=parse_cols) if parse_cols else pd.read_csv(path)
        df_all = pd.concat([old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=subset, keep="last")
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Monthly backtesting for EWS")
    ap.add_argument("--data", required=True, help="Path to CSV/Parquet data")
    ap.add_argument("--as-of-col", dest="as_of_col", default="as_of_date")
    ap.add_argument("--pd-col", dest="pd_col", default="pd_12m")
    ap.add_argument("--y-col", dest="y_col", default="y_event_6m")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--amber", type=float, default=0.02)
    ap.add_argument("--red", type=float, default=0.05)
    ap.add_argument("--outdir", default="artifacts/backtest/")
    ap.add_argument("--segment-col", default=None, help="Optional column for segment view (e.g., sector)")
    ap.add_argument("--min-records", type=int, default=200)
    ap.add_argument("--sweep-min", type=float, default=0.005)
    ap.add_argument("--sweep-max", type=float, default=0.10)
    ap.add_argument("--sweep-step", type=float, default=0.005)
    ap.add_argument("--drop-last-n", type=int, default=0, help="Drop last N months to ensure label maturity")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    data_path = Path(args.data)
    if data_path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Normalize month and drop last N months if labels may be immature
    df["as_of_month"] = _to_month_period(df[args.as_of_col])
    all_months = pd.period_range(args.start, args.end, freq="M")
    if args.drop_last_n > 0 and len(all_months) > args.drop_last_n:
        all_months = all_months[:-args.drop_last_n]

    metrics_rows = []
    calib_rows = []
    sweep_rows = []
    seg_rows = []

    cuts = np.arange(args.sweep_min, args.sweep_max + 1e-12, args.sweep_step)

    for m in all_months:
        cohort = df[df["as_of_month"] == m].copy()
        # De-dup: one row per customer per month
        if "customer_id" in cohort.columns:
            cohort = cohort.drop_duplicates(subset=["as_of_month", "customer_id"])
        if len(cohort) < args.min_records:
            continue

        # Compute cohort metrics
        res = metrics_for_cohort(cohort, args.pd_col, args.y_col, args.amber, args.red, min_records=args.min_records)
        if res is None:
            continue

        # metrics
        metrics_rows.append({
            "as_of_month": m.to_timestamp("M"),
            "pd_col": args.pd_col,
            "y_col": args.y_col,
            "amber": args.amber,
            "red": args.red,
            **{k: v for k, v in res.items() if k != "calibration"}
        })

        # calibration deciles
        cal = res["calibration"].copy()
        cal.insert(0, "as_of_month", m.to_timestamp("M"))
        calib_rows.append(cal)

        # threshold sweep
        y = cohort[args.y_col].astype(int).to_numpy()
        p = cohort[args.pd_col].clip(1e-6, 1 - 1e-6).astype(float).to_numpy()
        sw = sweep_thresholds(y, p, cuts)
        sw.insert(0, "as_of_month", m.to_timestamp("M"))
        sweep_rows.append(sw)

        # segment metrics (optional)
        if args.segment_col and args.segment_col in cohort.columns:
            seg_df = segment_metrics(cohort, args.segment_col, args.pd_col, args.y_col, args.amber, args.red, args.min_records)
            if len(seg_df) > 0:
                seg_df.insert(0, "as_of_month", m.to_timestamp("M"))
                seg_df.insert(1, "segment_col", args.segment_col)
                seg_rows.append(seg_df)

    # Build DataFrames
    if len(metrics_rows) == 0:
        print("No cohorts met minimum record threshold. Nothing to write.")
        return

    df_metrics = pd.DataFrame(metrics_rows).sort_values("as_of_month")
    # rolling 12m
    for col in ["auc", "ks", "brier", "pr_auc"]:
        df_metrics[f"{col}_roll12"] = df_metrics[col].rolling(12, min_periods=3).mean()

    df_calib = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame(columns=["as_of_month"])
    df_sweep = pd.concat(sweep_rows, ignore_index=True) if sweep_rows else pd.DataFrame(columns=["as_of_month"])
    df_seg = pd.concat(seg_rows, ignore_index=True) if seg_rows else pd.DataFrame(columns=["as_of_month"])

    # Append-safe writes
    def _save(name, df, subset):
        path = outdir / name
        if df.empty:
            return
        if path.exists():
            # Only parse columns that exist in the file
            old = pd.read_csv(path)
            parse_cols = [c for c in subset if (("date" in c) or ("month" in c)) and c in old.columns]
            if parse_cols:
                old[parse_cols] = old[parse_cols].apply(pd.to_datetime, errors='coerce')
            df_all = pd.concat([old, df], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=subset, keep="last")
        else:
            df_all = df
        df_all.to_csv(path, index=False)

    _save("monthly_metrics.csv", df_metrics, ["as_of_month", "pd_col", "y_col", "amber", "red"])
    _save("monthly_calibration.csv", df_calib, ["as_of_month", "decile"])
    _save("threshold_sweep.csv", df_sweep, ["as_of_month", "cut"])
    _save("segment_metrics.csv", df_seg, ["as_of_month", "segment_col", "segment"])

    print("✅ Backtest completed.")
    print(f"  - {outdir/'monthly_metrics.csv'}")
    print(f"  - {outdir/'monthly_calibration.csv'}")
    if not df_sweep.empty:
        print(f"  - {outdir/'threshold_sweep.csv'}")
    if not df_seg.empty:
        print(f"  - {outdir/'segment_metrics.csv'}")


if __name__ == "__main__":
    main()
