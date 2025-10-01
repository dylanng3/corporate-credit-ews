# src/features.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

ASOF_DEFAULT = "2025-06-30"
OW_DAYS = 180  # Observation Window

# ---------------------------
# IO helpers
# ---------------------------
def _read_any(path_no_ext: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    """Read Parquet if exists else CSV."""
    p_parquet = path_no_ext if path_no_ext.suffix==".parquet" else path_no_ext.with_suffix(".parquet")
    p_csv     = path_no_ext if path_no_ext.suffix==".csv"     else path_no_ext.with_suffix(".csv")
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv, parse_dates=parse_dates)
    raise FileNotFoundError(f"Not found: {p_parquet} or {p_csv}")

def load_raw(raw_dir: Path):
    fin = _read_any(raw_dir / "fin_quarterly", parse_dates=["fq_date"])
    cr  = _read_any(raw_dir / "credit_daily", parse_dates=["date"])
    cf  = _read_any(raw_dir / "cashflow_daily", parse_dates=["date"])
    cov = _read_any(raw_dir / "covenant", parse_dates=["date"])
    lab = _read_any(raw_dir / "labels", parse_dates=["asof_date"])
    return fin, cr, cf, cov, lab

# ---------------------------
# Basic utils
# ---------------------------
def winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

def sector_size_normalize(df: pd.DataFrame, cols: List[str], sector_col="sector_code", size_col="size_bucket"):
    """Robust z-score theo (sector, size) dùng median & IQR (q75-q25)."""
    df = df.copy()
    for c in cols:
        def _norm(g: pd.DataFrame):
            if len(g) < 10:
                med = df[c].median()
                iqr = df[c].quantile(0.75) - df[c].quantile(0.25)
            else:
                med = g[c].median()
                iqr = g[c].quantile(0.75) - g[c].quantile(0.25)
            iqr = iqr if iqr > 0 else (abs(med) if med != 0 else 1.0)
            return (g[c] - med) / iqr
        df[f"{c}__zs_sector_size"] = (
            df.groupby([sector_col, size_col], dropna=False, group_keys=False).apply(_norm)
        )
    return df

# ---------------------------
# Financial ratios (TTM + QoQ deltas)
# ---------------------------
def compute_financial_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    fin = fin.sort_values(["customer_id","fq_date"]).copy()
    # TTM aggregates
    fin["ebit_ttm"]     = fin.groupby("customer_id")["ebit"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    fin["ebitda_ttm"]   = fin.groupby("customer_id")["ebitda"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    fin["interest_ttm"] = fin.groupby("customer_id")["interest_expense"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    fin["revenue_ttm"]  = fin.groupby("customer_id")["revenue"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    fin["cogs_ttm"]     = fin.groupby("customer_id")["cogs"].transform(lambda s: s.rolling(4, min_periods=1).sum())

    last = fin.groupby("customer_id").tail(1).copy()

    # Liquidity / Leverage
    last["icr_ttm"]        = last["ebit_ttm"] / last["interest_ttm"].replace(0, np.nan)
    capex_approx           = 0.3 * last["ebitda_ttm"]  # proxy (không có principal, dùng interest)
    last["dscr_ttm_proxy"] = (last["ebitda_ttm"] - capex_approx) / last["interest_ttm"].replace(0, np.nan)
    last["debt_to_ebitda"] = last["total_debt"] / last["ebitda_ttm"].replace(0, np.nan)
    last["current_ratio"]  = last["current_assets"] / last["current_liab"].replace(0, np.nan)

    # Turnover & CCC (dùng balance cuối + flow TTM)
    last["dso"] = 365 * last["ar"]        / last["revenue_ttm"].replace(0, np.nan)
    last["dpo"] = 365 * last["ap"]        / last["cogs_ttm"].replace(0, np.nan)
    last["doh"] = 365 * last["inventory"] / last["cogs_ttm"].replace(0, np.nan)
    last["ccc"] = last["dso"] + last["doh"] - last["dpo"]

    # QoQ deltas
    prev = fin.groupby("customer_id").nth(-2).reset_index()
    for col in ["dso","ccc"]:
        prev_calc = prev.copy()
        prev_calc["dso"] = 365 * prev_calc["ar"] / prev_calc["revenue_ttm"].replace(0, np.nan)
        prev_calc["dpo"] = 365 * prev_calc["ap"] / prev_calc["cogs_ttm"].replace(0, np.nan)
        prev_calc["doh"] = 365 * prev_calc["inventory"] / prev_calc["cogs_ttm"].replace(0, np.nan)
        prev_calc["ccc"] = prev_calc["dso"] + prev_calc["doh"] - prev_calc["dpo"]
        last = last.merge(prev_calc[["customer_id", col]], on="customer_id", how="left", suffixes=("", "_prev"))
        last[f"delta_{col}_qoq"] = last[col] - last[f"{col}_prev"]
        last.drop(columns=[f"{col}_prev"], inplace=True, errors="ignore")

    return last[[
        "customer_id","sector_code","size_bucket",
        "icr_ttm","dscr_ttm_proxy","debt_to_ebitda","current_ratio",
        "dso","dpo","doh","ccc","delta_dso_qoq","delta_ccc_qoq"
    ]]

# ---------------------------
# Credit behavior (OW=180d)
# ---------------------------
def compute_behavioral_features(cr: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    start = asof - pd.Timedelta(days=OW_DAYS)
    win = cr[(cr["date"]>=start) & (cr["date"]<=asof)].copy()
    g = win.groupby("customer_id")

    # Utilization — lấy 60 ngày gần nhất của mỗi KH trong OW
    g_last60 = g.tail(60)  # groupby.tail giữ MultiIndex -> phải group lại
    util_mean_60 = (g_last60.groupby("customer_id")["utilized"].mean() /
                    g_last60.groupby("customer_id")["limit"].mean()).rename("%util_mean_60d")
    util_p95_60  = (g_last60.groupby("customer_id")["utilized"].quantile(0.95) /
                    g_last60.groupby("customer_id")["limit"].mean()).rename("%util_p95_60d")

    # Breach/DPD
    breach_cnt_90 = g.tail(90)["breach_flag"].groupby(level=0).sum().rename("limit_breach_cnt_90d")
    dpd_max_180   = g["dpd_days"].max().rename("dpd_max_180d")

    # dpd_trend: slope dpd theo ngày trong OW
    win["t"] = (win["date"] - start).dt.days # type: ignore
    slope = win.groupby("customer_id").apply(
        lambda x: np.polyfit(x["t"], x["dpd_days"], 1)[0] if len(x)>=2 else 0.0
    )
    slope.name = "dpd_trend_180d"

    # near_due_freq_7d: tỷ lệ ngày trong 7d gần nhất có 0 < DPD < 30
    last7 = win[win["date"]>asof - pd.Timedelta(days=7)].copy()
    near = (last7.assign(near=(last7["dpd_days"]>0) & (last7["dpd_days"]<30))
                 .groupby("customer_id")["near"].mean()
                 .rename("near_due_freq_7d"))

    feats = pd.concat([util_mean_60, util_p95_60, breach_cnt_90, dpd_max_180, slope, near], axis=1)
    return feats

# ---------------------------
# Cashflow (OW=180d)
# ---------------------------
def compute_cashflow_features(cf: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    start = asof - pd.Timedelta(days=OW_DAYS)
    win = cf[(cf["date"]>=start) & (cf["date"]<=asof)].copy()
    g = win.groupby("customer_id")
    inflow_mean_60  = g.tail(60)["inflow"].groupby(level=0).mean().rename("inflow_mean_60d")
    outflow_mean_60 = g.tail(60)["outflow"].groupby(level=0).mean().rename("outflow_mean_60d")
    inflow_med_6m   = g["inflow"].median().rename("inflow_median_6m")
    inflow_drop_60  = ((inflow_med_6m - inflow_mean_60) /
                       inflow_med_6m.replace(0,np.nan)).rename("inflow_drop_60d")
    return pd.concat([inflow_mean_60, outflow_mean_60, inflow_drop_60], axis=1)

# ---------------------------
# Covenant (OW=180d)
# ---------------------------
def compute_covenant_features(cov: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    start = asof - pd.Timedelta(days=OW_DAYS)
    win = cov[(cov["date"]>=start) & (cov["date"]<=asof)]
    g = win.groupby("customer_id")["breach_flag"]
    any_breach = g.max().rename("covenant_breach_any_180d")
    cnt_breach = g.sum().rename("covenant_breach_cnt_180d")
    return pd.concat([any_breach, cnt_breach], axis=1)

# ---------------------------
# Main make_features
# ---------------------------
def make_features(raw_dir: Path, asof: str = ASOF_DEFAULT,
                  winsor: bool=True, normalize: bool=True,
                  out_path: Path | None=None) -> pd.DataFrame:
    asof_ts = pd.Timestamp(asof)
    fin, cr, cf, cov, lab = load_raw(raw_dir)

    # Loại KH đang NPE/forborne (proxy): dpd_days ≥ 90 trong 7 ngày trước asof
    cr_asof = cr[(cr["date"]<=asof_ts) & (cr["date"]>asof_ts - pd.Timedelta(days=7))]
    bad_now = cr_asof.groupby("customer_id")["dpd_days"].max()
    exclude_ids = set(bad_now[bad_now>=90].index)

    fin_rat = compute_financial_ratios(fin)
    behav   = compute_behavioral_features(cr, asof_ts)
    cashf   = compute_cashflow_features(cf, asof_ts)
    covf    = compute_covenant_features(cov, asof_ts)

    df = (fin_rat
          .merge(behav, left_on="customer_id", right_index=True, how="left")
          .merge(cashf, left_on="customer_id", right_index=True, how="left")
          .merge(covf,  left_on="customer_id", right_index=True, how="left")
          .merge(lab[["customer_id","event_h12m"]], on="customer_id", how="left"))

    # Drop excluded
    df = df[~df["customer_id"].isin(exclude_ids)].reset_index(drop=True)

    # Winsorize & sector-size normalization
    ratio_cols = [
        "icr_ttm","dscr_ttm_proxy","debt_to_ebitda","current_ratio",
        "dso","dpo","doh","ccc","delta_dso_qoq","delta_ccc_qoq",
        "%util_mean_60d","%util_p95_60d","limit_breach_cnt_90d","dpd_max_180d","dpd_trend_180d","near_due_freq_7d",
        "inflow_mean_60d","outflow_mean_60d","inflow_drop_60d",
        "covenant_breach_cnt_180d"
    ]
    if winsor:
        for c in ratio_cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                df[c] = winsorize_series(df[c])

    if normalize:
        df = sector_size_normalize(df, [c for c in ratio_cols if c in df.columns],
                                   sector_col="sector_code", size_col="size_bucket")

    # Save
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_path.with_suffix(".parquet"), index=False)
        except Exception:
            df.to_csv(out_path.with_suffix(".csv"), index=False)

    return df

# ---------------------------
# CLI
# ---------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Build EWS features (Basel-aligned) for corporate portfolio.")
    p.add_argument("--raw-dir", type=str, required=True, help="Folder chứa fin_quarterly, credit_daily, cashflow_daily, covenant, labels (.parquet hoặc .csv)")
    p.add_argument("--asof", type=str, default=ASOF_DEFAULT)
    p.add_argument("--no-winsor", action="store_true", help="Tắt winsorize 1%/99%")
    p.add_argument("--no-normalize", action="store_true", help="Tắt chuẩn hoá theo (sector,size)")
    p.add_argument("--out", type=str, required=True, help="Đường dẫn output (không cần đuôi). VD: data/features_ews")
    return p.parse_args()

def main_cli():
    args = _parse_args()
    df = make_features(
        raw_dir=Path(args.raw_dir),
        asof=args.asof,
        winsor=(not args.no_winsor),
        normalize=(not args.no_normalize),
        out_path=Path(args.out)
    )
    print(f"Done. Features shape: {df.shape}. Saved to {Path(args.out).with_suffix('.parquet')} (or .csv).")

if __name__ == "__main__":
    main_cli()

# python src/features.py --raw-dir data/raw --asof 2025-06-30 --out data/processed/features_ews