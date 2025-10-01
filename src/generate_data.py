#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Corporate EWS dataset generator (Basel-aligned, compact)
Outputs (under --output-dir):
  - fin_quarterly.parquet|csv
  - credit_daily.parquet|csv
  - cashflow_daily.parquet|csv
  - covenant.parquet|csv
  - labels.parquet|csv
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Config ----------------
@dataclass
class Config:
    random_seed: int = 42
    n_customers: int = 1000
    sectors: Tuple[str, ...] = ("MFG","TRA","CON","AGR","ENG","CHE","RET","LOG","TEL","IT")
    sector_probs: Tuple[float, ...] = (0.18,0.12,0.10,0.08,0.08,0.10,0.12,0.10,0.06,0.06)
    size_buckets: Tuple[str, ...] = ("SME","Corp")
    size_probs: Tuple[float, ...] = (0.8,0.2)

    # Financials (quarterly history up to end_quarter)
    end_quarter: str = "2025-06-30"
    n_quarters: int = 12

    # Snapshot & windows
    asof_date: str = "2025-06-30"       # mốc snapshot (month-end)
    behavior_days: int = 180            # quan sát hành vi lùi lại từ asof
    label_horizon_days: int = 365       # gắn nhãn 12M sau asof

    # Label rule
    dpd_threshold: int = 90
    min_streak_days: int = 30

    # Optional bump (only if event=0)
    util_rate_bump_cutoff: float = 0.90
    util_bump: float = 0.20
    cov_breach_bump: float = 0.20

    # Output
    output_dir: str = "data/raw"

# --------------- Helpers ----------------
def max_consecutive_geq(vals: List[float], threshold: float) -> int:
    cnt = best = 0
    for v in vals:
        cnt = cnt + 1 if v >= threshold else 0
        if cnt > best: best = cnt
    return best

def save_table(df: pd.DataFrame, base: Path):
    """Try Parquet; fallback CSV."""
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(base.with_suffix(".csv"), index=False)

# --------------- Generators --------------
def make_customers(cfg: Config) -> pd.DataFrame:
    ids = [f"C{str(i).zfill(4)}" for i in range(1, cfg.n_customers+1)]
    return pd.DataFrame({
        "customer_id": ids,
        "sector_code": np.random.choice(cfg.sectors, size=cfg.n_customers, p=cfg.sector_probs),
        "size_bucket": np.random.choice(cfg.size_buckets, size=cfg.n_customers, p=cfg.size_probs)
    })

def generate_fin_quarterly(customers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    end_q = pd.Timestamp(cfg.end_quarter)
    fq_dates = pd.period_range(end=end_q, periods=cfg.n_quarters, freq="Q").to_timestamp(how="end")
    rows = []
    for cid, sector, size in tqdm(customers[["customer_id","sector_code","size_bucket"]].itertuples(index=False),
                                  total=len(customers), desc="Financials (quarterly)"):
        base_rev = np.random.lognormal(mean=12 if size=="Corp" else 10.5, sigma=0.5)
        growth = np.random.normal(0.02, 0.03)    # QoQ
        margin = np.clip(np.random.normal(0.15, 0.07), 0.02, 0.35)
        sector_risk = {"MFG":0.0,"TRA":0.05,"CON":0.03,"AGR":0.04,"ENG":0.0,"CHE":0.02,"RET":0.03,"LOG":0.02,"TEL":-0.01,"IT":-0.02}[sector]
        debt_mult = 0.8 + (0.6 if size=="Corp" else 0.4) + np.random.normal(0, 0.2)

        rev = base_rev
        for fq in fq_dates:
            rev = max(1e5, rev * (1 + growth + np.random.normal(0, 0.05)))
            cogs = rev * np.clip(0.75 + np.random.normal(0, 0.03), 0.6, 0.92)
            ebitda = rev * np.clip(margin + np.random.normal(0, 0.03), 0.02, 0.4)
            ebit = ebitda - rev * np.clip(0.02 + np.random.normal(0, 0.01), 0.0, 0.06)  # D&A proxy
            ar = rev * np.clip(0.18 + np.random.normal(0, 0.05), 0.05, 0.45)
            ap = cogs * np.clip(0.20 + np.random.normal(0, 0.05), 0.05, 0.5)
            inventory = cogs * np.clip(0.12 + np.random.normal(0, 0.03), 0.02, 0.35)
            current_assets = ar + inventory + rev * np.clip(0.03 + np.random.normal(0, 0.01), 0.005, 0.1)
            current_liab = ap + rev * np.clip(0.02 + np.random.normal(0, 0.01), 0.005, 0.1)
            total_debt = max(0.0, rev * (0.3 + sector_risk + np.random.normal(0, 0.1))) * (1 + (debt_mult - 1) * 0.5)
            interest = max(1e3, total_debt * np.clip(0.08 + np.random.normal(0, 0.02), 0.03, 0.2)) / 4
            rows.append([cid, fq, sector, size, rev, cogs, ebitda, ebit, interest,
                        total_debt, current_assets, current_liab, inventory, ar, ap])
    return pd.DataFrame(rows, columns=[
        "customer_id","fq_date","sector_code","size_bucket","revenue","cogs","ebitda","ebit",
        "interest_expense","total_debt","current_assets","current_liab","inventory","ar","ap"
    ])

def _daily_window(cfg: Config):
    asof = pd.Timestamp(cfg.asof_date)
    start = asof - pd.Timedelta(days=cfg.behavior_days - 1)
    end   = asof + pd.Timedelta(days=cfg.label_horizon_days)
    return pd.date_range(start, end, freq="D"), asof

def generate_credit_daily(customers: pd.DataFrame, fin_quarterly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates, _ = _daily_window(cfg)
    rows = []
    for cid in tqdm(customers["customer_id"].tolist(), desc="Credit (daily)"):
        last_rev = fin_quarterly.loc[fin_quarterly["customer_id"]==cid].sort_values("fq_date")["revenue"].iloc[-1]
        limit = max(1e5, last_rev * 0.3 * np.random.uniform(0.8, 1.2))
        util_level = np.clip(np.random.beta(2,2), 0.05, 0.98)
        dpd = 0
        for d in dates:
            util = np.clip(util_level + np.sin(d.dayofyear/365*2*np.pi)*0.05 + np.random.normal(0,0.05), 0, 1.2)
            utilized = np.clip(limit * util, 0, limit * 1.2)
            breach = int(utilized > limit)
            # Markov-ish DPD
            if np.random.rand() < 0.985:
                dpd = max(0, dpd - np.random.binomial(1, 0.3))
            else:
                dpd += np.random.choice([1,3,7], p=[0.6,0.3,0.1])
            product = np.random.choice(["OD","TERM","TR_LOAN"], p=[0.7,0.2,0.1])
            rows.append([cid, d, limit, utilized, breach, int(dpd), product])
    return pd.DataFrame(rows, columns=["customer_id","date","limit","utilized","breach_flag","dpd_days","product_type"])

def generate_cashflow_daily(customers: pd.DataFrame, fin_quarterly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates, _ = _daily_window(cfg)
    rows = []
    for cid in tqdm(customers["customer_id"].tolist(), desc="Cashflow (daily)"):
        last_rev = fin_quarterly.loc[fin_quarterly["customer_id"]==cid].sort_values("fq_date")["revenue"].iloc[-1]
        mu = last_rev/365*np.random.uniform(0.6,1.1)
        for d in dates:
            season = 1 + 0.2*np.sin((d.dayofyear/365)*2*np.pi)
            inflow = max(0, np.random.normal(mu*season, mu*0.3))
            outflow = max(0, np.random.normal(mu*season*0.9, mu*0.25))
            rows.append([cid, d, inflow, outflow])
    return pd.DataFrame(rows, columns=["customer_id","date","inflow","outflow"])

def generate_covenant(customers: pd.DataFrame, fin_quarterly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates, _ = _daily_window(cfg)
    months = pd.period_range(start=dates[0], end=dates[-1], freq="M").to_timestamp(how="end")
    rows = []
    for cid in tqdm(customers["customer_id"].tolist(), desc="Covenants (monthly)"):
        fq_hist = fin_quarterly.loc[fin_quarterly["customer_id"]==cid].sort_values("fq_date").tail(4)
        ebit_ttm = fq_hist["ebit"].sum(); int_ttm = fq_hist["interest_expense"].sum()
        icr_ttm = (ebit_ttm/(int_ttm+1e-6)) if int_ttm>0 else 5.0
        ebitda_ttm = fin_quarterly.loc[fin_quarterly["customer_id"]==cid].sort_values("fq_date").tail(4)["ebitda"].sum()
        debt_last = fq_hist["total_debt"].iloc[-1] if len(fq_hist) else 0
        ca = fq_hist["current_assets"].iloc[-1] if len(fq_hist) else 1.0
        cl = fq_hist["current_liab"].iloc[-1] if len(fq_hist) else 1.0
        de_ratio = (debt_last/(ebitda_ttm+1e-6)) if ebitda_ttm>0 else 10.0
        curr_ratio = ca/(cl+1e-6)
        dscr_proxy = (ebitda_ttm/(int_ttm+1e-6)) if int_ttm>0 else 5.0
        for m in months:
            rows.append([cid,m,"ICR_min",2.0,icr_ttm,int(icr_ttm < 2.0*np.random.uniform(0.9,1.1))])
            rows.append([cid,m,"Debt_to_EBITDA_max",4.0,de_ratio,int(de_ratio > 4.0*np.random.uniform(0.9,1.1))])
            rows.append([cid,m,"Current_Ratio_min",1.0,curr_ratio,int(curr_ratio < 1.0*np.random.uniform(0.9,1.1))])
            rows.append([cid,m,"DSCR_min_proxy",1.1,dscr_proxy,int(dscr_proxy < 1.1*np.random.uniform(0.9,1.1))])
    return pd.DataFrame(rows, columns=["customer_id","date","covenant_name","threshold","actual","breach_flag"])

def build_labels(customers: pd.DataFrame, credit_daily: pd.DataFrame, covenant: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    asof = pd.Timestamp(cfg.asof_date)
    horizon_end = asof + pd.Timedelta(days=cfg.label_horizon_days)
    labels = []
    for cid in tqdm(customers["customer_id"].tolist(), desc="Labels (12M)"):
        future_credit = credit_daily[(credit_daily["customer_id"]==cid) & (credit_daily["date"]>asof) & (credit_daily["date"]<=horizon_end)]
        streak90 = max_consecutive_geq(future_credit["dpd_days"].tolist(), cfg.dpd_threshold)
        event = 1 if streak90 >= cfg.min_streak_days else 0
        if event == 0:
            past30 = credit_daily[(credit_daily["customer_id"]==cid) & (credit_daily["date"]<=asof)].tail(30)
            util_rate = (past30["utilized"].mean()/past30["limit"].mean()) if len(past30)>0 and past30["limit"].mean()>0 else 0.0
            future_cov = covenant[(covenant["customer_id"]==cid) & (covenant["date"]>asof) & (covenant["date"]<=horizon_end) & (covenant["breach_flag"]==1)]
            bump = (cfg.util_bump if util_rate > cfg.util_rate_bump_cutoff else 0.0) + (cfg.cov_breach_bump if len(future_cov)>0 else 0.0)
            if np.random.random() < bump: event = 1
        labels.append([cid, asof, event])
    return pd.DataFrame(labels, columns=["customer_id","asof_date","event_h12m"])

# ---------------- Main -------------------
def run(cfg: Config) -> dict:
    np.random.seed(cfg.random_seed)
    outdir = Path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    customers = make_customers(cfg)
    fin_quarterly = generate_fin_quarterly(customers, cfg)
    credit_daily  = generate_credit_daily(customers, fin_quarterly, cfg)
    cashflow_daily= generate_cashflow_daily(customers, fin_quarterly, cfg)
    covenant     = generate_covenant(customers, fin_quarterly, cfg)
    labels       = build_labels(customers, credit_daily, covenant, cfg)

    save_table(fin_quarterly, outdir/"fin_quarterly")
    save_table(credit_daily,  outdir/"credit_daily")
    save_table(cashflow_daily,outdir/"cashflow_daily")
    save_table(covenant,      outdir/"covenant")
    save_table(labels,        outdir/"labels")

    return {
        "output_dir": str(outdir.resolve()),
        "summary": {
            "n_customers": len(customers),
            "fin_quarterly_shape": fin_quarterly.shape,
            "credit_daily_shape": credit_daily.shape,
            "cashflow_daily_shape": cashflow_daily.shape,
            "covenant_shape": covenant.shape,
            "labels_shape": labels.shape,
            "event_rate_h12m": float(labels["event_h12m"].mean()),
        },
    }

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate synthetic EWS dataset (Corporate/SME)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-customers", type=int, default=1000)
    p.add_argument("--end-quarter", type=str, default="2025-06-30")
    p.add_argument("--n-quarters", type=int, default=12)
    p.add_argument("--asof-date", type=str, default="2025-06-30")
    p.add_argument("--behavior-days", type=int, default=180)
    p.add_argument("--label-horizon-days", type=int, default=365)
    p.add_argument("--dpd-threshold", type=int, default=90)
    p.add_argument("--min-streak-days", type=int, default=30)
    p.add_argument("--util-cutoff", type=float, default=0.90)
    p.add_argument("--util-bump", type=float, default=0.20)
    p.add_argument("--cov-breach-bump", type=float, default=0.20)
    p.add_argument("--output-dir", type=str, default="data/raw")
    a = p.parse_args()
    return Config(
        random_seed=a.seed, n_customers=a.n_customers,
        end_quarter=a.end_quarter, n_quarters=a.n_quarters,
        asof_date=a.asof_date, behavior_days=a.behavior_days, label_horizon_days=a.label_horizon_days,
        dpd_threshold=a.dpd_threshold, min_streak_days=a.min_streak_days,
        util_rate_bump_cutoff=a.util_cutoff, util_bump=a.util_bump, cov_breach_bump=a.cov_breach_bump,
        output_dir=a.output_dir
    )

if __name__ == "__main__":
    cfg = parse_args()
    out = run(cfg)
    print("Output directory:", out["output_dir"])
    print("Summary:", out["summary"])

# python src/generate_data.py --n-customers 10000
