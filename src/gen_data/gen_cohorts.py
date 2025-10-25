#!/usr/bin/env python3
"""Generate monthly cohorts with as_of_date for backtesting"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_synthetic_cohorts(start, end, n_customers=10000, seed=42, output="data/processed/backtest_cohorts.parquet"):
    """Generate synthetic monthly snapshots (simple, fast)"""
    print(f"Generating {start} → {end}, {n_customers} customers/month...")
    
    months = pd.period_range(start, end, freq='M').to_timestamp(how='end')
    sectors = ["Manufacturing","Construction","Transportation","Retail","Chemicals","Logistics","Engineering","Technology","Agriculture","Telecommunications"]
    grades = ["A","B","C","D","E","F","G"]
    grade_weights = [0.82,0.08,0.04,0.02,0.01,0.01,0.02]
    grade_pd = {"A":0.005,"B":0.010,"C":0.020,"D":0.040,"E":0.080,"F":0.120,"G":0.200}
    sector_mult = {"Construction":1.3,"Retail":1.2,"Manufacturing":1.1,"Agriculture":1.1,"Technology":0.8,"Telecommunications":0.9}
    
    rows = []
    for month in months:
        shock = np.sin(month.month/12*2*np.pi)*0.002 + (month.year-2023)*0.001
        for i in range(n_customers):
            np.random.seed(seed + hash(f"{month}{i}") % 2**31)
            sector = np.random.choice(sectors)
            grade = np.random.choice(grades, p=grade_weights)
            pd_pred = np.clip(grade_pd[grade] * sector_mult.get(sector,1.0) * (1+shock) + np.random.normal(0,0.003), 1e-6, 0.5)
            true_pd = np.clip(pd_pred * np.random.lognormal(0,0.3), 1e-6, 0.5)
            rows.append({
                'customer_id': f"C{i:05d}",
                'as_of_date': month,
                'sector': sector,
                'grade': grade,
                'pd_12m': pd_pred,
                'y_event_6m': int(np.random.rand() < true_pd*0.5),
                'y_event_12m': int(np.random.rand() < true_pd),
                'ead': np.random.lognormal(12,1)*1000,
                'lgd': np.clip(np.random.beta(4,6)+0.2, 0.2, 0.7)
            })
    
    df = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"{len(df):,} rows → {output}")
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-31")
    p.add_argument("--end", default="2025-06-30")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", default="data/processed/backtest_cohorts.parquet")
    a = p.parse_args()
    generate_synthetic_cohorts(a.start, a.end, a.n, a.seed, a.o)
