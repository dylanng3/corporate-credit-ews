#!/usr/bin/env python3
"""
Calculate Population Stability Index (PSI) for backtest cohorts
Compares each month's distribution to baseline (first month)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_psi_numeric(baseline, current, n_bins=10):
    """
    Calculate PSI for a numeric feature
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    # Remove NaN
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()
    
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return 0.0
    
    # Create bins based on baseline quantiles
    bins = np.percentile(baseline_clean, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    
    if len(bins) < 2:
        return 0.0
    
    # Ensure bins cover the range
    bins[0] = min(bins[0], current_clean.min()) - 1e-10
    bins[-1] = max(bins[-1], current_clean.max()) + 1e-10
    
    # Count frequencies in each bin
    baseline_counts, _ = np.histogram(baseline_clean, bins=bins)
    current_counts, _ = np.histogram(current_clean, bins=bins)
    
    # Convert to proportions (add small epsilon to avoid log(0))
    epsilon = 1e-5
    baseline_props = (baseline_counts + epsilon) / (baseline_counts.sum() + epsilon * len(bins))
    current_props = (current_counts + epsilon) / (current_counts.sum() + epsilon * len(bins))
    
    # Calculate PSI
    psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
    return psi


def calculate_psi_categorical(baseline, current):
    """
    Calculate PSI for a categorical feature
    """
    # Get proportions
    baseline_props = baseline.value_counts(normalize=True)
    current_props = current.value_counts(normalize=True)
    
    # Get all unique categories
    all_categories = set(baseline_props.index) | set(current_props.index)
    
    # Calculate PSI
    epsilon = 1e-5
    psi = 0
    for cat in all_categories:
        exp = baseline_props.get(cat, 0) + epsilon
        act = current_props.get(cat, 0) + epsilon
        psi += (act - exp) * np.log(act / exp)
    
    return psi


def main():
    """Calculate PSI for all months and features"""
    # Load data
    data_path = 'data/processed/backtest_cohorts.parquet'
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    
    print(f"   Loaded {len(df):,} rows, {df['as_of_date'].nunique()} unique months")
    
    # Baseline = first month
    baseline_date = df['as_of_date'].min()
    baseline = df[df['as_of_date'] == baseline_date].copy()
    print(f"   Baseline: {baseline_date.strftime('%Y-%m-%d')} ({len(baseline):,} customers)")
    
    # Calculate PSI for each month
    results = []
    for month in sorted(df['as_of_date'].unique()):
        current = df[df['as_of_date'] == month].copy()
        
        # Numeric features
        psi_pd = calculate_psi_numeric(baseline['pd_12m'], current['pd_12m'])
        psi_ead = calculate_psi_numeric(baseline['ead'], current['ead'])
        psi_lgd = calculate_psi_numeric(baseline['lgd'], current['lgd'])
        
        # Categorical features
        psi_sector = calculate_psi_categorical(baseline['sector'], current['sector'])
        psi_grade = calculate_psi_categorical(baseline['grade'], current['grade'])
        
        results.append({
            'as_of_date': month,
            'pd_score': psi_pd,
            'sector_mix': psi_sector,
            'grade_mix': psi_grade,
            'ead': psi_ead,
            'lgd': psi_lgd
        })
        
        print(f"   {month.strftime('%Y-%m')} | PD:{psi_pd:.4f} Sector:{psi_sector:.4f} Grade:{psi_grade:.4f} EAD:{psi_ead:.4f} LGD:{psi_lgd:.4f}")
    
    # Save results
    psi_df = pd.DataFrame(results)
    output_path = 'artifacts/backtest/psi_monthly.csv'
    psi_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved PSI results to {output_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("PSI SUMMARY")
    print("="*60)
    for col in ['pd_score', 'sector_mix', 'grade_mix', 'ead', 'lgd']:
        max_psi = psi_df[col].max()
        mean_psi = psi_df[col].mean()
        status = "✅ Stable" if max_psi < 0.10 else ("⚠️ Moderate" if max_psi < 0.25 else "🔴 Significant")
        print(f"{col.upper():15s} | Max: {max_psi:.4f} | Mean: {mean_psi:.4f} | {status}")
    print("="*60)
    
    return psi_df


if __name__ == "__main__":
    main()
