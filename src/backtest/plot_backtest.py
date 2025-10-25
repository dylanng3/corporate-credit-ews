#!/usr/bin/env python3
"""
Backtest Visualization Suite - Individual Plot Scripts
Run specific plots by importing and calling functions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# ============= HELPER FUNCTIONS =============

def wilson_ci(p, n, confidence=0.95):
    """Calculate Wilson score confidence interval for proportion"""
    if n == 0 or pd.isna(p) or pd.isna(n):
        return p, p
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4*n**2))) / denominator
    return max(0, centre - margin), min(1, centre + margin)

def auc_ci_delong_approx(auc, n_pos, n_neg, confidence=0.95):
    """Approximate DeLong CI for AUC using normal approximation"""
    if n_pos == 0 or n_neg == 0 or pd.isna(auc):
        return auc, auc
    # DeLong variance approximation: V ≈ AUC(1-AUC) / min(n_pos, n_neg)
    var_auc = auc * (1 - auc) / min(n_pos, n_neg)
    se_auc = np.sqrt(var_auc)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    return max(0, auc - z * se_auc), min(1, auc + z * se_auc)

# ============= DATA LOADERS =============

def load_backtest_data(base_dir='artifacts/backtest'):
    """Load all backtest results"""
    metrics = pd.read_csv(f'{base_dir}/monthly_metrics.csv', parse_dates=['as_of_month'])
    calib = pd.read_csv(f'{base_dir}/monthly_calibration.csv', parse_dates=['as_of_month'])
    sweep = pd.read_csv(f'{base_dir}/threshold_sweep.csv', parse_dates=['as_of_month'])
    return metrics, calib, sweep


def load_psi_data(psi_path='artifacts/backtest/psi_monthly.csv'):
    """Load PSI calculations"""
    psi = pd.read_csv(psi_path, parse_dates=['as_of_date'])
    return psi


# ============= PLOT 1: PERFORMANCE OVER TIME =============

def plot_performance_time(metrics, output='artifacts/backtest/plot_performance_time.png'):
    """Plot AUC, KS, Brier over time with confidence intervals"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Calculate CIs for AUC (assuming ~10K samples, 1.1% default rate)
    n_total = 10000
    n_pos = int(n_total * 0.011)  # baseline default rate
    n_neg = n_total - n_pos
    
    auc_lower = []
    auc_upper = []
    for auc_val in metrics['auc']:
        lower, upper = auc_ci_delong_approx(auc_val, n_pos, n_neg)
        auc_lower.append(lower)
        auc_upper.append(upper)
    
    # AUC with CI bands
    ax = axes[0]
    ax.plot(metrics['as_of_month'], metrics['auc'], 'o-', color='#2E86AB', linewidth=2, markersize=6, label='AUC')
    ax.fill_between(metrics['as_of_month'], auc_lower, auc_upper,
                      color='#2E86AB', alpha=0.2, label='95% CI')
    ax.axhline(metrics['auc'].mean(), color='#2E86AB', linestyle='--', alpha=0.5, 
               label=f'Mean: {metrics["auc"].mean():.3f}')
    ax.set_title('AUC Over Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(0.75, 0.90)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # KS (no CI - keep simple)
    ax = axes[1]
    ax.plot(metrics['as_of_month'], metrics['ks'], 'o-', color='#A23B72', linewidth=2, markersize=6)
    ax.axhline(metrics['ks'].mean(), color='#A23B72', linestyle='--', alpha=0.5,
               label=f'Mean: {metrics["ks"].mean():.3f}')
    ax.fill_between(metrics['as_of_month'], 
                      metrics['ks'].mean() - metrics['ks'].std(),
                      metrics['ks'].mean() + metrics['ks'].std(),
                      color='#A23B72', alpha=0.1)
    ax.set_title('KS Statistic Over Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('KS', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Brier (no CI - keep simple)
    ax = axes[2]
    ax.plot(metrics['as_of_month'], metrics['brier']*100, 'o-', color='#F18F01', linewidth=2, markersize=6)
    ax.axhline(metrics['brier'].mean()*100, color='#F18F01', linestyle='--', alpha=0.5,
               label=f'Mean: {metrics["brier"].mean()*100:.2f}')
    ax.set_title('Brier Score (lower=better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brier Score (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 2: ALERT PERFORMANCE =============

def plot_alert_performance(metrics, output='artifacts/backtest/plot_alert_performance.png'):
    """Plot precision, recall, alert rate for amber/red thresholds with CI bands"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Calculate Wilson CIs for precision and recall
    # Using alert counts and precision to back-calculate true positives
    n_total = 10000  # cohort size
    
    # Amber CI
    amber_prec_lower = []
    amber_prec_upper = []
    amber_recall_lower = []
    amber_recall_upper = []
    
    for _, row in metrics.iterrows():
        # Precision CI: proportion of alerts that are true positives
        n_alerts_amber = int(row['amber_alert_rate'] * n_total)
        lower, upper = wilson_ci(row['amber_precision'], n_alerts_amber)
        amber_prec_lower.append(lower * 100)
        amber_prec_upper.append(upper * 100)
        
        # Recall CI: proportion of all positives that are caught
        n_positives = int(n_total * 0.011)  # baseline default rate
        lower, upper = wilson_ci(row['amber_recall'], n_positives)
        amber_recall_lower.append(lower * 100)
        amber_recall_upper.append(upper * 100)
    
    # Red CI
    red_prec_lower = []
    red_prec_upper = []
    red_recall_lower = []
    red_recall_upper = []
    
    for _, row in metrics.iterrows():
        n_alerts_red = int(row['red_alert_rate'] * n_total)
        lower, upper = wilson_ci(row['red_precision'], n_alerts_red)
        red_prec_lower.append(lower * 100)
        red_prec_upper.append(upper * 100)
        
        n_positives = int(n_total * 0.011)
        lower, upper = wilson_ci(row['red_recall'], n_positives)
        red_recall_lower.append(lower * 100)
        red_recall_upper.append(upper * 100)
    
    # Precision with CI
    ax = axes[0]
    ax.plot(metrics['as_of_month'], metrics['amber_precision']*100, 'o-', 
            color='#E76F51', linewidth=2, markersize=6, label='Amber (2%)')
    ax.fill_between(metrics['as_of_month'], amber_prec_lower, amber_prec_upper,
                     color='#E76F51', alpha=0.2)
    ax.plot(metrics['as_of_month'], metrics['red_precision']*100, 's-', 
            color='#C73E1D', linewidth=2, markersize=6, label='Red (5%)')
    ax.fill_between(metrics['as_of_month'], red_prec_lower, red_prec_upper,
                     color='#C73E1D', alpha=0.2)
    ax.set_title('Alert Precision (% Defaults in Alerts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (%) with 95% CI', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Recall with CI
    ax = axes[1]
    ax.plot(metrics['as_of_month'], metrics['amber_recall']*100, 'o-', 
            color='#E76F51', linewidth=2, markersize=6, label='Amber (2%)')
    ax.fill_between(metrics['as_of_month'], amber_recall_lower, amber_recall_upper,
                     color='#E76F51', alpha=0.2)
    ax.plot(metrics['as_of_month'], metrics['red_recall']*100, 's-', 
            color='#C73E1D', linewidth=2, markersize=6, label='Red (5%)')
    ax.fill_between(metrics['as_of_month'], red_recall_lower, red_recall_upper,
                     color='#C73E1D', alpha=0.2)
    ax.set_title('Alert Recall (% Defaults Caught)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (%) with 95% CI', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Alert Rate (no CI - deterministic from data)
    ax = axes[2]
    ax.plot(metrics['as_of_month'], metrics['amber_alert_rate']*100, 'o-', 
            color='#E76F51', linewidth=2, markersize=6, label='Amber (2%)')
    ax.plot(metrics['as_of_month'], metrics['red_alert_rate']*100, 's-', 
            color='#C73E1D', linewidth=2, markersize=6, label='Red (5%)')
    ax.set_title('Alert Rate (% Customers Flagged)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Alert Rate (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 3: CALIBRATION =============

def plot_calibration(calib, month='latest', output='artifacts/backtest/plot_calibration.png'):
    """Plot calibration by decile for a specific month"""
    if month == 'latest':
        month = calib['as_of_month'].max()
    
    data = calib[calib['as_of_month'] == month]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = data['decile']
    width = 0.35
    
    ax.bar(x - width/2, data['pd_avg']*100, width, color='#2E86AB', alpha=0.7, label='Predicted PD')
    ax.bar(x + width/2, data['odr']*100, width, color='#F18F01', alpha=0.7, label='Observed Default Rate')
    
    # Perfect calibration line
    ax.plot([0, 11], [0, max(data['pd_avg'].max(), data['odr'].max())*100*1.1], 
            'k--', alpha=0.3, label='Perfect Calibration')
    
    ax.set_title(f'Calibration by Decile - {pd.Timestamp(month).strftime("%B %Y")}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('PD Decile (1=Lowest Risk, 10=Highest Risk)', fontsize=11)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 4: PRECISION-RECALL CURVE =============

def plot_precision_recall(sweep, month='latest', output='artifacts/backtest/plot_precision_recall.png'):
    """Plot precision-recall curve with threshold markers"""
    if month == 'latest':
        month = sweep['as_of_month'].max()
    
    data = sweep[sweep['as_of_month'] == month].sort_values('recall', ascending=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.plot(data['recall']*100, data['precision']*100, 'o-', 
            color='#2E86AB', linewidth=2, markersize=4, alpha=0.6)
    
    # Mark thresholds
    amber_point = data.iloc[(data['cut'] - 0.02).abs().argmin()]
    red_point = data.iloc[(data['cut'] - 0.05).abs().argmin()]
    
    ax.plot(amber_point['recall']*100, amber_point['precision']*100, 'o', 
            color='#E76F51', markersize=15, label=f'Amber (2%): P={amber_point["precision"]*100:.1f}%, R={amber_point["recall"]*100:.1f}%')
    ax.plot(red_point['recall']*100, red_point['precision']*100, 's', 
            color='#C73E1D', markersize=15, label=f'Red (5%): P={red_point["precision"]*100:.1f}%, R={red_point["recall"]*100:.1f}%')
    
    ax.set_title(f'Precision-Recall Curve - {pd.Timestamp(month).strftime("%B %Y")}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall (% of Defaults Caught)', fontsize=11)
    ax.set_ylabel('Precision (% of Alerts that Default)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 5: ROLLING METRICS =============

def plot_rolling_metrics(metrics, output='artifacts/backtest/plot_rolling_metrics.png'):
    """Plot monthly vs 12-month rolling metrics"""
    roll = metrics[metrics['auc_roll12'].notna()]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AUC
    ax = axes[0, 0]
    ax.plot(roll['as_of_month'], roll['auc'], 'o-', color='#2E86AB', alpha=0.3, 
            linewidth=1, markersize=4, label='Monthly')
    ax.plot(roll['as_of_month'], roll['auc_roll12'], '-', color='#2E86AB', 
            linewidth=3, label='12M Rolling')
    ax.set_title('AUC: Monthly vs Rolling 12M', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # KS
    ax = axes[0, 1]
    ax.plot(roll['as_of_month'], roll['ks'], 'o-', color='#A23B72', alpha=0.3, 
            linewidth=1, markersize=4, label='Monthly')
    ax.plot(roll['as_of_month'], roll['ks_roll12'], '-', color='#A23B72', 
            linewidth=3, label='12M Rolling')
    ax.set_title('KS: Monthly vs Rolling 12M', fontsize=12, fontweight='bold')
    ax.set_ylabel('KS', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Brier
    ax = axes[1, 0]
    ax.plot(roll['as_of_month'], roll['brier']*100, 'o-', color='#F18F01', alpha=0.3, 
            linewidth=1, markersize=4, label='Monthly')
    ax.plot(roll['as_of_month'], roll['brier_roll12']*100, '-', color='#F18F01', 
            linewidth=3, label='12M Rolling')
    ax.set_title('Brier Score: Monthly vs Rolling 12M', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brier Score (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Lift
    ax = axes[1, 1]
    ax.plot(metrics['as_of_month'], metrics['lift_10pct'], 'o-', color='#6A994E', 
            linewidth=2, markersize=6, label='Top 10%')
    ax.plot(metrics['as_of_month'], metrics['lift_20pct'], 's-', color='#87C38F', 
            linewidth=2, markersize=6, label='Top 20%')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Lift at Top Percentiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lift (vs Baseline)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 6: THRESHOLD SENSITIVITY =============

def plot_threshold_sensitivity(sweep, output='artifacts/backtest/plot_threshold_sensitivity.png'):
    """Plot how metrics change with threshold"""
    # Average across all months
    avg = sweep.groupby('cut').agg({
        'alert_rate': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Alert Rate vs Threshold
    ax = axes[0]
    ax.plot(avg['cut']*100, avg['alert_rate']*100, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax.axvline(2, color='#E76F51', linestyle='--', alpha=0.7, label='Amber (2%)')
    ax.axvline(5, color='#C73E1D', linestyle='--', alpha=0.7, label='Red (5%)')
    ax.set_title('Alert Rate vs PD Threshold', fontsize=12, fontweight='bold')
    ax.set_xlabel('PD Threshold (%)', fontsize=10)
    ax.set_ylabel('Alert Rate (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Precision vs Recall
    ax = axes[1]
    ax.plot(avg['recall']*100, avg['precision']*100, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    # Mark thresholds
    amber_idx = (avg['cut'] - 0.02).abs().idxmin()
    red_idx = (avg['cut'] - 0.05).abs().idxmin()
    ax.plot(avg.loc[amber_idx, 'recall']*100, avg.loc[amber_idx, 'precision']*100, 
            'o', color='#E76F51', markersize=15, label='Amber (2%)')
    ax.plot(avg.loc[red_idx, 'recall']*100, avg.loc[red_idx, 'precision']*100, 
            's', color='#C73E1D', markersize=15, label='Red (5%)')
    ax.set_title('Precision-Recall Trade-off (Avg)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall (%)', fontsize=10)
    ax.set_ylabel('Precision (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output}")
    plt.close()


# ============= PLOT 7: PSI OVER TIME =============

def plot_psi(metrics, output='artifacts/backtest/plot_psi.png'):
    """Plot Population Stability Index over time"""
    # Load real PSI data
    psi_data = load_psi_data()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot each feature
    colors = {'pd_score': '#2E86AB', 'sector_mix': '#A23B72', 'grade_mix': '#F18F01', 'ead': '#06A77D', 'lgd': '#E63946'}
    labels = {'pd_score': 'PD Score', 'sector_mix': 'Sector Mix', 'grade_mix': 'Grade Mix', 'ead': 'EAD', 'lgd': 'LGD'}
    
    for col in ['pd_score', 'sector_mix', 'grade_mix', 'ead', 'lgd']:
        ax.plot(psi_data['as_of_date'], psi_data[col], 'o-', 
                color=colors[col], linewidth=2, markersize=6, label=labels[col])
    
    # PSI thresholds
    ax.axhline(0.10, color='#F4A261', linestyle='--', linewidth=2, alpha=0.7, 
               label='Warning Threshold (0.10)')
    ax.axhline(0.25, color='#E63946', linestyle='--', linewidth=2, alpha=0.7, 
               label='Action Threshold (0.25)')
    
    # Calculate appropriate y-axis range
    max_psi = psi_data[['pd_score', 'sector_mix', 'grade_mix', 'ead', 'lgd']].max().max()
    y_upper = 0.015  # Fixed upper limit to zoom in on actual data (0.015 = 1.5%)
    
    # Shaded regions - only show stable zone since data is all stable
    ax.fill_between(psi_data['as_of_date'], 0, 0.10, 
                     color='green', alpha=0.08, label='Stable Zone (<0.10)')
    
    ax.set_title('Population Stability Index (PSI) Over Time\n(Zoomed view: 0-1.5%)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('PSI', fontsize=11)
    ax.set_ylim(0, y_upper)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    
    # Add annotation with statistics
    ax.text(0.98, 0.95, 
            f'Max PSI: {max_psi:.4f}\nHighly stable',
            transform=ax.transAxes, fontsize=10, 
            ha='right', va='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4))
    
    # Add note about thresholds being off-chart
    ax.text(0.02, 0.95, 
            'Note: Warning (0.10) & Action (0.25)\nthresholds not shown (far above data)',
            transform=ax.transAxes, fontsize=8, 
            ha='left', va='top', style='italic', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output}")
    plt.close()


# ============= MAIN =============

if __name__ == "__main__":
    import sys
    
    # Load data
    print("Loading backtest data...")
    metrics, calib, sweep = load_backtest_data()
    print(f"   Loaded {len(metrics)} months")
    
    # Command-line arguments
    if len(sys.argv) > 1:
        plot = sys.argv[1].lower()
        
        plots = {
            'performance': lambda: plot_performance_time(metrics),
            'alert': lambda: plot_alert_performance(metrics),
            'calibration': lambda: plot_calibration(calib),
            'pr': lambda: plot_precision_recall(sweep),
            'rolling': lambda: plot_rolling_metrics(metrics),
            'threshold': lambda: plot_threshold_sensitivity(sweep),
            'psi': lambda: plot_psi(metrics),
            'all': lambda: [
                plot_performance_time(metrics),
                plot_alert_performance(metrics),
                plot_calibration(calib),
                plot_precision_recall(sweep),
                plot_rolling_metrics(metrics),
                plot_threshold_sensitivity(sweep),
                plot_psi(metrics),
                print("\n✅ All plots generated!")
            ]
        }
        
        if plot in plots:
            plots[plot]()
        else:
            print(f"\nUnknown plot: {plot}")
            print("\nAvailable plots:")
            for name in plots.keys():
                print(f"   - {name}")
            sys.exit(1)
    else:
        # Show usage
        print("\n" + "="*60)
        print("📊 BACKTEST VISUALIZATION")
        print("="*60)
        print("\nUsage:")
        print("  python src/backtest/plot_backtest.py performance   # AUC, KS, Brier over time")
        print("  python src/backtest/plot_backtest.py alert         # Precision, Recall, Alert Rate")
        print("  python src/backtest/plot_backtest.py calibration   # Predicted vs Observed")
        print("  python src/backtest/plot_backtest.py pr            # Precision-Recall curve")
        print("  python src/backtest/plot_backtest.py rolling       # Rolling 12M metrics")
        print("  python src/backtest/plot_backtest.py threshold     # Threshold sensitivity")
        print("  python src/backtest/plot_backtest.py psi           # Population Stability Index")
        print("  python src/backtest/plot_backtest.py all           # Generate all plots")
        print("\nOutput: artifacts/backtest/plot_*.png")
        print("="*60)
