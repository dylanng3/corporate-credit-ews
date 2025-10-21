#!/usr/bin/env python
"""
Create portfolio from scores_all.csv (calibrated PD) for stress testing.
Maps PD to grades, assigns EAD/LGD/APR by grade & sector.
"""

import argparse
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

# Grade thresholds: A-F cutoffs for PD
GRADE_THRESHOLDS = [0.005, 0.01, 0.02, 0.04, 0.07, 0.12]
GRADE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

# Sector mapping: short code -> full name
SECTOR_MAPPING = {
    "MFG": "Manufacturing",
    "CON": "Construction",
    "IT": "Technology",
    "ENG": "Engineering",
    "TRA": "Transportation",
    "CHE": "Chemicals",
    "LOG": "Logistics",
    "RET": "Retail",
    "AGR": "Agriculture",
    "SVC": "Services",
    "TEL": "Telecommunications",
}


def seed_from_id(customer_id: str, base: int = 42) -> int:
    """Deterministic seed from customer_id using blake2b hash"""
    h = hashlib.blake2b(str(customer_id).encode(), digest_size=4).digest()
    return base + int.from_bytes(h, "little")


def map_grade(pd_score: float, thresholds: list = GRADE_THRESHOLDS) -> str:
    """Map PD to grade (A-G)"""
    for i, threshold in enumerate(thresholds):
        if pd_score < threshold:
            return GRADE_LABELS[i]
    return GRADE_LABELS[-1]


def assign_ead(grade: str, sector: str, seed: int = 42) -> float:
    """Assign EAD based on grade & sector (better grades get higher limits)"""
    rng = np.random.default_rng(seed)
    
    # Base EAD by grade
    grade_base = {
        "A": 150_000, "B": 120_000, "C": 100_000, "D": 80_000,
        "E": 60_000, "F": 40_000, "G": 25_000
    }
    
    # Sector multiplier (using full names to match stress_test.py)
    sector_mult = {
        "Manufacturing": 1.3,
        "Construction": 1.2,
        "Technology": 1.1,
        "Engineering": 1.1,
        "Transportation": 1.0,
        "Chemicals": 1.0,
        "Logistics": 1.0,
        "Retail": 0.9,
        "Agriculture": 0.8,
        "Services": 0.85,
        "Telecommunications": 1.0,
    }
    
    base = grade_base.get(grade, 50_000)
    mult = sector_mult.get(sector, 1.0)
    
    # Add randomness ±20%
    return float(base * mult * rng.uniform(0.8, 1.2))


def assign_lgd(grade: str, sector: str) -> float:
    """Assign LGD (worse grades = higher LGD, clipped [0.25, 0.70])"""
    # Base LGD by grade
    grade_lgd = {
        "A": 0.30, "B": 0.35, "C": 0.40, "D": 0.45,
        "E": 0.50, "F": 0.55, "G": 0.60
    }
    
    # Sector adjustment (using full names)
    sector_adj = {
        "Manufacturing": -0.05,
        "Construction": 0.00,
        "Technology": -0.03,
        "Engineering": -0.02,
        "Transportation": 0.00,
        "Chemicals": 0.00,
        "Logistics": 0.02,
        "Retail": 0.05,
        "Agriculture": 0.08,
        "Services": 0.03,
        "Telecommunications": -0.01,
    }
    
    lgd = grade_lgd.get(grade, 0.45) + sector_adj.get(sector, 0.0)
    return float(np.clip(lgd, 0.25, 0.70))


def assign_apr(grade: str) -> float:
    """Assign APR (worse grades = higher APR)"""
    grade_apr = {
        "A": 0.08, "B": 0.10, "C": 0.12, "D": 0.14,
        "E": 0.16, "F": 0.18, "G": 0.22
    }
    return grade_apr.get(grade, 0.14)


def create_portfolio(
    scores_path: Path,
    output_path: Path,
    seed: int = 42,
    id_col: str = "customer_id",
    sector_col: str = "sector_code",
    pd_col: str = "prob_calibrated"
) -> pd.DataFrame:
    """Create portfolio: map PD to grades, assign EAD/LGD/APR"""
    # Read scores
    print(f"Reading scores from: {scores_path}")
    scores = pd.read_csv(scores_path)
    
    # Validate required columns
    required_cols = [id_col, sector_col, pd_col]
    missing_cols = [col for col in required_cols if col not in scores.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Portfolio size: {len(scores):,} customers")
    
    # Create portfolio
    portfolio = pd.DataFrame()
    portfolio["customer_id"] = scores[id_col]
    
    # Map sector code to full name
    portfolio["sector"] = scores[sector_col].map(SECTOR_MAPPING)
    
    # Handle unmapped sectors
    unmapped = portfolio["sector"].isna()
    if unmapped.any():
        print(f"⚠️  Warning: {unmapped.sum()} customers have unmapped sectors, defaulting to 'Services'")
        portfolio.loc[unmapped, "sector"] = "Services"
    
    # Clip PD to reasonable range [1e-6, 0.5] to avoid extreme values
    portfolio["pd_baseline"] = scores[pd_col].clip(1e-6, 0.5)
    
    # Map grade from PD
    print("Mapping credit grades from PD scores...")
    portfolio["grade"] = portfolio["pd_baseline"].apply(map_grade)
    
    # Assign EAD, LGD, APR using deterministic seeds
    print("Assigning EAD, LGD, APR...")
    
    portfolio["ead"] = portfolio.apply(
        lambda row: assign_ead(
            row["grade"],
            row["sector"],
            seed=seed_from_id(row["customer_id"], base=seed)
        ),
        axis=1
    )
    
    portfolio["lgd"] = portfolio.apply(
        lambda row: assign_lgd(row["grade"], row["sector"]),
        axis=1
    )
    
    portfolio["apr"] = portfolio["grade"].apply(assign_apr)
    
    # Reorder columns
    portfolio = portfolio[[
        "customer_id", "sector", "grade", "pd_baseline",
        "ead", "lgd", "apr"
    ]]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(output_path, index=False)
    print(f"Saved portfolio to: {output_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("PORTFOLIO SUMMARY")
    print("="*70)
    
    print("\nGrade Distribution:")
    grade_stats = portfolio.groupby("grade").agg({
        "customer_id": "count",
        "pd_baseline": "mean",
        "ead": "sum",
        "lgd": "mean",
        "apr": "mean"
    }).rename(columns={"customer_id": "count"})
    grade_stats["pct"] = (grade_stats["count"] / len(portfolio) * 100).round(1)
    grade_stats["ead_M"] = (grade_stats["ead"] / 1_000_000).round(2)
    print(grade_stats[["count", "pct", "pd_baseline", "ead_M", "lgd", "apr"]].to_string())
    
    print("\nSector Distribution:")
    sector_stats = portfolio.groupby("sector").agg({
        "customer_id": "count",
        "pd_baseline": "mean",
        "ead": "sum"
    }).rename(columns={"customer_id": "count"})
    sector_stats["pct"] = (sector_stats["count"] / len(portfolio) * 100).round(1)
    sector_stats["ead_M"] = (sector_stats["ead"] / 1_000_000).round(2)
    print(sector_stats[["count", "pct", "pd_baseline", "ead_M"]].to_string())
    
    print("\nOverall Statistics:")
    print(f"  Total EAD:        ${portfolio['ead'].sum()/1_000_000:.2f}M")
    print(f"  Avg PD:           {portfolio['pd_baseline'].mean():.4f} ({portfolio['pd_baseline'].mean()*100:.2f}%)")
    print(f"  Avg LGD:          {portfolio['lgd'].mean():.4f} ({portfolio['lgd'].mean()*100:.2f}%)")
    print(f"  Avg APR:          {portfolio['apr'].mean():.4f} ({portfolio['apr'].mean()*100:.2f}%)")
    print(f"  Expected Loss:    ${(portfolio['pd_baseline'] * portfolio['lgd'] * portfolio['ead']).sum()/1_000_000:.2f}M")
    print("="*70)
    
    return portfolio


def main():
    parser = argparse.ArgumentParser(description="Create portfolio for stress testing")
    parser.add_argument("--scores", type=Path, default=Path("data/processed/scores_calibrated.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/portfolio_scored.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-col", type=str, default="customer_id")
    parser.add_argument("--sector-col", type=str, default="sector_code")
    parser.add_argument("--pd-col", type=str, default="prob_calibrated")
    
    args = parser.parse_args()
    
    if not args.scores.exists():
        print(f"Error: Scores file not found: {args.scores}")
        return 1
    
    create_portfolio(
        args.scores,
        args.output,
        args.seed,
        args.id_col,
        args.sector_col,
        args.pd_col
    )
    return 0


if __name__ == "__main__":
    exit(main())
