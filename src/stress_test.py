import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p/(1-p))

def inv_logit(z):
    return 1/(1+np.exp(-z))

def load_scenarios(path: Path):
    if yaml is None:
        raise ImportError("PyYAML is required (pip install pyyaml).")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["scenarios"]

def apply_stress_pd(df, shocks, sector_beta, grade_beta, elasticities, sector_multipliers=None, pd_cap=0.5):
    z = logit(df["pd_baseline"].values)
    
    # Apply macro shock elasticities
    for k, v in shocks.items():
        if k not in ["sector_multipliers"]:  # Skip non-macro keys
            z = z + elasticities.get(k, 0.0) * float(v)
    
    # Add sector and grade betas
    z = z + df["sector"].map(sector_beta).fillna(0.0).values
    z = z + df["grade"].map(grade_beta).fillna(0.0).values
    
    pd_s = inv_logit(z)
    
    # Apply sector-specific multipliers if provided
    if sector_multipliers:
        for sector, multiplier in sector_multipliers.items():
            mask = df["sector"] == sector
            pd_s[mask] = pd_s[mask] * multiplier
    
    return np.clip(pd_s, 1e-6, pd_cap)

def compute_impacts(df, pd_stress, provision_coverage=1.0, capital_alpha=0.08):
    out = df.copy()
    out["pd_stress"] = pd_stress
    out["npl_proxy"] = out["pd_stress"] * out["ead"]
    out["el"] = out["pd_stress"] * out["lgd"] * out["ead"]
    out["provision"] = provision_coverage * out["el"]
    out["capital"] = capital_alpha * out["ead"] * np.sqrt(np.clip(out["pd_stress"]*out["lgd"], 0, 1))
    out["credit_income"] = out["apr"] * out["ead"] * (1.0 - out["pd_stress"]) - out["el"]
    return out

def aggregate(df2, scenario):
    overall = df2.agg({"ead":"sum","npl_proxy":"sum","el":"sum","provision":"sum","capital":"sum","credit_income":"sum"}).to_frame().T
    overall["level"]="overall"; overall["key"]="all"
    by_sector = df2.groupby("sector").agg({"ead":"sum","npl_proxy":"sum","el":"sum","provision":"sum","capital":"sum","credit_income":"sum"}).reset_index().rename(columns={"sector":"key"}); by_sector["level"]="sector"
    by_grade = df2.groupby("grade").agg({"ead":"sum","npl_proxy":"sum","el":"sum","provision":"sum","capital":"sum","credit_income":"sum"}).reset_index().rename(columns={"grade":"key"}); by_grade["level"]="grade"
    out = pd.concat([overall, by_sector, by_grade], ignore_index=True)
    out.insert(0,"scenario",scenario)
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", default="data/processed/portfolio_scored.csv")
    parser.add_argument("--scenarios", default="artifacts/stress_testing/stress_scenarios.yaml")
    parser.add_argument("--output", default="artifacts/stress_testing/stress_results.csv")
    parser.add_argument("--provision_coverage", type=float, default=1.0)
    parser.add_argument("--capital_alpha", type=float, default=0.08)
    args = parser.parse_args()

    portfolio_path = Path(args.portfolio)
    scenarios_path = Path(args.scenarios)
    output_path = Path(args.output)
    
    if not portfolio_path.exists():
        print(f"Error: Portfolio file not found: {portfolio_path}")
        print(f"Please run: python src/create_portfolio.py --output {portfolio_path}")
        return 1
    
    df = pd.read_csv(portfolio_path)
    scenarios = load_scenarios(scenarios_path)

    elasticities = {"gdp_growth": -0.20, "unemployment": 0.35, "rates": 0.15, "cpi": 0.10}
    sector_beta = {"Manufacturing":0.05,"Retail":0.10,"Services":0.00,"Construction":0.12,"Agriculture":0.03}
    grade_beta = {"A":-0.10,"B":-0.05,"C":0.0,"D":0.05,"E":0.10,"F":0.20,"G":0.30}

    frames=[]
    for scen in scenarios:
        name = scen["name"]
        shocks = scen["shocks"]
        sector_mult = scen.get("sector_multipliers", None)
        pd_stress = apply_stress_pd(df, shocks, sector_beta, grade_beta, elasticities, sector_mult)
        df_imp = compute_impacts(df, pd_stress, args.provision_coverage, args.capital_alpha)
        frames.append(aggregate(df_imp, name))
    res = pd.concat(frames, ignore_index=True)

    base = res[res["scenario"]=="Base"][["level","key","ead","npl_proxy","el","provision","capital","credit_income"]].set_index(["level","key"])
    deltas=[]
    for name in res["scenario"].unique():
        if name=="Base": 
            continue
        cur = res[res["scenario"]==name][["level","key","ead","npl_proxy","el","provision","capital","credit_income"]].set_index(["level","key"])
        joined = cur.join(base, lsuffix="_cur", rsuffix="_base", how="left").reset_index()
        joined.insert(0,"scenario",name)
        for col in ["npl_proxy","el","provision","capital","credit_income"]:
            joined[f"delta_{col}"] = joined[f"{col}_cur"] - joined[f"{col}_base"]
        deltas.append(joined)
    if deltas:
        deltas_df = pd.concat(deltas, ignore_index=True)
        res = res.merge(
            deltas_df[["scenario","level","key","delta_npl_proxy","delta_el","delta_provision","delta_capital","delta_credit_income"]],
            on=["scenario","level","key"], how="left"
        )
    res.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)
    
    overall = res[(res["level"]=="overall") & (res["key"]=="all")].copy()
    overall = overall.sort_values("scenario")
    
    print("\nPortfolio-wide Impact Summary:")
    print("-"*70)
    print(f"{'Scenario':<25} {'NPL (M)':<12} {'Δ NPL':<12} {'Provision (M)':<15}")
    print("-"*70)
    
    for _, row in overall.iterrows():
        scen = row["scenario"]
        npl = row["npl_proxy"] / 1e6
        prov = row["provision"] / 1e6
        delta_npl = row.get("delta_npl_proxy", 0) / 1e6 if pd.notna(row.get("delta_npl_proxy")) else 0
        
        print(f"{scen:<25} {npl:>10.2f}  {delta_npl:>+10.2f}  {prov:>12.2f}")
    
    print("="*70)
    print(f"\nTotal Scenarios Run: {len(scenarios)}")
    print(f"Portfolio Size: {len(df):,} customers")
    print(f"Total EAD: ${df['ead'].sum()/1e6:,.2f}M")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()