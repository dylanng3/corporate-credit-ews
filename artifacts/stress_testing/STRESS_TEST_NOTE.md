# STRESS TEST NOTE
**Corporate Credit Portfolio - Macroeconomic Stress Testing**  
*Date: October 21, 2025*  
*Model Version: 1.0.0*

---

## Executive Summary

This stress test evaluates the resilience of the corporate credit portfolio under various macroeconomic scenarios, ranging from mild recession to severe economic crisis. The analysis quantifies potential impacts on NPL formation, provisioning requirements, capital adequacy, and credit income.

**Key Findings:**
- **Severe Recession**: NPL increases by **$145.3M** (+451%), provisions rise to **$79.8M**
- **Moderate Recession**: NPL increases by **$100.1M** (+310%), provisions at **$59.6M**
- **Interest Rate Shock**: NPL increases by **$64.5M** (+200%), provisions at **$43.6M**
- **Base Case**: NPL at **$32.2M**, provisions at **$14.6M**

---

## Portfolio Overview

| Metric | Value |
|--------|-------|
| Total Customers | 5,000 |
| Total EAD | $510.6M |
| Baseline NPL | $32.2M (6.3% of EAD) |
| Average PD (Baseline) | 3.8% |
| Weighted LGD | 45% |

**Grade Distribution:**
- Investment Grade (A-C): 50%
- Sub-Investment Grade (D-E): 32%
- High Risk (F-G): 18%

**Sector Distribution:**
- Manufacturing, Retail, Services, Construction, Agriculture

---

## Scenarios Tested

### 1. **Mild Recession**
- GDP: -2%, Unemployment: +1.5pp, Rates: +0.5pp
- **Impact**: NPL +$42.1M, Provision +$19.0M

### 2. **Moderate Recession**
- GDP: -4%, Unemployment: +3pp, Rates: +1pp
- **Impact**: NPL +$100.1M, Provision +$45.0M

### 3. **Severe Recession** (2008-style)
- GDP: -6%, Unemployment: +5pp
- **Impact**: NPL +$145.3M, Provision +$65.2M
- **Credit Income**: Turns negative at -$31.2M

### 4. **Stagflation**
- GDP: -1%, Unemployment: +2pp, Rates: +3pp, CPI: +5pp
- **Impact**: NPL +$80.4M, Provision +$36.1M

### 5. **Interest Rate Shock**
- Rates: +4pp, GDP: -1.5%, Unemployment: +1pp
- **Impact**: NPL +$64.5M, Provision +$29.0M

### 6. **Sector-Specific (Construction Crisis)**
- Construction sector PD doubles, spillover to Manufacturing
- **Impact**: NPL +$38.0M, Provision +$17.1M

---

## Sectoral Analysis (Severe Recession)

| Sector | EAD ($M) | NPL ($M) | Δ NPL ($M) | Provision ($M) |
|--------|----------|----------|------------|----------------|
| Manufacturing | 102.1 | 35.6 | +29.1 | 16.0 |
| Retail | 102.3 | 35.5 | +29.0 | 16.0 |
| Services | 102.2 | 35.5 | +29.0 | 16.0 |
| Construction | 102.0 | 35.5 | +29.1 | 16.0 |
| Agriculture | 102.1 | 35.5 | +29.1 | 15.9 |

*Approximately equal distribution due to synthetic portfolio*

---

## Grade Migration Impact (Severe Recession)

| Grade | Baseline NPL ($M) | Stressed NPL ($M) | Δ NPL ($M) | % Increase |
|-------|-------------------|-------------------|------------|------------|
| A | 2.9 | 6.9 | +4.0 | +138% |
| B | 5.7 | 14.2 | +8.5 | +149% |
| C | 12.0 | 29.9 | +17.9 | +149% |
| D | 24.1 | 59.4 | +35.3 | +146% |
| E | 34.6 | 86.1 | +51.5 | +149% |
| F | 54.7 | 134.3 | +79.6 | +146% |
| G | 109.5 | 267.3 | +157.8 | +144% |

**Key Observations:**
- Higher grades show proportionally larger increases (leverage effect)
- Grade E-G contribute 75% of total NPL increase
- Grade A-C still manageable but require monitoring

---

## Capital & Provision Requirements

### Severe Recession Scenario
- **Additional Provisions Needed**: $65.2M (4.5x baseline)
- **Capital Buffer Required**: $15.5M (vs $5.8M baseline)
- **Credit Income Impact**: -$88.9M (turns deeply negative)

### Moderate Recession Scenario
- **Additional Provisions**: $45.0M (3.1x baseline)
- **Capital Buffer**: $11.5M
- **Credit Income Impact**: -$51.0M

### Break-even Analysis
- Portfolio can absorb **Mild Recession** with existing provisions
- **Moderate Recession** requires $45M additional capital
- **Severe Recession** would require significant capital injection or portfolio restructuring

---

## Methodology

### PD Uplift Model
```
PD_stressed = invlogit(
    logit(PD_baseline)
    + β_GDP × ΔGDP
    + β_unemployment × Δunemployment
    + β_rates × Δrates
    + β_sector
    + β_grade
)
```

**Elasticities Used:**
- GDP Growth: -0.20 (negative growth increases PD)
- Unemployment: +0.35
- Interest Rates: +0.15
- CPI: +0.10

**Sector Betas:** Manufacturing (+0.05), Retail (+0.10), Construction (+0.12)  
**Grade Betas:** A (-0.10), B (-0.05), C (0.0), D (+0.05), E (+0.10), F (+0.20), G (+0.30)

### Impact Calculations
- **NPL Proxy**: PD × EAD
- **Expected Loss**: PD × LGD × EAD
- **Provision**: Coverage × EL (100% coverage assumed)
- **Capital**: 8% × EAD × √(PD × LGD) (simplified Basel approach)
- **Credit Income**: APR × EAD × (1 - PD) - EL

---

## Risk Mitigation Recommendations

### Immediate Actions (0-3 months)
1. **Increase Provisions**: Build provision buffer to $80M to cover Moderate scenario
2. **Enhance Monitoring**: Weekly tracking of early warning signals for grades D-G
3. **Sector Review**: Deep dive into Construction and Retail sectors

### Medium-term (3-12 months)
1. **Portfolio Rebalancing**: Reduce exposure to high-risk grades (F-G) by 20%
2. **Pricing Adjustment**: Increase APR for grades D-E to compensate for higher risk
3. **Collateral Review**: Re-evaluate LGD assumptions, improve collateral coverage

### Long-term (12+ months)
1. **Stress Testing Cadence**: Quarterly stress tests with updated macro forecasts
2. **Dynamic Provisioning**: Implement forward-looking ECL model
3. **Capital Planning**: Maintain 15% capital buffer above regulatory minimum

---

## Sensitivity Analysis

**PD Uplift Sensitivity** (Severe Recession):
- Elasticity +20%: NPL increases to $195M (+$163M vs baseline)
- Elasticity -20%: NPL at $160M (+$128M vs baseline)

**LGD Sensitivity**:
- LGD +10pp (55% avg): Provision requirement +$13M
- LGD -10pp (35% avg): Provision requirement -$13M

---

## Conclusion

The stress test reveals significant vulnerabilities in severe downside scenarios. While the portfolio can withstand mild economic turbulence, a moderate-to-severe recession would require substantial capital injections and active portfolio management.

**Traffic Light Assessment:**
- 🟢 **Mild Recession**: Manageable with existing buffers
- 🟡 **Moderate Recession**: Requires immediate capital planning
- 🔴 **Severe Recession**: Critical - immediate action required

**Next Steps:**
1. Present findings to Credit Committee and Board
2. Develop contingency funding plan for $100M provision buffer
3. Implement enhanced monitoring for construction and high-risk grades
4. Update ICAAP and recovery planning with stress test results

---

*Generated by: Corporate Credit EWS - Stress Testing Module v1.0*  
*Contact: Risk Management Department*
