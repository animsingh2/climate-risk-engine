# Climate Risk Engine

A Python-based climate financial risk scoring engine that models how physical and transition climate risks flow through to credit metrics — PD, LGD, and EAD — and computes Climate VaR across multiple scenarios. Methodology follows TCFD recommendations and NGFS Phase 4 (2023) guidance.

---

## What it does

- Maps portfolio holdings to climate risk exposure using NGFS Phase 4 (2023), IPCC AR6 SSP, and IEA WEO 2023 scenarios
- Models transition risk via carbon price shocks to EBITDA margins (Earnings at Risk approach), adjusted by sector emissions intensity
- Models physical risk via hazard scores across six hazard types — flood, heat stress, wildfire, sea level rise, drought, and tropical cyclone — scaled to warming pathways
- Computes climate-adjusted PD (additive PD overlay), LGD, and expected loss per holding
- Aggregates to Climate VaR at 95% and 99% confidence via parametric and Monte Carlo methods
- Produces Expected Shortfall and sector-level correlation matrices
- Interactive Streamlit dashboard for scenario comparison

---

## Methodology

Aligned with TCFD recommendations and NGFS Phase 4 guidance. The transition risk approach follows the Earnings at Risk methodology documented in UNEP FI (2023) and referenced in the UNEP FI–SAS Climate Stress Testing report (2025) as the most common industry approach. Physical risk follows the Hazard–Vulnerability–Exposure framework from NGFS (2023) and IPCC AR6.

### NGFS scenarios

| Scenario | Warming | Transition Risk | Physical Risk |
|---|---|---|---|
| Net Zero 2050 | <1.5°C | High (early, orderly) | Low |
| Delayed Transition | ~2°C | High (abrupt post-2030) | Moderate |
| Current Policies | 4°C+ | Low | High |

Also supports IPCC AR6 SSPs (SSP1-2.6, SSP2-4.5, SSP5-8.5) and IEA WEO 2023 (NZE, APS, STEPS).

Climate VaR is computed here as a portfolio-level credit risk metric — expected loss at a given confidence interval — consistent with banking and UNEP FI frameworks. This is distinct from equity-valuation-based Climate VaR approaches (e.g. MSCI), which express climate costs as a percentage devaluation of enterprise value using a Merton-type credit model. Both are valid; the choice here reflects the credit risk use case rather than an asset management one.

---

## Project structure

```
climate-risk-engine/
├── src/
│   ├── schema.py             # Domain objects and enumerations (Holding, Portfolio, ClimateRiskResult)
│   ├── scenarios.py          # Scenario registry (NGFS, IPCC AR6, IEA WEO)
│   ├── data_loader.py        # CSV loaders for portfolio and carbon price inputs
│   ├── hazard_scorer.py      # Physical hazard scoring by country, sector, hazard type
│   ├── transition_risk.py    # Carbon cost → EBITDA shock → PD shift (logistic function)
│   ├── credit_aggregator.py  # Expected Loss, Climate VaR, Expected Shortfall
│   └── dashboard.py          # Streamlit dashboard
├── data/
│   ├── inputs/               # Portfolio template CSV
│   └── ngfs_scenarios/       # Carbon price paths by scenario
├── tests/                    # 54 automated tests across all modules
├── docs/                     # Technical methodology and plain-language summary
└── run_dashboard.py
```

---

## Setup

```bash
git clone https://github.com/animsingh2/climate-risk-engine.git
cd climate-risk-engine
pip install -r requirements.txt
python run_dashboard.py
```

---

## Stack

Python 3.13 · Streamlit · Pandas · Plotly · dataclasses

---

## Known limitations

- **Hazard scores are proxy-based.** Physical risk uses country-sector proxies rather than asset-level geolocation. Production-grade implementations would use licensed data (JRC, FEMA, or equivalent). This is flagged as a data gap even among large banks in UNEP FI (2025).
- **Static balance sheet.** Portfolio composition is held constant over the scenario horizon, consistent with the majority of banks per the UNEP FI–SAS survey (2025), but limits the ability to model strategic repositioning.
- **No macroeconomic transmission channels.** The model uses direct carbon cost transmission only; indirect channels (GDP, CPI, unemployment) are not modelled.
- **Single horizon year.** The engine evaluates risk at a specified horizon year rather than producing multi-year transition paths.
- **Physical and transition risks are additive, not compounded.** The UNEP FI–SAS survey (2025) finds only 24% of banks currently compound these interactions.

---

## References

- NGFS (2023). *NGFS Climate Scenarios Phase 4*
- IPCC (2021). *Sixth Assessment Report — SSP Scenarios*
- IEA (2023). *World Energy Outlook 2023*
- TCFD (2017). *Recommendations of the Task Force on Climate-related Financial Disclosures*
- UNEP FI & SAS (2025). *Climate Stress Testing Methodologies: Current Practices, Challenges, and the Road Ahead*
- UNEP FI (2023). *The 2023 Climate Risk Landscape*