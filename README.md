# Climate Risk Engine

A Python-based financial risk scoring engine that models how physical and transition climate risks flow through to credit metrics — PD, LGD, and EAD — and computes Climate VaR across multiple scenarios.

## What it does

- Maps portfolio holdings to climate risk exposure using NGFS Phase 4 (2023), IPCC AR6 SSP, and IEA WEO 2023 scenarios
- Models transition risk via carbon price shocks to EBITDA margins, adjusted by sector emissions intensity
- Models physical risk via hazard scores (flood, heat stress, wildfire, sea level rise, drought, cyclone) scaled to warming pathways
- Produces climate-adjusted PD, LGD, and expected loss per holding
- Aggregates to Climate VaR at 95% and 99% confidence levels across the portfolio
- Interactive Streamlit dashboard for scenario analysis

## Methodology

Aligned with TCFD recommendations and NGFS Phase 4 guidance. Three core NGFS scenarios:

| Scenario | Warming | Risk Profile |
|---|---|---|
| Net Zero 2050 | <1.5°C | High transition, low physical |
| Delayed Transition | ~2°C | Spike transition post-2030, moderate physical |
| Current Policies | 4°C+ | Low transition, high physical |

## Setup

```bash
git clone https://github.com/animsingh2/climate-risk-engine.git
cd climate-risk-engine
pip install -r requirements.txt
python run_dashboard.py
```

## Stack

Python 3.13 · Streamlit · Pandas · dataclasses

## Status

Work in progress. Hazard scores are proxy-based; production use would require licensed physical risk data (JRC, FEMA, or equivalent).
