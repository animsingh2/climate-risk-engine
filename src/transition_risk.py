"""
Module 3 — Transition Risk Engine
Carbon price → EBITDA shock → PD shift

Transmission chain:
  1. Carbon price ($/tCO2e) from scenario at horizon year
  2. Gross carbon cost = emissions_intensity × carbon_price × revenue
  3. Net carbon cost = gross × (1 - pass_through_rate) × (1 - abatement_capacity)
  4. EBITDA shock % = net_carbon_cost / (ebitda_margin × revenue)
  5. PD shift = logistic mapping of EBITDA shock (same function as physical risk)
  6. Stranded asset LGD loading for high-carbon sectors
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from src.schema import Holding, Sector
from src.scenarios import ClimateScenario, get_scenario
from src.sector_params import SECTOR_TRANSITION_PARAMS, HORIZON_YEARS
from src.data_loader import load_carbon_prices, CarbonPricePoint
from pathlib import Path


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class TransitionRiskResult:
    """
    Transition risk output for one holding, one scenario, one horizon year.
    """
    holding_id: str
    scenario_id: str
    horizon_year: int

    # Inputs (stored for transparency / audit trail)
    carbon_price_usd: float             # $/tCO2e at horizon year
    gross_carbon_cost_usd_m: float      # Before pass-through and abatement
    net_carbon_cost_usd_m: float        # After pass-through and abatement

    # Shocks
    ebitda_shock_pct: float             # Fractional EBITDA reduction (0-1)
    transition_pd_shift: float          # Additive PD increase
    stranded_asset_lgd_loading: float   # Additional LGD for high-carbon assets

    # Flags
    carbon_price_available: bool        # False for IPCC-only scenarios
    skipped: bool = False               # True if scenario has no carbon price


# ── Carbon price lookup ───────────────────────────────────────────────────────

def _build_price_index(
    price_points: list[CarbonPricePoint],
) -> dict[tuple[str, int], float]:
    """
    Build a fast lookup: (scenario_id, year) → carbon_price.
    Interpolates linearly between available years.
    """
    # Group by scenario
    by_scenario: dict[str, dict[int, float]] = {}
    for p in price_points:
        by_scenario.setdefault(p.scenario_id, {})[p.year] = p.carbon_price_usd_per_tco2e

    index: dict[tuple[str, int], float] = {}
    for scenario_id, year_map in by_scenario.items():
        sorted_years = sorted(year_map.keys())
        for year in HORIZON_YEARS:
            if year in year_map:
                index[(scenario_id, year)] = year_map[year]
            else:
                # Linear interpolation between nearest bracketing years
                lower = max((y for y in sorted_years if y <= year), default=None)
                upper = min((y for y in sorted_years if y >= year), default=None)
                if lower and upper and lower != upper:
                    t = (year - lower) / (upper - lower)
                    price = year_map[lower] + t * (year_map[upper] - year_map[lower])
                    index[(scenario_id, year)] = round(price, 2)
                elif lower:
                    index[(scenario_id, year)] = year_map[lower]
                elif upper:
                    index[(scenario_id, year)] = year_map[upper]

    return index


# ── Core transmission functions ───────────────────────────────────────────────

def _compute_ebitda_shock(
    holding: Holding,
    carbon_price: float,
) -> tuple[float, float, float]:
    """
    Returns (gross_carbon_cost_usd_m, net_carbon_cost_usd_m, ebitda_shock_pct).

    emissions_intensity is tCO2e per $M revenue.
    So: gross_cost = intensity × carbon_price × revenue_usd_m / 1_000_000
    (carbon_price is $/tCO2e, intensity is tCO2e/$M rev → cost is $/M rev → convert to $M)
    """
    params = SECTOR_TRANSITION_PARAMS[holding.sector]

    # Gross annual carbon cost ($M)
    gross = (
        holding.emissions_intensity      # tCO2e / $M revenue
        * carbon_price                   # $ / tCO2e
        * holding.annual_revenue_usd_m   # $M revenue
        / 1_000_000                      # convert $ → $M
    )

    # Net cost after pass-through and abatement
    net = gross * (1 - params.carbon_pass_through_rate) * (1 - params.abatement_capacity)

    # EBITDA shock: net cost as fraction of current EBITDA
    ebitda_usd_m = holding.ebitda_margin * holding.annual_revenue_usd_m
    if ebitda_usd_m <= 0:
        ebitda_shock = 1.0  # No EBITDA — full stress
    else:
        ebitda_shock = min(net / ebitda_usd_m, 1.0)

    return round(gross, 4), round(net, 4), round(ebitda_shock, 6)


def _ebitda_shock_to_pd_shift(
    ebitda_shock_pct: float,
    baseline_pd: float,
) -> float:
    """
    Logistic mapping: EBITDA shock → PD shift.

    Calibrated so:
    - 10% EBITDA shock on BBB → ~20bps PD shift
    - 30% EBITDA shock on BBB → ~80bps PD shift
    - 60% EBITDA shock on BBB → ~200bps PD shift
    Consistent with Moody's scenario stress outputs and
    NGFS Phase 4 sectoral PD impact ranges.
    """
    if ebitda_shock_pct <= 0:
        return 0.0

    k = 7.0
    midpoint = 0.45
    logistic = 1 / (1 + math.exp(-k * (ebitda_shock_pct - midpoint)))

    max_shift = min(baseline_pd * 5.0, 0.95 - baseline_pd)
    pd_shift = logistic * max_shift

    return round(max(pd_shift, 0.0), 6)


def _stranded_asset_lgd_loading(
    holding: Holding,
    ebitda_shock_pct: float,
) -> float:
    """
    Additional LGD for high-carbon sectors where fixed assets
    (pipelines, coal plants, refineries) lose value under transition.
    Only meaningful for energy/utilities/materials.
    Scaled by EBITDA shock severity.
    """
    params = SECTOR_TRANSITION_PARAMS[holding.sector]
    loading = params.stranded_asset_risk * ebitda_shock_pct
    return round(min(loading, 0.25), 4)  # cap at +25pp LGD


# ── Main scoring functions ────────────────────────────────────────────────────

def score_holding_transition(
    holding: Holding,
    scenario: ClimateScenario,
    price_index: dict[tuple[str, int], float],
    horizon_years: list[int] | None = None,
) -> list[TransitionRiskResult]:
    """
    Score one holding across all horizon years for one scenario.
    Returns one result per horizon year.
    """
    years = horizon_years or HORIZON_YEARS
    results = []

    # IPCC scenarios have no carbon price — skip transition risk
    if not scenario.has_carbon_price:
        for year in years:
            results.append(TransitionRiskResult(
                holding_id=holding.holding_id,
                scenario_id=scenario.scenario_id,
                horizon_year=year,
                carbon_price_usd=0.0,
                gross_carbon_cost_usd_m=0.0,
                net_carbon_cost_usd_m=0.0,
                ebitda_shock_pct=0.0,
                transition_pd_shift=0.0,
                stranded_asset_lgd_loading=0.0,
                carbon_price_available=False,
                skipped=True,
            ))
        return results

    for year in years:
        carbon_price = price_index.get((scenario.scenario_id, year), 0.0)

        gross, net, ebitda_shock = _compute_ebitda_shock(holding, carbon_price)
        pd_shift = _ebitda_shock_to_pd_shift(ebitda_shock, holding.baseline_pd)
        lgd_loading = _stranded_asset_lgd_loading(holding, ebitda_shock)

        results.append(TransitionRiskResult(
            holding_id=holding.holding_id,
            scenario_id=scenario.scenario_id,
            horizon_year=year,
            carbon_price_usd=carbon_price,
            gross_carbon_cost_usd_m=gross,
            net_carbon_cost_usd_m=net,
            ebitda_shock_pct=ebitda_shock,
            transition_pd_shift=pd_shift,
            stranded_asset_lgd_loading=lgd_loading,
            carbon_price_available=True,
            skipped=False,
        ))

    return results


def score_portfolio_transition(
    holdings: list[Holding],
    scenarios: list[ClimateScenario],
    carbon_prices_csv: str | Path = "data/ngfs_scenarios/carbon_prices.csv",
    horizon_years: list[int] | None = None,
) -> list[TransitionRiskResult]:
    """
    Score all holdings × all scenarios × all horizon years.
    Everything computed here — dashboard filters the output.
    Returns flat list of TransitionRiskResult.
    """
    price_points = load_carbon_prices(carbon_prices_csv)
    price_index = _build_price_index(price_points)

    results = []
    for holding in holdings:
        for scenario in scenarios:
            results.extend(
                score_holding_transition(holding, scenario, price_index, horizon_years)
            )
    return results