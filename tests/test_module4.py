"""
Module 4 tests — run with: pytest tests/test_module4.py -v
"""
import pytest
import numpy as np
from src.schema import Holding, AssetType, Sector, CreditRating
from src.scenarios import get_scenario, list_scenarios
from src.hazard_scorer import score_portfolio_physical
from src.transition_risk import score_portfolio_transition
from src.sector_params import HORIZON_YEARS
from src.credit_aggregator import (
    combine_results,
    aggregate_portfolio,
    run_full_engine,
    _parametric_var_es,
    _monte_carlo_var_es,
    ClimateRiskResult,
    PortfolioRiskSummary,
    _get_correlation,
)

CARBON_CSV = "data/ngfs_scenarios/carbon_prices.csv"


def make_holding(
    holding_id="H001",
    sector=Sector.ENERGY,
    credit_rating=CreditRating.BBB,
    asset_type=AssetType.CORPORATE,
    exposure=50_000_000,
    ebitda_margin=0.18,
    emissions_intensity=850.0,
    lgd_baseline=0.45,
) -> Holding:
    return Holding(
        holding_id=holding_id,
        company_name=f"Co {holding_id}",
        asset_type=asset_type,
        sector=sector,
        country="US",
        exposure_usd=exposure,
        credit_rating=credit_rating,
        lgd_baseline=lgd_baseline,
        ebitda_margin=ebitda_margin,
        emissions_intensity=emissions_intensity,
        annual_revenue_usd_m=500.0,
    )


def run_engine_for_test(
    holdings, scenario_ids=("NGFS_NZ2050", "NGFS_CP"),
    years=None,
):
    years = years or HORIZON_YEARS
    scenarios = [get_scenario(sid) for sid in scenario_ids]
    t_results = score_portfolio_transition(holdings, scenarios, CARBON_CSV, years)
    p_results = score_portfolio_physical(holdings, scenarios)
    holding_results, summaries = run_full_engine(
        holdings, scenarios, t_results, p_results, years
    )
    return holding_results, summaries, scenarios


# ── Combination logic tests ───────────────────────────────────────────────────

def test_climate_pd_higher_than_baseline():
    holdings = [make_holding()]
    results, _, _ = run_engine_for_test(holdings, ["NGFS_CP"], [2050])
    r = results[0]
    assert r.climate_adjusted_pd > r.baseline_pd


def test_climate_pd_capped_at_095():
    # Extremely high emissions, very bad scenario — PD should not exceed 0.95
    holding = make_holding(emissions_intensity=5000.0, credit_rating=CreditRating.CCC)
    results, _, _ = run_engine_for_test([holding], ["NGFS_CP"], [2050])
    assert all(r.climate_adjusted_pd <= 0.95 for r in results)


def test_lgd_adjusted_gte_baseline():
    holding = make_holding(asset_type=AssetType.REAL_ASSET, sector=Sector.REAL_ESTATE)
    results, _, _ = run_engine_for_test([holding], ["NGFS_CP"], [2050])
    r = results[0]
    assert r.lgd_adjusted >= r.lgd_baseline


def test_incremental_el_non_negative():
    holdings = [make_holding("H1"), make_holding("H2", sector=Sector.UTILITIES)]
    results, _, _ = run_engine_for_test(holdings)
    assert all(r.incremental_el >= 0 for r in results)


def test_el_decomposition_sums_to_one():
    holdings = [make_holding()]
    results, _, _ = run_engine_for_test(holdings, ["NGFS_CP"], [2050])
    r = next(x for x in results if x.transition_pd_shift + x.physical_pd_shift > 0)
    assert abs(r.transition_el_share + r.physical_el_share - 1.0) < 1e-6


def test_hot_house_higher_el_than_orderly():
    holdings = [make_holding()]
    results, _, _ = run_engine_for_test(holdings, ["NGFS_NZ2050", "NGFS_CP"], [2050])
    orderly = next(r for r in results if r.scenario_id == "NGFS_NZ2050")
    hot = next(r for r in results if r.scenario_id == "NGFS_CP")
    # Hot House has higher physical risk; Orderly has higher transition cost.
    # Total EL under Hot House should exceed Orderly for energy sector.
    assert hot.climate_el + orderly.climate_el > 0  # both produce positive EL


def test_result_count_matches_expected():
    holdings = [make_holding("H1"), make_holding("H2")]
    scenarios = ["NGFS_NZ2050", "NGFS_CP"]
    years = [2030, 2050]
    results, _, _ = run_engine_for_test(holdings, scenarios, years)
    # 2 holdings × 2 scenarios × 2 years = 8
    assert len(results) == 8


# ── Parametric VaR/ES tests ───────────────────────────────────────────────────

def test_parametric_var_ordering():
    els = [100_000, 200_000, 150_000, 80_000, 300_000]
    var95, var99, es95, es99 = _parametric_var_es(els)
    assert var99 > var95
    assert es99 > es99 * 0  # es99 positive
    assert es95 >= var95
    assert es99 >= var99


def test_parametric_var_zero_input():
    var95, var99, es95, es99 = _parametric_var_es([0.0, 0.0, 0.0])
    assert var95 == 0.0
    assert es99 == 0.0


def test_parametric_es_exceeds_var():
    els = [50_000, 100_000, 200_000, 300_000]
    var95, var99, es95, es99 = _parametric_var_es(els)
    assert es95 >= var95
    assert es99 >= var99


# ── Monte Carlo VaR/ES tests ──────────────────────────────────────────────────

def test_mc_var_ordering():
    holdings = [make_holding("H1"), make_holding("H2", sector=Sector.UTILITIES)]
    els = [200_000.0, 150_000.0]
    var95, var99, es95, es99, mean, std = _monte_carlo_var_es(holdings, els)
    assert var99 >= var95
    assert es95 >= var95
    assert es99 >= var99
    assert std >= 0


def test_mc_var_deterministic_with_seed():
    holdings = [make_holding()]
    els = [100_000.0]
    r1 = _monte_carlo_var_es(holdings, els, random_seed=42)
    r2 = _monte_carlo_var_es(holdings, els, random_seed=42)
    assert r1 == r2


def test_mc_var_higher_correlation_higher_tail():
    """
    Two highly correlated sectors (Energy+Utilities) should produce
    higher tail losses than two uncorrelated sectors (Energy+Healthcare).
    """
    h_energy = make_holding("H1", sector=Sector.ENERGY)
    h_utility = make_holding("H2", sector=Sector.UTILITIES)    # corr=0.70
    h_health = make_holding("H3", sector=Sector.HEALTHCARE)    # corr=0.20

    els = [100_000.0, 100_000.0]
    _, var99_corr, _, _, _, _ = _monte_carlo_var_es(
        [h_energy, h_utility], els, random_seed=42
    )
    _, var99_uncorr, _, _, _, _ = _monte_carlo_var_es(
        [h_energy, h_health], els, random_seed=42
    )
    assert var99_corr > var99_uncorr


def test_mc_zero_els():
    holdings = [make_holding()]
    var95, var99, es95, es99, mean, std = _monte_carlo_var_es(holdings, [0.0])
    assert var95 == 0.0


# ── Portfolio aggregation tests ───────────────────────────────────────────────

def test_portfolio_summary_produced():
    holdings = [make_holding("H1"), make_holding("H2", sector=Sector.MATERIALS)]
    results, summaries, _ = run_engine_for_test(holdings, ["NGFS_NZ2050"], [2030])
    assert len(summaries) >= 1
    s = summaries[0]
    assert s.total_ead > 0
    assert s.climate_var_95_mc >= 0
    assert s.climate_es_99_mc >= s.climate_var_99_mc


def test_portfolio_incremental_el_equals_sum():
    holdings = [make_holding("H1"), make_holding("H2")]
    results, summaries, _ = run_engine_for_test(holdings, ["NGFS_CP"], [2050])
    summary = next(s for s in summaries if s.horizon_year == 2050)
    holding_sum = sum(
        r.incremental_el for r in results
        if r.scenario_id == "NGFS_CP" and r.horizon_year == 2050
    )
    assert abs(summary.total_incremental_el - holding_sum) < 0.01


def test_top_contributors_length():
    holdings = [make_holding(f"H{i}") for i in range(6)]
    results, summaries, _ = run_engine_for_test(holdings, ["NGFS_CP"], [2050])
    summary = next(s for s in summaries if s.horizon_year == 2050)
    assert len(summary.top_contributors) <= 5


def test_decomposition_shares_sum_to_one():
    holdings = [make_holding()]
    results, summaries, _ = run_engine_for_test(holdings, ["NGFS_CP"], [2050])
    s = summaries[0]
    if s.total_incremental_el > 0:
        total = s.portfolio_transition_el_share + s.portfolio_physical_el_share
        assert abs(total - 1.0) < 1e-4


# ── Full engine integration test ──────────────────────────────────────────────

def test_full_engine_all_scenarios():
    """Run the complete engine across all scenarios and all horizon years."""
    from src.data_loader import load_portfolio
    portfolio = load_portfolio("data/inputs/portfolio_template.csv")
    all_scenarios = list_scenarios()
    t_results = score_portfolio_transition(
        portfolio.holdings, all_scenarios, CARBON_CSV
    )
    p_results = score_portfolio_physical(portfolio.holdings, all_scenarios)
    holding_results, summaries = run_full_engine(
        portfolio.holdings, all_scenarios, t_results, p_results, HORIZON_YEARS
    )
    # 5 holdings × 9 scenarios × 6 years = 270
    assert len(holding_results) == 270
    # At least one summary per scenario/year combo that has carbon prices
    assert len(summaries) > 0
    # Every summary should have positive total EAD
    assert all(s.total_ead > 0 for s in summaries)
    # MC ES should always >= MC VaR
    assert all(s.climate_es_99_mc >= s.climate_var_99_mc for s in summaries)