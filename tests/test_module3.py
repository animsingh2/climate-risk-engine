"""
Module 3 tests — run with: pytest tests/test_module3.py -v
"""
import pytest
from src.schema import Holding, AssetType, Sector, CreditRating
from src.scenarios import get_scenario, list_scenarios, ScenarioFramework
from src.sector_params import SECTOR_TRANSITION_PARAMS, HORIZON_YEARS
from src.transition_risk import (
    score_holding_transition,
    score_portfolio_transition,
    _build_price_index,
    _compute_ebitda_shock,
    _ebitda_shock_to_pd_shift,
    TransitionRiskResult,
)
from src.data_loader import load_carbon_prices, CarbonPricePoint


CARBON_CSV = "data/ngfs_scenarios/carbon_prices.csv"


def make_holding(
    holding_id="T001",
    sector=Sector.ENERGY,
    credit_rating=CreditRating.BBB,
    ebitda_margin=0.18,
    emissions_intensity=850.0,
    annual_revenue_usd_m=2200.0,
) -> Holding:
    return Holding(
        holding_id=holding_id,
        company_name="Test Co",
        asset_type=AssetType.CORPORATE,
        sector=sector,
        country="US",
        exposure_usd=50_000_000,
        credit_rating=credit_rating,
        lgd_baseline=0.45,
        ebitda_margin=ebitda_margin,
        emissions_intensity=emissions_intensity,
        annual_revenue_usd_m=annual_revenue_usd_m,
    )


# ── Price index tests ─────────────────────────────────────────────────────────

def test_price_index_builds():
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    assert ("NGFS_NZ2050", 2050) in index
    assert index[("NGFS_NZ2050", 2050)] == 250.0


def test_price_index_interpolation():
    # Add a scenario with only 2030 and 2050 — 2040 must be interpolated
    prices = [
        CarbonPricePoint("TEST", 2030, 100.0),
        CarbonPricePoint("TEST", 2050, 200.0),
    ]
    index = _build_price_index(prices)
    # 2040 is halfway → should be ~150
    assert abs(index[("TEST", 2040)] - 150.0) < 1.0


def test_ipcc_not_in_price_index():
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    # IPCC SSPs have no carbon price rows in CSV
    assert ("IPCC_SSP585", 2050) not in index


# ── EBITDA shock tests ────────────────────────────────────────────────────────

def test_ebitda_shock_zero_at_zero_price():
    holding = make_holding()
    gross, net, shock = _compute_ebitda_shock(holding, carbon_price=0.0)
    assert gross == 0.0
    assert net == 0.0
    assert shock == 0.0


def test_ebitda_shock_increases_with_carbon_price():
    holding = make_holding()
    _, _, shock_low = _compute_ebitda_shock(holding, carbon_price=30.0)
    _, _, shock_high = _compute_ebitda_shock(holding, carbon_price=250.0)
    assert shock_high > shock_low


def test_high_pass_through_reduces_net_cost():
    energy = make_holding(sector=Sector.ENERGY)    # pass-through 0.35
    utility = make_holding(sector=Sector.UTILITIES) # pass-through 0.75
    _, net_energy, _ = _compute_ebitda_shock(energy, 100.0)
    _, net_utility, _ = _compute_ebitda_shock(utility, 100.0)
    # Utility passes more cost to customers → lower net cost absorbed
    assert net_utility < net_energy


def test_pd_shift_zero_at_zero_shock():
    assert _ebitda_shock_to_pd_shift(0.0, baseline_pd=0.002) == 0.0


def test_pd_shift_increases_with_shock():
    pd_low = _ebitda_shock_to_pd_shift(0.10, 0.002)
    pd_high = _ebitda_shock_to_pd_shift(0.60, 0.002)
    assert pd_high > pd_low


def test_pd_shift_never_exceeds_095():
    # Even with 100% EBITDA shock, PD should stay below 0.95
    shift = _ebitda_shock_to_pd_shift(1.0, baseline_pd=0.002)
    assert shift + 0.002 <= 0.95


# ── Scenario-level tests ──────────────────────────────────────────────────────

def test_ipcc_scenario_skipped():
    holding = make_holding()
    scenario = get_scenario("IPCC_SSP585")
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    results = score_holding_transition(holding, scenario, index)
    assert all(r.skipped for r in results)
    assert all(r.transition_pd_shift == 0.0 for r in results)


def test_disorderly_pd_spike_post_2030():
    """
    Delayed Transition: carbon price spikes post-2030.
    PD shift at 2040 should be materially higher than at 2030.
    """
    holding = make_holding()
    scenario = get_scenario("NGFS_DT")
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    results = score_holding_transition(holding, scenario, index)
    by_year = {r.horizon_year: r for r in results}
    assert by_year[2040].transition_pd_shift > by_year[2030].transition_pd_shift * 2


def test_orderly_pd_rises_steadily():
    """Net Zero 2050: smooth price rise → monotonically increasing PD shifts."""
    holding = make_holding()
    scenario = get_scenario("NGFS_NZ2050")
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    results = sorted(
        score_holding_transition(holding, scenario, index),
        key=lambda r: r.horizon_year
    )
    shifts = [r.transition_pd_shift for r in results]
    assert shifts == sorted(shifts), "PD shifts should be non-decreasing for orderly scenario"


def test_low_emissions_intensity_low_shift():
    high_emitter = make_holding(emissions_intensity=1200.0)
    low_emitter = make_holding(emissions_intensity=50.0)
    scenario = get_scenario("NGFS_NZ2050")
    prices = load_carbon_prices(CARBON_CSV)
    index = _build_price_index(prices)
    high_results = score_holding_transition(high_emitter, scenario, index)
    low_results = score_holding_transition(low_emitter, scenario, index)
    high_shift_2050 = next(r for r in high_results if r.horizon_year == 2050).transition_pd_shift
    low_shift_2050 = next(r for r in low_results if r.horizon_year == 2050).transition_pd_shift
    assert high_shift_2050 > low_shift_2050


# ── Portfolio scoring tests ───────────────────────────────────────────────────

def test_portfolio_result_count():
    holdings = [make_holding("H1"), make_holding("H2")]
    scenarios = [get_scenario("NGFS_NZ2050"), get_scenario("NGFS_CP")]
    results = score_portfolio_transition(holdings, scenarios, CARBON_CSV)
    # 2 holdings × 2 scenarios × 6 horizon years = 24
    assert len(results) == 24


def test_all_sectors_have_params():
    for sector in Sector:
        assert sector in SECTOR_TRANSITION_PARAMS, f"Missing params for {sector}"


def test_horizon_years_complete():
    assert HORIZON_YEARS == [2025, 2030, 2035, 2040, 2045, 2050]


def test_compute_all_scenarios_all_horizons():
    """Full compute pass — all scenarios including IPCC."""
    holding = make_holding()
    all_scenarios = list_scenarios()
    results = score_portfolio_transition([holding], all_scenarios, CARBON_CSV)
    # 1 holding × 9 scenarios × 6 years = 54
    assert len(results) == 54