"""
Module 2 smoke tests — run with: pytest tests/test_module2.py -v
"""
import pytest
from src.schema import Holding, AssetType, Sector, CreditRating, HazardType
from src.scenarios import (
    get_scenario, list_scenarios, ScenarioFramework,
    WarmingPathway, WARMING_TO_PHYSICAL_SCALAR
)
from src.hazard_scorer import (
    score_holding_physical, score_portfolio_physical,
    _revenue_shock_to_pd_shift, SECTOR_VULNERABILITY
)


def make_holding(
    holding_id="T001",
    sector=Sector.ENERGY,
    country="US",
    credit_rating=CreditRating.BBB,
    asset_type=AssetType.CORPORATE,
    ebitda_margin=0.20,
) -> Holding:
    return Holding(
        holding_id=holding_id,
        company_name="Test Co",
        asset_type=asset_type,
        sector=sector,
        country=country,
        exposure_usd=10_000_000,
        credit_rating=credit_rating,
        lgd_baseline=0.45,
        ebitda_margin=ebitda_margin,
        emissions_intensity=500.0,
        annual_revenue_usd_m=100.0,
    )


# ── Scenario registry tests ───────────────────────────────────────────────────

def test_get_known_scenario():
    s = get_scenario("NGFS_NZ2050")
    assert s.display_name == "Net Zero 2050 (NGFS)"
    assert s.has_carbon_price is True


def test_get_unknown_scenario_raises():
    with pytest.raises(KeyError):
        get_scenario("NONEXISTENT")


def test_list_scenarios_all():
    all_s = list_scenarios()
    assert len(all_s) >= 9


def test_list_scenarios_by_framework():
    ngfs = list_scenarios(ScenarioFramework.NGFS)
    assert all(s.framework == ScenarioFramework.NGFS for s in ngfs)
    assert len(ngfs) == 3


def test_physical_scalar_ordering():
    # Hot House must have higher scalar than orderly
    hot_house = get_scenario("NGFS_CP").physical_risk_scalar
    orderly = get_scenario("NGFS_NZ2050").physical_risk_scalar
    assert hot_house > orderly


def test_ipcc_no_carbon_price():
    s = get_scenario("IPCC_SSP585")
    assert s.has_carbon_price is False


# ── Hazard scorer tests ───────────────────────────────────────────────────────

def test_score_holding_basic():
    holding = make_holding()
    scenario = get_scenario("NGFS_CP")
    result = score_holding_physical(holding, scenario)
    assert result.holding_id == "T001"
    assert 0 < result.composite_hazard_score <= 1.0
    assert result.physical_pd_shift >= 0


def test_hot_house_worse_than_orderly():
    holding = make_holding()
    hot = score_holding_physical(holding, get_scenario("NGFS_CP"))
    orderly = score_holding_physical(holding, get_scenario("NGFS_NZ2050"))
    assert hot.physical_pd_shift > orderly.physical_pd_shift
    assert hot.composite_hazard_score > orderly.composite_hazard_score


def test_real_asset_has_lgd_adjustment():
    real_asset = make_holding(asset_type=AssetType.REAL_ASSET, sector=Sector.REAL_ESTATE)
    corporate = make_holding(asset_type=AssetType.CORPORATE, sector=Sector.REAL_ESTATE)
    scenario = get_scenario("NGFS_CP")
    ra_result = score_holding_physical(real_asset, scenario)
    corp_result = score_holding_physical(corporate, scenario)
    assert ra_result.lgd_adjustment > 0
    assert corp_result.lgd_adjustment == 0


def test_high_ebitda_margin_buffers_pd_shift():
    low_margin = make_holding(ebitda_margin=0.05)
    high_margin = make_holding(ebitda_margin=0.40)
    scenario = get_scenario("NGFS_CP")
    low = score_holding_physical(low_margin, scenario)
    high = score_holding_physical(high_margin, scenario)
    assert low.physical_pd_shift > high.physical_pd_shift


def test_portfolio_scoring_dimensions():
    holdings = [make_holding("H1"), make_holding("H2"), make_holding("H3")]
    scenarios = [get_scenario("NGFS_NZ2050"), get_scenario("NGFS_CP")]
    results = score_portfolio_physical(holdings, scenarios)
    # 3 holdings × 2 scenarios = 6 results
    assert len(results) == 6


def test_all_hazard_types_scored():
    holding = make_holding()
    scenario = get_scenario("NGFS_DT")
    result = score_holding_physical(holding, scenario)
    scored_types = {p.hazard_type for p in result.hazard_profiles}
    assert scored_types == set(HazardType)


def test_sector_vulnerability_all_sectors_covered():
    for sector in Sector:
        assert sector in SECTOR_VULNERABILITY, f"Missing vulnerability for {sector}"