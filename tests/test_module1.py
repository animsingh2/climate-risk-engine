"""
Module 1 smoke tests — run with: pytest tests/test_module1.py -v
"""
import pytest
from pathlib import Path
from src.schema import Holding, Portfolio, AssetType, Sector, CreditRating, BASELINE_PD_MAP
from src.data_loader import load_portfolio, load_carbon_prices

PORTFOLIO_CSV = Path("data/inputs/portfolio_template.csv")
NGFS_CSV = Path("data/ngfs_scenarios/carbon_prices.csv")


def test_baseline_pd_map_complete():
    for rating in CreditRating:
        assert rating in BASELINE_PD_MAP, f"Missing PD for {rating}"


def test_holding_baseline_pd_assigned():
    h = Holding(
        holding_id="T001", company_name="Test Co",
        asset_type=AssetType.CORPORATE, sector=Sector.ENERGY,
        country="US", exposure_usd=1_000_000,
        credit_rating=CreditRating.BBB, lgd_baseline=0.45,
        ebitda_margin=0.20, emissions_intensity=500.0,
        annual_revenue_usd_m=100.0,
    )
    assert h.baseline_pd == 0.0020


def test_load_portfolio():
    portfolio = load_portfolio(PORTFOLIO_CSV)
    assert len(portfolio) == 5
    assert portfolio.holdings[0].holding_id == "H001"
    assert portfolio.total_exposure_usd == 200_000_000


def test_load_ngfs():
    prices = load_carbon_prices(NGFS_CSV)
    # CSV now has 3 NGFS + 3 IEA scenarios × 6 years = 36 rows
    assert len(prices) == 36
    orderly_2050 = [p for p in prices if p.year == 2050 and p.scenario_id == "NGFS_NZ2050"]
    assert orderly_2050[0].carbon_price_usd_per_tco2e == 250.0


def test_invalid_hazard_score():
    from src.schema import PhysicalHazardProfile, HazardType
    with pytest.raises(ValueError):
        PhysicalHazardProfile(hazard_type=HazardType.FLOOD, hazard_score=1.5)