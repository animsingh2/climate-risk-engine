"""
Module 1 — Data Schema & Domain Objects
Climate Risk Engine
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class Sector(str, Enum):
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    MATERIALS = "Materials"
    INDUSTRIALS = "Industrials"
    CONSUMER_STAPLES = "Consumer Staples"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    FINANCIALS = "Financials"
    REAL_ESTATE = "Real Estate"
    HEALTHCARE = "Healthcare"
    TECHNOLOGY = "Technology"
    COMMUNICATION = "Communication"
    OTHER = "Other"


class CreditRating(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    DEFAULT = "D"


class NGFSScenario(str, Enum):
    ORDERLY = "Net Zero 2050"            # Early, smooth policy action
    DISORDERLY = "Delayed Transition"    # Late, abrupt policy spike post-2030
    HOT_HOUSE = "Current Policies"       # No new climate policy; high physical risk


class HazardType(str, Enum):
    FLOOD = "Flood"
    HEAT_STRESS = "Heat Stress"
    WILDFIRE = "Wildfire"
    SEA_LEVEL_RISE = "Sea Level Rise"
    DROUGHT = "Drought"
    TROPICAL_CYCLONE = "Tropical Cyclone"


class AssetType(str, Enum):
    CORPORATE = "Corporate"
    REAL_ASSET = "Real Asset"           # Project finance, commercial RE
    SOVEREIGN = "Sovereign"


# ── Baseline PD by credit rating ─────────────────────────────────────────────
# Illustrative values consistent with Basel/S&P long-run averages

BASELINE_PD_MAP: dict[CreditRating, float] = {
    CreditRating.AAA:     0.0001,   # 0.01%
    CreditRating.AA:      0.0003,   # 0.03%
    CreditRating.A:       0.0008,   # 0.08%
    CreditRating.BBB:     0.0020,   # 0.20%
    CreditRating.BB:      0.0100,   # 1.00%
    CreditRating.B:       0.0400,   # 4.00%
    CreditRating.CCC:     0.1500,   # 15.00%
    CreditRating.DEFAULT: 1.0000,   # 100%
}


# ── Core domain objects ───────────────────────────────────────────────────────

@dataclass
class PhysicalHazardProfile:
    """
    Physical hazard exposure for an asset location.
    Hazard score is 0 (no exposure) to 1 (maximum exposure).
    """
    hazard_type: HazardType
    hazard_score: float             # 0.0 – 1.0
    return_period_years: int = 100  # e.g. 1-in-100-year event
    data_source: str = "Proxy"      # e.g. "JRC", "FEMA", "Proxy"

    def __post_init__(self) -> None:
        if not 0.0 <= self.hazard_score <= 1.0:
            raise ValueError(
                f"hazard_score must be between 0 and 1, got {self.hazard_score}"
            )


@dataclass
class Holding:
    """
    A single portfolio holding — corporate or real asset.
    This is the primary input object that feeds both
    the transition risk engine and the physical hazard scorer.
    """
    # Identification
    holding_id: str
    company_name: str
    asset_type: AssetType
    sector: Sector
    country: str                        # ISO 3166-1 alpha-2, e.g. "US", "DE"

    # Financial exposure
    exposure_usd: float                 # EAD — outstanding loan or investment ($)
    credit_rating: CreditRating
    lgd_baseline: float                 # Loss Given Default, e.g. 0.45 = 45%

    # Fundamentals for transition risk
    ebitda_margin: float                # e.g. 0.20 = 20% EBITDA margin
    emissions_intensity: float          # tCO2e per $M revenue
    annual_revenue_usd_m: float         # Revenue in $M USD

    # Physical risk location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = None        # Fallback if lat/lon not available

    # Physical hazard profiles (populated by hazard scorer)
    hazard_profiles: list[PhysicalHazardProfile] = field(default_factory=list)

    # Derived — set by engine, not by user
    baseline_pd: float = field(init=False)

    def __post_init__(self) -> None:
        self.baseline_pd = BASELINE_PD_MAP[self.credit_rating]
        if not 0.0 <= self.lgd_baseline <= 1.0:
            raise ValueError(f"lgd_baseline must be between 0 and 1")
        if not 0.0 <= self.ebitda_margin <= 1.0:
            raise ValueError(f"ebitda_margin must be between 0 and 1")
        if self.emissions_intensity < 0:
            raise ValueError(f"emissions_intensity cannot be negative")


@dataclass
class ClimateRiskResult:
    """
    Output of the climate risk engine for a single holding
    under a single NGFS scenario at a given year horizon.
    """
    holding_id: str
    scenario: NGFSScenario
    horizon_year: int

    # PD outputs
    baseline_pd: float
    transition_pd_shift: float          # Additive shift from carbon cost
    physical_pd_shift: float            # Additive shift from hazard exposure
    climate_adjusted_pd: float          # = baseline + transition shift + physical shift

    # Loss metrics
    lgd_adjusted: float                 # LGD after collateral impairment
    ead: float                          # Unchanged from financial exposure
    expected_loss: float                # = climate_adjusted_pd × lgd_adjusted × ead

    # Decomposition
    transition_ebitda_shock: float      # % EBITDA reduction from carbon cost
    physical_revenue_shock: float       # % revenue reduction from hazard

    # VaR contribution (populated by portfolio aggregator)
    climate_var_contribution_95: float = 0.0
    climate_var_contribution_99: float = 0.0


@dataclass
class NGFSCarbonPrice:
    """
    Carbon price path under a given NGFS scenario.
    Price is in USD per tCO2e.
    """
    scenario: NGFSScenario
    year: int
    carbon_price_usd_per_tco2e: float


@dataclass
class Portfolio:
    """
    Collection of holdings. Entry point for the full engine run.
    """
    portfolio_id: str
    holdings: list[Holding] = field(default_factory=list)

    def add_holding(self, holding: Holding) -> None:
        self.holdings.append(holding)

    def get_holding(self, holding_id: str) -> Optional[Holding]:
        for h in self.holdings:
            if h.holding_id == holding_id:
                return h
        return None

    @property
    def total_exposure_usd(self) -> float:
        return sum(h.exposure_usd for h in self.holdings)

    def __len__(self) -> int:
        return len(self.holdings)