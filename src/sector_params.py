"""
Sector-level parameters for transition risk engine.
These are the primary calibration inputs — tune with real data when available.

Sources:
- Carbon pass-through rates: MSCI Climate Value-at-Risk methodology (2023),
  IEA Net Zero by 2050 sector narratives
- Abatement capacity: IPCC AR6 WGIII Chapter 6 mitigation potentials
- Revenue sensitivity: NGFS Phase 4 sector impact assessments
"""

from __future__ import annotations
from dataclasses import dataclass
from src.schema import Sector


@dataclass(frozen=True)
class SectorTransitionParams:
    """
    Calibration parameters for one sector's transition risk exposure.

    carbon_pass_through_rate:
        Fraction of carbon cost a company can pass to customers (0-1).
        High = less margin compression. E.g. regulated utilities ~0.75.

    abatement_capacity:
        Fraction of emissions a company can reduce without revenue impact
        through operational changes (efficiency, fuel switching) by 2030.
        Reduces effective carbon cost exposure.

    revenue_sensitivity:
        Multiplier on EBITDA shock → revenue impact. Sectors with
        inelastic demand (utilities) have lower revenue sensitivity
        than discretionary sectors.

    stranded_asset_risk:
        Additional LGD loading for sectors with high fixed-asset
        carbon exposure (coal, oil & gas infrastructure). Scale 0-1.
    """
    carbon_pass_through_rate: float     # 0.0 – 1.0
    abatement_capacity: float           # 0.0 – 1.0
    revenue_sensitivity: float          # 0.5 – 2.0
    stranded_asset_risk: float          # 0.0 – 1.0


SECTOR_TRANSITION_PARAMS: dict[Sector, SectorTransitionParams] = {

    Sector.ENERGY: SectorTransitionParams(
        carbon_pass_through_rate=0.35,
        abatement_capacity=0.20,
        revenue_sensitivity=1.60,
        stranded_asset_risk=0.70,
    ),
    Sector.UTILITIES: SectorTransitionParams(
        carbon_pass_through_rate=0.75,
        abatement_capacity=0.35,
        revenue_sensitivity=0.70,
        stranded_asset_risk=0.45,
    ),
    Sector.MATERIALS: SectorTransitionParams(
        carbon_pass_through_rate=0.45,
        abatement_capacity=0.25,
        revenue_sensitivity=1.20,
        stranded_asset_risk=0.40,
    ),
    Sector.INDUSTRIALS: SectorTransitionParams(
        carbon_pass_through_rate=0.50,
        abatement_capacity=0.30,
        revenue_sensitivity=1.10,
        stranded_asset_risk=0.30,
    ),
    Sector.CONSUMER_STAPLES: SectorTransitionParams(
        carbon_pass_through_rate=0.60,
        abatement_capacity=0.25,
        revenue_sensitivity=0.80,
        stranded_asset_risk=0.10,
    ),
    Sector.CONSUMER_DISCRETIONARY: SectorTransitionParams(
        carbon_pass_through_rate=0.55,
        abatement_capacity=0.20,
        revenue_sensitivity=1.30,
        stranded_asset_risk=0.10,
    ),
    Sector.FINANCIALS: SectorTransitionParams(
        carbon_pass_through_rate=0.80,
        abatement_capacity=0.15,
        revenue_sensitivity=0.50,
        stranded_asset_risk=0.05,
    ),
    Sector.REAL_ESTATE: SectorTransitionParams(
        carbon_pass_through_rate=0.55,
        abatement_capacity=0.30,
        revenue_sensitivity=0.90,
        stranded_asset_risk=0.25,
    ),
    Sector.HEALTHCARE: SectorTransitionParams(
        carbon_pass_through_rate=0.70,
        abatement_capacity=0.20,
        revenue_sensitivity=0.60,
        stranded_asset_risk=0.05,
    ),
    Sector.TECHNOLOGY: SectorTransitionParams(
        carbon_pass_through_rate=0.75,
        abatement_capacity=0.35,
        revenue_sensitivity=0.55,
        stranded_asset_risk=0.05,
    ),
    Sector.COMMUNICATION: SectorTransitionParams(
        carbon_pass_through_rate=0.70,
        abatement_capacity=0.30,
        revenue_sensitivity=0.60,
        stranded_asset_risk=0.05,
    ),
    Sector.OTHER: SectorTransitionParams(
        carbon_pass_through_rate=0.55,
        abatement_capacity=0.25,
        revenue_sensitivity=1.00,
        stranded_asset_risk=0.15,
    ),
}


# Horizon years — all computed, user selects subset in dashboard
HORIZON_YEARS: list[int] = [2025, 2030, 2035, 2040, 2045, 2050]