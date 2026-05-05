"""
Module 2 — Physical Hazard Scorer
Assigns hazard profiles and computes physical PD shift per holding per scenario.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from src.schema import (
    Holding, PhysicalHazardProfile, HazardType, Sector
)
from src.scenarios import ClimateScenario


# ── Sector vulnerability factors ──────────────────────────────────────────────
# How exposed each sector is to physical damage / business interruption.
# Scale 0–1. Based on TCFD sector guidance and NGFS Phase 4 sector narratives.

SECTOR_VULNERABILITY: dict[Sector, dict[HazardType, float]] = {
    Sector.ENERGY: {
        HazardType.FLOOD:           0.70,
        HazardType.HEAT_STRESS:     0.50,
        HazardType.WILDFIRE:        0.60,
        HazardType.SEA_LEVEL_RISE:  0.65,
        HazardType.DROUGHT:         0.55,
        HazardType.TROPICAL_CYCLONE:0.75,
    },
    Sector.UTILITIES: {
        HazardType.FLOOD:           0.75,
        HazardType.HEAT_STRESS:     0.60,
        HazardType.WILDFIRE:        0.55,
        HazardType.SEA_LEVEL_RISE:  0.70,
        HazardType.DROUGHT:         0.65,
        HazardType.TROPICAL_CYCLONE:0.70,
    },
    Sector.REAL_ESTATE: {
        HazardType.FLOOD:           0.85,
        HazardType.HEAT_STRESS:     0.45,
        HazardType.WILDFIRE:        0.70,
        HazardType.SEA_LEVEL_RISE:  0.90,
        HazardType.DROUGHT:         0.35,
        HazardType.TROPICAL_CYCLONE:0.80,
    },
    Sector.MATERIALS: {
        HazardType.FLOOD:           0.65,
        HazardType.HEAT_STRESS:     0.55,
        HazardType.WILDFIRE:        0.50,
        HazardType.SEA_LEVEL_RISE:  0.55,
        HazardType.DROUGHT:         0.60,
        HazardType.TROPICAL_CYCLONE:0.65,
    },
    Sector.INDUSTRIALS: {
        HazardType.FLOOD:           0.60,
        HazardType.HEAT_STRESS:     0.50,
        HazardType.WILDFIRE:        0.45,
        HazardType.SEA_LEVEL_RISE:  0.55,
        HazardType.DROUGHT:         0.40,
        HazardType.TROPICAL_CYCLONE:0.60,
    },
    Sector.CONSUMER_STAPLES: {
        HazardType.FLOOD:           0.55,
        HazardType.HEAT_STRESS:     0.65,
        HazardType.WILDFIRE:        0.50,
        HazardType.SEA_LEVEL_RISE:  0.45,
        HazardType.DROUGHT:         0.70,
        HazardType.TROPICAL_CYCLONE:0.55,
    },
    Sector.CONSUMER_DISCRETIONARY: {
        HazardType.FLOOD:           0.50,
        HazardType.HEAT_STRESS:     0.35,
        HazardType.WILDFIRE:        0.40,
        HazardType.SEA_LEVEL_RISE:  0.45,
        HazardType.DROUGHT:         0.30,
        HazardType.TROPICAL_CYCLONE:0.50,
    },
    Sector.FINANCIALS: {
        HazardType.FLOOD:           0.30,
        HazardType.HEAT_STRESS:     0.20,
        HazardType.WILDFIRE:        0.25,
        HazardType.SEA_LEVEL_RISE:  0.35,
        HazardType.DROUGHT:         0.20,
        HazardType.TROPICAL_CYCLONE:0.30,
    },
    Sector.HEALTHCARE: {
        HazardType.FLOOD:           0.45,
        HazardType.HEAT_STRESS:     0.40,
        HazardType.WILDFIRE:        0.40,
        HazardType.SEA_LEVEL_RISE:  0.40,
        HazardType.DROUGHT:         0.30,
        HazardType.TROPICAL_CYCLONE:0.45,
    },
    Sector.TECHNOLOGY: {
        HazardType.FLOOD:           0.40,
        HazardType.HEAT_STRESS:     0.35,
        HazardType.WILDFIRE:        0.35,
        HazardType.SEA_LEVEL_RISE:  0.35,
        HazardType.DROUGHT:         0.25,
        HazardType.TROPICAL_CYCLONE:0.40,
    },
    Sector.COMMUNICATION: {
        HazardType.FLOOD:           0.35,
        HazardType.HEAT_STRESS:     0.25,
        HazardType.WILDFIRE:        0.30,
        HazardType.SEA_LEVEL_RISE:  0.30,
        HazardType.DROUGHT:         0.20,
        HazardType.TROPICAL_CYCLONE:0.35,
    },
    Sector.OTHER: {
        HazardType.FLOOD:           0.45,
        HazardType.HEAT_STRESS:     0.35,
        HazardType.WILDFIRE:        0.40,
        HazardType.SEA_LEVEL_RISE:  0.40,
        HazardType.DROUGHT:         0.35,
        HazardType.TROPICAL_CYCLONE:0.45,
    },
}


# ── Country baseline hazard scores ────────────────────────────────────────────
# Raw hazard exposure by country, per hazard type. Scale 0–1.
# Sources: JRC Global Disaster Risk Index, World Bank Climate Risk indicators,
# INFORM Risk Index 2023. These are portfolio-level proxies — not asset-level.
# Users with precise lat/lon data can override via custom hazard CSV upload.

COUNTRY_HAZARD_SCORES: dict[str, dict[HazardType, float]] = {
    "US": {
        HazardType.FLOOD:            0.65,
        HazardType.HEAT_STRESS:      0.55,
        HazardType.WILDFIRE:         0.60,
        HazardType.SEA_LEVEL_RISE:   0.50,
        HazardType.DROUGHT:          0.50,
        HazardType.TROPICAL_CYCLONE: 0.55,
    },
    "DE": {
        HazardType.FLOOD:            0.50,
        HazardType.HEAT_STRESS:      0.40,
        HazardType.WILDFIRE:         0.25,
        HazardType.SEA_LEVEL_RISE:   0.30,
        HazardType.DROUGHT:          0.35,
        HazardType.TROPICAL_CYCLONE: 0.10,
    },
    "AU": {
        HazardType.FLOOD:            0.60,
        HazardType.HEAT_STRESS:      0.80,
        HazardType.WILDFIRE:         0.85,
        HazardType.SEA_LEVEL_RISE:   0.55,
        HazardType.DROUGHT:          0.75,
        HazardType.TROPICAL_CYCLONE: 0.50,
    },
    "FR": {
        HazardType.FLOOD:            0.50,
        HazardType.HEAT_STRESS:      0.45,
        HazardType.WILDFIRE:         0.40,
        HazardType.SEA_LEVEL_RISE:   0.35,
        HazardType.DROUGHT:          0.40,
        HazardType.TROPICAL_CYCLONE: 0.10,
    },
    "GB": {
        HazardType.FLOOD:            0.55,
        HazardType.HEAT_STRESS:      0.35,
        HazardType.WILDFIRE:         0.20,
        HazardType.SEA_LEVEL_RISE:   0.45,
        HazardType.DROUGHT:          0.30,
        HazardType.TROPICAL_CYCLONE: 0.10,
    },
    "IN": {
        HazardType.FLOOD:            0.80,
        HazardType.HEAT_STRESS:      0.85,
        HazardType.WILDFIRE:         0.45,
        HazardType.SEA_LEVEL_RISE:   0.65,
        HazardType.DROUGHT:          0.70,
        HazardType.TROPICAL_CYCLONE: 0.75,
    },
    "CN": {
        HazardType.FLOOD:            0.75,
        HazardType.HEAT_STRESS:      0.65,
        HazardType.WILDFIRE:         0.40,
        HazardType.SEA_LEVEL_RISE:   0.60,
        HazardType.DROUGHT:          0.55,
        HazardType.TROPICAL_CYCLONE: 0.65,
    },
    "JP": {
        HazardType.FLOOD:            0.70,
        HazardType.HEAT_STRESS:      0.60,
        HazardType.WILDFIRE:         0.30,
        HazardType.SEA_LEVEL_RISE:   0.65,
        HazardType.DROUGHT:          0.35,
        HazardType.TROPICAL_CYCLONE: 0.80,
    },
    "BR": {
        HazardType.FLOOD:            0.75,
        HazardType.HEAT_STRESS:      0.70,
        HazardType.WILDFIRE:         0.70,
        HazardType.SEA_LEVEL_RISE:   0.55,
        HazardType.DROUGHT:          0.65,
        HazardType.TROPICAL_CYCLONE: 0.45,
    },
    "ZA": {
        HazardType.FLOOD:            0.60,
        HazardType.HEAT_STRESS:      0.65,
        HazardType.WILDFIRE:         0.55,
        HazardType.SEA_LEVEL_RISE:   0.45,
        HazardType.DROUGHT:          0.70,
        HazardType.TROPICAL_CYCLONE: 0.30,
    },
}

# Fallback for countries not in the lookup
DEFAULT_HAZARD_SCORES: dict[HazardType, float] = {
    HazardType.FLOOD:            0.50,
    HazardType.HEAT_STRESS:      0.50,
    HazardType.WILDFIRE:         0.40,
    HazardType.SEA_LEVEL_RISE:   0.40,
    HazardType.DROUGHT:          0.45,
    HazardType.TROPICAL_CYCLONE: 0.40,
}


# ── LGD haircut by hazard type (for real assets) ──────────────────────────────
# Physical damage impairs collateral value. These are maximum haircuts
# applied proportionally to hazard_score × scenario_scalar.

LGD_HAIRCUT_BY_HAZARD: dict[HazardType, float] = {
    HazardType.FLOOD:            0.20,
    HazardType.SEA_LEVEL_RISE:   0.25,
    HazardType.WILDFIRE:         0.18,
    HazardType.TROPICAL_CYCLONE: 0.22,
    HazardType.HEAT_STRESS:      0.08,
    HazardType.DROUGHT:          0.10,
}


# ── Scoring output ────────────────────────────────────────────────────────────

@dataclass
class HazardScoreResult:
    holding_id: str
    scenario_id: str
    composite_hazard_score: float       # Weighted across all hazard types
    revenue_shock_pct: float            # % revenue loss from physical hazard
    physical_pd_shift: float            # Additive PD increase
    lgd_adjustment: float               # Additional LGD from collateral impairment
    hazard_profiles: list[PhysicalHazardProfile]


# ── Core scoring functions ────────────────────────────────────────────────────

def _get_country_hazard(country: str, hazard: HazardType) -> float:
    scores = COUNTRY_HAZARD_SCORES.get(country.upper(), DEFAULT_HAZARD_SCORES)
    return scores.get(hazard, DEFAULT_HAZARD_SCORES[hazard])


def _revenue_shock_to_pd_shift(
    revenue_shock_pct: float,
    baseline_pd: float,
    ebitda_margin: float,
) -> float:
    """
    Maps revenue shock → PD shift using a logistic function.

    Logic:
    - Revenue shock erodes EBITDA (scaled by margin)
    - EBITDA erosion maps to credit stress via logistic curve
    - Output is additive PD shift, capped to keep PD < 1

    The logistic steepness (k=8) is calibrated so a 30% revenue
    shock on a BBB-rated company produces ~150–200bps PD shift,
    consistent with Moody's CreditEdge stress scenario outputs.
    """
    if revenue_shock_pct <= 0:
        return 0.0

    # EBITDA erosion — higher margin = more buffer against revenue shock
    ebitda_erosion = revenue_shock_pct / max(ebitda_margin, 0.05)
    ebitda_erosion = min(ebitda_erosion, 1.0)

    # Logistic mapping: 0 erosion → 0 shift; full erosion → ~baseline_pd × 4
    k = 8.0
    midpoint = 0.5
    logistic = 1 / (1 + math.exp(-k * (ebitda_erosion - midpoint)))

    # Scale to a max PD shift of 4× baseline, capped at 0.95
    max_shift = min(baseline_pd * 4.0, 0.95 - baseline_pd)
    pd_shift = logistic * max_shift

    return round(max(pd_shift, 0.0), 6)


def score_holding_physical(
    holding: Holding,
    scenario: ClimateScenario,
) -> HazardScoreResult:
    """
    Compute physical hazard score and PD shift for one holding under one scenario.
    """
    scalar = scenario.physical_risk_scalar
    hazard_profiles: list[PhysicalHazardProfile] = []
    weighted_hazard_sum = 0.0
    lgd_adjustment = 0.0

    for hazard_type in HazardType:
        raw_score = _get_country_hazard(holding.country, hazard_type)
        scaled_score = raw_score * scalar
        vulnerability = SECTOR_VULNERABILITY.get(
            holding.sector, SECTOR_VULNERABILITY[Sector.OTHER]
        )[hazard_type]

        effective_score = scaled_score * vulnerability

        hazard_profiles.append(PhysicalHazardProfile(
            hazard_type=hazard_type,
            hazard_score=round(min(effective_score, 1.0), 4),
            return_period_years=100,
            data_source="Country-Sector Proxy",
        ))

        weighted_hazard_sum += effective_score

        # LGD impairment — only meaningful for real assets with collateral
        from src.schema import AssetType
        if holding.asset_type == AssetType.REAL_ASSET:
            max_haircut = LGD_HAIRCUT_BY_HAZARD.get(hazard_type, 0.05)
            lgd_adjustment += max_haircut * effective_score

    # Composite score: mean across all hazard types
    composite = weighted_hazard_sum / len(list(HazardType))

    # Revenue shock: composite hazard → revenue loss %
    # Capped at 40% — extreme but not company-ending for a single scenario year
    revenue_shock_pct = min(composite * 0.60, 0.40)

    pd_shift = _revenue_shock_to_pd_shift(
        revenue_shock_pct=revenue_shock_pct,
        baseline_pd=holding.baseline_pd,
        ebitda_margin=holding.ebitda_margin,
    )

    lgd_adjustment = min(round(lgd_adjustment, 4), 0.30)  # cap at +30pp

    return HazardScoreResult(
        holding_id=holding.holding_id,
        scenario_id=scenario.scenario_id,
        composite_hazard_score=round(composite, 4),
        revenue_shock_pct=round(revenue_shock_pct, 4),
        physical_pd_shift=pd_shift,
        lgd_adjustment=lgd_adjustment,
        hazard_profiles=hazard_profiles,
    )


def score_portfolio_physical(
    holdings: list[Holding],
    scenarios: list[ClimateScenario],
) -> list[HazardScoreResult]:
    """
    Score all holdings across all scenarios. Returns flat list of results.
    """
    results = []
    for holding in holdings:
        for scenario in scenarios:
            results.append(score_holding_physical(holding, scenario))
    return results