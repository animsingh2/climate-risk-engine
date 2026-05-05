"""
Scenario Registry — NGFS Phase 4, IPCC AR6 SSPs, IEA WEO
Provides a unified interface so the engine is framework-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ScenarioFramework(str, Enum):
    NGFS = "NGFS"
    IPCC = "IPCC"
    IEA = "IEA"


class WarmingPathway(str, Enum):
    """
    Maps each scenario to a physical risk warming level.
    Used to scale hazard severity — framework-agnostic.
    """
    BELOW_1P5C = "Below 1.5°C"    # Best case — strong policy action
    TWO_C      = "~2°C"            # Moderate policy action
    THREE_C    = "~3°C"            # Weak / delayed policy action
    FOUR_PLUS_C = "4°C+"           # No new policy — worst physical risk


@dataclass
class ClimateScenario:
    """
    A single named scenario from any framework.
    This is the object the engine works with — not framework-specific enums.
    """
    framework: ScenarioFramework
    scenario_id: str                    # e.g. "NGFS_NZ2050", "IPCC_SSP126"
    display_name: str                   # e.g. "Net Zero 2050 (NGFS)"
    warming_pathway: WarmingPathway
    description: str

    # Physical risk scalar — applied to raw hazard scores
    # Derived from warming pathway; set by registry, not user
    physical_risk_scalar: float = field(init=False)

    # Transition risk present? IEA/NGFS yes; pure IPCC SSPs no
    has_carbon_price: bool = True

    def __post_init__(self) -> None:
        self.physical_risk_scalar = WARMING_TO_PHYSICAL_SCALAR[self.warming_pathway]


# Physical risk scalar by warming pathway
# Based on IPCC AR6 damage function calibration and NGFS Phase 4 physical risk
# multipliers. Hot House (4°C+) = full hazard exposure; 1.5°C = ~35% of that.
WARMING_TO_PHYSICAL_SCALAR: dict[WarmingPathway, float] = {
    WarmingPathway.BELOW_1P5C:  0.35,
    WarmingPathway.TWO_C:       0.55,
    WarmingPathway.THREE_C:     0.75,
    WarmingPathway.FOUR_PLUS_C: 1.00,
}


# ── Scenario Registry ─────────────────────────────────────────────────────────

SCENARIO_REGISTRY: dict[str, ClimateScenario] = {

    # NGFS Phase 4 (2023)
    "NGFS_NZ2050": ClimateScenario(
        framework=ScenarioFramework.NGFS,
        scenario_id="NGFS_NZ2050",
        display_name="Net Zero 2050 (NGFS)",
        warming_pathway=WarmingPathway.BELOW_1P5C,
        description="Early, orderly transition. Carbon price rises steadily to ~$250/tCO2e by 2050.",
        has_carbon_price=True,
    ),
    "NGFS_DT": ClimateScenario(
        framework=ScenarioFramework.NGFS,
        scenario_id="NGFS_DT",
        display_name="Delayed Transition (NGFS)",
        warming_pathway=WarmingPathway.TWO_C,
        description="Policy action delayed to post-2030, then abrupt carbon price spike.",
        has_carbon_price=True,
    ),
    "NGFS_CP": ClimateScenario(
        framework=ScenarioFramework.NGFS,
        scenario_id="NGFS_CP",
        display_name="Current Policies (NGFS)",
        warming_pathway=WarmingPathway.FOUR_PLUS_C,
        description="No new climate policy. Low carbon price, high physical risk materialises.",
        has_carbon_price=True,
    ),

    # IPCC AR6 SSPs — physical risk focused; no standalone carbon price path
    "IPCC_SSP126": ClimateScenario(
        framework=ScenarioFramework.IPCC,
        scenario_id="IPCC_SSP126",
        display_name="SSP1-2.6 (IPCC AR6)",
        warming_pathway=WarmingPathway.BELOW_1P5C,
        description="Sustainable development pathway. Strong mitigation, ~1.5–2°C by 2100.",
        has_carbon_price=False,
    ),
    "IPCC_SSP245": ClimateScenario(
        framework=ScenarioFramework.IPCC,
        scenario_id="IPCC_SSP245",
        display_name="SSP2-4.5 (IPCC AR6)",
        warming_pathway=WarmingPathway.TWO_C,
        description="Middle-of-the-road. Intermediate emissions, ~2.7°C by 2100.",
        has_carbon_price=False,
    ),
    "IPCC_SSP585": ClimateScenario(
        framework=ScenarioFramework.IPCC,
        scenario_id="IPCC_SSP585",
        display_name="SSP5-8.5 (IPCC AR6)",
        warming_pathway=WarmingPathway.FOUR_PLUS_C,
        description="Fossil-fuelled development. High emissions, ~4.4°C by 2100.",
        has_carbon_price=False,
    ),

    # IEA World Energy Outlook 2023
    "IEA_NZE": ClimateScenario(
        framework=ScenarioFramework.IEA,
        scenario_id="IEA_NZE",
        display_name="Net Zero Emissions 2050 (IEA)",
        warming_pathway=WarmingPathway.BELOW_1P5C,
        description="IEA NZE pathway. Aggressive clean energy deployment, 1.5°C aligned.",
        has_carbon_price=True,
    ),
    "IEA_APS": ClimateScenario(
        framework=ScenarioFramework.IPCC,
        scenario_id="IEA_APS",
        display_name="Announced Pledges Scenario (IEA)",
        warming_pathway=WarmingPathway.TWO_C,
        description="All announced national pledges met in full. ~1.7°C by 2100.",
        has_carbon_price=True,
    ),
    "IEA_STEPS": ClimateScenario(
        framework=ScenarioFramework.IEA,
        scenario_id="IEA_STEPS",
        display_name="Stated Policies Scenario (IEA)",
        warming_pathway=WarmingPathway.THREE_C,
        description="Only existing and stated policies implemented. ~2.5°C by 2100.",
        has_carbon_price=True,
    ),
}


def get_scenario(scenario_id: str) -> ClimateScenario:
    if scenario_id not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. "
            f"Available: {list(SCENARIO_REGISTRY.keys())}"
        )
    return SCENARIO_REGISTRY[scenario_id]


def list_scenarios(framework: Optional[ScenarioFramework] = None) -> list[ClimateScenario]:
    """Return all scenarios, optionally filtered by framework."""
    scenarios = list(SCENARIO_REGISTRY.values())
    if framework:
        scenarios = [s for s in scenarios if s.framework == framework]
    return scenarios