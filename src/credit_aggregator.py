"""
Module 4 — Credit Risk Aggregation & Climate VaR
Combines physical + transition risk into final credit metrics,
computes EL, and derives Climate VaR / ES at portfolio level.

Methods:
  - Parametric VaR/ES: analytical, assumes normal loss distribution
  - Monte Carlo VaR/ES: 10,000 simulations with sector correlations
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from src.schema import Holding, Sector
from src.scenarios import ClimateScenario
from src.hazard_scorer import HazardScoreResult
from src.transition_risk import TransitionRiskResult


# ── Sector correlation matrix ─────────────────────────────────────────────────
# Pairwise climate loss correlations between sectors.
# Based on NGFS Phase 4 sectoral co-movement and TCFD cross-sector
# contagion analysis. Higher = more correlated tail losses.
# Diagonal = 1.0 by definition.

_SECTORS_ORDERED = [
    Sector.ENERGY,
    Sector.UTILITIES,
    Sector.MATERIALS,
    Sector.INDUSTRIALS,
    Sector.CONSUMER_STAPLES,
    Sector.CONSUMER_DISCRETIONARY,
    Sector.FINANCIALS,
    Sector.REAL_ESTATE,
    Sector.HEALTHCARE,
    Sector.TECHNOLOGY,
    Sector.COMMUNICATION,
    Sector.OTHER,
]

# fmt: off
_CORRELATION_MATRIX = np.array([
#  ENE   UTL   MAT   IND   CST   CDI   FIN   RE    HLT   TEC   COM   OTH
  [1.00, 0.70, 0.60, 0.55, 0.30, 0.25, 0.35, 0.40, 0.20, 0.20, 0.20, 0.40],  # Energy
  [0.70, 1.00, 0.55, 0.50, 0.35, 0.25, 0.40, 0.45, 0.20, 0.20, 0.20, 0.40],  # Utilities
  [0.60, 0.55, 1.00, 0.65, 0.35, 0.30, 0.35, 0.40, 0.20, 0.25, 0.20, 0.45],  # Materials
  [0.55, 0.50, 0.65, 1.00, 0.40, 0.35, 0.35, 0.40, 0.25, 0.30, 0.25, 0.45],  # Industrials
  [0.30, 0.35, 0.35, 0.40, 1.00, 0.55, 0.35, 0.35, 0.35, 0.25, 0.25, 0.40],  # Con. Staples
  [0.25, 0.25, 0.30, 0.35, 0.55, 1.00, 0.35, 0.40, 0.30, 0.30, 0.30, 0.35],  # Con. Disc.
  [0.35, 0.40, 0.35, 0.35, 0.35, 0.35, 1.00, 0.55, 0.30, 0.35, 0.35, 0.40],  # Financials
  [0.40, 0.45, 0.40, 0.40, 0.35, 0.40, 0.55, 1.00, 0.25, 0.30, 0.25, 0.40],  # Real Estate
  [0.20, 0.20, 0.20, 0.25, 0.35, 0.30, 0.30, 0.25, 1.00, 0.30, 0.35, 0.30],  # Healthcare
  [0.20, 0.20, 0.25, 0.30, 0.25, 0.30, 0.35, 0.30, 0.30, 1.00, 0.55, 0.30],  # Technology
  [0.20, 0.20, 0.20, 0.25, 0.25, 0.30, 0.35, 0.25, 0.35, 0.55, 1.00, 0.30],  # Communication
  [0.40, 0.40, 0.45, 0.45, 0.40, 0.35, 0.40, 0.40, 0.30, 0.30, 0.30, 1.00],  # Other
], dtype=float)
# fmt: on

_SECTOR_INDEX: dict[Sector, int] = {s: i for i, s in enumerate(_SECTORS_ORDERED)}


def _get_correlation(s1: Sector, s2: Sector) -> float:
    return float(_CORRELATION_MATRIX[_SECTOR_INDEX[s1], _SECTOR_INDEX[s2]])


# ── Per-holding combined result ───────────────────────────────────────────────

@dataclass
class ClimateRiskResult:
    """
    Final combined credit risk output for one holding,
    one scenario, one horizon year.
    """
    holding_id: str
    company_name: str
    sector: Sector
    scenario_id: str
    horizon_year: int

    # Exposure
    ead: float

    # PD
    baseline_pd: float
    transition_pd_shift: float
    physical_pd_shift: float
    climate_adjusted_pd: float          # baseline + transition + physical, capped at 0.95

    # LGD
    lgd_baseline: float
    lgd_transition_loading: float       # stranded asset loading
    lgd_physical_loading: float         # collateral impairment
    lgd_adjusted: float                 # baseline + both loadings, capped at 0.95

    # Loss metrics
    baseline_el: float                  # baseline_pd × lgd_baseline × ead
    climate_el: float                   # climate_adjusted_pd × lgd_adjusted × ead
    incremental_el: float               # climate_el - baseline_el

    # Decomposition
    transition_el_share: float          # fraction of incremental EL from transition
    physical_el_share: float            # fraction of incremental EL from physical

    # Transition detail
    ebitda_shock_pct: float
    carbon_price_usd: float
    transition_skipped: bool

    # Physical detail
    composite_hazard_score: float
    physical_revenue_shock_pct: float


@dataclass
class PortfolioRiskSummary:
    """
    Aggregated portfolio-level risk metrics for one scenario, one horizon year.
    """
    scenario_id: str
    horizon_year: int
    n_holdings: int

    # Portfolio totals
    total_ead: float
    total_baseline_el: float
    total_climate_el: float
    total_incremental_el: float

    # Climate VaR — parametric (normal distribution assumption)
    climate_var_95_parametric: float
    climate_var_99_parametric: float
    climate_es_95_parametric: float     # Expected Shortfall at 95%
    climate_es_99_parametric: float     # Expected Shortfall at 99%

    # Climate VaR — Monte Carlo (empirical from simulated loss distribution)
    climate_var_95_mc: float
    climate_var_99_mc: float
    climate_es_95_mc: float
    climate_es_99_mc: float

    # Monte Carlo metadata
    n_simulations: int
    mc_loss_mean: float
    mc_loss_std: float

    # Top risk contributors (holding_ids sorted by incremental EL descending)
    top_contributors: list[str] = field(default_factory=list)

    # Decomposition
    portfolio_transition_el_share: float = 0.0
    portfolio_physical_el_share: float = 0.0


# ── Combination logic ─────────────────────────────────────────────────────────

def combine_results(
    holding: Holding,
    transition: TransitionRiskResult,
    physical: HazardScoreResult,
    scenario: ClimateScenario,
    horizon_year: int,
) -> ClimateRiskResult:
    """
    Merge transition + physical results into a single ClimateRiskResult.
    """
    # PD combination — additive, capped at 0.95
    climate_pd = min(
        holding.baseline_pd + transition.transition_pd_shift + physical.physical_pd_shift,
        0.95,
    )

    # LGD combination — additive, capped at 0.95
    lgd_adj = min(
        holding.lgd_baseline
        + transition.stranded_asset_lgd_loading
        + physical.lgd_adjustment,
        0.95,
    )

    # Expected losses
    baseline_el = holding.baseline_pd * holding.lgd_baseline * holding.exposure_usd
    climate_el = climate_pd * lgd_adj * holding.exposure_usd
    incremental_el = max(climate_el - baseline_el, 0.0)

    # Decompose incremental EL between transition and physical
    transition_pd_contribution = transition.transition_pd_shift
    physical_pd_contribution = physical.physical_pd_shift
    total_pd_shift = transition_pd_contribution + physical_pd_contribution

    if total_pd_shift > 0:
        transition_share = transition_pd_contribution / total_pd_shift
        physical_share = physical_pd_contribution / total_pd_shift
    else:
        transition_share = 0.0
        physical_share = 0.0

    return ClimateRiskResult(
        holding_id=holding.holding_id,
        company_name=holding.company_name,
        sector=holding.sector,
        scenario_id=scenario.scenario_id,
        horizon_year=horizon_year,
        ead=holding.exposure_usd,
        baseline_pd=holding.baseline_pd,
        transition_pd_shift=transition.transition_pd_shift,
        physical_pd_shift=physical.physical_pd_shift,
        climate_adjusted_pd=round(climate_pd, 6),
        lgd_baseline=holding.lgd_baseline,
        lgd_transition_loading=transition.stranded_asset_lgd_loading,
        lgd_physical_loading=physical.lgd_adjustment,
        lgd_adjusted=round(lgd_adj, 6),
        baseline_el=round(baseline_el, 2),
        climate_el=round(climate_el, 2),
        incremental_el=round(incremental_el, 2),
        transition_el_share=round(transition_share, 4),
        physical_el_share=round(physical_share, 4),
        ebitda_shock_pct=transition.ebitda_shock_pct,
        carbon_price_usd=transition.carbon_price_usd,
        transition_skipped=transition.skipped,
        composite_hazard_score=physical.composite_hazard_score,
        physical_revenue_shock_pct=physical.revenue_shock_pct,
    )


# ── Parametric VaR / ES ───────────────────────────────────────────────────────

# Standard normal quantiles
_Z_95 = 1.6449
_Z_99 = 2.3263


def _parametric_var_es(
    incremental_els: list[float],
) -> tuple[float, float, float, float]:
    """
    Parametric VaR and ES assuming normally distributed portfolio losses.
    Returns (var_95, var_99, es_95, es_99).

    ES (Expected Shortfall) = mean + std × φ(z) / (1 - confidence)
    where φ is the standard normal PDF.
    Standard result from normal distribution CVaR formula.
    """
    if not incremental_els or all(e == 0 for e in incremental_els):
        return 0.0, 0.0, 0.0, 0.0

    arr = np.array(incremental_els, dtype=float)
    mu = float(np.sum(arr))             # Portfolio total EL = sum of holding ELs
    sigma = float(np.sqrt(
        np.sum(arr ** 2)                # Simplified: assumes independence for parametric
    ))

    if sigma == 0:
        return mu, mu, mu, mu

    var_95 = mu + _Z_95 * sigma
    var_99 = mu + _Z_99 * sigma

    # ES = mu + sigma × φ(z) / (1 - α)
    phi_95 = math.exp(-0.5 * _Z_95 ** 2) / math.sqrt(2 * math.pi)
    phi_99 = math.exp(-0.5 * _Z_99 ** 2) / math.sqrt(2 * math.pi)
    es_95 = mu + sigma * phi_95 / 0.05
    es_99 = mu + sigma * phi_99 / 0.01

    return (
        round(max(var_95, 0), 2),
        round(max(var_99, 0), 2),
        round(max(es_95, 0), 2),
        round(max(es_99, 0), 2),
    )


# ── Monte Carlo VaR / ES ──────────────────────────────────────────────────────

def _monte_carlo_var_es(
    holdings: list[Holding],
    incremental_els: list[float],
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> tuple[float, float, float, float, float, float]:
    """
    Monte Carlo VaR and ES using correlated sector shocks.
    Returns (var_95, var_99, es_95, es_99, mc_mean, mc_std).

    Method:
    1. Build sector correlation matrix for the holdings in this portfolio
    2. Draw correlated standard normal shocks (Cholesky decomposition)
    3. Convert shocks to loss multipliers via normal CDF
    4. Scale each holding's incremental EL by its loss multiplier
    5. Sum across portfolio → simulated portfolio loss
    6. Derive VaR and ES from empirical loss distribution
    """
    n = len(holdings)
    if n == 0 or all(e == 0 for e in incremental_els):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(random_seed)

    # Build correlation sub-matrix for this portfolio's sectors
    corr = np.array([
        [_get_correlation(holdings[i].sector, holdings[j].sector) for j in range(n)]
        for i in range(n)
    ], dtype=float)

    # Ensure positive semi-definite (numerical safety)
    corr = (corr + corr.T) / 2
    min_eig = np.linalg.eigvalsh(corr).min()
    if min_eig < 0:
        corr += (-min_eig + 1e-8) * np.eye(n)

    # Cholesky decomposition for correlated draws
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(n)  # fallback to independence if decomposition fails

    # Simulate correlated standard normal shocks: shape (n_simulations, n_holdings)
    z_independent = rng.standard_normal((n_simulations, n))
    z_correlated = z_independent @ L.T

    # Convert to loss multipliers: shocked_el = el × (1 + shock_scale × z)
    # shock_scale controls how much individual ELs vary around their mean.
    # Set to 0.40 — a holding can range from ~0× to ~2× its expected EL.
    # Clip multipliers at 0 (no negative losses) and 3 (no infinite blowup).
    els = np.array(incremental_els, dtype=float)
    shock_scale = 0.40
    multipliers = np.clip(1.0 + shock_scale * z_correlated, 0.0, 3.0)

    # Portfolio loss per simulation: sum of shocked holding ELs
    simulated_losses = (multipliers * els[np.newaxis, :]).sum(axis=1)

    # Empirical VaR and ES
    var_95 = float(np.percentile(simulated_losses, 95))
    var_99 = float(np.percentile(simulated_losses, 99))

    tail_95 = simulated_losses[simulated_losses >= var_95]
    tail_99 = simulated_losses[simulated_losses >= var_99]
    es_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95
    es_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

    return (
        round(var_95, 2),
        round(var_99, 2),
        round(es_95, 2),
        round(es_99, 2),
        round(float(np.mean(simulated_losses)), 2),
        round(float(np.std(simulated_losses)), 2),
    )


# ── Portfolio aggregation ─────────────────────────────────────────────────────

def aggregate_portfolio(
    holdings: list[Holding],
    results: list[ClimateRiskResult],
    scenario_id: str,
    horizon_year: int,
    n_simulations: int = 10_000,
) -> PortfolioRiskSummary:
    """
    Aggregate holding-level results into portfolio summary with VaR/ES.
    """
    subset = [
        r for r in results
        if r.scenario_id == scenario_id and r.horizon_year == horizon_year
    ]
    if not subset:
        raise ValueError(
            f"No results found for scenario={scenario_id}, year={horizon_year}"
        )

    # Map holding_id → Holding for MC correlation
    holding_map = {h.holding_id: h for h in holdings}

    total_ead = sum(r.ead for r in subset)
    total_baseline_el = sum(r.baseline_el for r in subset)
    total_climate_el = sum(r.climate_el for r in subset)
    total_incremental_el = sum(r.incremental_el for r in subset)

    incremental_els = [r.incremental_el for r in subset]
    holdings_ordered = [holding_map[r.holding_id] for r in subset]

    # Parametric
    var95p, var99p, es95p, es99p = _parametric_var_es(incremental_els)

    # Monte Carlo
    var95m, var99m, es95m, es99m, mc_mean, mc_std = _monte_carlo_var_es(
        holdings_ordered, incremental_els, n_simulations
    )

    # Top contributors by incremental EL
    sorted_results = sorted(subset, key=lambda r: r.incremental_el, reverse=True)
    top_contributors = [r.holding_id for r in sorted_results[:5]]

    # Portfolio-level decomposition
    if total_incremental_el > 0:
        trans_share = sum(
            r.incremental_el * r.transition_el_share for r in subset
        ) / total_incremental_el
        phys_share = sum(
            r.incremental_el * r.physical_el_share for r in subset
        ) / total_incremental_el
    else:
        trans_share = phys_share = 0.0

    return PortfolioRiskSummary(
        scenario_id=scenario_id,
        horizon_year=horizon_year,
        n_holdings=len(subset),
        total_ead=round(total_ead, 2),
        total_baseline_el=round(total_baseline_el, 2),
        total_climate_el=round(total_climate_el, 2),
        total_incremental_el=round(total_incremental_el, 2),
        climate_var_95_parametric=var95p,
        climate_var_99_parametric=var99p,
        climate_es_95_parametric=es95p,
        climate_es_99_parametric=es99p,
        climate_var_95_mc=var95m,
        climate_var_99_mc=var99m,
        climate_es_95_mc=es95m,
        climate_es_99_mc=es99m,
        n_simulations=n_simulations,
        mc_loss_mean=mc_mean,
        mc_loss_std=mc_std,
        top_contributors=top_contributors,
        portfolio_transition_el_share=round(trans_share, 4),
        portfolio_physical_el_share=round(phys_share, 4),
    )


# ── Full engine run ───────────────────────────────────────────────────────────

def run_full_engine(
    holdings: list[Holding],
    scenarios: list[ClimateScenario],
    transition_results: list[TransitionRiskResult],
    physical_results: list[HazardScoreResult],
    horizon_years: list[int],
    n_simulations: int = 10_000,
) -> tuple[list[ClimateRiskResult], list[PortfolioRiskSummary]]:
    """
    Master function — combines all results and produces full output.

    Returns:
        holding_results: one ClimateRiskResult per holding/scenario/year
        portfolio_summaries: one PortfolioRiskSummary per scenario/year
    """
    # Index inputs for fast lookup
    transition_index: dict[tuple[str, str, int], TransitionRiskResult] = {
        (r.holding_id, r.scenario_id, r.horizon_year): r
        for r in transition_results
    }
    physical_index: dict[tuple[str, str], HazardScoreResult] = {
        (r.holding_id, r.scenario_id): r
        for r in physical_results
    }

    holding_results: list[ClimateRiskResult] = []

    for holding in holdings:
        for scenario in scenarios:
            for year in horizon_years:
                t_key = (holding.holding_id, scenario.scenario_id, year)
                p_key = (holding.holding_id, scenario.scenario_id)

                transition = transition_index.get(t_key)
                physical = physical_index.get(p_key)

                if transition is None or physical is None:
                    continue

                holding_results.append(
                    combine_results(holding, transition, physical, scenario, year)
                )

    # Aggregate portfolio summaries
    portfolio_summaries: list[PortfolioRiskSummary] = []
    for scenario in scenarios:
        for year in horizon_years:
            try:
                summary = aggregate_portfolio(
                    holdings, holding_results,
                    scenario.scenario_id, year, n_simulations
                )
                portfolio_summaries.append(summary)
            except ValueError:
                pass  # No results for this combo (e.g. IPCC physical-only)

    return holding_results, portfolio_summaries