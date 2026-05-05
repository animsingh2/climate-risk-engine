"""
Module 5 — Portfolio Dashboard
Streamlit app. Run with: streamlit run run_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pandas as pd
import numpy as np
import csv
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

from src.schema import Holding, Portfolio, AssetType, Sector, CreditRating
from src.scenarios import list_scenarios, get_scenario, ScenarioFramework, ClimateScenario
from src.hazard_scorer import score_portfolio_physical
from src.transition_risk import score_portfolio_transition
from src.credit_aggregator import run_full_engine, ClimateRiskResult, PortfolioRiskSummary
from src.sector_params import HORIZON_YEARS
from src.data_loader import load_portfolio, load_carbon_prices

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Climate Risk Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
        letter-spacing: -0.02em;
        color: #e2e8f0 !important;
    }

    .metric-card {
        background: #12161f;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }

    .metric-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.72rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 4px;
    }

    .metric-delta-pos { color: #EE6677; font-size: 0.78rem; }
    .metric-delta-neg { color: #009988; font-size: 0.78rem; }

    .risk-badge-high {
        background: rgba(238,102,119,0.12);
        color: #EE6677;
        border: 1px solid rgba(238,102,119,0.25);
        border-radius: 3px;
        padding: 2px 7px;
        font-size: 0.68rem;
        font-family: 'DM Mono', monospace;
    }
    .risk-badge-med {
        background: rgba(204,187,68,0.12);
        color: #CCBB44;
        border: 1px solid rgba(204,187,68,0.25);
        border-radius: 3px;
        padding: 2px 7px;
        font-size: 0.68rem;
        font-family: 'DM Mono', monospace;
    }
    .risk-badge-low {
        background: rgba(0,153,136,0.12);
        color: #009988;
        border: 1px solid rgba(0,153,136,0.25);
        border-radius: 3px;
        padding: 2px 7px;
        font-size: 0.68rem;
        font-family: 'DM Mono', monospace;
    }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.05rem;
        color: #94a3b8 !important;
        border-bottom: 1px solid #1e2530;
        padding-bottom: 7px;
        margin-bottom: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        color: #6b7280;
    }

    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important;
    }

    .upload-hint {
        font-size: 0.72rem;
        color: #4b5563;
        font-family: 'DM Mono', monospace;
    }

    div[data-testid="stSidebarContent"] {
        background-color: #0a0d13;
        border-right: 1px solid #1a2030;
    }

    div[data-testid="stSidebarContent"] label,
    div[data-testid="stSidebarContent"] .stMarkdown p,
    div[data-testid="stSidebarContent"] .stMarkdown h3 {
        color: #94a3b8 !important;
    }

    .stSelectbox label, .stMultiSelect label,
    .stSlider label, .stRadio label {
        color: #94a3b8 !important;
    }

    .stMarkdown p { color: #94a3b8; }

    /* Subtler info/warning boxes */
    .stAlert { border-radius: 6px; }
    
    .stMarkdown p { color: #94a3b8; }

    /* Subtler info/warning boxes */
    .stAlert { border-radius: 6px; }

    /* Prevent primary color bleeding onto bold text in markdown */
    .stMarkdown strong, .stMarkdown b,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
    title_font=dict(family="DM Serif Display, serif", color="#e2e8f0", size=15),
    xaxis=dict(
        gridcolor="#1e2530",
        linecolor="#2a3340",
        tickcolor="#4b5563",
        tickfont=dict(color="#6b7280", size=11),
        title_font=dict(color="#94a3b8", size=12),
    ),
    yaxis=dict(
        gridcolor="#1e2530",
        linecolor="#2a3340",
        tickcolor="#4b5563",
        tickfont=dict(color="#6b7280", size=11),
        title_font=dict(color="#94a3b8", size=12),
    ),
    legend=dict(
        bgcolor="rgba(14,17,23,0.7)",
        bordercolor="#2a3340",
        borderwidth=1,
        font=dict(color="#94a3b8", size=11),
    ),
    margin=dict(t=48, b=36, l=40, r=20),
)

# Okabe-Ito colorblind-safe palette — muted, accessible, distinct
# Reference: https://jfly.uni-koeln.de/color/
OKI = {
    "blue":        "#4477AA",
    "cyan":        "#66CCEE",
    "green":       "#228833",
    "yellow":      "#CCBB44",
    "red":         "#EE6677",
    "purple":      "#AA3377",
    "grey":        "#BBBBBB",
    "orange":      "#EE7733",
    "teal":        "#009988",
}

SCENARIO_COLORS = {
    "NGFS_NZ2050": OKI["teal"],
    "NGFS_DT":     OKI["yellow"],
    "NGFS_CP":     OKI["red"],
    "IPCC_SSP126": OKI["cyan"],
    "IPCC_SSP245": OKI["orange"],
    "IPCC_SSP585": OKI["purple"],
    "IEA_NZE":     OKI["green"],
    "IEA_APS":     OKI["blue"],
    "IEA_STEPS":   OKI["grey"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_usd(val: float, decimals: int = 0) -> str:
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.{decimals}f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.{decimals}f}K"
    return f"${val:.{decimals}f}"

def fmt_usd_plain(val: float) -> str:
    """USD formatter without $ sign — avoids LaTeX rendering in st.markdown."""
    if abs(val) >= 1_000_000:
        return f"USD {val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"USD {val/1_000:.0f}K"
    return f"USD {val:.0f}"

def fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{val*100:.{decimals}f}%"

def fmt_bps(val: float) -> str:
    return f"{val*10000:.1f} bps"

def risk_badge(pd: float) -> str:
    if pd > 0.05:
        return '<span class="risk-badge-high">HIGH</span>'
    if pd > 0.01:
        return '<span class="risk-badge-med">MEDIUM</span>'
    return '<span class="risk-badge-low">LOW</span>'

def metric_card(label: str, value: str, delta: str = "", delta_positive_is_bad: bool = True) -> str:
    delta_class = ""
    delta_html = ""
    if delta:
        is_increase = delta.startswith("+") or (not delta.startswith("-") and delta != "—")
        bad = is_increase if delta_positive_is_bad else not is_increase
        delta_class = "metric-delta-pos" if bad else "metric-delta-neg"
        delta_html = f'<div class="{delta_class}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# ── Engine runner (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running climate risk engine...")
def run_engine(
    portfolio_csv_bytes: bytes,
    carbon_csv_bytes: bytes,
    selected_scenario_ids: tuple[str, ...],
) -> tuple[list, list, list]:
    """
    Returns (holding_results, summaries, holdings_data).
    holdings_data is a list of dicts for serialisation — Portfolio object
    is not cacheable by Streamlit.
    """
    import io as _io
    from src.transition_risk import _build_price_index, score_holding_transition

    portfolio = load_portfolio(_io.StringIO(portfolio_csv_bytes.decode("utf-8")))
    carbon_prices = load_carbon_prices(_io.StringIO(carbon_csv_bytes.decode("utf-8")))
    price_index = _build_price_index(carbon_prices)
    scenarios = [get_scenario(sid) for sid in selected_scenario_ids]

    t_results = []
    for holding in portfolio.holdings:
        for scenario in scenarios:
            t_results.extend(
                score_holding_transition(holding, scenario, price_index, HORIZON_YEARS)
            )

    p_results = score_portfolio_physical(portfolio.holdings, scenarios)
    holding_results, summaries = run_full_engine(
        portfolio.holdings, scenarios, t_results, p_results, HORIZON_YEARS
    )

    # Serialise holdings as plain dicts so Streamlit can cache them
    holdings_data = [{
        "holding_id": h.holding_id,
        "company_name": h.company_name,
        "sector": h.sector.value,
        "country": h.country,
        "asset_type": h.asset_type.value,
        "exposure_usd": h.exposure_usd,
        "credit_rating": h.credit_rating.value,
        "lgd_baseline": h.lgd_baseline,
        "ebitda_margin": h.ebitda_margin,
        "emissions_intensity": h.emissions_intensity,
        "annual_revenue_usd_m": h.annual_revenue_usd_m,
        "baseline_pd": h.baseline_pd,
    } for h in portfolio.holdings]

    return holding_results, summaries, holdings_data


def load_default_files() -> tuple[bytes, bytes]:
    portfolio_path = Path("data/inputs/portfolio_template.csv")
    carbon_path = Path("data/ngfs_scenarios/carbon_prices.csv")
    return portfolio_path.read_bytes(), carbon_path.read_bytes()

def validate_portfolio_csv(csv_bytes: bytes) -> list[str]:
    """
    Returns a list of error messages. Empty list = valid.
    """
    required_columns = {
        "holding_id", "company_name", "asset_type", "sector",
        "country", "exposure_usd", "credit_rating", "lgd_baseline",
        "ebitda_margin", "emissions_intensity", "annual_revenue_usd_m",
    }
    valid_sectors = {s.value for s in Sector}
    valid_ratings = {r.value for r in CreditRating}
    valid_asset_types = {a.value for a in AssetType}
    errors = []

    try:
        import io as _io
        reader = csv.DictReader(_io.StringIO(csv_bytes.decode("utf-8")))
        headers = set(reader.fieldnames or [])
        missing = required_columns - headers
        if missing:
            errors.append(f"Missing columns: {', '.join(sorted(missing))}")
            return errors  # Can't validate rows without columns

        for i, row in enumerate(reader, start=2):
            hid = row.get("holding_id", f"row {i}")

            # Numeric fields
            for field in ["exposure_usd", "lgd_baseline", "ebitda_margin",
                          "emissions_intensity", "annual_revenue_usd_m"]:
                try:
                    float(row[field])
                except ValueError:
                    errors.append(f"[{hid}] '{field}' must be a number, got: '{row[field]}'")

            # LGD and margin range
            try:
                if not 0 <= float(row["lgd_baseline"]) <= 1:
                    errors.append(f"[{hid}] 'lgd_baseline' must be between 0 and 1")
                if not 0 <= float(row["ebitda_margin"]) <= 1:
                    errors.append(f"[{hid}] 'ebitda_margin' must be between 0 and 1")
            except ValueError:
                pass  # Already caught above

            # Enum fields
            if row["sector"].strip() not in valid_sectors:
                errors.append(f"[{hid}] Invalid sector '{row['sector']}'. Valid: {', '.join(sorted(valid_sectors))}")
            if row["credit_rating"].strip().upper() not in valid_ratings:
                errors.append(f"[{hid}] Invalid rating '{row['credit_rating']}'. Valid: {', '.join(sorted(valid_ratings))}")
            if row["asset_type"].strip() not in valid_asset_types:
                errors.append(f"[{hid}] Invalid asset_type '{row['asset_type']}'. Valid: {', '.join(sorted(valid_asset_types))}")

    except Exception as e:
        errors.append(f"Could not parse CSV: {e}")

    return errors


def validate_carbon_csv(csv_bytes: bytes) -> list[str]:
    """Returns a list of error messages. Empty list = valid."""
    required_columns = {"scenario_id", "year", "carbon_price_usd_per_tco2e"}
    from src.scenarios import SCENARIO_REGISTRY
    errors = []
    try:
        import io as _io
        reader = csv.DictReader(_io.StringIO(csv_bytes.decode("utf-8")))
        headers = set(reader.fieldnames or [])
        missing = required_columns - headers
        if missing:
            errors.append(f"Missing columns: {', '.join(sorted(missing))}")
            return errors
        for i, row in enumerate(reader, start=2):
            sid = row.get("scenario_id", "").strip()
            if sid not in SCENARIO_REGISTRY:
                errors.append(
                    f"Row {i}: Unknown scenario_id '{sid}'. "
                    f"Must match a registered scenario (e.g. NGFS_NZ2050)."
                )
            try:
                int(row["year"])
            except ValueError:
                errors.append(f"Row {i}: 'year' must be an integer, got '{row['year']}'")
            try:
                float(row["carbon_price_usd_per_tco2e"])
            except ValueError:
                errors.append(f"Row {i}: 'carbon_price_usd_per_tco2e' must be a number")
    except Exception as e:
        errors.append(f"Could not parse CSV: {e}")
    return errors

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Climate Risk Engine")
    st.markdown('<p class="upload-hint">NGFS Phase 4 · TCFD Aligned · Basel Credit Risk</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### Data Inputs")

    # Template download
    template_path = Path("data/inputs/portfolio_template.csv")
    if template_path.exists():
        st.download_button(
            label="Download portfolio template",
            data=template_path.read_bytes(),
            file_name="portfolio_template.csv",
            mime="text/csv",
            use_container_width=False,
        )

    portfolio_file = st.file_uploader(
        "Portfolio CSV",
        type="csv",
        help="Upload your portfolio. Must match the template schema.",
    )
    carbon_file = st.file_uploader(
        "Carbon Price CSV",
        type="csv",
        help="Optional: upload custom carbon price paths.",
    )
    if portfolio_file is None:
        st.info("Using sample portfolio. Upload your own CSV above.")

    st.divider()

    st.markdown("### Scenarios")
    all_scenarios = list_scenarios()

    framework_filter = st.multiselect(
        "Framework",
        options=[f.value for f in ScenarioFramework],
        default=[f.value for f in ScenarioFramework],
    )

    filtered_scenarios = [
        s for s in all_scenarios if s.framework.value in framework_filter
    ]

    selected_scenario_ids = st.multiselect(
        "Select scenarios",
        options=[s.scenario_id for s in filtered_scenarios],
        default=[
            s.scenario_id for s in filtered_scenarios
            if s.scenario_id in ("NGFS_NZ2050", "NGFS_DT", "NGFS_CP")
        ],
        format_func=lambda sid: get_scenario(sid).display_name,
    )

    st.divider()

    st.markdown("### Horizon Year")
    horizon_year = st.select_slider(
        "Analysis horizon",
        options=HORIZON_YEARS,
        value=2030,
    )

    st.divider()

    st.markdown("### Risk Metrics")
    var_method = st.radio(
        "VaR / ES method",
        options=["Monte Carlo", "Parametric"],
        index=0,
    )

    confidence_level = st.radio(
        "Confidence level",
        options=["95%", "99%"],
        index=1,
    )
    conf_key = "99" if confidence_level == "99%" else "95"

# ── Load data & run engine ────────────────────────────────────────────────────

portfolio_bytes, carbon_bytes = load_default_files()

if portfolio_file is not None:
    uploaded_portfolio_bytes = portfolio_file.getvalue()
    errors = validate_portfolio_csv(uploaded_portfolio_bytes)
    if errors:
        st.error("**Portfolio CSV has errors. Please fix and re-upload:**")
        for e in errors:
            st.markdown(f"- {e}")
        st.stop()
    portfolio_bytes = uploaded_portfolio_bytes

if carbon_file is not None:
    uploaded_carbon_bytes = carbon_file.getvalue()
    errors = validate_carbon_csv(uploaded_carbon_bytes)
    if errors:
        st.error("**Carbon price CSV has errors. Please fix and re-upload:**")
        for e in errors:
            st.markdown(f"- {e}")
        st.stop()
    carbon_bytes = uploaded_carbon_bytes

if not selected_scenario_ids:
    st.warning("Select at least one scenario to continue.")
    st.stop()

try:
    holding_results, summaries, holdings_data = run_engine(
        portfolio_bytes,
        carbon_bytes,
        tuple(sorted(selected_scenario_ids)),
    )
except Exception as e:
    st.error(f"Engine error: {e}")
    st.stop()

# Rebuild portfolio object from cached holdings_data (not cacheable directly)
portfolio = Portfolio(portfolio_id="active")
for hd in holdings_data:
    portfolio.add_holding(Holding(
        holding_id=hd["holding_id"],
        company_name=hd["company_name"],
        asset_type=AssetType(hd["asset_type"]),
        sector=Sector(hd["sector"]),
        country=hd["country"],
        exposure_usd=hd["exposure_usd"],
        credit_rating=CreditRating(hd["credit_rating"]),
        lgd_baseline=hd["lgd_baseline"],
        ebitda_margin=hd["ebitda_margin"],
        emissions_intensity=hd["emissions_intensity"],
        annual_revenue_usd_m=hd["annual_revenue_usd_m"],
    ))

# Build lookup dataframes
results_df = pd.DataFrame([vars(r) for r in holding_results])
summaries_df = pd.DataFrame([vars(s) for s in summaries])

if results_df.empty:
    st.warning("No results generated. Check that selected scenarios have data.")
    st.stop()

# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("# Climate Financial Risk Dashboard")
st.markdown(
    f"**{len(portfolio.holdings)} holdings** · "
    f"**{len(selected_scenario_ids)} scenarios** · "
    f"Total exposure: **{fmt_usd(portfolio.total_exposure_usd)}**"
)
st.divider()

# ── Results download ──────────────────────────────────────────────────────────
results_csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download full results CSV",
    data=results_csv,
    file_name="climate_risk_results.csv",
    mime="text/csv",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Overview",
    "Scenario Analysis",
    "Holding Drilldown",
    "Scenario Comparison",
    "Plain Summary",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="section-header">Portfolio Composition</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Sector breakdown — treemap
    with col1:
        holdings_df = pd.DataFrame([{
            "company": h.company_name,
            "sector": h.sector.value,
            "country": h.country,
            "exposure": h.exposure_usd,
            "rating": h.credit_rating.value,
            "asset_type": h.asset_type.value,
            "ebitda_margin": h.ebitda_margin,
            "emissions_intensity": h.emissions_intensity,
            "baseline_pd": h.baseline_pd,
        } for h in portfolio.holdings])

        fig_sector = px.treemap(
            holdings_df,
            path=["sector", "company"],
            values="exposure",
            color="exposure",
            color_continuous_scale=["#1a2744", "#2563eb", "#60a5fa"],
            title="Exposure by Sector",
        )
        fig_sector.update_layout(**PLOTLY_LAYOUT)
        fig_sector.update_traces(textfont_family="DM Sans")
        st.plotly_chart(fig_sector, width="stretch")

    # Rating distribution
    with col2:
        rating_data = holdings_df.groupby("rating")["exposure"].sum().reset_index()
        rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
        rating_data["rating"] = pd.Categorical(rating_data["rating"], categories=rating_order, ordered=True)
        rating_data = rating_data.sort_values("rating")

        fig_rating = go.Figure(go.Bar(
            x=rating_data["rating"],
            y=rating_data["exposure"] / 1_000_000,
            marker_color=[OKI["teal"], OKI["cyan"], OKI["green"], OKI["yellow"], OKI["orange"], OKI["red"], OKI["purple"], "#4a2040"],
            text=[fmt_usd(v * 1_000_000) for v in rating_data["exposure"] / 1_000_000],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10),
        ))
        fig_rating.update_layout(
            **PLOTLY_LAYOUT,
            title="Exposure by Credit Rating ($M)",
            yaxis_title="Exposure ($M)",
            showlegend=False,
        )
        st.plotly_chart(fig_rating, width="stretch")

    # Holdings table
    st.markdown('<p class="section-header">Holdings</p>', unsafe_allow_html=True)

    display_df = holdings_df.copy()
    display_df["exposure"] = display_df["exposure"].apply(fmt_usd)
    display_df["baseline_pd"] = display_df["baseline_pd"].apply(fmt_bps)
    display_df["ebitda_margin"] = display_df["ebitda_margin"].apply(fmt_pct)
    display_df["emissions_intensity"] = display_df["emissions_intensity"].apply(lambda x: f"{x:.0f} tCO₂e/$M")
    display_df.columns = [
        "Company", "Sector", "Country", "Exposure", "Rating",
        "Asset Type", "EBITDA Margin", "Emissions Intensity", "Baseline PD"
    ]
    st.dataframe(display_df, width="stretch", hide_index=True)

    # Emissions intensity heatmap
    st.markdown('<p class="section-header">Emissions Intensity by Sector</p>', unsafe_allow_html=True)
    col3, col4 = st.columns([2, 1])
    with col3:
        fig_emiss = px.bar(
            holdings_df.sort_values("emissions_intensity", ascending=True),
            x="emissions_intensity",
            y="company",
            orientation="h",
            color="sector",
            title="Emissions Intensity (tCO₂e per $M Revenue)",
            labels={"emissions_intensity": "tCO₂e / $M Revenue", "company": ""},
        )
        fig_emiss.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        st.plotly_chart(fig_emiss, width="stretch")

    with col4:
        st.markdown("**Portfolio Stats**")
        total_exp = portfolio.total_exposure_usd
        avg_pd = np.mean([h.baseline_pd for h in portfolio.holdings])
        avg_emiss = np.mean([h.emissions_intensity for h in portfolio.holdings])
        weighted_emiss = sum(
            h.emissions_intensity * h.exposure_usd for h in portfolio.holdings
        ) / total_exp

        st.markdown(metric_card("Total Exposure", fmt_usd(total_exp)), unsafe_allow_html=True)
        st.markdown(metric_card("Holdings", str(len(portfolio.holdings))), unsafe_allow_html=True)
        st.markdown(metric_card("Avg Baseline PD", fmt_bps(avg_pd)), unsafe_allow_html=True)
        st.markdown(metric_card("Wtd Avg Emissions", f"{weighted_emiss:.0f} tCO₂e/$M"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    scenario_focus = st.selectbox(
        "Scenario",
        options=selected_scenario_ids,
        format_func=lambda sid: get_scenario(sid).display_name,
    )

    # Filter results
    tab2_results = results_df[
        (results_df["scenario_id"] == scenario_focus) &
        (results_df["horizon_year"] == horizon_year)
    ]
    tab2_summary = summaries_df[
        (summaries_df["scenario_id"] == scenario_focus) &
        (summaries_df["horizon_year"] == horizon_year)
    ]

    if tab2_results.empty or tab2_summary.empty:
        st.warning("No results for this scenario/horizon combination.")
    else:
        s = tab2_summary.iloc[0]
        scenario_obj = get_scenario(scenario_focus)

        # Top-line metrics
        st.markdown('<p class="section-header">Portfolio Risk Metrics</p>', unsafe_allow_html=True)

        if var_method == "Monte Carlo":
            var_val = s[f"climate_var_{conf_key}_mc"]
            es_val = s[f"climate_es_{conf_key}_mc"]
        else:
            var_val = s[f"climate_var_{conf_key}_parametric"]
            es_val = s[f"climate_es_{conf_key}_parametric"]

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card(
                "Total Climate EL",
                fmt_usd(s["total_climate_el"]),
                f"+{fmt_usd(s['total_incremental_el'])} vs baseline",
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card(
                f"Climate VaR ({confidence_level})",
                fmt_usd(var_val),
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card(
                f"Expected Shortfall ({confidence_level})",
                fmt_usd(es_val),
            ), unsafe_allow_html=True)
        with c4:
            trans_pct = s["portfolio_transition_el_share"] * 100
            st.markdown(metric_card(
                "Transition Risk Share",
                f"{trans_pct:.1f}%",
            ), unsafe_allow_html=True)
        with c5:
            phys_pct = s["portfolio_physical_el_share"] * 100
            st.markdown(metric_card(
                "Physical Risk Share",
                f"{phys_pct:.1f}%",
            ), unsafe_allow_html=True)

        st.divider()

        col_a, col_b = st.columns(2)

        # PD shift waterfall
        with col_a:
            st.markdown('<p class="section-header">PD Shift by Holding</p>', unsafe_allow_html=True)
            fig_pd = go.Figure()
            fig_pd.add_trace(go.Bar(
                name="Baseline PD",
                x=tab2_results["holding_id"],
                y=tab2_results["baseline_pd"] * 10000,
                marker_color="#2a3340",
            ))
            fig_pd.add_trace(go.Bar(
                name="Transition Shift",
                x=tab2_results["holding_id"],
                y=tab2_results["transition_pd_shift"] * 10000,
                marker_color=OKI["yellow"],
            ))
            fig_pd.add_trace(go.Bar(
                name="Physical Shift",
                x=tab2_results["holding_id"],
                y=tab2_results["physical_pd_shift"] * 10000,
                marker_color=OKI["red"],
            ))
            fig_pd.update_layout(
                **PLOTLY_LAYOUT,
                barmode="stack",
                title="Climate-Adjusted PD (basis points)",
                yaxis_title="PD (bps)",
            )
            st.plotly_chart(fig_pd, width="stretch")

        # EL decomposition
        with col_b:
            st.markdown('<p class="section-header">Expected Loss Decomposition</p>', unsafe_allow_html=True)
            fig_el = go.Figure()
            fig_el.add_trace(go.Bar(
                name="Baseline EL",
                x=tab2_results["holding_id"],
                y=tab2_results["baseline_el"] / 1000,
                marker_color="#2a3340",
            ))
            fig_el.add_trace(go.Bar(
                name="Climate Increment",
                x=tab2_results["holding_id"],
                y=tab2_results["incremental_el"] / 1000,
                marker_color=SCENARIO_COLORS.get(scenario_focus, "#60a5fa"),
            ))
            fig_el.update_layout(
                **PLOTLY_LAYOUT,
                barmode="stack",
                title="Expected Loss by Holding ($K)",
                yaxis_title="EL ($K)",
            )
            st.plotly_chart(fig_el, width="stretch")

        # PD evolution over time
        st.markdown('<p class="section-header">PD Evolution Across Horizon Years</p>', unsafe_allow_html=True)

        time_results = results_df[results_df["scenario_id"] == scenario_focus]
        fig_time = go.Figure()
        for _, row in results_df[
            (results_df["scenario_id"] == scenario_focus)
        ].groupby("holding_id"):
            pass

        for hid in tab2_results["holding_id"].unique():
            hdata = time_results[time_results["holding_id"] == hid].sort_values("horizon_year")
            fig_time.add_trace(go.Scatter(
                x=hdata["horizon_year"],
                y=hdata["climate_adjusted_pd"] * 10000,
                mode="lines+markers",
                name=hid,
                line=dict(width=2),
                marker=dict(size=6),
            ))

        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            title="Climate-Adjusted PD Over Time (bps)",
            xaxis_title="Year",
            yaxis_title="Adjusted PD (bps)",
        )
        st.plotly_chart(fig_time, width="stretch")

        # Monte Carlo loss distribution
        if var_method == "Monte Carlo":
            st.markdown('<p class="section-header">Simulated Loss Distribution</p>', unsafe_allow_html=True)

            mc_mean = float(s["mc_loss_mean"])
            mc_std = float(s["mc_loss_std"])

            rng = np.random.default_rng(42)
            sim_losses = rng.normal(mc_mean, mc_std, 10_000)
            sim_losses = np.clip(sim_losses, 0, None)

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=sim_losses / 1000,
                nbinsx=80,
                marker_color=OKI["blue"],
                opacity=0.7,
                name="Simulated Losses",
            ))
            fig_mc.add_vline(
                x=var_val / 1000,
                line_dash="dash",
                line_color=OKI["yellow"],
                annotation_text=f"VaR {confidence_level}",
                annotation_font_color=OKI["yellow"],
            )
            fig_mc.add_vline(
                x=es_val / 1000,
                line_dash="dot",
                line_color=OKI["red"],
                annotation_text=f"ES {confidence_level}",
                annotation_font_color=OKI["red"],
            )
            fig_mc.update_layout(
                **PLOTLY_LAYOUT,
                title="Monte Carlo Portfolio Loss Distribution ($K)",
                xaxis_title="Portfolio Loss ($K)",
                yaxis_title="Frequency",
                showlegend=False,
            )
            st.plotly_chart(fig_mc, width="stretch")

        # Detailed results table
        st.markdown('<p class="section-header">Holding-Level Results</p>', unsafe_allow_html=True)
        detail_cols = [
            "holding_id", "baseline_pd", "transition_pd_shift",
            "physical_pd_shift", "climate_adjusted_pd",
            "lgd_baseline", "lgd_adjusted", "baseline_el",
            "climate_el", "incremental_el",
            "ebitda_shock_pct", "composite_hazard_score",
        ]
        detail = tab2_results[detail_cols].copy()
        for col in ["baseline_pd", "transition_pd_shift", "physical_pd_shift", "climate_adjusted_pd"]:
            detail[col] = detail[col].apply(fmt_bps)
        for col in ["lgd_baseline", "lgd_adjusted", "ebitda_shock_pct"]:
            detail[col] = detail[col].apply(fmt_pct)
        for col in ["baseline_el", "climate_el", "incremental_el"]:
            detail[col] = detail[col].apply(fmt_usd)
        detail["composite_hazard_score"] = detail["composite_hazard_score"].apply(lambda x: f"{x:.3f}")
        detail.columns = [
            "Holding", "Baseline PD", "Transition Shift", "Physical Shift",
            "Climate PD", "LGD Baseline", "LGD Adjusted",
            "Baseline EL", "Climate EL", "Incremental EL",
            "EBITDA Shock", "Hazard Score",
        ]
        st.dataframe(detail, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOLDING DRILLDOWN
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    selected_holding_id = st.selectbox(
        "Select holding",
        options=[h.holding_id for h in portfolio.holdings],
        format_func=lambda hid: next(
            h.company_name for h in portfolio.holdings if h.holding_id == hid
        ),
    )

    holding_obj = portfolio.get_holding(selected_holding_id)
    holding_data = results_df[results_df["holding_id"] == selected_holding_id]

    if holding_obj is None or holding_data.empty:
        st.warning("No data for selected holding.")
    else:
        st.markdown(f"## {holding_obj.company_name}")
        st.markdown(
            f"`{holding_obj.sector.value}` · "
            f"`{holding_obj.country}` · "
            f"`{holding_obj.asset_type.value}` · "
            f"Rating: `{holding_obj.credit_rating.value}` · "
            f"Exposure: `{fmt_usd(holding_obj.exposure_usd)}`"
        )

        # Company fundamentals
        st.markdown('<p class="section-header">Fundamentals</p>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown(metric_card("Baseline PD", fmt_bps(holding_obj.baseline_pd)), unsafe_allow_html=True)
        with f2:
            st.markdown(metric_card("EBITDA Margin", fmt_pct(holding_obj.ebitda_margin)), unsafe_allow_html=True)
        with f3:
            st.markdown(metric_card("Emissions Intensity", f"{holding_obj.emissions_intensity:.0f} tCO₂e/$M"), unsafe_allow_html=True)
        with f4:
            st.markdown(metric_card("LGD Baseline", fmt_pct(holding_obj.lgd_baseline)), unsafe_allow_html=True)

        st.divider()

        # PD across all scenarios and years
        st.markdown('<p class="section-header">Climate-Adjusted PD — All Scenarios × All Years</p>', unsafe_allow_html=True)

        fig_drill = go.Figure()
        for sid in selected_scenario_ids:
            sdata = holding_data[
                holding_data["scenario_id"] == sid
            ].sort_values("horizon_year")
            if sdata.empty:
                continue
            scenario_name = get_scenario(sid).display_name
            fig_drill.add_trace(go.Scatter(
                x=sdata["horizon_year"],
                y=sdata["climate_adjusted_pd"] * 10000,
                mode="lines+markers",
                name=scenario_name,
                line=dict(color=SCENARIO_COLORS.get(sid, "#60a5fa"), width=2.5),
                marker=dict(size=7),
            ))

        # Baseline reference line
        fig_drill.add_hline(
            y=holding_obj.baseline_pd * 10000,
            line_dash="dash",
            line_color="#3a4555",
            annotation_text="Baseline PD",
            annotation_font_color="#6b7280",
        )
        fig_drill.update_layout(
            **PLOTLY_LAYOUT,
            title="Climate-Adjusted PD by Scenario (bps)",
            xaxis_title="Year",
            yaxis_title="PD (bps)",
        )
        st.plotly_chart(fig_drill, width="stretch")

        # Transition vs Physical split
        col_x, col_y = st.columns(2)

        with col_x:
            st.markdown('<p class="section-header">Risk Driver Split at Selected Horizon</p>', unsafe_allow_html=True)
            horizon_data = holding_data[holding_data["horizon_year"] == horizon_year]
            if not horizon_data.empty:
                fig_split = go.Figure()
                for sid in selected_scenario_ids:
                    row = horizon_data[horizon_data["scenario_id"] == sid]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    fig_split.add_trace(go.Bar(
                        name=get_scenario(sid).display_name,
                        x=["Transition PD Shift", "Physical PD Shift"],
                        y=[r["transition_pd_shift"] * 10000, r["physical_pd_shift"] * 10000],
                        marker_color=SCENARIO_COLORS.get(sid, "#60a5fa"),
                    ))
                fig_split.update_layout(
                    **PLOTLY_LAYOUT,
                    barmode="group",
                    title=f"PD Shift Decomposition — {horizon_year}",
                    yaxis_title="PD Shift (bps)",
                )
                st.plotly_chart(fig_split, width="stretch")

        with col_y:
            st.markdown('<p class="section-header">EBITDA Shock by Scenario</p>', unsafe_allow_html=True)
            horizon_data2 = holding_data[holding_data["horizon_year"] == horizon_year]
            if not horizon_data2.empty:
                fig_ebitda = go.Figure()
                for sid in selected_scenario_ids:
                    row = horizon_data2[horizon_data2["scenario_id"] == sid]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    fig_ebitda.add_trace(go.Bar(
                        name=get_scenario(sid).display_name,
                        x=[get_scenario(sid).display_name],
                        y=[r["ebitda_shock_pct"] * 100],
                        marker_color=SCENARIO_COLORS.get(sid, "#60a5fa"),
                        text=f"{r['ebitda_shock_pct']*100:.1f}%",
                        textposition="outside",
                    ))
                fig_ebitda.update_layout(
                    **PLOTLY_LAYOUT,
                    title=f"EBITDA Shock (%) — {horizon_year}",
                    yaxis_title="EBITDA Shock (%)",
                    showlegend=False,
                )
                st.plotly_chart(fig_ebitda, width="stretch")

        # Incremental EL over time per scenario
        st.markdown('<p class="section-header">Incremental Expected Loss Over Time</p>', unsafe_allow_html=True)
        fig_iel = go.Figure()
        for sid in selected_scenario_ids:
            sdata = holding_data[holding_data["scenario_id"] == sid].sort_values("horizon_year")
            if sdata.empty:
                continue
            hex_color = SCENARIO_COLORS.get(sid, "#2563eb")
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            fig_iel.add_trace(go.Scatter(
                x=sdata["horizon_year"],
                y=sdata["incremental_el"] / 1000,
                mode="lines+markers",
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b},0.13)",
                line=dict(color=hex_color, width=2),
                name=get_scenario(sid).display_name,
                marker=dict(size=6),
            ))
        fig_iel.update_layout(
            **PLOTLY_LAYOUT,
            title="Incremental Expected Loss ($K)",
            xaxis_title="Year",
            yaxis_title="Incremental EL ($K)",
        )
        st.plotly_chart(fig_iel, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<p class="section-header">Side-by-Side Scenario Comparison</p>', unsafe_allow_html=True)

    compare_year = st.select_slider(
        "Comparison horizon year",
        options=HORIZON_YEARS,
        value=horizon_year,
        key="compare_year",
    )

    compare_summaries = summaries_df[summaries_df["horizon_year"] == compare_year]

    if compare_summaries.empty:
        st.warning("No summary data for selected year.")
    else:
        # Summary table
        summary_rows = []
        for sid in selected_scenario_ids:
            row = compare_summaries[compare_summaries["scenario_id"] == sid]
            if row.empty:
                continue
            r = row.iloc[0]
            scenario_obj = get_scenario(sid)

            if var_method == "Monte Carlo":
                var_val = r[f"climate_var_{conf_key}_mc"]
                es_val = r[f"climate_es_{conf_key}_mc"]
            else:
                var_val = r[f"climate_var_{conf_key}_parametric"]
                es_val = r[f"climate_es_{conf_key}_parametric"]

            summary_rows.append({
                "Scenario": scenario_obj.display_name,
                "Framework": scenario_obj.framework.value,
                "Warming Pathway": scenario_obj.warming_pathway.value,
                "Total Climate EL": fmt_usd(r["total_climate_el"]),
                "Incremental EL": fmt_usd(r["total_incremental_el"]),
                f"VaR ({confidence_level})": fmt_usd(var_val),
                f"ES ({confidence_level})": fmt_usd(es_val),
                "Transition Share": fmt_pct(r["portfolio_transition_el_share"]),
                "Physical Share": fmt_pct(r["portfolio_physical_el_share"]),
            })

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        st.divider()

        col_p, col_q = st.columns(2)

        # Total climate EL by scenario
        with col_p:
            fig_comp_el = go.Figure()
            for sid in selected_scenario_ids:
                row = compare_summaries[compare_summaries["scenario_id"] == sid]
                if row.empty:
                    continue
                r = row.iloc[0]
                fig_comp_el.add_trace(go.Bar(
                    name=get_scenario(sid).display_name,
                    x=[get_scenario(sid).display_name],
                    y=[r["total_incremental_el"] / 1000],
                    marker_color=SCENARIO_COLORS.get(sid, "#60a5fa"),
                    text=fmt_usd(r["total_incremental_el"]),
                    textposition="outside",
                    textfont=dict(family="DM Mono", size=10),
                ))
            fig_comp_el.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Incremental EL by Scenario — {compare_year} ($K)",
                yaxis_title="Incremental EL ($K)",
                showlegend=False,
            )
            st.plotly_chart(fig_comp_el, width="stretch")

        # VaR and ES comparison
        with col_q:
            fig_var_comp = go.Figure()
            sids = [
                s for s in selected_scenario_ids
                if not compare_summaries[compare_summaries["scenario_id"] == s].empty
            ]
            names = [get_scenario(s).display_name for s in sids]

            if var_method == "Monte Carlo":
                var_vals = [
                    compare_summaries[compare_summaries["scenario_id"] == s].iloc[0][f"climate_var_{conf_key}_mc"] / 1000
                    for s in sids
                ]
                es_vals = [
                    compare_summaries[compare_summaries["scenario_id"] == s].iloc[0][f"climate_es_{conf_key}_mc"] / 1000
                    for s in sids
                ]
            else:
                var_vals = [
                    compare_summaries[compare_summaries["scenario_id"] == s].iloc[0][f"climate_var_{conf_key}_parametric"] / 1000
                    for s in sids
                ]
                es_vals = [
                    compare_summaries[compare_summaries["scenario_id"] == s].iloc[0][f"climate_es_{conf_key}_parametric"] / 1000
                    for s in sids
                ]

            fig_var_comp.add_trace(go.Bar(
                name=f"VaR {confidence_level}",
                x=names,
                y=var_vals,
                marker_color=OKI["blue"],
            ))
            fig_var_comp.add_trace(go.Bar(
                name=f"ES {confidence_level}",
                x=names,
                y=es_vals,
                marker_color=OKI["purple"],
            ))
            fig_var_comp.update_layout(
                **PLOTLY_LAYOUT,
                barmode="group",
                title=f"Climate VaR vs ES by Scenario — {compare_year} ($K)",
                yaxis_title="Loss ($K)",
            )
            st.plotly_chart(fig_var_comp, width="stretch")

        # EL evolution over time — all scenarios on one chart
        st.markdown('<p class="section-header">Portfolio Incremental EL — All Scenarios Over Time</p>', unsafe_allow_html=True)
        fig_timeline = go.Figure()
        for sid in selected_scenario_ids:
            sdata = summaries_df[summaries_df["scenario_id"] == sid].sort_values("horizon_year")
            if sdata.empty:
                continue
            fig_timeline.add_trace(go.Scatter(
                x=sdata["horizon_year"],
                y=sdata["total_incremental_el"] / 1_000_000,
                mode="lines+markers",
                name=get_scenario(sid).display_name,
                line=dict(color=SCENARIO_COLORS.get(sid, "#60a5fa"), width=2.5),
                marker=dict(size=7),
            ))
        fig_timeline.update_layout(
            **PLOTLY_LAYOUT,
            title="Portfolio Incremental EL Over Time ($M)",
            xaxis_title="Year",
            yaxis_title="Incremental EL ($M)",
        )
        st.plotly_chart(fig_timeline, width="stretch")

        # Heatmap: holding × scenario incremental EL
        st.markdown('<p class="section-header">Risk Heatmap — Holding × Scenario</p>', unsafe_allow_html=True)
        pivot = results_df[results_df["horizon_year"] == compare_year].pivot_table(
            index="holding_id",
            columns="scenario_id",
            values="incremental_el",
            aggfunc="sum",
        )
        if not pivot.empty:
            pivot.columns = [get_scenario(c).display_name for c in pivot.columns]
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values / 1000,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale=[[0, "#0e1117"], [0.5, OKI["blue"]], [1.0, OKI["red"]]],
                text=[[fmt_usd(v * 1000) for v in row] for row in pivot.values / 1000],
                texttemplate="%{text}",
                textfont=dict(family="DM Mono", size=10),
                colorbar=dict(title="Incr. EL ($K)"),
            ))
            fig_heat.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Incremental EL Heatmap ($K) — {compare_year}",
                xaxis_title="Scenario",
                yaxis_title="Holding",
            )
            st.plotly_chart(fig_heat, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PLAIN SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### Plain Summary")
    st.markdown(
        "A simple explanation of the portfolio's climate risk — "
        "no finance or climate knowledge needed."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        summary_scenario = st.selectbox(
            "Scenario for summary",
            options=selected_scenario_ids,
            format_func=lambda sid: get_scenario(sid).display_name,
            key="summary_scenario",
        )
    with col_cfg2:
        summary_year = st.select_slider(
            "Horizon year for summary",
            options=HORIZON_YEARS,
            value=horizon_year,
            key="summary_year",
        )

    tab5_results = results_df[
        (results_df["scenario_id"] == summary_scenario) &
        (results_df["horizon_year"] == summary_year)
    ]
    tab5_summary = summaries_df[
        (summaries_df["scenario_id"] == summary_scenario) &
        (summaries_df["horizon_year"] == summary_year)
    ]

    if tab5_results.empty or tab5_summary.empty:
        st.warning("No data for this scenario and year combination.")
    else:
        s5 = tab5_summary.iloc[0]
        scenario_obj5 = get_scenario(summary_scenario)

        var_5 = float(s5["climate_var_99_mc"])
        es_5 = float(s5["climate_es_99_mc"])
        incr_el = float(s5["total_incremental_el"])
        total_ead = float(s5["total_ead"])
        trans_share = float(s5["portfolio_transition_el_share"])
        phys_share = float(s5["portfolio_physical_el_share"])

        top2 = tab5_results.nlargest(2, "incremental_el")[
            ["holding_id", "company_name", "sector",
             "incremental_el", "climate_adjusted_pd",
             "ebitda_shock_pct", "composite_hazard_score"]
        ].to_dict(orient="records")

        generate = st.button("Generate plain summary", type="primary")

        if generate:
            scenario_desc = scenario_obj5.description
            warming = scenario_obj5.warming_pathway.value

            if trans_share >= 0.6:
                driver = "policy and carbon pricing changes"
                driver_detail = (
                    "As governments raise the price on carbon emissions, "
                    "companies that produce a lot of pollution face higher costs. "
                    "This squeezes their profits and makes it harder to repay debts."
                )
            elif phys_share >= 0.6:
                driver = "physical climate events such as floods, heatwaves, and wildfires"
                driver_detail = (
                    "As the climate changes, extreme weather events become more frequent "
                    "and severe. This can damage buildings and equipment, disrupt operations, "
                    "and reduce revenues."
                )
            else:
                driver = "a combination of policy changes and physical climate events"
                driver_detail = (
                    "Both rising carbon costs and more frequent extreme weather events "
                    "are contributing roughly equally to the risk in this portfolio."
                )

            if incr_el / max(total_ead, 1) < 0.005:
                severity = "relatively modest"
            elif incr_el / max(total_ead, 1) < 0.02:
                severity = "moderate"
            else:
                severity = "material"

            holding_lines = []
            for h in top2:
                hazard_plain = (
                    "low"
                    if float(h["composite_hazard_score"]) < 0.2
                    else "moderate"
                    if float(h["composite_hazard_score"]) < 0.4
                    else "high"
                )
                holding_lines.append(
                    f"{h['company_name']} ({h['sector']}) has an extra expected loss "
                    f"of {fmt_usd_plain(float(h['incremental_el']))}, its chance of default rises to "
                    f"{float(h['climate_adjusted_pd'])*100:.2f}%, it faces a "
                    f"{float(h['ebitda_shock_pct'])*100:.1f}% hit to operating profits, "
                    f"and has {hazard_plain} physical exposure to climate hazards."
                )

            holding_text = " ".join(holding_lines)

            para1 = (
                f"What is this scenario? "
                f"The {scenario_obj5.display_name} scenario describes a world where "
                f"{scenario_desc.lower()} "
                f"Under this path, the planet warms by roughly {warming} compared to "
                f"pre-industrial levels. This analysis looks at how that world, by the year "
                f"{summary_year}, could affect the companies and assets in this portfolio."
            )

            para2 = (
                f"How much money is at risk? "
                f"The total amount invested across this portfolio is {fmt_usd_plain(total_ead)}. "
                f"Under this scenario, climate risk is expected to add an extra "
                f"{fmt_usd_plain(incr_el)} in losses on top of what would normally be expected "
                f"— a {severity} increase. In a bad but plausible outcome, total "
                f"climate-related losses could reach {fmt_usd_plain(var_5)}. In a very severe "
                f"but rare scenario, they could reach as high as {fmt_usd_plain(es_5)}. "
                f"The dominant source of risk is {driver}. {driver_detail}"
            )

            para3 = (
                f"Which companies are most exposed? "
                f"{holding_text} "
                f"These companies face the greatest strain because their businesses are "
                f"either heavily exposed to carbon pricing, located in areas prone to "
                f"climate hazards, or both. The rest of the portfolio carries lower but "
                f"still meaningful climate risk that grows over time."
            )

            summary_text = f"{para1}\n\n{para2}\n\n{para3}"

            st.session_state["plain_summary"] = summary_text
            st.session_state["plain_summary_key"] = (summary_scenario, summary_year)

        if (
            "plain_summary" in st.session_state
            and st.session_state.get("plain_summary_key") == (summary_scenario, summary_year)
        ):
            st.divider()
            st.markdown(st.session_state["plain_summary"])
            st.divider()
            st.download_button(
                label="Download summary as text",
                data=st.session_state["plain_summary"].encode("utf-8"),
                file_name=f"climate_risk_summary_{summary_scenario}_{summary_year}.txt",
                mime="text/plain",
                key="download_summary",
            )