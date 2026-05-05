"""
Module 1 — Data Loader
Reads portfolio CSV and carbon price CSV into domain objects.
Accepts file paths (str/Path) or file-like objects (StringIO) for dashboard uploads.
"""

from __future__ import annotations
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from src.schema import Holding, Portfolio, AssetType, Sector, CreditRating
from src.scenarios import get_scenario

FileInput = Union[str, Path, io.StringIO]


def _open_input(source: FileInput):
    """Normalize file path or StringIO into an open text stream."""
    if isinstance(source, io.StringIO):
        source.seek(0)
        return source, False   # (stream, should_close)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return open(path, newline="", encoding="utf-8"), True


@dataclass
class CarbonPricePoint:
    scenario_id: str
    year: int
    carbon_price_usd_per_tco2e: float


def load_portfolio(source: FileInput) -> Portfolio:
    stream, should_close = _open_input(source)
    portfolio_id = Path(source).stem if not isinstance(source, io.StringIO) else "uploaded"
    portfolio = Portfolio(portfolio_id=portfolio_id)
    try:
        reader = csv.DictReader(stream)
        for i, row in enumerate(reader, start=2):
            try:
                holding = Holding(
                    holding_id=row["holding_id"].strip(),
                    company_name=row["company_name"].strip(),
                    asset_type=AssetType(row["asset_type"].strip()),
                    sector=Sector(row["sector"].strip()),
                    country=row["country"].strip().upper(),
                    exposure_usd=float(row["exposure_usd"]),
                    credit_rating=CreditRating(row["credit_rating"].strip().upper()),
                    lgd_baseline=float(row["lgd_baseline"]),
                    ebitda_margin=float(row["ebitda_margin"]),
                    emissions_intensity=float(row["emissions_intensity"]),
                    annual_revenue_usd_m=float(row["annual_revenue_usd_m"]),
                    latitude=float(row["latitude"]) if row.get("latitude", "").strip() else None,
                    longitude=float(row["longitude"]) if row.get("longitude", "").strip() else None,
                    region=row.get("region", "").strip() or None,
                )
                portfolio.add_holding(holding)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error in portfolio CSV row {i}: {e}") from e
    finally:
        if should_close:
            stream.close()
    return portfolio


def load_carbon_prices(source: FileInput) -> list[CarbonPricePoint]:
    stream, should_close = _open_input(source)
    prices: list[CarbonPricePoint] = []
    try:
        reader = csv.DictReader(stream)
        for i, row in enumerate(reader, start=2):
            try:
                scenario_id = row["scenario_id"].strip()
                get_scenario(scenario_id)
                prices.append(CarbonPricePoint(
                    scenario_id=scenario_id,
                    year=int(row["year"]),
                    carbon_price_usd_per_tco2e=float(row["carbon_price_usd_per_tco2e"]),
                ))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Error in carbon price CSV row {i}: {e}") from e
    finally:
        if should_close:
            stream.close()
    return prices