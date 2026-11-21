"""Data layer (slice 2) — Tushare transport + point-in-time-correct fetchers.

* :mod:`qmt.data.tushare_client` — thin retrying HTTP client over the Tushare Pro API,
  with quota/rate-limit classification and ``fund_code``↔``ts_code`` helpers.
* :mod:`qmt.data.sources` — typed fetchers that tag each row's ``available_at``
  correctly (invariant 1) and cache into the :class:`~qmt.pit.PointInTimeStore`.

Nothing here reads a :class:`~qmt.types.UserPortfolio` (analysis/advice isolation,
invariant 3): the data layer is mode-agnostic and only ever serves point-in-time facts.
"""

from __future__ import annotations

from qmt.data.sources import (
    DATA_GAPS,
    TABLE_FUND_BASIC,
    TABLE_FUND_NAV,
    TABLE_FUND_PORTFOLIO,
    TABLE_INDEX_DAILY,
    TABLE_SW_DAILY,
    fund_nav,
    fund_portfolio,
    fund_universe,
    index_daily,
    sw_daily,
)
from qmt.data.tushare_client import (
    TushareAPIError,
    TushareClient,
    TushareError,
    TushareQuotaError,
    TushareRateLimitError,
    fund_code_to_ts_code,
    ts_code_to_fund_code,
)
from qmt.data.ingest import (
    IngestReport,
    build_industry_map,
    ingest_fund,
    ingest_universe,
    load_cached_industry_map,
)

__all__ = [
    # client
    "TushareClient",
    "TushareError",
    "TushareAPIError",
    "TushareQuotaError",
    "TushareRateLimitError",
    "fund_code_to_ts_code",
    "ts_code_to_fund_code",
    # sources
    "fund_universe",
    "fund_nav",
    "fund_portfolio",
    "index_daily",
    "sw_daily",
    # ingest / priming
    "ingest_fund",
    "ingest_universe",
    "build_industry_map",
    "load_cached_industry_map",
    "IngestReport",
    # store table names + data gaps
    "TABLE_FUND_BASIC",
    "TABLE_FUND_NAV",
    "TABLE_FUND_PORTFOLIO",
    "TABLE_INDEX_DAILY",
    "TABLE_SW_DAILY",
    "DATA_GAPS",
]
