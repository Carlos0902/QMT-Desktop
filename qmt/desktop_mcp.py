"""Local stdio MCP interface for fund research."""
from __future__ import annotations

from datetime import date, timedelta
import json
import re

from mcp.server.fastmcp import FastMCP
from qmt.desktop_profile import configure, read_key
from qmt.data.tushare_client import TushareClient, TushareError

server = FastMCP("QMT Desktop")


def _code(value):
    if not re.fullmatch(r"[0-9]{6}", value):
        raise ValueError("请输入六位基金代码。")
    return value


@server.tool()
def qmt_status() -> dict:
    """Check whether a data key is configured for this Windows user."""
    from qmt import __version__
    return {"version": __version__, "provider": "Tushare Pro", "configured": bool(read_key())}


@server.tool()
def fund_nav(fund_code: str, start: date, end: date) -> dict:
    """Get published NAV observations for one off-exchange fund, up to one year per call."""
    code = _code(fund_code)
    if end < start or end - start > timedelta(days=366):
        raise ValueError("每次查询最多支持一年，结束日期须晚于或等于开始日期。")
    key = read_key()
    if not key:
        raise ValueError("请在 QMT Desktop 的设置中保存 Tushare Pro Key。")
    client = TushareClient(token=key, timeout=20, max_retries=0)
    try:
        frame = client.query("fund_nav", {"ts_code": code + ".OF",
            "start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
            ["ts_code", "nav_date", "ann_date", "unit_nav", "adj_nav"])
    except TushareError:
        raise ValueError("Tushare 查询失败。请检查密钥、接口权限、网络和查询频率。") from None
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    return {"provider": "Tushare Pro", "fund_code": code, "observations": records}


@server.tool()
def analyze_cached_fund(fund_code: str, as_of: date) -> dict:
    """Analyze a cached fund as of a date. Use Desktop's analysis page to fetch data first."""
    from fastapi.testclient import TestClient
    from qmt.controlplane.api import create_app
    with TestClient(create_app()) as client:
        result = client.post("/api/analyze", json={"fund_code": _code(fund_code),
                             "as_of": as_of.isoformat(), "ingest": False})
    if result.status_code != 200:
        raise ValueError("该日期的缓存无法完成分析。请先在控制台运行此基金并检查数据覆盖。")
    return result.json()


if __name__ == "__main__":
    configure()
    server.run(transport="stdio")
