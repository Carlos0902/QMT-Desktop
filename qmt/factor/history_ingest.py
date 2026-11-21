"""Collect historical execution evidence without certifying archive completeness.

Tushare dividends and the public fund announcement index are secondary discovery
sources. Original issuer documents must confirm rates/status/events before strict
backtests. A successful query with zero records does not certify zero events.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import requests

from qmt.data.tushare_client import TushareClient

ANN_API = "https://api.fund.eastmoney.com/f10/JJGG"
DIV_DOC = "https://tushare.pro/document/2?doc_id=120"


def collect_dividends(codes, destination: Path, *, seal=date(2024, 12, 10)):
    destination.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for code in codes:
        try:
            frame = TushareClient(timeout=20, max_retries=0).query("fund_div", {"ts_code": code + ".OF"})
            if len(frame):
                frame = frame[(frame.ann_date < seal.strftime("%Y%m%d"))
                              & (frame.ex_date < seal.strftime("%Y%m%d"))].drop_duplicates()
            frame.to_json(destination / (code + ".json"), orient="records", force_ascii=False)
            outcomes[code] = {"rows": len(frame), "status": "collected"}
        except Exception as exc:
            outcomes[code] = {"status": "failed", "error_type": type(exc).__name__}
    return {"provider": "Tushare fund_div", "documentation": DIV_DOC,
            "seal_excluded": str(seal), "outcomes": outcomes,
            "primary_announcement_verified": False, "coverage_certified": False}


def collect_announcement_index(code, destination: Path, *, seal=date(2024, 12, 10), max_pages=30):
    """Page at exactly 100 to avoid server's cap causing skipped records."""
    destination.mkdir(parents=True, exist_ok=True)
    found = {}
    page_audit = []
    total = None
    complete_index = False
    session = requests.Session()
    for page in range(1, max_pages + 1):
        response = session.get(ANN_API, params={"fundcode": code, "pageIndex": page,
                               "pageSize": 100, "type": 0},
                               headers={"Referer": "https://fundf10.eastmoney.com/", "User-Agent": "Mozilla/5.0"}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if payload.get("ErrCode", 0) != 0:
            raise ValueError("announcement API returned nonzero ErrCode")
        rows = payload.get("Data") or []
        total = int(payload.get("TotalCount", 0))
        digest = hashlib.sha256(response.content).hexdigest()
        eligible = [r for r in rows if str(r.get("PUBLISHDATE", ""))[:10] < str(seal)]
        page_audit.append({"page": page, "n_rows": len(rows), "n_pre_seal": len(eligible),
                           "sha256_response": digest, "requested_url": response.url})
        # Persist only pre-seal announcement metadata; never use current status.
        (destination / f"{code}_page{page}.json").write_text(json.dumps({"Data": eligible,
                      "TotalCount_at_fetch": total, "response_sha256": digest}, ensure_ascii=False, indent=2))
        for record in eligible:
            found[record["ID"]] = record
        if not rows or page * 100 >= total:
            complete_index = True
            break
    relevant = [r for r in found.values() if re.search(
        "暂停|恢复|开放|大额|限额|分红|收益分配|招募|费率|收费|合同|清算|终止|份额折算", r.get("TITLE", ""))]
    result = {"fund_code": code, "provider": ANN_API, "pages": page_audit,
              "provider_total": total, "provider_index_pagination_finished": complete_index,
              "pre_seal_records": len(found), "relevant_records": relevant,
              "source_role": "secondary announcement discovery index",
              "official_complete_archive_verified": False,
              "fee_status_dividend_coverage_certified": False}
    (destination / (code + "_index.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data_cache/execution_history/universe.json"))
    parser.add_argument("--destination", type=Path, default=Path("data_cache/execution_history"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    codes = [r["fund_code"] for r in json.loads(args.universe.read_text())["domestic"]]
    outcomes = {}
    with ThreadPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
        futures = {pool.submit(collect_announcement_index, c, args.destination / "announcement_index"): c for c in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                r = future.result()
                outcomes[code] = {"status": "collected", "pre_seal_records": r["pre_seal_records"],
                                  "n_relevant": len(r["relevant_records"]),
                                  "index_pagination_finished": r["provider_index_pagination_finished"]}
            except Exception as exc:
                outcomes[code] = {"status": "failed", "error_type": type(exc).__name__}
            print(code, outcomes[code], flush=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "announcement_provider_manifest.json").write_text(json.dumps(
        {"provider": ANN_API, "outcomes": outcomes, "official_coverage_certified": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
