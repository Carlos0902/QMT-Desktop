"""Thin client over the Tushare Pro HTTP API (data-layer, tushare-mapping slice 2).

Tushare Pro exposes a single POST endpoint (``https://api.tushare.pro``) that takes
``{api_name, token, params, fields}`` and returns
``{"code": 0, "msg": null, "data": {"fields": [...], "items": [[...], ...]}}``.

This module is a *thin* transport layer only: it authenticates (token from
``qmt.config``), retries transient failures, backs off on the points-tiered
per-minute rate limit, and — crucially — surfaces the free/low-points quota errors
as a typed :class:`TushareQuotaError` instead of crashing. It performs **no**
point-in-time tagging or domain mapping; that is :mod:`qmt.data.sources`.

Quota behaviour (observed / documented, see tushare notes in the plan): Tushare gates
every interface behind an accumulated-points threshold. A free ~120-point account
cannot call most *fund* interfaces at all — the API returns a non-zero ``code`` with a
Chinese ``msg`` mentioning 权限 / 积分 ("no permission" / "insufficient points"). The
per-minute rate limit (tiered by points) returns a ``msg`` mentioning 每分钟 / 频率.
We classify the two: quota/permission errors are **non-retryable and surfaced**
(:class:`TushareQuotaError`); rate-limit errors are **retried after a wait**.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import requests

from qmt.config import CONFIG, get_settings

__all__ = [
    "TushareClient",
    "TushareError",
    "TushareAPIError",
    "TushareQuotaError",
    "TushareRateLimitError",
    "fund_code_to_ts_code",
    "ts_code_to_fund_code",
]

TUSHARE_API_URL = "https://api.tushare.pro"

# Constrain content negotiation to gzip/deflate. Left unset, the server may return a
# brotli (``content-encoding: br``) body that some installed brotli decoders fail to
# decompress on chunked transfer (ContentDecodingError), which would otherwise be
# misclassified as a transient network failure and retried to exhaustion. gzip is
# universally decodable by requests and returns the identical payload.
_REQUEST_HEADERS = {"Accept-Encoding": "gzip, deflate"}


# =============================================================================
# Exceptions
# =============================================================================
class TushareError(RuntimeError):
    """Base class for all Tushare transport/API failures."""


class TushareAPIError(TushareError):
    """A non-zero API ``code`` that is neither a quota nor a rate-limit condition."""

    def __init__(self, code: int, msg: str, api_name: str) -> None:
        self.code = code
        self.msg = msg
        self.api_name = api_name
        super().__init__(f"tushare api_name={api_name!r} code={code} msg={msg!r}")


class TushareQuotaError(TushareError):
    """Insufficient-points / no-permission error (the free/low-points tier).

    Non-retryable: retrying will never succeed until the account's points change, so
    the caller must surface this to the operator rather than loop. Raised instead of
    crashing so the data layer can degrade gracefully (a data gap, not a stack trace).
    """

    def __init__(self, msg: str, api_name: str) -> None:
        self.msg = msg
        self.api_name = api_name
        super().__init__(
            f"tushare quota/permission error on api_name={api_name!r}: {msg!r} "
            f"(the account lacks the points to call this interface; see plan quota notes)"
        )


class TushareRateLimitError(TushareError):
    """Per-minute rate limit hit and not clearing within ``max_retries`` waits."""

    def __init__(self, msg: str, api_name: str) -> None:
        self.msg = msg
        self.api_name = api_name
        super().__init__(f"tushare rate limit not clearing on api_name={api_name!r}: {msg!r}")


# =============================================================================
# ts_code <-> fund_code helpers (D14: canonical 6-digit code in the domain layer)
# =============================================================================
def fund_code_to_ts_code(fund_code: str, suffix: Optional[str] = None) -> str:
    """Map a canonical 6-digit ``fund_code`` to a Tushare ``ts_code``.

    OTC (场外) funds take the ``.OF`` suffix (config ``tushare.otc_ts_code_suffix``).
    If ``fund_code`` already carries a ``.`` suffix it is returned unchanged.
    """
    if "." in fund_code:
        return fund_code
    suf = CONFIG.tushare.otc_ts_code_suffix if suffix is None else suffix
    return f"{fund_code}{suf}"


def ts_code_to_fund_code(ts_code: str) -> str:
    """Strip the exchange/market suffix, returning the canonical 6-digit code."""
    return ts_code.split(".", 1)[0]


# =============================================================================
# Error classification (quota vs rate-limit vs generic)
# =============================================================================
# Keywords are matched against the Chinese ``msg`` Tushare returns.
_QUOTA_HINTS = ("权限", "积分", "没有足够", "不足", "开通", "membership", "permission")
_RATE_HINTS = ("每分钟", "每小时", "频率", "分钟内", "次数", "限制访问", "too many", "frequency")


def _classify_error(msg: Optional[str]) -> str:
    """Return one of ``'quota'`` / ``'rate_limit'`` / ``'error'`` for a non-zero code.

    Rate-limit is checked first: a message can mention both 次数 (rate) and other terms,
    but a per-minute limit is transient and worth retrying, whereas a points/permission
    problem is terminal — so when in doubt we prefer the retryable classification only
    when a rate hint is present *and* no quota hint is.
    """
    text = msg or ""
    has_quota = any(h in text for h in _QUOTA_HINTS)
    has_rate = any(h in text for h in _RATE_HINTS)
    if has_quota and not has_rate:
        return "quota"
    if has_rate and not has_quota:
        return "rate_limit"
    if has_quota:  # both present → treat as quota (terminal, fail-closed / surface)
        return "quota"
    if has_rate:
        return "rate_limit"
    return "error"


# =============================================================================
# Client
# =============================================================================
class TushareClient:
    """Minimal, retrying POST client for the Tushare Pro API.

    Parameters
    ----------
    token:
        Tushare Pro token. Defaults to ``qmt.config.get_settings().tushare_token``
        (loaded from ``.env`` / env var ``TUSHARE_TOKEN``; never hard-coded).
    base_url:
        API endpoint (override for testing).
    timeout:
        Per-request socket timeout (seconds).
    max_retries:
        How many times to retry a transient failure (network error or rate limit).
    backoff_base:
        Base seconds for exponential backoff on network errors (``backoff_base * 2**n``).
    rate_limit_wait_seconds:
        Seconds to wait before retrying after a per-minute rate-limit error. Tushare's
        window is one minute, so production should keep this ~60s; tests set it to 0.
    session:
        Injected ``requests.Session`` (tests inject a fake to run offline).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        base_url: str = TUSHARE_API_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        rate_limit_wait_seconds: float = 61.0,
        session: Optional[Any] = None,
    ) -> None:
        from os import environ
        if token is None and environ.get("QMT_DESKTOP") == "1":
            from qmt.desktop_profile import read_key
            token = read_key()
        self._token = token if token is not None else get_settings().tushare_token
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._rate_limit_wait = rate_limit_wait_seconds
        self._session = session if session is not None else requests.Session()

    @property
    def has_token(self) -> bool:
        """Whether a non-empty token is configured (False → every call will 40203)."""
        return bool(self._token)

    # --- the single low-level call -----------------------------------------
    def query(
        self,
        api_name: str,
        params: Optional[Mapping[str, Any]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Call ``api_name`` and return its result as a DataFrame (columns = fields).

        Raises
        ------
        TushareQuotaError
            The account lacks points/permission for this interface (non-retryable).
        TushareRateLimitError
            The per-minute rate limit did not clear within ``max_retries`` waits.
        TushareAPIError
            Any other non-zero API code.
        TushareError
            A network/transport failure that persisted across all retries.
        """
        if not self.has_token:
            raise TushareError("请先配置 Tushare Pro Key。")
        # Tushare's HTTP API expects ``fields`` as a comma-separated STRING (its canonical
        # form). A str is itself a Sequence[str], so ``list(fields)`` would explode
        # "ts_code,name" into single characters; Tushare then silently ignores the
        # unrecognised field list and dumps ALL columns in UPPERCASE, which the lowercase
        # domain mapping in qmt.data.sources reads as None and drops every row. Pass a
        # string through unchanged; join an explicit list of field names with commas.
        if not fields:
            fields_param: str = ""
        elif isinstance(fields, str):
            fields_param = fields
        else:
            fields_param = ",".join(fields)
        payload: dict[str, Any] = {
            "api_name": api_name,
            "token": self._token,
            "params": dict(params) if params else {},
            "fields": fields_param,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._session.post(
                    self._base_url, json=payload, timeout=self._timeout, headers=_REQUEST_HEADERS
                )
                resp.raise_for_status()
                body = resp.json()
            except requests.exceptions.RequestException as exc:  # network/transport
                last_exc = exc
                if attempt >= self._max_retries:
                    raise TushareError(
                        f"network failure calling api_name={api_name!r} after "
                        f"{self._max_retries + 1} attempts: {exc}"
                    ) from exc
                time.sleep(self._backoff_base * (2**attempt))
                continue

            code = int(body.get("code", 0) or 0)
            msg = body.get("msg")
            if code == 0:
                return self._to_frame(body.get("data") or {})

            kind = _classify_error(msg)
            if kind == "quota":
                # Terminal: surface (do not crash the process, do not retry).
                raise TushareQuotaError(msg or "", api_name)
            if kind == "rate_limit":
                if attempt >= self._max_retries:
                    raise TushareRateLimitError(msg or "", api_name)
                time.sleep(self._rate_limit_wait)
                continue
            raise TushareAPIError(code, msg or "", api_name)

        # Unreachable, but keeps the type checker happy.
        raise TushareError(f"exhausted retries for api_name={api_name!r}: {last_exc}")

    @staticmethod
    def _to_frame(data: Mapping[str, Any]) -> pd.DataFrame:
        fields = list(data.get("fields") or [])
        items = list(data.get("items") or [])
        if not fields:
            return pd.DataFrame()
        return pd.DataFrame(items, columns=fields)
