"""Daily cash-and-unit accounting for long-only OTC funds.

Orders are sized at the decision cutoff, before that session's valuation NAV is
known. Unit NAV converts money to shares. Optional automatic dividend
reinvestment reconstructs bonus shares from adjusted NAV / unit NAV changes.
Bonus shares form a new zero-subscription-fee lot with a fresh confirmation age;
this is an explicit modelling assumption, not observed cash-dividend data.
Management, custody and sales-service expenses already embedded in reported NAV
are NEVER deducted again. Only supplied subscription/redemption schedules are
charged. Settlement and confirmation delays count supplied valuation sessions;
redemption holding periods count calendar days between confirmations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
import inspect
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from qmt.fees import default_fee_schedule, redemption_amounts, subscription_amounts
from qmt.factor.event_history import DividendHistory, DividendEvent, TradingStatus

__all__ = [
    "ExecutionConfig", "FundLot", "TradeEvent", "CashReceivable", "BacktestResult",
    "DividendClaim", "run_fund_backtest", "scores_to_weights",
]

_EPS = 1e-10
WeightCallback = Callable[[date, pd.Series], pd.Series | Mapping[str, float] | None]


@dataclass
class ExecutionConfig:
    trading_dates: Sequence[date]
    initial_cash: float = 100_000.0
    top_k: int = 5
    sticky_rank_buffer: int = 0
    fill_lag_sessions: int = 1
    purchase_confirm_sessions: int = 1
    redemption_confirm_sessions: int = 1
    redemption_settlement_sessions: int = 2
    confirmation_requires_public_nav: bool = True
    reinvest_dividends: bool = True
    dividend_mode: str = "ratio_proxy"  # ratio_proxy | cash | reinvest
    adjustment_ratio_tolerance: float = 0.001
    seal_date: date | None = None
    end_date: date | None = None
    liquidate_at_end: bool = False
    status_callback: Callable[[str, date, str], bool | TradingStatus | None] | None = None
    require_status: bool = False
    fee_assumptions: tuple[str, ...] = ()
    covered_daily_buy_money_exclusive: float | None = None  # evidence scope, not a dealing cap
    research_minimum_remaining_units: float = 0.0  # common admissible policy, not inferred true minimum
    research_minimum_redemption_units: float = 0.0


@dataclass
class FundLot:
    lot_id: int
    fund_code: str
    units: float
    fill_date: date
    confirmation_due_date: date | None
    confirmation_date: date | None = None
    adjustment_ratio: float = 1.0
    adjustment_date: date | None = None
    redeem_available_date: date | None = None
    confirmation_date_fixed: bool = False
    fill_nav: float | None = None
    fill_adjustment_ratio: float = 1.0


@dataclass(frozen=True)
class TradeEvent:
    date: date
    event_type: str
    fund_code: str = ""
    decision_date: date | None = None
    fill_date: date | None = None
    units: float = 0.0
    nav: float | None = None
    gross_money: float = 0.0
    cash_money: float = 0.0
    cost_money: float = 0.0
    hold_days: int | None = None
    lot_id: int | None = None
    note: str = ""
    order_date: date | None = None


@dataclass
class CashReceivable:
    fund_code: str
    fill_date: date
    settlement_date: date | None
    amount: float
    confirmation_due_date: date | None = None
    confirmation_date: date | None = None
    purchase_confirmation_date: date | None = None
    gross_money: float = 0.0
    fee_money: float = 0.0
    fee_schedule: object | None = None
    lot_id: int | None = None
    fill_nav: float | None = None
    fill_adjustment_ratio: float = 1.0


@dataclass
class DividendClaim:
    event: DividendEvent
    entitled_units: float
    amount: float
    recognized: bool = False
    completed: bool = False


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    events: list[TradeEvent]
    lots: list[FundLot]
    receivables: list[CashReceivable]
    initial_cash: float
    assumptions: list[str] = field(default_factory=list)
    dividend_claims: list[DividendClaim] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return float(self.daily.equity.iloc[-1]) if len(self.daily) else self.initial_cash

    @property
    def net_return(self) -> float:
        return self.final_equity / self.initial_cash - 1.0

    @property
    def cost_money(self) -> float:
        return float(self.daily.cost_money.sum()) if len(self.daily) else 0.0

    @property
    def total_cost_money(self) -> float:
        return self.cost_money

    @property
    def total_traded_money(self) -> float:
        return float(self.daily.traded_gross_money.sum()) if len(self.daily) else 0.0

    @property
    def turnover(self) -> float:
        """Sum of daily traded NAV notional / (2 * pre-fill equity)."""
        return float(self.daily.turnover.sum()) if len(self.daily) else 0.0

    @property
    def max_drawdown(self) -> float:
        values = np.r_[self.initial_cash, self.daily.equity.to_numpy(float)]
        return float(np.min(values / np.maximum.accumulate(values) - 1.0))

    def summary(self) -> dict:
        return {"final_equity": self.final_equity, "net_return": self.net_return,
                "cost_money": self.cost_money, "total_traded_money": self.total_traded_money,
                "turnover": self.turnover, "max_drawdown": self.max_drawdown,
                "n_events": len(self.events),
                "pending_estimated_redemption_fees": sum(r.confirmation_date is None for r in self.receivables),
                "pending_dividend_claims": sum(c.recognized and not c.completed for c in self.dividend_claims),
                "assumptions": self.assumptions}


@dataclass
class _Order:
    fund_code: str
    side: str
    decision_date: date
    fill_date: date
    amount: float = 0.0  # buy: gross cash reserved; sell: fixed requested units
    fee_schedule: object | None = None
    order_date: date | None = None
    net_amount: float = 0.0
    funding_epoch: int = 0


@dataclass
class _FundingIntent:
    fund_code: str
    decision_date: date
    net_remaining: float
    epoch: int


def scores_to_weights(scores, drifted_weights: pd.Series, *, top_k: int,
                      sticky_rank_buffer: int = 0) -> pd.Series:
    """Deterministic equal-weight top K, with an optional held-name rank buffer."""
    if top_k < 1 or sticky_rank_buffer < 0:
        raise ValueError("top_k must be positive and sticky_rank_buffer nonnegative")
    s = pd.Series(scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    ranked = sorted(s.index, key=lambda c: (-float(s[c]), str(c)))
    buffer = set(ranked[:top_k + sticky_rank_buffer])
    held = {c for c, w in drifted_weights.items() if w > _EPS}
    selected = [c for c in ranked if c in held and c in buffer][:top_k]
    selected += [c for c in ranked if c not in selected][:max(0, top_k - len(selected))]
    return pd.Series(1.0 / len(selected), index=selected, dtype=float) if selected else pd.Series(dtype=float)


def _normalized_frame(frame: pd.DataFrame, sessions: list[date], *, numeric=True) -> pd.DataFrame:
    index = [pd.Timestamp(d).date() for d in frame.index]
    if len(index) != len(set(index)):
        raise ValueError("NAV frames must have unique valuation dates")
    if frame.columns.has_duplicates:
        raise ValueError("NAV frames must have unique fund columns")
    # Slice before numeric conversion; no values outside the simulation are used.
    normalized = frame.set_axis(index, axis=0).sort_index().reindex(sessions)
    return normalized.astype(float) if numeric else normalized


def run_fund_backtest(
    unit_nav: pd.DataFrame,
    adjusted_nav: pd.DataFrame | None,
    decision_dates: Sequence[date],
    score_callback: WeightCallback | None = None,
    *,
    weight_callback: WeightCallback | None = None,
    fees_resolver: Callable | Mapping[str, object] | None = None,
    config: ExecutionConfig,
    end_date: date | None = None,
    visible_unit_nav: pd.DataFrame | None = None,
    visible_adjusted_nav: pd.DataFrame | None = None,
    visible_nav_dates: pd.DataFrame | None = None,
    dividend_events: DividendHistory | None = None,
) -> BacktestResult:
    """Run a cash-funded OTC portfolio, stopping before ``config.seal_date``.

    Exactly one callback is required. ``None`` retains holdings without placing
    orders; an empty Series intentionally targets cash. A weight callback returns
    finite nonnegative weights summing to at most one. A score callback uses the
    common deterministic top-K/hysteresis rule. Both receive economic holdings
    weights, not target weights. Features must be PIT-safe upstream.

    Visible frames represent information published by the cutoff, and their
    ``visible_nav_dates`` identify the raw valuation dates. By default,
    confirmation/settlement waits for public fill-NAV publication plus configured
    delays. Set ``confirmation_requires_public_nav=False`` for normal contractual
    confirmation/payment timing; only confirmed own fills may then supplement
    public prices in account valuation. They never modify public feature frames.
    Without visible frames, prior-session publication is an explicit assumption
    suitable for synthetic tests, not late-disclosure data. Neither timing mode
    is a replay of actual broker confirmation/settlement receipts.
    ``daily.cost_money`` and traded money are DAILY increments, not cumulative.
    Unfunded net buy increments remain frozen until sale proceeds settle; fees are
    resolved on actual order placement, without re-scoring or drift chasing. A
    replacement target cancels outstanding unfunded intents; None retains them.
    End liquidation is optional and best effort: it places next-fill orders early
    enough for the final session, and cannot redeem unconfirmed units.
    """
    if (score_callback is None) == (weight_callback is None):
        raise ValueError("supply exactly one score_callback or weight_callback")
    if not np.isfinite(config.initial_cash) or config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if config.fill_lag_sessions not in (0, 1):
        raise ValueError("fill_lag_sessions must be 0 or 1")
    if config.purchase_confirm_sessions < 1 or config.redemption_confirm_sessions < 1:
        raise ValueError("purchase and redemption confirmation delays must be >= 1 session")
    if config.redemption_settlement_sessions < max(2, config.redemption_confirm_sessions):
        raise ValueError("redemption settlement must be >= 2 sessions and >= confirmation")
    if config.top_k < 1 or config.sticky_rank_buffer < 0:
        raise ValueError("invalid top_k or sticky_rank_buffer")
    if not np.isfinite(config.research_minimum_remaining_units) or config.research_minimum_remaining_units < 0:
        raise ValueError("research remaining-unit policy must be finite and nonnegative")
    if not np.isfinite(config.research_minimum_redemption_units) or config.research_minimum_redemption_units < 0:
        raise ValueError("research redemption-unit policy must be finite and nonnegative")
    if not isinstance(config.confirmation_requires_public_nav, bool):
        raise ValueError("confirmation_requires_public_nav must be a boolean")
    if (config.covered_daily_buy_money_exclusive is not None and
            (not np.isfinite(config.covered_daily_buy_money_exclusive) or config.covered_daily_buy_money_exclusive <= 0)):
        raise ValueError("covered daily purchase amount must be positive and finite")
    if config.dividend_mode not in {"ratio_proxy", "cash", "reinvest"}:
        raise ValueError("dividend_mode must be ratio_proxy, cash or reinvest")
    proxy_reinvest = config.dividend_mode == "ratio_proxy" and config.reinvest_dividends
    observed_dividends = config.dividend_mode != "ratio_proxy"
    if observed_dividends and not isinstance(dividend_events, DividendHistory):
        raise ValueError("observed dividend modes require DividendHistory with proven coverage")
    if proxy_reinvest and adjusted_nav is None:
        raise ValueError("adjusted_nav is required for assumed dividend reinvestment")
    if not np.isfinite(config.adjustment_ratio_tolerance) or not 0 <= config.adjustment_ratio_tolerance < 1:
        raise ValueError("adjustment_ratio_tolerance must be finite and lie in [0, 1)")
    if visible_adjusted_nav is not None and visible_unit_nav is None:
        raise ValueError("visible_adjusted_nav requires visible_unit_nav")
    if visible_nav_dates is not None and visible_unit_nav is None:
        raise ValueError("visible_nav_dates requires visible_unit_nav")

    calendar = sorted({pd.Timestamp(d).date() for d in config.trading_dates})
    limit = pd.Timestamp(end_date or config.end_date).date() if (end_date or config.end_date) else None
    seal = pd.Timestamp(config.seal_date).date() if config.seal_date else None
    sessions = [d for d in calendar if (limit is None or d <= limit) and (seal is None or d < seal)]
    if not sessions:
        raise ValueError("no observed sessions before end_date/seal_date")
    decisions = {pd.Timestamp(d).date() for d in decision_dates}
    if any(d not in set(calendar) for d in decisions):
        raise ValueError("decision_dates must lie on the supplied observed calendar")
    active_decisions = decisions & set(sessions)
    simulation_start = min(active_decisions) if active_decisions else sessions[0]
    raw = _normalized_frame(unit_nav, sessions)
    codes = sorted(str(c) for c in raw.columns)
    raw.columns = [str(c) for c in raw.columns]
    raw = raw.reindex(columns=codes)
    if observed_dividends:
        dividend_events.assert_coverage(codes, simulation_start, sessions[-1])
    adjusted = (_normalized_frame(adjusted_nav, sessions).rename(columns=str).reindex(columns=codes)
                if adjusted_nav is not None else raw)
    visible = (_normalized_frame(visible_unit_nav, sessions).rename(columns=str).reindex(columns=codes)
               if visible_unit_nav is not None else None)
    visible_adj = (_normalized_frame(visible_adjusted_nav, sessions).rename(columns=str).reindex(columns=codes)
                   if visible_adjusted_nav is not None else None)
    source_dates = (_normalized_frame(visible_nav_dates, sessions, numeric=False)
                    .rename(columns=str).reindex(columns=codes) if visible_nav_dates is not None else None)
    values = raw.to_numpy(float)
    adjustments = adjusted.to_numpy(float)
    positions = {d: i for i, d in enumerate(calendar)}
    col = {c: i for i, c in enumerate(codes)}
    cash = float(config.initial_cash)
    marks: dict[str, float] = {}
    ratios: dict[str, float] = {}
    mark_dates: dict[str, date] = {}
    private_quotes: dict[str, tuple[date, float, float]] = {}
    lots: list[FundLot] = []
    orders: list[_Order] = []
    funding_intents: dict[str, _FundingIntent] = {}
    funding_epoch = 0
    receivables: list[CashReceivable] = []
    events: list[TradeEvent] = []
    records = []
    dividend_claims: dict[tuple[str, str], DividendClaim] = {}
    registry_units: dict[tuple[str, date], float] = {}
    application_trades: list[tuple[str, str, date, date, float]] = []
    dividend_cash_arrival = 0.0
    next_lot_id = 1
    ever_subscribed: set[str] = set()
    assumptions = [
        f"Fill valuation lag is {config.fill_lag_sessions} supplied observed session(s); NAV is unknown at order cutoff.",
        "Management/custody/sales-service costs already embedded in NAV; no additional annual holding fee.",
        "Redemption is FIFO; calendar holding days run from purchase confirmation to redemption confirmation (exclusive).",
        "Cash receivables earn no interest and cannot fund purchases before settlement.",
        "Redemption fees are estimated at fill and reconciled at the selected contract/public confirmation date; terminal unconfirmed receivables retain estimated fees.",
        "Unfunded net buy increments from a rebalance execute when spendable proceeds settle; the original signal is frozen until replaced or expired.",
    ]
    assumptions.append("Confirmation and settlement wait for public fill NAV publication (late-public observer scenario)."
                       if config.confirmation_requires_public_nav else
                       "Normal broker confirmation follows configured contract delays independently of public NAV; own confirmed fills supply private account valuation only, not alpha features. Settlement delay is an assumed contract/payment scenario, not observed account receipts.")
    if observed_dividends:
        assumptions.append(f"Observed dividend mode={config.dividend_mode}; source record/ex/payment dates and application entitlement rules are used; adjusted NAV ratio is ignored.")
        assumptions.append("Same-session dated dividend payments are credited at EOD and become spendable next session; payments dated before a session are spendable at its cutoff.")
    else:
        assumptions.append("Automatic dividend reinvestment is reconstructed from adjusted/unit NAV ratios; bonus shares form new fee-free lots confirmed next session, not observed cash-dividend records."
                           if config.reinvest_dividends else "Cash dividends are not modelled; unit-NAV capital returns only.")
    if proxy_reinvest:
        assumptions.append(f"Adjusted/unit ratio changes <= {config.adjustment_ratio_tolerance:.2%} are treated as rounding; credited anchors persist so cumulative larger changes remain detectable.")
    if visible is None:
        assumptions.append("Cutoff valuation assumes prior-session NAV is published on time.")
    elif source_dates is None:
        assumptions.append("Published NAV supplied without source dates; fill publication by scheduled confirmation is assumed.")
    if config.status_callback is None:
        assumptions.append("Fund subscription/redemption status is unavailable; orders blocked."
                           if config.require_status else "Fund subscription/redemption status is unavailable; assumed open.")
    if fees_resolver is None:
        assumptions.append("Subscription/redemption rates use qmt.fees default assumptions, not certified prospectus/channel schedules.")
    else:
        assumptions.extend(config.fee_assumptions or ("Caller-supplied fee schedules; provenance is the caller's responsibility.",))

    def after(d, delay):
        i = positions[d] + delay
        return calendar[i] if 0 <= i < len(calendar) else None

    def valid_number(x):
        return np.isfinite(x) and x > 0

    def published(code, fill, row):
        if source_dates is None:
            return True
        src = source_dates.iloc[row, col[code]]
        return pd.notna(src) and pd.Timestamp(src).date() >= fill

    def remember_private(code, pricedate, nav, ratio, d):
        if config.confirmation_requires_public_nav or nav is None or not valid_number(nav) or pricedate >= d:
            return
        prior = private_quotes.get(code)
        if prior is None or pricedate >= prior[0]:
            private_quotes[code] = (pricedate, float(nav), float(ratio))

    def lot_information_known(lot, row):
        needed = lot.adjustment_date or lot.fill_date
        if published(lot.fund_code, needed, row):
            return True
        if config.confirmation_requires_public_nav or lot.confirmation_date is None:
            return False
        own = private_quotes.get(lot.fund_code)
        return own is not None and own[0] >= needed

    def schedule(code, d):
        if fees_resolver is None:
            return default_fee_schedule(code)
        if isinstance(fees_resolver, Mapping):
            return fees_resolver.get(code)
        try:
            inspect.signature(fees_resolver).bind(code, d)
        except TypeError:
            return fees_resolver(code)
        except ValueError:
            pass
        return fees_resolver(code, d)

    def holdings_value():
        return sum(lot.units * marks[lot.fund_code] for lot in lots)

    def purchase_supported(quote, code, amount):
        bound = getattr(quote, "supported_purchase_amount_max_exclusive", None)
        if bound is not None and amount >= bound:
            raise ValueError(f"unsupported fee amount band for {code}: gross={amount} >= {bound}")
        name = "minimum_additional_purchase_money" if code in ever_subscribed else "minimum_initial_purchase_money"
        minimum = float(getattr(quote, name, 0.0))
        return amount + _EPS >= minimum

    def redemption_supported(quote, code, units):
        minimum = max(float(getattr(quote, "minimum_redemption_units", 0.0)),
                      config.research_minimum_redemption_units)
        all_units = sum(l.units for l in lots if l.fund_code == code)
        return units + _EPS >= minimum or (all_units < minimum and units + _EPS >= all_units)

    def avoid_dust(code, units):
        all_units = sum(l.units for l in lots if l.fund_code == code)
        left = all_units - units
        floor = config.research_minimum_remaining_units
        return max(0.0, all_units - floor) if _EPS < left < floor - _EPS else units

    def dividend_value():
        return sum(claim.amount for claim in dividend_claims.values() if claim.recognized and not claim.completed)

    def entitlement(event):
        base = registry_units.get((event.fund_code, event.record_date))
        if base is None:
            return None
        for code, side, application, fill, units in application_trades:
            if code != event.fund_code or application != event.record_date:
                continue
            if side == "buy" and event.record_date_buy_entitled:
                base += units  # same-day applications have not registered at record close
            elif side == "sell":
                if fill <= event.record_date and event.record_date_sell_entitled:
                    base += units
                elif fill > event.record_date and not event.record_date_sell_entitled:
                    base -= units
        return max(0.0, base)

    def refresh_dividends(known, d):
        for event in known:
            if event.fund_code not in col:
                continue
            units = entitlement(event)
            if units is None:
                continue
            key = (event.fund_code, event.event_id)
            old = dividend_claims.get(key)
            amount = units * event.per_unit_cash
            if old is not None and old.completed:
                if (abs(old.amount - amount) > _EPS or old.event.cash_pay_date != event.cash_pay_date
                        or old.event.reinvest_price_date != event.reinvest_price_date):
                    raise ValueError("material dividend revision after payment/reinvestment requires explicit correction data")
                continue
            recognized = d >= event.ex_date
            if old is None:
                dividend_claims[key] = DividendClaim(event, units, amount, recognized)
                if recognized and amount > _EPS:
                    events.append(TradeEvent(d, "dividend_entitlement", event.fund_code,
                                             gross_money=amount, units=units, note=event.event_id))
            else:
                became_asset = recognized and not old.recognized
                old.event, old.entitled_units, old.amount, old.recognized = event, units, amount, recognized
                if became_asset and amount > _EPS:
                    events.append(TradeEvent(d, "dividend_entitlement", event.fund_code,
                                             gross_money=amount, units=units, note=event.event_id))

    def restore_unfilled_buy(order, unfilled_net=None):
        intent = funding_intents.get(order.fund_code)
        if intent is not None and intent.epoch == order.funding_epoch:
            intent.net_remaining += order.net_amount if unfilled_net is None else unfilled_net

    liquidation_decision = after(sessions[-1], -config.fill_lag_sessions) if config.liquidate_at_end else None
    for row, d in enumerate(sessions):
        cost = traded = 0.0
        covered_buy_used: dict[str, float] = {}
        settled_cash = dividend_cash_arrival
        dividend_cash_arrival = 0.0
        new_funding_target = False
        unknown_status = blocked_orders = 0
        # Confirmations and already-settled proceeds are usable before this cutoff.
        for lot in lots:
            if (lot.confirmation_date is None and lot.confirmation_due_date is not None
                    and lot.confirmation_due_date <= d
                    and (lot.confirmation_date_fixed or not config.confirmation_requires_public_nav
                         or published(lot.fund_code, lot.fill_date, row))):
                lot.confirmation_date = (lot.confirmation_due_date if lot.confirmation_date_fixed
                                         or not config.confirmation_requires_public_nav else d)
                events.append(TradeEvent(d, "buy_confirm", lot.fund_code, fill_date=lot.fill_date,
                                         units=lot.units, lot_id=lot.lot_id))
            if lot.confirmation_date is not None:
                remember_private(lot.fund_code, lot.fill_date, lot.fill_nav, lot.fill_adjustment_ratio, d)
        unsettled = []
        for receivable in receivables:
            if (receivable.confirmation_date is None and receivable.confirmation_due_date is not None
                    and receivable.confirmation_due_date <= d
                    and (not config.confirmation_requires_public_nav
                         or published(receivable.fund_code, receivable.fill_date, row))):
                receivable.confirmation_date = (d if config.confirmation_requires_public_nav
                                                else receivable.confirmation_due_date)
                actual_days = (receivable.confirmation_date - receivable.purchase_confirmation_date).days
                actual_rate = float(receivable.fee_schedule.redemption_rate(actual_days))
                actual_fee = redemption_amounts(receivable.gross_money, actual_rate).fee
                adjustment = actual_fee - receivable.fee_money
                receivable.amount -= adjustment
                receivable.fee_money = actual_fee
                cost += adjustment
                # Preserve the configured post-confirmation settlement interval
                # when delayed publication pushes confirmation later.
                if config.confirmation_requires_public_nav:
                    receivable.settlement_date = after(
                        d, config.redemption_settlement_sessions - config.redemption_confirm_sessions,
                    )
                events.append(TradeEvent(d, "redemption_confirm", receivable.fund_code,
                                         fill_date=receivable.fill_date, gross_money=receivable.gross_money,
                                         cost_money=adjustment, hold_days=actual_days, lot_id=receivable.lot_id,
                                         note="fee reconciliation; negative cost_money is a fee refund"))
            if receivable.confirmation_date is not None:
                remember_private(receivable.fund_code, receivable.fill_date, receivable.fill_nav,
                                 receivable.fill_adjustment_ratio, d)
            if (receivable.settlement_date is not None and receivable.settlement_date <= d
                    and receivable.confirmation_date is not None):
                cash += receivable.amount
                settled_cash += receivable.amount
                events.append(TradeEvent(d, "cash_settle", receivable.fund_code,
                                         fill_date=receivable.fill_date, cash_money=receivable.amount))
            else:
                unsettled.append(receivable)
        receivables = unsettled

        known_dividends = dividend_events.as_of(d) if observed_dividends else []
        if observed_dividends:
            for c in codes:
                registry_units[(c, d)] = sum(l.units for l in lots if l.fund_code == c and l.confirmation_date is not None)
            refresh_dividends(known_dividends, d)
            for claim in dividend_claims.values():
                if not claim.recognized or claim.completed or claim.amount <= _EPS:
                    continue
                event = claim.event
                if config.dividend_mode == "cash":
                    if event.cash_pay_date is None:
                        raise ValueError(f"cash dividend {event.event_id} lacks observed payment date")
                    if event.cash_pay_date < d:
                        cash += claim.amount
                        settled_cash += claim.amount
                        claim.completed = True
                        events.append(TradeEvent(d, "dividend_cash_pay", event.fund_code,
                                                 cash_money=claim.amount, note=event.event_id))

        status_constraints: dict[tuple[str, str], TradingStatus] = {}

        def tradable(code, side, decision=None):
            nonlocal unknown_status, blocked_orders
            status = config.status_callback(code, d, side) if config.status_callback else None
            if status is None:
                unknown_status += 1
                allowed = config.status_callback is None and not config.require_status
            elif isinstance(status, TradingStatus):
                allowed = status.allowed
                status_constraints[(code, side)] = status
            elif isinstance(status, (bool, np.bool_)):
                allowed = bool(status)
            else:
                raise ValueError("status_callback must return bool, TradingStatus or None")
            if not allowed:
                blocked_orders += 1
                events.append(TradeEvent(d, "order_blocked", code, decision_date=decision or d,
                                         note=f"{side}: status unavailable or closed"))
            return allowed

        if d in decisions or d == liquidation_decision:
            cut_marks = {}
            cut_ratios = {}
            cut_sources = {}
            for c in codes:
                if visible is None:
                    if c in marks:
                        cut_marks[c] = marks[c]
                        cut_ratios[c] = ratios.get(c, 1.0)
                        cut_sources[c] = mark_dates.get(c)
                else:
                    v = visible.iloc[row, col[c]]
                    if valid_number(v):
                        cut_marks[c] = v
                        src = source_dates.iloc[row, col[c]] if source_dates is not None else after(d, -1)
                        cut_sources[c] = pd.Timestamp(src).date() if pd.notna(src) else None
                        if not proxy_reinvest:
                            cut_ratios[c] = 1.0
                        elif visible_adj is not None:
                            a = visible_adj.iloc[row, col[c]]
                            if valid_number(a):
                                cut_ratios[c] = a / v
                        elif source_dates is None:
                            cut_ratios[c] = ratios.get(c, 1.0)
                own = private_quotes.get(c)
                if own is not None and (cut_sources.get(c) is None or own[0] >= cut_sources[c]):
                    cut_sources[c], cut_marks[c], cut_ratios[c] = own
            unknown_held = any(
                lot.fund_code not in cut_marks or lot.fund_code not in cut_ratios
                or not lot_information_known(lot, row) for lot in lots
            ) or any(not published(r.fund_code, r.fill_date, row)
                     and (config.confirmation_requires_public_nav or r.confirmation_date is None) for r in receivables)
            if unknown_held:
                blocked_orders += 1
                events.append(TradeEvent(d, "decision_blocked", note="held/fill NAV not published or visible adjustment missing"))
            else:
                if observed_dividends:
                    for c in cut_marks:
                        src = cut_sources.get(c)
                        if pd.isna(src):
                            continue
                        src = pd.Timestamp(src).date()
                        cut_marks[c] -= sum(e.per_unit_cash for e in known_dividends
                                            if e.fund_code == c and src < e.ex_date <= d)
                        if cut_marks[c] <= 0:
                            raise ValueError("announced ex-dividend adjustment exceeds visible unit NAV")
                held_value = {}
                for lot in lots:
                    relative = cut_ratios[lot.fund_code] / lot.adjustment_ratio
                    if abs(relative - 1.0) <= config.adjustment_ratio_tolerance:
                        relative = 1.0
                    visible_units = lot.units * relative
                    held_value[lot.fund_code] = held_value.get(lot.fund_code, 0.0) + visible_units * cut_marks[lot.fund_code]
                equity_cut = cash + sum(held_value.values()) + sum(r.amount for r in receivables) + dividend_value()
                held_weights = pd.Series({c: v / equity_cut for c, v in held_value.items()}, dtype=float)
                if d == liquidation_decision:
                    target = pd.Series(dtype=float)
                    # Cancel unfilled buy intents before explicit end liquidation.
                    orders = [o for o in orders if o.side != "buy"]
                elif score_callback is not None:
                    scores = score_callback(d, held_weights.copy())
                    target = (None if scores is None else scores_to_weights(scores, held_weights,
                              top_k=config.top_k, sticky_rank_buffer=config.sticky_rank_buffer))
                else:
                    target = weight_callback(d, held_weights.copy())
                if target is not None:
                    target = pd.Series(target, dtype=float)
                    target.index = target.index.map(str)
                    if (target.index.has_duplicates or not np.isfinite(target.to_numpy()).all()
                            or (target < 0).any() or target.sum() > 1.0 + _EPS):
                        raise ValueError("target weights must be unique, finite, nonnegative and sum <= 1")
                    if any(c not in col for c in target.index):
                        raise ValueError("target weights reference funds outside NAV columns")
                    for intent in funding_intents.values():
                        if intent.net_remaining > _EPS:
                            events.append(TradeEvent(d, "funding_cancelled", intent.fund_code,
                                                     decision_date=intent.decision_date,
                                                     gross_money=intent.net_remaining,
                                                     note="unfunded net increment replaced by a new target"))
                    funding_intents = {}
                    funding_epoch += 1
                    fill = after(d, config.fill_lag_sessions)
                    if fill is None or fill > sessions[-1]:
                        blocked_orders += 1
                        events.append(TradeEvent(d, "decision_blocked", note="next executable session outside simulation"))
                    else:
                        projected = dict(held_value)
                        if config.dividend_mode == "reinvest":
                            for claim in dividend_claims.values():
                                if claim.recognized and not claim.completed:
                                    c = claim.event.fund_code
                                    projected[c] = projected.get(c, 0.0) + claim.amount
                        reserved_units = {}
                        for o in orders:
                            if o.side == "buy":
                                rate = float(getattr(o.fee_schedule, "purchase_fee_rate"))
                                projected[o.fund_code] = projected.get(o.fund_code, 0.0) + subscription_amounts(o.amount, rate).net
                            else:
                                reserved_units[o.fund_code] = reserved_units.get(o.fund_code, 0.0) + o.amount
                                projected[o.fund_code] = projected.get(o.fund_code, 0.0) - o.amount * cut_marks[o.fund_code]
                        desired = {c: float(target.get(c, 0.0)) * equity_cut for c in set(projected) | set(target.index)}
                        for c in sorted(desired):
                            excess = projected.get(c, 0.0) - desired[c]
                            if excess <= _EPS or c not in cut_marks or not tradable(c, "sell"):
                                continue
                            min_hold = status_constraints.get((c, "sell"), TradingStatus(True)).min_hold_days or 0
                            available_units = sum(l.units for l in lots if l.fund_code == c and l.confirmation_date is not None
                                                  and (l.redeem_available_date is None or l.redeem_available_date <= d)
                                                  and (d - l.confirmation_date).days >= min_hold)
                            available_units -= reserved_units.get(c, 0.0)
                            amount = avoid_dust(c, min(excess / cut_marks[c], max(0.0, available_units)))
                            if amount <= _EPS:
                                continue
                            fee = schedule(c, d)
                            if fee is None:
                                blocked_orders += 1
                                events.append(TradeEvent(d, "order_blocked", c, note="redemption fee schedule unavailable"))
                                continue
                            if not redemption_supported(fee, c, amount):
                                blocked_orders += 1
                                events.append(TradeEvent(d, "order_blocked", c, decision_date=d,
                                                         units=amount, note="redemption units below contract minimum; not full redemption"))
                                continue
                            orders.append(_Order(c, "sell", d, fill, amount, fee))
                            events.append(TradeEvent(d, "sell_order", c, decision_date=d, fill_date=fill, units=amount))
                        for c in sorted(desired):
                            deficit = desired[c] - projected.get(c, 0.0)
                            if deficit <= _EPS:
                                continue
                            funding_intents[c] = _FundingIntent(c, d, deficit, funding_epoch)
                            events.append(TradeEvent(d, "funding_intent", c, decision_date=d,
                                                     gross_money=deficit, note="frozen net asset increment"))
                        new_funding_target = True

        # Execute the existing frozen signal when money becomes spendable. The
        # signal callback is not called here, and target amounts are not marked up
        # or down with subsequent market moves.
        if new_funding_target or settled_cash > _EPS:
            fill = after(d, config.fill_lag_sessions)
            free_cash = max(0.0, cash - sum(o.amount for o in orders if o.side == "buy"))
            if fill is not None and fill <= sessions[-1] and free_cash > _EPS:
                requests = []
                for c, intent in sorted(funding_intents.items()):
                    if intent.net_remaining <= _EPS or not tradable(c, "buy", intent.decision_date):
                        continue
                    fee = schedule(c, d)  # Published schedule on actual placement day.
                    if fee is None:
                        blocked_orders += 1
                        events.append(TradeEvent(d, "order_blocked", c, decision_date=intent.decision_date,
                                                 order_date=d, note="subscription fee schedule unavailable"))
                        continue
                    rate = float(getattr(fee, "purchase_fee_rate"))
                    subscription_amounts(0.0, rate)
                    gross = intent.net_remaining * (1.0 + rate)
                    cap = status_constraints.get((c, "buy"), TradingStatus(True)).purchase_limit_money
                    if cap is not None:
                        used = sum(o.amount for o in orders if o.side == "buy" and o.fund_code == c and o.order_date == d)
                        gross = min(gross, max(0.0, cap - used))
                    requests.append((c, gross, fee, intent))
                requested = sum(amount for _, amount, _, _ in requests)
                scale = min(1.0, free_cash / requested) if requested > _EPS else 0.0
                for c, amount, fee, intent in requests:
                    amount *= scale
                    if amount <= _EPS:
                        continue
                    if not purchase_supported(fee, c, amount):
                        blocked_orders += 1
                        events.append(TradeEvent(d, "order_blocked", c, decision_date=intent.decision_date,
                                                 order_date=d, gross_money=amount,
                                                 note="subscription amount below contract initial/additional minimum"))
                        continue
                    net = subscription_amounts(amount, float(getattr(fee, "purchase_fee_rate"))).net
                    intent.net_remaining = max(0.0, intent.net_remaining - net)
                    orders.append(_Order(c, "buy", intent.decision_date, fill, amount, fee,
                                         order_date=d, net_amount=net, funding_epoch=intent.epoch))
                    events.append(TradeEvent(d, "buy_order", c, decision_date=intent.decision_date,
                                             fill_date=fill, gross_money=amount, order_date=d))

        # EOD outcomes: only now may raw valuation NAV enter economic accounting.
        stale = stale_adjustment = 0
        grouped_lots = {}
        for lot in lots:
            grouped_lots.setdefault(lot.fund_code, []).append(lot)
        for c in codes:
            v, a = values[row, col[c]], adjustments[row, col[c]]
            fund_lots = grouped_lots.get(c, [])
            held = bool(fund_lots)
            if valid_number(v):
                marks[c] = float(v)
                mark_dates[c] = d
                if proxy_reinvest and valid_number(a):
                    ratio = float(a / v)
                    bonus = 0.0
                    for lot in fund_lots:
                        relative = ratio / lot.adjustment_ratio - 1.0
                        if abs(relative) <= config.adjustment_ratio_tolerance:
                            continue  # Keep the credited anchor, including for cumulative changes.
                        change = lot.units * relative
                        if change > _EPS:
                            bonus += change
                        elif change < -_EPS:
                            # Ratio reversals/revisions reduce existing shares; they
                            # are not positive dividends with a fresh holding age.
                            lot.units += change
                            events.append(TradeEvent(d, "adjustment_revision", c, units=change,
                                                     nav=float(v), lot_id=lot.lot_id))
                        lot.adjustment_ratio = ratio
                        lot.adjustment_date = d
                    if bonus > _EPS:
                        bonus_lot = FundLot(next_lot_id, c, bonus, d, after(d, 1),
                                            adjustment_ratio=ratio, fill_nav=float(v), fill_adjustment_ratio=ratio)
                        next_lot_id += 1
                        lots.append(bonus_lot)
                        events.append(TradeEvent(d, "dividend_reinvest", c, units=bonus,
                                                 nav=float(v), lot_id=bonus_lot.lot_id,
                                                 note="new bonus-share lot; zero subscription fee; fresh holding age"))
                    ratios[c] = ratio
                elif not proxy_reinvest:
                    ratios[c] = 1.0
                elif held:
                    stale_adjustment += 1
            elif held:
                stale += 1
        prefill_equity = cash + holdings_value() + sum(r.amount for r in receivables) + dividend_value()
        pending = []
        for order in orders:
            if order.fill_date != d:
                pending.append(order)
                continue
            c = order.fund_code
            v, a = values[row, col[c]], adjustments[row, col[c]]
            if (not valid_number(v) or (proxy_reinvest and not valid_number(a))):
                blocked_orders += 1
                events.append(TradeEvent(d, "order_blocked", c, decision_date=order.decision_date,
                                         fill_date=d, note=f"{order.side}: fill NAV/adjustment unavailable"))
                if order.side == "buy":
                    restore_unfilled_buy(order)
                continue
            if not tradable(c, order.side, order.decision_date):
                if order.side == "buy":
                    restore_unfilled_buy(order)
                continue
            if order.side == "buy":
                amount = min(order.amount, cash)
                covered = config.covered_daily_buy_money_exclusive
                if covered is not None and covered_buy_used.get(c, 0.0) + amount >= covered:
                    raise ValueError("daily subscription exceeds verified historical amount scope")
                covered_buy_used[c] = covered_buy_used.get(c, 0.0) + amount
                if not purchase_supported(order.fee_schedule, c, amount):
                    blocked_orders += 1
                    restore_unfilled_buy(order)
                    events.append(TradeEvent(d, "order_blocked", c, decision_date=order.decision_date,
                                             order_date=order.order_date, gross_money=amount,
                                             note="subscription amount below contract initial/additional minimum at fill"))
                    continue
                money = subscription_amounts(amount, float(getattr(order.fee_schedule, "purchase_fee_rate")))
                if money.net < order.net_amount - _EPS:
                    restore_unfilled_buy(order, order.net_amount - money.net)
                cash -= amount
                units = money.net / v
                lot = FundLot(next_lot_id, c, units, d, after(d, config.purchase_confirm_sessions),
                              adjustment_ratio=ratios.get(c, 1.0), fill_nav=float(v),
                              fill_adjustment_ratio=ratios.get(c, 1.0))
                next_lot_id += 1
                lots.append(lot)
                ever_subscribed.add(c)
                cost += money.fee
                traded += money.net
                events.append(TradeEvent(d, "buy_fill", c, order.decision_date, d, units, float(v),
                                         money.net, -amount, money.fee, lot_id=lot.lot_id,
                                         order_date=order.order_date))
                application_trades.append((c, "buy", order.order_date or order.decision_date, d, units))
            else:
                min_hold = status_constraints.get((c, "sell"), TradingStatus(True)).min_hold_days or 0
                executable = min(order.amount, sum(l.units for l in lots if l.fund_code == c
                                                   and l.confirmation_date is not None
                                                   and (l.redeem_available_date is None or l.redeem_available_date <= d)
                                                   and (d - l.confirmation_date).days >= min_hold))
                executable = avoid_dust(c, executable)
                if executable <= _EPS or not redemption_supported(order.fee_schedule, c, executable):
                    blocked_orders += 1
                    events.append(TradeEvent(d, "order_blocked", c, decision_date=order.decision_date,
                                             units=executable, note="redemption units below contract minimum/unavailable at fill"))
                    continue
                confirmation = after(d, config.redemption_confirm_sessions)
                if confirmation is None:
                    blocked_orders += 1
                    events.append(TradeEvent(d, "order_blocked", c, decision_date=order.decision_date,
                                             fill_date=d, note="redemption confirmation outside supplied calendar"))
                    continue
                remaining = executable
                for lot in sorted(lots, key=lambda l: (l.confirmation_date or date.max, l.lot_id)):
                    if (lot.fund_code != c or lot.confirmation_date is None or remaining <= _EPS
                            or (lot.redeem_available_date is not None and lot.redeem_available_date > d)
                            or (d - lot.confirmation_date).days < min_hold):
                        continue
                    units = min(lot.units, remaining)
                    hold_days = (confirmation - lot.confirmation_date).days
                    rate = float(order.fee_schedule.redemption_rate(hold_days))
                    money = redemption_amounts(units * v, rate)
                    lot.units -= units
                    remaining -= units
                    cost += money.fee
                    traded += money.gross
                    receivables.append(CashReceivable(
                        c, d, after(d, config.redemption_settlement_sessions), money.net,
                        confirmation_due_date=confirmation, purchase_confirmation_date=lot.confirmation_date,
                        gross_money=money.gross, fee_money=money.fee, fee_schedule=order.fee_schedule,
                        lot_id=lot.lot_id, fill_nav=float(v), fill_adjustment_ratio=ratios.get(c, 1.0),
                    ))
                    events.append(TradeEvent(d, "sell_fill", c, order.decision_date, d, units, float(v),
                                             money.gross, 0.0, money.fee, hold_days, lot.lot_id))
                    application_trades.append((c, "sell", order.order_date or order.decision_date, d, units))
        orders = pending
        lots = [l for l in lots if l.units > _EPS]
        if observed_dividends:
            for c in codes:
                registry_units[(c, d)] = sum(l.units for l in lots if l.fund_code == c and l.confirmation_date is not None)
            # Cutoff valuations use only14:30 announcements; EOD accounting may
            # also use notices released after the cutoff in the same session.
            closing_dividends = dividend_events.as_of(datetime.combine(d, time(23, 59, 59)))
            refresh_dividends(closing_dividends, d)
            for claim in dividend_claims.values():
                if not claim.recognized or claim.completed or claim.amount <= _EPS:
                    continue
                event = claim.event
                if config.dividend_mode == "cash":
                    if event.cash_pay_date is None:
                        raise ValueError(f"cash dividend {event.event_id} lacks observed payment date")
                    if event.cash_pay_date <= d:
                        cash += claim.amount
                        dividend_cash_arrival += claim.amount
                        claim.completed = True
                        events.append(TradeEvent(d, "dividend_cash_pay", event.fund_code,
                                                 cash_money=claim.amount, note=event.event_id))
                else:
                    if any(x is None for x in (event.reinvest_price_date, event.confirmation_date,
                                               event.redeem_available_date)):
                        raise ValueError(f"reinvestment dividend {event.event_id} lacks observed lifecycle dates")
                    if event.reinvest_price_date > d:
                        continue
                    price = event.reinvest_price
                    if price is None:
                        if event.reinvest_price_date not in raw.index:
                            raise ValueError("observed reinvestment pricing date missing from raw NAV")
                        price = float(raw.loc[event.reinvest_price_date, event.fund_code])
                    if not valid_number(price):
                        raise ValueError("observed reinvestment pricing NAV unavailable")
                    units = claim.amount / price
                    lot = FundLot(next_lot_id, event.fund_code, units, event.reinvest_price_date,
                                  event.confirmation_date, adjustment_ratio=1.0,
                                  redeem_available_date=event.redeem_available_date,
                                  confirmation_date_fixed=True, fill_nav=float(price), fill_adjustment_ratio=1.0)
                    if event.confirmation_date <= d:
                        lot.confirmation_date = event.confirmation_date
                    next_lot_id += 1
                    lots.append(lot)
                    claim.completed = True
                    events.append(TradeEvent(d, "dividend_reinvest", event.fund_code,
                                             fill_date=event.reinvest_price_date, units=units, nav=price,
                                             gross_money=claim.amount, lot_id=lot.lot_id,
                                             note=f"observed {event.event_id}; no subscription fee; new confirmation age"))
        held = holdings_value()
        receivable_value = sum(r.amount for r in receivables) + dividend_value()
        reserved = sum(o.amount for o in orders if o.side == "buy")
        equity = cash + held + receivable_value
        if cash < -1e-7 or reserved > cash + 1e-7 or any(l.units < -_EPS for l in lots):
            raise AssertionError("execution borrowed cash or shorted fund units")
        if d == sessions[-1]:
            for intent in funding_intents.values():
                if intent.net_remaining > _EPS:
                    events.append(TradeEvent(d, "funding_expired", intent.fund_code,
                                             decision_date=intent.decision_date,
                                             gross_money=intent.net_remaining,
                                             note="unfunded net increment expired at simulation/seal boundary"))
            funding_intents = {}
        if d < simulation_start:
            continue  # Warmup marks do not inflate the reported investment horizon.
        records.append({"date": d, "cash": cash, "available_cash": max(0.0, cash - reserved),
                        "reserved_cash": reserved, "holdings_value": held, "receivables": receivable_value,
                        "unfunded_buy_net_money": sum(i.net_remaining for i in funding_intents.values()),
                        "equity": equity, "cost_money": cost, "traded_gross_money": traded,
                        "turnover": traded / (2.0 * prefill_equity) if prefill_equity > 0 else 0.0,
                        "stale_holdings": stale, "stale_adjustments": stale_adjustment,
                        "pending_estimated_redemption_fees": sum(r.confirmation_date is None for r in receivables),
                        "dividend_receivables": dividend_value(),
                        "status_unknown_orders": unknown_status, "blocked_orders": blocked_orders})
    return BacktestResult(pd.DataFrame(records).set_index("date"), events, lots, receivables,
                          config.initial_cash, assumptions, list(dividend_claims.values()))
