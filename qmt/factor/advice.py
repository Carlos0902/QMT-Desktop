"""规则化调仓提示 — 每期必给结论(含「不建议调仓」),成本敏感,参数透明。

定位(合规):这是**个人研究工具的规则化输出**,由用户显式采用的主因子
(user_override)+ 赎回费档位 + 仓位规模三组规则推导;所有阈值written在
模块顶部、可审计;条件式表述(「若遵循已采用因子的纪律」),不是收益承诺,
也不是持牌投顾建议。用户明确要求每次 shadow-step 后强制输出。

规则(全部透明):
  1. 减持候选:截面分位 < REDUCE_BELOW 且仓位 >= MIN_WEIGHT_TO_ACT 的持仓;
     每条附赎回费档位(按真实持有天数)与「再等 X 天费率降档」提示。
  2. 保持:分位 >= HOLD_ABOVE,或处于中间带(变动的成本收益比不划算)。
  3. 微仓不动:仓位 < MIN_WEIGHT_TO_ACT 的,调整意义不大。
  4. 未排名持仓:明确写「无法评估,不给建议」。
  5. 现金部署:现金占比 > CASH_DEPLOY_ABOVE 时,列出观察组合打分最高、
     且未持有的前 N 只作为候选(示例性分批金额);同时写明保留现金的理由
     (主因子未通过全部验证门槛)。
  6. 若无任何减持候选且现金不超带:输出「本期不建议调仓」+ 原因。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from qmt.pit import PointInTimeStore

__all__ = ["rebalance_note", "rebalance_plan", "REDUCE_BELOW", "HOLD_ABOVE",
           "MIN_WEIGHT_TO_ACT", "CASH_DEPLOY_ABOVE", "N_CANDIDATES"]

REDUCE_BELOW = 0.30        # 分位低于此 → 减持候选
HOLD_ABOVE = 0.55          # 分位高于此 → 明确保持
MIN_WEIGHT_TO_ACT = 0.02   # 仓位低于 2% 的不值得动
CASH_DEPLOY_ABOVE = 0.20   # 现金占比高于此才谈部署
N_CANDIDATES = 5           # 现金部署候选数
# 赎回费档位(与 qmt.fees 缺省一致):(不足天数, 费率)
_FEE_TIERS = [(7, 0.015), (30, 0.005), (365, 0.0025)]


def _fee_for(hold_days: int) -> tuple[float, str]:
    """(当前费率, 降档提示)。"""
    for cap, rate in _FEE_TIERS:
        if hold_days < cap:
            nxt = [(c, r) for c, r in _FEE_TIERS if c > cap]
            nxt_rate = nxt[0][1] if nxt else 0.0
            tip = f"再持有 {cap - hold_days} 天费率降为 {nxt_rate:.2%}" if cap - hold_days > 0 else ""
            return rate, tip
    return 0.0, ""


def _names(store: PointInTimeStore, codes: list[str]) -> dict[str, str]:
    try:
        fb = store.get_asof("fund_basic", pd.Timestamp("2100-01-01"))
        m = dict(zip(fb["fund_code"].astype(str), fb["name"].astype(str)))
        return {c: m.get(c, "") for c in codes}
    except Exception:
        return {c: "" for c in codes}


def rebalance_note(snap: dict[str, Any], store: PointInTimeStore) -> str:
    """由快照生成本期调仓提示;没有持仓账本时给出说明。"""
    hr = snap.get("held_review")
    if not hr:
        return "(未初始化持仓账本 — 无法给出针对你的调仓提示;先 portfolio-init。)"
    total = hr.get("total_value") or 0.0
    cash_w = hr.get("cash_weight") or 0.0
    primary = snap.get("primary") or "观察名单综合分"

    reduce_list: list[str] = []
    keep_list: list[str] = []
    tiny_list: list[str] = []
    unranked: list[str] = []

    positions = hr.get("positions") or {}
    held_codes = list(positions)
    name_of = _names(store, held_codes + list(snap.get("watch_top") or []))

    for fc, r in sorted(positions.items(), key=lambda kv: -(kv[1].get("weight") or 0)):
        w = r.get("weight") or 0.0
        p = r.get("watchlist_percentile")
        hd = int(r.get("hold_days") or 0)
        label = f"{fc} {name_of.get(fc, '')}".strip()
        if p is None:
            unranked.append(f"  {label}(权重 {w:.1%})—— 未参与排名,无法评估,不给建议")
            continue
        if w < MIN_WEIGHT_TO_ACT:
            tiny_list.append(f"  {label}(权重仅 {w:.1%},分位 {p:.0%})—— 仓位太小,调整意义不大")
            continue
        if p < REDUCE_BELOW:
            rate, tip = _fee_for(hd)
            half = total * w * 0.5
            fee = half * rate
            line = (f"  {label}(权重 {w:.1%},分位 {p:.0%} < {REDUCE_BELOW:.0%} 阈值)\n"
                    f"    若按因子纪律:考虑减持约一半(≈¥{half:,.0f}),"
                    f"赎回费 {rate:.2%}(持有 {hd} 天)≈¥{fee:,.0f}")
            if tip:
                line += f";{tip}"
            reduce_list.append(line)
        elif p >= HOLD_ABOVE:
            keep_list.append(f"  {label}(分位 {p:.0%})—— 保持")
        else:
            keep_list.append(f"  {label}(分位 {p:.0%},中间带)—— 保持;变动的成本收益比不划算")

    lines: list[str] = []
    lines.append(f"━━━ 本期调仓提示(依据你采用的「{primary}」;规则化输出,非持牌投顾建议)━━━")

    do_deploy = cash_w > CASH_DEPLOY_ABOVE
    if not reduce_list and not do_deploy:
        lines.append("本期结论:**不建议调仓** —— 没有持仓触发减持阈值,现金占比也在常态带内;")
        lines.append("按因子纪律,不动是本期成本最低的选择。")
    else:
        verdict = []
        if reduce_list:
            verdict.append(f"{len(reduce_list)} 只触发减持阈值")
        if do_deploy:
            verdict.append(f"现金占比 {cash_w:.0%} 高于 {CASH_DEPLOY_ABOVE:.0%}(可考虑分批部署)")
        lines.append("本期结论:小幅调整 —— " + ";".join(verdict))

    if reduce_list:
        lines.append("")
        lines.append("【减持候选】")
        lines.extend(reduce_list)
    if keep_list:
        lines.append("")
        lines.append("【保持不动】")
        lines.extend(keep_list)
    if tiny_list:
        lines.extend(tiny_list)
    if unranked:
        lines.extend(unranked)

    if do_deploy:
        held = set(held_codes)
        cands = [c for c in (snap.get("watch_top") or []) if c not in held][:N_CANDIDATES]
        if cands:
            cash_amt = total * cash_w
            tranche = round(cash_amt * 0.5 / max(1, min(3, len(cands))) / 100) * 100
            lines.append("")
            lines.append(f"【现金部署(可选)】现金约 ¥{cash_amt:,.0f}({cash_w:.0%})")
            lines.append(f"  观察组合(按 {primary})打分最高、且你未持有的前 {len(cands)} 只:")
            for c in cands:
                nm = name_of.get(c, "")
                lines.append(f"    {c} {nm}".rstrip())
            lines.append(f"  示例节奏:先部署现金的一半,分 2–3 批、每批 ≈¥{tranche:,.0f};")
            lines.append("  若你不认同该因子的有效性,继续保留现金同样合理(它未通过全部验证门槛)。")

    lines.append("")
    lines.append(f"(规则参数:减持阈值 {REDUCE_BELOW:.0%} / 保持线 {HOLD_ABOVE:.0%} / "
                 f"微仓线 {MIN_WEIGHT_TO_ACT:.0%} / 现金带 {CASH_DEPLOY_ABOVE:.0%};全部可调,输出为条件式提示,非收益承诺。)")
    return "\n".join(lines)


def explain_plan(plan: dict[str, Any], store: PointInTimeStore) -> str:
    """把可执行订单渲染成人能读的中文方案。"""
    names = _names(store, [x["fund"] for x in plan["sells"] + plan["keeps"]
                          + plan["skipped"] + plan["topups"] + plan["new_candidates"]])
    L = [f"━━━ 强制调仓方案 @ {plan['as_of']}（按你采用的「{plan['primary']}」，规则化，非投顾建议）━━━",
         f"总资产 ¥{plan['total_value']:,.0f} · 现金 ¥{plan['cash']:,.0f}"]
    L.append("")
    if plan["sells"]:
        L.append("【卖出】")
        for s in plan["sells"]:
            nm = names.get(s["fund"], "")
            L.append(f"  {s['action']}  {s['fund']} {nm[:18]}  ¥{s['sell_amount']:,.0f}"
                     f"（分位 {s['pctile']:.0%}，持有 {s['hold_days']} 天，赎回费 {s['fee_rate']:.2%}≈¥{s['fee']:,.0f}）")
        L.append(f"  卖出合计 ¥{plan['total_sell_amount']:,.0f}，赎回费合计 ≈¥{plan['total_redemption_fee']:,.0f}")
    else:
        L.append("【卖出】无 —— 没有持仓跌破减持阈值。")
    L.append("")
    if plan["topups"]:
        L.append(f"【买入 / 补仓】可部署资金 ≈¥{plan['deployable']:,.0f}（释放现金 + 原有现金的一半）")
        for t in plan["topups"]:
            nm = names.get(t["fund"], "")
            L.append(f"  补仓  {t['fund']} {nm[:18]}  ¥{t['buy_amount']:,.0f}（分位 {t['pctile']:.0%}，你已持有的高分基金）")
    elif plan["new_candidates"]:
        L.append(f"【买入候选】可部署 ≈¥{plan['deployable']:,.0f} —— 你的持仓里没有够高分的可补，因子给出的场外候选：")
        for c in plan["new_candidates"]:
            nm = names.get(c["fund"], "")
            warn = f" ⚠{c['warn']}" if c.get("warn") else ""
            L.append(f"  {c['fund']} {nm[:20]}（{c['type']}）{warn}")
        L.append("  （买入陌生产品前请自查申赎规则与锁定期;不认同因子有效性则保留现金同样合理。）")
    else:
        L.append("【买入】无可部署现金或无合适标的。")
    L.append("")
    if plan["keeps"]:
        L.append("【保持不动】" + "、".join(f"{k['fund']}({k['pctile']:.0%})" for k in plan["keeps"]))
    for sk in plan["skipped"]:
        L.append(f"  {sk['fund']} {names.get(sk['fund'],'')[:16]} —— {sk['reason']}（不动）")
    L.append("")
    L.append(f"本次换手约 {plan['turnover_pct']:.0%}，卖出后现金 ¥{plan['cash_after_sells']:,.0f}。")
    L.append("提示:low_vol_60 仍未通过平台验证门槛(NO-GO);以上是「若贯彻该因子纪律」的规则化动作，"
             "不是收益承诺,也非持牌投顾建议。节奏建议:季频(约每 3 个月)复看一次。")
    return "\n".join(L)


def _fund_types(store: PointInTimeStore, codes: list[str]) -> dict[str, str]:
    try:
        fb = store.get_asof("fund_basic", pd.Timestamp("2100-01-01"))
        return dict(zip(fb["fund_code"].astype(str), fb["fund_type"].astype(str)))
    except Exception:
        return {}


def _redeem_rate(hold_days: int) -> float:
    for cap, rate in _FEE_TIERS:
        if hold_days < cap:
            return rate
    return 0.0


def rebalance_plan(snap: dict[str, Any], store: PointInTimeStore,
                   reduce_full_below: float = 0.20,
                   reduce_half_below: float = REDUCE_BELOW,
                   deploy_frac: float = 0.5) -> dict[str, Any]:
    """把「减持哪只、买入多少」落成**具体到元**的可执行订单(规则化,非投顾建议)。

    纪律(与验证过的季频+粘性一致):
      * 分位 < ``reduce_full_below``(明显最差)→ 全卖;< ``reduce_half_below`` → 减半;
        >= ``HOLD_ABOVE`` 保持;中间带保持(不值得为小幅差异付换手);
      * 未排名 / 微仓 → 明确不动;
      * 释放的现金 + 原有现金的 ``deploy_frac``,按 low_vol 打分优先**补仓已持有的
        高分基金**(保守、避免买入不熟悉产品);仅当没有可补的高分持仓时,才列出
        因子的场外新候选(并标注货币/持有期基金风险)。
    """
    hr = snap.get("held_review") or {}
    total = hr.get("total_value") or 0.0
    cash = (hr.get("cash_weight") or 0.0) * total
    positions = hr.get("positions") or {}
    ftypes = _fund_types(store, list(positions) + list(snap.get("watch_top") or []))

    sells: list[dict] = []
    keeps: list[dict] = []
    skipped: list[dict] = []
    freed = 0.0
    for fc, r in sorted(positions.items(), key=lambda kv: (kv[1].get("watchlist_percentile") or -1)):
        w = r.get("weight") or 0.0
        p = r.get("watchlist_percentile")
        hd = int(r.get("hold_days") or 0)
        value = w * total
        if p is None:
            skipped.append({"fund": fc, "value": value, "reason": "未排名,无法评估"})
            continue
        if w < MIN_WEIGHT_TO_ACT:
            skipped.append({"fund": fc, "value": value, "reason": f"微仓 {w:.1%},不动"})
            continue
        if p < reduce_full_below or p < reduce_half_below:
            frac = 1.0 if p < reduce_full_below else 0.5
            amt = value * frac
            rate = _redeem_rate(hd)
            fee = amt * rate
            freed += amt - fee
            sells.append({"fund": fc, "pctile": p, "sell_amount": round(amt, 0),
                          "fee_rate": rate, "fee": round(fee, 0), "hold_days": hd,
                          "action": "全部赎回" if frac == 1.0 else "赎回一半"})
        else:
            keeps.append({"fund": fc, "pctile": p, "value": round(value, 0)})

    deployable = freed + cash * deploy_frac
    # 优先补仓已持有的高分基金(分位 >= HOLD_ABOVE),等额分配
    topups: list[dict] = []
    good = [k for k in keeps if k["pctile"] >= HOLD_ABOVE]
    if deployable > 0 and good:
        each = deployable / len(good)
        for k in good:
            topups.append({"fund": k["fund"], "pctile": k["pctile"],
                           "buy_amount": round(each, 0), "type": ftypes.get(k["fund"], "")})
    # 新候选(仅当没有可补的高分持仓时列出,带风险标注)
    new_candidates: list[dict] = []
    if deployable > 0 and not good:
        held = set(positions)
        for fc in (snap.get("watch_top") or []):
            if fc in held:
                continue
            t = ftypes.get(fc, "")
            new_candidates.append({"fund": fc, "type": t,
                                   "warn": ("货币/持有期类,买入≈持有现金" if ("货币" in t or "债券" in t) else "")})
            if len(new_candidates) >= 5:
                break

    total_sell = sum(s["sell_amount"] for s in sells)
    total_fee = sum(s["fee"] for s in sells)
    return {
        "as_of": snap.get("as_of"),
        "primary": snap.get("primary"),
        "total_value": round(total, 0), "cash": round(cash, 0),
        "sells": sells, "keeps": keeps, "skipped": skipped,
        "topups": topups, "new_candidates": new_candidates,
        "total_sell_amount": round(total_sell, 0), "total_redemption_fee": round(total_fee, 0),
        "cash_after_sells": round(cash + freed, 0),
        "deployable": round(deployable, 0),
        "turnover_pct": round(total_sell / total, 3) if total else 0.0,
    }
