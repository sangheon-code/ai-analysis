"""
KNOAH 거래 분석 – 데이터 집계 및 Claude AI 딥 리포트
"""

import json
import pandas as pd
import numpy as np
from typing import Optional


def aggregate_data(trades: pd.DataFrame, deposits: pd.DataFrame,
                   exchange: str) -> dict:
    """거래 데이터를 요약 통계로 변환"""
    if trades.empty:
        return {}

    df = trades.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["net_pnl"] = df["pnl_usdt"] - df["fee_usdt"]

    n = len(df)
    wins = df[df["pnl_usdt"] > 0]
    losses = df[df["pnl_usdt"] <= 0]

    total_pnl = round(float(df["net_pnl"].sum()), 2)
    total_fee = round(float(df["fee_usdt"].sum()), 2)
    win_rate = round(len(wins) / n, 3) if n > 0 else 0

    gross_profit = float(wins["pnl_usdt"].sum()) if len(wins) > 0 else 0
    gross_loss = abs(float(losses["pnl_usdt"].sum())) if len(losses) > 0 else 1
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

    initial_balance = float(deposits["amount_usdt"].sum()) if not deposits.empty else 10_000
    equity = initial_balance + df.sort_values("datetime")["net_pnl"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = round(float(drawdown.min()), 2)
    max_dd_pct = round(abs(max_dd) / float(peak[drawdown.idxmin()]), 3) if float(peak[drawdown.idxmin()]) > 0 else 0

    per_symbol = []
    for sym, grp in df.groupby("symbol"):
        sw = grp[grp["pnl_usdt"] > 0]
        per_symbol.append({
            "symbol": str(sym), "trades": int(len(grp)),
            "pnl_usdt": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(len(sw) / len(grp), 2) if len(grp) > 0 else 0,
            "avg_holding_minutes": int(grp["holding_minutes"].mean()),
            "avg_leverage": round(float(grp["leverage"].mean()), 1),
        })
    per_symbol.sort(key=lambda x: x["pnl_usdt"])

    df["hour"] = df["datetime"].dt.hour
    df["hour_kst"] = (df["hour"] + 9) % 24
    df["weekday"] = df["datetime"].dt.day_name().str[:3]
    weekday_pnl = {}
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        day_df = df[df["weekday"] == day]
        weekday_pnl[day] = round(float(day_df["net_pnl"].sum()), 2) if len(day_df) > 0 else 0

    most_active = df.groupby("hour_kst").size().nlargest(4).index.tolist()

    market_orders = int(len(df[df["order_type"] == "MARKET"]))
    stoploss_set = int(len(df[df["stoploss_set"] == True]))

    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    revenge_count = 0
    dca_count = 0
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i - 1]
        curr = df_sorted.iloc[i]
        gap = (curr["datetime"] - prev["datetime"]).total_seconds() / 60
        if prev["pnl_usdt"] < 0 and gap <= 5:
            revenge_count += 1
        if (prev["symbol"] == curr["symbol"] and prev["side"] == curr["side"]
                and prev["pnl_usdt"] < 0 and gap <= 30):
            dca_count += 1

    day_groups = df.groupby(df["datetime"].dt.date).size()
    max_trades_day = int(day_groups.max()) if len(df) > 0 else 0
    avg_trades_day = round(float(day_groups.mean()), 1) if len(df) > 0 else 0

    notional = df["quantity_usdt"] * df["leverage"]
    avg_pos_pct = round(float(notional.mean() / initial_balance), 2) if initial_balance > 0 else 0
    max_pos_pct = round(float(notional.max() / initial_balance), 2) if initial_balance > 0 else 0

    dep_df = deposits if not deposits.empty else pd.DataFrame(columns=["type", "amount_usdt"])
    total_dep = float(dep_df[dep_df["type"] == "DEPOSIT"]["amount_usdt"].sum()) if len(dep_df) > 0 else 0
    total_wd = float(dep_df[dep_df["type"] == "WITHDRAWAL"]["amount_usdt"].sum()) if len(dep_df) > 0 else 0

    return {
        "exchange": exchange,
        "analysis_period": f"{df['datetime'].min().strftime('%Y-%m-%d')} ~ {df['datetime'].max().strftime('%Y-%m-%d')}",
        "account_type": "USDT-M Futures",
        "summary": {
            "total_trades": int(n), "total_pnl_usdt": total_pnl, "total_fee_usdt": total_fee,
            "win_rate": float(win_rate),
            "avg_win_usdt": round(float(wins["pnl_usdt"].mean()), 2) if len(wins) > 0 else 0,
            "avg_loss_usdt": round(float(losses["pnl_usdt"].mean()), 2) if len(losses) > 0 else 0,
            "profit_factor": float(profit_factor),
            "max_drawdown_usdt": max_dd, "max_drawdown_pct": float(max_dd_pct),
            "avg_leverage": round(float(df["leverage"].mean()), 1),
            "max_leverage_used": int(df["leverage"].max()),
            "initial_balance_usdt": round(initial_balance, 2),
            "final_balance_usdt": round(initial_balance + total_pnl, 2),
        },
        "per_symbol": per_symbol,
        "time_pattern": {"most_active_hours_kst": [int(h) for h in most_active], "weekday_pnl": weekday_pnl},
        "behavior": {
            "market_order_ratio": round(market_orders / n, 2) if n > 0 else 0,
            "stoploss_usage_ratio": round(stoploss_set / n, 2) if n > 0 else 0,
            "revenge_trade_count": revenge_count,
            "avg_position_size_pct": float(avg_pos_pct), "max_position_size_pct": float(max_pos_pct),
            "avg_trades_per_day": float(avg_trades_day), "max_trades_in_day": max_trades_day,
            "dca_down_count": dca_count,
        },
        "deposit_withdrawal": {"total_deposits_usdt": round(total_dep, 2), "total_withdrawals_usdt": round(total_wd, 2)},
    }


# ─────────────────────────────────────────────────
# 딥 리포트용 다변수 교차 분석 데이터
# ─────────────────────────────────────────────────
def aggregate_deep_data(trades: pd.DataFrame, deposits: pd.DataFrame) -> dict:
    """Claude 딥 리포트를 위한 교차 분석 데이터 생성"""
    if trades.empty:
        return {}

    df = trades.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["net_pnl"] = df["pnl_usdt"] - df["fee_usdt"]
    df["hour_kst"] = (df["datetime"].dt.hour + 9) % 24
    df["weekday"] = df["datetime"].dt.day_name().str[:3]
    df["is_win"] = df["pnl_usdt"] > 0

    result = {}

    # ── 1. 종목 × 방향 교차 분석 (상위/하위 5개) ────
    sym_side = []
    for (sym, side), grp in df.groupby(["symbol", "side"]):
        if len(grp) < 3:
            continue
        sym_side.append({
            "symbol": sym, "side": side, "trades": int(len(grp)),
            "win_rate": round(len(grp[grp["is_win"]]) / len(grp), 2),
            "pnl": round(float(grp["net_pnl"].sum()), 2),
            "avg_leverage": round(float(grp["leverage"].mean()), 1),
        })
    sym_side.sort(key=lambda x: x["pnl"])
    result["symbol_side_best"] = sym_side[-3:] if len(sym_side) >= 3 else sym_side
    result["symbol_side_worst"] = sym_side[:3]

    # ── 2. 종목 × 요일 교차 (손실 top 5) ────────────
    sym_day = []
    for (sym, day), grp in df.groupby(["symbol", "weekday"]):
        if len(grp) < 2:
            continue
        sym_day.append({
            "symbol": sym, "weekday": day, "trades": int(len(grp)),
            "pnl": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(len(grp[grp["is_win"]]) / len(grp), 2),
        })
    sym_day.sort(key=lambda x: x["pnl"])
    result["symbol_weekday_worst5"] = sym_day[:5]
    result["symbol_weekday_best5"] = sym_day[-5:] if len(sym_day) >= 5 else sym_day[-3:]

    # ── 3. 시간대별(KST) 수익/손실 구간 ─────────────
    hour_stats = []
    for hr, grp in df.groupby("hour_kst"):
        if len(grp) < 2:
            continue
        hour_stats.append({
            "hour_kst": int(hr), "trades": int(len(grp)),
            "pnl": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(len(grp[grp["is_win"]]) / len(grp), 2),
        })
    hour_stats.sort(key=lambda x: x["pnl"])
    result["hour_worst3"] = hour_stats[:3]
    result["hour_best3"] = hour_stats[-3:] if len(hour_stats) >= 3 else hour_stats

    # ── 4. 레버리지 구간별 성과 ──────────────────────
    df["lev_band"] = pd.cut(df["leverage"], bins=[0, 5, 10, 20, 200],
                            labels=["1-5x", "6-10x", "11-20x", "21x+"])
    lev_stats = []
    for band, grp in df.groupby("lev_band", observed=True):
        if len(grp) < 2:
            continue
        lev_stats.append({
            "leverage_band": str(band), "trades": int(len(grp)),
            "win_rate": round(len(grp[grp["is_win"]]) / len(grp), 2),
            "pnl": round(float(grp["net_pnl"].sum()), 2),
            "avg_pnl": round(float(grp["net_pnl"].mean()), 2),
        })
    result["leverage_bands"] = lev_stats

    # ── 5. 보유 시간별 성과 ──────────────────────────
    df["hold_band"] = pd.cut(df["holding_minutes"], bins=[0, 10, 60, 360, 99999],
                             labels=["스캘핑(<10분)", "단타(10-60분)", "스윙(1-6시간)", "장기(6시간+)"])
    hold_stats = []
    for band, grp in df.groupby("hold_band", observed=True):
        if len(grp) < 2:
            continue
        hold_stats.append({
            "hold_band": str(band), "trades": int(len(grp)),
            "win_rate": round(len(grp[grp["is_win"]]) / len(grp), 2),
            "pnl": round(float(grp["net_pnl"].sum()), 2),
        })
    result["holding_time_bands"] = hold_stats

    # ── 6. 연속 패턴 분석 (시퀀스) ───────────────────
    ds = df.sort_values("datetime").reset_index(drop=True)

    # 복수매매 상세
    revenge_trades = []
    for i in range(1, len(ds)):
        prev, curr = ds.iloc[i - 1], ds.iloc[i]
        gap = (curr["datetime"] - prev["datetime"]).total_seconds() / 60
        if prev["pnl_usdt"] < 0 and gap <= 5:
            revenge_trades.append({
                "pnl": round(float(curr["net_pnl"]), 2),
                "leverage_change": int(curr["leverage"]) - int(prev["leverage"]),
                "size_change_pct": round((float(curr["quantity_usdt"]) / max(float(prev["quantity_usdt"]), 1) - 1) * 100, 0),
            })
    if revenge_trades:
        rt_df = pd.DataFrame(revenge_trades)
        result["revenge_trading"] = {
            "count": len(rt_df),
            "total_pnl": round(float(rt_df["pnl"].sum()), 2),
            "avg_pnl": round(float(rt_df["pnl"].mean()), 2),
            "escalated_pct": round(len(rt_df[rt_df["size_change_pct"] > 10]) / len(rt_df) * 100, 0),
        }
    else:
        result["revenge_trading"] = {"count": 0, "total_pnl": 0, "avg_pnl": 0, "escalated_pct": 0}

    # 연패 후 행동
    streak = 0
    post_tilt = []
    for i in range(len(ds)):
        if ds.iloc[i]["pnl_usdt"] <= 0:
            streak += 1
        else:
            if streak >= 3 and i < len(ds) - 1:
                nxt = ds.iloc[i]
                post_tilt.append({
                    "streak_len": streak,
                    "next_leverage": int(nxt["leverage"]),
                    "next_pnl": round(float(nxt["net_pnl"]), 2),
                })
            streak = 0
    result["post_tilt_behavior"] = {
        "tilt_count": len(post_tilt),
        "avg_next_pnl": round(float(np.mean([t["next_pnl"] for t in post_tilt])), 2) if post_tilt else 0,
        "avg_next_leverage": round(float(np.mean([t["next_leverage"] for t in post_tilt])), 1) if post_tilt else 0,
    }

    # 연승 후 과매매
    win_streak = 0
    post_win_streak = []
    for i in range(len(ds)):
        if ds.iloc[i]["pnl_usdt"] > 0:
            win_streak += 1
        else:
            if win_streak >= 3 and i < len(ds) - 1:
                nxt = ds.iloc[i]
                post_win_streak.append({
                    "streak_len": win_streak,
                    "next_leverage": int(nxt["leverage"]),
                    "next_pnl": round(float(nxt["net_pnl"]), 2),
                })
            win_streak = 0
    result["post_win_streak"] = {
        "count": len(post_win_streak),
        "avg_next_pnl": round(float(np.mean([t["next_pnl"] for t in post_win_streak])), 2) if post_win_streak else 0,
    }

    # ── 7. 수익 거래 vs 손실 거래 보유 시간 비교 ─────
    result["holding_comparison"] = {
        "avg_win_hold_min": round(float(df[df["is_win"]]["holding_minutes"].mean()), 1) if len(df[df["is_win"]]) > 0 else 0,
        "avg_loss_hold_min": round(float(df[~df["is_win"]]["holding_minutes"].mean()), 1) if len(df[~df["is_win"]]) > 0 else 0,
    }

    return result


# ─────────────────────────────────────────────────
# Claude 딥 리포트
# ─────────────────────────────────────────────────
DEEP_REPORT_PROMPT = """당신은 15년 경력의 퀀트 트레이더이자 트레이딩 심리 코치입니다.
유저의 선물 거래 데이터를 분석하여, 기계적 규칙으로는 발견할 수 없는 인사이트를 제공합니다.

## 데이터 구조
- basic_stats: 기본 요약 통계 (이건 유저도 이미 대시보드에서 보고 있음)
- deep_data: 다변수 교차 분석 결과 (이게 핵심 — 여기서 패턴을 찾아야 함)

## 리포트 구조 (마크다운)

### 매매 스타일 진단
유저의 트레이딩 스타일을 분류하세요 (스캘퍼/단타/스윙/혼합).
보유 시간 데이터, 거래 빈도, 레버리지 패턴을 교차해서 판단.
스타일이 일관적인지, 아니면 이랬다저랬다 하는지도 진단.

### 숨은 패턴 발견
종목×방향, 종목×요일, 시간대별 교차 데이터에서 유저가 모를 만한 패턴을 찾으세요.
예: "BTC SHORT는 승률 70%인데 DOGE LONG은 12% — 같은 사람 맞나 싶을 정도"
예: "화요일 오전에만 집중적으로 손실 — 이 시간대에 뭔가 감정적 트리거가 있을 수 있음"
최소 3개, 최대 5개. 뻔한 건 빼고 의외인 것만.

### 심리 패턴 분석
복수매매, 연패 후 행동, 연승 후 과신 데이터를 보고 트레이딩 심리를 진단.
- 복수매매 시 포지션 크기가 커지는지 (에스컬레이션)
- 연패 후 레버리지를 올리는지
- 연승 후 방심해서 큰 손실이 나는지
- 수익 거래는 빨리 끊고 손실 거래는 오래 들고 있는지 (disposition effect)
구체적 수치 인용하되, 심리학적 용어로 이름을 붙여주세요.

### 맞춤 전략 제안
위 분석을 종합해서 이 유저만을 위한 구체적 전략을 제안하세요.
3개의 규칙을 만들어주세요. 각 규칙은:
- 무엇을: 구체적 행동 (종목명, 시간, 레버리지 수치 포함)
- 왜: 위 분석의 어느 데이터가 근거인지
- 어떻게: 거래소에서 실제로 어떻게 설정하는지
반드시 실행 가능해야 합니다. "마인드셋을 바꿔라" 같은 추상적 조언 금지.

## 규칙
- 한국어
- "$" 대신 "USD" 사용
- 뻔한 얘기 금지 (승률 낮으면 올려라, 레버리지 줄여라 같은 건 유저도 암)
- 데이터에서 발견한 비직관적인 패턴에 집중
- 거래가 30건 미만이면 "데이터가 부족합니다" 대신 있는 데이터에서 최대한 분석
"""


def call_claude_deep_report(basic_stats: dict, deep_data: dict, api_key: str) -> str:
    """Claude API로 딥 리포트 생성"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = f"""아래 거래 데이터를 분석해서 딥 리포트를 작성해주세요.

### 기본 통계 (유저가 이미 대시보드에서 보는 것)
```json
{json.dumps(basic_stats, ensure_ascii=False, indent=2, default=str)}
```

### 교차 분석 데이터 (여기서 패턴을 찾아주세요)
```json
{json.dumps(deep_data, ensure_ascii=False, indent=2, default=str)}
```"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=DEEP_REPORT_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text
