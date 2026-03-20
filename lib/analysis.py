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
        "analysis_period": f"{df['datetime'].min().strftime('%Y-%m-%d')} - {df['datetime'].max().strftime('%Y-%m-%d')}",
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
    df["hold_band"] = pd.cut(df["holding_minutes"], bins=[0, 10, 1440, 43200, 99999],
                             labels=["스캘핑(<10분)", "데이트레이딩(10분-1일)", "스윙(1일-30일)", "장기(30일+)"])
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
# 상세 분석 요약 (Haiku)
# ─────────────────────────────────────────────────
DETAIL_SUMMARY_PROMPT = """당신은 트레이딩 데이터 분석가입니다.
아래 상세 분석 데이터를 보고 3-4문장으로 핵심만 요약하세요.

규칙:
- "$" 대신 "USD", "~" 대신 "-" 사용
- 뻔한 소리 금지. 유저가 차트를 보면 알 수 있는 건 생략.
- 교차 분석에서 발견되는 비직관적 패턴 위주.
- 구체적 숫자 포함 (종목명, 승률, 금액).
- 한국어. 짧은 문장.
"""


def generate_detail_summary(basic_stats: dict, deep_data: dict, api_key: str) -> dict:
    """상세 분석 탭 상단 AI 요약 (Haiku)"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = f"""### 기본 통계
```json
{json.dumps(basic_stats, ensure_ascii=False, indent=1, default=str)}
```
### 교차 분석
```json
{json.dumps(deep_data, ensure_ascii=False, indent=1, default=str)}
```"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=DETAIL_SUMMARY_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    usage = message.usage
    cost = usage.input_tokens * 0.80 / 1_000_000 + usage.output_tokens * 4 / 1_000_000

    return {
        "summary": message.content[0].text,
        "cost_usd": round(cost, 6),
    }


# ─────────────────────────────────────────────────
# Claude 액션 플랜 (레거시, 현재 미사용)
# ─────────────────────────────────────────────────
ACTION_PLAN_PROMPT = """당신은 트레이딩 코치입니다.
유저는 이미 대시보드에서 차트와 지표를 다 봤습니다.
당신은 데이터에서 사람이 놓치기 쉬운 교차 패턴을 찾아서 구체적 행동을 제안하는 역할입니다.

## 글쓰기 규칙
- "~"(물결표) 사용 금지. 범위는 "-" 사용 (예: 3-5개)
- "$" 대신 "USD" 사용
- 중학생도 알아듣게. 전문용어 금지.
- 짧은 문장. 한 문장에 한 가지만.
- 뻔한 소리 금지 (승률 올려라, 레버리지 줄여라, 멘탈 관리해라)
- 비슷한 수치를 "큰 차이"라고 과장하지 마세요.

## 출력 형식 (마크다운)
정확히 3개의 행동을 제안하세요. 각각:

### 1. [한줄 제목]
- **근거**: 데이터에서 발견한 구체적 패턴 (종목명, 숫자 포함)
- **행동**: 뭘 어떻게 바꾸라는 건지 (구체적으로)
- **설정법**: 거래소에서 어떻게 세팅하는지

뻔한 패턴 말고, 교차 분석(종목x방향, 종목x요일, 시간대, 연패 후 행동 등)에서
사람이 직접 발견하기 어려운 것 위주로.
"""


def call_claude_action_plan(basic_stats: dict, deep_data: dict, api_key: str) -> dict:
    """Claude API로 액션 플랜 생성 (3개 구체적 행동)"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = f"""아래 데이터를 보고 구체적 행동 3개를 제안해주세요.

### 기본 통계
```json
{json.dumps(basic_stats, ensure_ascii=False, indent=2, default=str)}
```

### 교차 분석
```json
{json.dumps(deep_data, ensure_ascii=False, indent=2, default=str)}
```"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=ACTION_PLAN_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    usage = message.usage
    cost = usage.input_tokens * 3 / 1_000_000 + usage.output_tokens * 15 / 1_000_000

    return {
        "report": message.content[0].text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost, 4),
    }


def get_api_balance(api_key: str) -> dict:
    """Anthropic API 크레딧 잔고 조회"""
    import requests
    try:
        # Admin API로 잔고 조회 시도
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        # billing endpoint는 admin key만 가능, 일반 key는 실패할 수 있음
        resp = requests.get("https://api.anthropic.com/v1/billing/credit_balance",
                           headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {"ok": True, "balance": data.get("balance", data)}
    except Exception:
        pass
    return {"ok": False}


# ─────────────────────────────────────────────────
# AI 챗 (자연어 거래 질문)
# ─────────────────────────────────────────────────
CHAT_SYSTEM = """당신은 트레이딩 데이터 분석 어시스턴트입니다.
유저의 거래 데이터가 JSON으로 주어집니다. 유저의 질문에 데이터를 기반으로 답하세요.

규칙:
- 짧고 직접적으로 답하세요. 2-4문장.
- 숫자를 인용하세요 (날짜, 금액, 승률 등).
- "$" 대신 "USD" 사용.
- "~" 대신 "-" 사용.
- 데이터에 없는 건 "데이터에 없습니다"라고 하세요. 추측 금지.
- 한국어로 답하세요.
"""


def chat_with_data(question: str, basic_stats: dict, deep_data: dict,
                   chat_history: list, api_key: str) -> dict:
    """거래 데이터 기반 자연어 질문 응답 (Haiku)"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    data_context = f"""### 거래 요약
```json
{json.dumps(basic_stats, ensure_ascii=False, indent=1, default=str)}
```

### 교차 분석
```json
{json.dumps(deep_data, ensure_ascii=False, indent=1, default=str)}
```"""

    messages = [{"role": "user", "content": f"내 거래 데이터:\n{data_context}"}]
    messages.append({"role": "assistant", "content": "데이터를 확인했습니다. 질문해주세요."})

    # 이전 대화 이어서
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 현재 질문
    messages.append({"role": "user", "content": question})

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=CHAT_SYSTEM,
        messages=messages,
    )

    usage = message.usage
    # Haiku: input $0.80/MTok, output $4/MTok
    cost = usage.input_tokens * 0.80 / 1_000_000 + usage.output_tokens * 4 / 1_000_000

    return {
        "answer": message.content[0].text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost, 6),
    }
