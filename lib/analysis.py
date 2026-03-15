"""
KNOAH 거래 분석 – 데이터 집계 및 Claude API 호출
"""

import json
import pandas as pd
import numpy as np
from typing import Optional


def aggregate_data(trades: pd.DataFrame, deposits: pd.DataFrame,
                   exchange: str) -> dict:
    """거래 데이터를 Claude에 보낼 요약 통계로 변환"""
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

    # 최대 드로다운
    cumulative = df.sort_values("datetime")["net_pnl"].cumsum()
    running_max = cumulative.cummax()
    max_dd = round(float((cumulative - running_max).min()), 2)

    initial_balance = float(deposits["amount_usdt"].sum()) if not deposits.empty else 10_000
    max_dd_pct = round(abs(max_dd) / initial_balance, 3) if initial_balance > 0 else 0

    # 종목별
    per_symbol = []
    for sym, grp in df.groupby("symbol"):
        sw = grp[grp["pnl_usdt"] > 0]
        per_symbol.append({
            "symbol": str(sym),
            "trades": int(len(grp)),
            "pnl_usdt": round(float(grp["net_pnl"].sum()), 2),
            "win_rate": round(len(sw) / len(grp), 2) if len(grp) > 0 else 0,
            "avg_holding_minutes": int(grp["holding_minutes"].mean()),
            "avg_leverage": round(float(grp["leverage"].mean()), 1),
        })
    per_symbol.sort(key=lambda x: x["pnl_usdt"])

    # 시간대별
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name().str[:3]
    weekday_pnl = {}
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        day_df = df[df["weekday"] == day]
        weekday_pnl[day] = round(float(day_df["net_pnl"].sum()), 2) if len(day_df) > 0 else 0

    most_active = df.groupby("hour").size().nlargest(4).index.tolist()

    # 행동 패턴
    market_orders = int(len(df[df["order_type"] == "MARKET"]))
    stoploss_set = int(len(df[df["stoploss_set"] == True]))

    # 복수매매 감지
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

    # 포지션 사이즈
    notional = df["quantity_usdt"] * df["leverage"]
    avg_pos_pct = round(float(notional.mean() / initial_balance), 2) if initial_balance > 0 else 0
    max_pos_pct = round(float(notional.max() / initial_balance), 2) if initial_balance > 0 else 0

    # 입출금
    dep_df = deposits if not deposits.empty else pd.DataFrame(columns=["type", "amount_usdt"])
    total_dep = float(dep_df[dep_df["type"] == "DEPOSIT"]["amount_usdt"].sum()) if len(dep_df) > 0 else 0
    total_wd = float(dep_df[dep_df["type"] == "WITHDRAWAL"]["amount_usdt"].sum()) if len(dep_df) > 0 else 0

    return {
        "exchange": exchange,
        "analysis_period": f"{df['datetime'].min().strftime('%Y-%m-%d')} ~ {df['datetime'].max().strftime('%Y-%m-%d')}",
        "account_type": "USDT-M Futures",
        "summary": {
            "total_trades": int(n),
            "total_pnl_usdt": total_pnl,
            "total_fee_usdt": total_fee,
            "win_rate": float(win_rate),
            "avg_win_usdt": round(float(wins["pnl_usdt"].mean()), 2) if len(wins) > 0 else 0,
            "avg_loss_usdt": round(float(losses["pnl_usdt"].mean()), 2) if len(losses) > 0 else 0,
            "profit_factor": float(profit_factor),
            "max_drawdown_usdt": max_dd,
            "max_drawdown_pct": float(max_dd_pct),
            "avg_leverage": round(float(df["leverage"].mean()), 1),
            "max_leverage_used": int(df["leverage"].max()),
            "initial_balance_usdt": round(initial_balance, 2),
            "final_balance_usdt": round(initial_balance + total_pnl, 2),
        },
        "per_symbol": per_symbol,
        "time_pattern": {
            "most_active_hours_utc": [int(h) for h in most_active],
            "weekday_pnl": weekday_pnl,
        },
        "behavior": {
            "market_order_ratio": round(market_orders / n, 2) if n > 0 else 0,
            "stoploss_usage_ratio": round(stoploss_set / n, 2) if n > 0 else 0,
            "revenge_trade_count": revenge_count,
            "avg_position_size_pct": float(avg_pos_pct),
            "max_position_size_pct": float(max_pos_pct),
            "avg_trades_per_day": float(avg_trades_day),
            "max_trades_in_day": max_trades_day,
            "dca_down_count": dca_count,
        },
        "deposit_withdrawal": {
            "total_deposits_usdt": round(total_dep, 2),
            "total_withdrawals_usdt": round(total_wd, 2),
        },
    }


# ─────────────────────────────────────────────────
# 섹션별 인라인 AI 코멘트
# ─────────────────────────────────────────────────
SECTION_PROMPT = """당신은 KNOAH 플랫폼의 트레이딩 분석 AI입니다.
유저의 거래 데이터를 보고, 대시보드의 각 섹션에 표시할 짧은 인사이트 코멘트를 작성합니다.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

{
  "overview": "핵심 지표 전체를 한줄로 진단 (예: 승률은 양호하나 수수료가 수익의 40%를 잠식 중)",
  "exchange_comparison": "거래소간 비교 코멘트 (거래소가 1개면 해당 거래소 특징)",
  "equity_curve": "누적 손익 곡선의 흐름을 한줄로 진단 (예: 중반 드로다운 후 회복세, 하락 추세 등)",
  "symbol_pnl": "종목별 손익 패턴 한줄 진단 (예: BTC에서 수익을 내지만 알트에서 반납)",
  "symbol_winrate": "종목별 승률 패턴 한줄 진단 (예: DOGE 승률 20%인데 계속 거래 중 → 손절 기준 필요)",
  "weekday_pnl": "요일별 패턴 한줄 진단 (예: 주말 거래에서 집중적 손실 발생)",
  "hourly_pattern": "시간대별 패턴 한줄 진단 (예: 새벽 거래 수익률 낮음, 오후에 집중하는 것이 유리)",
  "pnl_distribution": "PnL 분포 한줄 진단 (예: 소액 수익을 자주 내지만 대규모 손실이 전체를 압도)",
  "action_items": ["바로 실행 가능한 행동 제안 1", "제안 2", "제안 3"]
}

## 규칙
- 한국어로 작성
- 각 코멘트는 반드시 1줄 (최대 60자)
- action_items는 2~3개, 각각 구체적이고 실행 가능한 행동
- 수치를 인용하여 구체적으로 작성
- "$" 기호 대신 "USD" 사용
- 부드러운 제안 톤
- 투자 조언이 아닌 매매 습관 진단
"""


def call_claude_sections(aggregated: dict, api_key: str) -> dict:
    """Claude API로 섹션별 코멘트 생성"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = f"""아래 거래 데이터를 보고 각 섹션별 인사이트를 JSON으로 작성해주세요.

```json
{json.dumps(aggregated, ensure_ascii=False, indent=2, default=str)}
```"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SECTION_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = message.content[0].text.strip()
    # JSON 블록 추출
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def generate_dummy_comments(aggregated: dict) -> dict:
    """API 키 없을 때 더미 섹션별 코멘트 생성"""
    s = aggregated["summary"]
    b = aggregated["behavior"]
    symbols = aggregated["per_symbol"]

    # 최악/최고 종목
    worst = symbols[0] if symbols else {"symbol": "N/A", "pnl_usdt": 0}
    best = symbols[-1] if symbols else {"symbol": "N/A", "pnl_usdt": 0}

    # 승률 기반 진단
    if s["win_rate"] < 0.4:
        ov = f"승률 {s['win_rate']*100:.0f}%로 낮은 편 — 손절 기준과 진입 타이밍 재점검 필요"
    elif s["total_pnl_usdt"] < 0:
        ov = f"승률 {s['win_rate']*100:.0f}%는 양호하지만, 평균 손실이 평균 수익보다 커서 순손실 구조"
    else:
        ov = f"수익 팩터 {s['profit_factor']}로 안정적이나, 수수료 {s['total_fee_usdt']:,.0f} USD가 수익을 깎는 중"

    # 누적 PnL 방향
    if s["total_pnl_usdt"] < -500:
        eq = f"전체 기간 하락 추세 — 최대 드로다운 {s['max_drawdown_pct']*100:.0f}% 기록"
    elif s["total_pnl_usdt"] < 0:
        eq = f"등락 반복 후 소폭 손실 마감 — 드로다운 구간에서 회복력 부족"
    else:
        eq = f"전반적 우상향 추세 — 다만 드로다운 {s['max_drawdown_pct']*100:.0f}% 구간 주의"

    return {
        "overview": ov,
        "exchange_comparison": f"평균 레버리지 {s['avg_leverage']}x, 최대 {s['max_leverage_used']}x — 리스크 관리 점검 필요",
        "equity_curve": eq,
        "symbol_pnl": f"{best['symbol']}에서 +{best['pnl_usdt']:,.0f} USD 수익, {worst['symbol']}에서 {worst['pnl_usdt']:,.0f} USD 손실 집중",
        "symbol_winrate": f"{worst['symbol']} 승률이 낮은데 거래 비중이 높음 — 해당 종목 거래 축소 검토",
        "weekday_pnl": "주말 거래 손실 비중이 높다면 평일 집중 전략이 유리할 수 있음",
        "hourly_pattern": f"주요 거래 시간대에 손익이 집중 — 컨디션 좋은 시간대로 거래 한정 추천",
        "pnl_distribution": f"평균 수익 {s['avg_win_usdt']:,.0f} USD vs 평균 손실 {s['avg_loss_usdt']:,.0f} USD — 손익비 개선 여지",
        "action_items": [
            f"레버리지를 {min(10, s['max_leverage_used'])}x 이하로 제한해보기",
            f"스탑로스 설정률 {b['stoploss_usage_ratio']*100:.0f}% → 100%로 올리기",
            f"복수매매 {b['revenge_trade_count']}회 발생 — 손실 후 30분 쿨다운 규칙 적용",
        ],
    }
