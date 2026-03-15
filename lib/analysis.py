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

    # 최대 드로다운 (최고 자산 대비 하락률)
    initial_balance = float(deposits["amount_usdt"].sum()) if not deposits.empty else 10_000
    equity = initial_balance + df.sort_values("datetime")["net_pnl"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = round(float(drawdown.min()), 2)  # 최대 드로다운 금액 (음수)
    max_dd_pct = round(abs(max_dd) / float(peak[drawdown.idxmin()]), 3) if float(peak[drawdown.idxmin()]) > 0 else 0

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
SECTION_PROMPT = """당신은 10년 경력의 퀀트 트레이더 겸 트레이딩 코치입니다.
유저의 실제 선물 거래 데이터를 분석하여, 대시보드의 각 섹션에 표시할 인사이트를 작성합니다.

## 분석 원칙
1. 반드시 데이터의 **구체적 수치**를 인용 (종목명, 금액, %, 횟수 등)
2. "좋다/나쁘다" 같은 모호한 표현 금지 — **왜** 문제인지, **어떤 수치가** 근거인지 명시
3. 손익비(RR ratio), 수익 팩터, 드로다운, 승률 간의 **관계**를 분석
4. 거래 빈도 + 레버리지 + 스탑로스 설정률의 **조합**으로 리스크를 진단
5. "$" 기호 대신 "USD" 사용

## 출력 형식
반드시 아래 JSON만 출력하세요. 다른 텍스트 없이 순수 JSON만.

{
  "overview": "전체 매매 스타일을 한 문장으로 진단. 반드시 핵심 수치 2개 이상 포함",
  "exchange_comparison": "거래소별 성과 비교 또는 단일 거래소 특징. 레버리지/승률/손익 수치 포함",
  "equity_curve": "자산 곡선 흐름 진단. 드로다운 구간, 회복 패턴, 추세 방향 구체적 언급",
  "symbol_pnl": "어떤 종목에서 수익이 나고 어떤 종목에서 까먹는지 종목명+금액으로 콕 집기",
  "symbol_winrate": "승률이 특히 낮거나 높은 종목을 지목하고, 거래 횟수 대비 효율성 진단",
  "weekday_pnl": "어떤 요일에 수익/손실이 집중되는지 구체적 요일+금액 언급",
  "hourly_pattern": "수익이 나는 시간대 vs 손실이 나는 시간대 구체적 비교",
  "pnl_distribution": "평균 수익 vs 평균 손실 금액 비교, 꼬리 리스크(큰 손실) 언급",
  "action_items": [
    "가장 임팩트 큰 개선 행동 (수치 근거 포함)",
    "두번째 행동 (수치 근거 포함)",
    "세번째 행동 (수치 근거 포함)"
  ]
}
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
    """API 키 없을 때 데이터 기반 더미 섹션별 코멘트 생성"""
    s = aggregated["summary"]
    b = aggregated["behavior"]
    symbols = aggregated["per_symbol"]
    tp = aggregated.get("time_pattern", {})
    wpnl = tp.get("weekday_pnl", {})

    # 종목 정렬 (pnl 기준 오름차순 — [0]이 최악)
    worst = symbols[0] if symbols else {"symbol": "?", "pnl_usdt": 0, "win_rate": 0, "trades": 0}
    best = symbols[-1] if symbols else {"symbol": "?", "pnl_usdt": 0, "win_rate": 0, "trades": 0}

    # 손익비 계산
    avg_win = abs(s["avg_win_usdt"]) if s["avg_win_usdt"] else 1
    avg_loss = abs(s["avg_loss_usdt"]) if s["avg_loss_usdt"] else 1
    rr_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0
    fee_pct = round(s["total_fee_usdt"] / max(abs(s["total_pnl_usdt"] + s["total_fee_usdt"]), 1) * 100, 0)

    # ── overview ─────────────────────────────────
    if s["win_rate"] < 0.4 and s["total_pnl_usdt"] < 0:
        ov = (f"승률 {s['win_rate']*100:.0f}%, 손익비 1:{rr_ratio} — "
              f"{s['total_trades']}건 중 {int(s['total_trades']*(1-s['win_rate']))}건이 손실, 진입 기준 재점검 필요")
    elif s["profit_factor"] < 1:
        ov = (f"수익팩터 {s['profit_factor']}(1 미만=순손실) — "
              f"평균 수익 {avg_win:,.0f} USD vs 평균 손실 {avg_loss:,.0f} USD, 손절 늦는 패턴")
    elif fee_pct > 30:
        ov = (f"승률 {s['win_rate']*100:.0f}%로 양호하지만 수수료가 총수익의 {fee_pct:.0f}%를 잠식 중 — "
              f"레버리지 {s['avg_leverage']}x 낮추면 수수료 절감 가능")
    else:
        ov = (f"수익팩터 {s['profit_factor']}, 승률 {s['win_rate']*100:.0f}% — "
              f"MDD {s['max_drawdown_pct']*100:.1f}% 관리하면서 수익 구조 유지 중")

    # ── equity_curve ─────────────────────────────
    dd_pct = s["max_drawdown_pct"] * 100
    dd_usd = abs(s["max_drawdown_usdt"])
    final = s["final_balance_usdt"]
    initial = s["initial_balance_usdt"]
    if s["total_pnl_usdt"] < -initial * 0.1:
        eq = f"자산이 {initial:,.0f} → {final:,.0f} USD로 감소, MDD -{dd_pct:.1f}%({dd_usd:,.0f} USD) — 하락 추세"
    elif s["total_pnl_usdt"] < 0:
        eq = f"등락 반복 후 {abs(s['total_pnl_usdt']):,.0f} USD 소폭 손실 마감, MDD -{dd_pct:.1f}% 구간에서 회복 부족"
    else:
        eq = f"{initial:,.0f} → {final:,.0f} USD(+{s['total_pnl_usdt']:,.0f}), MDD -{dd_pct:.1f}% 거쳐 회복"

    # ── exchange_comparison ──────────────────────
    exc = (f"평균 레버리지 {s['avg_leverage']}x(최대 {s['max_leverage_used']}x), "
           f"스탑로스 설정률 {b['stoploss_usage_ratio']*100:.0f}% — "
           f"{'고위험 세팅' if s['avg_leverage'] > 15 else '중위험 세팅' if s['avg_leverage'] > 8 else '보수적 세팅'}")

    # ── symbol_pnl ───────────────────────────────
    sym_pnl = (f"{best['symbol']} +{best['pnl_usdt']:,.0f} USD({best['trades']}건) vs "
               f"{worst['symbol']} {worst['pnl_usdt']:,.0f} USD({worst['trades']}건) — "
               f"{'알트코인 손실이 BTC 수익을 상쇄' if worst['pnl_usdt'] < -best['pnl_usdt'] else '수익 종목이 손실을 커버 중'}")

    # ── symbol_winrate ───────────────────────────
    low_wr = [s for s in symbols if s["win_rate"] < 0.35 and s["trades"] >= 5]
    if low_wr:
        lw = low_wr[0]
        sym_wr = f"{lw['symbol']} 승률 {lw['win_rate']*100:.0f}%인데 {lw['trades']}회 거래 — 해당 종목 거래 비중 줄이거나 전략 변경"
    else:
        sym_wr = f"전체 종목 승률 편차가 크지 않으나, {worst['symbol']}의 손익비가 가장 불리"

    # ── weekday_pnl ──────────────────────────────
    if wpnl:
        worst_day = min(wpnl, key=wpnl.get)
        best_day = max(wpnl, key=wpnl.get)
        wd = f"{worst_day} {wpnl[worst_day]:,.0f} USD(최악) vs {best_day} +{wpnl[best_day]:,.0f} USD(최고) — {worst_day} 거래 축소 검토"
    else:
        wd = "요일별 데이터 부족"

    # ── hourly_pattern ───────────────────────────
    active_hrs = tp.get("most_active_hours_utc", [])
    if active_hrs:
        hr = f"거래 집중 시간 {active_hrs[0]}~{active_hrs[-1]}시(UTC) — 해당 시간대 외 거래는 충동매매 가능성 체크"
    else:
        hr = "시간대 데이터 부족"

    # ── pnl_distribution ─────────────────────────
    if rr_ratio < 1:
        dist = f"평균 수익 {avg_win:,.0f} < 평균 손실 {avg_loss:,.0f} USD(손익비 {rr_ratio}) — 손실이 수익보다 커서 승률로 커버 불가"
    else:
        dist = f"평균 수익 {avg_win:,.0f} > 평균 손실 {avg_loss:,.0f} USD(손익비 {rr_ratio}) — 큰 손실 건만 관리하면 수익 구조 유지"

    # ── action_items ─────────────────────────────
    actions = []
    if s["avg_leverage"] > 10:
        actions.append(f"레버리지 평균 {s['avg_leverage']}x → 10x 이하로 제한 시 MDD {dd_pct:.0f}% 절반 이상 감소 기대")
    if b["stoploss_usage_ratio"] < 0.5:
        actions.append(f"스탑로스 설정률 {b['stoploss_usage_ratio']*100:.0f}% → 모든 포지션에 -2~3% SL 필수 적용")
    if b["revenge_trade_count"] > 3:
        actions.append(f"복수매매 {b['revenge_trade_count']}회 감지 — 연속 2회 손실 시 30분 강제 휴식 규칙 적용")
    if low_wr:
        lw = low_wr[0]
        actions.append(f"{lw['symbol']} 승률 {lw['win_rate']*100:.0f}% — 해당 종목 한달간 거래 중단 후 재평가")
    if b["max_trades_in_day"] > 15:
        actions.append(f"일 최대 {b['max_trades_in_day']}건 과매매 — 하루 8건 상한 설정")
    if len(actions) < 2:
        actions.append("매매 일지 작성 시작 — 진입 근거와 감정 상태 기록으로 패턴 파악")
    actions = actions[:3]

    return {
        "overview": ov,
        "exchange_comparison": exc,
        "equity_curve": eq,
        "symbol_pnl": sym_pnl,
        "symbol_winrate": sym_wr,
        "weekday_pnl": wd,
        "hourly_pattern": hr,
        "pnl_distribution": dist,
        "action_items": actions,
        ],
    }
