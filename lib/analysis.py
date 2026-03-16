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

## 핵심 규칙: 모든 코멘트는 반드시 "진단 → 처방" 구조
- 앞부분: 수치를 근거로 현상 진단 (1줄)
- 뒷부분: "→" 바로 이어서 구체적으로 뭘 해야 하는지 처방 (1줄)
- "—"(em dash) 사용 금지. 진단과 처방 사이는 "→" 하나만 사용
- 처방은 "~해보세요", "~하세요" 같은 직접적 행동 지시
- 모호한 "검토", "점검", "고려" 금지. "OO을 XX로 바꾸세요", "OO을 중단하세요" 같이 구체적으로.
- "$" 기호 대신 "USD" 사용

## 출력 형식
반드시 아래 JSON만 출력하세요. 다른 텍스트 없이 순수 JSON만.

{
  "overview": "[수치 진단] → [구체적 처방]",
  "exchange_comparison": "[거래소별 수치 비교] → [어느 거래소에서 뭘 바꿔야 하는지]",
  "equity_curve": "[곡선 흐름 진단] → [자산 관리 관점에서 뭘 해야 하는지]",
  "symbol_pnl": "[종목+금액 진단] → [어떤 종목을 줄이고 어떤 종목에 집중할지]",
  "symbol_winrate": "[승률 낮은 종목 지목] → [해당 종목 거래를 어떻게 할지]",
  "weekday_pnl": "[요일별 패턴] → [어떤 요일에 거래하고 어떤 요일은 쉴지]",
  "hourly_pattern": "[시간대별 패턴] → [언제 거래하고 언제 피할지]",
  "pnl_distribution": "[손익비 진단] → [익절/손절 기준을 구체적으로 어떻게 설정할지]",
  "action_items": [
    "내일부터 바로 할 수 있는 행동 1 (수치 포함, ~하세요 형태)",
    "이번 주 안에 할 행동 2 (수치 포함, ~하세요 형태)",
    "한달 동안 지킬 규칙 3 (수치 포함, ~하세요 형태)"
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

    # ── overview: 진단 → 처방 ─────────────────────
    if s["win_rate"] < 0.4 and s["total_pnl_usdt"] < 0:
        ov = (f"승률 {s['win_rate']*100:.0f}%에 손익비 {rr_ratio} "
              f"→ 승률 45% 이상 나오는 종목만 골라서 거래하세요. 나머지는 당장 끊으세요")
    elif s["profit_factor"] < 1:
        ov = (f"수익팩터 {s['profit_factor']}(1 미만=벌어도 까먹는 구조) "
              f"→ 손절 라인을 진입가 -{min(3, 100//max(s['avg_leverage'],1)):.0f}%에 무조건 걸고, 절대 옮기지 마세요")
    elif fee_pct > 30:
        ov = (f"수수료가 총수익의 {fee_pct:.0f}%를 먹고 있음 "
              f"→ 레버리지를 {s['avg_leverage']}x에서 {max(3, s['avg_leverage']//2)}x로 반으로 줄이세요. 수수료가 절반 됩니다")
    else:
        ov = (f"수익팩터 {s['profit_factor']}, 승률 {s['win_rate']*100:.0f}% "
              f"→ 현재 전략 유지하되, MDD {s['max_drawdown_pct']*100:.1f}% 넘으면 포지션 사이즈를 절반으로 줄이세요")

    # ── equity_curve: 진단 → 처방 ────────────────
    dd_pct = s["max_drawdown_pct"] * 100
    dd_usd = abs(s["max_drawdown_usdt"])
    final = s["final_balance_usdt"]
    initial = s["initial_balance_usdt"]
    if s["total_pnl_usdt"] < -initial * 0.1:
        eq = (f"{initial:,.0f} → {final:,.0f} USD, MDD -{dd_pct:.1f}% "
              f"→ 지금 전략은 돈을 잃는 구조입니다. 2주간 거래를 멈추고 복기하세요")
    elif s["total_pnl_usdt"] < 0:
        eq = (f"MDD -{dd_pct:.1f}% 후 회복 못 함 "
              f"→ 드로다운 {dd_pct/2:.0f}% 넘으면 당일 거래 중단하는 규칙을 만드세요")
    else:
        eq = (f"+{s['total_pnl_usdt']:,.0f} USD 수익, MDD -{dd_pct:.1f}% "
              f"→ 수익 중이지만 MDD 구간에서 포지션을 50% 줄였으면 회복이 더 빨랐을 겁니다")

    # ── exchange_comparison: 진단 → 처방 ─────────
    if s['avg_leverage'] > 15:
        exc = (f"평균 {s['avg_leverage']}x 레버리지에 스탑로스 {b['stoploss_usage_ratio']*100:.0f}% "
               f"→ 이건 도박입니다. 레버리지 10x 이하 + 모든 포지션 SL 필수로 바꾸세요")
    elif b['stoploss_usage_ratio'] < 0.3:
        exc = (f"스탑로스 설정률 {b['stoploss_usage_ratio']*100:.0f}%밖에 안 됨 "
               f"→ 진입 전에 SL 먼저 설정하고, SL 없으면 진입하지 마세요. 예외 없이")
    else:
        exc = (f"평균 레버리지 {s['avg_leverage']}x, SL 설정률 {b['stoploss_usage_ratio']*100:.0f}% "
               f"→ SL을 100%까지 올리고, 레버리지는 {max(3, int(s['avg_leverage']*0.7))}x로 낮추세요")

    # ── symbol_pnl: 진단 → 처방 ──────────────────
    sym_pnl = (f"{worst['symbol']}에서 {worst['pnl_usdt']:,.0f} USD 날림({worst['trades']}건) "
               f"→ {worst['symbol']} 거래를 당장 중단하세요. 대신 {best['symbol']}({best['trades']}건, +{best['pnl_usdt']:,.0f} USD)에 집중하세요")

    # ── symbol_winrate: 진단 → 처방 ──────────────
    low_wr = [sym for sym in symbols if sym["win_rate"] < 0.35 and sym["trades"] >= 5]
    if low_wr:
        lw = low_wr[0]
        sym_wr = (f"{lw['symbol']} 승률 {lw['win_rate']*100:.0f}%로 {lw['trades']}번 거래 = 돈 버리는 중 "
                  f"→ 이 종목은 한달간 거래 금지. 대신 승률 높은 종목 1~2개만 하세요")
    else:
        sym_wr = (f"{worst['symbol']} 승률이 가장 낮음 "
                  f"→ 종목을 3개 이하로 줄이세요. 많이 하면 집중력이 분산됩니다")

    # ── weekday_pnl: 진단 → 처방 ────────────────
    if wpnl:
        worst_day = min(wpnl, key=wpnl.get)
        best_day = max(wpnl, key=wpnl.get)
        wd = (f"{worst_day}에 {wpnl[worst_day]:,.0f} USD 손실 집중, {best_day}에 +{wpnl[best_day]:,.0f} USD "
              f"→ {worst_day}은 거래하지 마세요. {best_day}에 평소보다 포지션 10% 늘려보세요")
    else:
        wd = "요일별 데이터 부족 → 최소 2주 이상 데이터를 모아주세요"

    # ── hourly_pattern: 진단 → 처방 ──────────────
    active_hrs = tp.get("most_active_hours_utc", [])
    if active_hrs:
        hr = (f"주요 거래 시간 {active_hrs[0]}~{active_hrs[-1]}시(UTC) "
              f"→ 이 시간대에만 거래하고, 그 외 시간은 차트를 아예 끄세요. 새벽 충동매매가 계좌를 깎습니다")
    else:
        hr = "시간대 데이터 부족 → 최소 2주 이상 데이터를 모아주세요"

    # ── pnl_distribution: 진단 → 처방 ────────────
    if rr_ratio < 1:
        target_sl = max(1, int(avg_loss * 0.7))
        dist = (f"평균 수익 {avg_win:,.0f} < 평균 손실 {avg_loss:,.0f} USD, 벌어도 까먹는 구조 "
                f"→ 손절 라인을 {target_sl:,} USD로 줄이세요. 그러면 손익비가 1 이상 됩니다")
    else:
        dist = (f"평균 수익 {avg_win:,.0f} > 평균 손실 {avg_loss:,.0f} USD(손익비 {rr_ratio}) "
                f"→ 좋은 구조입니다. 익절을 {int(avg_win*1.2):,} USD까지 늘려보세요")

    # ── action_items: 당장 실행 가능한 행동 ──────
    actions = []
    if s["avg_leverage"] > 10:
        actions.append(f"오늘부터: 레버리지 상한을 {max(5, int(s['avg_leverage']*0.5))}x로 설정하세요. 거래소 설정에서 바로 바꿀 수 있습니다")
    if b["stoploss_usage_ratio"] < 0.5:
        sl_pct = min(3, 100 // max(int(s['avg_leverage']), 1))
        actions.append(f"오늘부터: 포지션 진입 전 -{sl_pct}% 스탑로스를 먼저 설정하세요. SL 없는 진입은 금지")
    if b["revenge_trade_count"] > 3:
        actions.append(f"이번 주부터: 2연패하면 30분 쉬세요. 복수매매 {b['revenge_trade_count']}회가 손실의 주범입니다")
    if low_wr:
        lw = low_wr[0]
        actions.append(f"오늘부터: {lw['symbol']} 거래를 중단하세요. 승률 {lw['win_rate']*100:.0f}%짜리에 돈 넣는 건 기부입니다")
    if b["max_trades_in_day"] > 15:
        actions.append(f"이번 주부터: 하루 거래 8건 넘으면 앱을 끄세요. {b['max_trades_in_day']}건은 과매매입니다")
    if worst["pnl_usdt"] < -500:
        actions.append(f"이번 달: {worst['symbol']}을 아예 관심 종목에서 빼세요. {worst['pnl_usdt']:,.0f} USD 손실의 원흉입니다")
    if rr_ratio < 1 and b["stoploss_usage_ratio"] >= 0.5:
        actions.append(f"이번 주부터: 익절 목표를 평균 손실({avg_loss:,.0f} USD)의 1.5배인 {int(avg_loss*1.5):,} USD로 설정하세요")
    if len(actions) < 2:
        actions.append("오늘부터: 매매 일지를 쓰세요. 진입 이유와 그때 기분을 적으면 패턴이 보입니다")
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
    }
