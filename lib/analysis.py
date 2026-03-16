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

    df["hour_kst"] = (df["hour"] + 9) % 24
    most_active = df.groupby("hour_kst").size().nlargest(4).index.tolist()  # KST 기준

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
            "most_active_hours_kst": [int(h) for h in most_active],
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

## 핵심 규칙: 모든 코멘트는 "한줄 진단 → 처방" 구조
- 앞부분: 현상을 짧게 진단. 수치는 차트에 있으니 최소한만
- 뒷부분: "→" 바로 이어서 뭘 해야 하는지 직접 행동 지시
- "검토", "점검", "고려" 금지. "~하세요", "~중단하세요" 사용
- "—"(em dash) 사용 금지. "→" 하나만 사용
- 짧고 임팩트 있게 1~2줄
- 각 섹션은 반드시 서로 다른 관점의 처방을 제시. 같은 말을 반복하지 말 것
  - overview: 전체 구조 진단 (수익팩터, 승률)
  - exchange_comparison: 거래소 세팅 (레버리지, SL)
  - equity_curve: 자산 관리 (포지션 사이징, 드로다운 대응)
  - symbol_pnl: 종목 선택 (집중/제외)
  - symbol_winrate: 종목별 전략 (승률 낮은 종목 대응)
  - weekday_pnl: 거래 스케줄 (요일별)
  - hourly_pattern: 거래 시간 (시간대별, KST 기준)
  - pnl_distribution: SL/TP 비율 설정
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
  "hourly_pattern": "[시간대별 패턴, KST 기준] → [언제 거래하고 언제 피할지]",
  "pnl_distribution": "[손익비 진단] → [SL:TP 비율을 구체적 숫자로 어떻게 설정할지]",
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

    # ── overview ─────────────────────────────────
    if s["win_rate"] < 0.4 and s["total_pnl_usdt"] < 0:
        ov = "지는 거래가 너무 많음 → 승률 높은 종목만 골라서 거래하세요. 나머지는 끊으세요"
    elif s["profit_factor"] < 1:
        ov = "벌어도 까먹는 구조 → 손절 라인을 진입 전에 무조건 걸고, 절대 옮기지 마세요"
    elif fee_pct > 30:
        ov = "수수료가 수익을 잠식 중 → 레버리지를 반으로 줄이세요. 수수료가 바로 절반 됩니다"
    else:
        ov = "수익 구조 양호 → 현재 전략 유지하되, 드로다운 깊어지면 포지션 사이즈 절반으로 줄이세요"

    # ── equity_curve ─────────────────────────────
    dd_pct = s["max_drawdown_pct"] * 100
    if s["total_pnl_usdt"] < -s["initial_balance_usdt"] * 0.1:
        eq = "돈을 잃는 전략 → 2주간 실거래를 멈추고, 지난 거래를 복기한 뒤 규칙을 다시 세우세요"
    elif s["total_pnl_usdt"] < 0:
        eq = f"드로다운 후 회복 못 함 → 드로다운 {dd_pct/2:.0f}% 넘으면 당일 거래 중단 규칙을 만드세요"
    else:
        eq = "수익 중이지만 드로다운 구간이 깊음 → 하락 시 포지션을 50% 줄이면 회복이 빨라집니다"

    # ── exchange_comparison ──────────────────────
    if s['avg_leverage'] > 15:
        exc = "레버리지가 도박 수준 → 10x 이하로 내리고, 모든 포지션에 SL 필수로 바꾸세요"
    elif b['stoploss_usage_ratio'] < 0.3:
        exc = "SL 거의 안 쓰는 중 → 진입 전에 SL 먼저 설정하세요. SL 없으면 진입 금지"
    else:
        exc = "세팅은 무난함 → SL 설정률 100%까지 올리고, 레버리지는 한 단계 더 낮추세요"

    # ── symbol_pnl ───────────────────────────────
    sym_pnl = (f"{worst['symbol']}이 손실 주범 → 이 종목 거래를 중단하고, "
               f"{best['symbol']}에 집중하세요")

    # ── symbol_winrate ───────────────────────────
    low_wr = [sym for sym in symbols if sym["win_rate"] < 0.35 and sym["trades"] >= 5]
    if low_wr:
        lw = low_wr[0]
        sym_wr = f"{lw['symbol']}은 계속 지고 있음 → 한달간 거래 금지. 승률 높은 종목 1~2개만 하세요"
    else:
        sym_wr = f"{worst['symbol']} 효율이 가장 낮음 → 종목을 3개 이하로 줄이세요"

    # ── weekday_pnl ──────────────────────────────
    if wpnl:
        worst_day = min(wpnl, key=wpnl.get)
        best_day = max(wpnl, key=wpnl.get)
        wd = f"{worst_day}에 손실 집중 → {worst_day}은 쉬세요. {best_day}에 더 집중하세요"
    else:
        wd = "데이터 부족 → 최소 2주 이상 데이터를 모아주세요"

    # ── hourly_pattern ───────────────────────────
    active_hrs = tp.get("most_active_hours_kst", tp.get("most_active_hours_utc", []))
    if active_hrs:
        hr_sorted = sorted(active_hrs)
        hr = f"수익 나는 시간대가 있음 → {hr_sorted[0]}~{hr_sorted[-1]}시(KST)에만 거래하고, 나머지 시간은 차트를 끄세요"
    else:
        hr = "데이터 부족 → 최소 2주 이상 데이터를 모아주세요"

    # ── pnl_distribution ─────────────────────────
    if rr_ratio < 1:
        dist = f"잃을 때 더 크게 잃는 구조 → SL:TP를 1:1.5 이상으로 설정하세요. TP가 SL보다 항상 커야 합니다"
    else:
        dist = f"손익비 양호 → 현재 SL:TP 비율 유지하되, TP를 조금 더 올려서 수익 구간을 늘려보세요"

    # ── action_items ─────────────────────────────
    actions = []
    if s["avg_leverage"] > 10:
        actions.append("오늘: 거래소 설정에서 레버리지 상한을 10x 이하로 바꾸세요")
    if b["stoploss_usage_ratio"] < 0.5:
        actions.append("오늘: 포지션 열기 전에 SL부터 설정하세요. SL 없는 진입은 금지")
    if b["revenge_trade_count"] > 3:
        actions.append("이번 주: 2연패하면 30분 쉬세요. 복수매매가 손실의 주범입니다")
    if low_wr:
        actions.append(f"오늘: {low_wr[0]['symbol']} 거래를 중단하세요")
    if b["max_trades_in_day"] > 15:
        actions.append("이번 주: 하루 8건 넘으면 앱을 끄세요")
    if worst["pnl_usdt"] < -500:
        actions.append(f"이번 달: {worst['symbol']}을 관심 종목에서 빼세요")
    if rr_ratio < 1 and b["stoploss_usage_ratio"] >= 0.5:
        actions.append("이번 주: TP를 SL의 1.5배로 설정하세요")
    if len(actions) < 2:
        actions.append("오늘: 매매 일지를 쓰세요. 진입 이유와 그때 기분을 적으면 패턴이 보입니다")
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
