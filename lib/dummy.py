"""
KNOAH 거래 분석 – 더미 데이터 생성
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .config import SYMBOLS, SIDES, PRICE_MAP


def generate_trades(n: int, exchange: str, days: int = 30,
                    start_id: int = 0) -> pd.DataFrame:
    """거래소 API 형태의 더미 거래 데이터 생성"""
    np.random.seed(None)
    rows = []
    base_time = datetime.now() - timedelta(days=days)

    skill = np.random.choice(
        ["beginner", "intermediate", "advanced"], p=[0.5, 0.35, 0.15]
    )
    win_base = {"beginner": 0.32, "intermediate": 0.45, "advanced": 0.55}[skill]

    sym_probs = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    lev_normal = [0.05, 0.10, 0.20, 0.20, 0.15, 0.12, 0.10, 0.05, 0.03]
    lev_btc    = [0.10, 0.20, 0.25, 0.20, 0.15, 0.05, 0.03, 0.01, 0.01]
    lev_opts   = [2, 3, 5, 8, 10, 15, 20, 25, 50]

    for i in range(n):
        symbol = np.random.choice(SYMBOLS, p=sym_probs)
        side = np.random.choice(SIDES, p=[0.6, 0.4])

        is_btc = symbol == "BTCUSDT"
        leverage = int(np.random.choice(lev_opts, p=lev_btc if is_btc else lev_normal))

        win = np.random.random() < (win_base + (0.1 if is_btc else -0.05))

        base_price = PRICE_MAP.get(symbol, 100)
        entry_price = base_price * np.random.uniform(0.92, 1.08)

        if win:
            pct = np.random.exponential(0.02)
            exit_price = entry_price * (1 + pct) if side == "LONG" else entry_price * (1 - pct)
        else:
            pct = np.random.exponential(0.025)
            exit_price = entry_price * (1 - pct) if side == "LONG" else entry_price * (1 + pct)

        qty = np.random.uniform(100, 2000)
        pnl = qty * leverage * pct * (1 if win else -1)
        fee = qty * leverage * 0.0008

        dt = base_time + timedelta(
            days=np.random.randint(0, days),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )

        rows.append({
            "id": start_id + i,
            "exchange": exchange,
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "side": side,
            "leverage": leverage,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity_usdt": round(qty, 2),
            "pnl_usdt": round(pnl, 2),
            "fee_usdt": round(fee, 2),
            "holding_minutes": int(np.random.exponential(120 if is_btc else 40)) + 1,
            "order_type": np.random.choice(["MARKET", "LIMIT"], p=[0.7, 0.3]),
            "stoploss_set": bool(np.random.choice([True, False], p=[0.2, 0.8])),
        })

    return pd.DataFrame(rows)


def generate_deposits(initial_balance: float = 10_000) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": 0,
        "datetime": (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d %H:%M"),
        "type": "DEPOSIT",
        "amount_usdt": initial_balance,
    }])
