"""
KNOAH 거래 분석 – 거래소 API 연동 (ccxt)
=========================================
Binance Futures, Bybit, OKX, Bitget USDT-M 선물 지원.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

try:
    import ccxt
except ImportError:
    ccxt = None  # UI에서 안내 메시지 표시


# ─────────────────────────────────────────────────
# 거래소 팩토리
# ─────────────────────────────────────────────────
_EXCHANGE_MAP = {
    "Binance": "binance",
    "Bybit": "bybit",
    "OKX": "okx",
    "Bitget": "bitget",
}

# OKX, Bitget은 passphrase 필요
NEEDS_PASSPHRASE = {"OKX", "Bitget"}


def create_exchange(name: str, api_key: str, api_secret: str,
                    passphrase: Optional[str] = None):
    """ccxt 거래소 인스턴스 생성 (USDT-M 선물 모드)"""
    if ccxt is None:
        raise ImportError("ccxt 패키지가 필요합니다: pip install ccxt")

    ccxt_id = _EXCHANGE_MAP.get(name)
    if not ccxt_id:
        raise ValueError(f"지원하지 않는 거래소: {name}")

    exchange_class = getattr(ccxt, ccxt_id)

    config = {
        "apiKey": api_key or "",
        "secret": api_secret or "",
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},  # USDT-M 선물
    }

    if name in NEEDS_PASSPHRASE:
        config["password"] = passphrase or ""

    # 거래소별 추가 설정
    if name == "Binance":
        config["options"]["defaultType"] = "future"
    elif name == "Bybit":
        config["options"]["defaultType"] = "swap"
    elif name == "OKX":
        config["options"]["defaultType"] = "swap"
    elif name == "Bitget":
        config["options"]["defaultType"] = "swap"
        config["options"]["defaultSubType"] = "linear"

    return exchange_class(config)


# ─────────────────────────────────────────────────
# 연결 테스트
# ─────────────────────────────────────────────────
def test_connection(exchange) -> dict:
    """API 연결 + 잔고 확인 (선물 엔드포인트 우선)"""
    # 바이낸스: sapi가 한국에서 차단됨 → fapi 직접 호출
    exchange_id = exchange.id if hasattr(exchange, "id") else ""

    try:
        if exchange_id == "binance":
            # fapi/v2/balance 직접 호출 (sapi 우회)
            resp = exchange.fapiPrivateV2GetBalance()
            usdt_item = next((x for x in resp if x.get("asset") == "USDT"), None)
            if usdt_item:
                total = round(float(usdt_item.get("balance", 0)), 2)
                free = round(float(usdt_item.get("availableBalance", 0)), 2)
            else:
                total, free = 0, 0
            return {"ok": True, "total_usdt": total, "free_usdt": free,
                    "msg": f"연결 성공 · 선물 잔고: ${total:,.2f} USDT"}
        else:
            balance = exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total = float(usdt.get("total", 0) or 0)
            free = float(usdt.get("free", 0) or 0)
            return {"ok": True, "total_usdt": round(total, 2), "free_usdt": round(free, 2),
                    "msg": f"연결 성공 · 잔고: ${total:,.2f} USDT"}
    except ccxt.AuthenticationError:
        return {"ok": False, "msg": "인증 실패: API Key / Secret을 확인해주세요."}
    except ccxt.PermissionDenied:
        return {"ok": False, "msg": "권한 부족: API 키에 선물 읽기 권한이 필요합니다."}
    except ccxt.NetworkError as e:
        err = str(e)
        if "restricted location" in err or "451" in err:
            return {"ok": False, "msg": "지역 제한: 해당 거래소는 현재 위치(한국)에서 API 접속이 차단됩니다. VPN을 사용하거나 다른 거래소를 이용해주세요."}
        return {"ok": False, "msg": f"네트워크 오류: {e}"}
    except Exception as e:
        return {"ok": False, "msg": f"연결 실패: {e}"}


# ─────────────────────────────────────────────────
# 거래 내역 가져오기
# ─────────────────────────────────────────────────
def fetch_trades(exchange, exchange_name: str,
                 days: int = 30, symbols: list = None) -> pd.DataFrame:
    """
    거래소에서 선물 거래 내역을 가져와 통합 DataFrame으로 반환.
    거래소별로 최적의 엔드포인트를 사용.
    """
    since_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    if exchange_name == "Binance":
        df = _fetch_binance(exchange, since_ms, symbols)
    elif exchange_name == "Bybit":
        df = _fetch_bybit(exchange, since_ms)
    elif exchange_name == "OKX":
        df = _fetch_okx(exchange)
    elif exchange_name == "Bitget":
        df = _fetch_bitget(exchange)
    else:
        raise ValueError(f"지원하지 않는 거래소: {exchange_name}")

    if not df.empty:
        df.insert(1, "exchange", exchange_name)
    return df


# ── Binance Futures ──────────────────────────────
def _fetch_binance(exchange, since_ms: int, symbols: list = None) -> pd.DataFrame:
    """
    Binance USDT-M: 전체 income 조회 (PnL + 수수료 + 펀딩피)
    """
    rows = []

    # 1) 모든 income 조회 (페이지네이션)
    all_income = []
    _cursor_time = since_ms
    for _ in range(100):
        batch = exchange.fapiPrivateGetIncome({
            "startTime": _cursor_time,
            "limit": 1000,
        })
        if not batch:
            break
        all_income.extend(batch)
        _cursor_time = int(batch[-1].get("time", 0)) + 1
        if len(batch) < 1000:
            break

    # income을 타입별로 분리
    income_data = [i for i in all_income if i.get("incomeType") == "REALIZED_PNL"]
    commission_data = [i for i in all_income if i.get("incomeType") == "COMMISSION"]
    funding_data = [i for i in all_income if i.get("incomeType") == "FUNDING_FEE"]

    # 심볼+시간 기준으로 수수료/펀딩피 매핑
    _fee_map = {}  # (symbol, time_bucket) → fee
    for c in commission_data:
        sym = c.get("symbol", "")
        ts = int(c.get("time", 0)) // 60000  # 분 단위 버킷
        _fee_map[(sym, ts)] = _fee_map.get((sym, ts), 0) + abs(float(c.get("income", 0)))
    _funding_map = {}
    for f in funding_data:
        sym = f.get("symbol", "")
        ts = int(f.get("time", 0)) // 60000
        _funding_map[(sym, ts)] = _funding_map.get((sym, ts), 0) + float(f.get("income", 0))

    # 2) User trades for entry/exit details (페이지네이션)
    trade_symbols = symbols or list(set(i.get("symbol", "") for i in income_data if i.get("symbol")))

    all_trades = []
    for sym in trade_symbols:
        _sym_cursor = since_ms
        for _ in range(20):
            try:
                trades = exchange.fapiPrivateGetUserTrades({
                    "symbol": sym.replace("/", "").replace(":USDT", ""),
                    "startTime": _sym_cursor,
                    "limit": 1000,
                })
                if not trades:
                    break
                all_trades.extend(trades)
                _sym_cursor = int(trades[-1].get("time", 0)) + 1
                if len(trades) < 1000:
                    break
            except Exception:
                break

    # income으로 포지션별 PnL 집계
    for inc in income_data:
        symbol = inc.get("symbol", "")
        pnl = float(inc.get("income", 0))
        ts = int(inc.get("time", 0))

        # 해당 심볼의 최근 거래에서 추가 정보 추출
        sym_trades = [t for t in all_trades if t.get("symbol") == symbol]
        recent = [t for t in sym_trades if abs(int(t.get("time", 0)) - ts) < 60000]

        if recent:
            last = recent[-1]
            side = "LONG" if last.get("side") == "BUY" and not last.get("buyer") else "SHORT"
            leverage = 1
            qty = sum(float(t.get("quoteQty", 0)) for t in recent)
            fee = sum(abs(float(t.get("commission", 0))) for t in recent)
            price = float(last.get("price", 0))
        else:
            side = "LONG" if pnl > 0 else "SHORT"
            leverage = 1
            qty = abs(pnl) * 10
            price = 0
            # fee_map에서 수수료 가져오기
            ts_bucket = ts // 60000
            fee = _fee_map.get((symbol, ts_bucket), 0)

        # 펀딩피 추가
        ts_bucket = ts // 60000
        funding = abs(_funding_map.get((symbol, ts_bucket), 0))
        total_fee = fee + funding

        rows.append({
            "datetime": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M"),
            "symbol": symbol,
            "side": side,
            "leverage": leverage,
            "entry_price": round(price, 4),
            "exit_price": 0,
            "quantity_usdt": round(abs(qty), 2),
            "pnl_usdt": round(pnl, 2),
            "fee_usdt": round(total_fee, 2),
            "holding_minutes": 0,
            "order_type": "MARKET",
            "stoploss_set": False,
        })

    return _to_dataframe(rows)



# ── Bybit ────────────────────────────────────────
def _fetch_bybit(exchange, since_ms: int) -> pd.DataFrame:
    """
    Bybit V5: /v5/position/closed-pnl — 포지션 단위 PnL 직접 제공
    """
    rows = []
    cursor = ""

    for _ in range(10):  # 최대 10페이지
        params = {"category": "linear", "limit": 200}
        if cursor:
            params["cursor"] = cursor

        resp = exchange.privateGetV5PositionClosedPnl(params)
        result = resp.get("result", {})
        items = result.get("list", [])

        if not items:
            break

        for item in items:
            ts = int(item.get("updatedTime", item.get("createdTime", 0)))
            if ts < since_ms:
                continue

            symbol = item.get("symbol", "")
            side = item.get("side", "Buy")
            side = "LONG" if side == "Buy" else "SHORT"

            rows.append({
                "datetime": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "side": side,
                "leverage": int(float(item.get("leverage", 1))),
                "entry_price": round(float(item.get("avgEntryPrice", 0)), 4),
                "exit_price": round(float(item.get("avgExitPrice", 0)), 4),
                "quantity_usdt": round(float(item.get("cumEntryValue", 0)), 2),
                "pnl_usdt": round(float(item.get("closedPnl", 0)), 2),
                "fee_usdt": 0,
                "holding_minutes": 0,
                "order_type": item.get("orderType", "Market").upper(),
                "stoploss_set": False,
            })

        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break

    return _to_dataframe(rows)


# ── OKX ──────────────────────────────────────────
def _fetch_okx(exchange) -> pd.DataFrame:
    """
    OKX: /api/v5/account/positions-history — 종료된 포지션 이력
    """
    rows = []

    for _ in range(5):  # 페이지네이션
        params = {"instType": "SWAP", "limit": "100"}

        resp = exchange.privateGetApiV5AccountPositionsHistory(params)
        data = resp.get("data", [])

        if not data:
            break

        for pos in data:
            inst_id = pos.get("instId", "")
            # OKX instId: BTC-USDT-SWAP → BTCUSDT
            symbol = inst_id.replace("-SWAP", "").replace("-", "")

            direction = pos.get("direction", "")
            side = "LONG" if direction == "long" else "SHORT"

            ts = int(pos.get("uTime", pos.get("cTime", 0)))

            rows.append({
                "datetime": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "side": side,
                "leverage": int(float(pos.get("lever", 1))),
                "entry_price": round(float(pos.get("openAvgPx", 0)), 4),
                "exit_price": round(float(pos.get("closeAvgPx", 0)), 4),
                "quantity_usdt": round(float(pos.get("openMaxPos", 0)) * float(pos.get("openAvgPx", 1)), 2),
                "pnl_usdt": round(float(pos.get("pnl", 0)), 2),
                "fee_usdt": round(abs(float(pos.get("fee", 0))) + abs(float(pos.get("fundingFee", 0))), 2),
                "holding_minutes": 0,
                "order_type": "MARKET",
                "stoploss_set": False,
            })

        break  # OKX는 after 파라미터로 페이지네이션 (간소화)

    return _to_dataframe(rows)


# ── Bitget ───────────────────────────────────────
def _fetch_bitget(exchange) -> pd.DataFrame:
    """
    Bitget: /api/v2/mix/position/history-position — 포지션 이력
    """
    rows = []

    try:
        resp = exchange.privateGetApiV2MixPositionHistoryPosition({
            "productType": "USDT-FUTURES",
            "limit": "200",
        })
        data = resp.get("data", {})
        items = data.get("list", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []

    except Exception:
        # fallback: fill history
        try:
            resp = exchange.privateGetApiV2MixOrderFillHistory({
                "productType": "USDT-FUTURES",
                "limit": "200",
            })
            items = resp.get("data", {}).get("fillList", [])
        except Exception:
            items = []

    for item in items:
        symbol = item.get("symbol", "")
        # Bitget symbol: BTCUSDT → BTCUSDT or BTCUSDT_UMCBL → BTCUSDT
        symbol = symbol.split("_")[0] if "_" in symbol else symbol

        side = item.get("holdSide", item.get("side", "long"))
        side = "LONG" if side.lower() in ("long", "buy") else "SHORT"

        ts = int(item.get("uTime", item.get("cTime", 0)))

        rows.append({
            "datetime": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "",
            "symbol": symbol,
            "side": side,
            "leverage": int(float(item.get("leverage", 1))),
            "entry_price": round(float(item.get("openAvgPrice", item.get("openPriceAvg", 0))), 4),
            "exit_price": round(float(item.get("closeAvgPrice", item.get("closePriceAvg", 0))), 4),
            "quantity_usdt": round(abs(float(item.get("margin", 0))) * float(item.get("leverage", 1)), 2),
            "pnl_usdt": round(float(item.get("achievedProfits", item.get("profit", 0))), 2),
            "fee_usdt": round(abs(float(item.get("totalFee", item.get("fee", 0)))), 2),
            "holding_minutes": 0,
            "order_type": "MARKET",
            "stoploss_set": False,
        })

    return _to_dataframe(rows)


# ── 입출금 내역 ──────────────────────────────────
def fetch_deposits_withdrawals(exchange, days: int = 90) -> pd.DataFrame:
    """입출금 내역 조회 (바이낸스: 선물 income TRANSFER 사용)"""
    rows = []
    since_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    exchange_id = exchange.id if hasattr(exchange, "id") else ""

    # 바이낸스: sapi 차단 우회 → fapi income에서 TRANSFER 내역 조회
    if exchange_id == "binance":
        try:
            _cursor = since_ms
            for _ in range(20):
                transfers = exchange.fapiPrivateGetIncome({
                    "incomeType": "TRANSFER",
                    "startTime": _cursor,
                    "limit": 1000,
                })
                if not transfers:
                    break
                for t in transfers:
                    amt = float(t.get("income", 0))
                    ts = int(t.get("time", 0))
                    rows.append({
                        "datetime": datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M"),
                        "type": "DEPOSIT" if amt > 0 else "WITHDRAWAL",
                        "amount_usdt": round(abs(amt), 2),
                    })
                _cursor = int(transfers[-1].get("time", 0)) + 1
                if len(transfers) < 1000:
                    break
        except Exception:
            pass

        if not rows:
            return pd.DataFrame(columns=["id", "datetime", "type", "amount_usdt"])
        df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
        df.insert(0, "id", range(len(df)))
        return df

    # 다른 거래소: 기존 방식
    try:
        deposits = exchange.fetch_deposits("USDT", since_ms, 100)
        for dep in deposits:
            rows.append({
                "datetime": datetime.fromtimestamp(dep["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M"),
                "type": "DEPOSIT",
                "amount_usdt": round(float(dep.get("amount", 0)), 2),
            })
    except Exception:
        pass

    try:
        withdrawals = exchange.fetch_withdrawals("USDT", since_ms, 100)
        for wd in withdrawals:
            rows.append({
                "datetime": datetime.fromtimestamp(wd["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M"),
                "type": "WITHDRAWAL",
                "amount_usdt": round(float(wd.get("amount", 0)), 2),
            })
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(columns=["id", "datetime", "type", "amount_usdt"])

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


# ── 유틸 ─────────────────────────────────────────
def _to_dataframe(rows: list) -> pd.DataFrame:
    """행 목록 → 정규화된 DataFrame"""
    if not rows:
        from .config import TRADE_COLUMNS
        return pd.DataFrame(columns=TRADE_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df
