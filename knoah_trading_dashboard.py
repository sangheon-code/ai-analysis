"""
KNOAH 거래 분석 대시보드
========================
멀티 거래소 API 연동 + Claude AI 딥 리포트
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from lib.config import (
    SYMBOLS, EXCHANGES, SIDES,
    CUSTOM_CSS, TRADE_COLUMNS, DEPOSIT_COLUMNS,
)
from lib.dummy import generate_trades, generate_deposits
from lib.analysis import aggregate_data, aggregate_deep_data, chat_with_data, generate_detail_summary

try:
    from lib.exchanges import (
        ccxt, create_exchange, test_connection,
        fetch_trades as exchange_fetch_trades,
        fetch_deposits_withdrawals, fetch_ohlcv,
    )
    HAS_CCXT = ccxt is not None
except ImportError:
    HAS_CCXT = False
    fetch_ohlcv = None


# ══════════════════════════════════════════════════
# Page Config & CSS
# ══════════════════════════════════════════════════
st.set_page_config(page_title="KNOAH Trading Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""<style>
.style-card { background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35; border-radius: 14px; padding: 24px 28px; }
.habit-card { background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35; border-radius: 14px; padding: 20px 24px; text-align: center; }
.habit-card .val { font-family: JetBrains Mono; font-size: 28px; font-weight: 700; margin: 8px 0 4px; }
.habit-card .lbl { font-size: 11px; color: #7b7b9e; text-transform: uppercase; letter-spacing: 1px; }
.habit-card .sub { font-size: 13px; margin-top: 6px; }
.action-card { background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35; border-radius: 14px; padding: 24px 28px; line-height: 1.8; font-size: 14px; }
.action-card h3 { color: #a78bfa; margin-top: 20px; font-size: 16px; }
.action-card strong { color: #e8e8ed; }
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════
_DEFAULTS = {
    "trades": pd.DataFrame(columns=TRADE_COLUMNS),
    "deposits": pd.DataFrame(columns=DEPOSIT_COLUMNS),
    "ai_deep_report": None,
    "trade_id_counter": 0,
    "connections": {},
    "chat_history": [],
    "chat_total_cost": 0.0,
    "detail_summary": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════
def connected_exchanges():
    return list(st.session_state.connections.keys())

def is_any_connected():
    return len(st.session_state.connections) > 0

def merge_exchange_trades(new_df, exchange_name):
    existing = st.session_state.trades
    if existing.empty:
        st.session_state.trades = new_df
    else:
        other = existing[existing["exchange"] != exchange_name]
        st.session_state.trades = pd.concat([other, new_df], ignore_index=True)
    st.session_state.trades = st.session_state.trades.sort_values("datetime").reset_index(drop=True)
    st.session_state.trades["id"] = range(len(st.session_state.trades))
    st.session_state.trade_id_counter = len(st.session_state.trades)

def merge_exchange_deposits(new_df, exchange_name):
    existing = st.session_state.deposits
    if "exchange" not in new_df.columns and not new_df.empty:
        new_df.insert(1, "exchange", exchange_name)
    if existing.empty:
        st.session_state.deposits = new_df
    else:
        if "exchange" not in existing.columns:
            existing.insert(1, "exchange", "Unknown")
        other = existing[existing["exchange"] != exchange_name]
        st.session_state.deposits = pd.concat([other, new_df], ignore_index=True)


# ══════════════════════════════════════════════════
# Plotly Theme
# ══════════════════════════════════════════════════
_CHART = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#a0a0b8"),
    margin=dict(l=40, r=20, t=32, b=36), height=300,
)
_C = {"profit": "#00e59b", "loss": "#ff4d6a", "primary": "#6b8aff", "secondary": "#a78bfa", "neutral": "#4a4a6a"}
_EX_COLOR = {"Binance": "#F0B90B", "Bybit": "#FF6500", "OKX": "#00C8FF", "Bitget": "#00D4AA", "Demo": "#6b8aff"}


# ══════════════════════════════════════════════════
# SIDEBAR (AI 챗만)
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# KNOAH")
    st.caption("Trading Analysis Platform")

    st.markdown("---")

    # ── AI 챗봇 ────────────────────────────────────
    st.markdown("### 💬 AI 챗")
    for msg in st.session_state.chat_history:
        _role_icon = "🧑" if msg["role"] == "user" else "🤖"
        _bg = "rgba(107,138,255,0.08)" if msg["role"] == "user" else "rgba(0,0,0,0)"
        st.markdown(f'<div style="background:{_bg};border-radius:8px;padding:6px 10px;margin:4px 0;font-size:13px">'
            f'{_role_icon} {msg["content"].replace("$", "USD ")}</div>', unsafe_allow_html=True)

    _chat_input = st.text_input("질문", placeholder="거래 데이터에 대해 질문...", key="sidebar_chat_input", label_visibility="collapsed")
    if _chat_input and st.session_state.get("_last_chat_input") != _chat_input:
        st.session_state._last_chat_input = _chat_input
        st.session_state.chat_history.append({"role": "user", "content": _chat_input})
        st.session_state._chat_pending = True
        st.rerun()

    if st.session_state.chat_history:
        _ch1, _ch2 = st.columns(2)
        with _ch1:
            st.caption(f"{len(st.session_state.chat_history)//2}회 · ${st.session_state.chat_total_cost:.4f}")
        with _ch2:
            if st.button("초기화", key="chat_clear", width="stretch"):
                st.session_state.chat_history = []
                st.session_state.chat_total_cost = 0.0
                st.session_state._last_chat_input = ""
                st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#55556a;font-size:11px'>KNOAH v1.0 · Powered by Claude</div>", unsafe_allow_html=True)


# ── Claude API Key (환경변수 / Streamlit Secrets) ──
api_key_claude = os.getenv("ANTHROPIC_API_KEY", "")
if not api_key_claude:
    try:
        api_key_claude = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key_claude = ""


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
st.markdown("# KNOAH Trading Analysis")

has_data = not st.session_state.trades.empty
conn_names = connected_exchanges()
badges = " ".join(f'<span class="badge badge-success">{n}</span>' for n in conn_names) if conn_names else '<span class="badge badge-info">DEMO</span>'
n_ex = int(st.session_state.trades["exchange"].nunique()) if has_data and "exchange" in st.session_state.trades.columns else 0
_hdr1, _hdr2 = st.columns([4, 1])
with _hdr1:
    st.markdown(f'{badges} &nbsp; 거래소 **{n_ex}개** · 거래 **{len(st.session_state.trades)}건**', unsafe_allow_html=True)
with _hdr2:
    if has_data and st.button("🗑 초기화", key="reset_main"):
        for k in ["trades", "deposits", "ai_deep_report", "trade_id_counter", "detail_summary"]:
            if k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
        st.session_state.connections = {}
        st.rerun()

if not has_data:
    # ── 데이터 없을 때: 거래소 연결 / 더미 데이터 UI ──
    col_conn, col_dummy = st.columns(2)

    with col_conn:
        st.markdown("### 🔗 거래소 연결")
        if not HAS_CCXT:
            st.warning("`pip install ccxt` 필요")
        exchange_name = st.selectbox("거래소", EXCHANGES, key="sel_exchange")
        api_key_exchange = st.text_input("API Key", type="password", key="api_key_exchange")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        passphrase = None
        if exchange_name in ("OKX", "Bitget"):
            passphrase = st.text_input("Passphrase", type="password", key="passphrase")
        if st.button("🔗 연결", width="stretch", disabled=not HAS_CCXT):
            if api_key_exchange and api_secret:
                with st.spinner(f"{exchange_name} 연결 중..."):
                    try:
                        ex = create_exchange(exchange_name, api_key_exchange, api_secret, passphrase)
                        result = test_connection(ex)
                        if result["ok"]:
                            st.session_state.connections[exchange_name] = {"instance": ex, "balance": result.get("total_usdt", 0), "msg": result["msg"]}
                            st.success(f"{exchange_name} 연결!")
                            st.rerun()
                        else:
                            st.error(result["msg"])
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("API Key/Secret 필요")

        if is_any_connected():
            st.markdown("---")
            fetch_days = st.number_input("조회 기간(일)", 7, 1095, 90, key="fetch_days")
            fetch_targets = st.multiselect("대상", connected_exchanges(), default=connected_exchanges(), key="ft")
            if st.button("📥 거래 내역 가져오기", width="stretch", type="primary"):
                total = 0
                for en in fetch_targets:
                    info = st.session_state.connections[en]
                    with st.spinner(f"{en}..."):
                        try:
                            tdf = exchange_fetch_trades(info["instance"], en, days=fetch_days)
                            if not tdf.empty:
                                merge_exchange_trades(tdf, en)
                                total += len(tdf)
                            ddf = fetch_deposits_withdrawals(info["instance"], days=fetch_days)
                            if not ddf.empty:
                                merge_exchange_deposits(ddf, en)
                            st.success(f"{en}: {len(tdf)}건")
                        except Exception as e:
                            st.error(f"{en}: {e}")
                if total > 0:
                    st.session_state.ai_deep_report = None
                    st.rerun()

    with col_dummy:
        st.markdown("### 🎲 더미 데이터")
        c_a, c_b = st.columns(2)
        with c_a:
            n_trades = st.number_input("건수", 10, 500, 100, step=10, key="n_trades")
        with c_b:
            n_days = st.number_input("기간(일)", 7, 90, 30, step=7, key="n_days")
        dummy_exchanges = st.multiselect("거래소", EXCHANGES, default=["Binance"], key="dummy_ex")
        ex_balances = {}
        for dex in (dummy_exchanges or ["Binance"]):
            ex_balances[dex] = st.number_input(f"{dex} 초기 잔고(USDT)", 100, 1_000_000, 10_000, step=1000, key=f"bal_{dex}")

        if st.button("더미 데이터 생성", width="stretch"):
            all_t, all_d = [], []
            exes = dummy_exchanges or ["Binance"]
            per_n = max(n_trades // len(exes), 10)
            for i, en in enumerate(exes):
                all_t.append(generate_trades(per_n, en, n_days, start_id=i * per_n))
                dep = generate_deposits(ex_balances.get(en, 10_000))
                dep.insert(1, "exchange", en)
                all_d.append(dep)
            combined = pd.concat(all_t, ignore_index=True).sort_values("datetime").reset_index(drop=True)
            combined["id"] = range(len(combined))
            st.session_state.trades = combined
            st.session_state.deposits = pd.concat(all_d, ignore_index=True).reset_index(drop=True)
            st.session_state.trade_id_counter = len(combined)
            st.session_state.ai_deep_report = None
            st.rerun()

    st.stop()

# ── 거래소 관리 (expander) ────────────────────────
with st.expander("거래소 관리 / 데이터 추가", expanded=False):
    _mgr1, _mgr2 = st.columns(2)
    with _mgr1:
        # 연결된 거래소 표시
        if st.session_state.connections:
            for ex_name, info in list(st.session_state.connections.items()):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f'<span class="conn-dot on"></span> **{ex_name}**', unsafe_allow_html=True)
                with c2:
                    st.caption(f"${info.get('balance', 0):,.2f}")
                with c3:
                    if st.button("✕", key=f"rm_{ex_name}"):
                        del st.session_state.connections[ex_name]
                        if not st.session_state.trades.empty and "exchange" in st.session_state.trades.columns:
                            st.session_state.trades = st.session_state.trades[st.session_state.trades["exchange"] != ex_name].reset_index(drop=True)
                        st.session_state.ai_deep_report = None
                        st.rerun()

        # 새 거래소 추가
        exchange_name = st.selectbox("거래소", EXCHANGES, key="sel_exchange_main")
        api_key_exchange = st.text_input("API Key", type="password", key="api_key_ex_main")
        api_secret = st.text_input("API Secret", type="password", key="api_secret_main")
        passphrase = None
        if exchange_name in ("OKX", "Bitget"):
            passphrase = st.text_input("Passphrase", type="password", key="passphrase_main")
        if st.button("🔗 연결", key="connect_main", width="stretch", disabled=not HAS_CCXT):
            if api_key_exchange and api_secret:
                with st.spinner(f"{exchange_name} 연결 중..."):
                    try:
                        ex = create_exchange(exchange_name, api_key_exchange, api_secret, passphrase)
                        result = test_connection(ex)
                        if result["ok"]:
                            st.session_state.connections[exchange_name] = {"instance": ex, "balance": result.get("total_usdt", 0), "msg": result["msg"]}
                            st.success(f"{exchange_name} 연결!")
                            st.rerun()
                        else:
                            st.error(result["msg"])
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("API Key/Secret 필요")

    with _mgr2:
        if is_any_connected():
            st.markdown("**거래 내역 가져오기**")
            fetch_days = st.number_input("조회 기간(일)", 7, 1095, 90, key="fetch_days_main")
            fetch_targets = st.multiselect("대상", connected_exchanges(), default=connected_exchanges(), key="ft_main")
            if st.button("📥 가져오기", key="fetch_main", width="stretch", type="primary"):
                total = 0
                for en in fetch_targets:
                    info = st.session_state.connections[en]
                    with st.spinner(f"{en}..."):
                        try:
                            tdf = exchange_fetch_trades(info["instance"], en, days=fetch_days)
                            if not tdf.empty:
                                merge_exchange_trades(tdf, en)
                                total += len(tdf)
                            ddf = fetch_deposits_withdrawals(info["instance"], days=fetch_days)
                            if not ddf.empty:
                                merge_exchange_deposits(ddf, en)
                            st.success(f"{en}: {len(tdf)}건")
                        except Exception as e:
                            st.error(f"{en}: {e}")
                if total > 0:
                    st.session_state.ai_deep_report = None
                    st.rerun()
        else:
            st.info("거래소를 연결하면 데이터를 가져올 수 있습니다.")

# ── 거래소 필터 ──────────────────────────────────
avail_ex = sorted(st.session_state.trades["exchange"].unique().tolist()) if "exchange" in st.session_state.trades.columns else []
if len(avail_ex) > 1:
    selected_ex = st.multiselect("분석 대상", avail_ex, default=avail_ex, key="gf")
else:
    selected_ex = avail_ex

def filtered_trades():
    d = st.session_state.trades
    if d.empty or not selected_ex: return d
    if "exchange" in d.columns: return d[d["exchange"].isin(selected_ex)]
    return d

df = filtered_trades().copy()
if df.empty: st.warning("선택된 거래소에 데이터가 없습니다."); st.stop()

df["datetime"] = pd.to_datetime(df["datetime"])
df["net_pnl"] = df["pnl_usdt"] - df["fee_usdt"]

# ── 잔고 계산 (입출금 기반) ───────────────────────
_connections = st.session_state.connections
_dep_all = st.session_state.deposits if not st.session_state.deposits.empty else pd.DataFrame(columns=["type", "amount_usdt", "exchange"])

# 거래소별 입출금 집계
_ex_initial = {}      # 초기 자산 (첫 입금)
_ex_deposits = {}     # 총 입금
_ex_withdrawals = {}  # 총 출금

for en in (df["exchange"].unique() if "exchange" in df.columns else ["default"]):
    if not _dep_all.empty and "exchange" in _dep_all.columns:
        ex_dep = _dep_all[_dep_all["exchange"] == en].sort_values("datetime") if "datetime" in _dep_all.columns else _dep_all[_dep_all["exchange"] == en]
        dep_list = ex_dep[ex_dep["type"] == "DEPOSIT"]["amount_usdt"]
        total_in = float(dep_list.sum()) if len(dep_list) > 0 else 0
        first_deposit = float(dep_list.iloc[0]) if len(dep_list) > 0 else 0
        total_out = float(ex_dep[ex_dep["type"] == "WITHDRAWAL"]["amount_usdt"].sum()) if len(ex_dep) > 0 else 0
    else:
        total_in, first_deposit, total_out = 10_000, 10_000, 0

    _ex_initial[en] = first_deposit if first_deposit > 0 else 10_000
    _ex_deposits[en] = total_in
    _ex_withdrawals[en] = total_out

# 현재 잔고 = API (연결 시) 또는 계산 (비연결 시)
_current_bal = 0
_ex_bal = {}  # 호환용
for en in (df["exchange"].unique() if "exchange" in df.columns else ["default"]):
    ex_pnl = float(df[df["exchange"] == en]["net_pnl"].sum()) if en != "default" else float(df["net_pnl"].sum())
    _ex_bal[en] = _ex_initial[en]  # equity curve용
    if en in _connections:
        _current_bal += float(_connections[en].get("balance", 0))
    else:
        # 현재자산 = 초기자산 + 추가입금 - 총출금 + 거래손익
        extra_dep = _ex_deposits[en] - _ex_initial[en]
        _current_bal += max(0, _ex_initial[en] + extra_dep - _ex_withdrawals[en] + ex_pnl)

# KNOAH 수익률 공식
# 초기자산 = 첫 입금
# 추가입금 = 총입금 - 초기자산
# 순이익 = 현재자산 - 초기자산 - 추가입금 + 총출금
# 수익률 = 순이익 / (초기자산 + 추가입금)
_init_total = sum(_ex_initial.values())
_dep_total = sum(_ex_deposits.values())
_wd_total = sum(_ex_withdrawals.values())
_extra_dep_total = _dep_total - _init_total  # 추가 입금

net_profit = _current_bal - _init_total - _extra_dep_total + _wd_total
roi_denom = _init_total + _extra_dep_total
roi = (net_profit / roi_denom * 100) if roi_denom > 0 else 0
init_bal_total = _init_total
win_count = len(df[df["pnl_usdt"] > 0])
win_rate = win_count / len(df) * 100
total_fee = float(df["fee_usdt"].sum())
avg_lev = float(df["leverage"].mean())
_roi_color = _C["profit"] if roi >= 0 else _C["loss"]
_pnl_sign = "+" if net_profit >= 0 else ""

# ── Equity 계산 (순수 거래 성과, 입출금 제외) ─────
# 입출금은 자금 이동이지 거래 성과가 아니므로 equity curve에서 제외
# 초기잔고(첫 입금) 기준으로 PnL만 누적
multi_ex = "exchange" in df.columns and df["exchange"].nunique() > 1

if multi_ex:
    _frames = []
    for en, grp in df.groupby("exchange"):
        g = grp.sort_values("datetime").copy()
        eb = _ex_bal.get(en, 10_000)
        g["ex_equity"] = eb + g["net_pnl"].cumsum()
        _frames.append(g[["datetime", "exchange", "net_pnl", "symbol", "side", "ex_equity"]])
    _all_eq = pd.concat(_frames).sort_values("datetime")
    _pivot = _all_eq.pivot_table(index="datetime", columns="exchange", values="ex_equity", aggfunc="last").ffill().bfill()
    _pivot["total"] = _pivot.sum(axis=1)
    _total_init = sum(_ex_bal.get(en, 10_000) for en in df["exchange"].unique())
    ds = _all_eq.merge(_pivot[["total"]].reset_index(), on="datetime", how="left")
    ds["equity"] = ds["total"]
else:
    ds = df.sort_values("datetime").copy()
    eb = list(_ex_bal.values())[0] if _ex_bal else init_bal_total
    ds["equity"] = eb + ds["net_pnl"].cumsum()
    _total_init = eb


# ── 사이드바 챗봇 응답 처리 ─────────────────────────
if st.session_state.get("_chat_pending") and api_key_claude and has_data:
    st.session_state._chat_pending = False
    _chat_q = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
    if _chat_q:
        try:
            _basic_chat = aggregate_data(df, st.session_state.deposits, "통합")
            _deep_chat = aggregate_deep_data(df, st.session_state.deposits)
            _recent = st.session_state.chat_history[-7:-1]
            result = chat_with_data(_chat_q, _basic_chat, _deep_chat, _recent, api_key_claude)
            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
            st.session_state.chat_total_cost += result["cost_usd"]
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"오류: {e}"})
        st.rerun()


# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
tab_dashboard, tab_detail, tab_whatif, tab_journal = st.tabs(["대시보드", "상세 분석", "What-if", "매매 일지"])


# ══════════════════════════════════════════════════
# TAB 1: 대시보드
# ══════════════════════════════════════════════════
with tab_dashboard:
    # ── Hero ──────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; margin:20px 0 28px 0;">
      <div style="font-size:11px; color:#55556a; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;">Current Balance</div>
      <div style="font-size:48px; font-family:JetBrains Mono; font-weight:700; color:#e8e8ed; line-height:1.1;">${_current_bal:,.2f}</div>
      <div style="margin-top:10px; display:inline-flex; gap:32px; align-items:baseline;">
        <div><span style="font-size:11px; color:#55556a; letter-spacing:1px;">ROI</span>
          <span style="font-size:32px; font-family:JetBrains Mono; font-weight:700; color:{_roi_color}; margin-left:8px;">{roi:+.2f}%</span></div>
        <div><span style="font-size:11px; color:#55556a; letter-spacing:1px;">P&L</span>
          <span style="font-size:24px; font-family:JetBrains Mono; font-weight:600; color:{_roi_color}; margin-left:8px;">{_pnl_sign}${net_profit:,.2f}</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 자산 곡선 ────────────────────────────────
    st.markdown('<div class="section-hdr">자산 곡선</div>', unsafe_allow_html=True)
    fig_eq = go.Figure()
    # 거래소별 라인 먼저 (아래 깔림 방지)
    if multi_ex:
        for en, grp in _all_eq.groupby("exchange"):
            fig_eq.add_trace(go.Scatter(x=grp["datetime"], y=grp["ex_equity"], mode="lines", name=en,
                line=dict(color=_EX_COLOR.get(en, "#888"), width=2)))
    # 통합 라인 위에
    fig_eq.add_trace(go.Scatter(x=ds["datetime"], y=ds["equity"], mode="lines", name="통합" if multi_ex else "자산",
        line=dict(color=_C["primary"], width=2.5 if not multi_ex else 2, dash="dot" if multi_ex else "solid")))
    _ref_bal = _total_init if multi_ex else (list(_ex_bal.values())[0] if _ex_bal else init_bal_total)
    fig_eq.add_hline(y=_ref_bal, line_dash="dash", line_color="#4a4a6a", line_width=1, annotation_text="초기 잔고", annotation_font_color="#7b7b9e")
    # y축 범위: 모든 라인 포함
    _eq_min = float(ds["equity"].min())
    _eq_max = float(ds["equity"].max())
    if multi_ex:
        _eq_min = min(_eq_min, float(_all_eq["ex_equity"].min()))
        _eq_max = max(_eq_max, float(_all_eq["ex_equity"].max()))
    _eq_pad = max((_eq_max - _eq_min) * 0.1, 1)
    fig_eq.update_layout(**{**_CHART, "height": 380}, xaxis_title="", yaxis_title="USDT",
        yaxis_range=[_eq_min - _eq_pad, _eq_max + _eq_pad],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_eq, width="stretch")

    # ── 핵심 지표 ────────────────────────────────
    st.markdown('<div class="section-hdr">핵심 지표</div>', unsafe_allow_html=True)
    _avg_win = float(df[df["pnl_usdt"] > 0]["net_pnl"].mean()) if win_count > 0 else 0
    _avg_loss = float(df[df["pnl_usdt"] <= 0]["net_pnl"].mean()) if len(df) - win_count > 0 else 0
    _pf = round(abs(float(df[df["net_pnl"] > 0]["net_pnl"].sum())) / max(abs(float(df[df["net_pnl"] <= 0]["net_pnl"].sum())), 1), 2)
    _peak = ds["equity"].cummax()
    _dd = ds["equity"] - _peak
    _mdd_idx = _dd.idxmin()
    _mdd_pct = float(_dd.min() / _peak[_mdd_idx] * 100) if _peak[_mdd_idx] > 0 else 0

    r1 = st.columns(4)
    r1[0].metric("총 거래", f"{len(df)}건"); r1[1].metric("승률", f"{win_rate:.1f}%")
    r1[2].metric("평균 수익", f"${_avg_win:,.2f}"); r1[3].metric("평균 손실", f"${_avg_loss:,.2f}")
    r2 = st.columns(4)
    r2[0].metric("수익 팩터", f"{_pf}"); r2[1].metric("평균 레버리지", f"{avg_lev:.1f}x")
    r2[2].metric("총 수수료", f"${total_fee:,.2f}"); r2[3].metric("MDD", f"{_mdd_pct:.1f}%")

    # ── 거래소별 성과 ────────────────────────────
    if multi_ex:
        st.markdown('<div class="section-hdr">거래소별 성과</div>', unsafe_allow_html=True)
        ex_cols = st.columns(df["exchange"].nunique())
        for i, (en, grp) in enumerate(df.groupby("exchange")):
            with ex_cols[i]:
                eb = _ex_bal.get(en, 10_000)
                ex_current = max(0, eb + float(grp["net_pnl"].sum()))
                ep = ex_current - eb
                ew = len(grp[grp["pnl_usdt"] > 0]) / len(grp) * 100
                ex_roi = (ep / eb * 100) if eb > 0 else 0
                color = _EX_COLOR.get(en, "#6b8aff")
                pc = _C["profit"] if ep >= 0 else _C["loss"]
                eps = "+" if ep >= 0 else ""
                st.markdown(
                    f'<div style="border-left:3px solid {color};padding:10px 14px;background:rgba(255,255,255,0.02);border-radius:0 10px 10px 0">'
                    f'<div style="font-size:13px;color:#7b7b9e;font-weight:600">{en}</div>'
                    f'<div style="font-size:22px;font-family:JetBrains Mono;color:{pc};font-weight:700">${ex_current:,.2f}</div>'
                    f'<div style="font-size:14px;font-family:JetBrains Mono;color:{pc}">{ex_roi:+.2f}% ({eps}${abs(ep):,.2f})</div>'
                    f'<div style="font-size:12px;color:#7b7b9e;margin-top:4px">초기 ${eb:,.0f} · {len(grp)}건 · 승률 {ew:.1f}%</div></div>',
                    unsafe_allow_html=True)

    # ── 종목별 차트 ──────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-hdr">종목별 손익</div>', unsafe_allow_html=True)
        sp = df.groupby("symbol")["net_pnl"].sum().sort_values()
        fig_s = go.Figure(go.Bar(x=sp.values, y=sp.index, orientation="h",
            marker_color=[_C["profit"] if v >= 0 else _C["loss"] for v in sp.values],
            text=[f"${v:+,.0f}" for v in sp.values], textposition="auto", textfont=dict(size=11)))
        fig_s.update_layout(**_CHART, xaxis_title="USDT", yaxis_title="")
        st.plotly_chart(fig_s, width="stretch")
    with col_r:
        st.markdown('<div class="section-hdr">종목별 승률</div>', unsafe_allow_html=True)
        ss = df.groupby("symbol").apply(lambda g: pd.Series({"wr": len(g[g["pnl_usdt"] > 0]) / len(g) * 100, "n": len(g)})).sort_values("wr")
        fig_w = go.Figure(go.Bar(x=ss["wr"], y=ss.index, orientation="h",
            marker_color=[_C["profit"] if v >= 50 else _C["loss"] for v in ss["wr"]],
            text=[f"{v:.0f}% ({int(n)}건)" for v, n in zip(ss["wr"], ss["n"])], textposition="auto", textfont=dict(size=11)))
        fig_w.add_vline(x=50, line_dash="dash", line_color="#4a4a6a")
        fig_w.update_layout(**_CHART, xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig_w, width="stretch")

    # ── 시간 패턴 ────────────────────────────────
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown('<div class="section-hdr">요일별 손익</div>', unsafe_allow_html=True)
        df["weekday"] = df["datetime"].dt.day_name()
        d_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        d_pnl = df.groupby("weekday")["net_pnl"].sum().reindex(d_order, fill_value=0)
        fig_d = go.Figure(go.Bar(x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], y=d_pnl.values,
            marker_color=[_C["profit"] if v >= 0 else _C["loss"] for v in d_pnl.values],
            text=[f"${v:+,.0f}" for v in d_pnl.values], textposition="outside", textfont=dict(size=10)))
        fig_d.update_layout(**_CHART, xaxis_title="", yaxis_title="USDT")
        st.plotly_chart(fig_d, width="stretch")
    with col_r2:
        st.markdown('<div class="section-hdr">시간대별 거래 (KST)</div>', unsafe_allow_html=True)
        df["hour_kst"] = (df["datetime"].dt.hour + 9) % 24
        hc = df.groupby("hour_kst").size().reindex(range(24), fill_value=0)
        hp = df.groupby("hour_kst")["net_pnl"].sum().reindex(range(24), fill_value=0)
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=list(range(24)), y=hc.values, name="거래수", marker_color=_C["neutral"]))
        fig_h.add_trace(go.Scatter(x=list(range(24)), y=hp.values, name="PnL", mode="lines+markers",
            line=dict(color=_C["primary"], width=2), marker=dict(size=4), yaxis="y2"))
        fig_h.update_layout(**_CHART, yaxis=dict(title="거래수", side="left"), yaxis2=dict(title="PnL", side="right", overlaying="y"),
            xaxis_title="시간 (KST)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode="overlay")
        st.plotly_chart(fig_h, width="stretch")

    # ── PnL 분포 ─────────────────────────────────
    st.markdown('<div class="section-hdr">개별 거래 PnL 분포</div>', unsafe_allow_html=True)
    _pnl_vals = df["net_pnl"].values
    _bin_size = max((float(_pnl_vals.max()) - float(_pnl_vals.min())) / 40, 1)
    _neg_edges = list(np.arange(0, float(_pnl_vals.min()) - _bin_size, -_bin_size))[::-1]
    _pos_edges = list(np.arange(0, float(_pnl_vals.max()) + _bin_size, _bin_size))
    _edges = sorted(set(_neg_edges + _pos_edges))
    _colors, _counts, _mids = [], [], []
    for j in range(len(_edges) - 1):
        lo, hi = _edges[j], _edges[j + 1]
        mid = (lo + hi) / 2
        _mids.append(mid); _counts.append(int(((df["net_pnl"] >= lo) & (df["net_pnl"] < hi)).sum()))
        _colors.append(_C["profit"] if mid >= 0 else _C["loss"])
    fig_hist = go.Figure(go.Bar(x=_mids, y=_counts, width=_bin_size * 0.9, marker_color=_colors, showlegend=False,
        hovertemplate="PnL: %{x:,.0f}<br>빈도: %{y}<extra></extra>"))
    fig_hist.update_layout(**_CHART, xaxis_title="PnL (USDT)", yaxis_title="빈도", bargap=0.05)
    st.plotly_chart(fig_hist, width="stretch")

    # ── 거래 내역 ────────────────────────────────
    st.markdown('<div class="section-hdr">거래 내역</div>', unsafe_allow_html=True)
    with st.expander(f"전체 거래 내역 ({len(df)}건)", expanded=False):
        fdf = df.copy()
        fdf["datetime"] = fdf["datetime"].astype(str)
        st.dataframe(fdf.sort_values("datetime", ascending=False).reset_index(drop=True), width="stretch", height=420)

    with st.expander("수동 거래 추가", expanded=False):
        with st.form("add_trade"):
            r0 = st.columns(5)
            with r0[0]: add_ex = st.selectbox("거래소", EXCHANGES, key="add_ex")
            with r0[1]: add_dt = st.text_input("일시", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
            with r0[2]: add_sym = st.selectbox("종목", SYMBOLS, key="add_sym")
            with r0[3]: add_side = st.selectbox("방향", SIDES, key="add_side")
            with r0[4]: add_lev = st.number_input("레버리지", 1, 125, 10, key="add_lev")
            r1 = st.columns(4)
            with r1[0]: add_entry = st.number_input("진입가", min_value=0.0, value=85000.0, format="%.4f")
            with r1[1]: add_exit = st.number_input("청산가", min_value=0.0, value=86000.0, format="%.4f")
            with r1[2]: add_qty = st.number_input("수량(USDT)", min_value=1.0, value=500.0, format="%.2f")
            with r1[3]: add_pnl = st.number_input("손익(USDT)", value=0.0, format="%.2f")
            r2x = st.columns(4)
            with r2x[0]: add_fee = st.number_input("수수료", min_value=0.0, value=1.0, format="%.2f")
            with r2x[1]: add_hold = st.number_input("보유(분)", min_value=1, value=60)
            with r2x[2]: add_type = st.selectbox("주문유형", ["MARKET", "LIMIT"], key="add_type")
            with r2x[3]: add_sl = st.checkbox("스탑로스")
            if st.form_submit_button("추가", width="stretch"):
                nid = st.session_state.trade_id_counter; st.session_state.trade_id_counter += 1
                nr = pd.DataFrame([{"id": nid, "exchange": add_ex, "datetime": add_dt, "symbol": add_sym, "side": add_side,
                    "leverage": int(add_lev), "entry_price": add_entry, "exit_price": add_exit, "quantity_usdt": add_qty,
                    "pnl_usdt": add_pnl, "fee_usdt": add_fee, "holding_minutes": add_hold, "order_type": add_type, "stoploss_set": add_sl}])
                st.session_state.trades = pd.concat([st.session_state.trades, nr], ignore_index=True)
                st.success(f"#{nid} 추가!"); st.rerun()


# ══════════════════════════════════════════════════
# TAB 2: AI Report
# ══════════════════════════════════════════════════
with tab_detail:
    # ── 교차 분석 데이터 계산 ──────────────────────
    _deep = aggregate_deep_data(df, st.session_state.deposits)

    if not _deep:
        st.info("데이터가 부족합니다.")
        st.stop()

    # ── AI 요약 (Haiku) ──────────────────────────
    def _run_detail_summary():
        _key = os.getenv("ANTHROPIC_API_KEY", "")
        if not _key:
            try: _key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception: _key = ""
        if not _key: return
        try:
            _basic = aggregate_data(st.session_state.trades.copy(), st.session_state.deposits, "통합")
            result = generate_detail_summary(_basic, _deep, _key)
            st.session_state.detail_summary = result["summary"]
        except Exception as e:
            st.session_state.detail_summary = f"요약 생성 실패: {e}"

    _sc1, _sc2 = st.columns([1, 5])
    with _sc1:
        st.button("AI 요약", width="stretch", type="primary",
                  disabled=not api_key_claude, key="btn_detail_summary", on_click=_run_detail_summary)
    with _sc2:
        if st.session_state.detail_summary:
            st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#151525);border:1px solid #1e1e35;'
                f'border-radius:10px;padding:14px 20px;font-size:14px;line-height:1.8">'
                f'{st.session_state.detail_summary.replace("$", "USD ")}</div>', unsafe_allow_html=True)
        else:
            st.caption("AI가 상세 분석 데이터를 요약합니다. (Haiku, 약 $0.002)")

    # ══════════════════════════════════════════════
    # Section 1: 매매 스타일
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-hdr">매매 스타일</div>', unsafe_allow_html=True)

    # 스타일 판별
    _hold_bands = {h["hold_band"]: h["trades"] for h in _deep.get("holding_time_bands", [])}
    _total_ht = sum(_hold_bands.values()) or 1
    _style_map = {"스캘핑(<10분)": "Scalper", "데이트레이딩(10분-1일)": "Day Trader",
                  "스윙(1일-30일)": "Swing Trader", "장기(30일+)": "Position Trader"}
    _dominant = max(_hold_bands, key=_hold_bands.get) if _hold_bands else "데이트레이딩(10분-1일)"
    _style_label = _style_map.get(_dominant, "Day Trader")
    _dominant_pct = round(_hold_bands.get(_dominant, 0) / _total_ht * 100)
    _is_mixed = _dominant_pct < 50

    _day_groups = df.groupby(df["datetime"].dt.date).size()
    _trades_per_day = round(float(_day_groups.mean()), 1)
    _avg_hold = float(df["holding_minutes"].mean())
    if _avg_hold < 10: _hold_str = f"{_avg_hold:.1f}분"
    elif _avg_hold < 60: _hold_str = f"{_avg_hold:.0f}분"
    elif _avg_hold < 1440: _hold_str = f"{_avg_hold / 60:.1f}시간"
    else: _hold_str = f"{_avg_hold / 1440:.1f}일"

    _style_color = {"Scalper": "#ff4d6a", "Day Trader": "#6b8aff", "Swing Trader": "#a78bfa", "Position Trader": "#00e59b"}

    col_style, col_dist = st.columns([2, 3])
    with col_style:
        _sc = _style_color.get(_style_label, "#6b8aff")
        _mix_tag = ' <span style="color:#ffbb33;font-size:12px">(혼합형)</span>' if _is_mixed else ""
        st.markdown(f"""<div class="style-card">
          <div style="font-size:11px;color:#7b7b9e;text-transform:uppercase;letter-spacing:1.5px">Trading Style</div>
          <div style="font-size:32px;font-weight:700;color:{_sc};margin:8px 0">{_style_label}{_mix_tag}</div>
          <div style="display:flex;gap:24px;margin-top:12px">
            <div><span style="color:#7b7b9e;font-size:11px">평균 보유</span><br><span style="font-family:JetBrains Mono;font-size:18px;font-weight:600">{_hold_str}</span></div>
            <div><span style="color:#7b7b9e;font-size:11px">일 평균 거래</span><br><span style="font-family:JetBrains Mono;font-size:18px;font-weight:600">{_trades_per_day}건</span></div>
            <div><span style="color:#7b7b9e;font-size:11px">평균 레버리지</span><br><span style="font-family:JetBrains Mono;font-size:18px;font-weight:600">{avg_lev:.1f}x</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with col_dist:
        # 보유시간 분포 바
        _band_labels = ["스캘핑(<10분)", "데이트레이딩(10분-1일)", "스윙(1일-30일)", "장기(30일+)"]
        _band_short = ["스캘핑", "데이트레이딩", "스윙", "장기"]
        _band_colors = ["#ff4d6a", "#6b8aff", "#a78bfa", "#00e59b"]
        _band_vals = [_hold_bands.get(b, 0) for b in _band_labels]
        _band_pcts = [v / _total_ht * 100 for v in _band_vals]

        fig_hold = go.Figure()
        for i, (lbl, pct, clr) in enumerate(zip(_band_short, _band_pcts, _band_colors)):
            if pct > 0:
                fig_hold.add_trace(go.Bar(x=[pct], y=[""], orientation="h", name=lbl,
                    marker_color=clr, text=f"{lbl} {pct:.0f}%" if pct > 8 else "",
                    textposition="inside", textfont=dict(size=11, color="white"),
                    hovertemplate=f"{lbl}: {pct:.1f}% ({_band_vals[i]}건)<extra></extra>"))
        fig_hold.update_layout(**{**_CHART, "height": 80, "margin": dict(l=0, r=0, t=0, b=0)},
            barmode="stack", showlegend=False,
            xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False))
        st.plotly_chart(fig_hold, width="stretch")

        # 레버리지 분포
        _lev_bands = _deep.get("leverage_bands", [])
        if _lev_bands:
            fig_lev = go.Figure()
            for lb in _lev_bands:
                clr = _C["profit"] if lb["avg_pnl"] >= 0 else _C["loss"]
                fig_lev.add_trace(go.Bar(x=[lb["leverage_band"]], y=[lb["avg_pnl"]], name=lb["leverage_band"],
                    marker_color=clr, text=f"승률 {lb['win_rate']*100:.0f}%<br>{lb['trades']}건",
                    textposition="outside", textfont=dict(size=10), showlegend=False,
                    hovertemplate=f"레버리지 {lb['leverage_band']}<br>건당 평균: {lb['avg_pnl']:+.1f} USD<br>승률: {lb['win_rate']*100:.0f}%<extra></extra>"))
            fig_lev.update_layout(**{**_CHART, "height": 200, "margin": dict(l=40, r=10, t=8, b=36)},
                xaxis_title="레버리지 구간", yaxis_title="건당 평균 PnL (USD)")
            st.plotly_chart(fig_lev, width="stretch")

    # ══════════════════════════════════════════════
    # Section 2: 패턴 발견
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-hdr">패턴 발견</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    # 종목 × 방향 성과
    with col_p1:
        st.caption("종목 x 방향 손익")
        _sym_side_all = _deep.get("symbol_side_worst", []) + _deep.get("symbol_side_best", [])
        if _sym_side_all:
            _seen = set()
            _sym_side_dedup = []
            for s in _sym_side_all:
                k = (s["symbol"], s["side"])
                if k not in _seen:
                    _seen.add(k)
                    _sym_side_dedup.append(s)

            fig_ss = go.Figure()
            for side, clr in [("LONG", _C["profit"]), ("SHORT", _C["loss"])]:
                side_data = [s for s in _sym_side_dedup if s["side"] == side]
                fig_ss.add_trace(go.Bar(
                    x=[s.get("symbol", "") for s in side_data],
                    y=[s["pnl"] for s in side_data],
                    name=side, marker_color=clr, opacity=0.85,
                    text=[f"WR {s['win_rate']*100:.0f}%" for s in side_data],
                    textposition="outside", textfont=dict(size=9),
                    hovertemplate="%{x} " + side + "<br>PnL: %{y:,.0f} USD<extra></extra>"))
            fig_ss.update_layout(**{**_CHART, "height": 280}, barmode="group",
                yaxis_title="PnL (USD)", xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_ss, width="stretch")
        else:
            st.caption("데이터 부족")

    # 시간대별 성과 히트맵
    with col_p2:
        st.caption("시간대별 성과 (KST)")
        _hr_pnl = df.groupby("hour_kst")["net_pnl"].sum().reindex(range(24), fill_value=0)

        fig_hr = go.Figure()
        _hr_colors = [_C["profit"] if v >= 0 else _C["loss"] for v in _hr_pnl.values]
        fig_hr.add_trace(go.Bar(x=list(range(24)), y=_hr_pnl.values, marker_color=_hr_colors,
            showlegend=False, hovertemplate="KST %{x}시<br>PnL: %{y:,.0f} USD<extra></extra>"))
        fig_hr.update_layout(**{**_CHART, "height": 280}, xaxis_title="시간 (KST)", yaxis_title="PnL (USD)",
            xaxis=dict(dtick=2))
        st.plotly_chart(fig_hr, width="stretch")

    # Row 2: 종목×요일 히트맵 + 보유시간별 성과
    col_p3, col_p4 = st.columns(2)

    with col_p3:
        st.caption("종목 x 요일 손익 히트맵")
        _day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _sym_day_pivot = df.pivot_table(index="symbol", columns=df["datetime"].dt.day_name().str[:3],
            values="net_pnl", aggfunc="sum", fill_value=0).reindex(columns=_day_order, fill_value=0)
        if not _sym_day_pivot.empty:
            _zmax = max(abs(_sym_day_pivot.values.min()), abs(_sym_day_pivot.values.max()), 1)
            fig_hm = go.Figure(go.Heatmap(
                z=_sym_day_pivot.values, x=_day_order, y=_sym_day_pivot.index.tolist(),
                colorscale=[[0, "#ff4d6a"], [0.5, "#1a1a2e"], [1, "#00e59b"]],
                zmin=-_zmax, zmax=_zmax,
                text=[[f"{v:+,.0f}" for v in row] for row in _sym_day_pivot.values],
                texttemplate="%{text}", textfont=dict(size=10),
                hovertemplate="%{y} %{x}<br>PnL: %{z:,.0f} USD<extra></extra>",
                colorbar=dict(title="USD", thickness=12)))
            fig_hm.update_layout(**{**_CHART, "height": 280, "margin": dict(l=80, r=10, t=8, b=36)},
                xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_hm, width="stretch")
        else:
            st.caption("데이터 부족")

    with col_p4:
        st.caption("보유시간별 성과")
        _hb = _deep.get("holding_time_bands", [])
        if _hb:
            fig_hb = go.Figure()
            for h in _hb:
                clr = _C["profit"] if h["pnl"] >= 0 else _C["loss"]
                fig_hb.add_trace(go.Bar(x=[h["hold_band"].split("(")[0]], y=[h["pnl"]], marker_color=clr,
                    text=f"WR {h['win_rate']*100:.0f}%<br>{h['trades']}건",
                    textposition="outside", textfont=dict(size=10), showlegend=False,
                    hovertemplate=f"{h['hold_band']}<br>PnL: {h['pnl']:+,.0f} USD<br>승률: {h['win_rate']*100:.0f}%<extra></extra>"))
            fig_hb.update_layout(**{**_CHART, "height": 280, "margin": dict(l=40, r=10, t=8, b=36)},
                yaxis_title="PnL (USD)")
            st.plotly_chart(fig_hb, width="stretch")
        else:
            st.caption("데이터 부족")

    # ══════════════════════════════════════════════
    # Section 3: 매매 습관
    # ══════════════════════════════════════════════
    st.markdown('<div class="section-hdr">매매 습관</div>', unsafe_allow_html=True)

    _revenge = _deep.get("revenge_trading", {})
    _tilt = _deep.get("post_tilt_behavior", {})
    _winstreak = _deep.get("post_win_streak", {})
    _holdcomp = _deep.get("holding_comparison", {})

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)

    with col_h1:
        _rv_count = _revenge.get("count", 0)
        _rv_pnl = _revenge.get("total_pnl", 0)
        _rv_esc = _revenge.get("escalated_pct", 0)
        _rv_color = _C["loss"] if _rv_pnl < 0 else _C["profit"]
        st.markdown(f"""<div class="habit-card">
          <div class="lbl">복수매매</div>
          <div class="val" style="color:{_rv_color}">{_rv_count}건</div>
          <div class="sub" style="color:{_rv_color}">PnL {_rv_pnl:+,.0f} USD</div>
          <div class="sub" style="color:#7b7b9e">에스컬레이션 {_rv_esc:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with col_h2:
        _tilt_cnt = _tilt.get("tilt_count", 0)
        _tilt_pnl = _tilt.get("avg_next_pnl", 0)
        _tilt_lev = _tilt.get("avg_next_leverage", 0)
        _tc = _C["profit"] if _tilt_pnl >= 0 else _C["loss"]
        st.markdown(f"""<div class="habit-card">
          <div class="lbl">연패 후 반응</div>
          <div class="val" style="color:{_tc}">{_tilt_cnt}회</div>
          <div class="sub" style="color:{_tc}">다음 거래 평균 {_tilt_pnl:+,.1f} USD</div>
          <div class="sub" style="color:#7b7b9e">레버리지 {_tilt_lev:.1f}x</div>
        </div>""", unsafe_allow_html=True)

    with col_h3:
        _ws_cnt = _winstreak.get("count", 0)
        _ws_pnl = _winstreak.get("avg_next_pnl", 0)
        _wc = _C["profit"] if _ws_pnl >= 0 else _C["loss"]
        st.markdown(f"""<div class="habit-card">
          <div class="lbl">연승 후 반응</div>
          <div class="val" style="color:{_wc}">{_ws_cnt}회</div>
          <div class="sub" style="color:{_wc}">다음 거래 평균 {_ws_pnl:+,.1f} USD</div>
        </div>""", unsafe_allow_html=True)

    with col_h4:
        _wh = _holdcomp.get("avg_win_hold_min", 0)
        _lh = _holdcomp.get("avg_loss_hold_min", 0)
        def _fmt_min(m):
            if m < 60: return f"{m:.0f}분"
            if m < 1440: return f"{m/60:.1f}시간"
            return f"{m/1440:.1f}일"
        _hold_diff_color = _C["loss"] if _lh > _wh * 1.5 else "#7b7b9e"
        st.markdown(f"""<div class="habit-card">
          <div class="lbl">보유시간 비교</div>
          <div style="margin-top:12px">
            <div style="color:{_C['profit']};font-family:JetBrains Mono;font-size:18px;font-weight:600">{_fmt_min(_wh)}</div>
            <div style="color:#7b7b9e;font-size:11px">수익 거래 평균</div>
          </div>
          <div style="margin-top:8px">
            <div style="color:{_hold_diff_color};font-family:JetBrains Mono;font-size:18px;font-weight:600">{_fmt_min(_lh)}</div>
            <div style="color:#7b7b9e;font-size:11px">손실 거래 평균</div>
          </div>
        </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════
# TAB 3: What-if 시뮬레이션
# ══════════════════════════════════════════════════
with tab_whatif:
    st.markdown("### What-if 시뮬레이션")
    st.caption("특정 조건의 거래를 제외했을 때 성과가 어떻게 달라지는지 확인합니다.")

    wf = df.copy()

    # ── 최적 조합 계산 (디폴트 추천) ─────────────────
    _base_pnl = float(wf["net_pnl"].sum())

    # ── 최적 조합 탐색: 각 필터를 개별 적용해서 PnL 개선되는 것만 디폴트 ──

    # 방향: 제외 시 PnL이 개선되는 경우만
    _def_side = []
    for side in ["LONG", "SHORT"]:
        if float(wf[wf["side"] != side]["net_pnl"].sum()) > _base_pnl:
            _def_side.append(side)
    # 둘 다 제외하면 0건이므로 더 효과 큰 1개만
    if len(_def_side) == 2:
        _side_gains = {s: float(wf[wf["side"] != s]["net_pnl"].sum()) - _base_pnl for s in _def_side}
        _def_side = [max(_side_gains, key=_side_gains.get)]

    # 종목: 해당 종목이 순손실인 것만
    _sym_pnl = wf.groupby("symbol")["net_pnl"].sum()
    _def_sym = sorted(_sym_pnl[_sym_pnl < 0].index.tolist(), key=lambda s: float(_sym_pnl[s]))

    # 시간대: 순손실 시간대 상위 3개
    _hr_pnl_wf = wf.groupby("hour_kst")["net_pnl"].sum()
    _loss_hrs = _hr_pnl_wf[_hr_pnl_wf < 0].nsmallest(3)
    _def_hrs = sorted([int(h) for h in _loss_hrs.index.tolist()])

    # 복수매매: 순손실인 경우만
    _wf_sorted_all = wf.sort_values("datetime").reset_index()
    _revenge_idx_all = set()
    for i in range(1, len(_wf_sorted_all)):
        prev, curr = _wf_sorted_all.iloc[i - 1], _wf_sorted_all.iloc[i]
        gap = (curr["datetime"] - prev["datetime"]).total_seconds() / 60
        if prev["pnl_usdt"] < 0 and gap <= 5:
            _revenge_idx_all.add(curr["index"])
    _revenge_pnl = float(wf.loc[wf.index.isin(_revenge_idx_all), "net_pnl"].sum()) if _revenge_idx_all else 0
    _def_revenge = _revenge_pnl < 0

    # 전체 조합 적용 시 오히려 악화되면 디폴트 비우기
    _test = wf.copy()
    if _def_side: _test = _test[~_test["side"].astype(str).isin(_def_side)]
    if _def_sym: _test = _test[~_test["symbol"].astype(str).isin(_def_sym)]
    if _def_hrs: _test = _test[~_test["hour_kst"].isin(_def_hrs)]
    if _def_revenge and _revenge_idx_all: _test = _test[~_test.index.isin(_revenge_idx_all)]
    if _test.empty or float(_test["net_pnl"].sum()) <= _base_pnl:
        # 조합이 악화시키면 개별 최대 개선 항목만 남기기
        _candidates = []
        if _def_side:
            _t = wf[~wf["side"].astype(str).isin(_def_side)]
            if not _t.empty: _candidates.append(("side", float(_t["net_pnl"].sum()) - _base_pnl))
        if _def_sym:
            _t = wf[~wf["symbol"].astype(str).isin(_def_sym)]
            if not _t.empty: _candidates.append(("sym", float(_t["net_pnl"].sum()) - _base_pnl))
        if _def_hrs:
            _t = wf[~wf["hour_kst"].isin(_def_hrs)]
            if not _t.empty: _candidates.append(("hrs", float(_t["net_pnl"].sum()) - _base_pnl))
        if _def_revenge and _revenge_idx_all:
            _t = wf[~wf.index.isin(_revenge_idx_all)]
            if not _t.empty: _candidates.append(("rev", float(_t["net_pnl"].sum()) - _base_pnl))
        _best = max(_candidates, key=lambda x: x[1]) if _candidates else None
        if not _best or _best[1] <= 0:
            _def_side, _def_sym, _def_hrs, _def_revenge = [], [], [], False
        else:
            if _best[0] != "side": _def_side = []
            if _best[0] != "sym": _def_sym = []
            if _best[0] != "hrs": _def_hrs = []
            if _best[0] != "rev": _def_revenge = False

    # ── 필터 조건 ──────────────────────────────────
    wf_c1, wf_c2, wf_c3, wf_c4, wf_c5 = st.columns(5)
    with wf_c1:
        wf_exclude_side = st.multiselect("방향 제외", ["LONG", "SHORT"], default=_def_side, key="wf_side")
    with wf_c2:
        wf_exclude_sym = st.multiselect("종목 제외", sorted(df["symbol"].unique().tolist()), default=_def_sym, key="wf_sym")
    with wf_c3:
        _hrs = list(range(24))
        wf_exclude_hrs = st.multiselect("시간대 제외 (KST)", _hrs, default=_def_hrs, key="wf_hrs",
            format_func=lambda x: f"{x}시")
    with wf_c4:
        _lev_options = sorted(wf["leverage"].unique().tolist())
        wf_exclude_lev = st.multiselect("레버리지 제외", _lev_options, key="wf_lev",
            format_func=lambda x: f"{int(x)}x")
    with wf_c5:
        wf_no_revenge = st.checkbox("복수매매 제외", value=_def_revenge, key="wf_revenge")

    # 복수매매 인덱스 (체크 시만 적용)
    _wf_sorted = wf.sort_values("datetime").reset_index()
    _revenge_idx = set()
    if wf_no_revenge:
        for i in range(1, len(_wf_sorted)):
            prev, curr = _wf_sorted.iloc[i - 1], _wf_sorted.iloc[i]
            gap = (curr["datetime"] - prev["datetime"]).total_seconds() / 60
            if prev["pnl_usdt"] < 0 and gap <= 5:
                _revenge_idx.add(curr["index"])

    # 필터 적용
    wf_filtered = wf.copy()
    if wf_exclude_side:
        wf_filtered = wf_filtered[~wf_filtered["side"].astype(str).isin(wf_exclude_side)]
    if wf_exclude_sym:
        wf_filtered = wf_filtered[~wf_filtered["symbol"].astype(str).isin(wf_exclude_sym)]
    if wf_exclude_hrs:
        wf_filtered = wf_filtered[~wf_filtered["hour_kst"].isin(wf_exclude_hrs)]
    if wf_exclude_lev:
        wf_filtered = wf_filtered[~wf_filtered["leverage"].isin(wf_exclude_lev)]
    if wf_no_revenge and _revenge_idx:
        wf_filtered = wf_filtered[~wf_filtered.index.isin(_revenge_idx)]

    _any_filter = bool(wf_exclude_side or wf_exclude_sym or wf_exclude_hrs or wf_exclude_lev or wf_no_revenge)
    _removed = len(wf) - len(wf_filtered)

    if _any_filter and _removed > 0:
        st.info(f"{_removed}건 제외됨 ({len(wf)}건 → {len(wf_filtered)}건)")

    # ── Before / After 비교 ────────────────────────
    def _calc_stats(d):
        if d.empty:
            return {"trades": 0, "pnl": 0, "win_rate": 0, "pf": 0, "avg_win": 0, "avg_loss": 0}
        w = d[d["pnl_usdt"] > 0]
        l = d[d["pnl_usdt"] <= 0]
        gp = float(w["net_pnl"].sum()) if len(w) > 0 else 0
        gl = abs(float(l["net_pnl"].sum())) if len(l) > 0 else 1
        return {
            "trades": len(d),
            "pnl": float(d["net_pnl"].sum()),
            "win_rate": len(w) / len(d) * 100,
            "pf": round(gp / gl, 2) if gl > 0 else 0,
            "avg_win": float(w["net_pnl"].mean()) if len(w) > 0 else 0,
            "avg_loss": float(l["net_pnl"].mean()) if len(l) > 0 else 0,
        }

    _before = _calc_stats(wf)
    _after = _calc_stats(wf_filtered) if _any_filter else _before

    def _delta(v1, v2, fmt=",.0f", suffix=""):
        d = v2 - v1
        if abs(d) < 0.01: return None
        return f"{d:+{fmt}}{suffix}"

    st.markdown('<div class="section-hdr">Before / After</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("총 PnL", f"${_after['pnl']:,.2f}",
        delta=_delta(_before["pnl"], _after["pnl"], ",.2f") if _any_filter else None)
    mc2.metric("승률", f"{_after['win_rate']:.1f}%",
        delta=_delta(_before["win_rate"], _after["win_rate"], ".1f", "%") if _any_filter else None)
    mc3.metric("수익 팩터", f"{_after['pf']}",
        delta=_delta(_before["pf"], _after["pf"], ".2f") if _any_filter else None)

    mc4, mc5, mc6 = st.columns(3)
    mc4.metric("거래 수", f"{_after['trades']}건",
        delta=f"{_after['trades'] - _before['trades']}건" if _any_filter and _removed > 0 else None)
    mc5.metric("평균 수익", f"${_after['avg_win']:,.2f}",
        delta=_delta(_before["avg_win"], _after["avg_win"], ",.2f") if _any_filter else None)
    mc6.metric("평균 손실", f"${_after['avg_loss']:,.2f}",
        delta=_delta(_before["avg_loss"], _after["avg_loss"], ",.2f") if _any_filter else None)

    # ── Equity 비교 차트 ──────────────────────────
    if _any_filter and len(wf_filtered) > 0:
        st.markdown('<div class="section-hdr">자산 곡선 비교</div>', unsafe_allow_html=True)
        _init = list(_ex_bal.values())[0] if len(_ex_bal) == 1 else sum(_ex_bal.values())

        _wf_sorted_eq = wf.sort_values("datetime").copy()
        _wf_sorted_eq["eq_before"] = _init + _wf_sorted_eq["net_pnl"].cumsum()
        # After: 제외된 거래의 PnL을 0으로 처리 (같은 시간축 유지)
        _excluded_idx = set(wf.index) - set(wf_filtered.index)
        _wf_sorted_eq["pnl_after"] = _wf_sorted_eq["net_pnl"].copy()
        _wf_sorted_eq.loc[_wf_sorted_eq.index.isin(_excluded_idx), "pnl_after"] = 0
        _wf_sorted_eq["eq_after"] = _init + _wf_sorted_eq["pnl_after"].cumsum()

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(x=_wf_sorted_eq["datetime"], y=_wf_sorted_eq["eq_before"],
            mode="lines", name="Before", line=dict(color="#4a4a6a", width=1.5, dash="dot")))
        fig_wf.add_trace(go.Scatter(x=_wf_sorted_eq["datetime"], y=_wf_sorted_eq["eq_after"],
            mode="lines", name="After", line=dict(color=_C["profit"], width=2)))
        fig_wf.add_hline(y=_init, line_dash="dash", line_color="#4a4a6a", line_width=1)
        fig_wf.update_layout(**{**_CHART, "height": 340},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="USDT")
        st.plotly_chart(fig_wf, width="stretch")

        # 차이 요약
        _diff_pnl = _after["pnl"] - _before["pnl"]
        _diff_color = _C["profit"] if _diff_pnl > 0 else _C["loss"]
        if _diff_pnl > 0:
            _diff_msg = f"제외한 {_removed}건 때문에 ${_diff_pnl:,.2f} 손해 보고 있었습니다"
        else:
            _diff_msg = f"제외한 {_removed}건이 ${abs(_diff_pnl):,.2f} 벌어주고 있었습니다"
        st.markdown(f'<div style="text-align:center;font-size:14px;color:{_diff_color};margin:8px 0">{_diff_msg}</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 4: 매매 일지
# ══════════════════════════════════════════════════
with tab_journal:
    st.markdown("### 매매 일지")
    st.caption("일별/주별 거래 요약과 특이사항을 자동으로 정리합니다.")

    _jdf = df.copy()
    _jdf["date"] = _jdf["datetime"].dt.date

    # ── 주간 / 일간 토글 ──────────────────────────
    _j_mode = st.radio("단위", ["일별", "주별"], horizontal=True, key="j_mode")

    if _j_mode == "주별":
        _jdf["period"] = _jdf["datetime"].dt.to_period("W").apply(lambda x: x.start_time.date())
        _period_label = "주간"
    else:
        _jdf["period"] = _jdf["date"]
        _period_label = "일별"

    # ── 기간별 요약 테이블 ──────────────────────────
    _j_groups = _jdf.groupby("period")

    _j_summary = []
    for period, grp in _j_groups:
        wins = grp[grp["pnl_usdt"] > 0]
        pnl = float(grp["net_pnl"].sum())
        wr = len(wins) / len(grp) * 100 if len(grp) > 0 else 0

        # 특이사항 감지
        flags = []
        # 복수매매
        gs = grp.sort_values("datetime").reset_index(drop=True)
        revenge_cnt = 0
        for i in range(1, len(gs)):
            gap = (gs.iloc[i]["datetime"] - gs.iloc[i-1]["datetime"]).total_seconds() / 60
            if gs.iloc[i-1]["pnl_usdt"] < 0 and gap <= 5:
                revenge_cnt += 1
        if revenge_cnt >= 3:
            flags.append(f"복수매매 {revenge_cnt}건")

        # 큰 손실 (평균의 3배 이상)
        _avg_abs = float(grp["net_pnl"].abs().mean())
        _big_loss = grp[grp["net_pnl"] < -_avg_abs * 3]
        if len(_big_loss) > 0:
            flags.append(f"큰 손실 {len(_big_loss)}건")

        # 과매매 (하루 평균의 2배)
        if _j_mode == "일별" and len(grp) > _trades_per_day * 2:
            flags.append(f"과매매 ({len(grp)}건)")

        # 최고 수익 종목
        _best_sym = grp.groupby("symbol")["net_pnl"].sum()
        if len(_best_sym) > 0:
            _top = _best_sym.idxmax()
            _top_pnl = float(_best_sym.max())
            _worst = _best_sym.idxmin()
            _worst_pnl = float(_best_sym.min())

        _j_summary.append({
            "period": str(period),
            "trades": len(grp),
            "pnl": pnl,
            "win_rate": wr,
            "best": f"{_top} (+${_top_pnl:,.0f})" if len(_best_sym) > 0 and _top_pnl > 0 else "-",
            "worst": f"{_worst} (${_worst_pnl:,.0f})" if len(_best_sym) > 0 and _worst_pnl < 0 else "-",
            "flags": ", ".join(flags) if flags else "-",
        })

    _j_df = pd.DataFrame(_j_summary)
    if not _j_df.empty:
        _j_df = _j_df.sort_values("period", ascending=False)

        # ── PnL 타임라인 ──────────────────────────
        fig_j = go.Figure()
        _j_colors = [_C["profit"] if v >= 0 else _C["loss"] for v in _j_df["pnl"]]
        fig_j.add_trace(go.Bar(x=_j_df["period"], y=_j_df["pnl"], marker_color=_j_colors,
            text=[f"${v:+,.0f}" for v in _j_df["pnl"]], textposition="outside", textfont=dict(size=10),
            hovertemplate="%{x}<br>PnL: %{y:,.0f} USD<br><extra></extra>", name="PnL"))
        _j_df_sorted = _j_df.sort_values("period")
        fig_j.add_trace(go.Scatter(x=_j_df_sorted["period"], y=_j_df_sorted["trades"],
            mode="lines+markers", name="거래 수", yaxis="y2",
            line=dict(color=_C["secondary"], width=1.5), marker=dict(size=4)))
        fig_j.update_layout(**{**_CHART, "height": 280}, xaxis_title="", yaxis_title="PnL (USD)",
            yaxis2=dict(title="거래 수", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_j, width="stretch")

        # ── 요약 테이블 ────────────────────────────
        st.markdown(f'<div class="section-hdr">{_period_label} 상세</div>', unsafe_allow_html=True)

        for _, row in _j_df.iterrows():
            _ps = "+" if row["pnl"] >= 0 else ""
            with st.expander(f'{row["period"]}  |  {row["trades"]}건  |  {_ps}${row["pnl"]:,.0f}'):
                ec1, ec2, ec3, ec4, ec5 = st.columns(5)
                ec1.metric("PnL", f"${row['pnl']:,.2f}")
                ec2.metric("거래 수", f"{row['trades']}건")
                ec3.metric("승률", f"{row['win_rate']:.1f}%")
                ec4.metric("최고 종목", row["best"])
                ec5.metric("최저 종목", row["worst"])
                if row["flags"] != "-":
                    st.warning(f"특이사항: {row['flags']}")

                # 해당 기간 거래 목록
                if _j_mode == "주별":
                    _period_trades = _jdf[_jdf["period"] == pd.Timestamp(row["period"]).date()]
                else:
                    _period_trades = _jdf[_jdf["date"] == pd.Timestamp(row["period"]).date()]
                if not _period_trades.empty:
                    # ── 캔들스틱 차트 + 진입/청산 마커 ────
                    _pt = _period_trades.sort_values("datetime")
                    _top_sym = _pt.groupby("symbol").size().nlargest(3).index.tolist()
                    _timeframe = "15m" if _j_mode == "일별" else "4h"
                    _tf_label = "15분봉" if _j_mode == "일별" else "4시간봉"

                    # 기간 범위
                    _p_start = pd.Timestamp(row["period"])
                    if _j_mode == "주별":
                        _p_end = _p_start + pd.Timedelta(days=7)
                    else:
                        _p_end = _p_start + pd.Timedelta(days=1)

                    _has_exchange = False
                    for _sym in _top_sym:
                        _sym_trades = _pt[_pt["symbol"] == _sym]
                        if _sym_trades.empty:
                            continue

                        # 거래소 연결 시 캔들 데이터 가져오기
                        _ohlcv = pd.DataFrame()
                        if is_any_connected() and fetch_ohlcv is not None:
                            _ex_name = str(_sym_trades.iloc[0].get("exchange", ""))
                            _ex_info = st.session_state.connections.get(_ex_name)
                            if _ex_info:
                                _ohlcv = fetch_ohlcv(
                                    _ex_info["instance"], _sym,
                                    _timeframe, _p_start.to_pydatetime(), _p_end.to_pydatetime())
                                if not _ohlcv.empty:
                                    _has_exchange = True

                        fig_candle = go.Figure()

                        if not _ohlcv.empty:
                            # 캔들스틱
                            fig_candle.add_trace(go.Candlestick(
                                x=_ohlcv["datetime"], open=_ohlcv["open"],
                                high=_ohlcv["high"], low=_ohlcv["low"], close=_ohlcv["close"],
                                increasing_line_color=_C["profit"], decreasing_line_color=_C["loss"],
                                increasing_fillcolor=_C["profit"], decreasing_fillcolor=_C["loss"],
                                name=_tf_label, showlegend=False))

                        # 진입 마커
                        for _, t in _sym_trades.iterrows():
                            _tc = _C["profit"] if t["net_pnl"] > 0 else _C["loss"]
                            _arrow = "triangle-up" if t["side"] == "LONG" else "triangle-down"
                            # 진입점
                            fig_candle.add_trace(go.Scatter(
                                x=[t["datetime"]], y=[t["entry_price"]],
                                mode="markers+text", marker=dict(size=10, color=_tc, symbol=_arrow,
                                    line=dict(width=1, color="white")),
                                text=["IN"], textposition="top center", textfont=dict(size=9, color=_tc),
                                showlegend=False,
                                hovertemplate=f"{t['side']} 진입<br>가격: {t['entry_price']:,.2f}<br>레버리지: {t['leverage']}x<extra></extra>"))
                            # 청산점
                            fig_candle.add_trace(go.Scatter(
                                x=[t["datetime"] + pd.Timedelta(minutes=t["holding_minutes"])],
                                y=[t["exit_price"]],
                                mode="markers+text", marker=dict(size=10, color=_tc, symbol="x",
                                    line=dict(width=1, color="white")),
                                text=["OUT"], textposition="bottom center", textfont=dict(size=9, color=_tc),
                                showlegend=False,
                                hovertemplate=f"{t['side']} 청산<br>가격: {t['exit_price']:,.2f}<br>PnL: {t['net_pnl']:+,.2f} USD<extra></extra>"))
                            # 진입→청산 연결선
                            fig_candle.add_trace(go.Scatter(
                                x=[t["datetime"], t["datetime"] + pd.Timedelta(minutes=t["holding_minutes"])],
                                y=[t["entry_price"], t["exit_price"]],
                                mode="lines", line=dict(color=_tc, width=1.5, dash="dot"),
                                showlegend=False, hoverinfo="skip"))

                        _chart_title = f"{_sym} {_tf_label}" if not _ohlcv.empty else f"{_sym} 진입/청산"
                        fig_candle.update_layout(**{**_CHART, "height": 350},
                            title=dict(text=_chart_title, font=dict(size=13, color="#7b7b9e")),
                            yaxis_title="가격", xaxis_title="",
                            xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig_candle, width="stretch")

                    if not _has_exchange and is_any_connected():
                        st.caption(f"캔들 데이터를 가져올 수 없습니다. ({_tf_label})")
                    elif not is_any_connected():
                        st.caption("거래소 연결 시 캔들스틱 차트가 표시됩니다.")

                    # ── 거래 목록 테이블 ──────────────
                    _show = _period_trades[["datetime", "symbol", "side", "entry_price", "exit_price", "leverage", "pnl_usdt", "fee_usdt", "holding_minutes"]].copy()
                    _show["datetime"] = _show["datetime"].astype(str)
                    _show.columns = ["시간", "종목", "방향", "진입가", "청산가", "레버리지", "PnL", "수수료", "보유(분)"]
                    st.dataframe(_show.sort_values("시간", ascending=False), width="stretch", height=200)
