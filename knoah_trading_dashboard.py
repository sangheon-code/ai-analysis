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
from lib.analysis import aggregate_data, aggregate_deep_data, call_claude_deep_report, get_api_balance

try:
    from lib.exchanges import (
        ccxt, create_exchange, test_connection,
        fetch_trades as exchange_fetch_trades,
        fetch_deposits_withdrawals,
    )
    HAS_CCXT = ccxt is not None
except ImportError:
    HAS_CCXT = False


# ══════════════════════════════════════════════════
# Page Config & CSS
# ══════════════════════════════════════════════════
st.set_page_config(page_title="KNOAH Trading Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""<style>
.deep-report { background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35; border-radius: 14px; padding: 28px 32px;
    line-height: 1.9; font-size: 14px; }
.deep-report h3 { color: #a78bfa; margin-top: 28px; font-size: 17px; border-bottom: 1px solid #1e1e35; padding-bottom: 8px; }
.deep-report strong { color: #e8e8ed; }
.deep-report table { width: 100%; border-collapse: collapse; margin: 12px 0; }
.deep-report th, .deep-report td { padding: 6px 12px; border: 1px solid #1e1e35; font-size: 13px; }
.deep-report th { background: #1a1a2e; color: #7b7b9e; }
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
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# KNOAH")
    st.caption("Trading Analysis Platform")

    st.markdown("---")
    st.markdown("### 🔗 거래소")
    if st.session_state.connections:
        for ex_name, info in list(st.session_state.connections.items()):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: st.markdown(f'<span class="conn-dot on"></span> **{ex_name}**', unsafe_allow_html=True)
            with c2: st.caption(f"${info.get('balance', 0):,.2f}")
            with c3:
                if st.button("✕", key=f"rm_{ex_name}"):
                    del st.session_state.connections[ex_name]
                    if not st.session_state.trades.empty and "exchange" in st.session_state.trades.columns:
                        st.session_state.trades = st.session_state.trades[st.session_state.trades["exchange"] != ex_name].reset_index(drop=True)
                    st.session_state.ai_deep_report = None
                    st.rerun()

    with st.expander("➕ 거래소 추가", expanded=not is_any_connected()):
        if not HAS_CCXT: st.warning("`pip install ccxt` 필요")
        exchange_name = st.selectbox("거래소", EXCHANGES, key="sel_exchange")
        api_key_exchange = st.text_input("API Key", type="password", key="api_key_exchange")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        passphrase = None
        if exchange_name in ("OKX", "Bitget"):
            passphrase = st.text_input("Passphrase", type="password", key="passphrase")
        if st.button("🔗 연결", use_container_width=True, disabled=not HAS_CCXT):
            if api_key_exchange and api_secret:
                with st.spinner(f"{exchange_name} 연결 중..."):
                    try:
                        ex = create_exchange(exchange_name, api_key_exchange, api_secret, passphrase)
                        result = test_connection(ex)
                        if result["ok"]:
                            st.session_state.connections[exchange_name] = {"instance": ex, "balance": result.get("total_usdt", 0), "msg": result["msg"]}
                            st.success(f"{exchange_name} 연결!")
                            st.rerun()
                        else: st.error(result["msg"])
                    except Exception as e: st.error(str(e))
            else: st.warning("API Key/Secret 필요")

    if is_any_connected():
        st.markdown("---")
        fetch_days = st.number_input("조회 기간(일)", 7, 1095, 90, key="fetch_days")
        fetch_targets = st.multiselect("대상", connected_exchanges(), default=connected_exchanges(), key="ft")
        if st.button("📥 거래 내역 가져오기", use_container_width=True, type="primary"):
            total = 0
            for en in fetch_targets:
                info = st.session_state.connections[en]
                with st.spinner(f"{en}..."):
                    try:
                        tdf = exchange_fetch_trades(info["instance"], en, days=fetch_days)
                        if not tdf.empty: merge_exchange_trades(tdf, en); total += len(tdf)
                        ddf = fetch_deposits_withdrawals(info["instance"], days=fetch_days)
                        if not ddf.empty: merge_exchange_deposits(ddf, en)
                        st.success(f"{en}: {len(tdf)}건")
                    except Exception as e: st.error(f"{en}: {e}")
            if total > 0: st.session_state.ai_deep_report = None; st.rerun()

    st.markdown("---")
    st.markdown("### 🎲 더미 데이터")
    c_a, c_b = st.columns(2)
    with c_a: n_trades = st.number_input("건수", 10, 500, 100, step=10, key="n_trades")
    with c_b: n_days = st.number_input("기간(일)", 7, 90, 30, step=7, key="n_days")
    dummy_exchanges = st.multiselect("거래소", EXCHANGES, default=["Binance"], key="dummy_ex")
    ex_balances = {}
    for dex in (dummy_exchanges or ["Binance"]):
        ex_balances[dex] = st.number_input(f"{dex} 초기 잔고(USDT)", 100, 1_000_000, 10_000, step=1000, key=f"bal_{dex}")

    if st.button("더미 데이터 생성", use_container_width=True):
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

    if st.button("🗑 초기화", use_container_width=True):
        for k in ["trades", "deposits", "ai_deep_report", "trade_id_counter"]:
            st.session_state[k] = _DEFAULTS[k]
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
st.markdown(f'{badges} &nbsp; 거래소 **{n_ex}개** · 거래 **{len(st.session_state.trades)}건**', unsafe_allow_html=True)

if not has_data:
    st.info("👈 사이드바에서 거래소를 연결하거나 더미 데이터를 생성해주세요.")
    st.stop()

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

# ── Equity 계산 ──────────────────────────────────
multi_ex = "exchange" in df.columns and df["exchange"].nunique() > 1
if multi_ex:
    _frames = []
    for en, grp in df.groupby("exchange"):
        g = grp.sort_values("datetime").copy()
        eb = _ex_bal.get(en, 10_000)
        g["ex_equity"] = (eb + g["net_pnl"].cumsum()).clip(lower=0)
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
    ds["equity"] = (eb + ds["net_pnl"].cumsum()).clip(lower=0)
    _total_init = eb


# ══════════════════════════════════════════════════
# TABS: 대시보드 | AI Report
# ══════════════════════════════════════════════════
tab_dashboard, tab_ai = st.tabs(["대시보드", "AI Report"])


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
    fig_eq.add_trace(go.Scatter(x=ds["datetime"], y=ds["equity"], mode="lines", name="통합" if multi_ex else "자산",
        line=dict(color=_C["primary"], width=2.5), fill="tonexty" if False else None))
    if multi_ex:
        for en, grp in _all_eq.groupby("exchange"):
            fig_eq.add_trace(go.Scatter(x=grp["datetime"], y=grp["ex_equity"], mode="lines", name=en,
                line=dict(color=_EX_COLOR.get(en, "#888"), width=1.5, dash="dot")))
    _ref_bal = _total_init if multi_ex else (list(_ex_bal.values())[0] if _ex_bal else init_bal_total)
    fig_eq.add_hline(y=_ref_bal, line_dash="dash", line_color="#4a4a6a", line_width=1, annotation_text="초기 잔고", annotation_font_color="#7b7b9e")
    # y축 범위: 데이터 범위에 맞춰서 변동이 잘 보이도록
    _eq_min = float(ds["equity"].min())
    _eq_max = float(ds["equity"].max())
    _eq_pad = max((_eq_max - _eq_min) * 0.1, 1)
    fig_eq.update_layout(**{**_CHART, "height": 380}, xaxis_title="", yaxis_title="USDT",
        yaxis_range=[_eq_min - _eq_pad, _eq_max + _eq_pad],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_eq, use_container_width=True)

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
        st.plotly_chart(fig_s, use_container_width=True)
    with col_r:
        st.markdown('<div class="section-hdr">종목별 승률</div>', unsafe_allow_html=True)
        ss = df.groupby("symbol").apply(lambda g: pd.Series({"wr": len(g[g["pnl_usdt"] > 0]) / len(g) * 100, "n": len(g)})).sort_values("wr")
        fig_w = go.Figure(go.Bar(x=ss["wr"], y=ss.index, orientation="h",
            marker_color=[_C["profit"] if v >= 50 else _C["loss"] for v in ss["wr"]],
            text=[f"{v:.0f}% ({int(n)}건)" for v, n in zip(ss["wr"], ss["n"])], textposition="auto", textfont=dict(size=11)))
        fig_w.add_vline(x=50, line_dash="dash", line_color="#4a4a6a")
        fig_w.update_layout(**_CHART, xaxis_title="%", yaxis_title="")
        st.plotly_chart(fig_w, use_container_width=True)

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
        st.plotly_chart(fig_d, use_container_width=True)
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
        st.plotly_chart(fig_h, use_container_width=True)

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
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── 거래 내역 ────────────────────────────────
    st.markdown('<div class="section-hdr">거래 내역</div>', unsafe_allow_html=True)
    with st.expander(f"전체 거래 내역 ({len(df)}건)", expanded=False):
        fdf = df.copy()
        fdf["datetime"] = fdf["datetime"].astype(str)
        st.dataframe(fdf.sort_values("datetime", ascending=False).reset_index(drop=True), use_container_width=True, height=420)

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
            if st.form_submit_button("추가", use_container_width=True):
                nid = st.session_state.trade_id_counter; st.session_state.trade_id_counter += 1
                nr = pd.DataFrame([{"id": nid, "exchange": add_ex, "datetime": add_dt, "symbol": add_sym, "side": add_side,
                    "leverage": int(add_lev), "entry_price": add_entry, "exit_price": add_exit, "quantity_usdt": add_qty,
                    "pnl_usdt": add_pnl, "fee_usdt": add_fee, "holding_minutes": add_hold, "order_type": add_type, "stoploss_set": add_sl}])
                st.session_state.trades = pd.concat([st.session_state.trades, nr], ignore_index=True)
                st.success(f"#{nid} 추가!"); st.rerun()


# ══════════════════════════════════════════════════
# TAB 2: AI Report
# ══════════════════════════════════════════════════
with tab_ai:
    st.markdown("### AI Deep Report")
    st.caption("Claude가 교차 분석 데이터를 기반으로 숨은 패턴, 심리 분석, 맞춤 전략을 제공합니다.")

    if not api_key_claude:
        st.warning("사이드바에서 Claude API Key를 입력해주세요.")
    elif len(df) < 10:
        st.warning("거래 데이터가 10건 미만입니다. 더 많은 데이터가 있으면 분석 정확도가 올라갑니다.")

    def _run_deep_report():
        _key = os.getenv("ANTHROPIC_API_KEY", "")
        if not _key:
            try: _key = st.secrets.get("ANTHROPIC_API_KEY", "")
            except Exception: _key = ""
        if not _key:
            return
        _df = st.session_state.trades.copy()
        _deps = st.session_state.deposits
        _basic = aggregate_data(_df, _deps, "통합")
        _deep = aggregate_deep_data(_df, _deps)
        try:
            result = call_claude_deep_report(_basic, _deep, _key)
            st.session_state.ai_deep_report = result["report"]
            st.session_state.ai_last_cost = result
        except Exception as e:
            st.session_state.ai_deep_report = f"리포트 생성 실패: {e}"
            st.session_state.ai_last_cost = None

    # API 잔고 조회
    _balance_info = None
    if api_key_claude:
        _balance_info = get_api_balance(api_key_claude)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.button("리포트 생성", use_container_width=True, type="primary",
                  disabled=not api_key_claude or len(df) < 1, key="btn_deep_report", on_click=_run_deep_report)
    with col_info:
        info_parts = ["Claude Sonnet · 약 10~15초 소요"]
        if _balance_info and _balance_info["ok"]:
            info_parts.append(f"잔고: {_balance_info['balance']}")
        info_parts.append("회당 약 \\$0.01-0.05")
        last = st.session_state.get("ai_last_cost")
        if last and isinstance(last, dict):
            info_parts.append(f"마지막: {last['input_tokens']+last['output_tokens']:,}토큰 \\${last['cost_usd']:.4f}")
        st.caption(" · ".join(info_parts))

    if st.session_state.ai_deep_report:
        # $를 USD로 치환 (LaTeX 해석 방지 + 마크다운 깨짐 방지)
        safe = st.session_state.ai_deep_report.replace("$", "USD ")
        st.markdown("---")
        st.markdown(safe)
        st.divider()
        st.download_button("리포트 다운로드 (.md)", data=st.session_state.ai_deep_report,
            file_name=f"knoah_deep_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown", use_container_width=True)
