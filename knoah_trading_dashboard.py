"""
KNOAH 거래 분석 대시보드
========================
멀티 거래소 API 연동 + Claude AI 섹션별 인라인 분석
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from lib.config import (
    SYMBOLS, EXCHANGES, SIDES,
    CUSTOM_CSS, TRADE_COLUMNS, DEPOSIT_COLUMNS,
)
from lib.dummy import generate_trades, generate_deposits
from lib.analysis import (
    aggregate_data, call_claude_sections, generate_dummy_comments,
)

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
st.set_page_config(
    page_title="KNOAH Trading Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# AI 코멘트 스타일 추가
_EXTRA_CSS = """
<style>
.ai-comment {
    background: linear-gradient(90deg, rgba(107,138,255,0.08) 0%, rgba(107,138,255,0.02) 100%);
    border-left: 3px solid #6b8aff;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 4px 0 16px 0;
    font-size: 13px;
    color: #b0b0d0;
    line-height: 1.5;
}
.ai-comment::before {
    content: "🤖 ";
    font-size: 12px;
}
.ai-actions {
    background: linear-gradient(135deg, rgba(167,139,250,0.10) 0%, rgba(107,138,255,0.06) 100%);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
}
.ai-actions-title {
    font-size: 14px;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 10px;
}
.ai-action-item {
    font-size: 13px;
    color: #c0c0e0;
    padding: 4px 0 4px 8px;
    border-left: 2px solid #a78bfa44;
    margin: 6px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(_EXTRA_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════
_DEFAULTS = {
    "trades": pd.DataFrame(columns=TRADE_COLUMNS),
    "deposits": pd.DataFrame(columns=DEPOSIT_COLUMNS),
    "ai_comments": None,        # dict with section keys
    "trade_id_counter": 0,
    "connections": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════
def connected_exchanges() -> list:
    return list(st.session_state.connections.keys())


def is_any_connected() -> bool:
    return len(st.session_state.connections) > 0


def merge_exchange_trades(new_df: pd.DataFrame, exchange_name: str):
    existing = st.session_state.trades
    if existing.empty:
        st.session_state.trades = new_df
    else:
        other = existing[existing["exchange"] != exchange_name]
        st.session_state.trades = pd.concat([other, new_df], ignore_index=True)
    st.session_state.trades = st.session_state.trades.sort_values("datetime").reset_index(drop=True)
    st.session_state.trades["id"] = range(len(st.session_state.trades))
    st.session_state.trade_id_counter = len(st.session_state.trades)


def merge_exchange_deposits(new_df: pd.DataFrame, exchange_name: str):
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


def ai(key: str):
    """세션에 저장된 AI 코멘트 가져오기"""
    if st.session_state.ai_comments and key in st.session_state.ai_comments:
        return st.session_state.ai_comments[key]
    return None


def render_ai(key: str):
    """AI 코멘트 렌더링 (있으면)"""
    comment = ai(key)
    if comment:
        st.markdown(f'<div class="ai-comment">{comment}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# Plotly Theme
# ══════════════════════════════════════════════════
_CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#a0a0b8"),
    margin=dict(l=40, r=20, t=32, b=36),
    height=300,
)
_C = {
    "profit": "#00e59b", "loss": "#ff4d6a", "primary": "#6b8aff",
    "secondary": "#a78bfa", "neutral": "#4a4a6a",
}
_EX_COLOR = {
    "Binance": "#F0B90B", "Bybit": "#F7A600",
    "OKX": "#FFFFFF", "Bitget": "#00D4AA", "Demo": "#6b8aff",
}


# ══════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# KNOAH")
    st.caption("Trading Analysis Platform")

    # ── 연결된 거래소 ────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 거래소")

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
                        st.session_state.trades = st.session_state.trades[
                            st.session_state.trades["exchange"] != ex_name
                        ].reset_index(drop=True)
                    st.session_state.ai_comments = None
                    st.rerun()

    with st.expander("➕ 거래소 추가", expanded=not is_any_connected()):
        if not HAS_CCXT:
            st.warning("`pip install ccxt` 필요")
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
                            st.session_state.connections[exchange_name] = {
                                "instance": ex, "balance": result.get("total_usdt", 0), "msg": result["msg"],
                            }
                            st.success(f"{exchange_name} 연결!")
                            st.rerun()
                        else:
                            st.error(result["msg"])
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("API Key/Secret 필요")

    # ── 데이터 가져오기 ──────────────────────────
    if is_any_connected():
        st.markdown("---")
        fetch_days = st.number_input("조회 기간(일)", 7, 90, 30, key="fetch_days")
        fetch_targets = st.multiselect("대상", connected_exchanges(), default=connected_exchanges(), key="ft")
        if st.button("📥 거래 내역 가져오기", use_container_width=True, type="primary"):
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
                st.session_state.ai_comments = None
                st.rerun()

    # ── 더미 데이터 ──────────────────────────────
    st.markdown("---")
    st.markdown("### 🎲 더미 데이터")
    c_a, c_b = st.columns(2)
    with c_a:
        n_trades = st.number_input("건수", 10, 500, 100, step=10, key="n_trades")
    with c_b:
        n_days = st.number_input("기간(일)", 7, 90, 30, step=7, key="n_days")
    dummy_exchanges = st.multiselect("거래소", EXCHANGES, default=["Binance"], key="dummy_ex")

    # 거래소별 초기 잔고 설정
    ex_balances = {}
    for dex in (dummy_exchanges or ["Binance"]):
        ex_balances[dex] = st.number_input(
            f"{dex} 초기 잔고(USDT)", 100, 1_000_000, 10_000, step=1000,
            key=f"bal_{dex}",
        )

    if st.button("더미 데이터 생성", use_container_width=True):
        all_t, all_d = [], []
        exes = dummy_exchanges or ["Binance"]
        per_n = max(n_trades // len(exes), 10)
        for i, en in enumerate(exes):
            tdf = generate_trades(per_n, en, n_days, start_id=i * per_n)
            all_t.append(tdf)
            bal = ex_balances.get(en, 10_000)
            dep = generate_deposits(bal)
            dep.insert(1, "exchange", en)
            all_d.append(dep)
        combined = pd.concat(all_t, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        combined["id"] = range(len(combined))
        st.session_state.trades = combined
        st.session_state.deposits = pd.concat(all_d, ignore_index=True).reset_index(drop=True)
        st.session_state.trade_id_counter = len(combined)
        st.session_state.ai_comments = None
        st.rerun()

    if st.button("🗑 초기화", use_container_width=True):
        st.session_state.trades = pd.DataFrame(columns=TRADE_COLUMNS)
        st.session_state.deposits = pd.DataFrame(columns=DEPOSIT_COLUMNS)
        st.session_state.ai_comments = None
        st.session_state.trade_id_counter = 0
        st.rerun()

    # ── Claude API ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Claude API")
    _default_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key_claude = st.text_input(
        "API Key", type="password", value=_default_key,
        help=".env ANTHROPIC_API_KEY 기본값", key="api_key_claude",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#55556a;font-size:11px'>"
        "KNOAH v0.4 · Powered by Claude</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
st.markdown("# 📊 KNOAH Trading Analysis")

# ── 상태 바 + AI 분석 버튼 ───────────────────────
conn_names = connected_exchanges()
badges = " ".join(f'<span class="badge badge-success">{n}</span>' for n in conn_names) if conn_names else '<span class="badge badge-info">DEMO</span>'
has_data = not st.session_state.trades.empty
n_ex = int(st.session_state.trades["exchange"].nunique()) if has_data and "exchange" in st.session_state.trades.columns else 0

top_left, top_right = st.columns([4, 1])
with top_left:
    st.markdown(f'{badges} &nbsp; 거래소 **{n_ex}개** · 거래 **{len(st.session_state.trades)}건**', unsafe_allow_html=True)
with top_right:
    run_ai = st.button(
        "🤖 AI 분석" if not st.session_state.ai_comments else "🔄 재분석",
        use_container_width=True,
        type="primary",
        disabled=not has_data,
    )

# ── AI 분석 실행 ─────────────────────────────────
if run_ai and has_data:
    df_for_ai = st.session_state.trades.copy()
    aggregated = aggregate_data(df_for_ai, st.session_state.deposits, "통합")
    if not api_key_claude:
        st.toast("API 키 없음 → 더미 분석 사용", icon="⚠️")
        st.session_state.ai_comments = generate_dummy_comments(aggregated)
        st.rerun()
    else:
        with st.spinner("Claude가 분석 중..."):
            try:
                st.session_state.ai_comments = call_claude_sections(aggregated, api_key_claude)
                st.rerun()
            except Exception as e:
                st.error(f"AI 분석 실패: {e}")
                st.session_state.ai_comments = generate_dummy_comments(aggregated)
                st.rerun()

# ── 거래소 필터 ──────────────────────────────────
avail_ex = sorted(st.session_state.trades["exchange"].unique().tolist()) if has_data and "exchange" in st.session_state.trades.columns else []
if len(avail_ex) > 1:
    selected_ex = st.multiselect("📊 분석 대상", avail_ex, default=avail_ex, key="gf")
else:
    selected_ex = avail_ex


def filtered_trades() -> pd.DataFrame:
    df = st.session_state.trades
    if df.empty or not selected_ex:
        return df
    if "exchange" in df.columns:
        return df[df["exchange"].isin(selected_ex)]
    return df


# ── 데이터 없으면 안내 ───────────────────────────
if not has_data:
    st.markdown("---")
    st.info("👈 사이드바에서 거래소를 연결하거나 더미 데이터를 생성해주세요.")
    st.stop()

df = filtered_trades().copy()
if df.empty:
    st.warning("선택된 거래소에 데이터가 없습니다.")
    st.stop()

df["datetime"] = pd.to_datetime(df["datetime"])
df["net_pnl"] = df["pnl_usdt"] - df["fee_usdt"]

total_pnl = float(df["net_pnl"].sum())
win_count = len(df[df["pnl_usdt"] > 0])
win_rate = win_count / len(df) * 100
total_fee = float(df["fee_usdt"].sum())
avg_lev = float(df["leverage"].mean())
init_bal = float(st.session_state.deposits["amount_usdt"].sum()) if not st.session_state.deposits.empty else 10_000
roi = total_pnl / init_bal * 100 if init_bal > 0 else 0


# ══════════════════════════════════════════════════
# SECTION: 핵심 지표
# ══════════════════════════════════════════════════
st.markdown('<div class="section-hdr">핵심 지표</div>', unsafe_allow_html=True)
render_ai("overview")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("총 거래", f"{len(df)}건")
m2.metric("총 손익", f"${total_pnl:,.2f}", delta=f"{roi:+.1f}%")
m3.metric("승률", f"{win_rate:.1f}%")
m4.metric("평균 레버리지", f"{avg_lev:.1f}x")
m5.metric("총 수수료", f"${total_fee:,.2f}")
m6.metric("순 ROI", f"{roi:+.1f}%")


# ── 거래소별 카드 ────────────────────────────────
if "exchange" in df.columns and df["exchange"].nunique() > 1:
    st.markdown('<div class="section-hdr">거래소별 성과</div>', unsafe_allow_html=True)
    render_ai("exchange_comparison")
    ex_cols = st.columns(df["exchange"].nunique())
    for i, (en, grp) in enumerate(df.groupby("exchange")):
        with ex_cols[i]:
            ep = float((grp["pnl_usdt"] - grp["fee_usdt"]).sum())
            ew = len(grp[grp["pnl_usdt"] > 0]) / len(grp) * 100
            color = _EX_COLOR.get(en, "#6b8aff")
            pnl_color = _C["profit"] if ep >= 0 else _C["loss"]
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:10px 14px;'
                f'background:rgba(255,255,255,0.02);border-radius:0 10px 10px 0;margin-bottom:8px">'
                f'<div style="font-size:13px;color:#7b7b9e;font-weight:600">{en}</div>'
                f'<div style="font-size:22px;font-family:JetBrains Mono;color:{pnl_color};font-weight:700">'
                f'{"+" if ep >= 0 else ""}${ep:,.2f}</div>'
                f'<div style="font-size:12px;color:#7b7b9e">'
                f'{len(grp)}건 · 승률 {ew:.1f}% · 평균 {grp["leverage"].mean():.1f}x</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
elif "exchange" in df.columns:
    render_ai("exchange_comparison")


# ══════════════════════════════════════════════════
# SECTION: 자산 곡선 (수익금 + 수익률)
# ══════════════════════════════════════════════════
st.markdown('<div class="section-hdr">자산 곡선</div>', unsafe_allow_html=True)
render_ai("equity_curve")

multi_ex = "exchange" in df.columns and df["exchange"].nunique() > 1

# 거래소별 초기 잔고 매핑
_dep = st.session_state.deposits
_ex_bal = {}
if not _dep.empty and "exchange" in _dep.columns:
    _ex_bal = _dep.groupby("exchange")["amount_usdt"].sum().to_dict()

# ── 거래소별 equity를 각각 계산 후 합산 (잔고 0 바닥 개별 적용) ──
# 각 거래소는 독립 계좌 → 잔고가 0이면 해당 거래소에서 거래 불가
# 통합 equity = sum(거래소별 equity)
if multi_ex:
    _ex_equity_frames = []
    for en, grp in df.groupby("exchange"):
        g = grp.sort_values("datetime").copy()
        eb = _ex_bal.get(en, 10_000)
        g["ex_equity"] = (eb + g["net_pnl"].cumsum()).clip(lower=0)
        g["ex_roi"] = ((g["ex_equity"] / eb) - 1) * 100
        g["_ex_bal"] = eb
        _ex_equity_frames.append(g[["datetime", "exchange", "net_pnl", "symbol", "side", "ex_equity", "ex_roi", "_ex_bal"]])
    _all_eq = pd.concat(_ex_equity_frames).sort_values("datetime")
    # 통합: 시점별 거래소 equity 합산
    ds = _all_eq.copy()
    # 각 거래소의 마지막 equity를 forward-fill해서 시점별 합산
    _pivot = _all_eq.pivot_table(index="datetime", columns="exchange", values="ex_equity", aggfunc="last")
    _pivot = _pivot.ffill().fillna(method="bfill")
    _pivot["total"] = _pivot.sum(axis=1)
    _total_init = sum(_ex_bal.get(en, 10_000) for en in df["exchange"].unique())
    _pivot["total_roi"] = ((_pivot["total"] / _total_init) - 1) * 100
    # ds에 통합 equity/roi 매핑
    ds = ds.merge(_pivot[["total", "total_roi"]].reset_index(), on="datetime", how="left")
    ds["equity"] = ds["total"]
    ds["roi_pct"] = ds["total_roi"]
else:
    ds = df.sort_values("datetime").copy()
    eb = list(_ex_bal.values())[0] if _ex_bal else init_bal
    ds["equity"] = (eb + ds["net_pnl"].cumsum()).clip(lower=0)
    ds["roi_pct"] = ((ds["equity"] / eb) - 1) * 100
    ds["ex_equity"] = ds["equity"]
    _total_init = eb

eq_col1, eq_col2 = st.columns(2)

# ── 수익금 그래프 ────────────────────────────────
with eq_col1:
    st.markdown("**수익금 (Equity)**")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=ds["datetime"], y=ds["equity"], mode="lines",
        name="통합" if multi_ex else "자산",
        line=dict(color=_C["primary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(107,138,255,0.06)",
    ))
    if multi_ex:
        for en, grp in _all_eq.groupby("exchange"):
            fig_eq.add_trace(go.Scatter(
                x=grp["datetime"], y=grp["ex_equity"], mode="lines", name=en,
                line=dict(color=_EX_COLOR.get(en, "#888"), width=1.5, dash="dot"),
            ))
    # 거래 마커
    wins_ds = ds[ds["net_pnl"] >= 0]
    loss_ds = ds[ds["net_pnl"] < 0]
    fig_eq.add_trace(go.Scatter(
        x=wins_ds["datetime"], y=wins_ds["equity"], mode="markers", name="수익",
        marker=dict(color=_C["profit"], size=4, opacity=0.5),
        hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        customdata=list(zip(wins_ds["symbol"], wins_ds["side"], [f"+${v:,.0f}" for v in wins_ds["net_pnl"]])),
    ))
    fig_eq.add_trace(go.Scatter(
        x=loss_ds["datetime"], y=loss_ds["equity"], mode="markers", name="손실",
        marker=dict(color=_C["loss"], size=4, opacity=0.5),
        hovertemplate="%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        customdata=list(zip(loss_ds["symbol"], loss_ds["side"], [f"${v:,.0f}" for v in loss_ds["net_pnl"]])),
    ))
    _ref_bal = _total_init if multi_ex else (list(_ex_bal.values())[0] if _ex_bal else init_bal)
    fig_eq.add_hline(y=_ref_bal, line_dash="dash", line_color="#4a4a6a", line_width=1,
                     annotation_text="초기 잔고", annotation_font_color="#7b7b9e")
    fig_eq.update_layout(**{**_CHART, "height": 360}, xaxis_title="", yaxis_title="USDT",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_eq, use_container_width=True)

# ── 수익률 그래프 ────────────────────────────────
with eq_col2:
    st.markdown("**수익률 (ROI %)**")
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=ds["datetime"], y=ds["roi_pct"], mode="lines",
        name="통합" if multi_ex else "ROI",
        line=dict(color=_C["secondary"], width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
    ))
    if multi_ex:
        for en, grp in _all_eq.groupby("exchange"):
            fig_roi.add_trace(go.Scatter(
                x=grp["datetime"], y=grp["ex_roi"], mode="lines", name=en,
                line=dict(color=_EX_COLOR.get(en, "#888"), width=1.5, dash="dot"),
            ))
    fig_roi.add_hline(y=0, line_dash="dash", line_color="#4a4a6a", line_width=1)
    fig_roi.update_layout(**{**_CHART, "height": 360}, xaxis_title="", yaxis_title="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_roi, use_container_width=True)


# ══════════════════════════════════════════════════
# SECTION: 종목별
# ══════════════════════════════════════════════════
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-hdr">종목별 손익</div>', unsafe_allow_html=True)
    render_ai("symbol_pnl")
    sp = df.groupby("symbol")["net_pnl"].sum().sort_values()
    fig_s = go.Figure(go.Bar(
        x=sp.values, y=sp.index, orientation="h",
        marker_color=[_C["profit"] if v >= 0 else _C["loss"] for v in sp.values],
        text=[f"${v:+,.0f}" for v in sp.values],
        textposition="auto", textfont=dict(size=11),
    ))
    fig_s.update_layout(**_CHART, xaxis_title="USDT", yaxis_title="")
    st.plotly_chart(fig_s, use_container_width=True)

with col_r:
    st.markdown('<div class="section-hdr">종목별 승률</div>', unsafe_allow_html=True)
    render_ai("symbol_winrate")
    ss = df.groupby("symbol").apply(
        lambda g: pd.Series({"wr": len(g[g["pnl_usdt"] > 0]) / len(g) * 100, "n": len(g)})
    ).sort_values("wr")
    fig_w = go.Figure(go.Bar(
        x=ss["wr"], y=ss.index, orientation="h",
        marker_color=[_C["profit"] if v >= 50 else _C["loss"] for v in ss["wr"]],
        text=[f"{v:.0f}% ({int(n)}건)" for v, n in zip(ss["wr"], ss["n"])],
        textposition="auto", textfont=dict(size=11),
    ))
    fig_w.add_vline(x=50, line_dash="dash", line_color="#4a4a6a")
    fig_w.update_layout(**_CHART, xaxis_title="%", yaxis_title="")
    st.plotly_chart(fig_w, use_container_width=True)


# ══════════════════════════════════════════════════
# SECTION: 시간 패턴
# ══════════════════════════════════════════════════
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown('<div class="section-hdr">요일별 손익</div>', unsafe_allow_html=True)
    render_ai("weekday_pnl")
    df["weekday"] = df["datetime"].dt.day_name()
    d_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d_pnl = df.groupby("weekday")["net_pnl"].sum().reindex(d_order, fill_value=0)
    d_lbl = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_d = go.Figure(go.Bar(
        x=d_lbl, y=d_pnl.values,
        marker_color=[_C["profit"] if v >= 0 else _C["loss"] for v in d_pnl.values],
        text=[f"${v:+,.0f}" for v in d_pnl.values],
        textposition="outside", textfont=dict(size=10),
    ))
    fig_d.update_layout(**_CHART, xaxis_title="", yaxis_title="USDT")
    st.plotly_chart(fig_d, use_container_width=True)

with col_r2:
    st.markdown('<div class="section-hdr">시간대별 거래</div>', unsafe_allow_html=True)
    render_ai("hourly_pattern")
    df["hour"] = df["datetime"].dt.hour
    hc = df.groupby("hour").size().reindex(range(24), fill_value=0)
    hp = df.groupby("hour")["net_pnl"].sum().reindex(range(24), fill_value=0)
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=list(range(24)), y=hc.values, name="거래수", marker_color=_C["neutral"]))
    fig_h.add_trace(go.Scatter(
        x=list(range(24)), y=hp.values, name="PnL", mode="lines+markers",
        line=dict(color=_C["primary"], width=2), marker=dict(size=4), yaxis="y2",
    ))
    fig_h.update_layout(
        **_CHART,
        yaxis=dict(title="거래수", side="left"),
        yaxis2=dict(title="PnL", side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════
# SECTION: PnL 분포
# ══════════════════════════════════════════════════
st.markdown('<div class="section-hdr">개별 거래 PnL 분포</div>', unsafe_allow_html=True)
render_ai("pnl_distribution")

fig_hist = go.Figure()
# 단일 히스토그램 + 색상으로 수익/손실 구분 (겹침 원천 차단)
import numpy as _np
_pnl = df["net_pnl"].values
_bin_size = max((float(_pnl.max()) - float(_pnl.min())) / 40, 1)
# 0을 경계로 bin edges 생성
_neg_edges = list(_np.arange(0, float(_pnl.min()) - _bin_size, -_bin_size))[::-1]
_pos_edges = list(_np.arange(0, float(_pnl.max()) + _bin_size, _bin_size))
_edges = sorted(set(_neg_edges + _pos_edges))
# 각 bin의 카운트와 색상 계산
_colors, _counts, _mids = [], [], []
for j in range(len(_edges) - 1):
    lo, hi = _edges[j], _edges[j + 1]
    mid = (lo + hi) / 2
    cnt = int(((df["net_pnl"] >= lo) & (df["net_pnl"] < hi)).sum())
    _mids.append(mid)
    _counts.append(cnt)
    _colors.append(_C["profit"] if mid >= 0 else _C["loss"])
fig_hist.add_trace(go.Bar(
    x=_mids, y=_counts, width=_bin_size * 0.9,
    marker_color=_colors, showlegend=False,
    hovertemplate="PnL: %{x:,.0f}<br>빈도: %{y}<extra></extra>",
))
fig_hist.update_layout(
    **_CHART, xaxis_title="PnL (USDT)", yaxis_title="빈도", bargap=0.05,
)
st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════
# SECTION: 액션 아이템 (AI 분석 결과가 있을 때)
# ══════════════════════════════════════════════════
actions = ai("action_items")
if actions and isinstance(actions, list):
    st.markdown("---")
    items_html = "".join(f'<div class="ai-action-item">{item}</div>' for item in actions)
    st.markdown(
        f'<div class="ai-actions">'
        f'<div class="ai-actions-title">🎯 AI 추천 액션 플랜</div>'
        f'{items_html}'
        f'<div style="font-size:11px;color:#666;margin-top:10px">'
        f'⚠️ 본 분석은 투자 조언이 아니며, 매매 습관 개선을 위한 참고 자료입니다.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════
# SECTION: 거래 내역 (토글)
# ══════════════════════════════════════════════════
st.markdown('<div class="section-hdr">거래 내역</div>', unsafe_allow_html=True)

with st.expander(f"📋 전체 거래 내역 ({len(df)}건)", expanded=False):
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        ex_f = st.multiselect("거래소", sorted(df["exchange"].unique()) if "exchange" in df.columns else [], key="ex_f")
    with fcol2:
        sym_f = st.multiselect("종목", sorted(df["symbol"].unique()), key="sym_f")
    with fcol3:
        side_f = st.multiselect("방향", ["LONG", "SHORT"], key="side_f")
    with fcol4:
        pnl_f = st.selectbox("손익", ["전체", "수익만", "손실만"], key="pnl_f")

    fdf = df.copy()
    if ex_f:
        fdf = fdf[fdf["exchange"].isin(ex_f)]
    if sym_f:
        fdf = fdf[fdf["symbol"].isin(sym_f)]
    if side_f:
        fdf = fdf[fdf["side"].isin(side_f)]
    if pnl_f == "수익만":
        fdf = fdf[fdf["pnl_usdt"] > 0]
    elif pnl_f == "손실만":
        fdf = fdf[fdf["pnl_usdt"] <= 0]

    fdf["datetime"] = fdf["datetime"].astype(str)
    st.data_editor(
        fdf.sort_values("datetime", ascending=False).reset_index(drop=True),
        use_container_width=True, num_rows="fixed", height=420,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "exchange": st.column_config.TextColumn("거래소", disabled=True, width="small"),
            "datetime": st.column_config.TextColumn("시간"),
            "symbol": st.column_config.SelectboxColumn("종목", options=SYMBOLS),
            "side": st.column_config.SelectboxColumn("방향", options=SIDES),
            "leverage": st.column_config.NumberColumn("레버리지", format="%dx"),
            "entry_price": st.column_config.NumberColumn("진입가", format="%.4f"),
            "exit_price": st.column_config.NumberColumn("청산가", format="%.4f"),
            "quantity_usdt": st.column_config.NumberColumn("수량", format="$%.2f"),
            "pnl_usdt": st.column_config.NumberColumn("손익", format="$%.2f"),
            "fee_usdt": st.column_config.NumberColumn("수수료", format="$%.2f"),
            "holding_minutes": st.column_config.NumberColumn("보유(분)", format="%d분"),
            "order_type": st.column_config.SelectboxColumn("유형", options=["MARKET", "LIMIT"]),
            "stoploss_set": st.column_config.CheckboxColumn("SL"),
        },
        key="trade_editor",
    )

with st.expander("➕ 수동 거래 추가", expanded=False):
    with st.form("add_trade"):
        r0 = st.columns(5)
        with r0[0]:
            add_ex = st.selectbox("거래소", EXCHANGES, key="add_ex")
        with r0[1]:
            add_dt = st.text_input("일시", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
        with r0[2]:
            add_sym = st.selectbox("종목", SYMBOLS, key="add_sym")
        with r0[3]:
            add_side = st.selectbox("방향", SIDES, key="add_side")
        with r0[4]:
            add_lev = st.number_input("레버리지", 1, 125, 10, key="add_lev")
        r1 = st.columns(4)
        with r1[0]:
            add_entry = st.number_input("진입가", min_value=0.0, value=85000.0, format="%.4f")
        with r1[1]:
            add_exit = st.number_input("청산가", min_value=0.0, value=86000.0, format="%.4f")
        with r1[2]:
            add_qty = st.number_input("수량(USDT)", min_value=1.0, value=500.0, format="%.2f")
        with r1[3]:
            add_pnl = st.number_input("손익(USDT)", value=0.0, format="%.2f")
        r2 = st.columns(4)
        with r2[0]:
            add_fee = st.number_input("수수료", min_value=0.0, value=1.0, format="%.2f")
        with r2[1]:
            add_hold = st.number_input("보유(분)", min_value=1, value=60)
        with r2[2]:
            add_type = st.selectbox("주문유형", ["MARKET", "LIMIT"], key="add_type")
        with r2[3]:
            add_sl = st.checkbox("스탑로스")
        if st.form_submit_button("추가", use_container_width=True):
            nid = st.session_state.trade_id_counter
            st.session_state.trade_id_counter += 1
            nr = pd.DataFrame([{
                "id": nid, "exchange": add_ex, "datetime": add_dt,
                "symbol": add_sym, "side": add_side, "leverage": int(add_lev),
                "entry_price": add_entry, "exit_price": add_exit,
                "quantity_usdt": add_qty, "pnl_usdt": add_pnl,
                "fee_usdt": add_fee, "holding_minutes": add_hold,
                "order_type": add_type, "stoploss_set": add_sl,
            }])
            st.session_state.trades = pd.concat([st.session_state.trades, nr], ignore_index=True)
            st.success(f"#{nid} 추가!")
            st.rerun()
