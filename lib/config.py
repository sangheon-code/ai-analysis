"""
KNOAH 거래 분석 – 설정 및 상수
"""

# ── 지원 종목 / 거래소 ──────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
    "BNBUSDT", "AVAXUSDT", "ADAUSDT", "MATICUSDT", "LINKUSDT",
]

EXCHANGES = ["Binance", "Bybit", "OKX", "Bitget"]
SIDES = ["LONG", "SHORT"]

# ── 심볼별 대략 기준가 (더미 데이터용) ──────────
PRICE_MAP = {
    "BTCUSDT": 85_000, "ETHUSDT": 3_200, "SOLUSDT": 180,
    "DOGEUSDT": 0.25, "XRPUSDT": 2.5, "BNBUSDT": 600,
    "AVAXUSDT": 35, "ADAUSDT": 0.8, "MATICUSDT": 1.2, "LINKUSDT": 18,
}

# ── Claude 시스템 프롬프트 ───────────────────────
SYSTEM_PROMPT = """당신은 KNOAH 플랫폼의 트레이딩 분석 AI입니다.
유저의 거래 데이터를 분석하여 객관적인 진단과 개선 방향을 제시합니다.

## 역할
- 감정 없이 데이터 기반으로 매매 패턴을 진단합니다.
- 구체적인 수치를 인용하며 분석합니다.
- 투자 조언이 아닌 매매 습관 진단에 집중합니다.

## 출력 구조 (마크다운, 반드시 이 순서로)

### 1. 한줄 진단
유저의 매매 스타일을 한 문장으로 요약.

### 2. 핵심 지표
총 손익, 승률, 수익 팩터, 최대 드로다운을 표로 정리.

### 3. 종목별 분석
수익/손실이 큰 종목 위주로 패턴 분석.

### 4. 잘한 점 (1~2개)
데이터에서 긍정적인 패턴을 찾아 구체적으로.

### 5. 개선 필요 (2~3개, 우선순위순)
가장 손익에 영향이 큰 문제부터. 각 항목마다:
- 문제 진단 (수치 근거)
- 왜 문제인지 설명
- 구체적 개선 행동 1가지

### 6. 다음 30일 액션 플랜
바로 실행 가능한 행동 2~3개.

## 규칙
- 한국어로 작성
- "~하세요" 대신 "~해보는 건 어떨까요" 같은 부드러운 제안 톤
- 절대 특정 코인 매수/매도를 권유하지 않음
- 마지막에 반드시 면책 문구 포함: "⚠️ 본 분석은 투자 조언이 아니며, 매매 습관 개선을 위한 참고 자료입니다."
"""

# ── 거래 DataFrame 컬럼 정의 ─────────────────────
TRADE_COLUMNS = [
    "id", "exchange", "datetime", "symbol", "side", "leverage",
    "entry_price", "exit_price", "quantity_usdt",
    "pnl_usdt", "fee_usdt", "holding_minutes",
    "order_type", "stoploss_set",
]

DEPOSIT_COLUMNS = ["id", "datetime", "type", "amount_usdt"]

# ── Custom CSS ───────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Global ── */
.block-container { max-width: 1280px; padding-top: 1.5rem; }
section[data-testid="stSidebar"] { width: 340px !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35;
    border-radius: 14px;
    padding: 18px 22px;
    transition: border-color 0.2s;
}
div[data-testid="stMetric"]:hover { border-color: #3d3d6b; }
div[data-testid="stMetric"] label {
    color: #7b7b9e !important; font-size: 12px !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important; font-weight: 700 !important;
}

/* ── Section headers ── */
.section-hdr {
    font-size: 13px; font-weight: 700;
    color: #7b7b9e; letter-spacing: 1.8px;
    text-transform: uppercase;
    margin: 32px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e1e35;
}

/* ── Analysis result card ── */
.analysis-card {
    background: linear-gradient(135deg, #0f0f1a 0%, #151525 100%);
    border: 1px solid #1e1e35;
    border-radius: 14px;
    padding: 28px 32px;
    line-height: 1.8;
    font-size: 14px;
}
.analysis-card h3 { color: #6b8aff; margin-top: 24px; font-size: 16px; }
.analysis-card strong { color: #e8e8ed; }

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-success { background: rgba(0,229,155,0.12); color: #00e59b; }
.badge-danger  { background: rgba(255,77,106,0.12); color: #ff4d6a; }
.badge-info    { background: rgba(77,139,255,0.12); color: #4d8bff; }
.badge-warn    { background: rgba(255,187,51,0.12); color: #ffbb33; }

/* ── Connection indicator ── */
.conn-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
}
.conn-dot.on  { background: #00e59b; box-shadow: 0 0 6px #00e59b88; }
.conn-dot.off { background: #ff4d6a; }

/* ── Chart container ── */
.chart-container {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 14px !important; font-weight: 600 !important; }
</style>
"""
