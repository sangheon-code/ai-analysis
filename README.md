# KNOAH Trading Analysis

멀티 거래소 선물 거래 분석 대시보드 + Claude AI 딥 리포트

## 데모

https://ai-analysis-knoah.streamlit.app

## 기능

### 대시보드
- 현재 잔고, 수익률(KNOAH 공식), 순이익 실시간 표시
- 자산곡선 (순수 거래 성과 기반, 거래소별 비교)
- 핵심지표: 승률, 평균수익/손실, 수익팩터, MDD 등
- 종목별 손익/승률, 요일별/시간대별(KST) 패턴, PnL 분포 차트
- 거래 내역 조회 및 수동 추가

### AI Deep Report
- Claude가 교차 분석 데이터를 기반으로 생성하는 맞춤 리포트
- 매매 스타일 진단 (스캘핑/데이트레이딩/스윙)
- 숨은 패턴 발견 (종목x방향, 종목x요일, 시간대별 교차)
- 매매 습관 분석 (복수매매, 연패 후 행동, 보유시간 비교)
- 실행 가능한 맞춤 전략 제안

### 거래소 연동
| 거래소 | 거래 내역 | 입출금 | 잔고 조회 |
|--------|-----------|--------|-----------|
| Binance | O (fapi) | O (TRANSFER) | O (fapi) |
| Bybit | O | O | O |
| OKX | O | O | O |
| Bitget | O | O | O |

- 여러 거래소 동시 연결 및 통합 분석
- 최대 3년치 데이터 조회 (페이지네이션)
- 바이낸스 한국 IP 차단 우회 (sapi 대신 fapi 사용)

### 수익률 계산 (KNOAH 공식)
```
순이익 = 현재자산 - 초기자산 - 추가입금 + 총출금
수익률(%) = 순이익 / (초기자산 + 추가입금) x 100
```

## 설치

```bash
git clone https://github.com/sangheon-code/ai-analysis.git
cd ai-analysis
pip install -r requirements.txt
```

## 실행

```bash
# .env 파일에 API 키 설정
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 실행
streamlit run knoah_trading_dashboard.py
```

## 환경 변수

| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 키 (AI 리포트용) |

거래소 API 키는 대시보드 사이드바에서 입력합니다.

## Streamlit Cloud 배포

1. GitHub repo 연결
2. Main file: `knoah_trading_dashboard.py`
3. Settings > Secrets에 추가:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## 프로젝트 구조

```
knoah_trading_dashboard.py  — 메인 대시보드 (Streamlit)
lib/
  config.py      — 상수, CSS, 컬럼 정의
  exchanges.py   — 거래소 API 연동 (ccxt)
  analysis.py    — 데이터 집계 + Claude 딥 리포트
  dummy.py       — 더미 데이터 생성 (청산 시뮬레이션 포함)
```

## 기술 스택

- Streamlit
- Plotly
- ccxt (거래소 API)
- Claude Sonnet (AI 분석)
- pandas / numpy
