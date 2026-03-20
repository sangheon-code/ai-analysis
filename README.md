# KNOAH Trading Analysis

멀티 거래소 선물 거래 분석 대시보드

## 데모

https://ai-analysis-dgcg8wrfbeawblsokatkqt.streamlit.app/

## 기능

### 대시보드
- 현재 잔고, 수익률(KNOAH 공식), 순이익 표시
- 자산곡선 (순수 거래 성과 기반, 거래소별 비교)
- 핵심지표: 승률, 평균수익/손실, 수익팩터, MDD 등
- 종목별 손익/승률, 요일별/시간대별(KST) 패턴, PnL 분포 차트
- 거래 내역 조회 및 수동 추가

### 상세 분석
- 매매 스타일 카드 (스캘핑/데이트레이딩/스윙/장기 비율)
- 패턴 발견: 종목x방향, 시간대 히트맵, 종목x요일 히트맵, 레버리지/보유시간별 성과
- 매매 습관 지표: 복수매매 빈도, 연패 후 평균 손실, 연승 후 과신율
- AI 요약 (Haiku): 상세 분석 데이터 기반 핵심 요약

### What-if 시뮬레이터
- 조건별 거래 제외 (방향, 종목, 시간대, 레버리지, 복수매매)
- Before/After 비교: PnL, 승률, 수익팩터 변화
- 자산곡선 오버레이 비교
- 스마트 디폴트: 수익이 가장 큰 제외 조합 자동 추천

### 매매 일지
- 일별/주별 PnL 타임라인
- 일별 상세: 15분봉 캔들스틱 + 진입/청산 마커
- 주별 상세: 4시간봉 캔들스틱 + 진입/청산 마커
- 이상치 감지 (평균 대비 큰 손익)

### AI 챗 (사이드바)
- Haiku 기반 자연어 질문 응답
- 거래 데이터를 context로 전달하여 데이터 기반 답변
- 질문당 약 $0.002

### 거래소 연동
| 거래소 | 거래 내역 | 입출금 | 잔고 조회 |
|--------|-----------|--------|-----------|
| Binance | O (fapi) | O (TRANSFER) | O (fapi) |
| Bybit | O | O | O |
| OKX | O | O | O |
| Bitget | O | O | O |

- 여러 거래소 동시 연결 및 통합 분석
- 최대 3년치 데이터 조회 (페이지네이션)
- 바이낸스: fapi 직접 호출 (sapi 우회). 클라우드 서버 IP는 바이낸스에서 차단될 수 있음
- 수수료/펀딩피 포함 (바이낸스: COMMISSION + FUNDING_FEE + REALIZED_PNL)

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
| `ANTHROPIC_API_KEY` | Claude API 키 (AI 요약/챗) |

거래소 API 키는 메인 화면에서 입력합니다.

## Streamlit Cloud 배포

1. GitHub repo 연결
2. Main file: `knoah_trading_dashboard.py`
3. Settings > Secrets에 추가:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## 프로젝트 구조

```
knoah_trading_dashboard.py  -- 메인 대시보드 (Streamlit)
lib/
  config.py      -- 상수, CSS, 컬럼 정의
  exchanges.py   -- 거래소 API 연동 (ccxt)
  analysis.py    -- 데이터 집계 + AI 요약/챗
  dummy.py       -- 더미 데이터 생성 (청산 시뮬레이션 포함)
```

## 기술 스택

- Streamlit 1.55+
- Plotly
- ccxt (거래소 API)
- Claude Haiku (AI 요약/챗)
- pandas / numpy
