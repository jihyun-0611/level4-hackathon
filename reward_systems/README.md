# 📊 민감 발언 탐지 평가 시스템

## 프로젝트 개요

- **문제점**: Upstage LLM API만을 사용했을 때 민감 발언 탐지 범위가 지나치게 광범위하고, 일관성이 부족.
- **제약사항**: API 모델은 비공개라 직접적인 파인튜닝 불가.
- **해결책**: LLM API 응답 결과를 추가적으로 **BERT 기반 모델 앙상블**을 통해 평가하는 2단계 필터링 시스템 구축.
- **참고 이론**: RLHF(Reward Learning from Human Feedback)에서 사용하는 Preference Tuning 기법을 참고.

## 데이터셋

### 학습 데이터
- 생성 방법: **Llama3 + Solar API**를 통한 문장 생성
- 라벨링 방법: **BERT 앙상블 모델**로 자동 분류
- 규모: **69,549개** 문장
  - Offensive: 44,844개
  - Non-offensive: 24,705개

### 평가 데이터
- 생성 방법: **Llama3 + Solar API**
- 라벨링 방법: **사람 직접 검수**
- 규모: **3,163개** 문장
  - Offensive: 894개
  - Non-offensive: 2,268개


## 모델 구성 및 학습 전략

### 1. Sentiment Analysis 모델
- [tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
- **전략**: feature extractor freezing, classifier 레이어만 학습
- **도입 이유**: Offensive 문장은 대개 강한 감정을 동반하는 경향이 있어, 보조적인 분류 지표로 활용

### 2. BERT 기반 분류 모델
- [klue/bert-base](https://huggingface.co/klue/bert-base)
- [klue/roberta-base](https://huggingface.co/klue/roberta-base)
- [klue/roberta-large](https://huggingface.co/klue/roberta-large)
- **전략**: 전체 파인튜닝
- **도입 이유**: 문맥 이해 기반의 고성능 문장 분류를 위해 BERT 아키텍처 선택

### 3. Reward Model (실험적)
- [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)
- **전략**: 파인튜닝 없이 점진적 unfreezing
- **결과**: 성능 미흡으로 실제 시스템에는 적용되지 않음


## 평가 지표 및 추론 방법

- **주요 지표**: F1-score
- **추론 방식**: 모델 앙상블 결과를 뉴스 카테고리별 가중치 기반으로 보정하여 최종 분류


## 결과

|                  | 초기 시스템 (LLM API 단독) | 최종 평가 시스템 (2단계 필터링 적용) |
|:----------------:|:--------------------------:|:---------------------------------:|
| **F1-score** | 0.79 | **0.85** |


## 디렉토리 구조

```
reward_systems/
├── config/               # 하이퍼파라미터 및 경로 설정
├── dataset/              # 데이터 로딩 및 전처리 스크립트
├── models/               # BERT, Sentiment, Reward 모델 정의 및 관리
├── trainers/             # 학습 loop 및 앙상블 학습 모듈
├── evaluators/           # 평가 및 추론 로직
├── utils/                # 공통 함수 및 헬퍼 스크립트
└── main.py               # 메인 엔트리포인트 (학습/추론 실행)
```


## 주요 기술 스택

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Upstage Solar API
- FastAPI (API 구성 시 활용)

