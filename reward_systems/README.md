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
- **전략**: Gradual unfreezing 
  - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
- **결과**: 성능 미흡으로 실제 시스템에는 적용되지 않음


## 평가 지표 및 추론 방법

- **주요 지표**: F1-score
- **추론 방식**: 모델 앙상블 결과를 뉴스 카테고리별 가중치 기반으로 보정하여 최종 분류


## 결과

|                  | 초기 시스템 (LLM API 단독) | 최종 평가 시스템 (2단계 필터링 적용) |
|:----------------:|:--------------------------:|:---------------------------------:|
| **F1-score** | 0.79 | **0.85** |


---
# 📈 Ongoing Improvements (지속 개선 작업)

## 1. 개선 배경

- 기존 평가 시스템은 LLM API (Solar 기반) + BERT 기반 모델을 조합해 민감 발언을 분류했지만, 데이터 품질과 recall 최적화에 개선 여지가 있음.
- 특히 초기 학습 데이터는 **LLAMA 모델을 통한 자동 생성**으로 구축되었기 때문에,
  - 문장 품질 편차가 크고
  - 미묘하거나 사회적 맥락이 중요한 케이스를 제대로 반영하지 못하는 한계가 있었음.
- 또한 비즈니스 관점에서는 민감 발언 누락을 최소화하기 위해 **Recall**을 Precision보다 더 중요하게 다루어야 함.

## 2. 주요 개선 방향

- **데이터 품질 개선**
  - KOLD 오픈 코퍼스를 기반으로 데이터 신뢰도 분석 및 정제 완료
  - 불확실성 높은 샘플 필터링 및 일부 직접 라벨링 검토
- **데이터 증강**
  - 유의어 치환 및 백트랜슬레이션을 통한 데이터 다양성 확보
  - 증강 데이터에 대한 신뢰도 검증 후 추가 학습
- **커스텀 평가 지표 도입**
  - Recall 가중치를 높인 커스텀 F1 Score 설계 및 적용
- **모델 파이프라인 개선**
  - Threshold 최적화 및 점수 기반 최종 판단 구조로 개선 예정

## 3. 목표 일정

- **2025년 5월 16일(금)** 까지 1차 개선 완료 목표
- 이후 추가 실험 및 최적화 진행 예정

---

## ✅ 진행 상황 체크리스트 (Progress Checklist)

- [x] KOLD 데이터 기반 베이스라인 학습
- [x] 데이터 신뢰도 분석 및 필터링
- [ ] 유의어 치환 데이터 증강
- [ ] 백트랜슬레이션 데이터 증강
- [ ] 증강 데이터 신뢰도 필터링 및 정제
- [ ] Recall-weighted Custom F1 metric 적용
- [ ] 증강 데이터 포함 학습 및 실험
- [ ] Threshold 최적화 및 결과 분석
- [ ] 최종 리포트 및 결과 문서화



## 주요 기술 스택

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Upstage Solar API
- FastAPI (API 구성 시 활용)

