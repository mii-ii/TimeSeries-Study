# ImageNet Image Classification: MLP vs CNN Performance Comparison

## 1. 프로젝트 개요 및 목표 (Project Overview)
* **분류 대상:** ImageNet 하위 4개 클래스 (고양이, 개, 말, 얼룩말)
* **학습 환경:** 해상도 128x128 픽셀 통일, Batch Size 32
* **핵심 목표:** 다층 퍼셉트론(MLP)과 합성곱 신경망(CNN)의 구조적 차이에 따른 이미지 데이터의 공간 정보 처리 능력 및 분류 성능 비교 검증.

## 2. 데이터 전처리 파이프라인 (Data Engineering)
* **딥 클렌징 (Deep Cleansing):**
  * 외부 수집 데이터에 흔히 혼입되는 0바이트 찌꺼기 파일 및 손상된 내부 헤더를 가진 이미지를 사전에 차단하여 TensorFlow 디코딩 에러를 원천적으로 방지함.
  * Python `PIL` 라이브러리를 활용한 데이터 무결성 검증 수행.
* **데이터 파이프라인 최적화 (Prefetching):**
  * `tf.data.AUTOTUNE`을 적용하여 GPU가 연산하는 동안 CPU가 다음 데이터를 메모리에 미리 적재하도록 유도, I/O 병목 현상(Bottleneck)을 최소화하고 학습 속도를 극대화함.

## 3. 모델 아키텍처 (Model Architecture)
### 대조군: 다층 퍼셉트론 (MLP)
* `Flatten` 레이어를 통해 2차원 공간 정보를 1차원 배열로 평탄화하여 입력.
* 픽셀 간의 공간적 연결성(윤곽선, 질감 등)이 파괴되어 발생하는 이미지 분류의 구조적 한계를 명확히 확인하기 위한 Baseline 모델로 활용.

### 실험군: 합성곱 신경망 (CNN)
* `Conv2D` 레이어를 통해 3x3 필터를 슬라이딩시키며 이미지의 공간적 특징(Feature)을 보존 및 추출.
* `MaxPooling2D` 레이어를 활용해 특징 맵의 핵심 신호만 남겨 연산량을 획기적으로 감축하고, 사물의 위치 변화에도 강건하게 대응하는 위치 불변성(Translation Invariance) 획득.

## 4. 과적합 제어 전략 (Regularization)
* **조기 종료 (Early Stopping):** 검증 손실(Validation Loss) 지표를 모니터링하여, 지정된 Epoch(5회) 동안 개선이 없을 시 무의미한 학습을 조기 종료하고 최적의 가중치(Weight)로 복원함.
* **드롭아웃 (Dropout):** 은닉층 뉴런을 무작위로 50% 비활성화하여 특정 노드에 대한 과도한 의존성을 배제하고, 모델의 전반적인 일반화(Generalization) 성능을 향상시킴.

## 5. 성능 평가 지표 (Evaluation Metrics)
* **정밀도(Precision) 및 재현율(Recall):** 단순 정확도(Accuracy)가 낳을 수 있는 데이터 불균형 함정을 보완하기 위한 교차 검증 수행.
* **AUROC (Area Under ROC Curve):** 판단 임계값(Threshold) 변동에도 흔들리지 않는 모델의 클래스 분별력을 종합적으로 측정하여 신뢰도 높은 성능 평가 진행.
