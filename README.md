# 딥러닝 기초 구현 프로젝트 모음 (`prj1` / `prj2` / `prj3`)

`prj1` → `prj2` → `prj3` 순서로, **기초 구현(벡터화/학습 루프/역전파)에서 CNN까지** 점진적으로 확장하는 실습 모음입니다.

> 노트북/코드 형태와 `eecs598`, `a3_helper` 등의 의존성으로 보아, **University of Michigan EECS 498/598 과제 스타일**(CS231n 계열) 기반 구현이 포함되어 있습니다.  
> 과제 배포용 헬퍼/데이터 로더가 빠져 있을 수 있어 **레포 단독 실행이 바로 안 될 수 있습니다.** (대신 “무엇을 구현했는지”를 중심으로 읽기 쉽게 정리했습니다)

## 목차

- [폴더 구조](#폴더-구조)
- [빠른 탐색 가이드](#빠른-탐색-가이드)
- [프로젝트 요약](#프로젝트-요약)
  - [`prj1` — KNN + PyTorch 기본기](#prj1--knn--pytorch-기본기)
  - [`prj2` — 선형 분류기 + 2층 MLP(수동 역전파)](#prj2--선형-분류기--2층-mlp수동-역전파)
  - [`prj3` — 모듈식 FCN/CNN + 최적화/정규화/체크포인트](#prj3--모듈식-fcncnn--최적화정규화체크포인트)
- [실행/재현 관련 메모](#실행재현-관련-메모)
- [실무 관점으로 연결고리](#실무-관점으로-연결고리)

## 폴더 구조

- `prj1/`
  - `knn.py`, `knn.ipynb`: KNN 분류기 + 거리 계산(2중 루프/1중 루프/벡터화) + 교차검증
  - `pytorch101.py`, `pytorch101.ipynb`: PyTorch 텐서 생성/인덱싱/브로드캐스팅/리쉐이프/배치 행렬곱 등 기본 연습
- `prj2/`
  - `linear_classifier.py`: **Linear SVM / Softmax** 손실(naive/vectorized) + SGD 학습 루프 + 하이퍼파라미터 탐색 유틸
  - `two_layer_net.py`: **2-layer MLP(FC→ReLU→FC)** forward/backward(수동 구현) + SGD 학습/검증 + 그리드서치
  - `hand_drawn_weights.jpg`: (실습 자료) 가중치/결정경계 직관화
- `prj3/`
  - `fully_connected_networks.py`, `fully_connected_networks.ipynb`: 모듈식 레이어(Linear/ReLU/Dropout) + 임의 깊이 FCN + 최적화(SGD, Momentum, RMSProp, Adam)
  - `convolutional_networks.py`, `convolutional_networks.ipynb`: Conv/Pool/BatchNorm(및 fast/sandwich 레이어) + 3-layer CNN + VGG 스타일 DeepConvNet
  - `*.pth`: 학습된 모델 체크포인트(예: `best_two_layer_net.pth`, `one_minute_deepconvnet.pth`)

## 빠른 탐색 가이드

- **“구현 내용”이 가장 잘 보이는 파일**
  - `prj1/knn.py`
  - `prj2/linear_classifier.py`, `prj2/two_layer_net.py`
  - `prj3/fully_connected_networks.py`, `prj3/convolutional_networks.py`
- **노트북으로 맥락(실험/검증/시각화)까지 보려면**
  - `prj1/knn.ipynb`
  - `prj3/fully_connected_networks.ipynb`
  - `prj3/convolutional_networks.ipynb`

## 프로젝트 요약

### `prj1` — KNN + PyTorch 기본기

- **핵심 구현**
  - 이미지 텐서를 Flatten 후 **유클리드 거리** 계산 → KNN 분류
  - 거리 계산을 **(1) 2중 루프 → (2) 1중 루프 → (3) 완전 벡터화**로 단계적으로 최적화
  - **K-fold 교차검증**으로 최적의 \(k\) 선택
- **학습 포인트**
  - 같은 알고리즘도 **벡터화/연산 형태**에 따라 실행 시간이 크게 달라짐
  - 텐서 shape/브로드캐스팅/인덱싱 실수는 모델 품질 이전에 파이프라인을 깨뜨림

### `prj2` — 선형 분류기 + 2층 MLP(수동 역전파)

- **핵심 구현**
  - Linear SVM / Softmax 손실을 **naive(루프) / vectorized(행렬 연산)**로 구현
  - SGD로 가중치 업데이트, 미니배치 샘플링, 기본적인 하이퍼파라미터 탐색
  - 2층 MLP(FC-ReLU-FC)에서 **forward/backward를 직접 구현**하여 기울기 흐름 확인
- **학습 포인트**
  - “모델”뿐 아니라 **학습 루프/로깅/재현성/튜닝 방식**이 결과를 크게 좌우함
  - 벡터화는 성능뿐 아니라 **코드 경로 단순화(버그 면적 감소)**에도 도움이 됨

### `prj3` — 모듈식 FCN/CNN + 최적화/정규화/체크포인트

- **핵심 구현**
  - 레이어 단위 `forward/backward`로 조립 가능한 구조(모듈식)
  - Dropout, 여러 최적화 기법(SGD+Momentum/RMSProp/Adam)
  - CNN 구성요소(Conv/Pool/BatchNorm) 및 “sandwich layer” 조합
  - 학습된 가중치를 `.pth`로 저장/로드
- **학습 포인트**
  - 구조가 복잡해질수록 **모듈화/인터페이스(입출력 shape 계약)**가 유지보수의 핵심
  - 체크포인트는 “저장”을 넘어 **실험 관리/롤백/재학습 비용 절감**의 기본 단위

## 실행/재현 관련 메모

- 일부 노트북/코드는 `eecs598`, `a3_helper.py` 같은 **과제 배포용 헬퍼**를 전제로 작성되어, 이 레포만으로는 실행이 바로 안 될 수 있습니다.
- 그럼에도 이 레포에서 확인 가능한 핵심은 다음입니다.
  - 레이어/손실/옵티마이저/학습 루프가 **어떤 형태로 구현되는지**
  - 벡터화/모듈화/체크포인트가 “왜 필요한지”

## 실무 관점으로 연결고리

이 레포는 “서비스를 직접 만든다”기보다는, 실제 ML 시스템에서 자주 마주치는 문제를 **작게 쪼개 구현 관점으로 이해**하는 데 가깝습니다.

### 계산/성능(벡터화, 배치 처리) → 인프라 비용과 직결

- `prj1`의 거리 계산 벡터화처럼, 같은 기능도 **배치 처리/행렬 연산/BLAS·GPU 활용** 여부에 따라
  - 처리량(QPS) / 지연시간(latency) / CPU·GPU 비용이 달라집니다.

### 학습 루프/하이퍼파라미터 탐색 → “훈련 잡” 운영의 기본

- `prj2`의 SGD 루프/검증/탐색은 실무에서 다음과 연결됩니다.
  - 배치 잡 오케스트레이션(스케줄러로 여러 실험 실행)
  - 실패 재시도/중단점 재개(resume)를 위한 체크포인트 정책
  - 실험 메타데이터(파라미터/버전/데이터 스냅샷) 기록
  - 재현성(같은 입력이면 같은 결과가 나는가) 확보

### 모듈식 레이어/네트워크 구성 → 팀 단위 개발/리뷰/테스트

- `prj3`의 레이어 단위 인터페이스는 모델 변경이 잦을 때
  - 코드 리뷰/테스트 범위를 줄이고
  - “일부 변경이 전체를 깨뜨리는” 상황을 완화하는 데 도움이 됩니다.
- 또한 `forward/backward`와 shape 계약은
  - 모델 서빙 시 입력 검증
  - 배치 차원/채널 순서 실수 방지 같은 운영 이슈로 이어집니다.

### 모델 아티팩트(`.pth`) → 저장/배포/롤백/감사

- `*.pth`는 실무에서 말하는 **모델 아티팩트**에 해당합니다.
  - 저장 위치(오브젝트 스토리지/레지스트리)
  - 메타데이터(학습 코드 버전/데이터 버전/성능 지표)
  - 배포/롤백 단위(“이 체크포인트로 되돌리기”) 같은 운영 결정과 직접 연결됩니다.

