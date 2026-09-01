# RoboMamba 논문 리뷰

## 1. 한 줄 요약

RoboMamba는 **CLIP 기반 vision encoder와 Mamba backbone을 결합한 멀티모달 로봇 모델**로, 일반 시각·언어 추론과 로봇 추론을 학습한 뒤 전체 backbone을 동결하고 작은 policy head만 학습해 manipulation을 수행합니다.

## 2. 문제의식

기존 Transformer 기반 MLLM/VLA는 다음 문제가 있습니다.

- 긴 sequence에서 attention 계산량이 큼
- 전체 모델 fine-tuning 비용이 높음
- 로봇 데이터가 상대적으로 적음
- task-specific fine-tuning 과정에서 기존 추론 능력이 손상될 수 있음

RoboMamba는 **선형 복잡도의 Mamba**와 작은 policy head를 사용해 효율적인 로봇 학습을 목표로 합니다.

## 3. 전체 구조

```text
이미지 ─→ CLIP Vision Encoder ─→ Projection ─┐
                                             ├─→ Mamba Backbone
텍스트 ─→ Token Embedding ──────────────────┘
                                             ├─→ 자연어 추론 / 계획
                                             └─→ Policy Head → 조작 pose
```

### 구성 요소

- **CLIP Vision Encoder**: 이미지를 visual token으로 변환
- **Projection Layer**: CLIP feature를 Mamba embedding space로 정렬
- **Mamba Backbone**: visual + text token을 함께 처리하며 추론
- **Policy Head**: manipulation을 위한 접촉 위치와 end-effector 방향을 예측

RoboMamba는 action token sequence 전체를 생성하는 generalist VLA라기보다 **멀티모달 reasoning backbone + manipulation head** 구조에 가깝습니다.

## 4. 학습 과정

### Stage 1.1 — Alignment Pre-training

- CLIP: Freeze
- Mamba: Freeze
- Projection Layer: Train

목표는 CLIP의 visual feature를 Mamba가 이해할 수 있는 embedding space에 맞추는 것입니다.

### Stage 1.2 — Instruction Co-training

- CLIP: Freeze
- Projection: Train
- Mamba: Train

일반 vision instruction 데이터와 robot reasoning 데이터를 함께 학습합니다.

로봇 관련 학습 항목에는 다음이 포함됩니다.

- Long-horizon planning
- Success classification
- Affordance reasoning
- Past action description
- Future prediction

### Stage 2 — Manipulation Fine-tuning

- CLIP: Freeze
- Projection: Freeze
- Mamba: Freeze
- **Policy head만 학습**

이미 학습된 multimodal representation은 유지하고 작은 head만 manipulation task에 맞게 학습합니다.

## 5. Mamba의 핵심 개념

일반적인 상태공간모델(SSM)은 다음 형태로 볼 수 있습니다.

```math
h_t = \bar{A}h_{t-1} + \bar{B}x_t
```

```math
y_t = Ch_t
```

Mamba의 Selective SSM에서는 현재 입력에 따라 상태 업데이트와 읽기 방식이 달라집니다.

```math
B_t = W_Bx_t
```

```math
C_t = W_Cx_t
```

```math
\Delta_t = softplus(W_\Delta x_t + b_\Delta)
```

핵심은 **모든 입력을 같은 방식으로 기억하는 것이 아니라 현재 token의 내용에 따라 무엇을 기억하고 무엇을 읽을지 선택한다는 점**입니다.

### 직관

```text
기존 SSM
모든 입력 → 거의 동일한 상태 업데이트 규칙

Mamba
현재 입력
   ↓
기억할 정보 / 잊을 정보 / 읽을 정보 선택
   ↓
Hidden State 업데이트
```

## 6. RoboMamba의 Action 출력

RoboMamba가 예측하는 주요 action 정보는 다음과 같습니다.

- Contact position
- End-effector orientation

이미지상의 접촉점을 예측하고 depth 정보를 이용해 3D 좌표로 변환합니다.

따라서 OpenVLA나 π₀처럼 긴 continuous action sequence를 직접 생성하기보다는 **manipulation을 위한 target pose prediction** 성격이 강합니다.

## 7. RT-2와의 차이

### RT-2

- VLM 기반
- Robot action을 language token처럼 생성
- Web-scale semantic knowledge를 기존 robot skill에 연결

### RoboMamba

- CLIP + Mamba multimodal backbone
- 자연어 출력과 manipulation action head가 분리
- 작은 policy head만 fine-tuning 가능
- 효율적인 reasoning + manipulation adaptation에 초점

## 8. 새로운 skill에 대한 일반화

RoboMamba가 일반 시각·언어 지식을 갖고 있더라도 학습하지 않은 복잡한 motor skill을 자동으로 발명할 수 있는 것은 아닙니다.

예를 들어 나사를 조이는 작업에는 다음이 필요합니다.

- Tool과 screw 정렬
- 축 방향 force 유지
- 손목 회전
- 미끄러짐 감지 및 재정렬
- 완료 상태 판단

언어적으로 작업 방법을 이해하는 것과 실제 continuous control을 수행하는 것은 별개의 문제입니다.

## 9. 장점과 한계

### 장점

- Attention 기반 모델보다 효율적인 sequence modeling을 목표로 함
- 작은 policy head만 학습 가능
- 일반 reasoning과 robot reasoning을 하나의 backbone에서 학습
- Mamba를 Physical AI / robot learning에 적용한 사례

### 한계

- full action trajectory를 생성하는 VLA와는 구조가 다름
- policy head가 task-specific manipulation output에 가까움
- Mamba 내부 state가 실제 closed-loop robot memory로 직접 활용되는 것은 아님

## 10. 연구적으로 흥미로운 질문

RoboMamba에서 특히 흥미로운 부분은 Mamba의 hidden state입니다.

현재 RoboMamba는 Mamba를 효율적인 multimodal backbone으로 사용하지만, 다음 방향은 별개의 연구 질문이 될 수 있습니다.

- 이전 observation/action history를 Mamba state에 누적할 수 있는가?
- state transition mismatch를 memory를 이용해 감지할 수 있는가?
- action correction 또는 recovery에 SSM state를 직접 활용할 수 있는가?

즉 **Mamba를 단순 backbone이 아니라 temporal state estimator / correction memory로 활용하는 방향**이 VLA robustness 연구와 연결될 수 있습니다.
