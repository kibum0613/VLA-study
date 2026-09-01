# RoboMamba 논문 리뷰

## 1. 논문 한 줄 요약

RoboMamba는 **CLIP 기반 시각 인코더와 Mamba 언어모델을 결합한 로봇 VLA 모델**임. 일반·로봇 추론 능력을 먼저 학습한 뒤 전체 backbone을 동결하고 작은 policy head만 학습하여 접촉 위치와 end-effector 방향을 예측함.

## 2. 연구 배경과 문제의식

기존 MLLM·VLA 기반 로봇 정책에는 다음 문제가 있음.

- Attention 기반 대형 언어모델의 계산량과 추론 비용이 큼.
- 로봇 조작을 위해 전체 모델을 fine-tuning하면 많은 데이터와 자원이 필요함.
- Task-specific fine-tuning 과정에서 기존의 시각·언어 추론 능력이 손상될 수 있음.

RoboMamba는 이를 줄이기 위해 선형 복잡도의 Mamba를 사용함. Manipulation 학습에서는 전체 backbone이 아니라 약 0.1% 규모의 작은 policy head만 학습함.

## 3. 전체 구조

```text
이미지 ─→ CLIP vision encoder ─→ Projection layer ─┐
                                                  ├─→ Mamba backbone
텍스트 명령 ─→ Token embedding ────────────────────┘
                                                    ├─→ 자연어 추론·계획 출력
                                                    └─→ Policy head → 위치·방향 예측
```

### 구성 요소별 역할

- **CLIP vision encoder**: 이미지를 visual token으로 변환함.
- **Projection layer**: CLIP의 시각 feature를 Mamba embedding space에 맞게 변환함.
- **Mamba backbone**: visual token과 text token을 함께 처리하며 일반 상식, 장면 이해, robot planning, affordance 등을 추론함.
- **Policy head**: Mamba 출력 representation을 바탕으로 접촉 위치와 end-effector 방향을 예측함.

RoboMamba는 자연어와 action의 두 출력 경로를 지원함. 다만 생성된 자연어 subgoal이 policy head로 다시 들어가는 명시적인 planner-controller hierarchy는 아님.

## 4. 학습 과정

### Stage 1.1 — Alignment Pre-training

- CLIP: 동결함.
- Mamba: 동결함.
- Projection layer: 학습함.
- 데이터: LLaVA-LCS 558K 이미지-텍스트 쌍을 사용함.

목적은 CLIP의 시각 feature를 Mamba가 언어 token과 함께 처리할 수 있는 표현 공간으로 정렬하는 것임. 이미 사전학습된 CLIP과 Mamba는 유지하고 두 모델 사이의 projection만 먼저 학습함.

### Stage 1.2 — Instruction Co-training

- CLIP: 동결함.
- Projection layer: 학습함.
- Mamba: 동결을 해제하고 학습함.
- 일반 vision instruction 데이터와 RoboVQA 기반 robot reasoning 데이터를 함께 사용함.

학습 항목은 다음과 같음.

- Long-horizon planning
- Success classification
- Discriminative affordance
- Generative affordance
- Past description
- Future prediction

일반 데이터는 장면과 상식을 이해하게 하고 robot data는 해당 장면에서 무엇을 할 수 있는지와 행동 순서를 학습하게 함.

### Stage 2 — Manipulation Fine-tuning

- CLIP: 동결함.
- Projection layer: 동결함.
- Mamba: 동결함.
- Policy head만 학습함.

Stage 1에서 학습한 representation을 유지하면서 작은 MLP head만 manipulation task에 맞게 학습함. 전체 backbone을 다시 조정하지 않으므로 학습 비용을 줄이고 기존 reasoning 능력 손상을 줄이는 방식임.

## 5. Mamba의 Selective SSM

기존 SSM은 다음 형태로 볼 수 있음.

```math
h_t = \bar A h_{t-1} + \bar B x_t
```

```math
y_t = C h_t
```

Mamba에서는 현재 입력에 따라 일부 상태공간 파라미터가 달라짐.

```math
B_t = W_B x_t
```

```math
C_t = W_C x_t
```

```math
\Delta_t = \operatorname{softplus}(W_\Delta x_t + b_\Delta)
```

실제 이산 상태 전이는 다음처럼 계산됨.

```math
\bar A_t = \exp(\Delta_t A)
```

```math
\bar B_t = (\Delta_t A)^{-1}(\exp(\Delta_t A)-I)\Delta_t B_t
```

### 각 항의 역할

- `A`: 모든 token에서 공유되는 기본 상태 동역학임.
- `Δ_t`: 현재 입력에 따라 상태를 얼마나 빠르게 갱신할지 조절함.
- `B_t`: 현재 입력을 hidden state의 어떤 방향으로 기록할지 결정함.
- `C_t`: 현재 hidden state에서 어떤 정보를 읽어 출력할지 결정함.
- `Ā_t`: 고정된 `A`와 입력별 `Δ_t`가 결합된 실제 상태 전이값임.
- `B̄_t`: `A`, `B_t`, `Δ_t`가 결합된 실제 입력 반영값임.

`A` 자체가 token마다 바뀐다기보다 기본 `A`는 공유되고 실제 연산에 사용되는 `Ā_t`가 입력에 따라 달라진다고 보는 것이 정확함.

## 6. Action 출력 방식

RoboMamba가 주로 예측하는 값은 다음과 같음.

- 접촉 위치 `a_pos`
- end-effector 방향 `a_dir`

위치 loss는 예측값과 정답값 사이의 절댓값 오차를 사용함. 회전 loss는 두 rotation matrix 사이의 각도 차이를 사용함.

시뮬레이션에서는 이미지상의 2D 접촉점을 예측한 뒤 depth 정보를 이용해 3D 좌표로 변환함. 따라서 일반적인 연속 action sequence 전체를 생성하는 VLA보다 **조작을 위한 단일 접촉 pose 예측 모델**에 가까움.

## 7. 실험 결과 핵심

- 약 2.7B 규모의 Mamba backbone을 사용함.
- RoboVQA를 함께 학습했을 때 robot reasoning과 unseen task manipulation 성능이 개선됨.
- 동일한 작은 policy head를 사용할 경우 reasoning 성능이 좋은 backbone이 manipulation 성공률도 높은 경향을 보임.
- A100 기준 약 9Hz의 제어 주파수를 보고함.
- 약 3.7M parameter 규모의 policy head만 manipulation 단계에서 학습함.

속도 차이에는 architecture뿐 아니라 전체 모델 크기 차이도 영향을 줌. 따라서 속도 향상을 전적으로 Mamba만의 효과로 해석하면 안 됨.

## 8. 계층형 구조 여부

전형적인 계층형 구조는 다음과 같음.

```text
상위 planner → subgoal 생성 → 하위 controller 실행 → 관찰 → 다음 subgoal
```

RoboMamba는 다음에 가까움.

```text
공통 backbone
├─ 자연어 계획·추론 출력
└─ 별도 policy head의 pose 출력
```

고수준 reasoning과 저수준 pose prediction 능력을 모두 갖지만 자연어 subgoal이 low-level policy로 직접 전달되는 구조는 아님. 따라서 공유 backbone을 사용하는 다중 출력 multimodal robot model로 보는 편이 적절함.

## 9. RT-2와의 차이

### RT-2

- 웹·이미지·언어로 사전학습된 VLM을 사용함.
- Robot action을 token화하여 언어 token과 같은 생성 경로에서 출력함.
- Web-scale semantic knowledge를 기존 robot skill에 연결하는 일반화를 목표로 함.

### RoboMamba

- CLIP과 Mamba를 결합한 multimodal backbone을 사용함.
- 자연어 출력과 action 출력 경로가 분리되어 있음.
- Action은 별도 MLP policy head가 연속 pose로 예측함.
- 작은 head만 학습해 manipulation skill을 추가하는 효율성에 초점을 둠.

RoboMamba는 RT-2형 generalist VLA보다는 multimodal reasoning backbone에 manipulation head를 붙인 구조에 가까움.

## 10. 새로운 action에 대한 일반화

RoboMamba와 RT-2 모두 학습하지 않은 새로운 운동 기술을 자동으로 만들어내는 모델은 아님.

예를 들어 나사 조이기에는 다음 능력이 필요함.

- 드라이버와 나사 홈 정렬
- 축 방향 힘 유지
- 손목 회전
- 미끄러짐 감지와 재정렬
- 토크 또는 완료 상태 판단

언어적으로 작업 방법을 이해하는 것과 실제 관절 제어를 수행하는 것은 별개의 문제임. 실제 수행을 위해서는 관련 demonstration이나 유사한 motor primitive가 필요함.

## 11. 장점

- Attention 기반 대형 VLA보다 효율적인 추론을 목표로 함.
- 일반 시각 추론과 robot reasoning을 공동 학습함.
- 전체 backbone을 유지하면서 작은 policy head만 학습할 수 있음.
- 기존 reasoning 능력을 크게 손상시키지 않고 manipulation 능력을 추가하는 전략을 제시함.
- Seen뿐 아니라 unseen articulated object에 대한 일반화 가능성을 보여줌.
- Mamba를 robot VLA에 적용한 초기 연구라는 의미가 있음.

## 12. 한계

- 언어 계획과 실제 action이 직접 연결된 계층형 구조가 아님.
- Action space가 단일 접촉 pose 중심으로 제한됨.
- 긴 trajectory나 복잡한 continuous action sequence를 직접 생성하지 않음.
- 실행 중 실패를 관찰하고 행동을 수정하는 closed-loop feedback 구조가 부족함.
- Grasp 성공 여부나 door open 상태 등을 지속적으로 점검하는 구조가 아님.
- 실제 환경 실험은 대규모 정량 평가보다 정성적 demonstration 성격이 강함.
- 2D 이미지 기반 예측이므로 3D geometry와 depth ambiguity에 한계가 있음.

## 13. 개인적인 해석

RoboMamba의 핵심은 단순히 Mamba를 사용했다는 점보다 다음 학습 전략에 있다고 봄.

> 시각·언어·robot reasoning 능력을 먼저 학습한 backbone을 만든 뒤 backbone을 동결하고 작은 action head만 학습하여 manipulation 능력을 추가함.

따라서 완성형 generalist VLA보다는 **강한 multimodal representation을 이용해 적은 비용으로 특정 manipulation skill을 추가할 수 있음을 보여준 연구**로 보는 것이 적절함.

추가로 관심 있는 확장 방향은 다음과 같음.

- 자연어 subgoal과 low-level policy를 실제로 연결하는 계층형 구조
- 실행 결과를 관찰하고 실패를 진단하는 closed-loop 제어
- 단일 이미지 대신 영상·이전 action·3D 정보를 사용하는 temporal VLA
- 접촉 pose 대신 연속 trajectory 또는 diffusion/flow 기반 action 생성
- 촉각·force·torque feedback을 사용하는 정밀 manipulation

## 14. 최종 정리

RoboMamba는 CLIP과 Mamba를 결합해 일반 및 로봇 관련 reasoning을 학습한 뒤 작은 policy head만으로 접촉 위치와 방향을 예측하는 효율적인 robot model임. Mamba의 selective SSM을 활용한 효율적인 backbone과 parameter-efficient manipulation adaptation 가능성을 제시했다는 점이 핵심임.
