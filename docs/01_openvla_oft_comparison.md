# OpenVLA vs OpenVLA-OFT vs OFT+

## 1. 한눈에 보는 구조 차이

| 항목 | OpenVLA | OpenVLA-OFT | OpenVLA-OFT+ |
|---|---|---|---|
| Action 표현 | Discrete token | Continuous action | Continuous action |
| 학습 목적함수 | Cross-entropy | L1 regression | L1 regression |
| Decoding | Autoregressive | Parallel / Bidirectional | Parallel / Bidirectional |
| Action 예측 | 1 timestep | Action chunk | Action chunk |
| 시각 입력 | 3인칭 이미지 | 3인칭 + Wrist camera | 3인칭 + Wrist camera |
| Robot state | 없음 | Proprioception | Proprioception |
| Action head | LM vocabulary | MLP regression head | MLP regression head |
| 추가 요소 | - | OFT recipe | FiLM language conditioning |

## 2. OpenVLA → OFT에서 바뀐 핵심

### Continuous action + L1 regression
OpenVLA는 7D action을 이산화해 언어 토큰처럼 생성하지만 OFT는 별도의 MLP regression head를 통해 연속 action을 직접 출력합니다.

로봇 action 예시는 다음과 같습니다.

```text
[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
```

### Parallel decoding
기존 autoregressive 방식은 action 차원을 순차 생성하지만 OFT는 action embedding 사이의 causal mask를 제거하고 bidirectional attention을 사용해 여러 action 차원을 동시에 예측합니다.

### Action chunking
한 번의 policy query에서 여러 미래 timestep의 action을 생성합니다. 매 timestep마다 모델을 다시 호출하는 것보다 추론 latency를 줄일 수 있습니다.

### 입력 modality 확장
OFT는 3인칭 카메라뿐 아니라 wrist camera와 proprioceptive state를 추가로 사용합니다. 따라서 시각 관측과 로봇 자체 상태를 함께 이용할 수 있습니다.

## 3. OFT → OFT+

OFT+는 OFT 구조를 유지하면서 **FiLM(Feature-wise Linear Modulation)**을 추가해 언어 조건이 시각 feature에 직접 개입하도록 합니다.

```math
feature' = γ(language) ⊙ feature + β(language)
```

즉 instruction을 마지막에 단순 결합하는 것이 아니라, 어떤 시각 feature를 강조할지 언어가 조절합니다.

## 4. LIBERO-Plus 직접 평가 결과

| 항목 | OFT | OFT+ | 차이 |
|---|---:|---:|---:|
| Overall | 71.20% | **80.96%** | +9.76%p |
| libero_spatial | 85.43% | 85.93% | +0.50%p |
| libero_object | 68.86% | **83.92%** | +15.06%p |
| libero_goal | 64.65% | **72.33%** | +7.68%p |
| libero_10 | 66.69% | **82.14%** | +15.45%p |

### Perturbation별

| Category | OFT | OFT+ |
|---|---:|---:|
| Background Textures | 94.05% | **96.47%** |
| Camera Viewpoints | 58.66% | **95.31%** |
| Language Instructions | 83.08% | **85.69%** |
| Light Conditions | 92.12% | **94.66%** |
| Objects Layout | **77.84%** | 77.11% |
| Robot Initial States | **33.29%** | 29.55% |
| Sensor Noise | 72.39% | **95.32%** |

## 5. 관찰한 점

### OFT+의 강점: Camera / Sensor Noise
OFT+는 특히 Camera Viewpoints와 Sensor Noise에서 큰 개선을 보였습니다.

- Camera Viewpoints: **+36.65%p**
- Sensor Noise: **+22.93%p**

FiLM의 원래 목적은 language grounding 강화이지만, 실험에서는 시각적 perturbation에서도 큰 개선이 나타났습니다.

**해석 가설**: instruction-conditioned visual feature modulation이 task와 무관한 시각 변화보다 물체와 gripper 관계에 집중하게 해 시각적 domain shift를 완화했을 가능성이 있습니다. 이는 논문에 직접 명시된 인과관계가 아니라 실험 결과를 바탕으로 한 해석입니다.

### 공통 취약점: Robot Initial States
OFT와 OFT+ 모두 Robot Initial States에서 30% 수준으로 크게 무너졌습니다.

두 모델의 주요 개선은 현재 관측의 feature quality와 action generation 방식에 집중되어 있습니다. 과거 observation과 action history를 누적해 상태를 추정하거나 실행 mismatch를 감지하는 명시적 temporal memory는 없습니다.

따라서 **관측 손상에 대한 robustness와 state-transition mismatch에 대한 robustness는 서로 다른 메커니즘을 요구할 가능성**이 있습니다.

## 6. 현재 연구 질문

- Camera perturbation과 Robot Initial State perturbation은 같은 방식으로 처리할 수 있는가?
- 현재 frame의 representation robustness만으로 state mismatch까지 해결할 수 있는가?
- Temporal memory 또는 closed-loop correction을 추가하면 Robot Initial States 취약점을 줄일 수 있는가?

---

> 이 문서는 논문에 명시된 구조적 사실, 직접 수행한 LIBERO-Plus 실험 결과, 그리고 실험에서 도출한 개인적 해석을 구분하여 작성했습니다.
