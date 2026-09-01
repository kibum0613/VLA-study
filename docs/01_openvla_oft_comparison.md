# OpenVLA vs OFT vs OFT+ 비교 분석

## 1. 세 모델 구조 비교

| 항목 | OpenVLA (base) | OpenVLA-OFT | OpenVLA-OFT+ |
|---|---|---|---|
| Action 표현 | discrete token (bin) | continuous | continuous |
| 학습 목적함수 | cross-entropy | L1 regression | L1 regression |
| Decoding | autoregressive, causal mask | parallel, bidirectional attention | 동일 |
| Action 예측 단위 | 1 timestep | action chunk | 동일 |
| 입력 | 3인칭 이미지 + 언어 | 3인칭 + wrist camera + proprioception | 동일 |
| 언어 조건화 | backbone 내부 결합 | backbone 내부 결합 | FiLM 추가 |
| Action head | LM vocabulary 재활용 | 별도 MLP regression head | 동일 |

## 2. OpenVLA → OFT 변경점

### 2.1 Discrete action token → continuous action

- OpenVLA는 7D action을 이산화한 뒤 Llama2 vocabulary의 저빈도 token에 매핑하여 학습함.
- Action 구성은 `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]` 형태임.
- OFT는 최종 hidden state를 별도 MLP regression head로 보내 연속 action을 직접 출력함.
- 학습 loss도 cross-entropy에서 L1 regression으로 변경됨.

### 2.2 Autoregressive → parallel decoding

- OpenVLA는 action token을 순차적으로 생성함.
- OFT는 action embedding 사이의 causal mask를 제거하고 bidirectional attention을 사용함.
- 여러 action 차원을 한 번의 forward에서 병렬로 예측함.

### 2.3 Action chunking

- 한 번의 policy query에서 여러 미래 timestep의 action을 함께 예측함.
- 매 timestep마다 모델을 다시 호출하는 방식보다 추론 latency를 줄일 수 있음.
- 원 논문에서는 base 대비 action generation 속도와 latency 개선을 보고함.

### 2.4 입력 modality 확장

- OpenVLA는 주로 3인칭 이미지와 언어를 입력으로 사용함.
- OFT는 wrist camera와 proprioceptive state를 추가로 사용함.
- 현재 장면뿐 아니라 로봇 자체 상태와 손목 시점 정보를 함께 활용할 수 있게 됨.

## 3. OFT → OFT+

- OFT의 parallel decoding, action chunking, continuous action, L1 regression 구조는 유지함.
- 여기에 **FiLM(Feature-wise Linear Modulation)**을 추가함.
- 언어 instruction으로부터 scale `γ`와 shift `β`를 생성해 visual feature를 변조함.

```math
feature' = \gamma(language) \odot feature + \beta(language)
```

- 언어 조건이 단순히 마지막에 결합되는 것이 아니라 시각 feature 처리 과정에 직접 개입함.
- 원 논문에서 FiLM의 명시적 목적은 language grounding 강화임.

---

## 4. LIBERO-Plus 직접 평가 결과

### 4.1 전체 및 suite별

| 항목 | OFT | OFT+ | 차이 |
|---|---:|---:|---:|
| Overall | 71.20% | 80.96% | +9.76p |
| libero_spatial | 85.43% | 85.93% | +0.50p |
| libero_object | 68.86% | 83.92% | +15.06p |
| libero_goal | 64.65% | 72.33% | +7.68p |
| libero_10 | 66.69% | 82.14% | +15.45p |

### 4.2 Perturbation별

| Category | OFT | OFT+ | 차이 |
|---|---:|---:|---:|
| Background Textures | 94.05% | 96.47% | +2.42p |
| Camera Viewpoints | 58.66% | 95.31% | +36.65p |
| Language Instructions | 83.08% | 85.69% | +2.61p |
| Light Conditions | 92.12% | 94.66% | +2.54p |
| Objects Layout | 77.84% | 77.11% | -0.73p |
| Robot Initial States | 33.29% | 29.55% | -3.74p |
| Sensor Noise | 72.39% | 95.32% | +22.93p |

### 4.3 난이도별

| 모델 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| OFT | 91.85 | 88.37 | 75.98 | 66.28 | 35.14 |
| OFT+ | 92.03 | 89.15 | 81.95 | 78.69 | 63.95 |

- 난이도가 높아질수록 OFT+의 개선폭이 커지는 경향이 확인됨.

---

## 5. 결과 해석

> 아래 내용에서 실측 수치와 논문에 명시된 구조는 확인된 사실임. 성능 차이의 원인은 실험 결과를 바탕으로 한 해석 또는 가설임.

### 5.1 Base → OFT 개선

- L1 회귀는 이산 분류와 다른 형태의 objective를 사용하므로 demonstration noise에 대한 반응도 달라짐.
- Bidirectional attention을 통해 action representation끼리 서로 참조할 수 있음.
- Wrist camera와 proprioception이 추가되면서 3인칭 시점 하나에만 의존하지 않게 됨.
- 특히 camera 변화에서 입력 modality 확장이 직접적인 이점으로 작용했을 가능성이 있음.

### 5.2 OFT+에서 Camera / Sensor Noise가 크게 오른 이유

- Camera Viewpoints는 +36.65p, Sensor Noise는 +22.93p 상승함.
- 반면 FiLM의 직접적인 목표와 가까운 Language Instructions는 +2.61p 상승에 그쳤음.
- 실측 패턴만 보면 FiLM이 언어 grounding뿐 아니라 task-relevant visual feature를 강조하는 역할을 했을 가능성이 있음.
- Instruction이 물체와 gripper 관계에 집중하게 하면서 배경·시점·픽셀 손상의 영향을 줄였을 수 있다는 해석임.
- 원 논문에 직접 제시된 인과관계는 아니므로 가설로만 기록함.

### 5.3 Robot Initial States가 해결되지 않은 점

- OFT와 OFT+ 모두 Robot Initial States에서 약 30% 수준으로 낮음.
- OFT 계열의 주요 변경점은 현재 관측의 입력 정보와 action generation 방식에 집중되어 있음.
- 과거 observation/action을 지속적으로 누적해 state transition을 추적하는 명시적 temporal memory는 없음.
- Action chunking은 미래 action을 여러 step 예측하는 방식이지 과거 실행 오차를 누적해 상태를 추정하는 방식은 아님.
- 따라서 camera perturbation과 robot state perturbation은 서로 다른 robustness 문제일 가능성이 있음.

## 6. 남은 질문

- Camera perturbation과 Robot Initial State perturbation을 같은 방식으로 처리할 수 있는지 확인 필요함.
- 현재 frame의 representation robustness만으로 physical state mismatch까지 해결 가능한지 검증 필요함.
- Temporal memory 또는 closed-loop correction이 Robot Initial States 취약점을 줄이는지 실험 필요함.
- Objects Layout이 두 모델 모두 77% 부근에서 정체되는 이유도 추가 분석 필요함.

---

본 문서의 성능 수치는 직접 수행한 LIBERO-Plus 평가 결과를 사용함. 구조적 사실과 개인적인 해석은 구분해서 기록함.
