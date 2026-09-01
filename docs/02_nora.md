# NORA 아키텍처와 LIBERO-Plus 강건성 분석

## 1. 모델 개요

- NORA는 **Qwen2.5-VL-3B**를 backbone으로 사용하는 약 3B 규모의 VLA 모델임.
- 본 평가에서는 `NORA-Long` 계열 체크포인트를 사용했으며 한 번의 추론에서 5-step action chunk를 생성함.

```text
3인칭 RGB + 자연어 instruction
            ↓
      Qwen2.5-VL-3B
            ↓
  FAST+ action token 생성
            ↓
      FAST+ decoder
            ↓
  5-step × 7-DoF action
```

- OpenVLA-OFT 계열과 달리 wrist camera와 proprioception을 추가 입력으로 사용하지 않음.
- 현재 시점의 3인칭 RGB와 언어 정보에 대한 의존도가 높음.

## 2. FAST+ Action Tokenizer

NORA는 연속 action을 직접 regression하지 않고 action sequence를 token으로 변환해 autoregressive하게 생성함.

1. Action sequence에 DCT(Discrete Cosine Transform)를 적용해 시간축 상관관계를 압축함.
2. 변환값을 정수 표현으로 변경함.
3. BPE(Byte-Pair Encoding)를 사용해 반복되는 action pattern을 token sequence로 압축함.
4. Qwen tokenizer vocabulary에 FAST+ action token을 추가함.
5. Next-token prediction과 cross-entropy loss로 학습함.
6. 생성된 token을 FAST+ decoder가 다시 연속 7-DoF action으로 복원함.

즉 연속 제어 문제를 압축된 이산 token 생성 문제로 바꾼 구조임.

## 3. OFT 계열과 구조 비교

| 항목 | NORA-Long | OpenVLA-OFT | OpenVLA-OFT+ |
|---|---|---|---|
| Backbone | Qwen2.5-VL-3B | OpenVLA/Llama2 7B | OpenVLA/Llama2 7B |
| Action 표현 | FAST+ token | continuous | continuous |
| Loss | cross-entropy | L1 regression | L1 regression |
| Decoding | autoregressive | parallel | parallel |
| Action chunk | 5 steps | 8 steps | 8 steps |
| 시각 입력 | 3인칭 RGB | 3인칭 + wrist | 3인칭 + wrist |
| Proprioception | 없음 | 있음 | 있음 |
| 언어 조건화 | Qwen VLM 내부 | backbone 내부 | FiLM 추가 |

## 4. LIBERO-Plus 직접 평가 결과

총 10,030 episode를 평가함.

| Category | NORA | OFT | OFT+ |
|---|---:|---:|---:|
| Background Textures | 61.25% | 94.05% | 96.47% |
| Camera Viewpoints | **3.25%** | 58.66% | 95.31% |
| Language Instructions | 68.97% | 83.08% | 85.69% |
| Light Conditions | 51.66% | 92.12% | 94.66% |
| Objects Layout | 59.21% | 77.84% | 77.11% |
| Robot Initial States | **36.13%** | 33.29% | 29.55% |
| Sensor Noise | **8.56%** | 72.39% | 95.32% |
| Overall | **39.49%** | 71.20% | 80.96% |

- 공식 LIBERO-Plus NORA 결과는 약 39.0%임.
- 본 평가에서는 39.49%를 기록해 공식 수치와 유사하게 재현됨.

## 5. 취약점

### Camera Viewpoints — 3.25%

- 가장 극단적인 취약점으로 확인됨.
- 시점이 바뀌면 물체 화면 좌표, 크기, 가림 관계, 로봇-물체 상대 관계가 함께 변함.
- 3인칭 단일 시점 의존 구조가 영향을 받았을 가능성이 있음.

### Sensor Noise — 8.56%

- 픽셀 수준의 관측 손상이 action token sequence 전체에 큰 영향을 미친 것으로 보임.
- OFT 계열과 가장 큰 성능 차이가 발생한 perturbation 중 하나임.

### 난이도별

| Level | Success Rate |
|---|---:|
| L1 | 62.77% |
| L2 | 45.46% |
| L3 | 41.79% |
| L4 | 31.34% |
| L5 | 18.43% |

- perturbation이 복합적으로 들어가거나 복구 과정이 길어질수록 성능이 급격히 하락함.

## 6. 상대적인 강점

### Robot Initial States

- 36.13%로 절대 성능은 낮지만 OFT 33.29%, OFT+ 29.55%보다 높았음.
- NORA가 모든 perturbation에서 일관되게 뒤처지는 구조는 아님을 확인함.

### Language Instructions

- 68.97%를 기록함.
- Camera / Sensor Noise 조건과 비교하면 상대적으로 높은 수준을 유지함.
- Qwen2.5-VL의 언어 및 instruction 처리 능력이 일정 부분 유지된 것으로 해석 가능함.

### 모델 크기

- 약 3B 규모로 7B OpenVLA 계열보다 작아 GPU 메모리 측면에서는 유리함.
- 다만 autoregressive action token 생성 방식이므로 작은 모델 크기가 곧 빠른 closed-loop inference를 의미하지는 않음.

## 7. 정리

- NORA의 failure pattern은 관측 변화와 robot state 변화에서 뚜렷하게 다르게 나타남.
- Camera / Sensor Noise에는 매우 취약했지만 Robot Initial States에서는 OFT 계열보다 조금 높았음.
- VLA robustness를 하나의 점수로 보기보다 perturbation 종류별로 나눠 보는 것이 필요함.
- 어떤 변화는 representation 문제가 되고 어떤 변화는 state tracking이나 replanning 문제가 될 수 있음.

---

실험 결과와 구조 설명은 직접 수행한 LIBERO-Plus 평가 및 논문 정리를 기반으로 작성함. 성능 원인에 대한 부분은 일부 해석을 포함함.
