# NORA 아키텍처와 LIBERO-Plus 강건성 분석

## 1. 모델 개요

NORA는 **Qwen2.5-VL-3B**를 backbone으로 사용하는 약 3B 규모의 Vision-Language-Action 모델입니다. 본 평가에서는 `NORA-Long` 계열 체크포인트를 사용했으며 한 번의 추론에서 5-step action chunk를 생성합니다.

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

NORA는 OpenVLA-OFT 계열과 달리 wrist camera와 proprioception을 추가 입력으로 사용하지 않고 현재 시점의 3인칭 영상과 언어에 주로 의존합니다.

## 2. FAST+ Action Tokenizer

NORA는 연속 action을 직접 regression하는 대신 action sequence를 token으로 변환해 autoregressive하게 생성합니다.

큰 흐름은 다음과 같습니다.

1. Action sequence에 DCT(Discrete Cosine Transform)를 적용해 상관관계를 압축
2. 변환값을 정수 표현으로 변환
3. BPE(Byte-Pair Encoding)를 사용해 반복되는 action pattern을 token으로 압축
4. Qwen tokenizer vocabulary에 FAST+ action token 추가
5. Next-token prediction과 cross-entropy로 학습
6. 생성된 token을 다시 연속 7-DoF action으로 복원

즉 **연속 제어 문제를 압축된 이산 token 생성 문제로 바꾸는 구조**입니다.

## 3. OFT 계열과 구조 비교

| 항목 | NORA-Long | OpenVLA-OFT | OpenVLA-OFT+ |
|---|---|---|---|
| Backbone | Qwen2.5-VL-3B | OpenVLA/Llama2 7B | OpenVLA/Llama2 7B |
| Action 표현 | FAST+ token | Continuous | Continuous |
| Loss | Cross-entropy | L1 regression | L1 regression |
| Decoding | Autoregressive | Parallel | Parallel |
| Action chunk | 5 steps | 8 steps | 8 steps |
| 시각 입력 | 3인칭 RGB | 3인칭 + wrist | 3인칭 + wrist |
| Proprioception | 없음 | 있음 | 있음 |
| 언어 조건화 | Qwen VLM 내부 | Backbone 내부 | FiLM 추가 |

## 4. LIBERO-Plus 직접 평가 결과

총 10,030 episode를 평가했습니다.

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

공식 LIBERO-Plus 결과의 NORA 값은 약 39.0%이며 본 평가에서는 **39.49%**를 기록했습니다.

## 5. 취약점

### Camera Viewpoints — 3.25%
가장 극단적인 취약점입니다. 카메라 시점이 변하면 물체의 화면 좌표, 크기, 가림 관계와 로봇-물체 상대 관계가 변하고 action generation이 크게 무너졌습니다.

### Sensor Noise — 8.56%
픽셀 수준의 관측 손상이 action token sequence 전체에 큰 영향을 미쳤습니다. OFT 계열과 가장 큰 차이가 발생한 perturbation 중 하나입니다.

### 난이도가 높아질수록 급격한 하락

- L1: 62.77%
- L2: 45.46%
- L3: 41.79%
- L4: 31.34%
- L5: **18.43%**

복합적인 perturbation과 긴 복구가 필요한 상황에서 취약성이 더 커졌습니다.

## 6. 상대적인 강점

### Robot Initial States
NORA는 Robot Initial States에서 **36.13%**로 OFT와 OFT+보다 오히려 조금 높았습니다.

절대 성공률 자체는 낮지만, NORA가 모든 perturbation에서 일관되게 뒤처지는 것은 아니라는 점이 중요합니다.

### 언어 변화
Language Instructions에서 68.97%를 기록해 Camera / Noise 조건보다는 상당히 높은 수준을 유지했습니다. Qwen2.5-VL의 언어 및 instruction 이해 능력이 일정 부분 유지된 것으로 볼 수 있습니다.

### 모델 크기
약 3B 규모로 7B OpenVLA 계열보다 작아 메모리 측면에서 유리합니다. 다만 autoregressive action token 생성 때문에 모델 크기가 작다고 곧바로 closed-loop inference가 더 빠르다고 볼 수는 없습니다.

## 7. 핵심 해석

NORA의 실험 결과에서 가장 눈에 띄는 것은 **시각적 관측 변화와 로봇 상태 변화에 대한 robustness pattern이 서로 다르다는 점**입니다.

- Camera / Sensor Noise → 매우 취약
- Robot Initial States → 상대적으로 OFT 계열보다 조금 높음

따라서 VLA robustness를 하나의 단일 지표로 보는 것보다 **어떤 종류의 변화에 어떤 내부 메커니즘이 필요한지 분리해서 분석할 필요가 있습니다.**

---

> 실험 결과와 구조 설명은 직접 수행한 LIBERO-Plus 평가 및 논문 정리를 기반으로 하며, 원인에 대한 설명은 일부 연구적 해석을 포함합니다.
