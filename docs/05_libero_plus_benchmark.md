# LIBERO-Plus 다중 VLA 강건성 평가

## 1. 왜 LIBERO-Plus를 사용했는가

기존 LIBERO는 로봇 manipulation policy의 task 수행 능력을 평가하는 데 유용하지만, 실제 환경에서 발생할 수 있는 다양한 변화에 대한 robustness를 세밀하게 보기 어렵습니다.

LIBERO-Plus는 기존 task를 기반으로 여러 perturbation을 적용해 **VLA가 환경 변화에 얼마나 강건한지** 평가할 수 있도록 구성된 benchmark입니다.

본 실험에서는 총 **10,030 episode**를 기준으로 여러 공개 VLA 모델을 직접 실행하고 비교했습니다.

## 2. Perturbation Categories

평가에서 사용한 주요 변화는 다음과 같습니다.

1. **Background Textures** — 배경/재질 변화
2. **Camera Viewpoints** — 카메라 위치, 각도, FOV 변화
3. **Robot Initial States** — manipulator 초기 pose 변화
4. **Language Instructions** — instruction 표현 변화
5. **Light Conditions** — 조명 세기, 방향, 색 변화
6. **Sensor Noise** — blur, photometric noise 등 이미지 손상
7. **Objects Layout** — target / distractor object 배치 변화

이 perturbation들은 모두 같은 종류의 robustness를 요구하지 않습니다.

예를 들어 Camera Viewpoint는 visual representation의 invariance가 중요하지만 Robot Initial State는 현재 로봇 상태와 trajectory를 다시 해석하고 수정하는 능력이 더 중요할 수 있습니다.

## 3. 평가 Suite

| Suite | Episodes |
|---|---:|
| libero_spatial | 2,402 |
| libero_object | 2,518 |
| libero_goal | 2,591 |
| libero_10 | 2,519 |
| **Total** | **10,030** |

모델별 inference stack이 달라도 비교가 가능하도록 suite별 최대 environment step budget을 통일했습니다.

| Suite | Max Steps |
|---|---:|
| libero_spatial | 220 |
| libero_object | 280 |
| libero_goal | 300 |
| libero_10 | 520 |

## 4. 직접 평가한 모델

현재까지 전체 benchmark를 완료한 주요 모델은 다음과 같습니다.

- OpenVLA-OFT
- OpenVLA-OFT+
- NORA / NORA-Long
- π₀

## 5. Overall Results

| Model | Overall Success Rate |
|---|---:|
| **OpenVLA-OFT+** | **80.96%** |
| **OpenVLA-OFT** | **71.20%** |
| **π₀** | **46.49%** |
| **NORA** | **39.49%** |

점수 자체뿐 아니라 모델마다 실패 pattern이 크게 달랐습니다.

## 6. OFT vs OFT+

| Category | OFT | OFT+ |
|---|---:|---:|
| Background Textures | 94.05% | 96.47% |
| Camera Viewpoints | 58.66% | **95.31%** |
| Language Instructions | 83.08% | 85.69% |
| Light Conditions | 92.12% | 94.66% |
| Objects Layout | 77.84% | 77.11% |
| Robot Initial States | **33.29%** | 29.55% |
| Sensor Noise | 72.39% | **95.32%** |

OFT+는 Camera Viewpoints와 Sensor Noise에서 큰 폭으로 개선됐지만 Robot Initial States는 여전히 매우 낮았습니다.

이 결과는 **시각적 domain shift와 로봇 state mismatch가 서로 다른 문제일 수 있음**을 보여주는 흥미로운 사례입니다.

## 7. NORA의 Failure Signature

NORA는 전체 성공률은 낮았지만 perturbation별 결과가 매우 특징적이었습니다.

| Category | NORA |
|---|---:|
| Camera Viewpoints | **3.25%** |
| Sensor Noise | **8.56%** |
| Robot Initial States | **36.13%** |
| Language Instructions | 68.97% |

Camera / Noise에는 극도로 취약했지만 Robot Initial States에서는 OFT 계열보다 조금 높은 결과를 보였습니다.

따라서 단순히 "모델 A가 모델 B보다 robust하다"고 표현하기보다 **어떤 perturbation에 어떤 구조가 강하거나 약한지** 보는 것이 중요합니다.

## 8. π₀ 결과

π₀는 전체 **46.49%**로 NORA보다 높았습니다.

특히 LIBERO-Object에서 NORA와 차이가 컸습니다.

| Suite | π₀ | NORA |
|---|---:|---:|
| Spatial | 54.20% | 51.87% |
| Object | **52.74%** | 33.32% |
| Goal | 37.75% | **39.06%** |
| LIBERO-10 | **41.88%** | 34.30% |

π₀의 flow-matching 기반 continuous action generation이 token 기반 NORA와 어떤 차이를 만드는지 추가 분석할 가치가 있습니다.

## 9. 실험에서 얻은 핵심 질문

### 1) Feature invariance만으로 충분한가?
Camera 변화처럼 task와 무관한 관측 변화는 invariant representation으로 처리할 수 있지만 Robot Initial State처럼 실제 physical state가 바뀐 경우에는 action도 달라져야 합니다.

### 2) Open-loop Action Chunking의 한계는 무엇인가?
여러 미래 action을 한 번에 생성하면 효율적이지만, 실행 도중 발생한 mismatch를 즉시 반영하지 못할 수 있습니다.

### 3) Temporal information이 필요한가?
현재 observation만 보는 것보다 다음 정보를 함께 사용하는 것이 robustness에 도움이 될 수 있습니다.

- Previous observations
- Previous actions
- Execution errors
- Task progress
- Correction history

### 4) Closed-loop correction은 어떻게 설계해야 하는가?
모든 변화에서 원래 action을 그대로 유지하는 것이 아니라, **task intent는 유지하면서 physical state에 맞게 action을 수정하는 구조**가 필요할 수 있습니다.

## 10. 다음 연구 방향

현재 관심 있는 방향은 다음과 같습니다.

```text
Robust VLA Representation
        +
Temporal State Tracking
        +
Closed-loop Correction
        ↓
Robust & Adaptive Manipulation
        ↓
Mobile / Quadruped Manipulation
```

고정형 manipulator benchmark에서 perturbation별 failure mechanism을 먼저 분석한 뒤, 향후 mobile manipulator 또는 quadruped + arm 환경으로 확장하는 것을 목표로 합니다.

---

> 이 저장소의 벤치마크 수치는 동일 연구 환경에서 직접 실행한 결과입니다. 일부 원인 분석은 실험 결과를 바탕으로 한 연구 가설이며 추가 검증이 필요합니다.
