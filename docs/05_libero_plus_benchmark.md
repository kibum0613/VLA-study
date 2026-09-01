# LIBERO-Plus 다중 VLA 강건성 평가

## 1. 실험 목적

기존 LIBERO는 manipulation policy의 task 수행 성능을 보기에는 적합하지만 실제 환경 변화에 대한 강건성을 세부적으로 보기에는 한계가 있음.

LIBERO-Plus는 기존 LIBERO task에 여러 perturbation을 추가해 **VLA가 환경 변화에 얼마나 강건한지** 확인할 수 있는 benchmark임.

본 실험에서는 총 10,030 episode를 기준으로 공개 VLA 모델을 직접 실행하고 비교함.

## 2. Perturbation Categories

1. **Background Textures** — 배경 및 재질 변화
2. **Camera Viewpoints** — 카메라 위치, 각도, FOV 변화
3. **Robot Initial States** — manipulator 초기 pose 변화
4. **Language Instructions** — instruction 표현 변화
5. **Light Conditions** — 조명 세기, 방향, 색 변화
6. **Sensor Noise** — blur, photometric noise 등 이미지 손상
7. **Objects Layout** — target/distractor object 배치 변화

각 perturbation이 요구하는 robustness 성격은 서로 다름.

- Camera Viewpoints는 visual representation의 invariance가 중요할 수 있음.
- Robot Initial States는 현재 로봇 상태와 trajectory를 다시 해석하고 수정하는 능력이 더 중요할 수 있음.
- Sensor Noise는 시각 입력의 안정성과 직접적으로 연결됨.

## 3. 평가 Suite

| Suite | Episodes | Max Steps |
|---|---:|---:|
| libero_spatial | 2,402 | 220 |
| libero_object | 2,518 | 280 |
| libero_goal | 2,591 | 300 |
| libero_10 | 2,519 | 520 |
| **Total** | **10,030** | - |

모델마다 inference stack이 달라도 비교가 가능하도록 environment step budget을 통일함.

## 4. 현재까지 평가한 모델

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

단순 overall 점수보다 perturbation별 failure pattern 차이가 더 흥미로웠음.

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

- OFT+는 Camera Viewpoints와 Sensor Noise에서 큰 폭으로 개선됨.
- Robot Initial States는 오히려 OFT+가 조금 낮았음.
- 시각적 domain shift와 robot state mismatch가 서로 다른 문제일 가능성이 있음.

## 7. NORA Failure Pattern

| Category | NORA |
|---|---:|
| Camera Viewpoints | **3.25%** |
| Sensor Noise | **8.56%** |
| Robot Initial States | **36.13%** |
| Language Instructions | 68.97% |

- Camera / Sensor Noise에서 극단적으로 낮았음.
- Robot Initial States에서는 OFT 계열보다 조금 높았음.
- 한 모델을 단순히 robust / non-robust로 나누기보다 어떤 perturbation에서 어떤 구조가 영향을 받는지 봐야 함.

## 8. π₀ 결과

| Suite | π₀ | NORA |
|---|---:|---:|
| Spatial | 54.20% | 51.87% |
| Object | **52.74%** | 33.32% |
| Goal | 37.75% | **39.06%** |
| LIBERO-10 | **41.88%** | 34.30% |

- π₀ overall은 46.49%로 NORA보다 7.00%p 높았음.
- 특히 Object에서 차이가 크게 나타남.
- Flow-matching 기반 continuous action generation과 token 기반 action generation의 차이를 추가로 분석할 가치가 있음.

## 9. 실험을 하면서 생긴 질문

### 9.1 Feature invariance만으로 충분한가?

- Camera 변화처럼 task와 무관한 관측 변화는 invariant representation이 도움될 수 있음.
- Robot Initial State처럼 실제 physical state가 바뀐 경우에는 action 자체가 달라져야 함.
- 모든 perturbation을 같은 invariance objective로 처리하면 안 될 가능성이 있음.

### 9.2 Open-loop action chunking의 한계는 무엇인가?

- 여러 미래 action을 한 번에 생성하면 효율적임.
- 실행 중 발생한 mismatch를 chunk 내부에서 즉시 반영하지 못할 수 있음.
- 긴 horizon에서 작은 오차가 누적될 가능성이 있음.

### 9.3 Temporal information이 필요한가?

현재 observation만 보는 것보다 아래 정보를 함께 사용하는 구조가 robustness에 도움될 수 있음.

- previous observations
- previous actions
- execution errors
- task progress
- correction history

### 9.4 Closed-loop correction은 어떻게 설계해야 하는가?

- 모든 변화에서 원래 action을 그대로 유지하는 것이 목적은 아님.
- Task intent는 유지하면서 physical state에 맞게 action을 수정할 수 있어야 함.
- 실제 실행 이후 새 observation을 다시 사용해 correction 결과를 확인하는 구조가 필요함.

## 10. 정리

현재까지의 결과에서 가장 크게 확인한 부분은 **perturbation 종류에 따라 failure mechanism이 다르게 나타난다는 점**임.

```text
Visual robustness
    ≠
Robot-state robustness
    ≠
Long-horizon recovery
```

고정형 manipulator benchmark에서 각 failure pattern을 먼저 분석한 뒤 mobile manipulator나 quadruped + arm 환경으로 확장하는 방향을 검토 중임.

---

본 문서의 benchmark 수치는 직접 수행한 실험 결과를 사용함. 원인에 대한 설명은 일부 연구적 해석을 포함하며 추가 검증이 필요함.
