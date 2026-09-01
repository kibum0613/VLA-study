# VLA Study

Vision-Language-Action(VLA), Physical AI, Robot Learning을 공부하고 직접 실험한 내용을 정리하는 저장소임.

논문 요약만 모으기보다 **모델 구조 이해 → 직접 실행 → benchmark 재현 → failure pattern 분석** 과정까지 기록하는 것을 목표로 함.

## Contents

### VLA Models / Benchmark

- [01. OpenVLA vs OFT vs OFT+ 비교 분석](docs/01_openvla_oft_comparison.md)
- [02. NORA 아키텍처와 LIBERO-Plus 강건성 분석](docs/02_nora.md)
- [03. π₀ 아키텍처와 LIBERO-Plus 평가](docs/03_pi0.md)
- [04. RoboMamba 논문 리뷰](docs/04_robomamba.md)
- [05. LIBERO-Plus 다중 VLA 강건성 평가](docs/05_libero_plus_benchmark.md)

### Hands-on / Reproduction

- [06. OpenVLA 논문 및 실습 정리](docs/06_openvla_practice.md)
- [07. SUREFlow + LIBERO 실습 및 재현 기록](docs/07_sureflow_reproduction.md)
- [08. Mamba 핵심 원리 및 VLA 적용 실습](docs/08_mamba_practice.md)
- [09. Mamba 원 논문 리뷰](docs/09_mamba_paper_review.md)

## 직접 수행한 LIBERO-Plus 결과

| Model | Overall Success Rate |
|---|---:|
| OpenVLA-OFT+ | **80.96%** |
| OpenVLA-OFT | **71.20%** |
| π₀ | **46.49%** |
| NORA | **39.49%** |

동일한 benchmark에서도 모델별 failure pattern이 크게 다르게 나타남. 따라서 overall score뿐 아니라 Camera Viewpoints, Robot Initial States, Sensor Noise 등 perturbation별 결과를 함께 분석함.

## 주요 공부 흐름

```text
OpenVLA
  ↓
OpenVLA-OFT / OFT+
  ↓
LIBERO-Plus robustness evaluation
  ↓
NORA / π₀ 비교
  ↓
Mamba / RoboMamba
  ↓
Temporal modeling / closed-loop VLA 문제 탐색
```

## 실습 내용

- OpenVLA pretrained checkpoint inference 수행함.
- OpenVLA와 LIBERO environment를 연결해 closed-loop rollout을 실행함.
- SUREFlow를 WSL2와 GPU 서버에서 직접 학습·평가함.
- SUREFlow 재현 과정에서 CUDA, Mamba, robosuite, LIBERO, MuJoCo dependency 문제를 해결함.
- OpenVLA-OFT, OFT+, NORA, π₀의 LIBERO-Plus 10,030 episode 평가를 수행함.
- Mamba, LSTM, Transformer를 Selective Copying task에서 직접 비교함.
- Mamba의 length generalization과 GPU memory scaling을 실험함.

## 관심 주제

- Vision-Language-Action Models
- Physical AI / Robot Learning
- Robot Manipulation
- Robust & Adaptive Manipulation
- Temporal Modeling / State Space Models
- Predictive Representation Learning
- World Models
- Mobile / Quadruped Manipulation

## 기록 원칙

- 논문에 명시된 사실과 직접 실험한 결과를 구분함.
- 실험 결과를 바탕으로 한 개인적인 해석은 별도로 표시함.
- 재현에 실패한 결과도 삭제하지 않고 원인과 함께 기록함.
- 실행 환경과 protocol 차이가 결과에 영향을 줄 수 있음을 고려함.
- 진행 중인 미공개 연구 아이디어나 내부 연구 내용은 공개하지 않음.
