# VLA Study

Vision-Language-Action(VLA), Physical AI, Robot Learning을 공부하고 직접 실험한 내용을 정리하는 저장소입니다.

모델 구조를 이해하고 실제 LIBERO-Plus 벤치마크에서 재현·비교한 결과를 중심으로 기록합니다.

## Contents

### VLA Models
- [OpenVLA / OpenVLA-OFT / OFT+ 비교](docs/01_openvla_oft_comparison.md)
- [NORA 아키텍처와 강건성 분석](docs/02_nora.md)
- [π₀ 아키텍처와 LIBERO-Plus 평가](docs/03_pi0.md)
- [RoboMamba 논문 리뷰](docs/04_robomamba.md)

### Benchmark & Experiments
- [LIBERO-Plus 다중 VLA 평가 정리](docs/05_libero_plus_benchmark.md)

## 현재까지의 주요 실험 결과

| Model | LIBERO-Plus Overall |
|---|---:|
| OpenVLA-OFT+ | **80.96%** |
| OpenVLA-OFT | **71.20%** |
| π₀ | **46.49%** |
| NORA | **39.49%** |

> 위 결과는 동일 연구 환경에서 직접 수행한 LIBERO-Plus 평가 결과를 정리한 값입니다. 모델별 입력 및 추론 구조가 서로 다르므로 단순 점수뿐 아니라 perturbation별 실패 특성을 함께 분석합니다.

## 관심 주제

- Vision-Language-Action Models
- Physical AI / Robot Learning
- Robust & Adaptive Manipulation
- Temporal Modeling / State Space Models
- World Models & Predictive Representation Learning
- Mobile / Quadruped Manipulation

## Notes

이 저장소는 개인 학습 및 연구 기록입니다. 논문에 명시된 사실과 실험 결과, 개인적인 해석·연구 가설을 구분해서 기록하려고 합니다.
