# Mamba 핵심 원리 및 VLA 적용 실습

## 1. 실습 요약

Mamba의 selective state space mechanism을 직접 실행하고 Selective Copying 과제에서 LSTM, Transformer와 정확도, 길이 일반화, 실행 시간, GPU memory를 비교함.

세 개 seed 반복 실험에서 Mamba는 길이 64~512 구간에서 가장 높은 sequence accuracy를 기록함. 긴 sequence에서 Transformer보다 훨씬 완만한 memory 증가를 보였음.

다만 Selective Copying은 Mamba에 유리한 synthetic memory task임. 본 결과만으로 실제 VLA 전체에서 Mamba가 Transformer보다 항상 우수하다고 결론 내릴 수는 없음.

## 2. 실습 목적

- Mamba block과 selective SSM의 동작을 CUDA 환경에서 확인함.
- Selective Copying 문제를 직접 학습함.
- Mamba, LSTM, Transformer의 학습 성능과 길이 일반화를 비교함.
- Mamba 내부의 `Δ`, `B`, `C`와 상태 변화의 관계를 확인함.
- Sequence length에 따른 실행 시간과 GPU memory 확장성을 비교함.
- 결과를 VLA의 observation/action sequence 처리 관점에서 해석함.

## 3. 실행 환경

| 항목 | 설정 |
|---|---|
| 운영 환경 | Windows 11 + WSL2 Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 3070 |
| Python | 3.10 / conda `mamba` |
| PyTorch | 2.5.1+cu121 |
| Mamba | `mamba-ssm 2.2.2` |
| 작업 경로 | `~/projects/mamba-practice` |

설치 과정에서는 `mamba-ssm`, `causal-conv1d` source build 문제와 API version mismatch가 발생함. 최종적으로 호환 version을 맞춘 뒤 Mamba CUDA operation과 minimal forward test가 정상 동작함을 확인함.

## 4. 이론적 배경

### 4.1 Transformer

Self-attention은 각 token이 다른 token을 직접 참조할 수 있음.

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
```

Sequence length가 `L`일 때 attention matrix가 `L × L` 크기가 되므로 긴 sequence에서 time/memory 사용량이 빠르게 증가함.

### 4.2 LSTM

과거 정보를 hidden state와 cell state에 압축해 다음 시점으로 전달함. Attention matrix를 만들지 않으므로 length에 대해 비교적 선형적으로 처리되지만 긴 구간에서 정보 유지가 어려울 수 있음.

### 4.3 Mamba

핵심 recurrence는 다음처럼 볼 수 있음.

```math
h_t=\bar A_t h_{t-1}+\bar B_t x_t
```

```math
y_t=C_t h_t
```

입력에 따라 `Δ_t`, `B_t`, `C_t`가 달라지는 selective mechanism을 사용함.

- `Δ_t`: 상태 전이의 시간 간격 또는 변화 강도를 조절함.
- `B_t`: 현재 입력이 상태에 기록되는 방식을 조절함.
- `C_t`: 상태에서 현재 출력으로 읽어내는 방식을 조절함.

큰 `Δ`가 곧 중요한 token이나 큰 state change를 의미하는 것은 아님. 실제 state 변화는 기존 상태의 전이와 입력 write term이 함께 결정함.

## 5. Selective Copying 과제

### 5.1 데이터 구성

| Token | 의미 |
|---|---|
| `0` | noise |
| `1~8` | 기억할 data value |
| `9` | delimiter |
| `10` | query |

길이 64 sequence의 앞부분에 네 개 data token을 random position에 배치함. 마지막 네 query 위치에서 해당 값을 등장 순서대로 출력하도록 학습함.

### 5.2 공통 학습 조건

| 설정 | 값 |
|---|---:|
| 학습 sequence length | 64 |
| 기억할 token 수 | 4 |
| Batch size | 64 |
| Model dimension | 64 |
| Layer 수 | 2 |
| Mamba state dimension | 16 |
| Transformer head 수 | 4 |
| Training step | 1500 |
| Learning rate | 1e-3 |
| Seed | 42, 123, 777 |

## 6. 실습 진행 순서

1. Mamba minimal forward test 수행함.
2. Mamba Selective Copying 학습함.
3. Mamba와 LSTM 비교함.
4. 학습 length 64 기준 128, 256, 512, 1024 길이 일반화 평가함.
5. Mamba 내부 `Δ`, `B`, `C` 분석함.
6. Selective scan의 write / hold / read 역할을 근사 분석함.
7. Mamba, LSTM, Transformer forward scaling 비교함.
8. 동일 Selective Copying task에서 세 모델 학습 비교함.
9. 세 개 seed 반복 실험함.
10. 결과를 시각화함.

## 7. 초기 Mamba vs LSTM 비교

약 6.7만 parameter 규모에서 비교함.

| 모델 | 학습 시간 | Token Accuracy | Sequence Accuracy |
|---|---:|---:|---:|
| Mamba | 15.74 s | 99.83% | 99.41% |
| LSTM | 9.62 s | 99.63% | 98.66% |

짧은 sequence에서는 cuDNN 최적화가 잘 된 LSTM이 wall-clock 기준으로 더 빨랐음. Mamba는 더 적은 step에서 높은 accuracy에 도달하는 경향을 보였음.

## 8. 길이 일반화

학습 length는 64로 고정함.

| Length | Mamba Sequence Acc. | LSTM Sequence Acc. |
|---:|---:|---:|
| 64 | 100.00% | 98.83% |
| 128 | 100.00% | 87.50% |
| 256 | 99.92% | 27.81% |
| 512 | 96.88% | 2.42% |
| 1024 | 30.00% | 0.23% |

- Mamba는 512까지 높은 성능을 유지함.
- 1024에서는 크게 하락함.
- Mamba도 무한히 기억을 유지하는 구조는 아니며 작은 감쇠가 긴 구간에서 누적될 수 있음을 확인함.

## 9. Mamba 내부 parameter 분석

Token 종류별 평균값을 비교함.

| Token 종류 | 평균 `Δ` | 평균 `|B|` | 평균 `|C|` |
|---|---:|---:|---:|
| noise | 0.051341 | 1.517282 | 1.209835 |
| data | 0.033239 | 1.302168 | 0.890093 |
| delimiter | 0.036526 | 1.241158 | 0.797572 |
| query | 0.029682 | 1.025471 | **2.066289** |

Noise의 평균 `Δ`가 가장 컸지만 실제 state 변화는 가장 크지 않았음. Query에서 `|C|`가 크게 증가해 저장된 state를 강하게 읽는 경향이 나타남.

교육용 근사 분석에서는 다음 pattern이 관찰됨.

- Data token: state 변화량이 큼. 중요한 정보를 state에 쓰는 역할로 해석 가능함.
- Query token: write량은 작지만 read 값이 큼. 저장된 정보를 읽는 역할로 해석 가능함.
- Noise token: 계산은 발생하지만 최종 state 변화는 상대적으로 작았음.

실제 fused selective scan 전체를 그대로 복원한 값은 아니므로 각 norm을 gate 값처럼 직접 해석하면 안 됨.

## 10. Forward scaling benchmark

Batch size 16, sequence length 64~4096에서 forward time과 peak allocated GPU memory를 측정함.

| 모델 | Length | Time | GPU Memory |
|---|---:|---:|---:|
| Mamba | 1024 | 1.422 ms | 69.03 MB |
| Mamba | 2048 | 2.714 ms | 129.41 MB |
| Mamba | 4096 | 5.015 ms | 250.41 MB |
| LSTM | 1024 | 1.319 ms | 108.87 MB |
| LSTM | 2048 | 2.518 ms | 209.00 MB |
| LSTM | 4096 | 4.936 ms | 409.25 MB |
| Transformer | 1024 | 10.995 ms | 613.66 MB |
| Transformer | 2048 | 62.139 ms | 4428.79 MB |
| Transformer | 4096 | OOM | OOM |

- Mamba와 LSTM은 긴 sequence 구간에서 비교적 선형 증가를 보임.
- Transformer는 self-attention의 `L²` intermediate matrix 때문에 memory 사용량이 급격히 증가함.
- RTX 3070에서는 length 4096 Transformer가 OOM 발생함.
- 짧은 length의 Mamba timing은 custom kernel 초기화와 GPU clock 영향으로 다소 불규칙했음.

## 11. 세 모델 단일 seed 학습 결과

| 모델 | Parameters | Train Time | Len 64 Seq. Acc. | Len 512 Seq. Acc. |
|---|---:|---:|---:|---:|
| Mamba | 67,083 | 19.10 s | **100.00%** | **85.89%** |
| LSTM | 68,107 | 11.57 s | 97.14% | 0.10% |
| Transformer | 101,515 | 17.66 s | 47.97% | 0.16% |

Transformer는 1500 step 종료 시점에도 accuracy가 상승 중이었음. 충분히 수렴한 상태로 보기 어려움. Position encoding을 더 긴 위치에 계산할 수 있어도 학습하지 않은 position pattern까지 잘 해석한다는 보장은 없음.

## 12. 세 개 seed 반복 결과

| 모델 | Length | Token Acc. | Sequence Acc. |
|---|---:|---:|---:|
| Mamba | 64 | 99.96 ± 0.02% | **99.84 ± 0.09%** |
| Mamba | 128 | 99.98 ± 0.03% | **99.93 ± 0.12%** |
| Mamba | 256 | 99.49 ± 0.41% | **97.97 ± 1.64%** |
| Mamba | 512 | 94.46 ± 2.76% | **81.04 ± 8.56%** |
| LSTM | 64 | 98.96 ± 0.66% | 95.97 ± 2.43% |
| LSTM | 128 | 79.78 ± 9.31% | 45.28 ± 18.64% |
| LSTM | 256 | 40.61 ± 12.16% | 3.23 ± 3.71% |
| LSTM | 512 | 27.99 ± 8.19% | 0.80 ± 0.80% |
| Transformer | 64 | 83.06 ± 5.83% | 45.90 ± 18.14% |
| Transformer | 128 | 38.42 ± 5.45% | 1.09 ± 0.79% |
| Transformer | 256 | 33.89 ± 1.52% | 0.28 ± 0.06% |
| Transformer | 512 | 29.39 ± 3.52% | 0.12 ± 0.06% |

- Mamba는 64~256에서 높은 평균 accuracy와 작은 standard deviation을 보임.
- 512에서는 평균 81.04%를 유지했지만 표준편차가 8.56%p로 커짐.
- 매우 긴 length에서는 initialization과 data order에 따른 편차가 커질 수 있음을 확인함.

## 13. VLA 관점에서의 해석

VLA action generation은 모델마다 다름.

- Continuous action을 discrete token으로 바꿔 autoregressive하게 생성하는 방식이 있음.
- 여러 timestep의 action chunk를 한 번에 생성하는 방식이 있음.
- Diffusion / flow matching으로 continuous trajectory를 생성하는 방식도 있음.

이번 실험에서 Mamba는 긴 observation / proprioception / previous action history를 효율적으로 처리하는 temporal module 후보로 볼 수 있었음.

다만 전체 vision-language backbone을 Transformer에서 Mamba로 바꾸는 것이 항상 좋은 선택이라는 의미는 아님.

다음과 같은 hybrid 구조가 현실적인 방향으로 보임.

```text
카메라 이미지 + 자연어 명령
        ↓
Transformer 기반 vision-language encoder
        ↓
시각·언어 feature + 관절·IMU·이전 action history
        ↓
Mamba temporal encoder / action decoder
        ↓
continuous action 또는 action chunk
```

- Transformer는 image patch와 language token의 복잡한 관계를 직접 참조하는 데 강함.
- Mamba는 긴 temporal history를 선형 비용과 고정 크기 state로 처리하는 데 장점이 있음.

## 14. 실험 한계

- Selective Copying은 희소한 중요 정보를 오래 기억하는 synthetic task라 Mamba에 유리한 inductive bias가 있음.
- Transformer hyperparameter와 training step을 별도로 최적화하지 않음.
- 세 seed만 사용했으므로 충분한 통계적 반복은 아님.
- 내부 state 분석은 selective scan의 교육용 근사임.
- 실제 VLA 성능은 vision encoder, robot data, action representation, control frequency, embodiment 등 많은 요소에 좌우됨.

## 15. 최종 결론

Mamba는 Selective Copying 문제를 안정적으로 학습했고 학습 length보다 긴 input에서도 LSTM과 Transformer보다 높은 accuracy를 유지함. 긴 sequence에서 Transformer보다 훨씬 낮은 memory 증가도 확인함.

핵심 장점은 `Transformer보다 무조건 정확함`이 아니라 **중요한 정보를 선택적으로 state에 기록하면서 긴 sequence를 효율적으로 처리할 수 있는 구조**라는 점임.

VLA에서는 전체 backbone 대체보다 observation/action history를 처리하는 temporal encoder 또는 action decoder로 활용하는 방향이 더 타당하다고 판단함.

## 16. 재현 파일

```text
~/projects/mamba-practice/
├── test_mamba.py
├── compare_mamba_lstm.py
├── benchmark_scaling.py
├── compare_three_models.py
├── multi_seed_experiment.py
├── multi_seed_results.csv
├── plot_multi_seed_results.py
└── figures/
```

실행 형태는 다음과 같음.

```bash
cd ~/projects/mamba-practice
PYTHONWARNINGS=ignore::FutureWarning python compare_three_models.py
PYTHONWARNINGS=ignore::FutureWarning python multi_seed_experiment.py
python plot_multi_seed_results.py
```
