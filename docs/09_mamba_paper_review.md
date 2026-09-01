# Paper Review: Mamba — Linear-Time Sequence Modeling with Selective State Spaces

## 0. 논문 정보

**논문명**  
*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*

**저자**  
Albert Gu, Tri Dao

**핵심 키워드**  
Mamba, SSM, Selective State Space Model, S6, Selective Scan, Long Context Modeling

**한 줄 요약**  
기존 SSM의 긴 sequence 처리 효율성은 유지하면서 입력 내용에 따라 무엇을 기억하고 버릴지 선택할 수 있도록 만든 모델임.

---

## 1. Transformer의 장점과 한계

Transformer의 핵심은 self-attention임.

각 token이 다른 token을 직접 참고할 수 있어 긴 문맥 안에서 중요한 정보를 찾는 데 강함.

문제는 sequence length가 길어질수록 attention matrix 크기가 커진다는 점임.

- Training 시 time / memory cost가 length에 대해 대략 quadratic하게 증가함.
- Autoregressive inference에서는 과거 key-value를 저장하는 KV cache가 필요함.
- Context가 길어질수록 memory 사용량도 커짐.

즉 성능은 강하지만 긴 sequence 처리 비용이 큼.

## 2. SSM이 등장한 이유

State Space Model은 긴 sequence를 효율적으로 처리하기 위해 다시 주목받은 구조임.

기본 형태는 다음과 같음.

```math
h_t=\bar A h_{t-1}+\bar B x_t
```

```math
y_t=C h_t
```

원래 SSM은 연속시간 상태방정식에서 출발함.

```math
h'(t)=Ah(t)+Bx(t)
```

```math
y(t)=Ch(t)
```

이 연속시간 system을 discretization하면 deep learning에서 사용하는 recurrence 형태가 됨.

## 3. SSM과 RNN의 차이

### 공통점

둘 다 이전 hidden state와 현재 입력으로 다음 state를 만듦.

### 차이

일반 RNN은 neural recurrence를 직접 학습함.

```math
h_t=f(W_hh_{t-1}+W_xx_t)
```

SSM은 연속시간 dynamical system에서 유도된 structured recurrence임.

기존 SSM의 중요한 특징은 `Ā`, `B̄`, `C`가 시간에 대해 고정되어 있다는 점임.

Recurrence를 펼치면 다음과 같은 convolution kernel을 만들 수 있음.

```math
K=(C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots)
```

```math
y=x*K
```

즉 inference에서는 recurrent하게 볼 수 있지만 training에서는 convolution 형태로 병렬 계산할 수 있었음.

## 4. 기존 SSM의 한계

기존 SSM의 핵심 약점은 **content-based selection이 약하다는 점**임.

모든 timestep에서 동일한 transition rule을 사용함.

중요한 token과 중요하지 않은 token이 들어와도 state update 규칙 자체는 크게 바뀌지 않음.

예를 들어 다음 sequence가 있다고 가정함.

```text
A / noise / noise / B / noise / C
```

중요한 정보 A, B, C만 기억하고 noise를 무시해야 하지만 기존 linear time-invariant SSM은 입력 내용에 따라 update rule을 바꾸기 어려움.

Transformer는 attention을 통해 필요한 token을 직접 다시 볼 수 있지만 SSM은 모든 정보를 fixed-size hidden state에 압축해야 함.

따라서 어떤 정보를 기억하고 어떤 정보를 버릴지가 중요해짐.

## 5. Mamba의 핵심 아이디어

Mamba는 SSM parameter 일부를 현재 입력의 함수로 만듦.

기존 SSM은 다음처럼 볼 수 있음.

```text
A, B, C, Δ = 시간에 대해 고정
```

Mamba에서는 다음과 같이 바뀜.

```math
B_t=B(x_t)
```

```math
C_t=C(x_t)
```

```math
\Delta_t=\Delta(x_t)
```

따라서 실제 recurrence도 token마다 달라짐.

```math
h_t=\bar A_t h_{t-1}+\bar B_t x_t
```

```math
y_t=C_t h_t
```

현재 입력을 보고 state update와 read 방식이 달라질 수 있음.

## 6. `B_t`, `C_t`, `Δ_t`의 역할

### `B_t`

현재 입력을 hidden state에 어떤 방향으로 기록할지 결정함.

쉽게 보면 **write** 역할에 가까움.

### `C_t`

현재 hidden state에서 어떤 정보를 출력으로 읽을지 결정함.

쉽게 보면 **read** 역할에 가까움.

### `Δ_t`

현재 입력에서 state transition의 시간 scale을 조절함.

작은 `Δ_t`는 이전 state를 더 오래 유지하는 방향으로 작동할 수 있고 큰 `Δ_t`는 현재 입력에 따라 state를 빠르게 변화시키는 방향으로 작동할 수 있음.

다만 `Δ_t` 하나만 보고 중요도를 판단하면 안 됨. 실제 state change는 `A`, `B_t`, `Δ_t`, 기존 state가 함께 결정함.

## 7. Projection의 의미

논문에서는 현재 input representation에서 선택적 parameter를 만들기 위해 linear projection을 사용함.

```math
s_B(x)=Linear_N(x)
```

```math
s_C(x)=Linear_N(x)
```

```math
s_\Delta(x)=Broadcast_D(Linear_1(x))
```

Projection은 현재 input vector를 SSM parameter 생성에 필요한 차원으로 바꾸는 학습 가능한 linear layer임.

예를 들어 512차원 token representation에서 16차원 `B_t`가 필요하면 linear projection으로 변환함.

`Δ`에는 양수 제약이 필요하므로 softplus를 사용함.

```math
softplus(z)=\log(1+e^z)
```

## 8. 왜 기존 SSM의 convolution을 그대로 못 쓰는가

기존 SSM은 transition parameter가 시간에 대해 고정되어 있음.

따라서 과거 input이 현재 output에 미치는 영향이 몇 timestep 떨어져 있는지에 따라 고정됨.

Mamba는 `Ā_t`, `B̄_t`, `C_t`가 입력에 따라 달라짐.

즉 동일한 거리의 과거 input이라도 중간에 어떤 token이 있었는지에 따라 영향이 달라짐.

고정 convolution kernel을 만들 수 없게 됨.

이 때문에 새로운 계산 방법이 필요해짐.

## 9. Selective Scan

Mamba recurrence는 다음처럼 표현할 수 있음.

```math
h_t=A_t h_{t-1}+b_t
```

각 step을 affine function으로 보면 다음과 같음.

```math
f_t(h)=A_t h+b_t
```

여러 step은 함수 합성으로 표현 가능함.

```math
h_t=f_t\circ f_{t-1}\circ \dots \circ f_1(h_0)
```

Affine update끼리 합치면 다시 affine update가 됨.

두 step을 예로 들면 다음과 같음.

```math
h_1=A_1h_0+b_1
```

```math
h_2=A_2h_1+b_2
```

대입하면 다음과 같음.

```math
h_2=A_2A_1h_0+A_2b_1+b_2
```

따라서 두 update를 다음처럼 하나로 합칠 수 있음.

```math
(A_2,b_2)\circ(A_1,b_1)=(A_2A_1, A_2b_1+b_2)
```

이 연산은 결합법칙을 만족함. 그래서 prefix scan 형태로 병렬 계산할 수 있음.

일반 RNN은 중간에 `tanh` 같은 nonlinear activation이 들어가 여러 step을 같은 간단한 형태로 합치기 어려움.

## 10. Hardware-aware Algorithm

Mamba는 selective scan을 GPU에서 효율적으로 실행하기 위해 hardware-aware 구현을 사용함.

### 10.1 Kernel Fusion

여러 연산을 따로 실행하면 intermediate tensor를 HBM에 반복적으로 읽고 써야 함.

관련 연산을 하나의 kernel 안에서 처리해 memory traffic을 줄임.

### 10.2 SRAM 활용

큰 hidden state tensor 전체를 HBM에 저장하지 않고 필요한 parameter를 SRAM으로 가져와 scan을 수행함.

최종 output만 HBM으로 내보내는 방향으로 memory movement를 줄임.

### 10.3 Recomputation

Backward에 필요한 모든 intermediate state를 forward에서 저장하지 않음.

Backward 시 일부 state를 다시 계산하는 방식으로 memory 사용량을 줄임.

Compute를 조금 더 쓰는 대신 memory를 절약함.

## 11. Mamba Block

Mamba block은 기존 SSM block과 MLP block을 단순화해 하나의 반복 구조로 구성함.

주요 요소는 다음과 같음.

- Input projection
- Local convolution
- Selective SSM
- SiLU / Swish activation
- Output projection
- Residual connection
- Normalization

Attention을 사용하지 않고 동일한 block을 반복해서 쌓는 구조임.

## 12. Selective mechanism이 유리한 상황

### Variable Spacing

중요한 token 사이의 noise 간격이 계속 바뀌는 경우임.

```text
A / □ / □ / B / □ / C
```

Fixed LTI filter는 위치 pattern이 계속 달라지면 어려워질 수 있음. Mamba는 token content를 보고 필요한 정보를 state에 기록할 수 있음.

### Context Filtering

긴 context에서 관련 없는 과거 정보까지 유지하면 오히려 성능이 떨어질 수 있음.

Selective mechanism은 불필요한 정보를 약하게 반영하거나 state를 빠르게 바꾸는 방식으로 context를 필터링할 수 있음.

### Boundary Resetting

여러 document를 이어 붙인 sequence에서는 document boundary를 넘어 정보가 섞이면 안 됨.

Transformer는 attention mask를 사용할 수 있음. Mamba는 selective state update를 이용해 boundary에서 이전 정보를 약화하는 방향으로 동작할 수 있음.

## 13. 논문 실험 결과 정리

### Synthetic Tasks

Selective Copying과 Induction Heads task에서 input-dependent selection의 효과를 확인함.

중요한 token 위치가 random하게 변하는 경우 기존 LTI SSM보다 Mamba가 높은 성능을 보임.

### Language Modeling

Pile 기반 language modeling에서 Hyena, RWKV, RetNet, H3++ 등 attention-free 모델보다 좋은 scaling을 보고함.

Transformer++와도 경쟁력 있는 결과를 보임.

### DNA Modeling

Long-context DNA sequence modeling에서도 context가 길어질수록 강한 결과를 보임.

### Audio Modeling

연속 신호인 audio에서도 기존 SSM 및 Transformer 계열과 경쟁력 있는 결과를 보임.

### Efficiency

Transformer autoregressive inference는 KV cache를 유지함.

Mamba는 fixed-size hidden state를 유지하므로 sequence가 길어져도 inference state memory가 크게 증가하지 않음.

논문에서는 비슷한 규모 Transformer 대비 높은 generation throughput을 보고함.

## 14. 핵심 기여

### 14.1 기존 SSM 한계 명확화

빠르고 긴 sequence에 강하지만 content-based selection이 부족한 기존 LTI SSM의 한계를 지적함.

### 14.2 Selective SSM

`B`, `C`, `Δ`를 input-dependent하게 만들어 중요한 정보를 선택적으로 기록하고 읽게 함.

### 14.3 Selective Scan

Convolution을 사용할 수 없게 된 selective recurrence를 GPU에서 효율적으로 병렬화함.

### 14.4 단순한 Architecture

Attention 없이 동일한 Mamba block을 반복해서 쌓는 구조를 제시함.

## 15. 내가 이해한 핵심 직관

Mamba의 출발점은 다음과 같음.

> 긴 sequence를 빠르게 처리하려면 모든 token을 계속 저장하는 방식은 비쌈.  
> Fixed-size state에 압축하려면 무엇을 기억하고 무엇을 버릴지 잘 선택해야 함.

Transformer는 token 자체를 보존하고 attention으로 필요한 정보를 다시 찾음.

기존 SSM은 모든 정보를 고정 규칙으로 state에 압축함.

Mamba는 전체 token을 저장하지 않으면서 현재 input에 따라 압축 규칙을 바꿈.

```text
Transformer
모든 token 유지 → attention으로 필요한 정보 검색

기존 SSM
모든 token → 고정된 update rule로 state에 압축

Mamba
현재 token 내용 확인 → write / hold / read 방식 변경
```

## 16. 헷갈렸던 부분 정리

### `h'(t)`가 어떻게 `h_t`가 되는가

`h'(t)`는 hidden state의 변화율임.

```math
h'(t)=\frac{dh(t)}{dt}
```

이 연속시간 방정식을 한 step 동안 풀고 discretization하면 `h_t` recurrence가 나옴.

직접 같은 값으로 바뀌는 것이 아니라 continuous dynamics를 discrete update로 변환한 결과임.

### SSM은 RNN인가

넓게 보면 recurrent model임.

다만 arbitrary neural recurrence가 아니라 continuous-time state space model에서 유도된 structured recurrence라는 차이가 있음.

### Mamba는 결국 RNN인가

Inference 관점에서는 recurrent하게 동작함.

하지만 일반 RNN과 달리 다음 특징이 있음.

- SSM 기반 structured recurrence 사용함.
- `B`, `C`, `Δ`가 input-dependent함.
- Selective scan으로 training 병렬화함.
- KV cache가 필요하지 않음.

따라서 단순 RNN이라기보다 selective SSM recurrence로 보는 편이 정확함.

### Scan은 언제 가능한가

각 step을 합칠 수 있고 합친 결과가 같은 형태로 유지되며 결합법칙이 성립해야 함.

Mamba step은 affine update라 이 조건을 만족함.

## 17. VLA 관점에서의 의미

VLA에서는 이미지와 언어뿐 아니라 이전 observation, proprioception, action history를 긴 sequence로 처리해야 할 수 있음.

Mamba의 장점은 모든 vision-language token을 대체하는 데 있다기보다 temporal history를 fixed-size state로 효율적으로 누적할 수 있다는 점에 있다고 봄.

따라서 다음 활용 방향이 흥미로움.

- Temporal observation encoder
- Proprioception / IMU history encoder
- Action history memory
- Long-horizon policy state estimator
- Action decoder

다만 actual robot control에서 Mamba state가 유의미한 task state를 안정적으로 유지하는지는 별도 검증이 필요함.

## 18. 한 줄 결론

Mamba는 기존 SSM의 선형 시간 처리 장점은 유지하면서 input에 따라 기억할 정보와 버릴 정보를 선택할 수 있도록 만든 attention-free sequence model임.
