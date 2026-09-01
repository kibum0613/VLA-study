# OpenVLA 논문 및 실습 정리

## 1. 기본 개념

- **Vision-Language-Action(VLA) Model**: 시각 정보, 언어 명령, 행동을 연결해 로봇을 제어하는 모델임.
- **Foundation Model**: 대규모 데이터로 사전학습한 뒤 여러 downstream task에 활용할 수 있는 모델임.
- **Parameter-Efficient Fine-Tuning(PEFT)**: 전체 parameter를 업데이트하지 않고 일부 parameter만 조정해 새 task에 적응시키는 방식임.
- **LoRA**: low-rank matrix를 추가해 적은 parameter만 학습하는 대표적인 PEFT 방식임.
- **Quantization**: weight precision을 낮춰 memory와 계산량을 줄이는 기법임.
- **Open X-Embodiment Dataset**: 여러 robot embodiment에서 수집한 대규모 robot demonstration dataset임.

---

## 2. OpenVLA 아키텍처

OpenVLA는 크게 세 부분으로 구성됨.

### 2.1 Vision Encoder

- DINOv2와 SigLIP feature를 결합함.
- DINOv2는 공간적·기하학적 정보 표현에 강점이 있음.
- SigLIP은 의미적·언어 정렬 정보에 강점이 있음.
- 이미지 입력을 visual patch embedding으로 변환함.

### 2.2 Projector

- Vision encoder의 출력 feature를 LLM이 사용할 수 있는 embedding space로 변환함.
- 시각 feature와 language token을 하나의 sequence에서 처리하기 위한 연결부 역할을 함.

### 2.3 LLM Backbone

- Llama2 7B 기반 Prismatic VLM을 사용함.
- 기존 VLM의 출력 공간을 robot action prediction에 맞게 fine-tuning함.
- 최종적으로 action을 language token과 유사한 방식으로 생성함.

---

## 3. Action 표현과 학습

- Robot action은 `[x, y, z, roll, pitch, yaw, gripper]` 형태의 7D vector임.
- 각 action dimension을 256개 bin으로 이산화함.
- Llama2 tokenizer에서 사용 빈도가 낮은 256개 token을 action token으로 재사용함.
- 학습은 next-token prediction 방식으로 수행함.
- Loss는 일반 language modeling과 동일하게 cross-entropy를 사용함.

즉 continuous control 문제를 discrete action token prediction 문제로 바꾼 구조임.

## 4. 학습 데이터

- Open X-Embodiment 기반 약 97만 건의 robot demonstration을 사용함.
- 여러 robot, scene, task를 포함하도록 dataset mixture를 구성함.
- 주로 단일 팔 end-effector control과 3인칭 camera를 사용하는 manipulation data를 활용함.
- Octo의 data mixture weighting 방식을 참고해 dataset 간 비중을 조정함.

## 5. 주요 설계 결정

- Prismatic VLM이 LLaVA 및 IDEFICS-1보다 multi-object / language understanding에서 좋은 결과를 보여 backbone으로 선택됨.
- 224×224 image resolution을 사용해 높은 해상도 대비 학습 비용을 줄임.
- Vision encoder까지 fine-tuning하는 것이 VLA 성능에 중요함을 확인함.
- 약 27 epoch 학습으로 높은 action token accuracy를 확보함.
- Learning rate는 `2e-5`를 사용함.

## 6. 인프라 및 효율화

- 원 논문 학습에는 64개의 A100 GPU를 사용함.
- Full fine-tuning은 큰 연산 비용이 필요함.
- LoRA를 사용하면 약 1.4% 수준의 parameter만 학습하면서 full fine-tuning과 유사한 성능을 보고함.
- 4-bit quantization으로 GPU memory 사용량을 줄이면서 성능 저하를 작게 유지함.

---

# 실습 기록

## 7. OpenVLA inference 테스트

실습 환경은 Windows + WSL2 Ubuntu 22.04, VS Code Remote WSL을 사용함.

공개 pretrained checkpoint를 이용해 단일 이미지 inference부터 확인함.

### 7.1 단일 action 출력

프롬프트 형태는 다음과 같음.

```text
What action should the robot take to {instruction}?
```

예시 출력은 다음과 같았음.

```text
[-0.00334023  0.00475603  0.00703386
 -0.01227217  0.00390846 -0.01940829
  0.99607843]
```

Action dimension은 다음과 같이 해석함.

| 값 | 의미 |
|---|---|
| Δx | end-effector x 방향 이동 |
| Δy | end-effector y 방향 이동 |
| Δz | end-effector z 방향 이동 |
| Δroll | roll 회전 변화 |
| Δpitch | pitch 회전 변화 |
| Δyaw | yaw 회전 변화 |
| gripper | gripper open/close 제어 |

여러 instruction을 입력했을 때 서로 다른 action vector가 출력되는 것을 확인함.

## 8. LIBERO observation 연결

LIBERO tutorial 환경에서 simulation observation을 가져와 OpenVLA 입력으로 사용함.

예시 instruction은 다음과 같음.

```text
pick up the black bowl between the plate and the ramekin and place it on the plate
```

예시 출력은 다음과 같았음.

```text
Action:
[ 0.00404102  0.00151844  0.01145089
 -0.00141607 -0.02288613 -0.00812570
  0.99607843]
```

LIBERO image가 OpenVLA를 통과해 7D action으로 변환되는 전체 경로가 정상 동작함을 확인함.

## 9. Closed-loop 연결

단순 inference에서 끝내지 않고 다음 구조를 구현함.

```text
LIBERO observation
        ↓
     OpenVLA
        ↓
   7D action
        ↓
LIBERO environment step
        ↓
 new observation
        ↓
      반복
```

Pretrained checkpoint를 LIBERO 환경과 연결해 simulation observation → action prediction → environment step이 반복되는 closed-loop 실행까지 성공함.

## 10. 실습에서 관찰한 문제

Fine-tuning 없이 pretrained checkpoint를 바로 사용했을 때 물체를 정확히 grasp하기보다 제자리에서 gripper가 반복적으로 동작하는 현상이 나타남.

가능한 원인은 다음과 같이 정리함.

- Action scale 불일치 가능성 있음.
- Normalization key가 task/robot과 맞지 않았을 가능성 있음.
- Gripper control direction 차이가 있을 수 있음.
- LIBERO task에 대한 domain adaptation이 부족함.
- Pretrained policy가 해당 embodiment와 observation distribution에 완전히 맞지 않음.

즉 모델 자체가 action을 출력하는 것과 실제 environment에서 task가 성공하는 것은 별개의 문제임.

## 11. OpenVLA 논문 결과 정리

### 11.1 다중 robot platform 평가

- WidowX 및 Google robot에서 평가함.
- Visual, motion, physical, semantic generalization과 language conditioning을 확인함.
- RT-1-X, RT-2-X, Octo 등과 비교함.
- 7B 규모 OpenVLA가 일부 설정에서 더 큰 RT-2-X보다 높은 task success를 기록함.

### 11.2 새로운 robot setting adaptation

- Franka-Tabletop과 Franka-DROID에서 fine-tuning을 평가함.
- Diffusion Policy, Octo 등과 비교함.
- Multi-instruction task에서 높은 평균 성능을 보임.
- 단일 instruction task에서는 Diffusion Policy도 경쟁력 있는 결과를 보임.

### 11.3 LoRA

- 전체 parameter 중 약 1.4%만 학습해 full fine-tuning과 유사한 결과를 보고함.
- Full fine-tuning 대비 compute requirement를 크게 줄일 수 있음.

### 11.4 Quantization

- 4-bit quantization으로 GPU memory 사용량을 크게 줄임.
- Consumer GPU에서 실행 가능한 형태를 제시함.

## 12. 한계

- 기본 구조는 단일 이미지 observation 중심임.
- 여러 frame의 temporal history나 explicit memory를 직접 사용하지 않음.
- 고주파 제어를 위해서는 inference throughput 개선이 필요함.
- 실제 robot에서 100%에 가까운 신뢰성을 보장하는 수준은 아님.
- 새로운 embodiment나 task에서는 별도의 adaptation이 필요함.

## 13. 실습을 통해 확인한 점

- Pretrained VLA를 단일 이미지 inference 수준에서 실행할 수 있었음.
- LIBERO observation을 OpenVLA 입력으로 연결할 수 있었음.
- 7D action vector의 의미와 environment 적용 구조를 확인함.
- Closed-loop simulation 연결까지 구현함.
- 다만 pretrained checkpoint를 그대로 사용하는 것만으로는 task success가 보장되지 않았음.
- Action normalization, embodiment alignment, fine-tuning이 실제 성공률에 중요한 요소임을 확인함.

## 14. 다음 단계

- LIBERO 전용 checkpoint 또는 LoRA fine-tuning 적용 필요함.
- Action normalization / unnormalization 설정 검증 필요함.
- Gripper direction과 controller convention 확인 필요함.
- 단순 action 출력이 아니라 rollout success를 기준으로 평가해야 함.
- 이후 OpenVLA-OFT와 비교하면서 continuous action / action chunking 구조 차이를 확인할 필요가 있음.
