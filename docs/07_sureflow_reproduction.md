# SUREFlow + LIBERO 실습 및 재현 기록

## 1. 실습 목적

SUREFlow를 직접 실행해 LIBERO manipulation policy의 학습부터 checkpoint 평가까지 전체 pipeline을 확인하는 것을 목표로 함.

단순 코드 실행이 아니라 다음 흐름을 직접 구성하고 문제를 해결함.

```text
LIBERO demonstration data
        ↓
SUREFlow Dataset / DataLoader
        ↓
SUREFlow Model Training
        ↓
Checkpoint 저장
        ↓
LIBERO Simulation Evaluation
        ↓
Success Rate / Rollout Video 분석
```

초기에는 WSL2 로컬 환경에서 smoke test를 수행했고 이후 연구실 GPU 서버에서 전체 학습과 평가를 진행함.

---

# Part 1. WSL2 로컬 실습

## 2. 로컬 환경

| 항목 | 설정 |
|---|---|
| OS | Windows + WSL2 Ubuntu 22.04 |
| Conda 환경 | `libero` |
| 프로젝트 경로 | `~/projects/SUREFlow` |
| 데이터셋 | LIBERO Spatial |
| Python | 3.8 |
| PyTorch | 2.4.1 + CUDA 12.1 |
| 실행 | WSL2 terminal / VS Code Remote WSL |

기본 실행 위치는 다음과 같음.

```bash
cd ~/projects/SUREFlow
conda activate libero
```

## 3. SUREFlow 구조

SUREFlow는 이미지, 언어 명령, 로봇 상태를 입력으로 받아 다음 action을 예측하는 robot policy임.

```text
입력
- 카메라 이미지
- 자연어 명령
- 로봇 상태

출력
- robot action vector
```

학습 대상은 simulation environment 자체가 아니라 environment 안에서 어떤 행동을 할지 결정하는 policy임.

전체 구조는 다음과 같음.

```text
LIBERO demonstration data (.hdf5)
→ LIBERO Dataset Loader
→ SUREFlow DataLoader
→ SUREFlow Model
→ Training Loop
→ Checkpoint
→ LIBERO / robosuite Simulation
→ Success Rate / Rollout Video
```

## 4. Mamba 관련 의존성 문제

초기 실행 과정에서 `causal-conv1d`, `mamba-ssm` 설치 문제가 발생함.

CUDA Toolkit과 `nvcc` 인식 문제를 먼저 해결한 뒤 설치함.

```bash
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

설치 확인은 다음과 같이 수행함.

```bash
python - <<'PY'
import causal_conv1d
import mamba_ssm
print("causal_conv1d OK")
print("mamba_ssm OK")
PY
```

추가 누락 package도 설치함.

```bash
pip install ftfy regex tqdm
pip install colorama
```

## 5. Python 3.8 타입 힌트 문제

코드 일부에서 Python 3.10 스타일 타입 힌트를 사용하고 있었음.

```python
str | None
list[str]
tuple[str, ...]
```

Python 3.8 환경에서 오류가 발생해 source file 상단에 아래 구문을 추가함.

```python
from __future__ import annotations
```

실습 당시 여러 `.py` 파일에 일괄 적용하여 실행을 진행함.

## 6. Dataset 로딩 확인

`libero_spatial` dataset이 정상적으로 로딩되는지 먼저 확인함.

정상 로딩 시 약 57,750개의 training sample이 확인됨.

```text
Training dataset size: target (62250, 7)
Number of training samples: 57750
```

Batch size가 2일 경우 1 epoch당 약 28,875 batch가 생성됨.

## 7. Mini subset smoke test

처음부터 전체 dataset을 학습시키지 않고 100개, 이후 1000개 sample로 smoke test를 수행함.

DataLoader에만 `Subset`을 적용하고 원본 `trainset`은 유지함. `self.trainset.get_all_actions()`가 scaler 구성에 사용되기 때문임.

```python
from torch.utils.data import Subset

max_train_samples = 1000
trainset_for_loader = self.trainset

if len(self.trainset) > max_train_samples:
    trainset_for_loader = Subset(self.trainset, range(max_train_samples))

self.train_dataloader = DataLoader(
    trainset_for_loader,
    batch_size=self.train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    prefetch_factor=None,
)
```

1000개 sample, 3 epoch 결과는 다음과 같았음.

```text
Epoch 0: Mean train loss = 0.8127
Epoch 1: Mean train loss = 0.4664
Epoch 2: Mean train loss = 0.3665
```

Loss는 정상 감소했지만 evaluation success rate는 0.000이었음.

Mini subset은 설치와 학습 loop 확인에는 적합하지만 task success를 기대하기에는 데이터가 부족하다고 판단함.

## 8. Evaluation 축소

기본 evaluation은 task와 episode 수가 많아 WSL2 RAM 부족으로 `Killed`가 발생함.

초기 확인용으로 아래와 같이 축소함.

```python
rollouts = 1
```

영상 확인 시에는 3 episode 정도로 늘려 사용함.

```text
Task Suite: 1 tasks
Episodes per Task: 3
Total Evaluations: 3
Multiprocessing: Disabled
Real-time Rendering: Disabled
```

## 9. Rollout 영상 저장

```python
save_video = True
render_image = False
use_multiprocessing = False
```

학습 성공 여부뿐 아니라 grasp, 접근, placement 실패가 어느 시점에서 발생하는지 영상으로 확인함.

## 10. 전체 dataset 학습으로 전환

Mini subset으로 pipeline을 확인한 뒤 전체 trainset을 사용함.

```python
self.train_dataloader = DataLoader(
    self.trainset,
    batch_size=self.train_batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=False,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

Memory가 불안정한 경우 `num_workers=0`, `persistent_workers=False`로 낮춰 사용함.

## 11. 로컬 실험 결과

### Mini subset 100 samples

```text
Epochs: 1
Mean train loss: 약 1.2883
Evaluation success rate: 0.000
```

Smoke test 용도로는 정상 동작함.

### Mini subset 1000 samples, 3 epochs

```text
Epoch 0 loss: 0.8127
Epoch 1 loss: 0.4664
Epoch 2 loss: 0.3665
Evaluation success rate: 0.000
```

Loss 감소가 곧 closed-loop task success를 의미하지 않음을 확인함.

### 전체 dataset 1 epoch

```text
Training samples: 57,750
Batch size: 2
Batches per epoch: 28,875
Checkpoint 저장 성공
Evaluation에서 1회 성공 확인
```

전체 dataset 학습이 실제 rollout success로 연결될 수 있음을 확인함.

---

# Part 2. 연구실 서버 재현

## 12. 서버 환경

로컬 WSL2 환경의 memory와 장시간 학습 제약 때문에 연구실 GPU 서버로 이동함.

| 항목 | 내용 |
|---|---|
| 사용자 계정 | `kbkim` |
| 프로젝트 | `/home/kbkim/projects/SUREFlow` |
| Conda 환경 | `sureflow` |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU Memory | 약 24GB |
| System Memory | 약 256GB |
| CUDA Toolkit | `/usr/local/cuda-12.4` |
| PyTorch | `2.4.1+cu121` |

실습 당시 GPU 0에 다른 process가 있어 주로 GPU 2를 사용함.

```bash
CUDA_VISIBLE_DEVICES=2
```

## 13. Conda 및 PyTorch 확인

```bash
conda create -n sureflow python=3.10
conda activate sureflow
```

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

확인 결과는 다음과 같았음.

```text
torch: 2.4.1+cu121
cuda: True
gpu: NVIDIA GeForce RTX 3090 Ti
```

## 14. SUREFlow 코드 및 주요 package

SUREFlow repository를 clone하고 필요한 package를 설치함.

```bash
pip install ftfy regex tqdm colorama h5py einops wandb omegaconf huggingface_hub
pip install matplotlib pandas scipy scikit-learn opencv-python
pip install causal-conv1d
pip install mamba-ssm
```

LIBERO / robosuite 계열 package는 version conflict를 피하기 위해 필요한 version만 선택해 설치함.

```bash
pip install hydra-core==1.2.0 easydict==1.9 robomimic==0.2.0 thop==0.1.1-2209072238
pip install robosuite==1.4.0 bddl==1.0.1 future==0.18.2 cloudpickle==2.1.0 gym==0.25.2
```

## 15. robosuite `/tmp` 로그 권한 문제

공용 서버에서 다음 오류가 발생함.

```text
PermissionError: [Errno 13] Permission denied: '/tmp/robosuite.log'
```

다른 사용자 소유의 `/tmp/robosuite.log` 때문에 발생한 문제였음.

System file permission을 변경하지 않고 개인 경로를 사용하도록 `robosuite/utils/log_utils.py`의 log path를 `/home/kbkim/tmp/robosuite.log`로 변경함.

## 16. LIBERO config 문제

`~/.libero/config.yaml`이 없으면 import 과정에서 dataset path prompt가 발생하고 비대화형 실행에서는 `EOFError`가 발생함.

개인 config를 생성해 해결함.

```yaml
benchmark_root: /home/kbkim/projects/LIBERO/libero/libero
bddl_files: /home/kbkim/projects/LIBERO/libero/libero/bddl_files
init_states: /home/kbkim/projects/LIBERO/libero/libero/init_files
datasets: /home/kbkim/datasets/robot
assets: /home/kbkim/projects/LIBERO/libero/libero/assets
```

## 17. PYTHONPATH 충돌

평가 중 다음 오류가 반복됨.

```text
ModuleNotFoundError: No module named 'libero.libero'
```

외부 LIBERO 경로와 SUREFlow 내부 `LIBERO-PRO`가 섞이면서 package가 잘못 잡힌 것이 원인이었음.

평가 전 `PYTHONPATH`를 초기화하고 SUREFlow 내부 경로만 사용함.

```bash
unset PYTHONPATH
export PYTHONPATH=/home/kbkim/projects/SUREFlow/LIBERO-PRO:/home/kbkim/projects/SUREFlow
```

이후 `libero.libero`와 benchmark module import가 정상 동작함.

## 18. MuJoCo version 문제

평가 중 다음 오류가 발생함.

```text
TypeError: mj_fullM(): incompatible function arguments
```

`robosuite==1.4.0`과 설치된 MuJoCo version의 API 호환 문제로 판단함.

```bash
pip uninstall -y mujoco
pip install mujoco==2.3.7
```

Offscreen rendering은 다음 환경변수를 사용함.

```bash
export MUJOCO_GL=egl
```

## 19. tmux 사용

장시간 학습 중 SSH disconnect에 대비해 tmux를 사용함.

```bash
tmux new -s sureflow100
```

Detach는 `Ctrl+B`, `D`를 사용함.

```bash
tmux attach -t sureflow100
tmux ls
```

## 20. Dataset

LIBERO Spatial dataset은 다음 경로에 저장함.

```text
/home/kbkim/datasets/robot/libero_spatial
```

Dataset download는 Hugging Face CLI를 사용함.

```bash
huggingface-cli download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --include "libero_spatial/*" \
  --local-dir /home/kbkim/datasets/robot
```

## 21. 학습 및 평가 설정

재현 과정에서 확인한 주요 설정은 다음과 같음.

| 항목 | 설정 |
|---|---:|
| Epoch | 최근 실험 200 |
| Batch size | 256 |
| Inference sampling steps | 50 |
| Rollouts per task | 최종 논문 조건 50 |
| Max step per episode | 350 |
| Demos per task | 70 |

실험 folder name과 실제 epoch 설정이 일치하지 않는 경우가 있었음. 이후 재현성을 위해 config snapshot과 checkpoint metadata를 함께 기록할 필요가 있다고 판단함.

## 22. 서버 학습 명령 형태

```bash
cd /home/kbkim/projects/SUREFlow
conda activate sureflow

unset PYTHONPATH
export PYTHONPATH=/home/kbkim/projects/SUREFlow/LIBERO-PRO:/home/kbkim/projects/SUREFlow

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python run.py --train_suite libero_spatial
```

## 23. Checkpoint 평가 형태

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled python run.py \
  --train_suite libero_spatial \
  --checkpoint_path /path/to/final_model.pth
```

Checkpoint를 자동으로 `tail -1` 등으로 선택하면 의도하지 않은 model을 평가할 수 있어 이후에는 정확한 path를 직접 지정함.

## 24. 최종 평가 결과

가장 최근 epoch 200 학습 checkpoint를 LIBERO Spatial에서 평가한 결과는 다음과 같았음.

```text
Overall Average Success Rate: 0.302

Task 0: 0.900
Task 1: 0.020
Task 2: 0.720
Task 3: 0.660
Task 4: 0.080
Task 5: 0.580
Task 6: 0.060
Task 7: 0.100
Task 8: 0.500
Task 9: 0.000
```

전체 평균 성공률은 **30.2%**였음.

Task별 편차가 매우 크게 나타남. Task 0, 2, 3은 비교적 높았지만 Task 1, 4, 6, 7, 9는 낮았음. Task 9는 0%였음.

## 25. 논문 결과와 차이

논문에서는 LIBERO Spatial 성공률을 약 94%로 보고함. 본 재현 결과는 30.2%로 큰 차이가 있었음.

가능한 원인은 다음과 같이 정리함.

1. 학습 / 평가 protocol이 논문과 완전히 동일하지 않았을 가능성 있음.
2. Folder name과 실제 epoch 설정이 달라 config 관리가 불명확했던 부분이 있음.
3. Checkpoint 선택 과정에서 다른 checkpoint를 선택했을 가능성이 있었음.
4. 외부 LIBERO와 내부 LIBERO-PRO package path가 섞이는 문제가 있었음.
5. MuJoCo / robosuite version이 논문 환경과 완전히 같지 않음.
6. 중간 평가에서는 rollout 수가 논문보다 적었음.
7. Long rollout에서 작은 action error가 누적됐을 가능성이 있음.

따라서 현재 결과는 **논문 성능 완전 재현 성공**이 아니라 **논문 코드 기반 서버 학습·평가 pipeline 구축과 troubleshooting까지 완료한 재현 실습**으로 기록하는 것이 적절함.

## 26. 실습에서 얻은 점

- WSL2와 공용 GPU 서버 모두에서 SUREFlow 실행 환경을 구성함.
- CUDA, Mamba, robosuite, MuJoCo, LIBERO version 및 path 문제를 직접 해결함.
- Mini subset → full dataset 순서로 학습 pipeline을 검증함.
- Checkpoint 저장과 simulation evaluation을 분리해 실행함.
- Rollout video와 task별 success rate를 이용해 실제 policy 성능을 확인함.
- Training loss 감소와 closed-loop task success가 동일하지 않음을 확인함.
- 논문 재현에서는 model code뿐 아니라 dependency, environment, controller, evaluation protocol까지 함께 고정해야 함을 확인함.

## 27. 이후 재현 시 개선할 점

- 새 experiment folder에서 config를 처음부터 고정함.
- 실행 시 config snapshot과 git commit hash를 함께 저장함.
- Checkpoint path를 직접 지정함.
- Epoch별 training/validation loss를 남김.
- Task별 failure video를 grasp failure / approach failure / placement failure / drift 등으로 분류함.
- 논문과 동일한 rollout 수와 max step으로 full evaluation을 수행함.

## 28. 한 줄 정리

SUREFlow를 WSL2와 GPU 서버에서 직접 학습하고 LIBERO simulation evaluation까지 연결함. 완전한 논문 성능 재현에는 실패했지만 실제 연구 코드 재현 과정에서 발생하는 dependency, path, simulator version, checkpoint, evaluation protocol 문제를 직접 해결하며 전체 pipeline을 구축함.
