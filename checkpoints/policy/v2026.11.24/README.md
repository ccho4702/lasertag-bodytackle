# v2026.11.24 — 배포용 체크포인트 (baseline / ours)

2026-11-24 공개. 과거 구간 × 프레임 수 격자(13칸, 3-fold 교차검증)에서 고른 설정 **T8 K4**(8프레임, 간격 4, 약 3초)를
**전체 데이터**(chaser 90 세션 47.7분, evader 75 세션 37.7분)로 학습한 배포용 체크포인트다.
ours 는 baseline 과 같은 설정에 SmolVLA plan 입력만 추가한 것이다.

## 파일

| | chaser (Misty) | evader (Cozmo) |
|---|---|---|
| baseline | `baseline_chaser_T8K4.pt` | `baseline_evader_T8K4.pt` |
| ours (+SmolVLA) | `ours_chaser_T8K4.pt` | 없음 |

각 약 1.1 MB.

- 입력: ego 카메라 → YOLO 히트맵 + depth(Depth-Anything-V2-Small, 252) → 정책. CPU / MacBook 에서 도는 구성이다.
- 출력: 축별 3-클래스(전진·정지·후진 / 좌·정지·우) → `[v, ω]`.
- `ours_chaser_T8K4.pt` 는 SmolVLA 릴리스 **v2026.11.24** 와 짝이다 (`bash scripts/download_smolvla.sh`). v0.2 SmolVLA 와 섞어 쓰지 않는다.
- 프레임 수·간격·depth 모델·VLA 주기는 체크포인트 `cfg` 에 들어 있어 `policy_server.py` 가 자동으로 맞춘다.
- 학습 설정과 지표 전체: `selection.json`. 재현: `baseline/final_top3.py`.

고른 방식:
- 설정은 하나만 올렸다. 격자 상위 3개(T8K4, T4K5, T16K1)는 baseline 성능이 서로 오차 범위 안이고, ours 이득이 여러 seed 에서
  반복 확인된 것은 T8K4 뿐이다.
- chaser 두 개는 같은 설정·같은 데이터로 두 번(11-14, 11-18) 학습한 것 중 검증 AE 가 낮은 11-14 학습분이다
  (baseline 0.375 대 0.401, ours 0.382 대 0.412). 선택 기준과 후보 점수는 `selection.json` 의 `candidates`.
- evader ours 는 올리지 않았다. 교차검증에서 baseline 과 차이가 없었고, 이 설정의 전체 데이터 학습을 끝내지 않았다.
  SmolVLA evader 는 릴리스에 들어 있어 나중에 학습하면 바로 쓸 수 있다.

## 실제 환경에서 볼 것

| 로봇 | 돌릴 것 | 볼 이유 |
|---|---|---|
| chaser | `baseline_chaser_T8K4.pt` | 기준 |
| chaser | `ours_chaser_T8K4.pt` + SmolVLA | 오프라인에서 유일하게 반복 확인된 이득(AE 0.533 → 0.479)이 실제 추격에서도 보이는지 |
| evader | `baseline_evader_T8K4.pt` | 기준 |
| evader | `../baseline_cpu_evader.pt` (2026-11-08, 옛 데이터 46 세션, 간격 8) | 11-12 수집분을 넣은 뒤 회전 방향 정확도가 떨어졌다. 옛 모델이 실제로 더 잘 도는지 |

같은 시작 위치·같은 장애물 배치에서 번갈아 돌리고, overhead 첫 히트 시간으로 비교한다 (HANDOFF 1-3).

아직 만들지 않은 후보: 정책에 자기 이전 명령을 넣는 변형 (`python final_top3.py` 의 `prevact_*`). 오프라인 수치는 매우 높지만
(chaser AE 0.190, evader 0.244) 정답 명령을 넣어 준 값이라, 실제로 좋아지는지 같은 명령만 반복하는지는 rollout 으로만 알 수 있다.

## 실행

MacBook 설치와 로봇 PC 쪽 절차는 [QUICKSTART_MAC.md](../../../QUICKSTART_MAC.md) 그대로이고, `--ckpt` 경로만 바꾼다.

```bash
git pull                                   # 체크포인트는 git 에 들어 있다
cd baseline

# baseline — 터미널 1: chaser(Misty), 터미널 2: evader(Cozmo)
python policy_server.py --ckpt ../checkpoints/policy/v2026.11.24/baseline_chaser_T8K4.pt --port 8770 --device mps
OMP_NUM_THREADS=4 python policy_server.py --ckpt ../checkpoints/policy/v2026.11.24/baseline_evader_T8K4.pt --port 8771 --device cpu

# ours (chaser) — SmolVLA 를 1회 받은 뒤 (릴리스 v2026.11.24, 로봇당 약 900 MB → checkpoints/smolvla/)
bash ../scripts/download_smolvla.sh
python policy_server.py --ckpt ../checkpoints/policy/v2026.11.24/ours_chaser_T8K4.pt --vla ../checkpoints/smolvla/chaser --port 8770 --device mps
```

로봇 없이 녹화 세션으로 점검 (프레임당 지연과 예측 출력):

```bash
python deploy.py --ckpt ../checkpoints/policy/v2026.11.24/baseline_chaser_T8K4.pt --session <chaser 세션 폴더> --n 50
```

2026-11-24 서버(Xeon, CPU 8 스레드)에서 세 체크포인트 모두 이 점검을 통과했다: 프레임당 중앙값 chaser 97~100 ms (budget 102 ms), evader 80 ms (budget 119 ms).
릴리스에 올린 SmolVLA chaser 묶음도 풀어서 `python -m vla.runner` 로 plan 이 나오는 것을 확인했다.

확인할 것:
- 서버 시작 로그의 `inference ≈ NN ms/frame` 이 budget(약 100 ms) 이하인지 본다. 넘으면 `--device` 를 mps ↔ cpu 로 바꾼다.
- ours 는 시작 로그의 SmolVLA 지연을 본다. 학습은 지연 chaser 4 / evader 3 프레임(약 340 ms)을 가정했다. Mac 에서 1회가 수 초면 plan 이 낡아 효과가 줄어든다. 이 경우 SmolVLA 는 GPU 가 있는 PC 에서 돌린다.
- `ours_chaser_T8K4.pt` 를 `--vla` 없이 띄우면 plan 없이 동작한다 (경고 출력).
- 정책을 켠 직후에는 과거 창을 첫 프레임으로 채우므로, 처음 약 3초는 정지 영상을 본 것처럼 반응할 수 있다.

## 오프라인 성능 (3-fold 교차 세션 검증, 학습에 안 쓴 세션)

같은 설정을 세션 2/3 로 학습해 나머지 1/3 로 평가한 값이다. 배포 체크포인트 자체는 전체 데이터로 학습했으므로 이 값은 추정치다.
T8K4 는 3 seed × 3 fold = 9회 평균, 참고로 적은 나머지 두 설정은 3회 평균이다 (체크포인트는 T8K4 만 있음).

| 로봇 | 설정 | 모델 | AE (낮을수록 좋음) | joint acc | balanced acc | 회전 방향 |
|---|---|---|---|---|---|---|
| chaser | T8K4 | baseline | 0.533 | 0.454 | 0.257 | 0.616 |
| chaser | T8K4 | ours | **0.479** | **0.490** | **0.277** | **0.668** |
| chaser | T4K5 | baseline | 0.525 | 0.470 | 0.267 | 0.618 |
| chaser | T4K5 | ours | 0.575 | 0.443 | 0.251 | 0.598 |
| chaser | T16K1 | baseline | 0.540 | 0.471 | 0.258 | 0.623 |
| chaser | T16K1 | ours | 0.518 | 0.472 | 0.256 | 0.643 |
| evader | T8K4 | baseline | 0.459 | 0.470 | 0.202 | 0.322 |
| evader | T8K4 | ours | 0.472 | 0.479 | 0.208 | 0.334 |
| evader | T4K5 | baseline | 0.452 | 0.487 | 0.193 | 0.286 |
| evader | T4K5 | ours | 0.454 | 0.500 | 0.211 | 0.303 |
| evader | T16K1 | baseline | 0.452 | 0.489 | 0.198 | 0.344 |
| evader | T16K1 | ours | 0.426 | 0.510 | 0.198 | 0.299 |

기준선: chaser 는 항상 정지 AE 0.451, 최빈 클래스 acc 0.307. evader 는 0.550, 0.509.

- AE: 정규화한 `[v, ω]` 의 평균 절대 오차. joint acc: 9개 조합 클래스 정확도. balanced acc: 클래스별 재현율 평균. 회전 방향: 사람이 회전한 프레임에서 좌/우를 맞힌 비율.
- 세 설정의 baseline 은 서로 오차 범위 안이다.
- chaser T8K4 의 ours 이득은 9회 중 7~8회에서 나타났지만 크지 않다 (paired t ≈ 2).
- evader 는 joint acc 가 최빈 클래스(0.509~0.533) 아래이고 회전 방향이 0.3 수준이다. 실제로 잘 도망칠지는 낮게 본다. 최근 수집분의 회전 비중이 낮은 것이 원인으로 보인다.
- 이 값들은 사람 조종을 얼마나 따라 하는지이고, 태그 성공률이 아니다. 실제 성능은 로봇 rollout 으로만 확인된다.

## 학습 설정

| | chaser | evader |
|---|---|---|
| 세션 | 90 (전체) | 75 (전체) |
| epoch / 체크포인트 선택 | 30 / val AE | 20 / val balanced acc |
| 공통 | improved preset, batch 32, `obs_dsmall252.npz`, val 10 %, 세션별 마지막 5 % 는 확인용 | 〃 |
| ours 추가 입력 | SmolVLA plan (10프레임마다 갱신, 지연 4프레임, 20 % dropout) | 〃 (지연 3프레임) |

ours 학습에 쓴 plan 은 out-of-fold(`vla_plans_oof_v3.npz`)이고, 배포용 SmolVLA 는 같은 설정으로 전체 세션에 학습한 것이다
(현재 프레임 1장 + 고정 지시문, action expert 만 1000 step 학습).

체크포인트 선택이 검증 AE 최소라서 이른 epoch 가 뽑힐 수 있다 (chaser baseline 22/30, ours 4/30, evader baseline 8/20 epoch).
