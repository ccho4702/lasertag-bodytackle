# Pipeline

Misty 로봇 레이저태그 데모 데이터로 VisuomotorPolicy를 학습하는 파이프라인입니다.

```
baseline/
├── preprocess.py   이미지 → YOLO 히트맵(enemy/ally/obstacle) + DATv2 depth, 224×224 (obs.npz)
├── dataset.py      슬라이딩 윈도우 시퀀스 데이터셋
├── model.py        VisuomotorPolicy (encoder: cnn|flat|spatial, LSTM, head: tanh|linear|cls)
├── train.py        학습 루프 (train / val / test 분할, preset, ablation 플래그)
├── augment.py      GPU 배치 증강 (hflip, jitter, detection dropout)
├── metrics.py      정확도/MSE/zero·mean·majority 기준선
├── eval.py         평가 및 예측 시각화
├── outputs/        eval 결과 그래프 (PNG)
└── runs/           학습 결과 저장 폴더
    ├── best.pt              모델 체크포인트 (gitignore)
    └── train_log.csv        에폭별 loss 로그
```

추가 파일: `robots.py` (로봇별 입출력 프로필), `deploy.py` (카메라 프레임 → 로봇 명령),
`detector/` (Cozmo 시점 YOLO), `cv.py` (세션 단위 CV), `vla/` (SmolVLA 주기 호출).

---

## 논문(arXiv 2603.11980) 정렬 + 로봇 입출력 (2026-11 데이터)

| | 논문 student | 이 레포 (`--preset paper`) |
|---|---|---|
| 입력 | YOLO enemy/ally Gaussian 히트맵 (커널 크기 s = λ/D, D = 실제 폭·초점거리/박스 폭) + DATv2-Large depth (5–95% clip, [0,1]), 224×224 | 동일 (`preprocess.py`). obstacle 히트맵도 저장하지만 paper preset 은 미사용 |
| 인코더 | ResNet18, 임베딩 1024 | 동일 (ImageNet init, 1채널 conv1 은 RGB 평균 가중치) |
| 시퀀스 | LSTM 1층 128, 길이 5 | 동일 |
| 출력 | 2층 MLP(128) → 속도 명령, L2 | 동일, [v, ω] 를 축별 act_scale 로 정규화해 L2 |
| 학습 | DAgger (MADDPG teacher), lr 1e-6 / 1e-5, 100 ep | teacher 없음 → 키보드 시연 BC. lr·epoch 동일 |
| 평가 | AE, Hit@15/30, CR (rollout) | AE(정규화 MSE) + 명령 정확도 (오프라인; Hit/CR 은 실기 rollout 필요) |

로봇 프로필 (`robots.py`) — 정책은 항상 몸체 속도 [v, ω] (ω>0 = 좌회전) 를 출력하고
`Robot.to_command()` 가 로봇 API 입력으로 변환합니다 (`deploy.Agent.step(frame)` 이 이 경로 전체를 수행).

| robot | 데이터 | 라벨 → [v, ω] | 시연 명령 값 | 로봇 API 입력 |
|---|---|---|---|---|
| `chaser` (Misty, 공격) | 480×640 세로, 9.8 fps | `requested_command` (운영자가 낸 명령; `--label_source executed` 로 ack 된 명령) | v {-25,0,25} %, ω {-25,0,+30} % | `/api/drive {LinearVelocity, AngularVelocity}` |
| `evader` (Cozmo, 회피) | 320×240, 8.4 fps | v = (L+R)/2, ω = (R−L)/2 | v {-90,0,90}, ω {-60,0,60} mm/s | `drive_wheels(l=v−ω, r=v+ω)` mm/s |

- 이미지는 기록된 그대로 사용 (예전 180° flip / 상단 90행 crop 제거 — 새 데이터에선 오검출 원인).
- YOLO: chaser `misty_yolo26s_v1.pt`, evader `runs/detect/runs_det/cozmo_v1/weights/best.pt` (`detector/README.md`).
- 각 로봇은 자기 카메라만 사용 (상단 카메라/ArUco 위치는 입력에 쓰지 않음).
- 가정값 확인 필요: `robots.py` 의 `hfov_deg` (Misty 55° 가정), 로봇 실제 폭 `width_m` → 커널 크기에만 영향.
- `--head cls` 의 bin 은 로봇 시연 값 그대로 (ω {-25,0,30}), 좌우반전 증강은 좌↔우 명령으로 매핑.

```bash
ROBOTS="chaser evader" ./run_paper_cv.sh     # 전처리 → OOF VLA plan → 5-fold CV (paper / +VLA)
```

---

## 환경 설치

```bash
# 모든 단계 (YOLO, DATv2, SmolVLA) 는 conda env `lsma` 하나로 실행
python ...   # 새 서버: ../SETUP.md 참고
```

---

## Step 1 — 전처리 (obs.npz 생성)

세션마다 한 번 (이미 있으면 건너뜀, `--force` 로 재실행). labels.csv 행과 1:1 정렬을 유지합니다.

```bash
cd baseline
python preprocess.py --robot chaser --session "../data_download/chaser/*/*/*/"
python preprocess.py --robot evader --session "../data_download/evader/*/*/"
```

옵션:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--robot` | (필수) | `chaser` / `evader` (YOLO, 라벨 변환, 초점거리) |
| `--session` | (필수) | 세션 디렉터리 (glob 가능) |
| `--label_source` | requested | chaser 라벨: 운영자 명령(requested) / ack 된 명령(executed) |
| `--lam` | 40 | 커널 크기 λ/D 의 λ (px·m) |
| `--no_depth` | False | DATv2 depth 채널 생략 |
| `--conf` / `--iou` | 0.25 / 0.45 | YOLO threshold |
| `--force` | False | 기존 obs.npz 덮어쓰기 |

---

## Step 2 — 학습

### 기본 방식 (temporal split)

각 세션의 **앞 80%를 train+val**, **뒤 20%를 test**로 사용합니다.  
Val은 train+val 구간의 **마지막 15% 시간 블록**입니다 (이전의 윈도우 랜덤 샘플링은 인접 윈도우가
7/8 프레임을 공유해 val이 train과 사실상 동일했음 → val MSE 15 vs test MSE 145의 원인).

```bash
python train.py \
  --sessions \
    ../data/sample/20261028T003936_723807Z \
    ../data/sample/20261028T004158_644470Z \
    ../data/sample/20261028T004644_157053Z \
  --run_dir runs/sample
```

### 세션 간 분할 (cross-session split)

특정 세션을 통째로 test로 분리하고, 나머지로 학습합니다.

```bash
python train.py \
  --sessions \
    ../data/sample/20261028T003936_723807Z \
    ../data/sample/20261028T004158_644470Z \
  --test_sessions \
    ../data/sample/20261028T004644_157053Z \
  --run_dir runs/sample_cross
```

### Preset / ablation

```bash
python train.py --sessions S1 S2 S3 --run_dir runs/base                       # baseline 모델
python train.py --sessions S1 S2 S3 --preset improved --run_dir runs/improved # 개선 모델
python train.py --sessions S1 S2 S3 --preset improved --encoder cnn --run_dir runs/abl_cnn  # 하나만 끄기
```

`--preset improved` = `--encoder spatial --coordconv --head cls --n_bins 3 --lstm_layers 1 --all_steps
--w_ang 2 --lr 1e-3 --aug_jitter 1.5 --aug_det_drop 0.1 --ema 0.99`. 명시한 플래그가 preset보다 우선합니다.

| 그룹 | 인자 | 기본값 | 설명 |
|---|---|---|---|
| 모델 | `--encoder` | `cnn` | `cnn`(원본, avgpool 4×4) / `flat`(풀링 없이 8×8 flatten) / `spatial`(spatial-softmax 키포인트 + presence) |
| | `--coordconv` | off | 입력에 x,y 좌표 채널 추가 |
| | `--head` | `tanh` | `tanh` / `linear`(clamp) / `cls`(bin 분류, `--n_bins`) |
| | `--n_bins` | 21 | Misty 키보드 데이터는 라벨이 {-15,0,15} → **3** |
| | `--lstm_layers` | 2 | |
| | `--prev_action` | off | 이전 행동 입력 (`--prev_action_dropout` 0.5). 오프라인 평가는 teacher forcing이라 낙관적 |
| | `--robot` | chaser | 로봇 프로필 (robots.py): 라벨 단위, 축별 act_scale·명령 값 |
| | `--channels` | enemy,ally,depth | obs.npz 채널 선택 (논문 입력) |
| 손실 | `--all_steps` | off | T개 스텝 전부 supervise |
| | `--loss` | `mse` | `mse` / `huber`(`--huber_delta`) — 회귀 헤드용 |
| | `--w_ang` | 1.0 | 각속도 손실 가중치 |
| 증강 | `--aug_hflip` | 0 | 좌우 반전 + ω 부호 반전 (검출 있는 시퀀스만). 시연자 탐색 행동이 비대칭이면 해로움 |
| | `--aug_jitter` | 0 | 프레임별 독립 평행이동(px) — YOLO 박스 노이즈 |
| | `--aug_det_drop` | 0 | 프레임별 채널 drop 확률 — 미검출 모사 |
| 최적화 | `--ema` | 0 | EMA decay (스텝 단위). 짧은 학습에 0.999는 초기값에 끌려가 성능 붕괴 — 0.99 권장 |

### 주요 옵션

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--sessions` | (필수) | train 세션 경로 (여러 개 가능) |
| `--test_sessions` | - | test 세션 경로 (생략 시 temporal split) |
| `--run_dir` | `runs` | 결과 저장 폴더 |
| `--epochs` | 100 | 학습 에폭 수 |
| `--batch_size` | 64 | 배치 크기 |
| `--lr` | 3e-4 | 학습률 |
| `--seq_len` | 8 | 입력 시퀀스 길이 |
| `--val_frac` / `--test_frac` | 0.15 / 0.2 | 시간 블록 split 비율 |
| `--resume` | - | 체크포인트 경로 (이어 학습: `last.pt`. opt/sched/EMA/best_val 복원. `--epochs`는 원래 값 유지) |
| `--eval_only` | False | 학습 없이 평가만 실행 |

### 결과 위치

학습이 끝나면 `--run_dir`에 저장됩니다.

```
runs/sample/
├── best.pt            최적 모델 체크포인트 (val MSE 기준)
├── last.pt            마지막 에폭 (resume용)
├── config.json        사용한 설정
├── test_metrics.json  test 지표
└── train_log.csv      에폭별 train_loss / val_mse / val_acc_joint / lr / elapsed_s
```

> `*.pt` 파일은 `.gitignore`에 포함되어 Git에 올라가지 않습니다.

---

## Step 3 — 평가

```bash
python eval.py \
  --session ../data/sample/20261028T004644_157053Z \
  --ckpt    runs/sample/best.pt \
  --plot_out outputs/eval_sample.png
```

`outputs/eval_sample.png`에 ground truth vs 예측 velocity 그래프가 저장됩니다.

옵션:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--session` | (필수) | 평가할 세션 경로 |
| `--ckpt` | (필수) | 모델 체크포인트 경로 |
| `--split` | `test` | 평가 구간 (`all` / `train` / `val` / `test`). cross-session으로 뺀 세션은 `all` |
| `--no_ema` | off | EMA 가중치 대신 raw 가중치 사용 |
| `--plot_out` | `outputs/eval_plot.png` | 그래프 저장 경로 |

---

## 학습 결과 요약

> ⚠️ 아래는 **이전 코드** 결과입니다. 이 데이터의 라벨은 각 차원 {-15, 0, +15}라 MSE 기준선은
> zero 예측 = **80.9** (temporal test) / **87.3** (cross, S3) 입니다. 즉 아래 test MSE는 trivial 기준선보다
> 나쁘고, val은 랜덤 split 누수로 과소평가돼 있습니다. 새 코드로 재학습 후 `acc_joint` vs
> `acc_joint_major`(최빈 클래스 비율)로 비교하세요. 이전 체크포인트는 새 `model.py`와 호환되지 않습니다.

| 실험 | split 방식 | Val MSE | Test MSE |
|---|---|---|---|
| `runs/` | temporal (data/test 3세션) | 21.34 | 145.33 |
| `runs/sample/` | temporal (data/sample 3세션) | 15.40 | 131.76 |
| `runs/sample_cross/` | cross-session (train 2, test 1) | 14.12 | 153.79 |

---

## VLA (SmolVLA) 연동 — 주기적 호출

SmolVLA(450M, 학습 파라미터 ~100M)는 RTX 3090에서 batch-1 추론이 **~350–450 ms (≈ 4 프레임 @10 Hz)** 라
매 프레임 호출이 불가능합니다. 구조:

```
매 프레임 (~10 Hz)   heatmap ─► encoder ─┐
                                        ├─► LSTM ─► 행동
Δ 프레임마다 1회      RGB + 지시문 ─► SmolVLA ─► action chunk (16 step)
                     L 프레임 뒤 도착 ─► 현재 시점에 해당하는 chunk 원소 + age + has_plan ─┘
```

환경: lerobot 은 별도 conda env `lsma` (python 3.10, torch 2.10, lerobot 0.4.4) 에 설치되어 있습니다.
`PY` 환경변수 (기본 `python`)

```bash
cd baseline
# 1) 정방향 원본 이미지 캐시 (YOLO 와 달리 180° 회전 안 함)
$PY -m vla.cache_images --session ../data/sample/*/
# 2) SmolVLA fine-tune (heatmap 정책과 동일한 train/val/test 시간 split)
$PY -m vla.train_smolvla --sessions ../data/sample/*/ --out runs_vla/smolvla_c16 --steps 2000
# 3) 모든 프레임에서 chunk 미리 계산 → <session>/vla_plans_<tag>.npz
$PY -m vla.extract_plans --ckpt runs_vla/smolvla_c16/best --tag smolvla_c16 --sessions ../data/sample/*/
# 4) heatmap 정책을 VLA plan 조건으로 학습 (주기 Δ, 지연 L 을 학습 시 시뮬레이션)
$PY train.py --sessions ../data/sample/*/ --preset improved --vla_tag smolvla_c16 \
    --vla_period 10 --vla_latency 4 --run_dir runs/improved_vla
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--vla_tag` | '' | `vla_plans_<tag>.npz` 사용 (빈 값이면 VLA 없음) |
| `--vla_period` | 10 | VLA 호출 주기 Δ (프레임) |
| `--vla_latency` | 4 | 결과 도착 지연 L (프레임) — `extract_plans` 가 측정한 latency 로 설정 |
| `--vla_latency_jitter` | 1 | 학습 시 지연을 샘플마다 [L−j, L+j] 에서 뽑음 (평가는 L 고정) |
| `--vla_dropout` | 0.2 | 학습 중 시퀀스 단위로 VLA 입력 제거 → VLA 없이도 동작 |
| `--vla_eval_phase` | 0 | 평가 시 호출 위상 (학습 시는 매 샘플 랜덤) |

`train_smolvla` 는 chunk offset k 별 정확도(`acc_joint_k`)를 보고합니다 = "k 프레임 묵은 계획"의 품질.
heatmap 학습/평가 로그의 `acc_joint_vla_only` 는 VLA 계획만 그대로 실행했을 때의 정확도입니다.

### Cross-fitting (필수) + 세션 단위 CV

3) 의 `extract_plans` 를 전체 데이터로 학습한 VLA 로 돌리면, VLA 가 학습한 프레임의 plan 은 거의 정답이라
heatmap 정책이 VLA 를 베끼도록 학습됩니다 (sample 데이터에서 in-sample plan 의 정확도 0.61 vs 실제 ~0.3).
실험에는 반드시 OOF plan 을 쓰세요:

```bash
# 세션 단위 K-fold: fold f 의 세션 plan 은 그 세션을 못 본 VLA 가 생성 → vla_plans_oof.npz
$PY -m vla.crossfit --sessions ../data/sample/*/ --tag oof --steps 1000
# leave-one-session-out CV (fold × seed), runs/cv/<name>/summary.json
$PY cv.py --sessions ../data/sample/*/ --name improved_vla --seeds 0 1 2 -- --preset improved --vla_tag oof
```
`run_cv_experiments.sh` 가 위 전체(crossfit → base/improved/±VLA/주기 ablation)를 재개 가능하게 실행합니다.

주의
- chunk ≥ Δ + L + jitter 이어야 합니다 (기본 16 ≥ 10 + 4 + 1).
- cv.py 와 crossfit 의 fold 분할(`sessions[f::k]`)은 같아야 test 세션이 VLA 학습에 안 들어갑니다 (기본값 동일).
- LeRobot SmolVLA `forward` 는 `actions_id_pad`(오타) 키를 읽어 패딩 마스크가 적용되지 않으므로,
  chunk 가 split 끝을 넘는 샘플은 데이터셋에서 제외합니다.
