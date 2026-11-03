# Misty / Cozmo Laser Tag — 인수인계 (2026-11-03)

새 서버에서 이 번들만으로 처음부터 이어서 진행하기 위한 문서입니다.
세부 사용법은 `baseline/README.md`, Cozmo 탐지기는 `baseline/detector/README.md`.

## 1. 목표와 제약

- 베이스라인: arXiv 2603.11980 "Learning Visuomotor Policy for Multi-Robot Laser Tag Game"의 student 정책
  (YOLO 히트맵 + depth → ResNet18 → LSTM → MLP → 속도 명령).
- **목표: 여기에 VLA 하나를 붙여 성능을 올리기.** VLA는 지연(latency) 때문에 매 프레임이 아니라
  **일정 간격(Δ 프레임)으로만** 호출되고, 그 사이에는 빠른 히트맵 정책이 마지막 VLA 계획을 조건으로 받아 매 프레임 동작.
  현재 VLA = SmolVLA (`lerobot/smolvla_base`, lerobot 0.4.4), 측정 지연 ≈ 330–342 ms (RTX 3090).
- 로봇 두 종류, **각각 따로 학습**:
  - `chaser` = Misty (쫓아가서 조준, 논문 Team I)
  - `evader` = Cozmo (도망, 논문 Team II)
- **각 로봇은 자기 카메라 화면만 입력으로 사용.** 상단(overhead) 카메라 이미지와 ArUco 위치는 입력에 절대 쓰지 않음
  (데이터에는 들어 있으나 분석용으로만).
- 데이터는 키보드 원격조작 시연 → 논문의 DAgger/teacher 대신 BC.

## 2. 번들 구성

```
lasertag/
├── HANDOFF.md                 (이 문서)
├── env/requirements.txt       핵심 패키지 고정 버전 (Python 3.10, CUDA 12.8 torch)
├── env/requirements_lsma_full.txt   원 서버 conda env `lsma` 의 pip freeze 전체
├── baseline/                  코드 (README.md, robots.py, preprocess.py, train.py, cv.py, deploy.py, vla/, detector/)
│   └── runs/
│       ├── detect/runs_det/cozmo_v1/   Cozmo 시점 YOLO (weights/best.pt, results.csv, args.yaml)
│       └── verify/paper_evader_f0/     paper 설정 검증 학습 1회 결과
├── yolo_extracted/release_v1/checkpoint/misty_yolo26s_v1.{pt,onnx}   Misty 시점 YOLO (원본)
├── data_download/
│   ├── chaser/<장애물 N개>/상대방 움직임 없음/<session>/   30 세션
│   ├── evader/misty_stop_obs{0,1,2}/<session>/            18 세션
│   │     각 세션: images/ overhead_images/ labels.csv metadata.json obs.npz(전처리 완료)
│   └── evader_yolo/           Cozmo 탐지기 라벨 (autolabels*.json, manual_review.json)
├── hf_cache/hub/              HuggingFace 모델 (smolvla_base, SmolVLM2-500M, Depth-Anything-V2-Large)
└── torch_cache/hub/checkpoints/resnet18-f37072fd.pth
```

`baseline/` 안의 경로는 모두 `../data_download`, `../yolo_extracted` 처럼 **상대경로**라서 이 구조만 유지하면 됨.

## 3. 새 서버 셋업

```bash
tar xf lasertag_handoff_20261103.tar && cd lasertag
conda create -n lsma python=3.10 -y && conda activate lsma
pip install -r env/requirements.txt          # 실패 시 env/requirements_lsma_full.txt 참고
# opencv-python 5.x 와 headless 4.x 가 같이 깔려 있던 환경. 충돌 나면 opencv-python-headless 만 설치해도 됨

export HF_HOME=$PWD/hf_cache                 # 모델 재다운로드 방지 (없으면 자동 다운로드됨)
export TORCH_HOME=$PWD/torch_cache
cd baseline
python -c "import torch, ultralytics, transformers, lerobot; print(torch.cuda.is_available())"
```

## 4. 데이터

| | chaser (Misty) | evader (Cozmo) |
|---|---|---|
| 세션 / 프레임 | 30 / 5,540 (장애물 0·1·2개 × 10) | 18 / 2,262 (장애물 0·1·2개 = 6·7·5) |
| 카메라 | 480×640 세로 RGB, 이미 똑바로 섬, 9.8 fps | 320×240 RGB(붉은 색감, 낮은 시점), 8.4 fps |
| 상대 로봇 | Cozmo (정지) | Misty (정지) |
| 라벨 → [v, ω] | `requested_command` → v {-25,0,25}%, ω {-25(우),0,+30(좌)}% | v=(L+R)/2 ∈ {-90,0,90}, ω=(R−L)/2 ∈ {-60,0,60} mm/s |
| 로봇 API | Misty `POST /api/drive {LinearVelocity, AngularVelocity}` (%) | `drive_wheels(l=v−ω, r=v+ω)` mm/s |

- 모든 변환은 `baseline/robots.py` 한 곳에. 정책은 항상 [v, ω] (ω>0 = 좌회전)를 출력, `Robot.to_command()`가 API 입력으로 변환.
- chaser 라벨의 `command`/`linear_velocity_percent` 는 로봇이 ack 한 명령(키 전환 직후 6.7% 프레임이 이전 명령).
  기본은 운영자가 낸 `requested_command` (`--label_source executed` 로 바꿀 수 있음).
- **obs.npz (전처리 결과, 이미 포함)**: `obs (N,4,224,224) fp16` = enemy/ally/obstacle 히트맵 + DATv2-L depth,
  `actions (N,2)` [v, ω], `boxes` YOLO 원시 검출, `valid`. labels.csv 행과 1:1.
  - 히트맵: 박스 중심에 크기 s = λ/D (λ=40 px·m) 가우시안, D = 초점거리·실제폭/박스폭 (0.3–3 m clip).
  - depth: 5–95% clip → [0,1].
  - 검출 비율(세션 평균): chaser enemy 37% / ally 8% (이 데이터엔 아군이 없음 → 오검출) / obstacle 54%,
    evader enemy 17% / obstacle 33%.

## 5. 실행 순서 (새 서버)

전처리는 끝나 있으므로 바로 학습 가능. 다시 하려면 `--force`.

```bash
cd baseline
# (선택) 전처리 재실행 — YOLO + DATv2
python preprocess.py --robot chaser --session "../data_download/chaser/*/*/*/" --force
python preprocess.py --robot evader --session "../data_download/evader/*/*/" --force

# 단일 학습 (paper 설정)
python train.py --robot evader --preset paper --sessions S... --test_sessions T... --run_dir runs/x

# 본 실험: 로봇별 전처리 → VLA 이미지 캐시 → OOF SmolVLA 계획(5-fold) → 5-fold CV (paper / +VLA Δ10 / +VLA Δ5)
ROBOTS="chaser evader" PY=python ./run_paper_cv.sh
# 결과: runs/cv/<name>/summary.json, 개별 run: runs/cv/<name>/f<fold>_s<seed>/test_metrics.json
```

- `run_paper_cv.sh` 은 seed 3개 (`--seeds 0 1 2`). 원 서버(RTX 3090 1장) 기준 run 1개: evader ≈ 15분, chaser ≈ 45분
  → 전체 90 run ≈ 45시간, seed 1개면 ≈ 15시간 (+ VLA cross-fitting ≈ 1.3시간). GPU 여러 장이면 fold/robot 단위로 나눠 병렬 실행 권장
  (`cv.py` 는 끝난 fold 를 건너뛰므로 같은 `--name` 으로 여러 프로세스가 다른 seed 를 맡아도 됨).
- VLA 지연(프레임): chaser 4, evader 3 (342 ms × fps). 다른 GPU면 `extract_plans.py` 가 찍는 latency 로 다시 계산해 `--vla_latency` 수정.
- 배포: `deploy.Agent(ckpt).step(rgb_frame)` → 로봇 API 인자 dict (학습과 같은 전처리 코드 사용).

## 6. 지금까지 결과

**Cozmo 시점 YOLO (val 306장, 장애물 조건별 1세션)**

| 모델 | enemy mAP50 / 50-95 | obstacle mAP50 / 50-95 |
|---|---|---|
| Misty YOLO 그대로 | 0.000 / 0.000 | 0.175 / 0.145 |
| cozmo_v1 (Misty 모델에서 fine-tune) | 0.981 / 0.967 | 0.982 / 0.970 |

라벨 = Grounding DINO 자동 라벨 → CLIP 검증 → 규칙 필터 → 전수 육안 검수 (enemy 357→41 삭제, obstacle 782→109 삭제, 18 추가).
val 라벨도 같은 방식이라 점수는 다소 낙관적. 사용자 육안 확인: OK.

**paper 설정 검증 (evader, fold 0: 학습 14 / 테스트 4 세션, seed 0)**

| 예측기 | 테스트 AE ↓ | joint 명령 정확도 |
|---|---|---|
| 항상 정지 | 0.598 | – |
| 항상 최빈(직진) | 0.383 | 0.443 |
| 항상 학습 평균 | 0.342 | – |
| paper 설정 | 0.402 | 0.076 (bacc 0.167) |

AE = v/90, ω/60 로 정규화한 MSE. best val 은 4 epoch (이후 train loss 0.44→0.07, val 악화) → 1.4k 윈도우에 ResNet18(12M) 과적합.

**이전 샘플 데이터(3세션, 폐기)**: 모든 설정이 최빈 기준선 이하, VLA 효과 판별 불가 → 데이터 규모 문제로 결론.

## 7. 결정 / 가정 사항

- 이미지는 기록된 그대로 사용 (이전 코드의 180° flip, 상단 90행 crop 제거 — 새 데이터에서 오검출 원인).
- ultralytics 는 numpy 입력을 **BGR** 로 받음 → `preprocess.Perception` 에서 변환 (빠뜨리면 evader enemy 검출 0).
- 장애물 = 벽돌 블록만. 경기장 벽, 가장자리 흰/크림색 상자, 사람, 소화기, 장비는 장애물 아님.
- `robots.py` 가정값: Misty 세로 스트림 HFOV 55°(미측정), 로봇 폭 Misty 0.25 m / Cozmo 0.055 m → 히트맵 커널 크기에만 영향. 실측 시 수정.
- 논문 지표 Hit@15/30, 충돌률은 실기 rollout 필요 → 오프라인은 AE + 명령 정확도만.
- VLA 계획은 반드시 OOF(cross-fitting): 같은 세션을 본 VLA 의 계획은 라벨 누설 (샘플 데이터에서 in-sample 0.605 vs OOF 0.22 확인).
- LeRobot SmolVLA 는 `actions_id_pad` 오타로 패딩 마스크를 안 씀 → chunk 가 split 끝을 넘지 않게 데이터셋에서 처리.

## 8. 남은 일

1. 본 실험 실행 (`run_paper_cv.sh`). 추천: 먼저 seed 1개로 paper vs 데이터 규모에 맞춘 설정
   (`--head cls --n_bins 3` 시연 명령 값 분류, 얕은 인코더/정규화, early stopping) 비교 후 VLA 효과 측정.
2. chaser 는 아직 학습 검증 전 (데이터 약 3배).
3. 실측값 반영: Misty 카메라 HFOV, 로봇 폭.
4. 실기 rollout (`deploy.py`) 로 Hit/충돌률 측정.
5. 보류 아이디어(요청 시): VLM 임베딩 FiLM 조건, VLA 분류 헤드, 다중 프레임 VLA 입력, 게이팅, VLA 증강.
6. Overleaf 제안서(https://www.overleaf.com/read/phnqjmcrmkjy#a5aff8)는 로그인 필요로 미확인 — 내용 반영 필요 시 본문 공유.
