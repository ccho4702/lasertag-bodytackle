# HANDOFF — 몸통박치기 술래잡기 (Misty chaser / Cozmo evader)

2026-11-03 기준. 태스크 정의는 [`TASK.md`](TASK.md), 이전(레이저 태그) 핸드오프는 [`docs/HANDOFF_v1_laser.md`](docs/HANDOFF_v1_laser.md).

## 0. 한눈에

| 항목 | 상태 |
|---|---|
| 학습 파이프라인 | ✅ 동작 (ego 영상 → YOLO 히트맵 + depth → 인코더 → LSTM → [v, ω]). train loss 정상 감소 |
| 체크포인트 → 로봇 API 출력 | ✅ `deploy.Agent.step(frame)` → Misty `/api/drive` dict / Cozmo `drive_wheels` dict (아래 1-2 에서 검증) |
| 오프라인 성능 | ⚠️ 학습에 안 쓴 세션에서 **최빈 명령 기준선 수준** (과적합). 실기에서 잘 움직일 거란 보장 없음 → **그래서 실기 테스트가 필요** |
| VLA | SmolVLA 하나로 고정 (π0/π0.5 대비 동급 성능, 1/4 메모리, 1/3 지연). 아직 정책 성능 이득 없음. 파인튜닝 체크포인트(chaser/evader, 각 865 MB)는 GitHub Release — `scripts/download_smolvla.sh` |
| overhead 히트 판정 | ✅ YOLO(mAP50-95 0.96) + 박스 간격 규칙 → 첫 히트 시간, 웹 UI |
| 실시간 | ⚠️ depth(DATv2-Large)가 병목. GPU 필수. CPU 는 1–2 s/step |

## 1. 설치 & 바로 테스트

```bash
git clone <this repo> lasertag && cd lasertag
python3.10 -m venv .venv && .venv/bin/pip install -r env/requirements_infer.txt   # 추론만 (학습·VLA 까지: env/requirements.txt)
source env/activate.sh
cd baseline
```
- 로봇 PC 쪽 컨트롤러 연결은 1-2 참고 (정책 서버 + `robot_client/`).
- 첫 실행 때 Depth-Anything-V2-Large (`depth-anything/Depth-Anything-V2-Large-hf`, ≈1.3 GB) 를 HF 에서 자동 다운로드 (인터넷 1회 필요).
- YOLO 가중치는 `checkpoints/yolo/` 에서 자동으로 찾는다 (`robots.py`).

### 1-1. 녹화 세션으로 재생 테스트 (로봇 없이)
녹화 세션 폴더(`labels.csv` + `images/`, 수집기 출력 그대로)를 하나 넣으면 프레임마다 명령·라벨·지연을 출력:
```bash
python deploy.py --ckpt ../checkpoints/policy/improved_chaser.pt --session <chaser 세션 폴더> --n 50
python deploy.py --ckpt ../checkpoints/policy/improved_evader.pt --session <evader 세션 폴더> --n 50
```
출력 예 (이 서버, CPU):
```
4 {'LinearVelocity': 25.0, 'AngularVelocity': 0.0} raw [25 0] label [0 0] 1931 ms
step latency (excluding the first 3 warm-up frames): median 1947 ms (budget ≈ 102 ms at 9.8 fps)
```
→ GPU 에서 **median 이 budget(chaser ≈ 102 ms, evader ≈ 119 ms) 안**인지 먼저 확인. 넘으면 1-4 참고.

### 1-2. 실기 (closed-loop) 테스트 — LSMA-Laser-tag 컨트롤러와 연결

로봇 PC 의 [LSMA-Laser-tag](https://github.com/gyeomhb/LSMA-Laser-tag) 는 Windows + Python 3.8 (Cozmo SDK) 라 torch 를 못 쓴다.
그래서 **정책은 서버로 띄우고**(이 레포 env, GPU 권장), 컨트롤러는 HTTP 로 프레임을 보내 명령을 받는다.

```
LSMA-Laser-tag (py3.8)                                  이 레포 (py3.10, GPU)
 camera frame (저장 형식 그대로) ──POST /step (raw RGB)──▶ policy_server.py ─ deploy.Agent ─ (+SmolVLA 스레드)
 Misty drive/time · Cozmo drive_wheel_motors ◀── {"v","w","name","api"} ──┘
```

**① 정책 서버** (이 레포, GPU 머신 또는 로봇 PC 의 별도 py3.10 env)
```bash
cd baseline
python policy_server.py --ckpt ../checkpoints/policy/improved_chaser_full.pt --port 8770    # Misty
python policy_server.py --ckpt ../checkpoints/policy/improved_evader_full.pt --port 8771    # Cozmo
# VLA 포함: --ckpt ../checkpoints/policy/improved_ram_chaser.pt --vla ../checkpoints/smolvla/chaser
```
시작 시 warm-up 후 `policy server: chaser (misty) on http://0.0.0.0:8770` 출력. 응답의 `ms` 가 step 지연 (budget: Misty ≈102 ms, Cozmo ≈119 ms).

**② 컨트롤러** — `robot_client/*.py` 4개를 LSMA-Laser-tag 폴더에 복사 (추가 패키지 없음: stdlib + numpy + Pillow + aiohttp)
```powershell
.\.venv\Scripts\python.exe -m unittest test_policy_misty                                     # 하드웨어 없이 점검
.\.venv\Scripts\python.exe .\policy_misty.py --policy-url http://<서버>:8770                 # wasd_misty_only 기반
.\.venv\Scripts\python.exe .\policy_misty.py --policy-url http://<서버>:8770 --with-cozmo    # wasd_misty 기반: Cozmo 랜덤 이동 토글 + tracking + overhead 저장
.\.venv\Scripts\python.exe .\policy_cozmo.py --policy-url http://<서버>:8771 [--record]      # wasd_cozmo 기반
```
- 키: **P** 정책 ON/OFF (ON 때 에피소드 리셋) · **W/A/S/D** 누르는 동안 수동 우선 · **Space** 정지(정책 OFF) · **Esc** 종료.
  서버 오류·카메라 2 s 끊김이면 정지.
- 입출력은 원래 컨트롤러와 동일:
  - Misty: 카메라 WebSocket 프레임 → 90° 시계방향 회전(480×640) 그대로 전송 / 명령은 `POST /api/drive/time`
    (LinearVelocity ±25, AngularVelocity +30 좌 / −25 우, TimeMs 500, 0.15 s 마다 재전송), 정지는 `drive/stop`.
  - Cozmo: 컬러 스트림, 머리 −5°, `latest_image.raw_image`(320×240) 그대로 전송 / `drive_wheel_motors(l, r)` (직진 ±90, 회전 ±60 mm/s).
  - 정책 출력은 학습 데이터의 명령 레벨로 스냅 (`--no_snap` 으로 끔) → 원래 WASD 명령과 같은 값.
- 녹화: 기존 녹화 버튼/`--record` 그대로 → `datasets_misty4/`, `dataset_cozmo/` 에 정책 주행이 저장 (`keys` 열 = `policy`).
  `--with-cozmo` 모드는 overhead 이미지도 같이 저장 → 1-3 의 히트 채점에 바로 사용.
- **비동기 구조** (추론이 느려도 로봇 제어가 멈추지 않음):
  - 컨트롤러는 새 카메라 프레임을 `POST /frame` 으로 올리고 바로 돌아온다 (왕복 ≈ 2 ms) — 응답은 서버의 *최신* 명령.
  - 서버는 별도 스레드에서 항상 가장 최근 프레임으로 추론 (못 따라간 프레임은 건너뜀).
  - Misty 는 기존 drive 루프가 최신 명령을 0.15 s 마다 `drive/time`(500 ms) 로 재전송, Cozmo 는 `drive_wheel_motors` 유지 → 연속 주행.
  - 카메라 프레임이 1 s(`--stale_ms`) 이상 안 오면 서버가 stop 을 돌려준다. 서버 오류도 stop.
  - 창 상단에 `infer ms`(추론 시간)·`frame age`(명령이 몇 ms 전 프레임 기준인지)·`fps` 표시. 추론이 budget(≈100 ms)보다 느리면
    명령이 그만큼 늦게 바뀐다 (LSTM 입력 간격도 학습과 달라짐) → CPU/Mac 은 `improved_cpu_*` 사용.
  - **추론 속도는 학습 데이터 fps 에 맞춰 제한** (Misty ≈ 9.8, Cozmo ≈ 8.4 fps; 수집기 설정은 15 fps 지만 실제 기록은 이 값).
    정책 입력 8 프레임이 학습 때와 같은 ≈ 0.8 s 를 덮게 하려는 것. 카메라가 15 fps 로 와도 최신 프레임만 골라 씀. `--infer_fps 0` 이면 매 프레임.
  - 명령이 한 프레임씩 튀면 서버에 `--vote 3` (최근 3회 다수결, 1–2 스텝 지연).

직접 파이썬에서 쓰려면: `from deploy import Agent; cmd, raw = Agent(ckpt).step(rgb_uint8_hwc)` (`baseline/live_example.py`).

### 1-3. 실기 결과 채점 — overhead 영상으로 첫 히트 시간
천장 카메라 영상(mp4 또는 jpg 폴더)을 녹화해 두면:
```bash
python -m overhead.hit_detect --weights ../checkpoints/yolo/overhead_yolo_v1.pt --video game.mp4 --fps 10 --out overhead/results
# 수집기 세션 폴더(overhead_images/ 포함)라면:  --sessions <세션 폴더들>
cd overhead/ui && ln -sfn ../results results && python -m http.server 8765    # http://localhost:8765
```
- 히트: 두 로봇 박스 간격 ≤ 10 px 가 3 프레임 연속, 한 번 30 px 이상 떨어진 뒤부터 판정 (`TASK.md` §3).
- UI: 영상 + 박스 + 간격 그래프 + 히트 시점. JSON 에 `time_to_first_hit_s`.
- 비교 기준(사람 조종 데모, chaser 30 세션): 28/30 히트, 첫 히트 중앙값 15.7 s.
- overhead 카메라 위치/해상도(1280×720)가 바뀌면 `--gap_px` 등 픽셀 임계값 재조정 필요.

### 1-4. 지연이 budget 을 넘으면
- 원인은 거의 다 depth (DATv2-**Large**). 선택지: (a) `preprocess.py` 의 `DEPTH_MODEL` 을 `...-Small-hf` 로 바꾸고
  **재전처리·재학습**, (b) TensorRT/fp16, (c) depth 채널 없는 모델 재학습 (`--channels enemy,ally,obstacle`).
- 체크포인트는 학습 때의 depth 모델과 짝이므로 depth 모델만 바꿔 끼우면 안 된다.

### 1-4b. baseline vs ours

**ours = baseline + 주기적 SmolVLA 보조 입력** (그 외 구조·데이터·학습 설정 동일). 둘 다 CPU(MacBook)용·GPU용이 있다.

| | CPU / MacBook (depth Small@252) | GPU (depth Large) |
|---|---|---|
| **baseline** | `policy/baseline_cpu_{chaser,evader}.pt` | `policy/baseline_gpu_{chaser,evader}.pt` |
| **ours** | `policy/ours_cpu_{chaser,evader}.pt` + `--vla checkpoints/smolvla/<robot>` | `policy/ours_gpu_{chaser,evader}.pt` + `--vla ...` |

```bash
# baseline (Misty)
python policy_server.py --ckpt ../checkpoints/policy/baseline_cpu_chaser.pt --port 8770
# ours (Misty) — SmolVLA 는 scripts/download_smolvla.sh 로 받기
python policy_server.py --ckpt ../checkpoints/policy/ours_cpu_chaser.pt --vla ../checkpoints/smolvla/chaser --port 8770
```
- ours 는 VLA 를 Δ=10 프레임마다 별도 스레드에서 호출하고 정책은 매 프레임 최신 plan + 그 나이(프레임)를 받는다.
  학습은 VLA 지연 L = chaser 4 / evader 3 프레임(RTX 3090 342 ms 기준) ± 1 로 시뮬레이션했다 → Mac 에서 SmolVLA 1회가 이보다
  훨씬 느리면 plan 이 오래돼 효과가 줄어든다 (최대 16 프레임 청크 안에서만 사용). 서버 시작 로그에 VLA 지연을 확인.
- ours 체크포인트를 `--vla` 없이 띄우면 경고 후 plan 없이(has_plan=0) 동작 — 학습 때 20 % 확률로 plan 을 빼고 학습해서 동작은 한다.
- 비교 실험: 같은 세션·같은 시작 위치에서 baseline / ours 를 번갈아 돌리고, overhead 첫 히트 시간(1-3)으로 비교.

### 1-5. GPU 없이 — CPU / MacBook (Apple silicon)

입력은 그대로(YOLO 히트맵 + **depth**)이고, depth 모델만 CPU 에서 빠른 것으로 바꾼 **CPU 전용 체크포인트**를 쓴다.

| depth 모델 (입력 크기) | depth 1장, CPU 8 스레드 (서버 Xeon 측정) | 용도 |
|---|---|---|
| Depth-Anything-V2-**Large** (518, 기본) | 1,636 ms | GPU 체크포인트 (`improved_full_*`) |
| Large (252) | 364 ms | – |
| Base (252) | 103 ms | – |
| **Small (252)** | **40 ms** | **CPU 체크포인트 (`improved_cpu_*`)** |

YOLO 히트맵(Misty 29–39 ms / Cozmo 17–20 ms)과 정책(수 ms)까지 합쳐 프레임당 ≈ 80 ms < budget(≈100 ms).
depth 모델은 학습과 같아야 하므로 체크포인트 cfg 에 `depth_model` / `depth_size` 가 저장돼 있고 `deploy.Agent` 가 자동으로 같은 것을 쓴다.

```bash
python policy_server.py --ckpt ../checkpoints/policy/improved_cpu_chaser.pt --port 8770 --device cpu   # Misty
python policy_server.py --ckpt ../checkpoints/policy/improved_cpu_evader.pt --port 8771 --device cpu   # Cozmo
# MacBook (M 시리즈): --device mps (Apple GPU) 와 --device cpu 둘 다 해 보고 시작 로그 / deploy.py median 이 작은 쪽 사용
```
- 메모리: 서버 2개(chaser + evader) 합쳐 ≈ 1–2 GB → 24 GB MacBook 에 둘 다 충분. Large depth(GPU 체크포인트)를 CPU/MPS 로 돌려도
  메모리는 ≈ 3–4 GB 로 들어가지만 CPU 에선 1.6 s/프레임이라 실시간 불가.
- MPS 는 지원 안 되는 연산을 CPU 로 넘기도록 `PYTORCH_ENABLE_MPS_FALLBACK=1` 이 자동 설정된다. 보통 depth(ViT)는 MPS 가 빠르고,
  Small@252 처럼 작은 입력은 CPU↔GPU 복사 비용 때문에 차이가 작거나 CPU 가 나을 수도 있다.
- SmolVLA 는 CPU 에서 수 초/회라 Mac 에서는 VLA 없이 테스트 권장.
- 서버 시작 로그의 `ms`(또는 `python deploy.py --ckpt ... --session ...` 의 median)를 Mac 에서 먼저 확인.

## 2. 체크포인트 (`checkpoints/`)

| 파일 | 설명 | 학습 / 선택 | test (학습에 안 쓴 세션) |
|---|---|---|---|
| `policy/improved_chaser.pt` | **먼저 써볼 것.** spatial-softmax 인코더, 축별 3-클래스 | 5-fold 중 val 최고 fold (f2, 세션 24개 학습) | acc 이동 0.63 / 회전 0.41 / joint 0.33 (최빈 0.32) |
| `policy/improved_evader.pt` | 동일 구조, Cozmo | 5-fold 중 val 최고 (f4) | 이동 0.69 / 회전 0.48 / joint 0.36 (최빈 0.32) |
| `policy/paper_chaser.pt` | 논문 student 그대로 (ResNet18, L2 회귀) | f3 | joint 0.12 (최빈 0.34) |
| `policy/paper_evader.pt` | 〃 | f0 | joint 0.08 (최빈 0.44) |
| `policy/improved_cpu_{chaser,evader}.pt` | improved + **Small@252 depth**, 전체 세션 — **CPU / MacBook 1순위** | 세션별 마지막 5 % 는 확인용 | 5-fold 비교는 `selection.json` 갱신 예정 |
| `policy/improved_full_{chaser,evader}.pt` | improved + Large depth, **전체 세션으로 학습** (배치 32, 60 epoch, bf16) — GPU 1순위 | 세션별 마지막 5 % 는 확인용 | – |
| `policy/improved_ram_chaser.pt` | improved + SmolVLA 조건 (Δ10, 지연 4) — `--vla` 와 함께 | 5-fold 중 val 최고 | `selection.json` |
| `policy/improved_vla_evader.pt` | improved + SmolVLA 조건 (Δ10, 지연 3) | 5-fold 중 val 최고 | `selection.json` |
| `smolvla/{chaser,evader}/` | 파인튜닝 SmolVLA (Release 다운로드) | 전체 세션, 1000 step | VLA 단독 정확도 0.29–0.34 (OOF) |
| `yolo/misty_yolo26s_v1.pt` | Misty 카메라용 YOLO (enemy/ally/obstacle) | 기존 release_v1 | |
| `yolo/cozmo_yolo_v1.pt` | Cozmo 카메라용 YOLO | 기존 | |
| `yolo/overhead_yolo_v1.pt` | 천장 카메라용 YOLO (misty/cozmo/obstacle) — 히트 판정 | DINO 자동라벨 7.8k 프레임 | val mAP50-95 0.96 |

- 선택 근거·지표: `checkpoints/policy/selection.json`. 정책 체크포인트는 optimizer 상태를 뺀 경량본 (deploy 에 필요한 `model/ema/cfg` 만).
- 모두 **세션 일부(80 %)로 학습한 CV fold 모델**이다. 전체 데이터로 다시 학습한 최종본은 아직 없음.
- 정직한 예상: chaser 는 회전 방향이 거의 우연 수준 → 상대를 향해 제대로 돌지 못할 가능성 큼. evader improved 는 fold 다수가
  한 명령만 내는 붕괴 상태였음 (선택된 f4 는 아님). 실기에서 확인하려는 게 바로 이것.

## 3. 다시 학습하기

데이터 레이아웃은 수집기 출력 그대로 (`data_download/chaser/<조건>/<조건>/<ts>/`, `data_download/evader/<조건>/<ts>/`).
```bash
python preprocess.py --robot chaser --session '../data_download/chaser/*/*/*/'     # → <세션>/obs.npz (히트맵+depth+라벨)
python train.py --preset improved --robot chaser --sessions <학습 세션들> --test_sessions <테스트 세션들> --run_dir runs/my_chaser
python run_parallel.py --robots chaser --seeds 0 --no_lc --only improved_chaser     # 5-fold CV 런처 (GPU 메모리 예산으로 병렬)
python report.py                                                                      # CV 결과 표
```
- 프리셋: `paper` (논문 student), `improved` (축별 3-클래스, spatial softmax, 증강, EMA).
- `--label_shift -2`: 카메라 프레임이 명령보다 ≈0.2 s 늦게 도착하는 것 보정 (실험 중, `TASK.md` §2).
- 시간 감 (B200 단독): epoch 당 chaser 13 s / evader 6 s, 100 epoch ≈ 20 / 10 분. 데이터에 비례.

## 4. VLA (SmolVLA, 선택)

- 매 프레임이 아니라 Δ 프레임(5 또는 10)마다 호출, 지연 L 프레임(3090 기준 342 ms → chaser 4 / evader 3) 뒤 도착한 plan 을
  정책의 보조 입력으로 사용: `agent.step(frame, vla_plan=..., vla_age_frames=k)`.
- 학습: `python -m vla.cache_images ...` → `python -m vla.crossfit --sessions ... --tag oof_ram --folds 5` (OOF plan) →
  `train.py ... --vla_tag oof_ram --vla_latency 4`. 지시문은 `robots.py` 의 `task`.
- 현재 결과: VLA 단독 명령 정확도 0.29–0.34 (최빈 수준), 붙여도 정책 개선 없음 → **실기 첫 테스트는 VLA 없이** 권장.
- **다운로드**: `bash scripts/download_smolvla.sh` → `checkpoints/smolvla/{chaser,evader}/` (`best/` + `vla_config.json`).
  GitHub Release `v0.1` 첨부파일 (파일당 865 MB 라 git 에는 없음, `gh auth login` 필요).
  재학습용 base 모델은 Hugging Face `lerobot/smolvla_base` 에서 자동 다운로드.
- 파인튜닝본: 로봇별 **전체 세션**으로 1000 step, 몸통박치기 지시문 (`vla_config.json` 의 `task`).
- 실기에서 VLA 쓰기 — VLA 조건부 정책과 짝으로:
  ```bash
  python live_example.py --ckpt ../checkpoints/policy/improved_ram_chaser.pt --vla ../checkpoints/smolvla/chaser --robot-ip <ip>
  python -m vla.runner --ckpt ../checkpoints/smolvla/chaser --session <세션>      # VLA 단독 지연 측정
  ```
  `vla/runner.py` 의 `AsyncVLA` 가 Δ 프레임마다 별도 스레드로 VLA 를 호출하고, 정책은 매 프레임 가장 최근 plan 과 그 나이(프레임)를 받는다.
- evader 의 VLA 조건부 정책(`improved_vla_evader.pt`)은 예전 지시문으로 만든 plan 으로 학습됨 (VLA 는 새 지시문) — 성능 이득이 없던 상태라 참고용.

## 5. 알려진 이슈 / 다음 할 일

1. 일반화 부족: train loss 는 0.97→0.35 로 떨어지지만 val/test 는 최빈 기준선 수준. 데이터 양(세션 24–30개)과
   운전자 명령 일관성(좌 28 % / 우 28 % 를 오감)이 병목. 학습곡선(25→50→100 %)이 평평.
2. chaser 카메라에 상대가 보이는 프레임 33 % — 안 보일 때 어느 쪽으로 돌지 영상만으로는 정해지지 않음.
3. 데이터 수집 권장: 운전 규칙 통일(안 보이면 한 방향 회전 탐색 → 보이면 중앙 정렬 → 직진), 60–70 s 세션,
   시작 위치·장애물 매번 변경, 두 로봇 모두 움직이는 세션. 목표 로봇당 ≈ 2만 프레임(≈35 분 주행), 먼저 5천 프레임으로 학습곡선 확인.
4. 실시간: GPU 에서 step 지연 측정 후 depth 경량화 여부 결정 (1-4).
