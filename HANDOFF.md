# HANDOFF — 몸통박치기 술래잡기 (Misty chaser / Cozmo evader)

2026-11-03 기준. 태스크 정의는 [`TASK.md`](TASK.md), 이전(레이저 태그) 핸드오프는 [`docs/HANDOFF_v1_laser.md`](docs/HANDOFF_v1_laser.md).

## 0. 한눈에

| 항목 | 상태 |
|---|---|
| 학습 파이프라인 | ✅ 동작 (ego 영상 → YOLO 히트맵 + depth → 인코더 → LSTM → [v, ω]). train loss 정상 감소 |
| 체크포인트 → 로봇 API 출력 | ✅ `deploy.Agent.step(frame)` → Misty `/api/drive` dict / Cozmo `drive_wheels` dict (아래 1-2 에서 검증) |
| 오프라인 성능 | ⚠️ 학습에 안 쓴 세션에서 **최빈 명령 기준선 수준** (과적합). 실기에서 잘 움직일 거란 보장 없음 → **그래서 실기 테스트가 필요** |
| VLA | SmolVLA 하나로 고정 (π0/π0.5 대비 동급 성능, 1/4 메모리, 1/3 지연). 아직 정책 성능 이득 없음. 체크포인트(865 MB)는 이 레포에 미포함 |
| overhead 히트 판정 | ✅ YOLO(mAP50-95 0.96) + 박스 간격 규칙 → 첫 히트 시간, 웹 UI |
| 실시간 | ⚠️ depth(DATv2-Large)가 병목. GPU 필수. CPU 는 1–2 s/step |

## 1. 설치 & 바로 테스트

```bash
git clone <this repo> lasertag && cd lasertag
python3.10 -m venv .venv && .venv/bin/pip install -r env/requirements_infer.txt   # 추론만 (학습·VLA 까지: env/requirements.txt)
source env/activate.sh
cd baseline
```
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

### 1-2. 실기 (closed-loop) 테스트
```python
from deploy import Agent
agent = Agent('../checkpoints/policy/improved_chaser.pt')   # 로봇 종류는 체크포인트에 들어있음
agent.reset()                                                # 에피소드 시작마다
while True:
    frame = <로봇 카메라 RGB uint8 HxWx3 — 수집기가 저장한 것과 똑같은 해상도/방향>
    cmd, raw = agent.step(frame)
    requests.post(f'http://{misty_ip}/api/drive', json=cmd)   # Misty: {'LinearVelocity','AngularVelocity'} %
    # Cozmo: robot.drive_wheels(**cmd)                         # {'l_wheel_speed','r_wheel_speed'} mm/s
```
- 예제 루프: `baseline/live_example.py` (Misty REST). `get_frame()` 은 **수집기와 같은 카메라 코드**로 채워야 함
  (학습 이미지와 회전·해상도가 다르면 YOLO 히트맵이 틀어진다).
- 매 프레임 호출. 프레임을 건너뛰면 LSTM 시퀀스 간격이 학습(≈10 fps)과 달라진다.
- `Agent(..., snap=True)` 로 출력값을 시연 명령 레벨(예: ω ∈ {−25, 0, 30})에 맞춰 스냅 가능.

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

## 2. 체크포인트 (`checkpoints/`)

| 파일 | 설명 | 학습 / 선택 | test (학습에 안 쓴 세션) |
|---|---|---|---|
| `policy/improved_chaser.pt` | **먼저 써볼 것.** spatial-softmax 인코더, 축별 3-클래스 | 5-fold 중 val 최고 fold (f2, 세션 24개 학습) | acc 이동 0.63 / 회전 0.41 / joint 0.33 (최빈 0.32) |
| `policy/improved_evader.pt` | 동일 구조, Cozmo | 5-fold 중 val 최고 (f4) | 이동 0.69 / 회전 0.48 / joint 0.36 (최빈 0.32) |
| `policy/paper_chaser.pt` | 논문 student 그대로 (ResNet18, L2 회귀) | f3 | joint 0.12 (최빈 0.34) |
| `policy/paper_evader.pt` | 〃 | f0 | joint 0.08 (최빈 0.44) |
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
- 파인튜닝된 SmolVLA 체크포인트(865 MB)는 GitHub 파일 한도 때문에 미포함. 필요하면 별도 전달.

## 5. 알려진 이슈 / 다음 할 일

1. 일반화 부족: train loss 는 0.97→0.35 로 떨어지지만 val/test 는 최빈 기준선 수준. 데이터 양(세션 24–30개)과
   운전자 명령 일관성(좌 28 % / 우 28 % 를 오감)이 병목. 학습곡선(25→50→100 %)이 평평.
2. chaser 카메라에 상대가 보이는 프레임 33 % — 안 보일 때 어느 쪽으로 돌지 영상만으로는 정해지지 않음.
3. 데이터 수집 권장: 운전 규칙 통일(안 보이면 한 방향 회전 탐색 → 보이면 중앙 정렬 → 직진), 60–70 s 세션,
   시작 위치·장애물 매번 변경, 두 로봇 모두 움직이는 세션. 목표 로봇당 ≈ 2만 프레임(≈35 분 주행), 먼저 5천 프레임으로 학습곡선 확인.
4. 실시간: GPU 에서 step 지연 측정 후 depth 경량화 여부 결정 (1-4).
