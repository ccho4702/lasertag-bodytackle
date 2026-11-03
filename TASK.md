# 태스크 정의 — 몸통박치기(Body-tackle) 술래잡기

> 2026-11-03 변경. 논문(arXiv 2603.11980)의 "레이저 조준·발사" 대신 **몸통박치기**: 일정 거리 이하로 붙으면 히트.
> 논문의 teacher→student 증류(distillation)는 쓰지 않는다. 논문 파이프라인 중 **student 쪽(ego 영상 → 정책)만**
> 가져와서 사람이 조종한 데모로 직접 학습(behavior cloning)하고, 여기에 **주기적 VLA 보조 신호**를 붙인다.

## 1. 역할과 데이터

| 역할 | 로봇 | 목표 | 데이터 (현재) |
|---|---|---|---|
| chaser (술래) | Misty | Cozmo 에게 **몸으로 부딪힌다** (최대한 빨리) | `data_download/chaser/장애물 {없음,1개,2개}/상대방 움직임 없음/` 30 세션, 5,540 프레임 — Cozmo 정지 |
| evader (도망자) | Cozmo | 쫓아오는 Misty 로부터 **도망친다** — 장애물 뒤에 숨는 것도 도망의 한 방법 (닿지 않는 것이 목적) | `data_download/evader/misty_stop_obs{0,1,2}/` 18 세션, 2,262 프레임 — Misty 정지 |

## 2. 입력 · 출력 · 라벨

- 입력: 로봇 자기 카메라(ego) 영상 → YOLO 상대/아군/장애물 히트맵 + depth (논문 student 입력과 동일)
- 출력 / 라벨: **운전자가 누른 명령** (`labels.csv` `requested_command`) 두 축
  - 회전: 왼쪽 / 없음 / 오른쪽 (Misty ω ∈ {+30, 0, −25} %, Cozmo ω ∈ {+60, 0, −60} mm/s)
  - 이동: 앞 / 없음 / 뒤 (Misty v ∈ {+25, 0, −25} %, Cozmo v ∈ {+90, 0, −90} mm/s)
  - `improved` 설정은 두 축을 각각 3-클래스로 분류, `paper` 설정은 [v, ω] 회귀(L2)
- 발사·조준(crosshair) 개념 없음. 로봇 API 출력은 기존과 동일 (`deploy.py`, `robots.py`).
- **영상-라벨 시간 정렬**: 화면 속 상대 위치와 회전 명령의 상관이 라벨을 2 프레임 앞당겼을 때 최대
  (k=−2: 0.275, k=0: 0.259, k=+4: 0.149) → 카메라 프레임이 명령보다 ≈0.2 s 늦게 도착.
  `train.py --label_shift -2` 로 보정 실험.
- VLA 지시문 (`robots.py` task, 몸통박치기로 변경):
  - chaser: "Drive into the enemy robot and bump it with your body while avoiding obstacles."
  - evader: "Run away from the chasing robot so that it cannot touch you; use obstacles to hide when you can."

## 3. 히트(hit) 정의 — overhead 카메라 기준

천장 카메라(1280×720, ≈ 620 px/m) 영상에서 overhead YOLO(`baseline/overhead/`)로 두 로봇 박스를 잡는다.

```
gap  = 두 박스 가장자리 사이 최단 거리 (px, 겹치면 0)
armed: 한 번이라도 gap > 30 px 로 떨어진 이후부터 판정 (시작 자세에서 붙어 있는 것은 무시)
cond : armed  AND  gap ≤ 10 px
hit  : cond 가 3 프레임 연속 (≈ 0.3 s)  →  첫 프레임 시각 = 히트 시각
박스가 잠깐 안 잡히면 직전 박스를 최대 5 프레임 유지
```

- 구현: `baseline/overhead/hit_detect.py` (`--gap_px 10 --arm_gap_px 30 --hit_frames 3 --hold 5 --dist_px 0`)
- 검증 UI: `baseline/overhead/ui` (`python -m http.server 8765`) — 영상 + 박스 + gap 그래프 + 히트 판정

## 4. 평가 지표

| 구분 | 지표 | 비고 |
|---|---|---|
| **주 지표 (실기)** | **첫 히트까지 시간 (s)** — chaser 는 짧을수록, evader 는 길수록 (제한시간 내 무히트 = evader 성공) | overhead 히트 판정으로 자동 측정 |
| 실기 보조 | 히트 성공률, 장애물 충돌 | |
| evader 보조 (참고) | 숨은 시간 비율 / 처음 숨기까지 시간 — overhead 에서 Misty 중심→Cozmo 중심 선분이 장애물 박스를 지나면 "숨음" | 장애물 0 개 세션은 해당 없음 |
| 오프라인 (학습 점검) | 회전·이동 축별 정확도와 균형 정확도, 최빈 클래스 기준선 대비 | 5-fold 세션 CV, 실기 성능의 대리 지표 |

참고 — 운전자 조종 데이터의 첫 히트 시간 (chaser 30 세션): 28/30 히트, 2.0–38.6 s, 중앙값 15.7 s.
히트 후 녹화 종료까지 중앙값 1.6 s (= 박치기 성공 직후 세션 종료). 실패 2 세션: 끝까지 375 px 떨어진 채 종료 1, 두 로봇이 동시에 검출된 프레임 없음 1.
evader 18 세션은 Misty 정지 상태라 히트 0/18. 숨기(overhead 선분-장애물 교차) 기준으로는
장애물 1개 세션 7/7, 2개 세션 5/5 가 숨는 데 성공했고, 숨은 프레임 비율 평균 39 % / 47 %,
마지막 10 프레임 기준 100 % / 60 % → 현재 데모(Misty 정지)는 "장애물 뒤로 가서 멈추기" 위주.
앞으로 두 로봇이 모두 움직이는 데이터를 수집할 예정 → evader 는 도망(+숨기), chaser 는 움직이는 상대 추적.

## 5. 알려진 제약

- chaser ego 카메라에 Cozmo 가 보이는 프레임은 33 % 뿐 (overhead 기하로 계산: 상대가 정면 ±15° 면 84 %,
  60° 이상 옆/뒤면 6 %). 안 보이는 구간의 회전 방향은 영상만으로는 정해지지 않아 오프라인 정확도 상한이 낮다.
- 운전자 명령 비율 (chaser): 좌회전 28 %, 우회전 28 %, 전진 25 %, 정지 17 %.

## 6. 코드 위치

| 파일 | 역할 |
|---|---|
| `baseline/overhead/hit_detect.py` | overhead 히트 판정 · 첫 히트 시간 · 결과 영상/JSON |
| `baseline/train.py --label_shift -2` | 영상-라벨 정렬 보정 |
| `baseline/run_parallel.py` | `improved_shift2_*`, `improved_ram_*` (몸통박치기 프롬프트 SmolVLA, VLA tag `oof_ram`) 실험 |

## 7. VLA 모델 선택 — SmolVLA 하나만 사용

| | SmolVLA | π0.5 | π0 |
|---|---|---|---|
| VLA 단독 명령 정확도 (evader, OOF) | 0.343 | 0.330 (Δ10) / 0.317 (Δ5) | 미실행 (GPU OOM) |
| +VLA 정책 test AE (evader, paper 설정) | 0.410 (Δ10) | 0.416 | – |
| VRAM | 4.5 GB | 20.7 GB | 15.9 GB |
| batch-1 지연 | 141 ms (B200 단독), 342 ms (RTX 3090) | ≈1 s (공유 GPU 측정) | ≈0.9 s (공유 GPU 측정) |

성능은 동급이고 비용은 SmolVLA 가 압도적으로 낮다 → **SmolVLA (`lerobot/smolvla_base`) 하나로 고정**.
3090(24 GB)에서 depth·YOLO 와 같이 올리려면 π 계열은 메모리·지연 모두 부담. π 전용 venv(`.venv-pi`)와
`vla/policies.py` 의 π 분기는 남겨두지만 실험 런처에서는 제외.
