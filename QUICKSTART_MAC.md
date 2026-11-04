# QUICKSTART — MacBook 에서 정책 서버 + 로봇 PC 컨트롤러

```
 MacBook (이 레포)                                  로봇 PC (Windows, LSMA-Laser-tag)
 ┌─────────────────────────────┐   LAN (HTTP)   ┌─────────────────────────────────────┐
 │ policy_server :8770  chaser │◀──── frame ────│ policy_misty.py  (Misty 카메라/주행) │
 │ policy_server :8771  evader │──── command ──▶│ policy_cozmo.py  (Cozmo 카메라/주행) │
 └─────────────────────────────┘                └─────────────────────────────────────┘
```
- 정책(torch)은 MacBook, 로봇 제어는 기존 LSMA-Laser-tag 그대로. 프레임 업로드와 추론·주행은 비동기라 추론이 느려도 끊기지 않는다.
- 체크포인트: **baseline** = `baseline_cpu_*`, **ours** = `ours_cpu_*` + SmolVLA. (CPU/Mac 용 = depth Small@252)

---

## 1. MacBook — 설치 (1회)

```bash
# Python 3.10–3.12 필요 (없으면: brew install python@3.11)
brew install gh && gh auth login                  # private 레포 접근 (SmolVLA 다운로드에도 사용)
gh repo clone ccho4702/lasertag-bodytackle lasertag && cd lasertag
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r env/requirements_infer.txt
```

이미 받아뒀다면 `cd lasertag && git pull`.

## 2. MacBook — 정책 서버 2개 (터미널 2개)

```bash
# 터미널 1 — chaser (Misty), Apple GPU
cd lasertag && source .venv/bin/activate && cd baseline
python policy_server.py --ckpt ../checkpoints/policy/baseline_cpu_chaser.pt --port 8770 --device mps

# 터미널 2 — evader (Cozmo), CPU 4 스레드
cd lasertag && source .venv/bin/activate && cd baseline
OMP_NUM_THREADS=4 python policy_server.py --ckpt ../checkpoints/policy/baseline_cpu_evader.pt --port 8771 --device cpu
```
- 첫 실행은 depth 모델(Depth-Anything-V2-Small, ≈100 MB)을 Hugging Face 에서 받는다 (인터넷 필요, 이후 캐시).
- 준비되면 이렇게 출력된다:
  `policy server: infer ≤ 9.8 fps | chaser (misty, mps) on http://0.0.0.0:8770  inference ≈ NN ms/frame (budget 102 ms)`
  → **inference 가 budget(≈100 ms) 이하인지 확인.** 넘으면 `--device` 를 mps↔cpu 바꿔서 빠른 쪽 사용.
- 맥 IP 확인: `ipconfig getifaddr en0` (Wi-Fi). 방화벽 팝업이 뜨면 python 의 수신 연결을 **허용**.
- 맥에서 확인: `curl http://localhost:8770/info` → `{"robot": "chaser", ...}`

### ours (+SmolVLA) 로 돌릴 때
```bash
bash scripts/download_smolvla.sh                # 1회, checkpoints/smolvla/{chaser,evader}
python policy_server.py --ckpt ../checkpoints/policy/ours_cpu_chaser.pt --vla ../checkpoints/smolvla/chaser --port 8770 --device mps
OMP_NUM_THREADS=4 python policy_server.py --ckpt ../checkpoints/policy/ours_cpu_evader.pt --vla ../checkpoints/smolvla/evader --port 8771 --device cpu
```
메모리: baseline 서버 ≈ 1 GB 씩, ours ≈ 2.5–3 GB 씩 → 16 GB MacBook Air 에 두 개 모두 가능.
SmolVLA 는 Mac 에서 1회 수 초일 수 있어 plan 이 늦게 도착한다 (제어는 막히지 않음).

## 3. 로봇 PC (Windows) — 컨트롤러 (1회 복사)

이 레포의 `robot_client/` 4개 파일을 **LSMA-Laser-tag 폴더**(`C:\Users\user\LSMA`)에 복사:
`policy_client.py`, `policy_misty.py`, `policy_cozmo.py`, `test_policy_misty.py` (추가 설치 없음, 기존 py3.8 `.venv` 그대로).

```powershell
cd C:\Users\user\LSMA
.\.venv\Scripts\python.exe -m unittest test_policy_misty          # 하드웨어 없이 점검 → OK
Invoke-RestMethod http://<맥IP>:8770/info                          # 맥 서버 연결 확인
```

## 4. 실행

```powershell
# Misty (chaser) — 기존 wasd_misty_only 와 동일 화면
.\.venv\Scripts\python.exe .\policy_misty.py --policy-url http://<맥IP>:8770

# Cozmo (evader) — 기존 wasd_cozmo 와 동일 (별도 PowerShell 창)
.\.venv\Scripts\python.exe .\policy_cozmo.py --policy-url http://<맥IP>:8771
```
| 키 (창 클릭 후) | 동작 |
|---|---|
| **P** | 정책 ON / OFF (ON 할 때마다 에피소드 리셋) |
| W/A/S/D | 누르는 동안 수동 조종 우선 |
| **Space** | 즉시 정지 + 정책 OFF |
| Esc | 종료 |

- 창 상단: `POLICY ON: forward_left | infer 85 ms | frame age 110 ms | 9.7 fps` — 추론 시간 / 명령이 몇 ms 전 프레임 기준인지.
- 서버가 죽거나 카메라가 1 s 끊기면 자동 정지.
- 둘 다 한 PC 에서 돌릴 때: Cozmo SDK 는 한 프로그램만 잡을 수 있으므로 `policy_misty.py` 에 `--with-cozmo` 를 **붙이지 않는다**.
  (`--with-cozmo` 는 Cozmo 를 정책 대신 기존 랜덤 주행 토글로 움직이는 모드 — chaser 만 테스트할 때 사용.)

## 5. 첫 히트 시간 측정 (overhead / bird's eye view)

overhead YOLO(`checkpoints/yolo/overhead_yolo_v1.pt`)로 두 로봇 박스를 잡고, 박스 간격 ≤ 10 px 가 3 프레임 연속이면 히트
(시작 자세에서 붙어 있는 것은 무시). 맥에서:

```bash
cd lasertag/baseline
# (a) 실시간 — Jetson overhead 스트림을 바로 받아 첫 히트 순간 출력
python -m overhead.hit_detect --weights ../checkpoints/yolo/overhead_yolo_v1.pt \
    --video http://192.168.1.37:8088/stream.mjpg --stop_on_hit --max_s 120 --out overhead/results
#   → "*** FIRST HIT at 12.34 s (frame 123)"   (시간 = 이 명령 시작 시점부터; 정책 P 와 같이 시작)
# (b) 녹화 후 — 녹화 세션(overhead_images/ 포함, policy_misty.py --with-cozmo 로 녹화) 또는 mp4
python -m overhead.hit_detect --weights ../checkpoints/yolo/overhead_yolo_v1.pt --sessions <세션폴더들> --out overhead/results
python -m overhead.hit_detect --weights ../checkpoints/yolo/overhead_yolo_v1.pt --video game.mp4 --out overhead/results
# 결과 확인 UI: 영상 + 박스 + 간격 그래프 + 히트 시점
cd overhead/ui && ln -sfn ../results results && python -m http.server 8765      # 브라우저 http://localhost:8765
```
- 결과: `overhead/results/<이름>.json` (`time_to_first_hit_s`, 프레임별 박스/간격) + 주석 영상 `.mp4`.
- 판정 파라미터: `--gap_px 10 --hit_frames 3 --arm_gap_px 30` (기본값). overhead 카메라 위치·해상도(1280×720)가 바뀌면 재조정.
- 장치는 자동 선택 (cuda → mps → cpu). `--device cpu` 로 지정 가능.

## 6. 문제 해결

| 증상 | 조치 |
|---|---|
| `inference` 가 100 ms 보다 큼 | `--device` 를 mps ↔ cpu 바꿔보기, 서버 하나만 띄워 비교, `OMP_NUM_THREADS` 조정 |
| 명령이 한 프레임씩 튐 | 서버에 `--vote 3` (최근 3회 다수결) |
| 컨트롤러가 `Policy stopped: ...` | 맥 IP/포트, 방화벽, 서버 터미널 에러 확인 후 P 다시 |
| `policy server runs a chaser model, not Cozmo` | 포트를 바꿔 연결함 — Misty 8770 / Cozmo 8771 |
| ours 서버에서 `WARNING ... --vla was not given` | `--vla ../checkpoints/smolvla/<robot>` 추가 |

자세한 내용: [HANDOFF.md](HANDOFF.md) · 태스크 정의: [TASK.md](TASK.md)
