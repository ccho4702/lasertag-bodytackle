# 맥 한 대로 Misty + Cozmo 정책 주행

Windows 로봇 PC 없이, 맥에서 정책 서버와 로봇 제어를 모두 돌리는 방법. 2026-11-12에 실제로 돌려 본 순서 그대로다.

```
 Misty 카메라 (WebSocket) ─▶ misty_live.py ─▶ policy_server :8770 (chaser) ─▶ misty_live.py ─▶ Misty /api/drive/time
 Cozmo 카메라 (SDK, 태블릿 USB) ─▶ cozmo_live.py ─▶ policy_server :8771 (evader) ─▶ cozmo_live.py ─▶ drive_wheel_motors
```

기존 `QUICKSTART_MAC.md` 방식(맥 = 서버, Windows PC = 컨트롤러)과 서버는 같고, 컨트롤러만 이 폴더의 스크립트로 바꾼 것이다.

## 파일

| 파일 | 역할 | 실행 환경 |
|---|---|---|
| `misty_live.py` | Misty 카메라 → chaser 서버 → Misty 주행 API. 화면 없이 터미널에서만 돈다 | `.venv` (Python 3.11) |
| `cozmo_live.py` | Cozmo 카메라 → evader 서버 → 바퀴 명령. 화면 없이 돈다 | `.venv-cozmo` (Python 3.8) |
| `offline_eval.py` | 녹화 세션을 정책에 순서대로 넣어 운전자 명령과 비교 (로봇 불필요) | `.venv` |
| `../baseline/policy_server.py` | 정책 서버 (기존 코드, 수정 없음) | `.venv` |
| `../robot_client/policy_client.py` | 서버에 프레임을 보내고 명령을 받는 클라이언트 (기존 코드) | 둘 다 |

## 1. 설치 (1회)

레포 루트(`lasertag-bodytackle/`)에서 실행한다. `uv`가 필요하다.

```bash
# 정책 서버 + Misty용
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r env/requirements_infer.txt aiohttp

# Cozmo용 (Cozmo SDK가 Python 3.8에 맞춰져 있어 따로 만든다)
uv venv --python 3.8 .venv-cozmo
uv pip install --python .venv-cozmo/bin/python "cozmo[camera]==1.4.10" cozmoclad==3.4.0 numpy==1.24.4 Pillow==10.4.0 aiohttp==3.10.11

# Cozmo 태블릿(Android / Amazon Fire)과 통신하는 adb
brew install --cask android-platform-tools
```

확인: `.venv/bin/python -m unittest discover -s tests` → 12개 통과.

## 2. 실행 순서

### ① 네트워크

맥을 Misty와 같은 Wi-Fi에 붙인다 (Misty는 `192.168.1.38`).

```bash
curl -s http://192.168.1.38/api/battery     # 응답이 오면 연결됨
```

### ② 정책 서버 2개 (터미널 2개, 계속 켜 둔다)

```bash
cd baseline
../.venv/bin/python policy_server.py --ckpt ../checkpoints/policy/baseline_cpu_chaser.pt --port 8770 --device mps
OMP_NUM_THREADS=4 ../.venv/bin/python policy_server.py --ckpt ../checkpoints/policy/baseline_cpu_evader.pt --port 8771 --device cpu
```

`policy server: ... inference ≈ NN ms/frame (budget ...)` 줄이 나오면 준비된 것이다. M5 맥 측정값: chaser 약 30 ms(예산 102), evader 약 71 ms(예산 119).

```bash
curl -s localhost:8770/info    # {"robot": "chaser", ...}
curl -s localhost:8771/info    # {"robot": "evader", ...}
```

### ③ Misty

Misty를 **충전 패드에서 내려놓는다.** 패드 위에서는 주행 명령에 `Success`로 답하면서도 움직이지 않았다.

```bash
# 레포 루트에서
.venv/bin/python mac_direct/misty_live.py --dry --seconds 5     # 카메라 + 정책만 확인, 주행 명령 없음
.venv/bin/python mac_direct/misty_live.py --seconds 30          # 실제 주행
```

- 0.5초마다 `v=+0 w=-25 right  infer 41 ms  age 105 ms  frames 123` 형식으로 출력한다.
- `--seconds`가 지나거나 Ctrl+C를 누르면 정지 명령을 보내고 끝난다.
- `frames`가 0에서 안 올라가면 카메라 스트림이 꼬인 것이다. 스트림을 끄고 다시 실행한다.
  ```bash
  curl -s -X POST http://192.168.1.38/api/videostreaming/stop -H 'Content-Type: application/json' -d '{}'
  ```

### ④ Cozmo

1. 태블릿을 맥에 USB로 꽂고, 태블릿의 "USB 디버깅 허용" 창에서 허용한다 ("이 컴퓨터에서 항상 허용" 체크).
2. `adb devices`에 기기가 `device`로 나와야 한다. `unauthorized`면 허용 창을 아직 안 누른 것이다.
3. 태블릿에서 Cozmo 앱을 열어 Cozmo에 연결하고, 설정에서 SDK 모드를 켠다.

```bash
.venv-cozmo/bin/python mac_direct/cozmo_live.py --dry --seconds 5    # 연결 + 카메라 + 정책만 확인
.venv-cozmo/bin/python mac_direct/cozmo_live.py --seconds 30         # 실제 주행
```

### ⑤ 종료

각 터미널에서 Ctrl+C. 백그라운드로 띄웠다면 `SIGINT`로 끝내야 정지 명령이 나간다.

```bash
pkill -INT -f misty_live.py
pkill -INT -f cozmo_live.py
curl -s -X POST http://192.168.1.38/api/drive/stop -H 'Content-Type: application/json' -d '{"Hold":false}'   # Misty 확실히 정지
```

## 3. 오프라인 확인 (로봇 없이)

녹화 세션 폴더(`labels.csv` + `images/`)를 정책에 넣어 운전자 명령과 비교한다.

```bash
cd baseline
PYTHONPATH=. ../.venv/bin/python ../mac_direct/offline_eval.py ../checkpoints/policy/baseline_cpu_chaser.pt <세션 폴더들>
```

`~/Downloads/sample/`의 Misty 세션 3개(3,094 프레임) 결과: 이동 0.614(최빈 0.636), 회전 0.381(최빈 0.464), 둘 다 일치 0.305(최빈 0.315).

## 4. 동작 방식과 측정값

| | Misty (chaser) | Cozmo (evader) |
|---|---|---|
| 모델 입력 | 프레임 8장, 4프레임 간격 → 약 2.9초 | 프레임 8장, 8프레임 간격 → 약 6.7초 |
| 서버 추론 속도 | 9.8 fps로 제한 | 8.4 fps로 제한 |
| 추론 1회 (중앙값) | 41 ms | 71 ms |
| 프레임 도착 → 명령 (중앙값 / 최대) | 109 / 210 ms | 174 / 305 ms |
| 명령 전송 주기 | 0.15초마다 `drive/time` 500 ms | 명령이 바뀔 때만 |

- 카메라 프레임이 2초 넘게 안 오면 두 스크립트 모두 정지 명령으로 바꾼다.
- Misty의 주행 명령은 500 ms짜리라, 스크립트가 죽어도 0.5초 안에 스스로 멈춘다.

## 5. 알려진 문제

- **Cozmo는 연결이 끊기면 정지 명령을 못 받는다.** 11-12 주행 중 174초 시점에 태블릿 연결이 끊겼고(`Lost connection to the device`), 마지막으로 전달된 명령은 전진이었다. Cozmo의 바퀴 명령에는 시간 제한이 없으므로, 끊기면 직접 Cozmo를 확인해야 한다.
- **Cozmo에 가장자리 보호가 없다.** 원래 컨트롤러(`wasd_misty.py --with-cozmo`)는 경기장 가장자리 15 cm 안에서 자동 후진하지만 `cozmo_live.py`에는 없다.
- **녹화 기능이 없다.** 주행 영상·명령을 데이터셋으로 저장하려면 Windows PC의 `policy_misty.py` / `policy_cozmo.py`를 쓴다.
- **Misty chaser는 전진을 거의 안 한다.** 30초 주행에서 정지 40 %, 좌회전 25 %, 우회전 25 %, 전진 9 %였다 (Cozmo가 시야에 없는 상태).

## 6. 웹 조작 패널 (켜기 / 끄기 버튼)

터미널 명령 대신 브라우저에서 로봇별로 모델을 고르고 켜고 끈다.

```bash
# 레포 루트에서
.venv/bin/python mac_direct/control_panel.py      # → http://127.0.0.1:8780
```

- **켜기**: 고른 모델의 정책 서버가 없거나 다른 모델이면 서버를 띄우거나 바꾼 뒤(20~60초), 로봇 루프를 시작한다.
- **끄기**: 로봇 루프를 중단한다. 루프가 로봇에 정지 명령을 보내고, Misty에는 정지를 한 번 더 보낸다.
- **모두 정지**: 두 로봇 루프를 모두 중단한다.
- **정책 서버 끄기**: 주행 중이 아닌 로봇의 정책 서버를 내린다. 평소에는 켜 두면 다음 켜기가 빠르다.
- 모델 선택: Misty는 baseline / ours(+SmolVLA), Cozmo는 baseline(11-24) / 구버전(11-08).
- 카드에 로봇 연결(Misty 배터리, Cozmo 태블릿), 서버 상태, 마지막 명령, 추론 시간, 최근 로그가 1초마다 갱신된다.
- 패널은 이 맥(127.0.0.1)에서만 열리고, 패널 페이지에서 보낸 요청만 받는다.
- ours는 `.venv-vla` 환경(lerobot 포함)과 `checkpoints/smolvla/chaser`가 필요하다.
  `env/requirements.txt`는 맥에서 numpy 고정 버전이 lerobot과 충돌해 설치되지 않으므로, numpy 줄을 빼고 설치한다.
- 로그: `mac_direct/logs/{misty,cozmo}_{loop,server}.log`

### 경기 (Misty + Cozmo + 녹화 한 번에)

맨 위 "경기" 카드의 버튼 하나로 두 로봇과 천장 카메라 녹화를 같이 다룬다. 모델은 아래 로봇 카드에서 고른다.

| 상태 | 로봇 | 녹화 |
| --- | --- | --- |
| 준비 중 | 정책 서버를 띄우고 로봇에 연결한다. 움직이지 않는다. | 없음 |
| 스탠바이 | 모델이 떠 있고 카메라가 들어온다. 멈춰서 기다린다. | 없음 |
| 경기 중 | 두 로봇이 주행한다. | 녹화 중 |

- **시작**: 두 로봇을 준비시키고, Misty·Cozmo·천장 카메라가 모두 준비되면 녹화를 시작한 뒤 두 로봇을 같은 순간에 출발시킨다.
  "셋 다 준비되면 바로 출발"을 끄면 스탠바이에서 멈춰 기다리고, **출발**을 눌러야 움직인다.
- **Cozmo 정지 — Misty만 주행**: 도망자가 멈춰 있는 조건용. 체크하고 시작하면 Cozmo는 켜지 않고(서버·루프·태블릿 불필요)
  Misty와 녹화만 같이 다룬다. `.meta.json`의 `game`에 `"cozmo": "static"`으로 남는다(주행한 로봇은 모델 이름).
- **모델 / 조건 버튼**: 한 번 누르면 Misty 모델, "Cozmo 정지" 여부, 저장 폴더가 함께 맞춰진다. 저장 폴더는
  `mac_direct/overhead_records/<baseline|ours>/<nonobs_stop|obs_stop|nonobs_moving|obs_moving>/` (obs = 장애물 있음,
  stop = Cozmo 정지). Misty 모델 선택이나 "Cozmo 정지" 체크를 바꿔도 폴더가 따라 바뀐다. 경기 중에는 장애물 조건만 바꿀 수 있다.
  **test** 버튼은 연습 판용 `mac_direct/overhead_records/test/`로 바꾼다(경고 없음).
- **일시 정지**: 로봇은 스탠바이로 서고(모델은 그대로 떠 있음) 녹화는 저장되어 히트 판정이 돌아간다.
  로봇 위치를 다시 잡은 뒤 **출발**을 누르면 바로 움직이고 새 파일로 녹화한다. 출발할 때 정책의 이전 프레임 기록은 비운다.
- **시간 제한 (2분)**: 한 판이 120초를 넘으면 실패(시간 초과)로 보고 자동으로 일시 정지한다. 로봇은 스탠바이로 서고
  녹화는 저장되며, `.meta.json`에 `"ended": "timeout"`이 남고 표에는 "실패 (시간 초과)"로 나온다. 위치를 다시 잡고 **출발**을 누르면 다음 판이다. 제한 시간은 `GAME_MAX_S` 환경 변수로 바꾼다.
  손으로 멈춘 판은 `"ended": "paused"`, 완전 종료로 끝난 판은 `"stopped"`.
- **실패로 표시**: 경기 중에 누르면 시간 제한 전에도 그 판을 실패로 기록하고 일시 정지한다(`"ended": "failure"`).
  시간 초과와 직접 표시 둘 다 `.meta.json`에 `"result": "failure"`가 남고, 표에는 히트 판정과 상관없이 실패로 나온다
  (판정상 히트가 있었으면 옆에 함께 적힌다). 그 밖의 판은 `result`가 비어 있고 히트 판정으로 성공 여부를 본다.
- **완전 종료**: 로봇 루프와 녹화를 모두 끝낸다. 준비 중에 누르면 취소다. 정책 서버는 남겨 두어 다음 시작이 빠르다.
- 저장되는 영상은 두 로봇이 출발한 순간부터 시작한다. 녹화 연결은 출발보다 약 0.8초 먼저 열어 두고, 저장할 때
  그 앞부분 프레임을 잘라낸다(잘라낸 길이는 `.meta.json`의 `trimmed_lead_s`). 그래서 첫 히트 시간은 출발 기준이다.
- **행동 기록**: 판마다 번호 폴더에 `<이름>.actions_misty.csv`, `.actions_cozmo.csv`가 같이 저장된다. 정책이 낸 출력마다
  한 줄(초당 8~10줄)이고 열은 `t_s`(출발 후 초 = 영상 시간), `t_unix`, `cmd_a`, `cmd_b`(Misty: v %, w % / Cozmo: 좌·우 바퀴 mm/s,
  실제로 로봇에 보낸 값), `name`(정책이 고른 행동), `infer_ms`, `frame_age_ms`, `stale`(1이면 강제 정지). 정지시킨 Cozmo는 파일이 없다.
- 한쪽 루프가 죽거나 녹화가 끊기면(최대 10분 포함) 경기를 자동으로 종료하거나 일시 정지하고 이유를 띄운다.
- 경기 중에는 로봇 카드의 켜기/끄기와 수동 녹화 버튼이 잠긴다.
- 구현: 두 루프를 `--gate mac_direct/logs/gate.txt`로 띄운다. 파일 내용이 `run`일 때만 주행하고, 아니면 스탠바이다.
- 리허설: `PANEL_DRY=1`로 패널을 띄우면 루프가 주행 명령을 보내지 않는다(연결·추론·녹화만 확인).

### SmolVLA 지시문 (ours)

Misty 카드의 "SmolVLA 지시문" 칸에서 SmolVLA에 주는 문장을 바꿀 수 있다. **적용**을 누르면 서버를 다시 띄우지 않고 바로
바뀌고, **학습 때 지시문으로**는 체크포인트에 저장된 원래 문장으로 되돌린다. 패널이 문장을 기억해 두었다가 ours 서버를
새로 띄울 때마다 다시 적용하고, 경기 녹화의 `.meta.json`(`game.vla_task`)에도 남긴다.

- 서버 쪽: `POST /vla_task {"task": "..."}` (빈 문자열 = 학습 때 지시문), `GET /info`의 `vla_task`, `vla_task_default`.
- SmolVLA는 문장 하나로만 미세 조정됐다. 다른 문장은 학습 범위 밖이라 뜻대로 따르지 않는다. 녹화 프레임 4장으로 재 보니
  계획이 조금 달라지기는 하지만(평균 차이 0.07~0.20, 범위 −1~1) 방향이 문장과 맞지 않았다("Stop and do not move."에서도
  전진 값이 줄지 않음). 보고서용 ours 측정은 학습 때 지시문으로 해야 한다.

### 과거 프레임을 보는 SmolVLA (ours + SmolVLA h2s8, 실험용)

릴리스 `v2026.11.16-smolvla-h2s8`의 SmolVLA는 현재 프레임에 더해 8프레임, 16프레임 전(약 1.6초 전까지) 두 장을 함께 본다.

```bash
# 기존 checkpoints/smolvla/ 를 덮어쓰지 않도록 다른 폴더에 받는다 (chaser 만, 약 900 MB)
mkdir -p checkpoints/smolvla_h2s8
gh release download v2026.11.16-smolvla-h2s8 -R ccho4702/lasertag-bodytackle -p smolvla_chaser.tar -D checkpoints/smolvla_h2s8
tar -xf checkpoints/smolvla_h2s8/smolvla_chaser.tar -C checkpoints/smolvla_h2s8 && rm checkpoints/smolvla_h2s8/smolvla_chaser.tar
```

패널 Misty 모델에서 "ours + SmolVLA h2s8"을 고르면 `ours_chaser_T8K4.pt` + 이 SmolVLA로 서버를 띄운다. 저장 폴더는 `ours_h2s8/…`.

- `baseline/vla/runner.py`가 과거 프레임을 모아 넘긴다(`vla_config.json`의 `hist`, `hist_stride`를 읽음. 출발 직후에는 첫 프레임으로 채움).
- **짝이 맞지 않는 조합이다.** `ours_chaser_T8K4.pt`는 한 장짜리 SmolVLA(v2026.11.24)의 계획으로 학습됐다. 이 SmolVLA는
  5-fold 중 fold 0 모델(90개 중 72개 세션)이고, 오프라인 정확도도 한 장짜리와 같았다(chaser 0.233 vs 0.230).
  보고서의 ours 수치로 쓰지 말고 참고용으로만 본다.
- 계획 1회에 걸리는 시간이 길다: 같은 Mac에서 343 ms(한 장짜리 183 ms, 다른 정책 서버가 떠 있는 상태에서 잼).

### 상태 점검 (추론 속도)

로봇을 켜면 6초 뒤부터 1초마다 정책 서버의 최신 명령을 읽어 상태를 표시한다.

| 상태 | 뜻 |
|---|---|
| 정상 | 추론 시간이 예산(Misty 102 ms, Cozmo 119 ms) 이하 |
| 예산 초과 | 명령은 나가지만 추론이 느려 학습 때와 프레임 간격이 다름 |
| 강제 정지 중 | 명령이 0.5초 넘게 낡아 서버가 정지를 내보내는 중 (추론 지연 또는 카메라 끊김) |

- 문제가 생기면 페이지 상단 배너와 탭 제목(⚠/⛔)에 표시된다. 카드에는 현재 추론 시간, 최근 30초 중앙값·최대, 강제 정지 비율이 나온다.
- **문제 생기면 자동 정지** 체크박스: 강제 정지 3초 또는 예산 초과 10초가 이어지면 두 로봇을 모두 멈춘다. 그 경기는 무효로 본다.
- 모델이 스스로 정지를 내는 경우(추론은 정상)는 이 점검에 걸리지 않는다. 상태가 "정상"인데 로봇이 서 있으면 모델 출력이다.
- 주의: 패널은 프로세스 확인에 `lsof`를 쓰지 않는다. 1초마다 `lsof`를 돌리면 두 정책 서버가 약 5배 느려진다
  (evader 72 ms → 600 ms, 2026-11-24 측정). 프로세스 목록(`ps`)만 본다.

### Bird's-eye view (천장 카메라)

Jetson의 스트림(`http://192.168.1.37:8088`, `/stream.mjpg`·`/snapshot.jpg`)을 쓴다. 다른 주소는 `OVERHEAD_URL` 환경변수로 준다.

- **실시간 화면**: 스트림이 켜져 있으면 0.4초마다 갱신된다.
- **녹화 시작 / 중지**: 스트림을 디코딩 없이 그대로 저장한다(최대 10분). 주행 중 추론에 부담을 주지 않기 위해서다.
- **저장 위치**: 녹화마다 `mac_direct/overhead_records/<번호>/` 폴더(001, 002, …)를 새로 만들고, 그 안에
  `<날짜_시각>.mkv`, `.meta.json`(프레임 수, 녹화 시간, 프레임 속도), 히트 판정 `.json`, 주석 영상 `.mp4`를 함께 둔다.
  저장 폴더는 패널의 **수정** 버튼으로 바꿀 수 있다(이후 녹화부터 적용, 녹화 중에는 변경 불가, 패널을 다시 켜도 유지).
- **히트 판정**: 녹화를 멈추면 바로 `overhead.hit_detect`를 자동으로 돌린다(로봇이 켜져 있어도 기다리지 않는다).
  결과는 같은 번호 폴더의 `<이름>.json`(첫 히트 시간)과 `<이름>.mp4`(박스가 그려진 영상)이고, 패널 표에 경로와 함께 나온다.
- 첫 히트 시간은 "받은 프레임 수 ÷ 녹화 시간"으로 잰 프레임 속도로 계산한다. 가짜 스트림으로만 시험했고
  (20초 녹화 → 19.9초), 실제 천장 카메라에서는 아직 검증하지 않았다.

### 기타

- `misty_live.py`는 시작할 때 항상 Misty 머리를 끝까지 숙인다(pitch +26°).
- `cozmo_live.py`는 Ctrl+C·종료 신호를 받으면 정지 명령을 보낸 뒤 끝난다(이전에는 정지가 나가지 않았다).
- 패널 주소를 바꾸려면 `PANEL_PORT` 환경변수를 쓴다.
