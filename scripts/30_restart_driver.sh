#!/usr/bin/env bash
# 실행 중인 driver 는 시작 시점의 코드를 메모리에 들고 있다. 코드가 바뀌면 phase 경계에서 재시작해야
# 새 손실/단계 정의가 적용된다. 현재 phase 가 끝나 state 가 넘어가면 driver 를 재시작한다.
#   nohup setsid bash scripts/30_restart_driver.sh > outputs/checkpoints/route/restart.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
CFG=configs/pipeline_route.yaml
STATE=outputs/checkpoints/route/pipeline_state.json
LOCK=outputs/checkpoints/route/pipeline.lock
WANT=${WANT:-1}                      # 이 phase_idx 이상이 되면 재시작
echo "[restart] $(date) phase_idx >= $WANT 대기"
while true; do
  idx=$(python3 -c "import json,pathlib;p=pathlib.Path('$STATE');print(json.loads(p.read_text())['phase_idx'] if p.exists() else 0)" 2>/dev/null || echo 0)
  [ "$idx" -ge "$WANT" ] && break
  sleep 20
done
echo "[restart] $(date) phase_idx=$idx — driver 재시작"
PID=$(pgrep -f "[1]9_train_pipeline.py --config $CFG" | head -1)
if [ -n "$PID" ]; then
  kill "$PID"
  for _ in $(seq 90); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$PID" 2>/dev/null
fi
rm -f "$LOCK"
sleep 15
echo "[restart] $(date) 새 코드로 재기동"
exec python scripts/19_train_pipeline.py --config "$CFG"
