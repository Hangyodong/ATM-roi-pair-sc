#!/usr/bin/env bash
# edge segment 분해가 학습셋 전체에서 끝나고 현재 phase 가 마무리되면,
# segment 분기 + edge count + 전역 배율을 s1 부터 켠 새 파이프라인(route2)으로 다시 시작한다.
#   nohup setsid bash scripts/31_start_route2.sh > outputs/checkpoints/route2/start.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/checkpoints/route2
CFG=configs/pipeline_route.yaml
CK=outputs/checkpoints/route/s1_route/route_sc_corr_step3000.pt

echo "[route2] $(date) edge segment (train 144) 대기"
while true; do
  n=$(python3 - <<'PY'
from pathlib import Path
subs = [l.strip() for l in open("outputs/splits/train.txt") if l.strip()]
print(sum((Path("outputs/roi_pairs") / s / "edge_segments.npz").exists() for s in subs))
PY
)
  echo "[route2] $(date) edge segment train ${n}/144"
  [ "$n" -ge 144 ] && break
  sleep 120
done

echo "[route2] $(date) 현재 phase(s1) 마무리 대기: $CK"
for _ in $(seq 120); do [ -f "$CK" ] && break; sleep 30; done
[ -f "$CK" ] || echo "[route2] 경고: $CK 없음 — 마지막 latest 로 시작한다"

PID=$(pgrep -f "[1]9_train_pipeline.py --config configs/pipeline_route.yaml" | head -1)
if [ -n "$PID" ]; then
  echo "[route2] 기존 driver $PID 종료"
  kill "$PID"; for _ in $(seq 90); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done; kill -9 "$PID" 2>/dev/null
fi
rm -f outputs/checkpoints/route/pipeline.lock outputs/checkpoints/route2/pipeline.lock outputs/checkpoints/route2/preempt
sleep 20
echo "[route2] $(date) 새 파이프라인 시작 (s1 부터, segment+count+scale 포함)"
exec python scripts/19_train_pipeline.py --config "$CFG"
